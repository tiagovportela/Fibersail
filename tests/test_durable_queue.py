"""Tests for the crash-safe durable file queue (Part 3).

The headline is the brief-required durability test: batches survive a process
restart and are recovered in FIFO order.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from fibersail_edge import FeatureFrame
from fibersail_edge.cloud import BatchHeader, DurableQueue, serialize_batch
from fibersail_edge.cloud.serialization import build_object_key, read_header


def test_put_peek_ack_fifo_order() -> None:
    with tempfile.TemporaryDirectory() as d:
        q = DurableQueue(d, fsync=False, fsync_dir=False)
        for i in range(3):
            q.put(f"batch-{i}".encode())
        assert len(q) == 3
        drained = []
        while (item := q.peek()) is not None:
            drained.append(item[1])
            q.ack(item[0])
        assert drained == [b"batch-0", b"batch-1", b"batch-2"]
        assert len(q) == 0


def test_survives_restart_recovers_in_fifo_order() -> None:
    """Brief-required: un-acked batches survive a restart and recover in FIFO order.

    We drop the queue object without a clean shutdown (no ack, no close) and open a
    fresh instance on the same directory — durability rides on ``put``'s fsync, not
    on any finalizer.
    """
    with tempfile.TemporaryDirectory() as d:
        q = DurableQueue(d)  # fsync on (real durability)
        payloads = [f"batch-{i}".encode() for i in range(5)]
        for p in payloads:
            q.put(p)
        del q  # simulate crash: no clean shutdown

        recovered = DurableQueue(d)
        assert len(recovered) == 5
        drained = []
        while (item := recovered.peek()) is not None:
            drained.append(item[1])
            recovered.ack(item[0])
        assert drained == payloads  # exact content, exact FIFO order


def test_crash_between_upload_and_ack_is_idempotent() -> None:
    """Re-delivery of an uploaded-but-unacked batch overwrites the same S3 key."""
    frames = [FeatureFrame(t=0.0, raw_value=0.0, rms=1.0, mean=0.0, std=1.0,
                           dominant_freq_hz=50.0, is_anomaly=False, score=0.0)]
    header = BatchHeader("s", "sess", 0, 1, 0.0, 0.0, 1_753_100_000.0)
    blob = serialize_batch(frames, header)
    with tempfile.TemporaryDirectory() as d:
        q = DurableQueue(d, fsync=False, fsync_dir=False)
        q.put(blob)
        # "upload" the peeked batch but crash before ack:
        bid1, data1 = q.peek()
        key1 = build_object_key("p", read_header(data1))
        del q
        # restart → same batch re-delivered → same deterministic key:
        q2 = DurableQueue(d, fsync=False, fsync_dir=False)
        bid2, data2 = q2.peek()
        key2 = build_object_key("p", read_header(data2))
        assert data1 == data2  # byte-identical body
        assert key1 == key2     # same key → overwrite, not duplicate


def test_recover_ignores_and_cleans_tmp() -> None:
    with tempfile.TemporaryDirectory() as d:
        q = DurableQueue(d, fsync=False, fsync_dir=False)
        q.put(b"real")
        # Hand-place an orphaned in-progress write.
        with open(os.path.join(d, "00000000000000000099.batch.tmp"), "wb") as fh:
            fh.write(b"partial")
        recovered = DurableQueue(d, fsync=False, fsync_dir=False)
        assert len(recovered) == 1  # tmp ignored
        assert not any(n.endswith(".tmp") for n in os.listdir(d))  # and cleaned up


def test_bounded_disk_drop_oldest() -> None:
    with tempfile.TemporaryDirectory() as d:
        q = DurableQueue(d, fsync=False, fsync_dir=False, max_files=3)
        for i in range(5):
            q.put(f"b{i}".encode())
        assert len(q) == 3
        assert q.dropped_batches == 2
        remaining = []
        while (item := q.peek()) is not None:
            remaining.append(item[1])
            q.ack(item[0])
        assert remaining == [b"b2", b"b3", b"b4"]  # the 3 newest, in order


def test_ack_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as d:
        q = DurableQueue(d, fsync=False, fsync_dir=False)
        bid = q.put(b"x")
        q.ack(bid)
        q.ack(bid)  # second ack is a no-op, not an error
        assert len(q) == 0


def test_invalid_config() -> None:
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            DurableQueue(d, max_files=0)
        with pytest.raises(ValueError):
            DurableQueue(d, max_bytes=-1)
