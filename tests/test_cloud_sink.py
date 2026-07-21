"""Tests for DurableCloudSink — the non-blocking, durable, retrying sink (Part 3).

All tests keep the threads bounded: sinks are always closed, backoff waits use an
injected (non-sleeping) hook, and drains that can't succeed use ``drain=False``.
"""

from __future__ import annotations

import random
import tempfile
import time
import tracemalloc

from fibersail_edge import FeatureFrame, FrameSink
from fibersail_edge.cloud import (
    BackoffConfig,
    DurableCloudSink,
    DurableQueue,
    FlakyUploader,
    InMemoryUploader,
    SinkConfig,
    deserialize_batch,
)

_CFG = SinkConfig(batch_max_frames=50, batch_max_seconds=1000.0, poll_timeout_s=0.02)


def _frames(n: int) -> list[FeatureFrame]:
    return [
        FeatureFrame(t=i * 0.1, raw_value=0.1, rms=1.0, mean=0.0, std=1.0,
                     dominant_freq_hz=50.0, is_anomaly=False, score=0.0)
        for i in range(n)
    ]


def _wait_until(pred, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def _queue(d: str) -> DurableQueue:
    return DurableQueue(d, fsync=False, fsync_dir=False)


def _collect(uploader: InMemoryUploader) -> list[FeatureFrame]:
    out: list[FeatureFrame] = []
    for blob in uploader.objects.values():
        out += deserialize_batch(blob)[1]
    out.sort(key=lambda f: f.t)
    return out


def test_durablecloudsink_is_a_framesink() -> None:
    with tempfile.TemporaryDirectory() as d:
        sink = DurableCloudSink(InMemoryUploader(), _queue(d))
        assert isinstance(sink, FrameSink)


def test_emit_is_nonblocking_under_stalled_uploader() -> None:
    with tempfile.TemporaryDirectory() as d:
        dq = _queue(d)
        stalled = FlakyUploader(InMemoryUploader(), fail_prob=1.0, rng=random.Random(0))
        sink = DurableCloudSink(stalled, dq, config=_CFG, sleep=lambda s: time.sleep(0.001))
        sink.start()
        try:
            worst = 0.0
            for f in _frames(500):
                t0 = time.perf_counter()
                sink.emit(f)
                worst = max(worst, time.perf_counter() - t0)
            assert worst < 0.05  # emit never blocks on the (perpetually failing) uploader
            assert _wait_until(lambda: len(dq) > 0)  # frames still reach durable disk
            assert sink.uploaded_batches == 0
        finally:
            sink.close(drain=False)


def test_emit_bounded_memory_under_backpressure() -> None:
    """With no consumer, emit fills the bounded queue then drops — RAM stays flat."""
    with tempfile.TemporaryDirectory() as d:
        cfg = SinkConfig(queue_maxsize=100, batch_max_frames=50, batch_max_seconds=1000.0)
        sink = DurableCloudSink(InMemoryUploader(), _queue(d), config=cfg)
        # Deliberately do NOT start() — nothing drains the in-memory queue.
        frame = _frames(1)[0]

        def peak_for(n: int) -> int:
            tracemalloc.start()
            for _ in range(n):
                sink.emit(frame)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return peak

        peak_small = peak_for(10_000)
        peak_large = peak_for(100_000)
        assert peak_large < peak_small + 200_000  # bounded by queue_maxsize, not emit count
        assert sink.dropped_frames > 0  # overflow dropped + counted


def test_backoff_retry_sequence_with_injected_sleep_and_flaky() -> None:
    recorded: list[float] = []
    with tempfile.TemporaryDirectory() as d:
        inner = InMemoryUploader()
        flaky = FlakyUploader(inner, fail_first_n=3)
        cfg = SinkConfig(
            batch_max_frames=50, batch_max_seconds=1000.0, poll_timeout_s=0.02,
            backoff=BackoffConfig(base_s=0.5, factor=2.0, max_s=100.0, jitter=0.0),
        )
        sink = DurableCloudSink(flaky, _queue(d), config=cfg, sleep=recorded.append)
        sink.start()
        for f in _frames(5):
            sink.emit(f)
        sink.close(drain=True, timeout=10.0)  # flush partial, then drain: 3 fails + 1 success
        assert recorded == [0.5, 1.0, 2.0]  # exact backoff prefix, no real sleeping
        assert sink.uploaded_batches == 1
        assert len(inner.objects) == 1


def test_graceful_shutdown_flushes_partial_batch() -> None:
    with tempfile.TemporaryDirectory() as d:
        inner = InMemoryUploader()
        sink = DurableCloudSink(inner, _queue(d), config=_CFG)
        sink.start()
        for f in _frames(5):  # fewer than batch_max_frames
            sink.emit(f)
        sink.close()  # drain=True default → flush the partial and upload it
        assert len(inner.objects) == 1
        assert len(_collect(inner)) == 5


def test_no_data_loss_end_to_end_with_intermittent_failures() -> None:
    """sink1 (always failing) buffers to disk; sink2 recovers the spool and uploads all."""
    emitted = _frames(250)
    with tempfile.TemporaryDirectory() as d:
        sink1 = DurableCloudSink(
            FlakyUploader(InMemoryUploader(), fail_prob=1.0, rng=random.Random(0)),
            _queue(d), config=_CFG, sleep=lambda s: time.sleep(0.001),
        )
        sink1.start()
        for f in emitted:
            sink1.emit(f)
        sink1.close(drain=False)  # crash-like: leave everything durable on disk
        assert sink1.uploaded_batches == 0
        assert sink1.enqueued_batches == 5  # 250 / 50

        inner2 = InMemoryUploader()
        dq2 = _queue(d)  # fresh queue on the same spool → recovers sink1's batches
        sink2 = DurableCloudSink(inner2, dq2, config=_CFG)
        sink2.start()
        assert _wait_until(lambda: len(dq2) == 0, timeout=10.0)  # healthy uploader drains it
        sink2.close()

        assert _collect(inner2) == emitted  # every frame present, exact, no loss
