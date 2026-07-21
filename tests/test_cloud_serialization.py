"""Tests for the NDJSON+gzip batch codec and the S3 key layout (Part 3)."""

from __future__ import annotations

import gzip

import pytest

from fibersail_edge import FeatureFrame
from fibersail_edge.cloud import (
    BatchHeader,
    build_object_key,
    deserialize_batch,
    read_header,
    serialize_batch,
)


def _frames(n: int) -> list[FeatureFrame]:
    return [
        FeatureFrame(
            t=i * 0.1, raw_value=0.5, rms=1.0 + i, mean=0.0, std=1.0,
            dominant_freq_hz=50.0, is_anomaly=(i % 5 == 0), score=float(i),
        )
        for i in range(n)
    ]


def _header(n: int, *, seq: int = 0) -> BatchHeader:
    frames = _frames(n)
    return BatchHeader(
        sensor_id="press-042", session_id="9f3c2b", batch_seq=seq, frame_count=n,
        t_min=frames[0].t, t_max=frames[-1].t, created_at_utc=1_753_100_000.0,
    )


def test_roundtrip_header_and_frames() -> None:
    frames = _frames(64)
    header = _header(64)
    got_header, got_frames = deserialize_batch(serialize_batch(frames, header))
    assert got_header == header
    assert got_frames == frames  # exact round-trip


def test_serialize_is_byte_deterministic() -> None:
    """Same input → identical bytes (gzip mtime=0), so a re-uploaded batch is idempotent."""
    frames, header = _frames(20), _header(20)
    assert serialize_batch(frames, header) == serialize_batch(frames, header)


def test_gzip_magic_and_compresses() -> None:
    blob = serialize_batch(_frames(200), _header(200))
    assert blob[:2] == b"\x1f\x8b"  # gzip magic
    assert len(blob) < len(gzip.decompress(blob))  # actually smaller


def test_non_finite_becomes_null() -> None:
    frames = [FeatureFrame(t=0.0, raw_value=0.0, rms=float("nan"), mean=0.0, std=0.0,
                           dominant_freq_hz=float("inf"), is_anomaly=False, score=0.0)]
    text = gzip.decompress(serialize_batch(frames, _header(1))).decode()
    assert "null" in text
    assert "NaN" not in text and "Infinity" not in text  # invalid JSON never emitted


def test_read_header_matches_full_deserialize() -> None:
    blob = serialize_batch(_frames(10), _header(10))
    assert read_header(blob) == deserialize_batch(blob)[0]


def test_build_object_key_layout_and_determinism() -> None:
    header = _header(10)
    key = build_object_key("telemetry/v1", header)
    assert key == (
        "telemetry/v1/sensor_id=press-042/date=2025-07-21/hour=12/part-9f3c2b-00000000.ndjson.gz"
    )
    assert build_object_key("telemetry/v1", header) == key  # deterministic
    # A different batch_seq → a different key; the padding preserves lexical order.
    assert build_object_key("telemetry/v1", _header(10, seq=1)).endswith("part-9f3c2b-00000001.ndjson.gz")


def test_batchheader_validation() -> None:
    for bad in (
        dict(sensor_id=""),
        dict(session_id=""),
        dict(batch_seq=-1),
        dict(frame_count=0),
        dict(created_at_utc=0.0),
    ):
        kwargs = dict(sensor_id="s", session_id="x", batch_seq=0, frame_count=1,
                      t_min=0.0, t_max=0.0, created_at_utc=1.0)
        kwargs.update(bad)
        with pytest.raises(ValueError):
            BatchHeader(**kwargs)
    with pytest.raises(ValueError):
        BatchHeader(sensor_id="s", session_id="x", batch_seq=0, frame_count=1,
                    t_min=5.0, t_max=1.0, created_at_utc=1.0)  # t_max < t_min
