"""Batch serialization + the S3 object-key layout (stdlib only).

A batch of :class:`~fibersail_edge.edge.features.FeatureFrame`\\ s is encoded as
**gzip-compressed NDJSON**: line 0 is a self-describing header, lines 1..N are one
frame each (the raw ``FeatureFrame._asdict()``). NDJSON + gzip is chosen because it
adds **zero runtime dependencies** (``json`` and ``gzip`` are stdlib) — consistent
with the numpy-only lean-core philosophy of the edge path — while staying
human-inspectable (``… | gunzip | head``), streamable, and directly queryable by
Athena/Glue. The only new runtime dependency in all of Part 3 is ``boto3`` (in
:mod:`~fibersail_edge.cloud.s3`), purely for transport.

Two details are load-bearing:

* **Deterministic bytes.** ``gzip.compress(..., mtime=0)`` omits the timestamp the
  gzip header normally embeds, so the same batch always compresses to identical
  bytes. Combined with the deterministic object key below, a retried upload after a
  crash overwrites the *same* S3 object with the *same* bytes — turning
  at-least-once delivery into an idempotent no-op.
* **Valid JSON only.** ``json.dumps(..., allow_nan=False)`` refuses to emit bare
  ``NaN``/``Infinity`` (which are invalid JSON and break Athena); any non-finite
  float (e.g. a degenerate FFT peak) is sanitized to ``null`` instead. The Part 2
  ``bool()`` cast on ``is_anomaly`` is also load-bearing — ``numpy.bool_`` is not
  JSON-serializable, so the frame must already carry a plain ``bool``.
"""

from __future__ import annotations

import gzip
import io
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

from ..edge.features import FeatureFrame

_SEPARATORS = (",", ":")  # compact → smaller objects, better gzip ratio


@dataclass(frozen=True)
class BatchHeader:
    """Self-describing metadata for one uploaded batch.

    ``session_id`` + ``batch_seq`` uniquely and *stably* identify a batch: they are
    frozen into the serialized bytes at creation, so the S3 key (a pure function of
    the header) is reproducible across a process restart — the basis of idempotent
    retries — while a fresh ``session_id`` per process keeps different runs from
    ever colliding.

    Attributes:
        sensor_id: Logical device/sensor the batch came from (S3 partition key).
        session_id: Stable id for one continuous producer run (per process start).
        batch_seq: Monotonic batch index within the session (0, 1, 2, ...).
        frame_count: Number of frames in the batch.
        t_min: Stream-relative time of the first frame (seconds).
        t_max: Stream-relative time of the last frame (seconds).
        created_at_utc: Wall-clock epoch seconds at batch close. Used ONLY for S3
            date/hour partitioning — frame ``t`` is stream-relative and resets per
            run, so it cannot bucket batches globally.
        schema_version: Envelope schema version (for forward evolution).
        producer: Producing package/version string.
    """

    sensor_id: str
    session_id: str
    batch_seq: int
    frame_count: int
    t_min: float
    t_max: float
    created_at_utc: float
    schema_version: int = 1
    producer: str = "fibersail-edge/0.3.0"

    def __post_init__(self) -> None:
        if not self.sensor_id:
            raise ValueError("sensor_id must be non-empty")
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if self.batch_seq < 0:
            raise ValueError("batch_seq must be >= 0")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be > 0")
        if self.t_max < self.t_min:
            raise ValueError("t_max must be >= t_min")
        if self.created_at_utc <= 0:
            raise ValueError("created_at_utc must be > 0")


# -- Serialization ------------------------------------------------------------


def _clean(record: Dict[str, Any]) -> Dict[str, Any]:
    """Replace non-finite floats with ``None`` so the record is valid JSON."""
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in record.items()
    }


def serialize_batch(frames: Sequence[FeatureFrame], header: BatchHeader) -> bytes:
    """Encode ``frames`` + ``header`` as gzip-compressed NDJSON (deterministic bytes)."""
    lines: List[str] = [
        json.dumps({"type": "header", **asdict(header)}, allow_nan=False, separators=_SEPARATORS)
    ]
    lines.extend(
        json.dumps(_clean(f._asdict()), allow_nan=False, separators=_SEPARATORS) for f in frames
    )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return gzip.compress(payload, mtime=0)


def deserialize_batch(blob: bytes) -> Tuple[BatchHeader, List[FeatureFrame]]:
    """Inverse of :func:`serialize_batch`. Returns ``(header, frames)``."""
    text = gzip.decompress(blob).decode("utf-8")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    header = _header_from_record(records[0])
    frames = [FeatureFrame(**rec) for rec in records[1:]]
    return header, frames


def read_header(blob: bytes) -> BatchHeader:
    """Cheaply read just the header (decompresses only up to the first line)."""
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as gz:
        first = gz.readline()
    return _header_from_record(json.loads(first))


def _header_from_record(record: Dict[str, Any]) -> BatchHeader:
    fields = {k: v for k, v in record.items() if k != "type"}
    return BatchHeader(**fields)


# -- S3 object layout ---------------------------------------------------------


def build_object_key(prefix: str, header: BatchHeader) -> str:
    """Deterministic Hive-partitioned S3 key for a batch (boto3-free).

    ``<prefix>/sensor_id=<id>/date=YYYY-MM-DD/hour=HH/part-<session>-<seq:08d>.ndjson.gz``

    Hive ``key=value`` partitions give Athena/Glue partition pruning; ``sensor_id``
    first spreads writes across prefixes and enables per-device lifecycle rules; the
    ``(session_id, batch_seq)`` filename is deterministic, so a retried upload
    targets the identical key (idempotent overwrite, never a duplicate). ``date``/
    ``hour`` come from ``created_at_utc`` (the edge wall-clock at batch close).
    """
    dt = datetime.fromtimestamp(header.created_at_utc, tz=timezone.utc)
    return (
        f"{prefix.rstrip('/')}"
        f"/sensor_id={header.sensor_id}"
        f"/date={dt:%Y-%m-%d}/hour={dt:%H}"
        f"/part-{header.session_id}-{header.batch_seq:08d}.ndjson.gz"
    )


__all__ = [
    "BatchHeader",
    "serialize_batch",
    "deserialize_batch",
    "read_header",
    "build_object_key",
]
