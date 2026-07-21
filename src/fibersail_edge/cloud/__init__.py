"""Part 3 — cloud sync.

Batches + compresses feature frames, buffers them **durably on local disk** so
nothing is lost across a process restart, and uploads them to S3 with retry/backoff
under intermittent connectivity — **without blocking the edge loop**.

    EdgeProcessor ─▶ DurableCloudSink.emit (O(1), non-blocking)
                        │  batcher thread: batch + gzip + fsync
                        ▼
                     DurableQueue (disk)  ──uploader thread (retry/backoff)──▶ Uploader ─▶ S3

- :mod:`~fibersail_edge.cloud.serialization` — NDJSON+gzip batch codec + S3 key layout.
- :mod:`~fibersail_edge.cloud.durable_queue` — crash-safe, restart-recoverable file FIFO.
- :mod:`~fibersail_edge.cloud.backoff` — exponential backoff with jitter.
- :mod:`~fibersail_edge.cloud.uploader` — the ``Uploader`` protocol + stdlib doubles.
- :mod:`~fibersail_edge.cloud.sink` — ``DurableCloudSink`` (implements ``FrameSink``).
- :mod:`~fibersail_edge.cloud.s3` — the boto3 S3 transport (imported separately).

Everything here is pure stdlib and importable without boto3; only
:mod:`fibersail_edge.cloud.s3` needs the ``cloud`` extra
(``uv sync --extra cloud``). That is why ``s3`` is **not** re-exported below —
``from fibersail_edge.cloud import DurableCloudSink`` must work with no boto3.
"""

from __future__ import annotations

from .backoff import BackoffConfig, backoff_delay
from .durable_queue import DurableQueue
from .serialization import (
    BatchHeader,
    build_object_key,
    deserialize_batch,
    read_header,
    serialize_batch,
)
from .sink import DurableCloudSink, SinkConfig
from .uploader import (
    FlakyUploader,
    InMemoryUploader,
    PermanentUploadError,
    TransientUploadError,
    Uploader,
)

__all__ = [
    "BatchHeader",
    "serialize_batch",
    "deserialize_batch",
    "read_header",
    "build_object_key",
    "DurableQueue",
    "BackoffConfig",
    "backoff_delay",
    "Uploader",
    "TransientUploadError",
    "PermanentUploadError",
    "InMemoryUploader",
    "FlakyUploader",
    "SinkConfig",
    "DurableCloudSink",
]
