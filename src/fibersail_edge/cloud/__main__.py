"""End-to-end cloud-sync demo — ``python -m fibersail_edge.cloud``.

Wires the whole pipeline against a mock S3 (``moto``, in-process — no Docker):

    sensor(+fault) -> EdgeProcessor -> DurableCloudSink -> [intermittent] S3Uploader -> moto S3

An outage window (and/or a random failure rate) makes uploads fail, so batches
buffer durably on disk and drain once connectivity returns. Afterwards the demo
lists and reads back every S3 object and prints a summary proving **no data loss**
across the outage, plus the compression ratio.

    uv run python -m fibersail_edge.cloud
    uv run python -m fibersail_edge.cloud --outage-start-s 0 --outage-duration-s 2 --failure-rate 0.2

The same ``S3Uploader`` code targets a real S3-compatible endpoint by setting
``S3Config.endpoint_url`` (e.g. LocalStack ``http://localhost:4566``); the demo
uses moto so it runs anywhere with no external services.
"""

from __future__ import annotations

import argparse
import gzip
import random
import shutil
import tempfile
import time

from ..edge import EdgeProcessor
from ..sensor import DampedOscillatorSensor, FaultConfig, SensorConfig
from .backoff import BackoffConfig
from .durable_queue import DurableQueue
from .serialization import build_object_key, deserialize_batch
from .sink import DurableCloudSink, SinkConfig
from .uploader import TransientUploadError, Uploader


class _IntermittentUploader:
    """Wraps an :class:`Uploader`; fails during a wall-clock outage window and/or randomly."""

    def __init__(
        self,
        inner: Uploader,
        *,
        failure_rate: float,
        outage_start_s: float,
        outage_duration_s: float,
        rng: random.Random,
    ) -> None:
        self._inner = inner
        self._rate = failure_rate
        self._start = outage_start_s
        self._dur = outage_duration_s
        self._rng = rng
        self._t0 = time.monotonic()
        self.failures = 0

    def upload(self, key: str, data: bytes) -> None:
        now = time.monotonic() - self._t0
        in_outage = self._dur > 0 and self._start <= now < self._start + self._dur
        if in_outage or (self._rate > 0 and self._rng.random() < self._rate):
            self.failures += 1
            raise TransientUploadError(f"simulated connectivity loss @ {now:.1f}s")
        self._inner.upload(key, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fibersail cloud-sync demo (mock S3 via moto).")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--sensor-id", default="press-042")
    parser.add_argument("--bucket", default="fibersail-telemetry")
    parser.add_argument("--prefix", default="telemetry/v1")
    parser.add_argument("--batch-frames", type=int, default=100)
    parser.add_argument("--batch-seconds", type=float, default=5.0)
    parser.add_argument("--failure-rate", type=float, default=0.0, help="random upload failure prob")
    parser.add_argument("--outage-start-s", type=float, default=0.0, help="outage window start (wall-clock s)")
    parser.add_argument("--outage-duration-s", type=float, default=2.0, help="outage window length (s)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from moto import mock_aws
    except ImportError:  # pragma: no cover
        raise SystemExit(
            "The demo uses moto for mock S3. Install dev deps with `uv sync` "
            "(moto is in the dev group)."
        )

    from .s3 import S3Config, S3Uploader, ensure_bucket, list_batches, make_client, read_batch

    spool = tempfile.mkdtemp(prefix="fibersail-cloud-")
    try:
        with mock_aws():  # patches botocore process-wide → the uploader thread must run inside this block
            s3cfg = S3Config(bucket=args.bucket, prefix=args.prefix)
            client = make_client(s3cfg)
            ensure_bucket(client, s3cfg.bucket, s3cfg.region)

            uploader = _IntermittentUploader(
                S3Uploader(s3cfg, client=client),
                failure_rate=args.failure_rate,
                outage_start_s=args.outage_start_s,
                outage_duration_s=args.outage_duration_s,
                rng=random.Random(args.seed),
            )
            sink = DurableCloudSink(
                uploader,
                DurableQueue(spool),
                key_fn=lambda header: build_object_key(s3cfg.prefix, header),
                config=SinkConfig(
                    sensor_id=args.sensor_id,
                    batch_max_frames=args.batch_frames,
                    batch_max_seconds=args.batch_seconds,
                    # Snappy backoff so the demo recovers quickly after the outage.
                    backoff=BackoffConfig(base_s=0.2, factor=2.0, max_s=2.0),
                ),
            )

            sensor = DampedOscillatorSensor(
                SensorConfig(
                    duration_s=args.duration_s,
                    seed=args.seed,
                    fault=FaultConfig(start_s=args.duration_s * 0.5, duration_s=3.0,
                                      omega_n_factor=0.7, zeta_factor=0.4),
                )
            )
            processor = EdgeProcessor.for_source(sensor)

            print(f"Streaming {args.duration_s:.0f}s from '{args.sensor_id}' → edge processor → cloud sink ...")
            print(f"Simulated outage: [{args.outage_start_s:.1f}s, "
                  f"{args.outage_start_s + args.outage_duration_s:.1f}s)  failure-rate={args.failure_rate}")

            frames_emitted = 0
            sink.start()
            for frame in processor.process_stream(sensor):
                sink.emit(frame)
                frames_emitted += 1
            sink.close(drain=True, timeout=30.0)  # block until the spool drains (outage ends → retries succeed)

            # --- verify: read every uploaded object back and count frames ---
            objects = list_batches(client, s3cfg.bucket, s3cfg.prefix)
            frames_in_s3 = raw_bytes = gz_bytes = 0
            for key, size in objects:
                blob = read_batch(client, s3cfg.bucket, key)
                gz_bytes += len(blob)
                raw_bytes += len(gzip.decompress(blob))
                frames_in_s3 += len(deserialize_batch(blob)[1])

            ratio = raw_bytes / gz_bytes if gz_bytes else 0.0
            ok = frames_in_s3 == frames_emitted and sink.dropped_frames == 0
            print()
            print(f"  simulated upload failures : {uploader.failures}")
            print(f"  batches enqueued / uploaded: {sink.enqueued_batches} / {sink.uploaded_batches}")
            print(f"  objects in S3             : {len(objects)}")
            if objects:
                print(f"  example key               : {objects[0][0]}")
            print(f"  frames emitted            : {frames_emitted}")
            print(f"  frames in S3              : {frames_in_s3}   "
                  f"{'✓ no loss' if ok else '✗ MISMATCH'}")
            print(f"  dropped (backpressure)    : {sink.dropped_frames}")
            print(f"  bytes gz / raw            : {gz_bytes:,} / {raw_bytes:,}   "
                  f"(compression {ratio:.2f}x)")
    finally:
        shutil.rmtree(spool, ignore_errors=True)


if __name__ == "__main__":
    main()
