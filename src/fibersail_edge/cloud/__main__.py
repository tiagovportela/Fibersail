"""End-to-end cloud-sync demo — ``python -m fibersail_edge.cloud``.

Wires the whole pipeline:

    sensor(+fault) -> EdgeProcessor -> DurableCloudSink -> [intermittent] S3Uploader -> S3

An outage window (and/or a random failure rate) makes uploads fail, so batches
buffer durably on disk and drain once connectivity returns. Afterwards the demo
reads every object back and prints a summary proving **no data loss**.

Two backends, same ``S3Uploader`` code — only ``S3Config.endpoint_url`` changes:

* **moto** (default): an in-process mock. Runs anywhere, no Docker — but the
  objects live only inside this process and vanish when it exits.
      uv run python -m fibersail_edge.cloud

* **LocalStack / real AWS** (``--localstack`` or ``--endpoint-url``): a real S3 API
  server over HTTP. Objects **persist** for the life of the service, so you can
  browse them afterward with the AWS CLI. This is the closest setup to production.
      docker compose up -d localstack
      uv run python -m fibersail_edge.cloud --localstack

The identical code path targets real AWS: drop the endpoint and supply real
credentials/region instead of the dummy ones.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import random
import shutil
import tempfile
import time
from typing import Iterator, Optional

from ..edge import EdgeProcessor
from ..sensor import DampedOscillatorSensor, FaultConfig, SensorConfig
from .backoff import BackoffConfig
from .durable_queue import DurableQueue
from .serialization import build_object_key, deserialize_batch
from .sink import DurableCloudSink, SinkConfig
from .uploader import TransientUploadError, Uploader

_LOCALSTACK_URL = "http://localhost:4566"


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


@contextlib.contextmanager
def _backend(endpoint_url: Optional[str]) -> Iterator[str]:
    """Enter the chosen S3 backend. Yields a human-readable backend label.

    With no endpoint we start ``moto`` (patches botocore process-wide, so the
    uploader thread must run inside this context). With an endpoint we use the real
    boto3 client — no patching, and the objects persist in the target service.
    """
    if endpoint_url is None:
        try:
            from moto import mock_aws
        except ImportError:  # pragma: no cover
            raise SystemExit(
                "moto is needed for the default in-process backend. Run `uv sync` "
                "(moto is in the dev group), or use --localstack against a container."
            )
        with mock_aws():
            yield "moto (in-process mock; objects vanish when this process exits)"
    else:
        yield f"{endpoint_url} (real S3 endpoint; objects persist)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fibersail cloud-sync end-to-end demo.")
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
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--endpoint-url", default=None,
                        help="target a real S3 endpoint (e.g. LocalStack) instead of the moto mock")
    parser.add_argument("--localstack", action="store_true",
                        help=f"shortcut for --endpoint-url {_LOCALSTACK_URL}")
    args = parser.parse_args()

    endpoint_url = args.endpoint_url or (_LOCALSTACK_URL if args.localstack else None)

    from .s3 import S3Config, S3Uploader, ensure_bucket, list_batches, make_client, read_batch

    s3cfg = S3Config(bucket=args.bucket, prefix=args.prefix, region=args.region, endpoint_url=endpoint_url)
    spool = tempfile.mkdtemp(prefix="fibersail-cloud-")
    try:
        with _backend(endpoint_url) as backend_label:
            print(f"Backend: {backend_label}")
            client = make_client(s3cfg)
            try:
                ensure_bucket(client, s3cfg.bucket, s3cfg.region)
            except Exception as exc:  # noqa: BLE001 - surface a friendly hint for a down endpoint
                if endpoint_url is not None:
                    raise SystemExit(
                        f"Could not reach S3 at {endpoint_url}: {exc}\n"
                        f"Is LocalStack running?  docker compose up -d localstack"
                    )
                raise

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
            sink.close(drain=True, timeout=30.0)

            # Verify against THIS run only: a persistent endpoint (LocalStack/AWS)
            # accumulates objects across runs, so filter by our session id.
            all_objects = list_batches(client, s3cfg.bucket, s3cfg.prefix)
            objects = [(k, s) for k, s in all_objects if f"part-{sink.session_id}-" in k]
            frames_in_s3 = raw_bytes = gz_bytes = 0
            for key, _size in objects:
                blob = read_batch(client, s3cfg.bucket, key)
                gz_bytes += len(blob)
                raw_bytes += len(gzip.decompress(blob))
                frames_in_s3 += len(deserialize_batch(blob)[1])

            ratio = raw_bytes / gz_bytes if gz_bytes else 0.0
            ok = frames_in_s3 == frames_emitted and sink.dropped_frames == 0
            print()
            print(f"  simulated upload failures : {uploader.failures}")
            print(f"  batches enqueued / uploaded: {sink.enqueued_batches} / {sink.uploaded_batches}")
            if len(all_objects) != len(objects):
                print(f"  objects in S3 (all runs)  : {len(all_objects)}")
            print(f"  objects in S3 (this run)  : {len(objects)}")
            if objects:
                print(f"  example key               : {objects[0][0]}")
            print(f"  frames emitted            : {frames_emitted}")
            print(f"  frames in S3              : {frames_in_s3}   "
                  f"{'✓ no loss' if ok else '✗ MISMATCH'}")
            print(f"  dropped (backpressure)    : {sink.dropped_frames}")
            print(f"  bytes gz / raw            : {gz_bytes:,} / {raw_bytes:,}   "
                  f"(compression {ratio:.2f}x)")

            if endpoint_url is not None:
                print("\nObjects persist — browse them (dummy creds work for LocalStack):")
                print(f"  export AWS_ACCESS_KEY_ID=testing AWS_SECRET_ACCESS_KEY=testing")
                print(f"  aws --endpoint-url={endpoint_url} s3 ls s3://{args.bucket}/{args.prefix}/ --recursive")
                if objects:
                    print(f"  aws --endpoint-url={endpoint_url} s3 cp "
                          f"s3://{args.bucket}/{objects[0][0]} - | gunzip | head")
    finally:
        shutil.rmtree(spool, ignore_errors=True)


if __name__ == "__main__":
    main()
