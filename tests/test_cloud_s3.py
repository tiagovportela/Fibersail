"""Tests for the S3 transport against an in-process mock (moto), Part 3."""

from __future__ import annotations

import random
import tempfile

from moto import mock_aws

from fibersail_edge import (
    DampedOscillatorSensor,
    EdgeProcessor,
    FaultConfig,
    FeatureFrame,
    SensorConfig,
)
from fibersail_edge.cloud import (
    BackoffConfig,
    BatchHeader,
    DurableCloudSink,
    DurableQueue,
    FlakyUploader,
    SinkConfig,
    build_object_key,
    deserialize_batch,
    serialize_batch,
)
from fibersail_edge.cloud import s3
from fibersail_edge.cloud.s3 import S3Config, S3Uploader


def _batch(n: int, seq: int = 0) -> tuple[bytes, BatchHeader]:
    frames = [
        FeatureFrame(t=i * 0.1, raw_value=0.5, rms=1.0, mean=0.0, std=1.0,
                     dominant_freq_hz=50.0, is_anomaly=False, score=0.0)
        for i in range(n)
    ]
    header = BatchHeader("press-042", "sess", seq, n, frames[0].t, frames[-1].t, 1_753_100_000.0)
    return serialize_batch(frames, header), header


def test_upload_list_read_round_trips() -> None:
    with mock_aws():
        cfg = S3Config(bucket="fibersail-telemetry")
        client = s3.make_client(cfg)
        s3.ensure_bucket(client, cfg.bucket, cfg.region)
        blob, header = _batch(128)
        key = build_object_key(cfg.prefix, header)
        S3Uploader(cfg, client=client).upload(key, blob)

        listed = s3.list_batches(client, cfg.bucket, cfg.prefix)
        assert [k for k, _ in listed] == [key]
        got_header, got_frames = deserialize_batch(s3.read_batch(client, cfg.bucket, key))
        assert got_header == header
        assert len(got_frames) == 128


def test_ensure_bucket_is_idempotent() -> None:
    with mock_aws():
        cfg = S3Config(bucket="fibersail-telemetry")
        client = s3.make_client(cfg)
        s3.ensure_bucket(client, cfg.bucket, cfg.region)
        s3.ensure_bucket(client, cfg.bucket, cfg.region)  # second call must not raise


def test_content_type_and_encoding_set() -> None:
    with mock_aws():
        cfg = S3Config(bucket="fibersail-telemetry")
        client = s3.make_client(cfg)
        s3.ensure_bucket(client, cfg.bucket, cfg.region)
        blob, header = _batch(4)
        key = build_object_key(cfg.prefix, header)
        S3Uploader(cfg, client=client).upload(key, blob)
        head = client.head_object(Bucket=cfg.bucket, Key=key)
        assert head["ContentType"] == "application/x-ndjson"
        assert head["ContentEncoding"] == "gzip"


def test_idempotent_key_produces_no_duplicate() -> None:
    with mock_aws():
        cfg = S3Config(bucket="fibersail-telemetry")
        client = s3.make_client(cfg)
        s3.ensure_bucket(client, cfg.bucket, cfg.region)
        blob, header = _batch(10)
        key = build_object_key(cfg.prefix, header)
        up = S3Uploader(cfg, client=client)
        up.upload(key, blob)
        up.upload(key, blob)  # retry of the same batch
        assert len(s3.list_batches(client, cfg.bucket, cfg.prefix)) == 1  # overwrite, not duplicate


def test_end_to_end_sensor_to_s3_survives_failures() -> None:
    """Full pipeline into mock S3 with ~30% upload failures → every frame lands, no loss."""
    with mock_aws():
        cfg = S3Config(bucket="fibersail-e2e")
        client = s3.make_client(cfg)
        s3.ensure_bucket(client, cfg.bucket, cfg.region)
        uploader = FlakyUploader(S3Uploader(cfg, client=client), fail_prob=0.3, rng=random.Random(1))
        with tempfile.TemporaryDirectory() as d:
            sink = DurableCloudSink(
                uploader,
                DurableQueue(d, fsync=False, fsync_dir=False),
                key_fn=lambda header: build_object_key(cfg.prefix, header),
                config=SinkConfig(sensor_id="e2e", batch_max_frames=50, batch_max_seconds=1000.0,
                                  poll_timeout_s=0.02,
                                  backoff=BackoffConfig(base_s=0.01, factor=2.0, max_s=0.05, jitter=0.0)),
                sleep=lambda s: None,  # deterministic + no real sleeping
            )
            sensor = DampedOscillatorSensor(
                SensorConfig(duration_s=15.0, seed=42,
                             fault=FaultConfig(start_s=7.0, duration_s=3.0,
                                               omega_n_factor=0.7, zeta_factor=0.4))
            )
            processor = EdgeProcessor.for_source(sensor)
            emitted = 0
            sink.start()
            for frame in processor.process_stream(sensor):
                sink.emit(frame)
                emitted += 1
            sink.close(drain=True, timeout=30.0)

            total = 0
            for key, _ in s3.list_batches(client, cfg.bucket, cfg.prefix):
                total += len(deserialize_batch(s3.read_batch(client, cfg.bucket, key))[1])
            assert emitted > 0
            assert total == emitted  # no loss despite intermittent failures
            assert sink.dropped_frames == 0
