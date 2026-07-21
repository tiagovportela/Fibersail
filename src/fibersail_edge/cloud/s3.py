"""The S3 transport — the only module that imports boto3.

Everything else in :mod:`fibersail_edge.cloud` is pure stdlib and testable without
boto3 or Docker; isolating the dependency here is what makes that possible. The
client takes a configurable ``endpoint_url``, so the *identical* code targets
``moto`` (in-process, no endpoint), LocalStack (``http://localhost:4566``), or real
AWS — only the config changes.

``S3Uploader`` implements the :class:`~fibersail_edge.cloud.uploader.Uploader`
protocol, classifying failures into transient (retry) vs permanent (dead-letter) so
the sink's retry loop does the right thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .uploader import PermanentUploadError, TransientUploadError, Uploader

try:  # boto3 is an optional (cloud) dependency; fail loudly only when actually used.
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, EndpointConnectionError

    _IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    boto3 = None  # type: ignore[assignment]
    BotoCoreError = ClientError = EndpointConnectionError = Exception  # type: ignore[misc,assignment]
    _IMPORT_ERROR = exc


def _require_boto3() -> None:
    if boto3 is None:
        raise ImportError(
            "boto3 is required for S3 sync — install the cloud extra: "
            "`uv sync --extra cloud` (or `pip install 'fibersail-edge[cloud]'`)."
        ) from _IMPORT_ERROR


@dataclass(frozen=True)
class S3Config:
    """Connection + layout settings for S3.

    ``endpoint_url=None`` targets real AWS; set it to a LocalStack URL to use a
    container. moto needs no endpoint (it patches botocore in-process). The dummy
    default credentials let moto/LocalStack work with no environment setup.
    """

    bucket: str
    prefix: str = "telemetry/v1"
    region: str = "us-east-1"
    endpoint_url: Optional[str] = None
    access_key: str = "testing"
    secret_key: str = "testing"

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("bucket must be non-empty")


def make_client(config: S3Config):  # type: ignore[no-untyped-def]
    """Create a boto3 S3 client from ``config``."""
    _require_boto3()
    return boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        region_name=config.region,
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
    )


def ensure_bucket(client, bucket: str, region: str = "us-east-1") -> None:  # type: ignore[no-untyped-def]
    """Create ``bucket`` if it doesn't exist. Idempotent.

    Handles the ``us-east-1`` quirk: ``create_bucket`` there must NOT pass a
    ``LocationConstraint`` (other regions must).
    """
    try:
        if region == "us-east-1":
            client.create_bucket(Bucket=bucket)
        else:
            client.create_bucket(
                Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region}
            )
    except ClientError as exc:  # already-exists is fine — makes this callable twice
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            return
        raise


def read_batch(client, bucket: str, key: str) -> bytes:  # type: ignore[no-untyped-def]
    """Fetch an object's raw bytes.

    Note: S3 does not auto-decompress on ``Content-Encoding: gzip`` for
    ``get_object`` — the bytes come back still gzipped, so pass the result to
    :func:`~fibersail_edge.cloud.serialization.deserialize_batch` (which gunzips).
    """
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def list_batches(client, bucket: str, prefix: str) -> List[Tuple[str, int]]:  # type: ignore[no-untyped-def]
    """List ``(key, size)`` for every object under ``prefix`` (handles >1000)."""
    out: List[Tuple[str, int]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append((obj["Key"], int(obj["Size"])))
    return out


class S3Uploader:
    """Uploads batches to S3. Implements :class:`~fibersail_edge.cloud.uploader.Uploader`.

    Args:
        config: Bucket/prefix/endpoint settings.
        client: An existing boto3 client (e.g. one created inside a ``moto``
            context); if omitted, one is built from ``config``.
    """

    def __init__(self, config: S3Config, *, client=None) -> None:  # type: ignore[no-untyped-def]
        _require_boto3()
        self._cfg = config
        self._client = client if client is not None else make_client(config)

    def upload(self, key: str, data: bytes) -> None:
        try:
            self._client.put_object(
                Bucket=self._cfg.bucket,
                Key=key,
                Body=data,
                ContentType="application/x-ndjson",
                ContentEncoding="gzip",
            )
        except EndpointConnectionError as exc:  # network unreachable → retry
            raise TransientUploadError(str(exc)) from exc
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
            if status >= 500 or status == 429:  # server-side / throttling → retry
                raise TransientUploadError(str(exc)) from exc
            raise PermanentUploadError(str(exc)) from exc  # 4xx auth/validation → don't retry
        except BotoCoreError as exc:  # timeouts, connection resets, etc. → retry
            raise TransientUploadError(str(exc)) from exc

    @property
    def client(self):  # type: ignore[no-untyped-def]
        """The underlying boto3 client (for verification/reads in demos and tests)."""
        return self._client


__all__ = [
    "S3Config",
    "S3Uploader",
    "make_client",
    "ensure_bucket",
    "read_batch",
    "list_batches",
]
