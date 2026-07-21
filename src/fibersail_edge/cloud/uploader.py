"""The upload transport abstraction and stdlib test doubles.

The sink depends only on the :class:`Uploader` protocol, never on boto3 — so the
whole durable-queue / batching / retry core is testable without the cloud extra or
Docker. The real S3 implementation lives in :mod:`~fibersail_edge.cloud.s3`.

Failures are typed so the retry loop can tell them apart:

* :class:`TransientUploadError` — network drop, 5xx, throttling → retry with backoff.
* :class:`PermanentUploadError` — auth/validation/4xx, bad schema → do NOT retry
  forever (the sink dead-letters it so it can't block the queue head).
"""

from __future__ import annotations

import random
import threading
from typing import Dict, Optional, Protocol, Sequence, runtime_checkable


class TransientUploadError(RuntimeError):
    """A retryable upload failure (connectivity, 5xx, throttling)."""


class PermanentUploadError(RuntimeError):
    """A non-retryable upload failure (auth, validation, bad request)."""


@runtime_checkable
class Uploader(Protocol):
    """Uploads one serialized batch to durable remote storage under ``key``."""

    def upload(self, key: str, data: bytes) -> None:
        """Store ``data`` at ``key``. Raise :class:`TransientUploadError` to retry."""
        ...


class InMemoryUploader:
    """An :class:`Uploader` that stores objects in a dict. For tests and AWS-free demos.

    Thread-safe (the uploader thread calls it while tests read :attr:`objects`).
    Idempotent by construction: re-uploading the same key overwrites.
    """

    def __init__(self) -> None:
        self._objects: Dict[str, bytes] = {}
        self._lock = threading.Lock()

    def upload(self, key: str, data: bytes) -> None:
        with self._lock:
            self._objects[key] = data

    @property
    def objects(self) -> Dict[str, bytes]:
        """A snapshot copy of the stored objects, keyed by object key."""
        with self._lock:
            return dict(self._objects)


class FlakyUploader:
    """Wraps any :class:`Uploader` and injects deterministic failures.

    Simulates intermittent connectivity so the retry/backoff/durability paths are
    exercised without real flakiness. Three modes (checked in this order):

    * ``fail_first_n`` — fail the first N attempts, then delegate (ideal for
      asserting an exact backoff sequence then eventual success).
    * ``pattern`` — a cycled sequence of booleans (``True`` = fail this attempt).
    * ``fail_prob`` — fail with this probability, using a seeded ``rng`` for
      reproducibility.
    """

    def __init__(
        self,
        inner: Uploader,
        *,
        fail_first_n: int = 0,
        pattern: Optional[Sequence[bool]] = None,
        fail_prob: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        if fail_first_n < 0:
            raise ValueError("fail_first_n must be >= 0")
        if not (0.0 <= fail_prob <= 1.0):
            raise ValueError("fail_prob must be in [0, 1]")
        if pattern is not None and len(pattern) == 0:
            raise ValueError("pattern must be non-empty when set")
        self._inner = inner
        self._fail_first_n = fail_first_n
        self._pattern = tuple(bool(x) for x in pattern) if pattern is not None else None
        self._fail_prob = fail_prob
        self._rng = rng or random.Random()
        self._attempts = 0
        self._lock = threading.Lock()

    def upload(self, key: str, data: bytes) -> None:
        with self._lock:
            self._attempts += 1
            attempt = self._attempts
        if self._fails(attempt):
            raise TransientUploadError(f"simulated connectivity failure (attempt {attempt})")
        self._inner.upload(key, data)

    @property
    def attempts(self) -> int:
        """Total number of upload attempts seen (including failed ones)."""
        return self._attempts

    def _fails(self, attempt: int) -> bool:
        if attempt <= self._fail_first_n:
            return True
        if self._pattern is not None:
            return self._pattern[(attempt - 1) % len(self._pattern)]
        if self._fail_prob > 0.0:
            return self._rng.random() < self._fail_prob
        return False


__all__ = [
    "Uploader",
    "TransientUploadError",
    "PermanentUploadError",
    "InMemoryUploader",
    "FlakyUploader",
]
