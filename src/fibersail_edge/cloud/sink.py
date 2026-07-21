"""``DurableCloudSink`` — the non-blocking, durable, retrying cloud sink.

This is the Part 2/3 boundary made real: it implements the
:class:`~fibersail_edge.edge.sink.FrameSink` protocol, so ``EdgeProcessor.run``
hands it frames, and it does everything slow (batch, compress, fsync, upload,
retry) on **background threads** — the edge loop only ever does an O(1),
non-blocking ``emit``.

Data flow::

    emit(frame) ──▶ bounded in-memory Queue ──▶ [batcher thread] ──▶ DurableQueue (disk)
                     (drop-newest if full)         batch+gzip+fsync        │
                                                                           ▼
                                          S3 ◀── [uploader thread] ◀── peek/upload/ack
                                                  retry w/ backoff

Durability boundary (stated honestly): a frame is durable once its batch is
``fsync``'d by the ``DurableQueue``. Frames still in the in-memory queue or in the
batcher's current partial batch (≤ ``queue_maxsize + batch_max_frames``, sub-second
at 10 Hz) are lost on a hard crash — but ``close()`` flushes them first, and the
brief's durability unit is the *batch*, which always survives.

Memory is bounded independently of S3 state: the in-memory queue is capped
(overflow dropped + counted), the batcher holds ≤ one batch, and the durable buffer
lives on disk, not in RAM.
"""

from __future__ import annotations

import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Callable, Optional

from ..edge.features import FeatureFrame
from .backoff import BackoffConfig, backoff_delay
from .durable_queue import DurableQueue
from .serialization import BatchHeader, build_object_key, read_header, serialize_batch
from .uploader import PermanentUploadError, TransientUploadError, Uploader


@dataclass(frozen=True)
class SinkConfig:
    """Configuration for :class:`DurableCloudSink`.

    Attributes:
        sensor_id: Logical device id stamped into each batch header (S3 partition).
        queue_maxsize: Bound on the in-memory frame queue (backpressure valve).
        batch_max_frames: Flush a batch once it reaches this many frames.
        batch_max_seconds: Flush a batch at least this often (so a partial batch
            can't sit forever when the stream idles).
        poll_timeout_s: Thread wake granularity for idle-flush and stop checks.
        backoff: Retry timing policy for the uploader.
    """

    sensor_id: str = "sensor-0"
    queue_maxsize: int = 10_000
    batch_max_frames: int = 100
    batch_max_seconds: float = 10.0
    poll_timeout_s: float = 0.5
    backoff: BackoffConfig = field(default_factory=BackoffConfig)

    def __post_init__(self) -> None:
        if self.queue_maxsize <= 0:
            raise ValueError("queue_maxsize must be > 0")
        if self.batch_max_frames <= 0:
            raise ValueError("batch_max_frames must be > 0")
        if self.batch_max_seconds <= 0:
            raise ValueError("batch_max_seconds must be > 0")
        if self.poll_timeout_s <= 0:
            raise ValueError("poll_timeout_s must be > 0")


class DurableCloudSink:
    """A :class:`~fibersail_edge.edge.sink.FrameSink` that batches, persists, and uploads.

    Args:
        uploader: The transport (an :class:`~fibersail_edge.cloud.uploader.Uploader`).
        queue: The :class:`~fibersail_edge.cloud.durable_queue.DurableQueue` backing store.
        key_fn: Maps a :class:`BatchHeader` to an object key. Defaults to the
            Hive-partitioned :func:`build_object_key` under ``telemetry/v1``.
        config: :class:`SinkConfig`.
        session_id: Stable per-run id baked into every batch header (defaults to a
            fresh random id, so different runs never produce colliding S3 keys).
        rng / sleep / now: Injection seams for deterministic tests (`sleep` replaces
            the interruptible backoff wait; `now` supplies the batch-close wall-clock).
    """

    def __init__(
        self,
        uploader: Uploader,
        queue: DurableQueue,
        key_fn: Optional[Callable[[BatchHeader], str]] = None,
        config: Optional[SinkConfig] = None,
        *,
        session_id: Optional[str] = None,
        rng: Optional[random.Random] = None,
        sleep: Optional[Callable[[float], None]] = None,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        self._uploader = uploader
        self._dq = queue
        self._cfg = config or SinkConfig()
        self._key_fn = key_fn or (lambda header: build_object_key("telemetry/v1", header))
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._rng = rng or random.Random()
        self._sleep = sleep
        self._now = now or time.time

        self._mem: "Queue[FeatureFrame]" = Queue(maxsize=self._cfg.queue_maxsize)
        self._batch_stop = threading.Event()  # tell batcher to flush + exit
        self._drain = threading.Event()        # tell uploader to exit once queue empties
        self._stop = threading.Event()         # hard-stop the uploader now
        self._batch_seq = 0
        self._batcher_thread: Optional[threading.Thread] = None
        self._uploader_thread: Optional[threading.Thread] = None
        self._started = False
        self._closed = False

        # Counters — each written by exactly one thread, read after join().
        self._dropped_frames = 0     # edge thread (emit)
        self._enqueued_batches = 0   # batcher thread
        self._enqueue_errors = 0     # batcher thread
        self._uploaded_batches = 0   # uploader thread
        self._upload_failures = 0    # uploader thread
        self._dead_lettered = 0      # uploader thread

    # -- FrameSink interface ---------------------------------------------------

    def emit(self, frame: FeatureFrame) -> None:
        """Accept a frame. O(1), non-blocking; drops the newest frame if the queue is full."""
        try:
            self._mem.put_nowait(frame)
        except Full:
            self._dropped_frames += 1  # backpressure: never block the edge loop

    # -- Lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Start the batcher and uploader threads (idempotent)."""
        if self._started:
            return
        self._started = True
        self._batcher_thread = threading.Thread(
            target=self._batch_loop, name="fibersail-batcher", daemon=False
        )
        self._uploader_thread = threading.Thread(
            target=self._upload_loop, name="fibersail-uploader", daemon=False
        )
        self._batcher_thread.start()
        self._uploader_thread.start()

    def close(self, *, drain: bool = True, timeout: Optional[float] = None) -> None:
        """Stop cleanly: flush the final partial batch, then (optionally) drain to the uploader.

        The batcher is always stopped first and joined, so its final partial batch
        is flushed durably to disk. If ``drain`` (default), the uploader is then
        given ``timeout`` seconds (``None`` = unbounded) to empty the durable queue
        before it is hard-stopped; anything still unsent stays durable on disk for
        the next process start. ``drain=False`` stops the uploader immediately.

        Note: ``drain=True`` with ``timeout=None`` blocks until the queue empties —
        don't use it with a permanently-unreachable uploader (use ``drain=False``).
        """
        if self._closed:
            return
        self._closed = True
        if not self._started:
            return
        self._batch_stop.set()
        assert self._batcher_thread is not None and self._uploader_thread is not None
        self._batcher_thread.join()
        if drain:
            self._drain.set()
            self._uploader_thread.join(timeout)
        self._stop.set()  # force-stop the uploader and interrupt any backoff wait
        self._uploader_thread.join()

    def __enter__(self) -> "DurableCloudSink":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- Counters --------------------------------------------------------------

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    @property
    def enqueued_batches(self) -> int:
        return self._enqueued_batches

    @property
    def uploaded_batches(self) -> int:
        return self._uploaded_batches

    @property
    def upload_failures(self) -> int:
        return self._upload_failures

    @property
    def dead_lettered(self) -> int:
        return self._dead_lettered

    @property
    def session_id(self) -> str:
        return self._session_id

    # -- Threads ---------------------------------------------------------------

    def _batch_loop(self) -> None:
        frames: list[FeatureFrame] = []
        deadline = time.monotonic() + self._cfg.batch_max_seconds
        while True:
            try:
                frames.append(self._mem.get(timeout=self._cfg.poll_timeout_s))
            except Empty:
                pass
            now = time.monotonic()
            if frames and (len(frames) >= self._cfg.batch_max_frames or now >= deadline):
                self._flush(frames)
                frames = []
                deadline = time.monotonic() + self._cfg.batch_max_seconds
            if self._batch_stop.is_set():
                # Drain whatever the edge loop already enqueued, flushing in
                # batch_max_frames chunks so a burst can't produce one huge object.
                while True:
                    try:
                        frames.append(self._mem.get_nowait())
                    except Empty:
                        break
                    if len(frames) >= self._cfg.batch_max_frames:
                        self._flush(frames)
                        frames = []
                if frames:
                    self._flush(frames)
                return

    def _flush(self, frames: list[FeatureFrame]) -> None:
        header = BatchHeader(
            sensor_id=self._cfg.sensor_id,
            session_id=self._session_id,
            batch_seq=self._batch_seq,
            frame_count=len(frames),
            t_min=frames[0].t,
            t_max=frames[-1].t,
            created_at_utc=float(self._now()),
        )
        self._batch_seq += 1
        data = serialize_batch(frames, header)
        try:
            self._dq.put(data)
            self._enqueued_batches += 1
        except OSError:
            # e.g. ENOSPC: count it and keep the thread alive rather than crashing.
            self._enqueue_errors += 1

    def _upload_loop(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            item = self._dq.peek()
            if item is None:
                if self._drain.is_set():
                    return  # queue empty and we were asked to drain → done
                self._stop.wait(self._cfg.poll_timeout_s)
                attempt = 0
                continue
            batch_id, data = item
            try:
                header = read_header(data)
                self._uploader.upload(self._key_fn(header), data)
                self._dq.ack(batch_id)
                self._uploaded_batches += 1
                attempt = 0
            except PermanentUploadError:
                # Poison batch: drop it so it can't block the queue head forever.
                self._dq.ack(batch_id)
                self._dead_lettered += 1
                attempt = 0
            except TransientUploadError:
                self._upload_failures += 1
                max_attempts = self._cfg.backoff.max_attempts
                if max_attempts is not None and attempt + 1 >= max_attempts:
                    # Configured max_attempts reached: dead-letter (drop + count) so a
                    # single batch can't block the queue head forever. This IS data
                    # loss — the default max_attempts=None never triggers it (an edge
                    # device should keep retrying across a long outage instead).
                    self._dq.ack(batch_id)
                    self._dead_lettered += 1
                    attempt = 0
                    continue
                self._wait_backoff(backoff_delay(attempt, self._cfg.backoff, self._rng))
                attempt += 1

    def _wait_backoff(self, delay: float) -> None:
        if self._sleep is not None:
            self._sleep(delay)  # test seam: record + return immediately
        else:
            self._stop.wait(delay)  # production: interruptible by close()


__all__ = ["SinkConfig", "DurableCloudSink"]
