"""The seam that decouples the edge processor (Part 2) from cloud sync (Part 3).

Why a sink, not just an iterator
--------------------------------
:meth:`EdgeProcessor.process_stream <fibersail_edge.processor.EdgeProcessor.process_stream>`
already yields frames lazily, which is enough for tests and simple in-process use.
But a lazy iterator does **not**, on its own, satisfy the brief's "the edge
processor shouldn't block on S3": if the consumer does blocking network I/O inline
it merely *relocates* the stall — the consumer blocks, so the processor blocks, so
the sensor read blocks. True decoupling needs the edge loop to hand each frame to a
**non-blocking** sink and let a *separate* worker deal with the network.

This module defines that contract (:class:`FrameSink`) and ships only in-memory
test doubles. Part 3 implements the production sink: :meth:`~FrameSink.emit`
enqueues the frame onto a bounded in-memory queue backed by a durable on-disk file
queue and returns immediately; a separate uploader thread drains it to S3 with
retry/backoff, and on a prolonged outage frames spill to disk (durable) rather than
blocking the edge loop. That production sink must be thread-safe; the in-memory
doubles here need not be (they are used single-threaded in tests/demos).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Protocol, runtime_checkable

from .features import FeatureFrame


@runtime_checkable
class FrameSink(Protocol):
    """A destination for emitted :class:`FeatureFrame`\\ s.

    The one hard rule: :meth:`emit` **must be non-blocking / O(1)**. It may
    enqueue, buffer, or hand off, but it must never do slow work (network, fsync
    of a large batch) on the caller's thread — that thread is the real-time edge
    loop.
    """

    def emit(self, frame: FeatureFrame) -> None:
        """Accept one frame. Must return promptly without blocking the edge loop."""
        ...


class ListSink:
    """Collects frames into an in-memory list. For tests and small demos.

    Args:
        maxlen: If set, keep only the most recent ``maxlen`` frames (bounded
            memory). ``None`` keeps everything — fine for short demo runs, not for
            an unbounded stream.
    """

    def __init__(self, maxlen: Optional[int] = None) -> None:
        if maxlen is not None and maxlen <= 0:
            raise ValueError(f"maxlen must be > 0 when set, got {maxlen}")
        self._maxlen = maxlen
        self._frames: List[FeatureFrame] = []

    def emit(self, frame: FeatureFrame) -> None:
        self._frames.append(frame)
        if self._maxlen is not None and len(self._frames) > self._maxlen:
            # Drop the oldest to stay bounded (O(1) amortized for our sizes).
            del self._frames[0]

    @property
    def frames(self) -> List[FeatureFrame]:
        """The collected frames, oldest-to-newest."""
        return self._frames


class CallbackSink:
    """Forwards each frame to a user callback.

    A thin adapter — e.g. push to a queue, log a line, or (in Part 3) enqueue for
    upload. The callback is responsible for staying non-blocking.
    """

    def __init__(self, fn: Callable[[FeatureFrame], None]) -> None:
        self._fn = fn

    def emit(self, frame: FeatureFrame) -> None:
        self._fn(frame)


__all__ = ["FrameSink", "ListSink", "CallbackSink"]
