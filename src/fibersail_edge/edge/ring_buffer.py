"""A fixed-capacity ring (circular) buffer over a preallocated numpy array.

Why a custom buffer rather than ``collections.deque(maxlen=N)``
---------------------------------------------------------------
A ``deque(maxlen)`` is the idiomatic O(1) append-and-evict structure, but it
stores *boxed* Python floats and is non-contiguous, so every time the edge
processor wants to run an FFT or a vectorized reduction it must first copy the
deque into a numpy array (an O(N) iterate-and-unbox in Python).

This buffer instead writes into a single ``np.empty(capacity)`` allocated once
at construction. Two properties matter for the edge use-case:

* **Vectorized features for free.** :meth:`snapshot` returns a contiguous
  ``float64`` array, so RMS/mean/std and ``np.fft.rfft`` run as pure C with no
  per-sample Python cost.
* **Provably bounded memory.** After construction the buffer never allocates
  again, no matter how long the stream runs. That is the crispest possible
  expression of the brief's "memory must not grow unbounded" — a test can push
  ten thousand vs. a million samples through the same buffer and observe an
  essentially identical peak allocation.

The buffer holds only ``value``\\ s (plain floats); timestamps are carried by the
processor, which knows the newest sample's time when it takes a snapshot.
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """A bounded FIFO of floats backed by a preallocated numpy array.

    Once the buffer is full, each :meth:`push` overwrites the oldest value, so
    the memory footprint is fixed at ``capacity`` floats for the lifetime of the
    stream.

    Example:
        >>> rb = RingBuffer(3)
        >>> for x in (1.0, 2.0, 3.0, 4.0):
        ...     rb.push(x)
        >>> rb.snapshot().tolist()   # oldest (2.0) was evicted
        [2.0, 3.0, 4.0]
    """

    def __init__(self, capacity: int, *, dtype: type = float) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._buf = np.empty(capacity, dtype=dtype)
        self._cursor = 0  # index the next push writes to
        self._count = 0  # number of valid values (<= capacity)

    # -- Introspection ---------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of values retained."""
        return self._buf.shape[0]

    @property
    def is_full(self) -> bool:
        """True once ``capacity`` values have been pushed."""
        return self._count == self._buf.shape[0]

    def __len__(self) -> int:
        """Number of valid values currently held (``<= capacity``)."""
        return self._count

    # -- Mutation --------------------------------------------------------------

    def push(self, value: float) -> None:
        """Append ``value``; overwrite the oldest once full. O(1), no allocation."""
        capacity = self._buf.shape[0]
        self._buf[self._cursor] = value
        self._cursor = (self._cursor + 1) % capacity
        if self._count < capacity:
            self._count += 1

    def snapshot(self) -> np.ndarray:
        """Return the held values oldest-to-newest as a fresh contiguous array.

        A copy is returned (not a view) so callers may detrend/window it in place
        without corrupting the buffer. Length equals ``len(self)``.
        """
        if self._count < self._buf.shape[0]:
            # Not full yet: valid values sit contiguously in ``[0, count)``.
            return self._buf[: self._count].copy()
        # Full: the oldest value is at ``_cursor``; stitch the two halves.
        return np.concatenate((self._buf[self._cursor :], self._buf[: self._cursor]))

    def clear(self) -> None:
        """Drop all values and reset to empty (retains the allocation)."""
        self._cursor = 0
        self._count = 0


__all__ = ["RingBuffer"]
