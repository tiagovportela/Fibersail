"""Tests for the bounded ring buffer (Part 2).

The load-bearing property for the whole edge service is that memory does not grow
with stream length — that is the brief's explicit ring-buffer requirement.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from fibersail_edge import RingBuffer


def test_push_and_snapshot_order() -> None:
    rb = RingBuffer(5)
    for x in (1.0, 2.0, 3.0):
        rb.push(x)
    assert len(rb) == 3
    assert not rb.is_full
    assert rb.snapshot().tolist() == [1.0, 2.0, 3.0]


def test_overwrites_oldest_when_full() -> None:
    rb = RingBuffer(3)
    for x in (1.0, 2.0, 3.0, 4.0, 5.0):
        rb.push(x)
    assert rb.is_full
    assert len(rb) == 3
    # Oldest two (1.0, 2.0) evicted; snapshot stays chronological.
    assert rb.snapshot().tolist() == [3.0, 4.0, 5.0]


def test_snapshot_is_a_copy() -> None:
    rb = RingBuffer(3)
    for x in (1.0, 2.0, 3.0):
        rb.push(x)
    snap = rb.snapshot()
    snap[0] = -999.0  # mutating the snapshot must not corrupt the buffer
    assert rb.snapshot().tolist() == [1.0, 2.0, 3.0]


def test_clear_resets() -> None:
    rb = RingBuffer(3)
    for x in (1.0, 2.0, 3.0, 4.0):
        rb.push(x)
    rb.clear()
    assert len(rb) == 0
    assert not rb.is_full
    assert rb.snapshot().size == 0


def test_invalid_capacity_rejected() -> None:
    with pytest.raises(ValueError):
        RingBuffer(0)
    with pytest.raises(ValueError):
        RingBuffer(-1)


def test_memory_is_bounded() -> None:
    """Pushing 100x more samples through the same buffer must not grow memory.

    A preallocated ring buffer allocates once at construction and never again, so
    the traced peak for a huge push count matches that of a small one (allocator
    noise aside). This is the crisp bounded-memory guarantee the edge loop needs.
    """
    rb = RingBuffer(2_000)

    def peak_for(n: int) -> int:
        tracemalloc.start()
        for i in range(n):
            rb.push(float(i))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    peak_small = peak_for(10_000)
    peak_large = peak_for(1_000_000)
    # 100x more pushes; peak must stay essentially flat (generous slack for noise).
    assert peak_large < peak_small + 50_000


def test_snapshot_dtype_is_float() -> None:
    rb = RingBuffer(4)
    rb.push(1.0)
    assert rb.snapshot().dtype == np.float64
