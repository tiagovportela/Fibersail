"""Tests for the retry backoff policy (Part 3)."""

from __future__ import annotations

import random

import pytest

from fibersail_edge.cloud import BackoffConfig, backoff_delay


def test_no_jitter_is_capped_exponential() -> None:
    cfg = BackoffConfig(base_s=0.5, factor=2.0, max_s=10.0, jitter=0.0)
    rng = random.Random(0)
    delays = [backoff_delay(n, cfg, rng) for n in range(6)]
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 10.0]  # doubles, then capped at max_s


def test_jitter_within_bounds_and_reproducible() -> None:
    cfg = BackoffConfig(base_s=0.5, factor=2.0, max_s=10.0, jitter=1.0)
    a = [backoff_delay(n, cfg, random.Random(7)) for n in range(6)]
    b = [backoff_delay(n, cfg, random.Random(7)) for n in range(6)]
    assert a == b  # seeded → reproducible
    for n, delay in enumerate(b):
        cap = min(cfg.max_s, cfg.base_s * cfg.factor ** n)
        assert 0.0 <= delay <= cap  # full jitter stays within [0, raw]


def test_overflow_guard_returns_cap() -> None:
    """A long outage pushes ``attempt`` high enough to overflow ``factor**attempt``."""
    cfg = BackoffConfig(base_s=0.5, factor=2.0, max_s=30.0, jitter=0.0)
    assert backoff_delay(5000, cfg, random.Random(0)) == 30.0  # no OverflowError, clamped


def test_config_validation() -> None:
    for bad in (
        dict(base_s=0.0),
        dict(factor=0.5),
        dict(max_s=0.1),  # < base_s (0.5)
        dict(jitter=1.5),
        dict(max_attempts=0),
    ):
        with pytest.raises(ValueError):
            BackoffConfig(**bad)
