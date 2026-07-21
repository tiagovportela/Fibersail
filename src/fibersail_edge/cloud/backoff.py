"""Exponential backoff with jitter for upload retries (stdlib only).

Pure policy: given an attempt number, return how long to wait before the next
retry. The *waiting* itself is owned by the sink (so it can be interrupted at
shutdown); this module only computes the delay, which keeps it trivially testable
with an injected RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BackoffConfig:
    """Retry timing policy.

    Attributes:
        base_s: Delay for the first retry (attempt 0), in seconds.
        factor: Multiplier per attempt (``base_s * factor**attempt``).
        max_s: Ceiling on the delay — the cap that actually matters for an edge
            device weathering a long outage.
        jitter: Fraction of the delay that is randomized, in ``[0, 1]``. ``0`` =
            deterministic; ``1`` = full jitter. Jitter decorrelates a fleet of
            devices all reconnecting at once after a regional outage.
        max_attempts: Optional cap on retries. ``None`` (default) retries forever —
            an edge device must keep trying across a multi-hour outage; the batch
            stays durable on disk the whole time.
    """

    base_s: float = 0.5
    factor: float = 2.0
    max_s: float = 30.0
    jitter: float = 1.0
    max_attempts: Optional[int] = None

    def __post_init__(self) -> None:
        if self.base_s <= 0:
            raise ValueError("base_s must be > 0")
        if self.factor < 1.0:
            raise ValueError("factor must be >= 1")
        if self.max_s < self.base_s:
            raise ValueError("max_s must be >= base_s")
        if not (0.0 <= self.jitter <= 1.0):
            raise ValueError("jitter must be in [0, 1]")
        if self.max_attempts is not None and self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1 when set")


def backoff_delay(attempt: int, config: BackoffConfig, rng: random.Random) -> float:
    """Delay (seconds) before retry ``attempt`` (0-based), capped and jittered.

    Uses an equal-jitter blend: a fixed ``(1 - jitter)`` share keeps the backoff
    growing, and the remaining ``jitter`` share is uniformly random — so delays
    still increase but never synchronize across devices.
    """
    # A long outage can push ``attempt`` high enough that ``factor**attempt``
    # overflows a float long before the ``min`` clamps it — treat that as "well
    # past the ceiling" and return the cap.
    try:
        raw = min(config.max_s, config.base_s * config.factor ** attempt)
    except OverflowError:
        raw = config.max_s
    return (1.0 - config.jitter) * raw + rng.uniform(0.0, config.jitter * raw)


__all__ = ["BackoffConfig", "backoff_delay"]
