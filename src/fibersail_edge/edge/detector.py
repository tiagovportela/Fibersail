"""Streaming anomaly detection over rolling features.

The detector consumes one feature reading at a time and emits a boolean flag plus
a score. It is **streaming and O(1) in memory** — it holds a handful of scalars,
never a history of frames — so it runs happily on the edge device alongside the
processor.

Approach — self-calibrating baseline z-score
--------------------------------------------
Which features to watch (and which to ignore) is a deliberate choice:

* **``dominant_freq_hz`` is the primary signature.** The injected fault is a
  *spectral step* (the resonance slides, e.g. 50 -> 35 Hz). A frequency drop is
  specific and amplitude-invariant — unlike RMS it is not moved by benign gain,
  load, or temperature changes.
* **``rms`` is the secondary signature.** The fault also drops damping, so the
  resonance rings harder and overall energy rises.
* **``mean`` and ``std`` are dropped.** On the AC acceleration channel ``mean ≈ 0``
  (uninformative), and with ``mean ≈ 0`` we have ``rms ≈ std`` — monitoring both
  is redundant. So the detector watches one amplitude channel (RMS) and one
  spectral channel (dominant frequency).

The detector spends its first ``calibration_frames`` learning a healthy baseline
mean and standard deviation for each feature (via Welford's online algorithm,
O(1) and numerically stable). Thereafter it flags a frame when *either* feature
deviates beyond ``k`` standard deviations. A two-counter debounce (a Schmitt
trigger) requires several consecutive deviating frames to fire and several calm
frames to clear, trading a little detection latency for far fewer false positives.

A note on honesty (see the README): consecutive frames overlap heavily (a ~1.5 s
window advanced by ~0.1 s), so they are strongly autocorrelated and *not*
independent Gaussian draws. ``k`` is therefore chosen a priori as a control-chart
style threshold, and the false-positive rate is validated empirically on a healthy
stream — it is **not** justified from Gaussian tail probabilities.

:class:`Detector` is a ``Protocol``, so an alternative drops in without touching the
processor. Two adaptive-baseline alternatives — an EWMA control chart and a scalar
Kalman filter — live in :mod:`~fibersail_edge.edge.adaptive` and share this module's
:class:`_Debounce`, so :mod:`~fibersail_edge.edge.compare` can run all three head to
head with only the baseline model differing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

# Detector lifecycle states (plain string constants — no Enum dependency, matching
# the lightweight style of the rest of the package).
CALIBRATING = "calibrating"
NORMAL = "normal"
ANOMALOUS = "anomalous"


@runtime_checkable
class Detector(Protocol):
    """A streaming, O(1)-memory anomaly detector over rolling features.

    Implementations consume the two monitored features per frame and return
    ``(is_anomaly, score)``. Keeping the contract this small lets the processor
    stay agnostic to the detection strategy.
    """

    def update(self, rms: float, dominant_freq_hz: float) -> Tuple[bool, float]:
        """Ingest one frame's features; return ``(is_anomaly, score)``."""
        ...

    def reset(self) -> None:
        """Return to the initial (uncalibrated) state."""
        ...


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for :class:`BaselineZScoreDetector`.

    Attributes:
        calibration_frames: Number of initial (assumed healthy) frames used to
            learn the baseline. At the default ~10 Hz frame rate, 30 frames ≈ 3 s.
        k: Deviation threshold in standard deviations. Chosen a priori (a
            control-chart-style value), not tuned against the evaluation.
        min_consecutive: Consecutive deviating frames required to *raise* a flag
            (debounce → fewer false positives, at a little onset latency).
        clear_consecutive: Consecutive calm frames required to *clear* a flag
            (hysteresis → no flapping near the threshold).
        min_rms_std: Absolute floor on the baseline RMS std (prevents divide-by-
            zero / z-score blow-up on a near-silent baseline).
        min_freq_std_hz: Absolute floor on the baseline dominant-frequency std.
            The FFT peak can sit in a single bin during calm operation, giving a
            near-zero std; flooring at ~one bin (default 0.5 Hz) stops one-bin
            jitter from producing spurious flags.
    """

    calibration_frames: int = 30
    k: float = 4.0
    min_consecutive: int = 3
    clear_consecutive: int = 5
    min_rms_std: float = 1e-9
    min_freq_std_hz: float = 0.5

    def __post_init__(self) -> None:
        if self.calibration_frames < 2:
            raise ValueError("calibration_frames must be >= 2")
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.min_consecutive < 1:
            raise ValueError("min_consecutive must be >= 1")
        if self.clear_consecutive < 1:
            raise ValueError("clear_consecutive must be >= 1")
        if self.min_rms_std <= 0 or self.min_freq_std_hz <= 0:
            raise ValueError("sigma floors must be > 0")


class _Welford:
    """Online mean/variance accumulator (Welford's algorithm). O(1) memory."""

    def __init__(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        self._m2 += delta * (x - self._mean)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        """Population standard deviation (matches ``numpy.std`` default ddof=0)."""
        if self.n < 1:
            return 0.0
        return (self._m2 / self.n) ** 0.5


class _Debounce:
    """Two-counter Schmitt trigger over a stream of raw boolean verdicts.

    Fires only after ``min_consecutive`` consecutive *deviating* frames and clears
    only after ``clear_consecutive`` consecutive *calm* ones — hysteresis that
    trades a little onset latency for far fewer false positives and no flapping.

    Factored out of :class:`BaselineZScoreDetector` so every detector variant shares
    the *identical* debounce; a head-to-head comparison then isolates the baseline
    model, which is the only thing that differs. O(1) memory (two counters + a flag).
    """

    def __init__(self, min_consecutive: int, clear_consecutive: int) -> None:
        self.min_consecutive = min_consecutive
        self.clear_consecutive = clear_consecutive
        self.reset()

    def reset(self) -> None:
        self._fired = False
        self._on = 0
        self._off = 0

    @property
    def fired(self) -> bool:
        return self._fired

    def update(self, raw_anomalous: bool) -> bool:
        """Feed one raw verdict; return the debounced (stable) flag."""
        if not self._fired:
            if raw_anomalous:
                self._on += 1
                if self._on >= self.min_consecutive:
                    self._fired = True
                    self._off = 0
            else:
                self._on = 0
        else:
            if raw_anomalous:
                self._off = 0
            else:
                self._off += 1
                if self._off >= self.clear_consecutive:
                    self._fired = False
                    self._on = 0
        return self._fired


class BaselineZScoreDetector:
    """Baseline z-score detector with debounce. Implements :class:`Detector`.

    Example:
        >>> det = BaselineZScoreDetector(DetectorConfig(calibration_frames=2))
        >>> det.update(1.0, 50.0)      # calibrating
        (False, 0.0)
        >>> det.update(1.0, 50.0)      # calibrating (baseline now fixed)
        (False, 0.0)
        >>> det.is_calibrated
        True
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self.reset()

    # -- Detector interface ----------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """True once the baseline has been established."""
        return self._calibrated

    @property
    def state(self) -> str:
        """Current lifecycle state: ``calibrating`` / ``normal`` / ``anomalous``."""
        if not self._calibrated:
            return CALIBRATING
        return ANOMALOUS if self._debounce.fired else NORMAL

    def update(self, rms: float, dominant_freq_hz: float) -> Tuple[bool, float]:
        cfg = self.config

        # -- Calibration phase: learn the healthy baseline, never flag. --------
        if not self._calibrated:
            self._rms_stats.add(rms)
            self._freq_stats.add(dominant_freq_hz)
            if self._rms_stats.n >= cfg.calibration_frames:
                self._finalize_baseline()
            return (False, 0.0)

        # -- Scoring phase: z-score against the frozen baseline. ---------------
        z_rms = (rms - self._rms_mean) / self._rms_std
        z_freq = (dominant_freq_hz - self._freq_mean) / self._freq_std
        score = float(max(abs(z_rms), abs(z_freq)))
        raw_anomalous = abs(z_rms) > cfg.k or abs(z_freq) > cfg.k
        fired = self._debounce.update(raw_anomalous)
        return (fired, score)

    def reset(self) -> None:
        self._calibrated = False
        self._rms_stats = _Welford()
        self._freq_stats = _Welford()
        self._rms_mean = 0.0
        self._rms_std = 1.0
        self._freq_mean = 0.0
        self._freq_std = 1.0
        self._debounce = _Debounce(self.config.min_consecutive, self.config.clear_consecutive)

    # -- Internals -------------------------------------------------------------

    def _finalize_baseline(self) -> None:
        """Freeze the baseline mean/std (floored) and end calibration."""
        cfg = self.config
        self._rms_mean = self._rms_stats.mean
        self._rms_std = max(self._rms_stats.std, cfg.min_rms_std)
        self._freq_mean = self._freq_stats.mean
        self._freq_std = max(self._freq_stats.std, cfg.min_freq_std_hz)
        self._calibrated = True


__all__ = [
    "Detector",
    "DetectorConfig",
    "BaselineZScoreDetector",
    "CALIBRATING",
    "NORMAL",
    "ANOMALOUS",
]
