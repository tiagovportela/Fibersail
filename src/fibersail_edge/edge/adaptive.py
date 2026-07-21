"""Adaptive-baseline detectors — EWMA and Kalman — as alternatives to compare.

Both implement the same :class:`~fibersail_edge.edge.detector.Detector` protocol as
:class:`~fibersail_edge.edge.detector.BaselineZScoreDetector`, watch the same two
features (dominant frequency as the primary spectral signature, RMS as the secondary
amplitude one), and pass through the *same*
:class:`~fibersail_edge.edge.detector._Debounce`. So a head-to-head run
(:mod:`~fibersail_edge.edge.compare`) isolates the one thing that differs: the
**baseline model**.

Why an adaptive baseline at all
-------------------------------
The baseline z-score detector learns a healthy mean/std over the calibration window,
then **freezes** it forever. That is simple and robust on a stationary bench signal.
A real machine, though, is not stationary over hours: temperature, load, and
lubrication slowly shift the healthy operating point. A permanently frozen baseline
eventually reads that benign drift as an anomaly (false positives climb), while naive
periodic re-calibration risks re-baselining on an already-degrading machine.

Both detectors here instead track a *slowly adapting* baseline, so benign drift is
followed rather than flagged. The cost of adaptivity is the obvious failure mode: a
genuine, **sustained** fault could be absorbed into the baseline and so mask itself.
The fix both share is **freeze-on-anomaly** — the baseline is updated only from frames
that look healthy (``|deviation| <= k``); a frame that looks anomalous scores against
the frozen baseline but does not move it. This is the freeze-on-anomaly behavior the
README's "what I'd change" section called out.

EWMA vs. Kalman — two views of one local-level model
----------------------------------------------------
* :class:`EwmaDetector` — an EWMA control chart. Baseline mean and variance are
  exponentially weighted with smoothing ``alpha``; a frame is scored by the familiar
  ``z = (x - mean) / std``. The single intuitive knob: ``alpha ≈ 1 / (memory in
  frames)`` — ``alpha = 0.05`` keeps ~20 frames (~2 s at 10 Hz) of effective memory.
* :class:`KalmanDetector` — a scalar Kalman filter with a random-walk state (each
  feature is a slowly-varying hidden *level* observed in noise). Its steady state *is*
  an EWMA, but it scores with the **normalized innovation** ``y / sqrt(S)`` (``S`` =
  innovation variance) — the statistically principled residual, unit-variance under
  the healthy hypothesis — and its knob is the process/measurement-noise ratio
  ``q / r``, the physical analogue of ``alpha`` (bigger ratio ⇒ tracks faster).

Both remain streaming and O(1) in memory (a handful of scalars), like the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .detector import CALIBRATING, NORMAL, ANOMALOUS, _Debounce, _Welford


def _ewma_step(mean: float, var: float, x: float, alpha: float) -> Tuple[float, float]:
    """One incremental EWMA mean/variance step (West's weighted update).

    Returns the updated ``(mean, var)``. ``var`` is the exponentially-weighted
    variance and stays ``>= 0`` by construction.
    """
    diff = x - mean
    incr = alpha * diff
    mean = mean + incr
    var = (1.0 - alpha) * (var + diff * incr)
    return mean, var


# --------------------------------------------------------------------------- EWMA


@dataclass(frozen=True)
class EwmaConfig:
    """Configuration for :class:`EwmaDetector`.

    Attributes:
        calibration_frames: Initial (assumed healthy) frames used to seed the EWMA
            baseline mean/variance. ~30 frames ≈ 3 s at the default 10 Hz.
        alpha: EWMA smoothing factor in ``(0, 1]``. Effective memory ≈ ``1 / alpha``
            frames, so ``0.05`` ≈ 20 frames (~2 s). Larger ⇒ adapts faster to drift
            but tolerates a shorter anomaly before absorbing it.
        k: Deviation threshold in standard deviations (control-chart value, a priori).
        min_consecutive: Consecutive deviating frames required to raise a flag.
        clear_consecutive: Consecutive calm frames required to clear a flag.
        min_rms_std: Floor on the baseline RMS std (prevents z-score blow-up).
        min_freq_std_hz: Floor on the baseline dominant-frequency std (~one FFT bin).
    """

    calibration_frames: int = 30
    alpha: float = 0.05
    k: float = 4.0
    min_consecutive: int = 3
    clear_consecutive: int = 5
    min_rms_std: float = 1e-9
    min_freq_std_hz: float = 0.5

    def __post_init__(self) -> None:
        if self.calibration_frames < 2:
            raise ValueError("calibration_frames must be >= 2")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.min_consecutive < 1:
            raise ValueError("min_consecutive must be >= 1")
        if self.clear_consecutive < 1:
            raise ValueError("clear_consecutive must be >= 1")
        if self.min_rms_std <= 0 or self.min_freq_std_hz <= 0:
            raise ValueError("sigma floors must be > 0")


class EwmaDetector:
    """EWMA control-chart detector with freeze-on-anomaly. Implements :class:`Detector`.

    Example:
        >>> det = EwmaDetector(EwmaConfig(calibration_frames=2))
        >>> det.update(1.0, 50.0)      # calibrating
        (False, 0.0)
        >>> det.update(1.0, 50.0)      # calibrating (baseline now seeded)
        (False, 0.0)
        >>> det.is_calibrated
        True
    """

    def __init__(self, config: Optional[EwmaConfig] = None) -> None:
        self.config = config or EwmaConfig()
        self.reset()

    # -- Detector interface ----------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def state(self) -> str:
        """Current lifecycle state: ``calibrating`` / ``normal`` / ``anomalous``."""
        if not self._calibrated:
            return CALIBRATING
        return ANOMALOUS if self._debounce.fired else NORMAL

    def update(self, rms: float, dominant_freq_hz: float) -> Tuple[bool, float]:
        cfg = self.config

        # -- Calibration: seed the baseline mean/variance, never flag. ---------
        if not self._calibrated:
            self._rms_stats.add(rms)
            self._freq_stats.add(dominant_freq_hz)
            if self._rms_stats.n >= cfg.calibration_frames:
                self._finalize_baseline()
            return (False, 0.0)

        # -- Score against the current adaptive baseline. ----------------------
        rms_std = max(self._rms_var ** 0.5, cfg.min_rms_std)
        freq_std = max(self._freq_var ** 0.5, cfg.min_freq_std_hz)
        z_rms = (rms - self._rms_mean) / rms_std
        z_freq = (dominant_freq_hz - self._freq_mean) / freq_std
        score = float(max(abs(z_rms), abs(z_freq)))
        raw_anomalous = abs(z_rms) > cfg.k or abs(z_freq) > cfg.k
        fired = self._debounce.update(raw_anomalous)

        # -- Adapt only on healthy-looking frames (freeze-on-anomaly). ---------
        if not raw_anomalous:
            self._rms_mean, self._rms_var = _ewma_step(self._rms_mean, self._rms_var, rms, cfg.alpha)
            self._freq_mean, self._freq_var = _ewma_step(
                self._freq_mean, self._freq_var, dominant_freq_hz, cfg.alpha
            )

        return (fired, score)

    def reset(self) -> None:
        self._calibrated = False
        self._rms_stats = _Welford()
        self._freq_stats = _Welford()
        self._rms_mean = 0.0
        self._rms_var = 1.0
        self._freq_mean = 0.0
        self._freq_var = 1.0
        self._debounce = _Debounce(self.config.min_consecutive, self.config.clear_consecutive)

    # -- Internals -------------------------------------------------------------

    def _finalize_baseline(self) -> None:
        cfg = self.config
        self._rms_mean = self._rms_stats.mean
        self._rms_var = max(self._rms_stats.std, cfg.min_rms_std) ** 2
        self._freq_mean = self._freq_stats.mean
        self._freq_var = max(self._freq_stats.std, cfg.min_freq_std_hz) ** 2
        self._calibrated = True


# ------------------------------------------------------------------------- Kalman


class _ScalarKalman:
    """Scalar local-level (random-walk) Kalman filter: hidden level in noise.

    Model: ``mu_t = mu_{t-1} + w``, ``x_t = mu_t + v`` with ``Var(w)=q``, ``Var(v)=r``.
    The predicted innovation variance ``S = P + q + r`` normalizes the residual so
    ``y / sqrt(S)`` is unit-variance under the healthy hypothesis.
    """

    __slots__ = ("mu", "P", "r", "q")

    def __init__(self, mu: float, var: float, process_var_ratio: float) -> None:
        self.mu = mu
        self.r = var                       # measurement noise ≈ healthy feature variance
        self.q = process_var_ratio * var   # process noise sets the tracking speed
        self.P = var                       # initial state-estimate variance

    def normalized_innovation(self, x: float) -> float:
        """Residual scaled to unit variance: ``(x - mu) / sqrt(P + q + r)``."""
        s = self.P + self.q + self.r
        return (x - self.mu) / (s ** 0.5)

    def learn(self, x: float) -> None:
        """One predict+update step, incorporating ``x`` into the state."""
        p_pred = self.P + self.q
        s = p_pred + self.r
        k = p_pred / s
        self.mu = self.mu + k * (x - self.mu)
        self.P = (1.0 - k) * p_pred


@dataclass(frozen=True)
class KalmanConfig:
    """Configuration for :class:`KalmanDetector`.

    Attributes:
        calibration_frames: Initial (assumed healthy) frames used to seed each
            filter's state and to estimate its measurement-noise variance ``r``.
        process_var_ratio: The process/measurement-noise ratio ``q / r``. This is
            the physical analogue of the EWMA ``alpha``: larger ⇒ the level tracks
            drift faster (and tolerates a shorter fault before absorbing it).
        k: Threshold on the absolute normalized innovation (unit-variance residual).
        min_consecutive: Consecutive deviating frames required to raise a flag.
        clear_consecutive: Consecutive calm frames required to clear a flag.
        min_rms_std: Floor on the RMS measurement std (``r`` floored at its square).
        min_freq_std_hz: Floor on the dominant-frequency measurement std.
    """

    calibration_frames: int = 30
    process_var_ratio: float = 0.01
    k: float = 4.0
    min_consecutive: int = 3
    clear_consecutive: int = 5
    min_rms_std: float = 1e-9
    min_freq_std_hz: float = 0.5

    def __post_init__(self) -> None:
        if self.calibration_frames < 2:
            raise ValueError("calibration_frames must be >= 2")
        if self.process_var_ratio <= 0:
            raise ValueError("process_var_ratio must be > 0")
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.min_consecutive < 1:
            raise ValueError("min_consecutive must be >= 1")
        if self.clear_consecutive < 1:
            raise ValueError("clear_consecutive must be >= 1")
        if self.min_rms_std <= 0 or self.min_freq_std_hz <= 0:
            raise ValueError("sigma floors must be > 0")


class KalmanDetector:
    """Scalar-Kalman detector with freeze-on-anomaly. Implements :class:`Detector`.

    Each feature is tracked by an independent local-level filter; a frame is flagged
    when either feature's absolute normalized innovation exceeds ``k``. The filters
    learn only from healthy-looking frames, so a sustained fault cannot pull the level
    onto itself and mask the anomaly.

    Example:
        >>> det = KalmanDetector(KalmanConfig(calibration_frames=2))
        >>> det.update(1.0, 50.0)
        (False, 0.0)
        >>> det.update(1.0, 50.0)
        (False, 0.0)
        >>> det.is_calibrated
        True
    """

    def __init__(self, config: Optional[KalmanConfig] = None) -> None:
        self.config = config or KalmanConfig()
        self.reset()

    # -- Detector interface ----------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def state(self) -> str:
        """Current lifecycle state: ``calibrating`` / ``normal`` / ``anomalous``."""
        if not self._calibrated:
            return CALIBRATING
        return ANOMALOUS if self._debounce.fired else NORMAL

    def update(self, rms: float, dominant_freq_hz: float) -> Tuple[bool, float]:
        cfg = self.config

        # -- Calibration: seed the filters from the healthy baseline. ----------
        if not self._calibrated:
            self._rms_stats.add(rms)
            self._freq_stats.add(dominant_freq_hz)
            if self._rms_stats.n >= cfg.calibration_frames:
                self._finalize_baseline()
            return (False, 0.0)

        # -- Score with the normalized innovation of each filter. --------------
        nu_rms = self._rms_kf.normalized_innovation(rms)
        nu_freq = self._freq_kf.normalized_innovation(dominant_freq_hz)
        score = float(max(abs(nu_rms), abs(nu_freq)))
        raw_anomalous = abs(nu_rms) > cfg.k or abs(nu_freq) > cfg.k
        fired = self._debounce.update(raw_anomalous)

        # -- Adapt only on healthy-looking frames (freeze-on-anomaly). ---------
        if not raw_anomalous:
            self._rms_kf.learn(rms)
            self._freq_kf.learn(dominant_freq_hz)

        return (fired, score)

    def reset(self) -> None:
        self._calibrated = False
        self._rms_stats = _Welford()
        self._freq_stats = _Welford()
        self._rms_kf: Optional[_ScalarKalman] = None
        self._freq_kf: Optional[_ScalarKalman] = None
        self._debounce = _Debounce(self.config.min_consecutive, self.config.clear_consecutive)

    # -- Internals -------------------------------------------------------------

    def _finalize_baseline(self) -> None:
        cfg = self.config
        rms_var = max(self._rms_stats.std, cfg.min_rms_std) ** 2
        freq_var = max(self._freq_stats.std, cfg.min_freq_std_hz) ** 2
        self._rms_kf = _ScalarKalman(self._rms_stats.mean, rms_var, cfg.process_var_ratio)
        self._freq_kf = _ScalarKalman(self._freq_stats.mean, freq_var, cfg.process_var_ratio)
        self._calibrated = True


__all__ = [
    "EwmaConfig",
    "EwmaDetector",
    "KalmanConfig",
    "KalmanDetector",
]
