"""Rolling feature extraction over a bounded window (no look-ahead).

Given a snapshot of the most recent samples (the *trailing* window
``[t - window_s, t]``), :class:`FeatureExtractor` computes the health features
the brief asks for — RMS, mean, std, and the dominant frequency (FFT peak) — and
packages them into an immutable :class:`FeatureFrame`.

Two deliberate choices, both documented in the README:

* **Recompute per hop, don't accumulate.** Statistics are recomputed from the
  window snapshot each time a frame is emitted (~10 Hz), using numpy's
  pairwise-summation ``mean``/``std``. The tempting O(1) alternative — a running
  ``sum``/``sum-of-squares`` with add/evict — suffers catastrophic cancellation
  on a DC-heavy signal (the CSV strain sits near 1500, so ``E[x**2] ≈ E[x]**2``
  and their difference loses precision, even going negative) and drifts without
  bound on an infinite stream. Recompute is both more robust and, at hop cadence
  over a ~1500-sample window, negligibly cheap.
* **Detrend before the FFT.** The window mean is subtracted before windowing, so
  the DC component cannot dominate the spectrum's ``argmax`` (again, critical for
  the ~1500-offset CSV source). This mirrors the Part 1 test helper.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class FeatureFrame(NamedTuple):
    """One window's worth of rolling features plus its anomaly verdict.

    A ``NamedTuple`` for the same reasons as :class:`~fibersail_edge.sources.Sample`:
    immutable, cheap, and ``._replace``/``._asdict`` make it trivial for Part 3 to
    stamp the anomaly verdict and serialize the record.

    Attributes:
        t: Window-*end* time — the timestamp of the newest sample in the window,
            in seconds from stream start. There is no look-ahead, so ``t`` is the
            latest information the frame reflects.
        raw_value: The newest raw sample value. Emitted once per hop, this doubles
            as the ~10 Hz decimated telemetry stream Part 3 uploads alongside the
            features.
        rms: Root-mean-square of the window (overall vibration energy).
        mean: Window mean.
        std: Window standard deviation.
        dominant_freq_hz: Frequency of the strongest spectral peak (Hz).
        is_anomaly: Whether the detector flagged this frame. Defaults ``False``;
            the processor stamps the real verdict via ``_replace``.
        score: Detector anomaly score (0.0 when healthy/calibrating).
    """

    t: float
    raw_value: float
    rms: float
    mean: float
    std: float
    dominant_freq_hz: float
    is_anomaly: bool = False
    score: float = 0.0


class FeatureExtractor:
    """Computes a :class:`FeatureFrame` from a window snapshot.

    The Hann window and the FFT bin-frequency table are precomputed once at
    construction and reused for every frame, so the per-hop cost is a single
    ``rfft`` plus a few reductions.

    Args:
        window_samples: Number of samples per window ``N``. The FFT resolution is
            ``fs / N`` Hz.
        sample_rate_hz: Sampling rate ``fs`` used to map FFT bins to Hz.
        detrend: Subtract the window mean before the FFT (recommended; prevents a
            DC offset from dominating the peak search).
        interpolate_peak: Refine the peak frequency to sub-bin accuracy with a
            parabolic fit on the log-magnitude spectrum.
    """

    def __init__(
        self,
        window_samples: int,
        sample_rate_hz: float,
        *,
        detrend: bool = True,
        interpolate_peak: bool = True,
    ) -> None:
        if window_samples < 2:
            raise ValueError(f"window_samples must be >= 2, got {window_samples}")
        if sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0, got {sample_rate_hz}")
        self._n = window_samples
        self._fs = sample_rate_hz
        self._detrend = detrend
        self._interpolate = interpolate_peak
        # Precompute once: the Hann taper (leakage suppression) and the bin->Hz map.
        self._hann = np.hanning(window_samples)
        self._freqs = np.fft.rfftfreq(window_samples, d=1.0 / sample_rate_hz)

    # -- Public API ------------------------------------------------------------

    def extract(self, window: np.ndarray, t: float, raw_value: float) -> FeatureFrame:
        """Build a :class:`FeatureFrame` from a window snapshot at time ``t``.

        Args:
            window: The trailing-window samples, oldest-to-newest.
            t: Window-end time (seconds from stream start).
            raw_value: The newest raw sample (carried as the decimated telemetry).
        """
        return FeatureFrame(
            t=float(t),
            raw_value=float(raw_value),
            rms=float(np.sqrt(np.mean(window * window))),
            mean=float(window.mean()),
            std=float(window.std()),
            dominant_freq_hz=self._dominant_frequency(window),
        )

    # -- Internals -------------------------------------------------------------

    def _dominant_frequency(self, window: np.ndarray) -> float:
        """Frequency (Hz) of the strongest spectral component in ``window``.

        Detrend (optional) -> Hann taper -> real FFT -> peak bin -> optional
        parabolic sub-bin refinement. Only the peak *location* is needed, so the
        Hann amplitude normalization is irrelevant.
        """
        w = window - window.mean() if self._detrend else window
        spectrum = np.abs(np.fft.rfft(w * self._hann))
        k = int(np.argmax(spectrum))

        if not self._interpolate:
            return float(self._freqs[k])

        offset = self._parabolic_peak(spectrum, k)
        # Convert the (possibly fractional) bin index to Hz via the uniform grid.
        df = self._fs / self._n
        return float(max(0.0, (k + offset) * df))

    @staticmethod
    def _parabolic_peak(magnitude: np.ndarray, k: int) -> float:
        """Sub-bin offset of the peak from a 3-point parabolic fit (log-magnitude).

        Returns a fractional offset in ``[-0.5, 0.5]`` to add to ``k``. Falls back
        to ``0.0`` at the spectrum edges or when the three points are degenerate.
        """
        if k <= 0 or k >= magnitude.shape[0] - 1:
            return 0.0
        # Log-magnitude gives an exact parabola for a Gaussian-like main lobe.
        eps = 1e-12
        a = np.log(magnitude[k - 1] + eps)
        b = np.log(magnitude[k] + eps)
        c = np.log(magnitude[k + 1] + eps)
        denom = a - 2.0 * b + c
        if denom == 0.0:
            return 0.0
        offset = 0.5 * (a - c) / denom
        # Guard against numerical blow-ups; a real peak sits within +/- half a bin.
        if not np.isfinite(offset) or abs(offset) > 0.5:
            return 0.0
        return float(offset)


__all__ = ["FeatureFrame", "FeatureExtractor"]
