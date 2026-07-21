"""Tests for rolling feature extraction (Part 2).

Feature correctness is checked against signals with known answers (pure sinusoids),
and the two subtle requirements are pinned down: detrending (so a DC offset can't
hijack the FFT peak) and no look-ahead (features depend only on the given window).
"""

from __future__ import annotations

import numpy as np

from fibersail_edge import FeatureExtractor, FeatureFrame


def _sinusoid(freq_hz: float, fs: float, n: int, amplitude: float = 1.0, offset: float = 0.0) -> np.ndarray:
    t = np.arange(n) / fs
    return offset + amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def test_dominant_frequency_recovers_pure_sinusoid() -> None:
    fs, n = 1000.0, 1000
    w = _sinusoid(50.0, fs, n)
    fx = FeatureExtractor(n, fs)
    frame = fx.extract(w, t=1.0, raw_value=float(w[-1]))
    assert abs(frame.dominant_freq_hz - 50.0) < 1.5  # within a bin or so


def test_rms_of_sinusoid() -> None:
    fs, n = 1000.0, 1000
    amplitude = 2.0
    w = _sinusoid(50.0, fs, n, amplitude=amplitude)  # integer number of periods
    fx = FeatureExtractor(n, fs)
    frame = fx.extract(w, t=1.0, raw_value=float(w[-1]))
    assert abs(frame.rms - amplitude / np.sqrt(2.0)) < 0.02


def test_mean_std_match_numpy() -> None:
    fs, n = 1000.0, 512
    rng = np.random.default_rng(0)
    w = rng.normal(3.0, 1.7, size=n)
    fx = FeatureExtractor(n, fs)
    frame = fx.extract(w, t=0.5, raw_value=float(w[-1]))
    assert abs(frame.mean - w.mean()) < 1e-9
    assert abs(frame.std - w.std()) < 1e-9


def test_detrending_ignores_dc_offset() -> None:
    """A large DC offset (like the ~1500 CSV strain baseline) must not win the FFT."""
    fs, n = 1000.0, 1000
    w = _sinusoid(50.0, fs, n, amplitude=1.0, offset=1500.0)
    fx = FeatureExtractor(n, fs, detrend=True)
    frame = fx.extract(w, t=1.0, raw_value=float(w[-1]))
    assert abs(frame.dominant_freq_hz - 50.0) < 1.5  # not 0 Hz (the DC bin)


def test_parabolic_interpolation_improves_offbin_freq() -> None:
    fs, n = 1000.0, 1000  # 1 Hz bins → 50.4 Hz falls between bins
    true_freq = 50.4
    w = _sinusoid(true_freq, fs, n)
    coarse = FeatureExtractor(n, fs, interpolate_peak=False).extract(w, 1.0, 0.0)
    refined = FeatureExtractor(n, fs, interpolate_peak=True).extract(w, 1.0, 0.0)
    assert abs(refined.dominant_freq_hz - true_freq) < abs(coarse.dominant_freq_hz - true_freq)


def test_no_lookahead_features_depend_only_on_window() -> None:
    """Features for a trailing window are identical regardless of any future samples.

    Extracting over a window slice must give the same result whether that slice is
    the tail of a short array or of a longer one — i.e. only past/current data is used.
    """
    fs, n = 1000.0, 256
    rng = np.random.default_rng(1)
    long_signal = rng.normal(size=n + 500)
    window = long_signal[:n].copy()  # the "trailing window up to now"
    fx = FeatureExtractor(n, fs)

    frame_now = fx.extract(window, t=n / fs, raw_value=float(window[-1]))
    # The same window, computed after 500 more (future) samples exist — unchanged.
    frame_again = fx.extract(long_signal[:n], t=n / fs, raw_value=float(window[-1]))
    assert frame_now == frame_again


def test_returns_feature_frame() -> None:
    fx = FeatureExtractor(64, 1000.0)
    frame = fx.extract(np.ones(64), t=0.0, raw_value=1.0)
    assert isinstance(frame, FeatureFrame)
    assert frame.is_anomaly is False and frame.score == 0.0  # defaults before detection
