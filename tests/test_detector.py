"""Tests for the streaming anomaly detector (Part 2).

The headline test is the brief-required one: run the detector against the known
fault-injection window and check it actually finds it, with an honest (untuned)
precision/recall bar. The rest pin down calibration, debounce, robustness, and
the O(1)-memory guarantee.
"""

from __future__ import annotations

import itertools
import tracemalloc

import pytest

from fibersail_edge import (
    BaselineZScoreDetector,
    DampedOscillatorSensor,
    Detector,
    DetectorConfig,
    EdgeProcessor,
    FaultConfig,
    ProcessorConfig,
    SensorConfig,
    evaluate,
)


def _faulted_sensor(seed: int = 42) -> DampedOscillatorSensor:
    return DampedOscillatorSensor(
        SensorConfig(
            duration_s=15.0,
            seed=seed,
            fault=FaultConfig(start_s=7.0, duration_s=3.0, omega_n_factor=0.7, zeta_factor=0.4),
        )
    )


def test_detector_implements_protocol() -> None:
    assert isinstance(BaselineZScoreDetector(), Detector)


def test_detects_injected_fault() -> None:
    """Brief-required: the detector must find the known fault window.

    Bars are deliberately loose/qualitative (not tuned to a magic number): the
    fault is caught as an event, most faulty windows are flagged, and false alarms
    stay in check. The windowed view is the honest one for a causal detector.
    """
    sensor = _faulted_sensor()
    processor = EdgeProcessor(ProcessorConfig(window_s=1.5, hop_s=0.1))
    report = evaluate(sensor, processor)

    assert report.event_detected
    assert report.detection_latency_s is not None
    assert report.windowed.recall > 0.6
    assert report.windowed.precision > 0.5
    # A True flag actually lands inside the ground-truth fault window.
    assert report.detection_latency_s < report.fault_window[1] - report.fault_window[0] + report.window_s


def test_no_anomalies_on_healthy_stream() -> None:
    """A healthy stream should raise essentially no flags."""
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=15.0, seed=1))  # no fault
    processor = EdgeProcessor(ProcessorConfig(window_s=1.5, hop_s=0.1))
    frames = list(processor.process_stream(sensor))
    flagged = sum(1 for f in frames if f.is_anomaly)
    assert flagged / len(frames) < 0.05


def test_calibration_suppresses_early_flags() -> None:
    """No flag is raised while calibrating, even on wild inputs."""
    det = BaselineZScoreDetector(DetectorConfig(calibration_frames=10))
    for _ in range(10):
        is_anomaly, score = det.update(rms=1e6, dominant_freq_hz=1.0)  # absurd, but calibrating
        assert is_anomaly is False and score == 0.0
    assert det.is_calibrated


def test_zero_std_baseline_does_not_divide_by_zero() -> None:
    """A perfectly constant baseline (std 0) must not blow up; floors protect it."""
    det = BaselineZScoreDetector(DetectorConfig(calibration_frames=5, min_consecutive=1))
    for _ in range(5):
        det.update(rms=1.0, dominant_freq_hz=50.0)  # constant → std == 0
    # A large deviation is finite-scored and (eventually) flagged, no exception.
    is_anomaly, score = det.update(rms=100.0, dominant_freq_hz=50.0)
    assert score > 0.0
    assert is_anomaly is True


def test_debounce_requires_consecutive_frames() -> None:
    det = BaselineZScoreDetector(DetectorConfig(calibration_frames=5, k=3.0, min_consecutive=3))
    for _ in range(5):
        det.update(rms=1.0, dominant_freq_hz=50.0)
    # One deviating frame is not enough to fire.
    assert det.update(rms=50.0, dominant_freq_hz=50.0)[0] is False
    assert det.update(rms=50.0, dominant_freq_hz=50.0)[0] is False
    # Third consecutive deviating frame fires.
    assert det.update(rms=50.0, dominant_freq_hz=50.0)[0] is True


def test_config_validation() -> None:
    for bad in (
        dict(calibration_frames=1),
        dict(k=0.0),
        dict(min_consecutive=0),
        dict(clear_consecutive=0),
        dict(min_rms_std=0.0),
        dict(min_freq_std_hz=-1.0),
    ):
        with pytest.raises(ValueError):
            DetectorConfig(**bad)


def test_detector_memory_is_bounded() -> None:
    det = BaselineZScoreDetector(DetectorConfig(calibration_frames=20))

    def peak_for(n: int) -> int:
        tracemalloc.start()
        for i in itertools.islice(itertools.count(), n):
            det.update(rms=1.0 + (i % 3), dominant_freq_hz=50.0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    peak_small = peak_for(2_000)
    peak_large = peak_for(20_000)
    assert peak_large < peak_small + 50_000
