"""Tests for the adaptive-baseline detectors (EWMA and Kalman).

Mirrors the baseline detector's contract tests — protocol conformance, catches the
known fault, quiet on a healthy stream, calibration suppression, O(1) memory — and
adds the property that motivates these variants: **freeze-on-anomaly** keeps a
sustained fault from being absorbed into the adaptive baseline.
"""

from __future__ import annotations

import itertools
import tracemalloc
from typing import Callable

import pytest

from fibersail_edge import (
    DampedOscillatorSensor,
    Detector,
    EdgeProcessor,
    EwmaConfig,
    EwmaDetector,
    FaultConfig,
    KalmanConfig,
    KalmanDetector,
    ProcessorConfig,
    SensorConfig,
    evaluate,
)

# Each entry builds a fresh adaptive detector with the given calibration frames.
DetectorFactory = Callable[[int], Detector]

ADAPTIVE: list[tuple[str, DetectorFactory]] = [
    ("ewma", lambda cal: EwmaDetector(EwmaConfig(calibration_frames=cal))),
    ("kalman", lambda cal: KalmanDetector(KalmanConfig(calibration_frames=cal))),
]
IDS = [name for name, _ in ADAPTIVE]


def _faulted_sensor(seed: int = 42) -> DampedOscillatorSensor:
    return DampedOscillatorSensor(
        SensorConfig(
            duration_s=15.0,
            seed=seed,
            fault=FaultConfig(start_s=7.0, duration_s=3.0, omega_n_factor=0.7, zeta_factor=0.4),
        )
    )


@pytest.mark.parametrize("name,factory", ADAPTIVE, ids=IDS)
def test_implements_protocol(name: str, factory: DetectorFactory) -> None:
    assert isinstance(factory(30), Detector)


@pytest.mark.parametrize("name,factory", ADAPTIVE, ids=IDS)
def test_detects_injected_fault(name: str, factory: DetectorFactory) -> None:
    """Same brief-required bar as the baseline detector, same loose thresholds."""
    sensor = _faulted_sensor()
    processor = EdgeProcessor(ProcessorConfig(window_s=1.5, hop_s=0.1), detector=factory(30))
    report = evaluate(sensor, processor)

    assert report.detector_name == type(processor.detector).__name__
    assert report.event_detected
    assert report.detection_latency_s is not None
    assert report.windowed.recall > 0.6
    assert report.windowed.precision > 0.5


@pytest.mark.parametrize("name,factory", ADAPTIVE, ids=IDS)
def test_quiet_on_healthy_stream(name: str, factory: DetectorFactory) -> None:
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=15.0, seed=1))  # no fault
    processor = EdgeProcessor(ProcessorConfig(window_s=1.5, hop_s=0.1), detector=factory(30))
    frames = list(processor.process_stream(sensor))
    flagged = sum(1 for f in frames if f.is_anomaly)
    assert flagged / len(frames) < 0.05


@pytest.mark.parametrize("name,factory", ADAPTIVE, ids=IDS)
def test_calibration_suppresses_early_flags(name: str, factory: DetectorFactory) -> None:
    det = factory(10)
    for _ in range(10):
        is_anomaly, score = det.update(rms=1e6, dominant_freq_hz=1.0)  # absurd, but calibrating
        assert is_anomaly is False and score == 0.0
    assert det.is_calibrated


@pytest.mark.parametrize("name,factory", ADAPTIVE, ids=IDS)
def test_freeze_on_anomaly_keeps_sustained_fault_flagged(name: str, factory: DetectorFactory) -> None:
    """A sustained deviation must NOT be absorbed into the adaptive baseline.

    Without freeze-on-anomaly the adaptive mean would chase the fault value and the
    flag would clear within ~1/alpha frames; with it, the flag persists for as long
    as the deviation does.
    """
    det = factory(5)
    for _ in range(5):
        det.update(rms=1.0, dominant_freq_hz=50.0)  # calibrate on a constant baseline
    assert det.is_calibrated

    fired_history = [det.update(rms=1.0, dominant_freq_hz=30.0)[0] for _ in range(120)]
    assert any(fired_history[:10]), "should raise the flag shortly after the step"
    assert fired_history[-1], "freeze-on-anomaly must keep the sustained fault flagged"
    assert all(fired_history[10:]), "flag must not clear while the deviation persists"


@pytest.mark.parametrize("name,factory", ADAPTIVE, ids=IDS)
def test_memory_is_bounded(name: str, factory: DetectorFactory) -> None:
    det = factory(20)

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


def test_ewma_config_validation() -> None:
    for bad in (
        dict(calibration_frames=1),
        dict(alpha=0.0),
        dict(alpha=1.5),
        dict(k=0.0),
        dict(min_consecutive=0),
        dict(clear_consecutive=0),
        dict(min_rms_std=0.0),
        dict(min_freq_std_hz=-1.0),
    ):
        with pytest.raises(ValueError):
            EwmaConfig(**bad)


def test_kalman_config_validation() -> None:
    for bad in (
        dict(calibration_frames=1),
        dict(process_var_ratio=0.0),
        dict(process_var_ratio=-0.1),
        dict(k=0.0),
        dict(min_consecutive=0),
        dict(clear_consecutive=0),
        dict(min_rms_std=0.0),
        dict(min_freq_std_hz=-1.0),
    ):
        with pytest.raises(ValueError):
            KalmanConfig(**bad)


def test_reset_returns_to_calibrating() -> None:
    for _, factory in ADAPTIVE:
        det = factory(3)
        for _ in range(3):
            det.update(rms=1.0, dominant_freq_hz=50.0)
        assert det.is_calibrated
        det.reset()
        assert not det.is_calibrated
        assert det.state == "calibrating"
