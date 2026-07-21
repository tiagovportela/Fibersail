"""Tests for the detector comparison harness (Part 2 stretch goal).

Covers the two things the comparison must actually deliver: it runs all three
detectors on identical input and scores them, and its drift demonstration shows the
adaptive detectors' advantage — far fewer false positives under benign baseline
drift, while all three still catch the sharp fault.
"""

from __future__ import annotations

from fibersail_edge import DampedOscillatorSensor, FaultConfig, SensorConfig
from fibersail_edge.edge.compare import (
    build_detectors,
    compare_detectors,
    drift_robustness_demo,
    format_comparison,
    format_drift_demo,
    synthesize_drift_features,
)
from fibersail_edge.edge.processor import DetectorConfig, ProcessorConfig


def _sensor_factory() -> DampedOscillatorSensor:
    return DampedOscillatorSensor(
        SensorConfig(
            duration_s=15.0,
            seed=42,
            fault=FaultConfig(start_s=7.0, duration_s=3.0, omega_n_factor=0.7, zeta_factor=0.4),
        )
    )


def test_build_detectors_has_all_three() -> None:
    dets = build_detectors()
    assert set(dets) == {"baseline-zscore", "ewma", "kalman"}


def test_compare_runs_all_three_and_detects_event() -> None:
    config = ProcessorConfig(window_s=1.5, hop_s=0.1, detector=DetectorConfig(k=4.0))
    results = compare_detectors(_sensor_factory, config)

    names = [name for name, _ in results]
    assert names == ["baseline-zscore", "ewma", "kalman"]
    for name, report in results:
        assert report.event_detected, f"{name} missed the fault"
        assert report.windowed.recall > 0.6


def test_format_comparison_renders() -> None:
    config = ProcessorConfig(window_s=1.5, hop_s=0.1)
    text = format_comparison(compare_detectors(_sensor_factory, config))
    for name in ("baseline-zscore", "ewma", "kalman"):
        assert name in text


def test_drift_demo_adaptive_beats_frozen_baseline() -> None:
    """The whole point of the adaptive detectors: robustness to benign drift."""
    features, labels = synthesize_drift_features()
    results = {r.name: r for r in drift_robustness_demo(features, labels)}

    # All three still catch the sharp fault.
    assert all(r.caught_fault for r in results.values())

    frozen_fp = results["baseline-zscore"].false_positives
    # The frozen baseline reads the drift as anomalous many times over.
    assert frozen_fp > 50
    # The adaptive detectors follow the drift and stay largely quiet — at least an
    # order of magnitude fewer drift-induced false positives.
    for name in ("ewma", "kalman"):
        assert results[name].false_positives * 10 < frozen_fp


def test_format_drift_demo_renders() -> None:
    features, labels = synthesize_drift_features()
    text = format_drift_demo(drift_robustness_demo(features, labels))
    assert "Drift robustness" in text
    for name in ("baseline-zscore", "ewma", "kalman"):
        assert name in text
