"""Tests for the honest precision/recall evaluation (Part 2).

The confusion-matrix math and the analytic window-overlap are pure functions with
exact expected values. The guard-band behaviour (windowed labels differ from point
labels at the fault boundaries) is the crux of the "honest evaluation" story.
"""

from __future__ import annotations

import math

from fibersail_edge import (
    DampedOscillatorSensor,
    EdgeProcessor,
    ProcessorConfig,
    SensorConfig,
    evaluate,
    frame_confusion,
    window_faulty_fraction,
)


def test_frame_confusion_counts_and_rates() -> None:
    labels = [True, True, False, False]
    preds = [True, False, True, False]
    r = frame_confusion(labels, preds)
    assert (r.tp, r.fn, r.fp, r.tn) == (1, 1, 1, 1)
    assert r.precision == 0.5
    assert r.recall == 0.5
    assert r.f1 == 0.5
    assert r.false_positive_rate == 0.5
    assert r.support_positive == 2 and r.support_negative == 2


def test_confusion_rates_nan_when_undefined() -> None:
    r = frame_confusion([False, False], [False, False])  # no positives, no predictions
    assert math.isnan(r.precision)  # 0/0
    assert math.isnan(r.recall)


def test_window_faulty_fraction_boundaries() -> None:
    fault = (0.0, 10.0)
    # Trailing window fully inside the fault.
    assert window_faulty_fraction(5.0, 1.0, fault) == 1.0
    # Fully outside (after the fault, past the window length).
    assert window_faulty_fraction(15.0, 1.0, fault) == 0.0
    # Partial overlap: window [9, 11], 1 s of 2 s overlaps → 0.5.
    assert window_faulty_fraction(11.0, 2.0, fault) == 0.5
    # Exactly at the fault end: [8, 10] still fully inside.
    assert window_faulty_fraction(10.0, 2.0, fault) == 1.0


def test_windowed_labeling_creates_guard_band() -> None:
    """Windowed labels lag point labels at both fault boundaries, by design."""
    fault = (7.0, 10.0)
    window_s, guard_frac = 1.5, 0.5

    # Just after onset: point-faulty, but the trailing window is < 50% faulty.
    t_onset = 7.5
    point = fault[0] <= t_onset < fault[1]
    windowed = window_faulty_fraction(t_onset, window_s, fault) >= guard_frac
    assert point is True and windowed is False

    # Just after the fault ends: point-healthy, but the window is still majority faulty.
    t_offset = 10.5
    point = fault[0] <= t_offset < fault[1]
    windowed = window_faulty_fraction(t_offset, window_s, fault) >= guard_frac
    assert point is False and windowed is True


def test_evaluate_handles_healthy_source() -> None:
    """No ground truth → recall undefined, no event, no latency; FP still measured."""
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=6.0, seed=3))  # no fault
    report = evaluate(sensor, EdgeProcessor(ProcessorConfig(window_s=1.5, hop_s=0.1)))
    assert report.fault_window is None
    assert report.event_detected is False
    assert report.detection_latency_s is None
    assert report.raw.support_positive == 0
    assert math.isnan(report.raw.recall)


def test_evaluate_is_repeatable() -> None:
    def run() -> tuple:
        sensor = DampedOscillatorSensor(SensorConfig(duration_s=8.0, seed=42))
        rep = evaluate(sensor, EdgeProcessor(ProcessorConfig(window_s=1.5, hop_s=0.1)))
        return (rep.n_frames, rep.windowed.tp, rep.windowed.fp)

    assert run() == run()


def test_format_summary_renders() -> None:
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=6.0, seed=3))
    report = evaluate(sensor, EdgeProcessor())
    text = report.format_summary()
    assert "detection evaluation" in text
    assert "precision" in text and "recall" in text
