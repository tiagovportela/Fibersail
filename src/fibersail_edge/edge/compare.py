"""Head-to-head detector comparison — the stretch-goal "compare EWMA/Kalman".

Runs the three detectors — the frozen-baseline z-score, the EWMA control chart, and
the scalar Kalman filter — over the **same** faulted stream, through the **same**
window/hop and the **same** debounce, and prints their honest evaluation metrics side
by side. Only the baseline model differs, so the table is a fair comparison.

    uv run python -m fibersail_edge.edge.compare
    uv run python -m fibersail_edge.edge.compare --fault-start-s 7 --k 4.0

Each detector is scored by :func:`~fibersail_edge.edge.evaluation.evaluate`, which
labels frames only from the ground-truth fault window (never fed to the detector) and
reports the windowed (causal-honest) precision/recall/F1/FPR plus detection latency.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .adaptive import EwmaConfig, EwmaDetector, KalmanConfig, KalmanDetector
from .detector import BaselineZScoreDetector, Detector, DetectorConfig
from .evaluation import EvaluationReport, evaluate
from .processor import EdgeProcessor, ProcessorConfig
from ..sources import SampleSource


def build_detectors(
    *,
    calibration_frames: int = 30,
    k: float = 4.0,
) -> Dict[str, Detector]:
    """The three detectors under test, each configured identically where it matters.

    Same ``calibration_frames``, ``k``, and (via their defaults) the same debounce, so
    the only difference is how each maintains its healthy baseline.
    """
    return {
        "baseline-zscore": BaselineZScoreDetector(
            DetectorConfig(calibration_frames=calibration_frames, k=k)
        ),
        "ewma": EwmaDetector(EwmaConfig(calibration_frames=calibration_frames, k=k)),
        "kalman": KalmanDetector(KalmanConfig(calibration_frames=calibration_frames, k=k)),
    }


def compare_detectors(
    source_factory: Callable[[], SampleSource],
    processor_config: ProcessorConfig,
    *,
    detectors: Dict[str, Detector] | None = None,
    guard_frac: float = 0.5,
) -> List[Tuple[str, EvaluationReport]]:
    """Evaluate every detector on a fresh copy of the same stream.

    ``source_factory`` is called once per detector so each sees byte-identical input
    (same seed) with no state carried over. Returns ``(name, report)`` pairs in the
    order given by ``detectors``.
    """
    detectors = detectors or build_detectors(
        calibration_frames=processor_config.detector.calibration_frames,
        k=processor_config.detector.k,
    )
    results: List[Tuple[str, EvaluationReport]] = []
    for name, detector in detectors.items():
        processor = EdgeProcessor(processor_config, detector=detector)
        report = evaluate(source_factory(), processor, guard_frac=guard_frac)
        results.append((name, report))
    return results


def format_comparison(results: List[Tuple[str, EvaluationReport]]) -> str:
    """Render the comparison as an aligned plain-text table."""

    def num(x: float) -> str:
        return "  n/a" if x != x else f"{x:5.3f}"  # x != x ⇒ NaN

    def latency(r: EvaluationReport) -> str:
        return f"{r.detection_latency_s:.2f}s" if r.detection_latency_s is not None else "  —  "

    header = (
        f"  {'detector':<16}{'event':>6}{'latency':>9}"
        f"{'   |   '}{'wP':>5} {'wR':>5} {'wF1':>5} {'wFPR':>5}"
        f"{'   |   '}{'rawF1':>6}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [
        "Detector comparison — same stream, same window, same debounce",
        "=" * 62,
        header,
        sep,
    ]
    for name, r in results:
        lines.append(
            f"  {name:<16}{('YES' if r.event_detected else 'no'):>6}{latency(r):>9}"
            f"   |   {num(r.windowed.precision)} {num(r.windowed.recall)} "
            f"{num(r.windowed.f1)} {num(r.windowed.false_positive_rate)}"
            f"   |   {num(r.raw.f1):>6}"
        )
    lines += [
        sep,
        "  wP/wR/wF1/wFPR = windowed (majority-faulty, causal-honest) metrics.",
        "  Better = higher wF1, lower wFPR, lower latency. Same k and debounce",
        "  throughout, so differences are the baseline model alone.",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class DriftResult:
    """Per-detector outcome on the synthetic drift stream."""

    name: str
    caught_fault: bool
    false_positives: int   # flags outside the fault window (drift-induced)
    healthy_frames: int    # denominator for the false-positive count


def synthesize_drift_features(
    *,
    n_frames: int = 400,
    drift_hz: float = 5.0,
    baseline_freq_hz: float = 50.0,
    fault_freq_hz: float = 30.0,
    fault_start: int = 260,
    fault_len: int = 40,
    noise_hz: float = 0.15,
    seed: int = 7,
) -> Tuple[List[Tuple[float, float]], List[bool]]:
    """A ``(rms, dominant_freq)`` feature stream with *benign* baseline drift + a fault.

    The dominant frequency drifts slowly and linearly by ``drift_hz`` across the run —
    the kind of slow shift a real machine shows with temperature/load — with a sharp
    downward fault stamped into ``[fault_start, fault_start + fault_len)``. RMS is held
    roughly constant so the frequency channel carries the signal. Returns the feature
    pairs and their ground-truth fault labels. Deterministic given ``seed``.
    """
    rng = random.Random(seed)
    features: List[Tuple[float, float]] = []
    labels: List[bool] = []
    for i in range(n_frames):
        faulty = fault_start <= i < fault_start + fault_len
        drift = drift_hz * (i / (n_frames - 1))
        freq = (fault_freq_hz if faulty else baseline_freq_hz + drift) + rng.gauss(0.0, noise_hz)
        rms = 1.0 + rng.gauss(0.0, 0.01)
        features.append((rms, freq))
        labels.append(faulty)
    return features, labels


def drift_robustness_demo(
    features: List[Tuple[float, float]],
    labels: List[bool],
    *,
    detectors: Dict[str, Detector] | None = None,
) -> List[DriftResult]:
    """Run each detector over a pre-computed feature stream; tally drift false positives.

    A frozen baseline eventually reads the benign drift as an anomaly (false positives
    climb); the adaptive detectors follow the drift and should stay quiet outside the
    fault while still catching it.
    """
    detectors = detectors or build_detectors()
    results: List[DriftResult] = []
    for name, det in detectors.items():
        det.reset()
        caught = False
        fp = healthy = 0
        for (rms, freq), faulty in zip(features, labels):
            flag, _ = det.update(rms, freq)
            if faulty:
                caught = caught or flag
            else:
                healthy += 1
                if flag:
                    fp += 1
        results.append(DriftResult(name, caught, fp, healthy))
    return results


def format_drift_demo(results: List[DriftResult]) -> str:
    """Render the drift-robustness tally as an aligned table."""
    lines = [
        "Drift robustness — benign baseline drift + one fault",
        "=" * 62,
        f"  {'detector':<16}{'fault caught':>14}{'drift false-positives':>24}",
        "  " + "-" * 52,
    ]
    for r in results:
        fp = f"{r.false_positives}/{r.healthy_frames}"
        lines.append(f"  {r.name:<16}{('YES' if r.caught_fault else 'no'):>14}{fp:>24}")
    lines += [
        "  " + "-" * 52,
        "  A frozen baseline flags the slow drift; the adaptive detectors follow",
        "  it and stay quiet, while all three still catch the sharp fault.",
    ]
    return "\n".join(lines)


def main() -> None:
    from ..sensor import DampedOscillatorSensor, FaultConfig, SensorConfig

    parser = argparse.ArgumentParser(description="Compare anomaly detectors on one faulted stream.")
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--sample-rate-hz", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fault-start-s", type=float, default=7.0)
    parser.add_argument("--fault-duration-s", type=float, default=3.0)
    parser.add_argument("--omega-factor", type=float, default=0.7)
    parser.add_argument("--zeta-factor", type=float, default=0.4)
    parser.add_argument("--window-s", type=float, default=1.5)
    parser.add_argument("--hop-s", type=float, default=0.1)
    parser.add_argument("--k", type=float, default=4.0)
    parser.add_argument("--calibration-frames", type=int, default=30)
    args = parser.parse_args()

    def source_factory() -> SampleSource:
        return DampedOscillatorSensor(
            SensorConfig(
                sample_rate_hz=args.sample_rate_hz,
                duration_s=args.duration_s,
                seed=args.seed,
                fault=FaultConfig(
                    start_s=args.fault_start_s,
                    duration_s=args.fault_duration_s,
                    omega_n_factor=args.omega_factor,
                    zeta_factor=args.zeta_factor,
                ),
            )
        )

    processor_config = ProcessorConfig(
        sample_rate_hz=args.sample_rate_hz,
        window_s=args.window_s,
        hop_s=args.hop_s,
        detector=DetectorConfig(k=args.k, calibration_frames=args.calibration_frames),
    )
    results = compare_detectors(source_factory, processor_config)

    fault = f"{args.fault_start_s:.1f}–{args.fault_start_s + args.fault_duration_s:.1f}s"
    print(f"Stream: {args.duration_s:.0f}s @ {args.sample_rate_hz:.0f} Hz, fault {fault} "
          f"(seed {args.seed}, k={args.k})")
    print()
    print(format_comparison(results))
    print()
    features, labels = synthesize_drift_features()
    print(format_drift_demo(drift_robustness_demo(features, labels)))


if __name__ == "__main__":
    main()
