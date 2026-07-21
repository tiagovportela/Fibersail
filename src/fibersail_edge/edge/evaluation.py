"""Honest offline evaluation of the detector against the ground-truth fault.

Ground-truth labels come *only* from ``source.fault_window`` and are computed
here, in the scorer — they are never fed into the detector, preserving the Part 1
invariant that a detector cannot peek at labels.

Why more than one precision/recall number
------------------------------------------
The features are computed over a *trailing* window with no look-ahead, which
creates two structural, unavoidable artifacts at the fault boundaries:

* **Onset lag → false negatives → depressed recall.** Right after the fault
  starts, the window is still mostly healthy samples, so RMS/frequency haven't
  shifted enough to trip the detector. Those frames are labeled faulty but can't
  be caught without look-ahead.
* **Trailing lag → false positives → depressed precision.** Right after the fault
  ends, the window still contains faulty samples, so the detector (correctly,
  given its input) keeps firing. Those frames are labeled healthy.

Reporting only the raw frame-level numbers would understate the detector; hiding
the effect by silently shifting labels would be dishonest. So :func:`evaluate`
reports **four complementary views** side by side:

1. **Raw** frame-level metrics (point labels ``is_faulty(frame.t)``) — the
   pessimistic, honest floor.
2. **Windowed** metrics — a frame is labeled faulty iff its trailing window is
   *majority* faulty (``window_faulty_fraction >= guard_frac``). This is a
   first-principles guard band of ``window_s * guard_frac`` at each boundary,
   derived from how the detector actually sees data — not a hand-tuned fudge.
3. **Detection latency** vs. a derived reference (``window_s * guard_frac`` to
   reach a majority-faulty window, plus the debounce delay). A latency near the
   reference means the detector is about as fast as a causal window allows;
   *below* it is possible and good — a strong fault (a big frequency drop) trips
   the threshold before the window is even majority-faulty.
4. **Event detected** — the operational question: did we raise *any* flag during
   the fault (allowing for the trailing window)?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from .processor import EdgeProcessor
from ..sources import SampleSource


class _Confusion:
    """Mutable confusion-matrix accumulator (O(1) memory)."""

    def __init__(self) -> None:
        self.tp = self.fp = self.tn = self.fn = 0

    def add(self, label: bool, pred: bool) -> None:
        if label and pred:
            self.tp += 1
        elif label and not pred:
            self.fn += 1
        elif not label and pred:
            self.fp += 1
        else:
            self.tn += 1

    def result(self) -> "EvalResult":
        return EvalResult(tp=self.tp, fp=self.fp, tn=self.tn, fn=self.fn)


@dataclass(frozen=True)
class EvalResult:
    """A confusion matrix and the rates derived from it.

    Ratios return ``nan`` when their denominator is zero (e.g. recall with no
    positive labels); the CLI renders those as ``n/a`` and always prints the raw
    counts, so a degenerate "never fires" detector is visible rather than hidden.
    """

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else math.nan

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else math.nan

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return math.nan
        return 2.0 * p * r / (p + r)

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else math.nan

    @property
    def support_positive(self) -> int:
        return self.tp + self.fn

    @property
    def support_negative(self) -> int:
        return self.tn + self.fp


def frame_confusion(labels: Iterable[bool], preds: Iterable[bool]) -> EvalResult:
    """Confusion matrix for paired ``(label, pred)`` booleans. Pure, order-agnostic."""
    conf = _Confusion()
    for label, pred in zip(labels, preds):
        conf.add(bool(label), bool(pred))
    return conf.result()


def window_faulty_fraction(
    t: float, window_s: float, fault_window: Tuple[float, float]
) -> float:
    """Fraction of the trailing window ``[t - window_s, t]`` that is faulty.

    Analytic overlap of the trailing window with ``fault_window = (start, end)``,
    normalized by ``window_s``. Returns a value in ``[0, 1]``. O(1) — no sampling.
    """
    start, end = fault_window
    lo = max(t - window_s, start)
    hi = min(t, end)
    overlap = max(0.0, hi - lo)
    return overlap / window_s


@dataclass(frozen=True)
class EvaluationReport:
    """The full result of scoring one run. See module docstring for the rationale."""

    detector_name: str
    window_s: float
    guard_s: float
    feature_rate_hz: float
    fault_window: Optional[Tuple[float, float]]
    n_frames: int
    raw: EvalResult
    windowed: EvalResult
    detection_latency_s: Optional[float]
    latency_reference_s: float
    event_detected: bool

    def format_summary(self) -> str:
        """A plain-text report suitable for a CLI or a log line."""

        def rate(x: float) -> str:
            return "n/a" if math.isnan(x) else f"{x:6.3f}"

        def block(title: str, r: EvalResult) -> str:
            return (
                f"  {title}\n"
                f"    TP={r.tp:<5d} FP={r.fp:<5d} FN={r.fn:<5d} TN={r.tn:<5d}\n"
                f"    precision={rate(r.precision)}  recall={rate(r.recall)}  "
                f"F1={rate(r.f1)}  FPR={rate(r.false_positive_rate)}"
            )

        fw = (
            f"{self.fault_window[0]:.2f}–{self.fault_window[1]:.2f} s"
            if self.fault_window is not None
            else "none (healthy data)"
        )
        latency = (
            f"{self.detection_latency_s:.2f} s (reference ≈ {self.latency_reference_s:.2f} s)"
            if self.detection_latency_s is not None
            else "not detected"
        )
        lines = [
            "Edge processing — detection evaluation",
            "=" * 44,
            f"  detector       : {self.detector_name}",
            f"  window         : {self.window_s:.2f} s   guard band: {self.guard_s:.2f} s",
            f"  frame rate     : {self.feature_rate_hz:.1f} Hz",
            f"  fault window   : {fw}",
            f"  frames scored  : {self.n_frames}",
            f"  event detected : {'YES' if self.event_detected else 'NO'}",
            f"  detect latency : {latency}",
            "",
            block("raw (point labels — pessimistic floor):", self.raw),
            "",
            block(f"windowed (majority-faulty, ±{self.guard_s:.2f} s guard):", self.windowed),
        ]
        return "\n".join(lines)


def evaluate(
    source: SampleSource,
    processor: EdgeProcessor,
    *,
    guard_frac: float = 0.5,
) -> EvaluationReport:
    """Run ``processor`` over ``source`` once and score it against the fault window.

    A single streaming, O(1)-memory pass — frames are consumed and scored on the
    fly, never buffered. ``source`` must be finite (set ``duration_s`` on the
    sensor, or use the CSV replay). The processor is reset first so the call is
    repeatable.

    Args:
        guard_frac: A frame counts as faulty in the *windowed* view when at least
            this fraction of its trailing window overlaps the fault. ``0.5``
            (majority-faulty) yields a symmetric guard band of ``window_s / 2``.
    """
    if not (0.0 <= guard_frac <= 1.0):
        raise ValueError("guard_frac must be in [0, 1]")

    processor.reset()
    window_s = processor.window_s
    guard_s = window_s * guard_frac
    fault_window = source.fault_window
    fault_start = fault_window[0] if fault_window is not None else None
    fault_end = fault_window[1] if fault_window is not None else None

    raw = _Confusion()
    windowed = _Confusion()
    n_frames = 0
    first_detection_t: Optional[float] = None
    event_detected = False

    for frame in processor.process_stream(source):
        n_frames += 1
        pred = frame.is_anomaly

        if fault_window is None:
            point_label = False
            window_label = False
        else:
            point_label = fault_start <= frame.t < fault_end
            window_label = window_faulty_fraction(frame.t, window_s, fault_window) >= guard_frac
            if pred and frame.t >= fault_start:
                if first_detection_t is None:
                    first_detection_t = frame.t
                if frame.t < fault_end + window_s:
                    event_detected = True

        raw.add(point_label, pred)
        windowed.add(window_label, pred)

    detection_latency_s = (
        first_detection_t - fault_start
        if (first_detection_t is not None and fault_start is not None)
        else None
    )
    # Reference latency: time for the window to become majority-faulty plus the
    # debounce-to-fire delay. Observed latency near this ⇒ near-optimal for a
    # causal window; below it ⇒ a strong fault tripped the threshold on a
    # partially-faulty window (good).
    min_consecutive = processor.config.detector.min_consecutive
    latency_reference_s = guard_s + min_consecutive / processor.feature_rate_hz

    return EvaluationReport(
        detector_name=type(processor.detector).__name__,
        window_s=window_s,
        guard_s=guard_s,
        feature_rate_hz=processor.feature_rate_hz,
        fault_window=fault_window,
        n_frames=n_frames,
        raw=raw.result(),
        windowed=windowed.result(),
        detection_latency_s=detection_latency_s,
        latency_reference_s=latency_reference_s,
        event_detected=event_detected,
    )


__all__ = [
    "EvalResult",
    "EvaluationReport",
    "frame_confusion",
    "window_faulty_fraction",
    "evaluate",
]
