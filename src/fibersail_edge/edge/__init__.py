"""Part 2 — the streaming edge processing service.

The pipeline, in dependency order:

    Sample ─▶ RingBuffer ─▶ FeatureExtractor ─▶ Detector ─▶ FeatureFrame ─▶ FrameSink ─▶ (Part 3)

- :mod:`~fibersail_edge.edge.ring_buffer` — bounded, O(1)/sample rolling window.
- :mod:`~fibersail_edge.edge.features` — RMS/mean/std + dominant-frequency (FFT) features.
- :mod:`~fibersail_edge.edge.detector` — streaming, self-calibrating anomaly detector.
- :mod:`~fibersail_edge.edge.sink` — the non-blocking seam that decouples Part 3.
- :mod:`~fibersail_edge.edge.processor` — ties it together into ``EdgeProcessor``.
- :mod:`~fibersail_edge.edge.evaluation` — honest precision/recall scoring.
- :mod:`~fibersail_edge.edge.benchmark` — throughput harness (``python -m fibersail_edge.edge.benchmark``).
"""

from __future__ import annotations

from .detector import BaselineZScoreDetector, Detector, DetectorConfig
from .evaluation import (
    EvalResult,
    EvaluationReport,
    evaluate,
    frame_confusion,
    window_faulty_fraction,
)
from .features import FeatureExtractor, FeatureFrame
from .processor import EdgeProcessor, ProcessorConfig
from .ring_buffer import RingBuffer
from .sink import CallbackSink, FrameSink, ListSink

__all__ = [
    "RingBuffer",
    "FeatureFrame",
    "FeatureExtractor",
    "Detector",
    "DetectorConfig",
    "BaselineZScoreDetector",
    "FrameSink",
    "ListSink",
    "CallbackSink",
    "ProcessorConfig",
    "EdgeProcessor",
    "EvalResult",
    "EvaluationReport",
    "frame_confusion",
    "window_faulty_fraction",
    "evaluate",
]
