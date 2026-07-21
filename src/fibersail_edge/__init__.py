"""fibersail_edge — edge sensor processing & cloud sync.

* **Part 1** — the synthetic vibration sensor and the shared streaming
  data-source interface (:class:`Sample`, :class:`SampleSource`).
* **Part 2** — the streaming edge processor: a bounded :class:`RingBuffer`, a
  :class:`FeatureExtractor` (RMS/mean/std/dominant-frequency), a streaming
  :class:`BaselineZScoreDetector`, the :class:`EdgeProcessor` that ties them
  together, an honest precision/recall :func:`evaluate`, and the
  :class:`FrameSink` seam that decouples the edge loop from Part 3's cloud sync.
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
from .sensor import DampedOscillatorSensor, FaultConfig, SensorConfig
from .sink import CallbackSink, FrameSink, ListSink
from .sources import CsvReplaySource, Sample, SampleSource

__all__ = [
    # Part 1 — sources & sensor
    "Sample",
    "SampleSource",
    "CsvReplaySource",
    "SensorConfig",
    "FaultConfig",
    "DampedOscillatorSensor",
    # Part 2 — edge processing
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

__version__ = "0.2.0"
