"""fibersail_edge — edge sensor processing & cloud sync.

Organized by exercise part, over one shared streaming contract:

* :mod:`~fibersail_edge.sources` — the shared contract (``Sample``, ``SampleSource``,
  ``CsvReplaySource``) that decouples every part from where the data comes from.
* :mod:`fibersail_edge.sensor` — **Part 1**: the synthetic vibration sensor.
* :mod:`fibersail_edge.edge` — **Part 2**: the streaming edge processor (bounded
  ring buffer, rolling features, anomaly detection, honest evaluation, sink seam).

The whole public API is re-exported here, so ``from fibersail_edge import X`` works
for any part; the subpackages (``fibersail_edge.edge``, ``fibersail_edge.sensor``)
are available too for callers that prefer to be explicit about the layer.
"""

from __future__ import annotations

from .edge import (
    BaselineZScoreDetector,
    CallbackSink,
    Detector,
    DetectorConfig,
    EdgeProcessor,
    EvalResult,
    EvaluationReport,
    FeatureExtractor,
    FeatureFrame,
    FrameSink,
    ListSink,
    ProcessorConfig,
    RingBuffer,
    evaluate,
    frame_confusion,
    window_faulty_fraction,
)
from .sensor import DampedOscillatorSensor, FaultConfig, SensorConfig
from .sources import CsvReplaySource, Sample, SampleSource

__all__ = [
    # Shared streaming contract
    "Sample",
    "SampleSource",
    "CsvReplaySource",
    # Part 1 — sensor
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

__version__ = "0.3.0"
