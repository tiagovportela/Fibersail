"""fibersail_edge — edge sensor processing & cloud sync.

Part 1 (this module set) provides the synthetic vibration sensor and the shared
streaming data-source interface that Parts 2 (edge processing) and 3 (cloud
sync) build on.
"""

from __future__ import annotations

from .sensor import DampedOscillatorSensor, FaultConfig, SensorConfig
from .sources import CsvReplaySource, Sample, SampleSource

__all__ = [
    "Sample",
    "SampleSource",
    "CsvReplaySource",
    "SensorConfig",
    "FaultConfig",
    "DampedOscillatorSensor",
]

__version__ = "0.1.0"
