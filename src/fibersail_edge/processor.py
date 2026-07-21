"""The edge processing service — the streaming heart of Part 2.

:class:`EdgeProcessor` consumes a :class:`~fibersail_edge.sources.SampleSource`
one sample at a time and emits :class:`~fibersail_edge.features.FeatureFrame`\\ s
at a fixed *hop* cadence. It ties together the three primitives:

    sample --> RingBuffer (bounded window) --> FeatureExtractor --> Detector --> FeatureFrame

Design points (expanded in the README):

* **Bounded, causal window.** A :class:`~fibersail_edge.ring_buffer.RingBuffer`
  of ``window_samples`` holds the most recent ``window_s`` seconds — a *trailing*
  window ``[t - window_s, t]`` with no look-ahead. Nothing is emitted until the
  buffer first fills (a ~``window_s`` silent start, well before any injected
  fault), after which frames come every ``hop_s`` seconds.
* **Decoupled cadence.** The per-sample path is just a ring-buffer push (O(1));
  the expensive work (an FFT + a few reductions) runs only once per hop. With
  ``hop_s = 0.1`` that is 10 frames/s regardless of the 1 kHz input — and those
  10 Hz frames double as the decimated telemetry Part 3 uploads.
* **Non-blocking output.** :meth:`process_stream` yields lazily; :meth:`run`
  pushes to a :class:`~fibersail_edge.sink.FrameSink`. The processor imports
  nothing from the cloud layer — dependency flow is one-way, so it can never
  block on S3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from .detector import BaselineZScoreDetector, Detector, DetectorConfig
from .features import FeatureExtractor, FeatureFrame
from .ring_buffer import RingBuffer
from .sink import FrameSink
from .sources import Sample, SampleSource


@dataclass(frozen=True)
class ProcessorConfig:
    """Configuration for :class:`EdgeProcessor`.

    Attributes:
        sample_rate_hz: Input sampling rate ``fs``. Should match the source's rate
            (use :meth:`EdgeProcessor.for_source` to wire it automatically).
        window_s: Rolling window length in seconds (the brief suggests 1–2 s).
        hop_s: Seconds between emitted frames. ``0.1`` → ~10 Hz frames.
        detrend: Subtract the window mean before the FFT (recommended).
        interpolate_peak: Parabolic sub-bin refinement of the FFT peak.
        detector: Configuration for the built-in baseline z-score detector.
    """

    sample_rate_hz: float = 1000.0
    window_s: float = 1.5
    hop_s: float = 0.1
    detrend: bool = True
    interpolate_peak: bool = True
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.window_s <= 0:
            raise ValueError("window_s must be > 0")
        if not (0 < self.hop_s <= self.window_s):
            raise ValueError("hop_s must satisfy 0 < hop_s <= window_s")
        if self.window_samples < 8:
            raise ValueError(
                f"window is too short for a meaningful FFT: "
                f"window_s * sample_rate_hz = {self.window_samples} samples (need >= 8)"
            )

    @property
    def window_samples(self) -> int:
        """Window length in samples, ``round(window_s * fs)``."""
        return int(round(self.window_s * self.sample_rate_hz))

    @property
    def hop_samples(self) -> int:
        """Samples between emitted frames, ``round(hop_s * fs)`` (>= 1)."""
        return max(1, int(round(self.hop_s * self.sample_rate_hz)))

    @property
    def feature_rate_hz(self) -> float:
        """Effective frame emission rate, ``fs / hop_samples``."""
        return self.sample_rate_hz / self.hop_samples


class EdgeProcessor:
    """Streaming edge processor: samples in, feature frames out.

    Example:
        >>> from fibersail_edge import DampedOscillatorSensor, SensorConfig
        >>> sensor = DampedOscillatorSensor(SensorConfig(duration_s=3.0))
        >>> proc = EdgeProcessor.for_source(sensor)
        >>> frames = list(proc.process_stream(sensor))
        >>> # first frame lands as the trailing window fills (~window_s), never before
        >>> len(frames) > 0 and abs(frames[0].t - proc.config.window_s) < 0.01
        True
    """

    def __init__(
        self,
        config: Optional[ProcessorConfig] = None,
        *,
        detector: Optional[Detector] = None,
    ) -> None:
        """Build a processor.

        Args:
            config: Windowing / feature configuration.
            detector: Any object satisfying the :class:`~fibersail_edge.detector.Detector`
                protocol. Defaults to a :class:`~fibersail_edge.detector.BaselineZScoreDetector`
                built from ``config.detector``. Inject your own to swap the detection
                strategy (e.g. an EWMA/Kalman variant) without changing the processor;
                when a detector is injected, ``config.detector`` is unused.
        """
        self.config = config or ProcessorConfig()
        self._ring = RingBuffer(self.config.window_samples)
        self._extractor = FeatureExtractor(
            self.config.window_samples,
            self.config.sample_rate_hz,
            detrend=self.config.detrend,
            interpolate_peak=self.config.interpolate_peak,
        )
        self._detector: Detector = (
            detector if detector is not None else BaselineZScoreDetector(self.config.detector)
        )
        self._first_full = True
        self._since_emit = 0

    @classmethod
    def for_source(
        cls,
        source: SampleSource,
        *,
        detector: Optional[Detector] = None,
        **overrides: object,
    ) -> "EdgeProcessor":
        """Build a processor whose ``sample_rate_hz`` matches ``source``.

        Any :class:`ProcessorConfig` field can be overridden by keyword, e.g.
        ``EdgeProcessor.for_source(src, window_s=2.0)``, and a custom ``detector``
        can be injected. This makes the processor work unchanged on the ~91 Hz CSV
        replay source as well as the 1 kHz sensor.
        """
        config = ProcessorConfig(sample_rate_hz=source.sample_rate_hz, **overrides)  # type: ignore[arg-type]
        return cls(config, detector=detector)

    # -- Convenience passthroughs ---------------------------------------------

    @property
    def detector(self) -> Detector:
        """The detector instance (exposed for evaluation/introspection)."""
        return self._detector

    @property
    def feature_rate_hz(self) -> float:
        return self.config.feature_rate_hz

    @property
    def window_s(self) -> float:
        return self.config.window_s

    # -- Streaming interface ---------------------------------------------------

    def process(self, sample: Sample) -> Optional[FeatureFrame]:
        """Ingest one sample; return a :class:`FeatureFrame` on a hop, else ``None``.

        Returns ``None`` while the window is still filling and on non-hop samples.
        """
        self._ring.push(sample.value)
        if not self._ring.is_full:
            return None

        self._since_emit += 1
        if not (self._first_full or self._since_emit >= self.config.hop_samples):
            return None

        # Emit: take a window snapshot, extract features, score, stamp the verdict.
        self._first_full = False
        self._since_emit = 0
        window = self._ring.snapshot()
        frame = self._extractor.extract(window, sample.t, sample.value)
        is_anomaly, score = self._detector.update(frame.rms, frame.dominant_freq_hz)
        return frame._replace(is_anomaly=is_anomaly, score=score)

    def process_stream(self, source: SampleSource) -> Iterator[FeatureFrame]:
        """Consume ``source.stream()`` lazily and yield one frame per hop."""
        for sample in source.stream():
            frame = self.process(sample)
            if frame is not None:
                yield frame

    def run(self, source: SampleSource, sink: FrameSink) -> None:
        """Drive the whole stream, pushing every frame to ``sink``.

        The driver never inspects the sink's progress, so a well-behaved
        (non-blocking) sink keeps the edge loop running at full speed.
        """
        for frame in self.process_stream(source):
            sink.emit(frame)

    def reset(self) -> None:
        """Clear all state so the processor can be reused on a fresh stream."""
        self._ring.clear()
        self._detector.reset()
        self._first_full = True
        self._since_emit = 0


__all__ = ["ProcessorConfig", "EdgeProcessor"]
