"""Tests for the streaming edge processor (Part 2).

Covers the windowing contract (nothing before the window fills, ~10 Hz hop
cadence, decimated sample carried), laziness, bounded memory over a long run,
source-agnosticism (works on the CSV replay too), and config validation.
"""

from __future__ import annotations

import inspect
import itertools
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from fibersail_edge import (
    CsvReplaySource,
    DampedOscillatorSensor,
    Detector,
    EdgeProcessor,
    FeatureFrame,
    ProcessorConfig,
    Sample,
    SensorConfig,
)


class _CountingDetector:
    """A trivial injected detector: flags every frame and counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def update(self, rms: float, dominant_freq_hz: float) -> tuple[bool, float]:
        self.calls += 1
        return True, 1.0

    def reset(self) -> None:
        self.calls = 0

SAMPLE_CSV = (
    Path(__file__).resolve().parent.parent / "take_home_exercise" / "sample_dataset_small.csv"
)


def _frames(duration_s: float = 5.0, **cfg_overrides: object) -> list[FeatureFrame]:
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=duration_s, seed=42))
    proc = EdgeProcessor(ProcessorConfig(**cfg_overrides))  # type: ignore[arg-type]
    return list(proc.process_stream(sensor))


def test_no_frames_before_window_full() -> None:
    fs, window_s = 1000.0, 1.5
    frames = _frames(duration_s=5.0, sample_rate_hz=fs, window_s=window_s, hop_s=0.1)
    assert frames, "expected some frames"
    # The window fills at sample index window_samples-1 → t = window_s - 1/fs.
    earliest = window_s - 1.0 / fs - 1e-9
    assert all(f.t >= earliest for f in frames)
    assert frames[0].t == pytest.approx(window_s - 1.0 / fs, abs=1.0 / fs)


def test_emits_at_hop_cadence() -> None:
    fs, hop_s = 1000.0, 0.1
    frames = _frames(duration_s=5.0, sample_rate_hz=fs, window_s=1.5, hop_s=hop_s)
    times = np.array([f.t for f in frames])
    gaps = np.diff(times)
    assert np.allclose(gaps, hop_s, atol=1.0 / fs)  # ~10 Hz


def test_frame_carries_decimated_sample_at_feature_rate() -> None:
    proc = EdgeProcessor(ProcessorConfig(sample_rate_hz=1000.0, window_s=1.5, hop_s=0.1))
    assert proc.feature_rate_hz == pytest.approx(10.0)
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=5.0, seed=42))
    frames = list(proc.process_stream(sensor))
    # ~10 Hz over ~3.5 s of post-warmup stream.
    assert 30 <= len(frames) <= 40
    assert all(np.isfinite(f.raw_value) for f in frames)


def test_process_stream_is_lazy() -> None:
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=None))  # infinite
    proc = EdgeProcessor()
    stream = proc.process_stream(sensor)
    assert inspect.isgenerator(stream)
    first = next(stream)
    assert isinstance(first, FeatureFrame)


def test_process_returns_none_during_warmup_and_off_hop() -> None:
    proc = EdgeProcessor(ProcessorConfig(sample_rate_hz=1000.0, window_s=0.1, hop_s=0.05))
    # window_samples = 100; nothing emitted for the first 99 samples.
    outs = [proc.process(Sample(t=i / 1000.0, value=float(i))) for i in range(100)]
    assert all(o is None for o in outs[:99])
    assert isinstance(outs[99], FeatureFrame)  # emits the instant the window fills


def test_bounded_memory_over_long_run() -> None:
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=None, seed=7))
    proc = EdgeProcessor()
    stream = sensor.stream()

    def peak_for(n: int) -> int:
        tracemalloc.start()
        for s in itertools.islice(stream, n):
            proc.process(s)  # discard frames; only fixed state is retained
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    peak_small = peak_for(5_000)
    peak_large = peak_for(50_000)
    assert peak_large < peak_small + 200_000


def test_reset_allows_reuse() -> None:
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=3.0, seed=42))
    proc = EdgeProcessor()
    first = [f.rms for f in proc.process_stream(sensor)]
    proc.reset()
    second = [f.rms for f in proc.process_stream(DampedOscillatorSensor(SensorConfig(duration_s=3.0, seed=42)))]
    assert first == second


def test_works_with_csv_source() -> None:
    """Source-agnostic: the CSV replay (~91 Hz, no ground truth) still yields frames."""
    src = CsvReplaySource(str(SAMPLE_CSV))
    proc = EdgeProcessor.for_source(src)
    frames = list(proc.process_stream(src))
    assert frames
    assert all(isinstance(f, FeatureFrame) for f in frames)
    assert all(np.isfinite(f.dominant_freq_hz) for f in frames)


def test_causality_prefix_consistency() -> None:
    """Outputs up to time T are identical whether or not data after T ever arrives.

    This is the strongest no-look-ahead statement for the whole pipeline
    (window + features + detector state): every frame emitted while processing
    only a prefix of the stream must be bit-identical to the corresponding frame
    of the full run. Any dependence on future samples would break the equality.
    """
    cfg = SensorConfig(duration_s=10.0, seed=42)
    full = list(EdgeProcessor().process_stream(DampedOscillatorSensor(cfg)))

    proc = EdgeProcessor()
    prefix: list[FeatureFrame] = []
    for s in itertools.islice(DampedOscillatorSensor(cfg).stream(), 6000):
        frame = proc.process(s)
        if frame is not None:
            prefix.append(frame)

    assert prefix  # the prefix is long enough to emit frames
    assert prefix == full[: len(prefix)]


def test_injected_detector_is_used() -> None:
    """A custom Detector can be injected without touching the processor."""
    fake = _CountingDetector()
    assert isinstance(fake, Detector)  # structural conformance
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=4.0, seed=42))
    proc = EdgeProcessor(detector=fake)
    frames = list(proc.process_stream(sensor))
    assert proc.detector is fake
    assert frames and all(f.is_anomaly for f in frames)  # the fake flags everything
    assert fake.calls == len(frames)  # called once per emitted frame


def test_for_source_accepts_injected_detector() -> None:
    fake = _CountingDetector()
    src = CsvReplaySource(str(SAMPLE_CSV))
    proc = EdgeProcessor.for_source(src, detector=fake, window_s=1.0)
    assert proc.detector is fake
    assert proc.config.sample_rate_hz == src.sample_rate_hz  # rate still wired from source


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ProcessorConfig(hop_s=2.0, window_s=1.0)  # hop > window
    with pytest.raises(ValueError):
        ProcessorConfig(sample_rate_hz=0.0)
    with pytest.raises(ValueError):
        ProcessorConfig(sample_rate_hz=1000.0, window_s=0.001)  # window too short for FFT
