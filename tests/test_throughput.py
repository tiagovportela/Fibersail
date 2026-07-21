"""Throughput tests (Part 2) — proves the >= 1000 samples/sec requirement.

Run with ``-s`` to see the measured rates:

    uv run pytest tests/test_throughput.py -s
"""

from __future__ import annotations

from fibersail_edge import DampedOscillatorSensor, ProcessorConfig, SensorConfig
from fibersail_edge.edge.benchmark import (
    bench_end_to_end,
    bench_processor_isolated,
    make_samples,
)

TARGET_SAMPLES_PER_S = 1000.0


def test_throughput_at_least_1000_per_sec() -> None:
    """Isolated processing must sustain well over the 1 kHz single-core bar."""
    cfg = ProcessorConfig(sample_rate_hz=1000.0, window_s=1.5, hop_s=0.1)
    samples = make_samples(60_000, SensorConfig(sample_rate_hz=1000.0, seed=42))
    result = bench_processor_isolated(samples, cfg)
    print(
        f"\nisolated: {result['samples_per_s']:,.0f} samp/s "
        f"({result['x_realtime']:.0f}x real-time, {result['per_sample_us']:.2f} us/sample)"
    )
    assert result["samples_per_s"] >= TARGET_SAMPLES_PER_S


def test_end_to_end_faster_than_realtime() -> None:
    """Sensor synthesis + processing together still beats real time comfortably."""
    fs = 1000.0
    cfg = ProcessorConfig(sample_rate_hz=fs, window_s=1.5, hop_s=0.1)
    source = DampedOscillatorSensor(SensorConfig(sample_rate_hz=fs, seed=42))
    result = bench_end_to_end(source, 60_000, cfg)
    print(
        f"\nend-to-end: {result['samples_per_s']:,.0f} samp/s "
        f"({result['x_realtime']:.0f}x real-time)"
    )
    assert result["samples_per_s"] >= TARGET_SAMPLES_PER_S
    assert result["x_realtime"] > 1.0
