"""Tests for the synthetic vibration sensor (Part 1).

These focus on the properties the rest of the pipeline relies on:
  * the stream is lazy and runs in bounded memory,
  * output is reproducible from a seed,
  * the injected fault is actually detectable (so Part 2 has a real target),
  * the physics is sane (spectral peak sits at the natural frequency),
  * timestamps/rate are correct and the signal is numerically stable.
"""

from __future__ import annotations

import inspect
import itertools
import tracemalloc

import numpy as np
import pytest

from fibersail_edge import (
    DampedOscillatorSensor,
    FaultConfig,
    Sample,
    SampleSource,
    SensorConfig,
)


def _collect(sensor: DampedOscillatorSensor) -> tuple[np.ndarray, np.ndarray]:
    """Drain a finite stream into (t, value) arrays."""
    data = np.array([(s.t, s.value) for s in sensor.stream()], dtype=float)
    return data[:, 0], data[:, 1]


def _dominant_freq(values: np.ndarray, fs: float) -> float:
    v = values - values.mean()
    spectrum = np.abs(np.fft.rfft(v * np.hanning(len(v))))
    freqs = np.fft.rfftfreq(len(v), 1.0 / fs)
    return float(freqs[np.argmax(spectrum)])


def test_implements_sample_source_protocol() -> None:
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=0.1))
    assert isinstance(sensor, SampleSource)


def test_stream_is_lazy() -> None:
    """stream() must be a generator and yield without precomputing the array."""
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=None))  # infinite
    stream = sensor.stream()
    assert inspect.isgenerator(stream)
    first = next(stream)
    assert isinstance(first, Sample)
    assert first.t == 0.0


def test_memory_is_bounded() -> None:
    """Peak allocation while consuming must not grow with the number of samples.

    We consume the stream one sample at a time (never retaining it), so a
    10x-longer run should use essentially the same peak memory.
    """
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=None))
    stream = sensor.stream()

    def peak_for(n: int) -> int:
        tracemalloc.start()
        acc = 0.0
        for s in itertools.islice(stream, n):
            acc += s.value  # touch the value; never store the samples
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert acc == acc  # keep `acc` alive / not NaN
        return peak

    peak_small = peak_for(2_000)
    peak_large = peak_for(20_000)
    # Allow generous slack for allocator noise; the point is it does NOT scale
    # with sample count (it would be ~10x here if the series were materialized).
    assert peak_large < peak_small + 200_000


def test_reproducible_with_seed() -> None:
    a = _collect(DampedOscillatorSensor(SensorConfig(duration_s=0.5, seed=7)))[1]
    b = _collect(DampedOscillatorSensor(SensorConfig(duration_s=0.5, seed=7)))[1]
    c = _collect(DampedOscillatorSensor(SensorConfig(duration_s=0.5, seed=8)))[1]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_timestamps_and_rate() -> None:
    fs = 1000.0
    t, _ = _collect(DampedOscillatorSensor(SensorConfig(duration_s=1.0, sample_rate_hz=fs)))
    assert len(t) == 1000
    dts = np.diff(t)
    assert np.allclose(dts, 1.0 / fs)
    assert np.all(dts > 0)  # strictly increasing


def test_signal_is_finite() -> None:
    for channel in ("displacement", "velocity", "acceleration"):
        _, v = _collect(
            DampedOscillatorSensor(SensorConfig(duration_s=1.0, channel=channel))
        )
        assert np.all(np.isfinite(v)), f"non-finite values on channel {channel}"


def test_all_channels_produce_signal() -> None:
    for channel in ("displacement", "velocity", "acceleration"):
        _, v = _collect(
            DampedOscillatorSensor(SensorConfig(duration_s=1.0, channel=channel))
        )
        assert np.std(v) > 0, f"channel {channel} produced a constant signal"


def test_dc_offset_only_on_displacement() -> None:
    offset = 1500.0
    _, disp = _collect(
        DampedOscillatorSensor(
            SensorConfig(duration_s=1.0, channel="displacement", dc_offset=offset)
        )
    )
    _, accel = _collect(
        DampedOscillatorSensor(
            SensorConfig(duration_s=1.0, channel="acceleration", dc_offset=offset)
        )
    )
    assert abs(disp.mean() - offset) < 5.0        # displacement sits near the offset
    assert abs(accel.mean()) < abs(offset) / 10   # acceleration is not shifted


def test_dominant_frequency_matches_natural_frequency() -> None:
    """A healthy run's spectral peak should sit near the configured f_n."""
    fs = 1000.0
    f_n = 50.0
    sensor = DampedOscillatorSensor(
        SensorConfig(duration_s=8.0, sample_rate_hz=fs, natural_freq_hz=f_n)
    )
    _, v = _collect(sensor)
    peak = _dominant_freq(v, fs)
    assert abs(peak - f_n) < 3.0  # within a few Hz of the natural frequency


def test_fault_window_exposed_and_kept_out_of_stream() -> None:
    fault = FaultConfig(start_s=4.0, duration_s=2.0, omega_n_factor=0.6)
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=1.0, fault=fault))
    assert sensor.fault_window == (4.0, 6.0)
    assert sensor.is_faulty(5.0) and not sensor.is_faulty(3.9)
    # Samples carry only (t, value) — no label leaks into the stream.
    assert set(Sample._fields) == {"t", "value"}

    healthy = DampedOscillatorSensor(SensorConfig(duration_s=1.0))
    assert healthy.fault_window is None


def test_fault_is_detectable() -> None:
    """The injected fault must shift the spectrum toward the faulted frequency.

    This validates that the ground-truth window corresponds to a *real*,
    findable change — the precondition for Part 2's detector evaluation.
    """
    fs = 1000.0
    f_n = 50.0
    factor = 0.6  # bearing-wear-like drop in natural frequency -> ~30 Hz
    sensor = DampedOscillatorSensor(
        SensorConfig(
            duration_s=10.0,
            sample_rate_hz=fs,
            natural_freq_hz=f_n,
            fault=FaultConfig(start_s=4.0, duration_s=2.0, omega_n_factor=factor),
        )
    )
    t, v = _collect(sensor)

    healthy_seg = v[(t >= 1.0) & (t < 3.0)]
    fault_seg = v[(t >= 4.2) & (t < 5.8)]  # trimmed to avoid edge transients

    f_healthy = _dominant_freq(healthy_seg, fs)
    f_fault = _dominant_freq(fault_seg, fs)

    # During the fault the peak should be much closer to the faulted frequency
    # than to the baseline frequency.
    assert abs(f_fault - f_n * factor) < abs(f_fault - f_n)
    assert abs(f_healthy - f_n) < abs(f_healthy - f_n * factor)
    # And the shift is substantial, not marginal.
    assert (f_healthy - f_fault) > 10.0


def test_natural_freq_above_nyquist_rejected() -> None:
    with pytest.raises(ValueError):
        SensorConfig(sample_rate_hz=100.0, natural_freq_hz=60.0)


def test_invalid_channel_rejected() -> None:
    with pytest.raises(ValueError):
        SensorConfig(channel="pressure")
