"""Tests for the streaming data-source interface and the CSV replay source."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from fibersail_edge import CsvReplaySource, Sample, SampleSource

SAMPLE_CSV = (
    Path(__file__).resolve().parent.parent
    / "take_home_exercise"
    / "sample_dataset_small.csv"
)


def test_sample_csv_present() -> None:
    assert SAMPLE_CSV.exists(), f"expected sample dataset at {SAMPLE_CSV}"


def test_csv_replay_implements_protocol() -> None:
    source = CsvReplaySource(str(SAMPLE_CSV))
    assert isinstance(source, SampleSource)
    assert source.fault_window is None  # no known ground truth for recorded data


def test_csv_replay_streams_lazily() -> None:
    source = CsvReplaySource(str(SAMPLE_CSV))
    stream = source.stream()
    assert inspect.isgenerator(stream)
    first = next(stream)
    assert isinstance(first, Sample)
    assert first.t == 0.0  # t is relative to the first populated row


def test_csv_replay_skips_blank_rows_and_is_monotonic() -> None:
    source = CsvReplaySource(str(SAMPLE_CSV))
    samples = list(source.stream())
    # The dataset populates strain on ~1/4 of its rows.
    assert len(samples) == 1818
    times = [s.t for s in samples]
    assert all(b > a for a, b in zip(times, times[1:])), "timestamps not increasing"
    # Strain values in the provided file sit around 1500.
    values = [s.value for s in samples]
    assert 1400.0 < min(values) < max(values) < 1600.0


def test_csv_replay_infers_sample_rate() -> None:
    source = CsvReplaySource(str(SAMPLE_CSV))
    # Populated rows are ~11 ms apart -> ~91 Hz effective rate.
    assert 80.0 < source.sample_rate_hz < 100.0


def test_csv_stream_full_carries_temperature() -> None:
    source = CsvReplaySource(str(SAMPLE_CSV))
    t, strain, temperature = next(source.stream_full())
    assert t == 0.0
    assert strain > 0
    assert temperature is not None and 30.0 < temperature < 50.0


def test_csv_missing_file_raises() -> None:
    source = CsvReplaySource("does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        next(source.stream())
