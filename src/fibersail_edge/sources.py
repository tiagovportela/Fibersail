"""Streaming sample-source interface and a CSV replay source.

The whole pipeline (Part 1 sensor → Part 2 edge processor → Part 3 cloud sync)
is decoupled from *where* samples come from by a single small contract:
:class:`SampleSource`. Both the synthetic :class:`~fibersail_edge.sensor.DampedOscillatorSensor`
and the :class:`CsvReplaySource` below implement it, so downstream code can be
pointed at either without change.

Design notes
------------
* ``Sample`` is a ``NamedTuple`` on purpose: it is immutable, tuple-cheap in
  both memory and construction cost, and unpacks cleanly (``t, value = sample``).
  Part 2 will pull these at >=1 kHz, so per-sample overhead matters.
* Sources are *lazy*: ``stream()`` returns an iterator that produces one sample
  at a time. Nothing ever materializes the full series in memory — the edge
  service "should not assume it has the whole array up front".
* Ground truth (the fault window) is exposed as source *metadata*, never mixed
  into the sample stream, so a detector cannot accidentally peek at labels.
"""

from __future__ import annotations

import csv
from datetime import datetime
from statistics import median
from typing import Iterator, NamedTuple, Optional, Protocol, Tuple, runtime_checkable


class Sample(NamedTuple):
    """A single timestamped sensor reading.

    Attributes:
        t: Seconds since the start of the stream (``t = 0`` is the first sample).
        value: The emitted channel value (acceleration by default for the
            synthetic sensor; strain for the CSV replay source).
    """

    t: float
    value: float


@runtime_checkable
class SampleSource(Protocol):
    """Contract every data source implements.

    Keeping this deliberately tiny is what lets Parts 2 and 3 stay decoupled
    from the origin of the data (synthetic physics vs. recorded CSV vs. a real
    device later).
    """

    @property
    def sample_rate_hz(self) -> float:
        """Nominal sampling rate in Hz. Downstream windowing/FFT code needs this.

        Declared as a read-only property (both implementations expose it that
        way); a plain attribute here would demand a *settable* member and reject
        read-only conformers under type checking.
        """
        ...

    def stream(self) -> Iterator[Sample]:
        """Yield samples lazily, one at a time, in time order."""
        ...

    @property
    def fault_window(self) -> Optional[Tuple[float, float]]:
        """Ground-truth anomaly window ``(start_s, end_s)`` or ``None``.

        ``None`` means "no known ground truth" (e.g. recorded data). This is
        metadata for *offline* evaluation only — it is never part of ``stream()``.
        """
        ...


class CsvReplaySource:
    """Replays the provided ``sample_dataset_small.csv`` through :class:`SampleSource`.

    The sample dataset is ``;``-delimited with columns
    ``date_time;strain;temperature``. It sits on a fixed ~364 Hz time grid, but
    ``strain``/``temperature`` are only populated on roughly every 4th row
    (an effective ~91 Hz), leaving the rest blank.

    Simplifying assumptions (documented in the README):
      * We treat a row as a real sample only when ``strain`` is present, and
        skip the blank filler rows rather than forward-filling them — forward
        filling would fabricate spectral content that is not in the signal.
      * ``value`` is the ``strain`` reading; ``temperature`` is available via
        :meth:`stream_full` for callers that want the auxiliary channel.
      * ``sample_rate_hz`` is inferred from the median spacing of populated rows.
      * The file has no known injected fault, so ``fault_window`` is ``None``.

    The CSV is read with the stdlib ``csv`` reader line-by-line, so the whole
    file is never loaded into memory at once — consistent with the streaming,
    bounded-footprint constraints of the edge device.
    """

    def __init__(
        self,
        path: str,
        *,
        delimiter: str = ";",
        value_column: str = "strain",
        time_column: str = "date_time",
        temperature_column: str = "temperature",
    ) -> None:
        self.path = path
        self._delimiter = delimiter
        self._value_column = value_column
        self._time_column = time_column
        self._temperature_column = temperature_column
        # Inferred lazily on first access so construction stays cheap and does
        # not require a full pass if the caller only wants to stream.
        self._sample_rate_hz: Optional[float] = None

    # -- SampleSource interface ------------------------------------------------

    @property
    def sample_rate_hz(self) -> float:
        if self._sample_rate_hz is None:
            self._sample_rate_hz = self._infer_sample_rate()
        return self._sample_rate_hz

    @property
    def fault_window(self) -> Optional[Tuple[float, float]]:
        return None  # No known ground truth for recorded data.

    def stream(self) -> Iterator[Sample]:
        """Yield ``Sample(t, strain)`` for each populated row, ``t`` from file start."""
        for t, value, _temperature in self._iter_rows():
            yield Sample(t=t, value=value)

    # -- Extras ----------------------------------------------------------------

    def stream_full(self) -> Iterator[Tuple[float, float, Optional[float]]]:
        """Yield ``(t, strain, temperature)`` — temperature may be ``None``."""
        yield from self._iter_rows()

    # -- Internals -------------------------------------------------------------

    def _iter_rows(self) -> Iterator[Tuple[float, float, Optional[float]]]:
        """Stream populated rows as ``(t_seconds, strain, temperature)``.

        Blank ``strain`` rows are skipped. ``t`` is seconds relative to the
        first populated row's timestamp.
        """
        t0: Optional[datetime] = None
        with open(self.path, newline="") as fh:
            reader = csv.DictReader(fh, delimiter=self._delimiter)
            for row in reader:
                raw_value = (row.get(self._value_column) or "").strip()
                if not raw_value:
                    continue  # skip blank filler rows
                ts = datetime.fromisoformat(row[self._time_column].strip())
                if t0 is None:
                    t0 = ts
                t = (ts - t0).total_seconds()

                raw_temp = (row.get(self._temperature_column) or "").strip()
                temperature = float(raw_temp) if raw_temp else None
                yield t, float(raw_value), temperature

    def _infer_sample_rate(self) -> float:
        """Median 1/Δt over populated rows (robust to the occasional gap)."""
        times = [t for t, _v, _temp in self._iter_rows()]
        if len(times) < 2:
            raise ValueError(
                f"Need at least two populated rows to infer a sample rate "
                f"from {self.path!r}"
            )
        deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
        if not deltas:
            raise ValueError(f"Timestamps in {self.path!r} are not increasing")
        return 1.0 / median(deltas)
