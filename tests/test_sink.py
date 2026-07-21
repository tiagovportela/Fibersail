"""Tests for the Part 2/3 decoupling seam (Part 2)."""

from __future__ import annotations

from fibersail_edge import (
    CallbackSink,
    DampedOscillatorSensor,
    EdgeProcessor,
    FeatureFrame,
    FrameSink,
    ListSink,
    SensorConfig,
)


def _frame(t: float = 0.0) -> FeatureFrame:
    return FeatureFrame(t=t, raw_value=1.0, rms=1.0, mean=0.0, std=1.0, dominant_freq_hz=50.0)


def test_list_and_callback_sinks_are_frame_sinks() -> None:
    assert isinstance(ListSink(), FrameSink)
    assert isinstance(CallbackSink(lambda f: None), FrameSink)


def test_callback_sink_receives_every_frame() -> None:
    received: list[FeatureFrame] = []
    sink = CallbackSink(received.append)
    for i in range(5):
        sink.emit(_frame(float(i)))
    assert [f.t for f in received] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_list_sink_is_bounded() -> None:
    sink = ListSink(maxlen=3)
    for i in range(10):
        sink.emit(_frame(float(i)))
    assert len(sink.frames) == 3
    assert [f.t for f in sink.frames] == [7.0, 8.0, 9.0]  # newest retained


def test_run_drives_every_frame_into_sink() -> None:
    sensor = DampedOscillatorSensor(SensorConfig(duration_s=4.0, seed=42))
    proc = EdgeProcessor()
    sink = ListSink()
    proc.run(sensor, sink)
    # The sink saw exactly the frames the stream would have yielded.
    proc.reset()
    expected = list(proc.process_stream(DampedOscillatorSensor(SensorConfig(duration_s=4.0, seed=42))))
    assert len(sink.frames) == len(expected) > 0
    assert sink.frames[0] == expected[0]
