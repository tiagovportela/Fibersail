"""Part 1 — synthetic vibration sensor.

A noise-driven damped harmonic oscillator, integrated with fixed-step RK4 and
streamed one sample at a time, with an injectable fault that gives a known
ground-truth anomaly window. See :mod:`fibersail_edge.sensor.oscillator`.
"""

from __future__ import annotations

from .oscillator import DampedOscillatorSensor, FaultConfig, SensorConfig

__all__ = ["DampedOscillatorSensor", "FaultConfig", "SensorConfig"]
