"""Synthetic vibration sensor — a noise-driven damped harmonic oscillator.

Physics
-------
The sensor models a single-degree-of-freedom vibrating structure (a mass on a
damped spring — a decent first approximation of an accelerometer mounted on a
machine)::

    x''(t) + 2*zeta*omega_n*x'(t) + omega_n**2 * x(t) = F(t)

where ``omega_n`` is the undamped natural frequency (rad/s), ``zeta`` the
damping ratio, and ``F(t)`` the driving force. In normal operation ``F(t)`` is
Gaussian noise (broadband ambient excitation). A *fault* — bearing wear or a
loosening mount — is modeled as a temporary shift in ``omega_n`` and/or ``zeta``.

Numerical method
----------------
We integrate the state ``[x, v]`` with a hand-rolled **fixed-step RK4** rather
than an adaptive ``scipy.integrate`` solver, because the edge use-case wants:

* **Bounded O(1) memory** — only the current 2-element state is retained; we
  never build an array of the whole series.
* **Deterministic, constant per-step cost** — real-time streaming needs
  predictable timing; adaptive step-size control does not give that.

The driving force is held **constant across each RK4 step** (one Gaussian draw
per step). This matters: a literally white-noise-driven ODE is a stochastic
differential equation for which RK4 is not formally convergent. Holding the
force constant over a step makes it a *band-limited* excitation — which is both
physically reasonable (real forcing is band-limited) and numerically well-posed.

To keep the physical response invariant to the choice of sample rate, the
per-step force standard deviation is scaled by ``sqrt(fs)``
(Euler–Maruyama-consistent): doubling ``fs`` halves the step but doubles the
per-step variance, leaving the low-frequency excitation PSD — and hence the
oscillator's response around ``omega_n`` — unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

from .sources import Sample

#: Emittable channels. Acceleration is the default because that is what a real
#: vibration/accelerometer sensor reports, and a fault shows up in it as both an
#: amplitude and a frequency change.
_CHANNELS = ("displacement", "velocity", "acceleration")


@dataclass(frozen=True)
class FaultConfig:
    """A temporary shift of the oscillator parameters — the injected anomaly.

    The fault multiplies the baseline parameters for a window of time, then the
    system returns to baseline. Physical interpretation:

    * ``omega_n_factor < 1`` — loss of effective stiffness (e.g. bearing wear),
      lowering the resonant frequency.
    * ``zeta_factor < 1`` — a loosening mount damps less, giving a sharper,
      higher-amplitude resonance. ``zeta_factor > 1`` models added friction.

    Attributes:
        start_s: Fault onset, in seconds from stream start.
        duration_s: How long the fault lasts, in seconds.
        omega_n_factor: Multiplier applied to ``omega_n`` during the fault.
        zeta_factor: Multiplier applied to ``zeta`` during the fault.
    """

    start_s: float
    duration_s: float
    omega_n_factor: float = 1.0
    zeta_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.start_s < 0:
            raise ValueError("fault start_s must be >= 0")
        if self.duration_s <= 0:
            raise ValueError("fault duration_s must be > 0")
        if self.omega_n_factor <= 0 or self.zeta_factor <= 0:
            raise ValueError("fault factors must be > 0")

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    def contains(self, t: float) -> bool:
        """True if time ``t`` (seconds) falls inside the fault window."""
        return self.start_s <= t < self.end_s


@dataclass(frozen=True)
class SensorConfig:
    """Configuration for :class:`DampedOscillatorSensor`.

    Defaults are chosen so the resonance (``natural_freq_hz`` = 50 Hz) sits well
    below the 500 Hz Nyquist of the default 1 kHz rate, and light damping
    (``zeta`` = 0.05) yields a clear spectral peak that a detector can track.

    Attributes:
        sample_rate_hz: Output sampling rate ``fs``.
        natural_freq_hz: Baseline natural frequency ``f_n`` (``omega_n = 2*pi*f_n``).
        zeta: Baseline damping ratio.
        force_std: Std. dev. of the (sample-rate-normalized) Gaussian forcing.
            Controls overall vibration amplitude.
        channel: One of ``"displacement" | "velocity" | "acceleration"``.
        seed: RNG seed for reproducibility. ``None`` uses fresh entropy.
        duration_s: If set, the stream ends after this many seconds; ``None``
            streams indefinitely.
        warmup_s: Simulated time integrated *before* ``t = 0`` so the emitted
            signal is already stationary at the first sample (the transient has
            a time constant ``tau = 1/(zeta*omega_n)``).
        dc_offset: Constant added to the displacement channel — handy to mirror
            a strain-gauge baseline (the sample dataset sits near 1500).
        fault: Optional injected fault; ``None`` means healthy baseline only.
    """

    sample_rate_hz: float = 1000.0
    natural_freq_hz: float = 50.0
    zeta: float = 0.05
    force_std: float = 1.0
    channel: str = "acceleration"
    seed: Optional[int] = 42
    duration_s: Optional[float] = None
    warmup_s: float = 0.2
    dc_offset: float = 0.0
    fault: Optional[FaultConfig] = None

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.natural_freq_hz <= 0:
            raise ValueError("natural_freq_hz must be > 0")
        if self.natural_freq_hz >= self.sample_rate_hz / 2:
            raise ValueError(
                f"natural_freq_hz ({self.natural_freq_hz}) must be below the "
                f"Nyquist frequency ({self.sample_rate_hz / 2})"
            )
        if self.zeta <= 0:
            raise ValueError("zeta must be > 0")
        if self.channel not in _CHANNELS:
            raise ValueError(f"channel must be one of {_CHANNELS}, got {self.channel!r}")
        if self.warmup_s < 0:
            raise ValueError("warmup_s must be >= 0")
        if self.duration_s is not None and self.duration_s <= 0:
            raise ValueError("duration_s must be > 0 when set")

    @property
    def omega_n(self) -> float:
        """Baseline natural frequency in rad/s."""
        return 2.0 * math.pi * self.natural_freq_hz


class DampedOscillatorSensor:
    """A synthetic vibration sensor that streams samples one at a time.

    Implements the :class:`~fibersail_edge.sources.SampleSource` protocol, so it
    is interchangeable with :class:`~fibersail_edge.sources.CsvReplaySource`.

    Example:
        >>> import itertools
        >>> cfg = SensorConfig(duration_s=1.0, fault=FaultConfig(0.4, 0.2, 0.8))
        >>> sensor = DampedOscillatorSensor(cfg)
        >>> first_three = list(itertools.islice(sensor.stream(), 3))
        >>> sensor.fault_window
        (0.4, 0.6000000000000001)
    """

    def __init__(self, config: Optional[SensorConfig] = None) -> None:
        self.config = config or SensorConfig()
        self._dt = 1.0 / self.config.sample_rate_hz
        # Per-step force std: sqrt(fs) scaling makes the excitation PSD — and so
        # the physical response — independent of the sample rate.
        self._force_step_std = self.config.force_std * math.sqrt(self.config.sample_rate_hz)

    # -- SampleSource interface ------------------------------------------------

    @property
    def sample_rate_hz(self) -> float:
        return self.config.sample_rate_hz

    @property
    def fault_window(self) -> Optional[Tuple[float, float]]:
        """Ground-truth anomaly window ``(start_s, end_s)`` or ``None``.

        Exposed as metadata for offline evaluation only — it is never part of
        the emitted stream, so a detector cannot peek at labels.
        """
        if self.config.fault is None:
            return None
        return (self.config.fault.start_s, self.config.fault.end_s)

    def is_faulty(self, t: float) -> bool:
        """Whether time ``t`` (seconds) lies within the injected fault window."""
        return self.config.fault is not None and self.config.fault.contains(t)

    def stream(self) -> Iterator[Sample]:
        """Yield :class:`Sample` values lazily until ``duration_s`` (or forever).

        Memory is O(1): only the 2-element state ``[x, v]`` and a handful of
        scalars are retained regardless of how long the stream runs.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        dt = self._dt

        # State: x = displacement, v = velocity. Start from rest and integrate
        # through the warmup so the emitted signal is already stationary at t=0.
        x = 0.0
        v = 0.0
        n_warmup = int(round(cfg.warmup_s * cfg.sample_rate_hz))
        for _ in range(n_warmup):
            x, v, _a = self._rk4_step(x, v, self._draw_force(rng), cfg.omega_n, cfg.zeta)

        n_total = None if cfg.duration_s is None else int(round(cfg.duration_s * cfg.sample_rate_hz))
        i = 0
        while n_total is None or i < n_total:
            t = i * dt
            omega_n, zeta = self._params_at(t)
            force = self._draw_force(rng)
            x, v, a = self._rk4_step(x, v, force, omega_n, zeta)
            yield Sample(t=t, value=self._emit(x, v, a))
            i += 1

    # -- Internals -------------------------------------------------------------

    def _params_at(self, t: float) -> Tuple[float, float]:
        """Effective ``(omega_n, zeta)`` at time ``t``, applying any active fault."""
        cfg = self.config
        if cfg.fault is not None and cfg.fault.contains(t):
            return (cfg.omega_n * cfg.fault.omega_n_factor, cfg.zeta * cfg.fault.zeta_factor)
        return (cfg.omega_n, cfg.zeta)

    def _draw_force(self, rng: np.random.Generator) -> float:
        return float(rng.normal(0.0, self._force_step_std))

    @staticmethod
    def _deriv(x: float, v: float, force: float, omega_n: float, zeta: float) -> Tuple[float, float]:
        """Right-hand side of the state ODE: returns ``(x', v')``.

        ``x' = v`` and ``v' = F - 2*zeta*omega_n*v - omega_n**2 * x`` (the
        acceleration).
        """
        acc = force - 2.0 * zeta * omega_n * v - omega_n * omega_n * x
        return v, acc

    def _rk4_step(
        self, x: float, v: float, force: float, omega_n: float, zeta: float
    ) -> Tuple[float, float, float]:
        """Advance ``[x, v]`` by one ``dt`` with RK4; force constant over the step.

        Returns ``(x_next, v_next, acceleration_at_end)`` where the acceleration
        is evaluated at the new state (used for the acceleration channel).
        """
        dt = self._dt
        d = self._deriv

        k1x, k1v = d(x, v, force, omega_n, zeta)
        k2x, k2v = d(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, force, omega_n, zeta)
        k3x, k3v = d(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, force, omega_n, zeta)
        k4x, k4v = d(x + dt * k3x, v + dt * k3v, force, omega_n, zeta)

        x_next = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        # Acceleration consistent with the new state and the same forcing.
        _, acc_next = d(x_next, v_next, force, omega_n, zeta)
        return x_next, v_next, acc_next

    def _emit(self, x: float, v: float, a: float) -> float:
        """Map the integrator state to the configured output channel."""
        channel = self.config.channel
        if channel == "acceleration":
            return a
        if channel == "velocity":
            return v
        # displacement
        return x + self.config.dc_offset
