"""End-to-end Part 2 demo: synthesize a faulted stream, process it, score it.

    uv run python -m fibersail_edge
    uv run python -m fibersail_edge --k 3.5 --window-s 2.0 --duration-s 20

Builds a :class:`~fibersail_edge.sensor.DampedOscillatorSensor` with an injected
fault, runs it through the :class:`~fibersail_edge.edge.processor.EdgeProcessor`, and
prints the honest precision/recall report from
:func:`~fibersail_edge.edge.evaluation.evaluate`.
"""

from __future__ import annotations

import argparse

from .edge import DetectorConfig, EdgeProcessor, ProcessorConfig, evaluate
from .sensor import DampedOscillatorSensor, FaultConfig, SensorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Fibersail edge processing demo + evaluation.")
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--sample-rate-hz", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    # Fault (ground truth).
    parser.add_argument("--fault-start-s", type=float, default=7.0)
    parser.add_argument("--fault-duration-s", type=float, default=3.0)
    parser.add_argument("--omega-factor", type=float, default=0.7, help="natural-freq multiplier during fault")
    parser.add_argument("--zeta-factor", type=float, default=0.4, help="damping multiplier during fault")
    # Processor / detector.
    parser.add_argument("--window-s", type=float, default=1.5)
    parser.add_argument("--hop-s", type=float, default=0.1)
    parser.add_argument("--k", type=float, default=4.0, help="z-score threshold")
    parser.add_argument("--calibration-frames", type=int, default=30)
    args = parser.parse_args()

    sensor = DampedOscillatorSensor(
        SensorConfig(
            sample_rate_hz=args.sample_rate_hz,
            duration_s=args.duration_s,
            seed=args.seed,
            fault=FaultConfig(
                start_s=args.fault_start_s,
                duration_s=args.fault_duration_s,
                omega_n_factor=args.omega_factor,
                zeta_factor=args.zeta_factor,
            ),
        )
    )
    processor = EdgeProcessor(
        ProcessorConfig(
            sample_rate_hz=args.sample_rate_hz,
            window_s=args.window_s,
            hop_s=args.hop_s,
            detector=DetectorConfig(k=args.k, calibration_frames=args.calibration_frames),
        )
    )

    report = evaluate(sensor, processor)
    print(report.format_summary())


if __name__ == "__main__":
    main()
