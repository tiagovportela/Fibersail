"""Throughput benchmark for the edge processor.

Demonstrates the brief's requirement to *sustain at least 1000 samples/sec on a
single core without falling behind*, and shows there is comfortable headroom.

Run it::

    uv run python -m fibersail_edge.benchmark
    uv run python -m fibersail_edge.benchmark --samples 200000 --window-s 2.0

Two measurements are reported:

* **isolated** — samples are pre-generated once, then only the ``process()`` loop
  is timed. This is the pure edge-processing throughput (excludes the cost of
  synthesizing the signal, which a real device would not pay — it reads a sensor).
* **end-to-end** — the sensor's RK4 integration *and* processing are timed
  together, i.e. the whole synthetic pipeline's rate.

Because the per-sample path is just a ring-buffer push (the FFT runs only ~10x/s),
isolated throughput is typically 10^4–10^5 samples/s on one core — 10–100x over
the 1 kHz bar.
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Dict, List, Optional

from .processor import EdgeProcessor, ProcessorConfig
from ..sensor import DampedOscillatorSensor, FaultConfig, SensorConfig
from ..sources import Sample, SampleSource


def make_samples(n: int, config: Optional[SensorConfig] = None) -> List[Sample]:
    """Pre-generate ``n`` samples into a list (so benchmarking excludes synthesis)."""
    sensor = DampedOscillatorSensor(config or SensorConfig())
    return list(itertools.islice(sensor.stream(), n))


def bench_processor_isolated(samples: List[Sample], config: ProcessorConfig) -> Dict[str, float]:
    """Time only the ``process()`` loop over pre-materialized samples."""
    proc = EdgeProcessor(config)
    n = len(samples)
    frames = 0
    start = time.perf_counter()
    for s in samples:
        if proc.process(s) is not None:
            frames += 1
    elapsed = time.perf_counter() - start
    return _summarize("isolated", n, frames, elapsed, config.sample_rate_hz)


def bench_end_to_end(source: SampleSource, n: int, config: ProcessorConfig) -> Dict[str, float]:
    """Time sensor synthesis + processing together over ``n`` samples."""
    proc = EdgeProcessor(config)
    frames = 0
    seen = 0
    start = time.perf_counter()
    for s in itertools.islice(source.stream(), n):
        seen += 1
        if proc.process(s) is not None:
            frames += 1
    elapsed = time.perf_counter() - start
    return _summarize("end-to-end", seen, frames, elapsed, config.sample_rate_hz)


def _summarize(name: str, n: int, frames: int, elapsed: float, fs: float) -> Dict[str, float]:
    rate = n / elapsed if elapsed > 0 else float("inf")
    return {
        "name": name,
        "samples": float(n),
        "frames": float(frames),
        "elapsed_s": elapsed,
        "samples_per_s": rate,
        "x_realtime": rate / fs,
        "per_sample_us": (elapsed / n * 1e6) if n else 0.0,
    }


def _print_row(r: Dict[str, float]) -> None:
    print(
        f"  {r['name']:<11s} "
        f"{int(r['samples']):>9d} samples  "
        f"{r['elapsed_s']:>7.3f} s  "
        f"{r['samples_per_s']:>12,.0f} samp/s  "
        f"{r['x_realtime']:>7.1f}x RT  "
        f"{int(r['frames']):>6d} frames  "
        f"{r['per_sample_us']:>6.2f} us/sample"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge processor throughput benchmark.")
    parser.add_argument("--samples", type=int, default=120_000, help="samples per run")
    parser.add_argument("--sample-rate-hz", type=float, default=1000.0)
    parser.add_argument("--window-s", type=float, default=1.5)
    parser.add_argument("--hop-s", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sensor_cfg = SensorConfig(
        sample_rate_hz=args.sample_rate_hz,
        seed=args.seed,
        # A fault in the mix so the detector's scoring path is exercised too.
        fault=FaultConfig(start_s=args.samples / args.sample_rate_hz * 0.5,
                          duration_s=1.0, omega_n_factor=0.7, zeta_factor=0.4),
    )
    proc_cfg = ProcessorConfig(
        sample_rate_hz=args.sample_rate_hz,
        window_s=args.window_s,
        hop_s=args.hop_s,
    )

    print(f"Generating {args.samples:,} samples @ {args.sample_rate_hz:.0f} Hz ...")
    samples = make_samples(args.samples, sensor_cfg)

    print("\nThroughput (single core):")
    isolated = bench_processor_isolated(samples, proc_cfg)
    _print_row(isolated)
    e2e = bench_end_to_end(DampedOscillatorSensor(sensor_cfg), args.samples, proc_cfg)
    _print_row(e2e)

    target = 1000.0
    ok = isolated["samples_per_s"] >= target and e2e["samples_per_s"] >= target
    headroom = isolated["samples_per_s"] / target
    print(
        f"\n  target >= {target:,.0f} samp/s: "
        f"{'PASS' if ok else 'FAIL'} "
        f"({headroom:,.0f}x headroom, isolated)"
    )


if __name__ == "__main__":
    main()
