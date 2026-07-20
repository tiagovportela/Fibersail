# Take-Home Exercise: Edge Sensor Processing & Cloud Sync

Role: Mid/Senior Python Software Engineer
Time budget: 3-4 hours (do not over-invest , see "What we're not grading" below)
Format: Submit a git repo (zip or link) with code, tests, and a short README

## Scenario

Our product monitors industrial equipment health using eg vibration sensors mounted directly on machines. Each sensor is read by a small edge device (think limited CPU, limited RAM, intermittent network). The edge device must process sensor data locally in near real time -> computing rolling health metrics and flagging anomalies, and periodically sync summarized data to AWS. The device cannot assume a reliable network connection at all times and cannot afford to lose data during outages.

You'll build a small but realistic slice of this pipeline: a synthetic physical sensor, an edge-side streaming processor, and a cloud sync layer.

## Part 1 , Simulated sensor (physics)

Implement a synthetic vibration sensor as a damped harmonic oscillator driven by noise, sampled at a configurable rate (default 1 kHz):

```
x''(t) + 2*zeta*omega_n*x'(t) + omega_n^2*x(t) = F(t)
```

Requirements:

- Numerically integrate the system (e.g., RK4 or `scipy.integrate`) to produce a displacement or acceleration time series.
- `F(t)` should be mostly Gaussian noise, representing normal operating vibration.
- Add a mechanism to inject a fault: at a chosen time, shift `omega_n` and/or `zeta` (simulating bearing wear or a loosening mount) for a configurable duration, then return to baseline.
- The generator should emit samples as a stream (an iterator/generator or callback), not a precomputed array , the edge service should not assume it has the whole array up front.

You don't need to be a controls engineer , the point is a plausible synthetic time series with a known ground-truth anomaly window, so we can check whether your detector actually finds it. **If you are blocked here, use the sample_dataset_small.csv as base dataset**, state your simplifying assumptions in the README.

## Part 2 , Edge processing service

Build a streaming processor that consumes the sensor stream in real time (or at simulated real-time speed) and:

- Maintains a bounded rolling window (e.g., 1-2 seconds) using an efficient structure, memory must not grow unbounded as the stream runs.
- Computes rolling features per window: RMS, rolling mean/std, and dominant frequency (FFT peak). No look-ahead , only use data available up to the current point.
- Flags anomalies and reports precision/recall-style stats against the known fault-injection window from Part 1.
- Sustains at least 1000 samples/sec of throughput on a single core on typical laptop hardware, without falling behind. Include a way to demonstrate this (a benchmark script or test with timing output).

## Part 3 , Cloud sync (data engineering)

The edge device batches processed windows (features + anomaly flags and 10Hz samples) and uploads them to S3.

- Batch and compress windowed output before upload.
- Use (eg LocalStack) to mock S3 , we do not expect a real AWS account or live credentials.
- Simulate intermittent connectivity: uploads should sometimes fail. Buffer unsent batches durably on local disk (eg. file based queue) so no data is lost across a simulated process restart, and retry with backoff once connectivity returns.
- CreateDesign the S3 object layout/partitioning and briefly justify it in the README.

## Constraints & non-functional requirements

- Pure Python + secure libs is fine; you may use additional libraries but you need justify anything non-obvious.
- The edge processing loop must run within a bounded, predictable memory footprint , call out in the README what that footprint is and how you'd verify it on constrained hardware.
- Code should be structured so Part 2 (edge) and Part 3 (cloud sync) are decoupled , e.g., the edge processor shouldn't block on S3 availability.

## Deliverables

1. Code (Parts 1-3) with tests covering at least: the anomaly detector against the known fault window, the ring buffer's bounded-memory behavior, and the durable-queue-survives-a-restart behavior.
2. `README.md` Open source grade covering:
   - How to run everything (generator → processor → sync) end to end.
   - Key design decisions and trade-offs, and what you'd change with more time.
   - A short (few paragraphs) section: "From prototype to production edge deployment" , how would this actually get deployed and operated on real edge hardware talking to AWS? Touch on things like: edge runtime/orchestration (e.g., AWS IoT Greengrass,, bare systemd), OTA updates, observability/alerting with constrained connectivity, and how you'd size the throughput/memory numbers for real hardware rather than a laptop.

## What we're not grading

- Production-grade AWS IAM/networking setup , mocked S3 is sufficient.
- Front-end/visualization , plain logs or a simple CLI summary are fine.
- Perfect anomaly detection accuracy , we care about a sound approach and honest evaluation of it, not a tuned model.
- Exhaustive test coverage , a handful of meaningful tests beats blanket coverage.

## Stretch goals (optional, only if time remains)

- Replace the threshold detector with a simple Kalman filter or EWMA-based detector and compare.
- Multi-sensor fusion (simulate 2-3 sensors, correlate anomalies across them).
- A one-page design sketch for the real AWS ingestion path at scale (connection/transformation/database) as an alternative to batched S3.

Good luck, we're more interested in your reasoning and trade-offs than a "complete" solution. If you run out of time, tell us what's missing and how you'd approach it during the interview.
