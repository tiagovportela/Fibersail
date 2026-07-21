# Fibersail Edge — Sensor Processing & Cloud Sync

A realistic slice of an industrial equipment-health pipeline, implemented in three
decoupled parts:

```
DampedOscillatorSensor ─▶ EdgeProcessor ─▶ DurableCloudSink ─▶ S3
   Part 1: synthetic       Part 2: rolling     Part 3: batch+gzip,   (moto / LocalStack
   vibration physics,      features + anomaly  durable disk queue,    / real AWS via
   injectable fault        detection @ 1 kHz   retry w/ backoff       endpoint_url)
```

> Full brief: [`take_home_exercise/take_home_exercise.md`](take_home_exercise/take_home_exercise.md).

```
src/fibersail_edge/
  sources.py   # shared streaming contract (Sample, SampleSource, CsvReplaySource)
  sensor/      # Part 1 — noise-driven damped harmonic oscillator (RK4), fault injection
  edge/        # Part 2 — ring buffer, features, detectors, evaluation, benchmark, compare
  cloud/       # Part 3 — serialization, durable queue, backoff, sink, S3 (only boto3 user)
tests/         # 123 tests: physics, detector vs fault window, bounded memory,
               # ≥1000 samp/s, durable-queue-survives-restart, no-data-loss end-to-end
notebooks/     # pre-executed exploration notebooks for Parts 1 & 2
```

---

## How to run everything (generator → processor → sync)

```bash
uv sync                                # one-time setup; no Docker needed
uv run python -m fibersail_edge.cloud  # the full pipeline, end to end
```

That one command streams 1 kHz samples with an injected fault, reduces them to
10 Hz feature frames with anomaly flags, batches + gzips the frames, buffers them
durably on disk through a **simulated connectivity outage**, uploads to (mock) S3
with retry/backoff, then reads every object back and proves nothing was lost:

```
  frames emitted            : 186
  frames in S3              : 186   ✓ no loss
  bytes gz / raw            : 12,742 / 37,883   (compression 2.97x)
```

Knobs: `--duration-s`, `--outage-start-s/--outage-duration-s`, `--failure-rate`,
`--batch-frames`, `--seed` (see `--help`). S3 is mocked in-process with **moto** by
default. To run against a real S3 endpoint whose objects persist and can be browsed
afterwards with the AWS CLI:

```bash
docker compose up -d localstack                       # real S3 API server on :4566
uv run python -m fibersail_edge.cloud --localstack    # same code, real endpoint
```

Each stage on its own:

```bash
uv run python -m fibersail_edge                  # Parts 1→2: pipeline + precision/recall report
uv run python -m fibersail_edge.edge.benchmark   # Part 2: throughput proof (≥1000 samp/s)
uv run python -m fibersail_edge.edge.compare     # Part 2: baseline vs EWMA vs Kalman detectors
uv run pytest                                    # 123 tests, ~11 s
```

Prefer pip? `pip install -e .` works (PEP 621). Notebooks for Parts 1–2 are under
`notebooks/` (pre-executed; re-run with `uv sync --extra viz`).

---

## Key design decisions & trade-offs

**Two tiny protocol seams decouple the three parts.** `SampleSource` (Part 1→2)
makes the synthetic sensor, the provided CSV replay, and a future real device
interchangeable. `FrameSink` (Part 2→3) is the non-blocking hand-off to the cloud —
a lazy iterator alone would *not* decouple, because a consumer doing blocking S3 I/O
just relocates the stall upstream into the edge loop. Dependency flow is one-way;
the edge processor imports nothing from the cloud layer.

**Sensor: hand-rolled fixed-step RK4, not `scipy.integrate`.** Bounded O(1) memory
(only the 2-element state) and constant per-step cost — what a real-time stream
needs and adaptive solvers don't give. Simplifying assumptions: the Gaussian forcing
is held constant across each RK4 step (band-limited excitation — well-posed, unlike
a literal white-noise ODE), with per-step std scaled by `√fs` so the physics is
sample-rate-invariant; the integrator warms up before `t=0` so the stream starts
stationary. The ground-truth fault window is exposed as **metadata only**
(`fault_window`, `is_faulty(t)`) — never in the sample stream — so the detector
cannot peek at labels and the evaluation stays honest.

**Edge loop: O(1) per sample, expensive work at hop cadence.** Samples land in a
**preallocated numpy ring buffer** (chosen over `deque(maxlen)`: the snapshot is a
contiguous array, so RMS/std and the FFT run as vectorized C, and the buffer never
allocates after construction — which makes the bounded-memory test crisp). Features
(RMS, mean, std, Hann-windowed FFT dominant frequency) are recomputed per 0.1 s hop
rather than maintained incrementally — a running sum-of-squares suffers catastrophic
cancellation on DC-offset signals and drifts on infinite streams. The window is
strictly **causal** (`[t − 1.5 s, t]`, no look-ahead), proven by a prefix-consistency
test: frames from a stream prefix are bit-identical to the full run's. Measured
throughput: **~420,000 samples/s isolated single-core** (~400× the 1 kHz requirement);
the 10 Hz frame stream doubles as the decimated telemetry Part 3 uploads.

**Detection: a self-calibrating z-score detector, evaluated honestly.** It learns a
healthy baseline over the first ~3 s (Welford), then flags when the **dominant
frequency** (primary — the fault is a spectral step, amplitude-invariant) or **RMS**
(secondary — damping drop raises energy) deviates beyond `k = 4σ`, debounced by a
Schmitt trigger. `k` is chosen a priori, not tuned: overlapping windows make frames
autocorrelated, so Gaussian-tail reasoning would be bogus — the false-positive rate
is validated empirically on a healthy stream instead. Because a causal window lags
fault onset and lingers past fault end, `evaluate()` reports **four views side by
side** rather than one flattering number — raw frame metrics (the pessimistic
floor), windowed guard-band metrics (labels a frame faulty iff its trailing window
is majority-faulty — derived, not tuned), detection latency vs. a derived reference,
and event-level detection:

```
raw:      precision 0.647  recall 0.733        detect latency: 0.90 s
windowed: precision 0.853  recall 0.967        event detected: YES
```

**Alternative detectors (stretch goal): EWMA and Kalman, compared.** `Detector` is a
protocol, so both drop in without touching the processor. All three share the same
features, `k`, and debounce, isolating the baseline model; the adaptive pair add
**freeze-on-anomaly** (the baseline learns only from healthy-looking frames, so a
sustained fault can't absorb itself). `python -m fibersail_edge.edge.compare` shows
they are equivalent on a stationary stream — and shows why the adaptive ones exist:
under benign 5 Hz baseline drift the frozen baseline false-positives on **176/360**
frames while EWMA/Kalman stay at **4/360**, all still catching the sharp fault.
Honest takeaway: prefer the simple frozen baseline on stationary signals; switch to
adaptive once the healthy operating point moves over hours (the realistic case).

**Cloud sync: at-least-once delivery + idempotent keys, not exactly-once.**
Exactly-once to S3 needs distributed transactions; instead the one unavoidable
duplicate (a crash between upload-success and ack) is harmless because the object
key is a pure function of the batch's **persisted** identity (`session_id` +
`batch_seq`, frozen into the batch bytes) and the gzip body is byte-deterministic
(`mtime=0`) — a retry overwrites the same key with the same bytes. The **durable
queue** is one file per batch: write `.tmp` → `fsync` → atomic `os.replace`, so a
crash never exposes a half-written batch, and recovery rescans the spool with the
next sequence derived from filenames (no counter file to disagree). Durability unit
is the **batch** (per the brief): frames are durable once their batch hits disk;
`close()` flushes the sub-second in-RAM tail. A hard-crash (`os._exit`) subprocess
test proves recovery does not depend on clean shutdown.

**Batch format: NDJSON + gzip (~3× on this telemetry).** Zero added dependencies
(stdlib `json` + `gzip`), human-inspectable (`… | gunzip | head`), Athena-queryable.
Parquet compresses/queries better at scale, but pyarrow would dwarf every other
dependency on the edge — the production pattern is to land compact NDJSON.gz and
convert downstream where CPU is cheap.

**Libraries (justifying the non-obvious).** Core runtime is **numpy only** — the
edge path must stay lean. `boto3` (the only Part 3 runtime addition) is an opt-in
`[cloud]` extra and imported in exactly one module, so the durable queue, batching,
backoff, and sink are pure stdlib and testable without it. **moto** (dev-only) mocks
S3 in-process — no Docker, deterministic, CI-friendly — while the `endpoint_url`
seam lets the identical client code hit LocalStack or real AWS; scipy/matplotlib
live in the `[viz]` extra for notebooks only.

### Bounded memory footprint (and how to verify it)

The edge loop holds a fixed set of allocations regardless of stream length: the
ring buffer (1.5 s @ 1 kHz = 1500 × 8 B ≈ 12 KB) + cached Hann/rfftfreq tables +
bounded FFT scratch + the detector's scalars — **well under 100 KB, constant
forever**. The cloud sink adds a capped in-memory queue and ≤ one batch in flight;
its RAM is independent of outage length because the durable buffer lives on disk
(optionally capped, drop-oldest). Verified with `tracemalloc` tests that assert flat
peaks across 10–100× longer runs (`test_processor.py`, `test_ring_buffer.py`,
`test_cloud_sink.py`). On constrained hardware: run the same probes (or sample RSS)
over a multi-hour soak under the deployment cgroup limits and confirm a flat curve.

### S3 object layout (and why)

```
telemetry/v1/sensor_id=<id>/date=YYYY-MM-DD/hour=HH/part-<session>-<seq:08d>.ndjson.gz
```

- **Hive-style `key=value` partitions** → Athena/Glue partition pruning for
  time-range queries; cheap prefix listing.
- **`sensor_id` first** → per-device lifecycle/retention rules and IAM prefix
  scoping, and it spreads write load across prefixes (no hot "today" prefix).
- **Deterministic filename** (`session_id` per process run + zero-padded
  `batch_seq`) → a retried upload overwrites rather than duplicates, lexical order
  is chronological within a run, and restarts can never clobber a previous run.
- **`v1`** versions the layout independently of the record `schema_version` inside
  the envelope.
- Caveat, stated honestly: frame time `t` is stream-relative and resets per run, so
  partitioning uses the edge wall-clock at batch close. A real deployment would add
  a per-frame ingest/device timestamp.

---

## What I'd change with more time

- An exact state-space (matrix-exponential) discretization as an alternative
  integrator — unconditionally stable for the linear system at high `fs`.
- Anti-aliased decimation (`scipy.signal.decimate`) for the 10 Hz telemetry, and a
  PR curve by sweeping the detector score threshold.
- Multi-sensor fusion (2–3 correlated sensors) and property-based tests (Hypothesis).
- Part 3: a `dead_letter/` spool (forensics over silent drop) for poison batches, a
  per-frame ingest timestamp, downstream Parquet compaction, and a LocalStack CI job
  alongside the moto unit tests.

---

## From prototype to production edge deployment

**Runtime / orchestration.** Start with a bare `systemd` unit (`Restart=always`,
`MemoryMax`/`CPUQuota` cgroup limits, `WatchdogSec`) running the packaged CLI — lean
and dependency-light for a small homogeneous fleet. Graduate to **AWS IoT
Greengrass v2** when fleet operations dominate: managed component deploys, OTA,
fleet provisioning, and IoT-Core-issued short-lived credentials with offline
spooling, at the cost of a heavier runtime and tighter AWS coupling. The package
supports both unchanged — a plain entry point for `ExecStart`, and config-driven
endpoint/credentials that Greengrass can inject.

**OTA updates.** Ship the versioned, `uv.lock`-pinned wheel as a Greengrass
component version (or a signed apt/OCI artifact) onto an A/B partition with a
post-install health check and automatic rollback. The `telemetry/v1` layout prefix
and the envelope `schema_version` let old and new firmware coexist during a phased
rollout — a half-updated fleet writes objects every reader understands.

**Observability with constrained connectivity.** Structured JSON logs into a
bounded local ring buffer; metrics shipped through the *same* durable-queue pattern
when a link is up. The signals that matter: queue depth, disk-spill high-water,
dropped-frame count, and **time-since-last-successful-sync** — alerted on
**cloud-side as a dead-man's switch**, because an offline device cannot raise its
own alarm. Rate-limit anomaly events so a fault storm can't saturate a thin uplink.

**Sizing for real hardware.** The laptop benchmark (~400× headroom) is a ceiling,
not a spec: re-run `benchmark.py` on the target SoC under the deployment cgroup
limits with BLAS/OMP threads pinned to 1, and verify the flat memory curve over a
multi-day soak. Size the disk buffer from the outage budget —
`compressed_batch_size × max_outage_s / batch_period_s` (a 6-hour outage at one
~3.4 KB batch per 5 s ≈ 15 MB) — and cap it with the queue's `max_bytes`.

**At scale** (stretch sketch): keep landing compact NDJSON.gz batches in S3, then
compact/convert to Parquet downstream (Firehose/Glue) under the same partition
scheme for Athena; if some signals need seconds-level reaction, add a hot path
(IoT Core → Kinesis/Flink → alerting) beside S3 as the durable cold store — a
latency/cost trade-off, not a correctness one.
