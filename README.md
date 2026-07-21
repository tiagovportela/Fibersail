# Fibersail Edge — Sensor Processing & Cloud Sync

A small, realistic slice of an industrial equipment-health pipeline: a synthetic
vibration sensor (physics) → a streaming edge processor → a durable cloud sync
layer. This repository implements all three parts — **Part 1** the simulated
sensor, **Part 2** the edge processing service, and **Part 3** the durable cloud
sync — each decoupled behind a small protocol so the next drops in without rework.

> Full brief: [`take_home_exercise/take_home_exercise.md`](take_home_exercise/take_home_exercise.md).

---

## Status

| Part | Scope | Status |
|------|-------|--------|
| **1** | Simulated sensor — damped harmonic oscillator, streamed, injectable fault | ✅ implemented |
| **2** | Edge processing — bounded ring buffer, RMS/std/dominant-freq, anomaly detection, ≥1 kHz | ✅ implemented |
| **3** | Cloud sync — batch + compress, durable file queue, retry/backoff, mock S3 | ✅ implemented |

---

## Repository layout

The package is organized by exercise part, over one shared streaming contract:

```
src/fibersail_edge/
  sources.py         # Sample, SampleSource protocol, CsvReplaySource   (shared contract)
  __main__.py        # end-to-end demo + eval   (python -m fibersail_edge)
  py.typed           # PEP 561 typing marker
  sensor/            # ── Part 1 — synthetic sensor ──
    oscillator.py    #   SensorConfig, FaultConfig, DampedOscillatorSensor
  edge/              # ── Part 2 — edge processing ──
    ring_buffer.py   #   RingBuffer — preallocated numpy circular buffer
    features.py      #   FeatureFrame, FeatureExtractor (RMS/std/dominant freq)
    detector.py      #   Detector protocol, BaselineZScoreDetector
    sink.py          #   FrameSink seam (ListSink/CallbackSink) → Part 3
    processor.py     #   ProcessorConfig, EdgeProcessor (ties it together)
    evaluation.py    #   precision/recall, guard-band + latency, evaluate()
    benchmark.py     #   throughput harness (python -m fibersail_edge.edge.benchmark)
  cloud/             # ── Part 3 — cloud sync ──
    serialization.py #   BatchHeader, NDJSON+gzip codec, S3 key layout
    durable_queue.py #   DurableQueue — crash-safe, restart-recoverable file FIFO
    backoff.py       #   exponential backoff with jitter
    uploader.py      #   Uploader protocol + InMemory/Flaky doubles
    sink.py          #   DurableCloudSink (implements FrameSink)
    s3.py            #   boto3 S3 transport (the ONLY boto3 importer)
    __main__.py      #   end-to-end demo (python -m fibersail_edge.cloud)
tests/
  test_sensor.py / test_sources.py                 # Part 1
  test_ring_buffer.py test_features.py test_detector.py test_processor.py
  test_evaluation.py test_throughput.py test_sink.py   # Part 2
  test_cloud_serialization.py test_durable_queue.py test_backoff.py test_uploader.py
  test_cloud_sink.py test_cloud_s3.py test_cloud_restart_subprocess.py   # Part 3
notebooks/
  01_sensor_exploration.ipynb   # Part 1 — signal viz + consistency (pre-executed)
  02_edge_processing.ipynb      # Part 2 — features, detection, throughput (pre-executed)
take_home_exercise/             # the provided brief + sample_dataset_small.csv
pyproject.toml                  # PEP 621; uv_build backend; viz + cloud extras + dev group
uv.lock                         # pinned, reproducible resolution (committed)
.python-version                 # interpreter pin (3.13)
```

The whole public API is re-exported from the top level, so `from fibersail_edge
import EdgeProcessor` works regardless of layout; `from fibersail_edge.edge import
…` / `from fibersail_edge.sensor import …` are available for callers that prefer to
name the layer. The `cloud` subpackage is imported explicitly
(`from fibersail_edge.cloud import DurableCloudSink`) and is **not** re-exported at
the top level, so `import fibersail_edge` stays numpy-only (no boto3).

Standard **src layout** built with uv's native **`uv_build`** backend.

---

## Install

Managed with [**uv**](https://docs.astral.sh/uv/). The core runtime needs only
**numpy**; analysis/visualization tooling lives in the `viz` extra so the edge
path stays lean, and dev tooling (pytest) is a PEP 735 dependency group.

```bash
uv sync --extra viz     # creates .venv, installs core + viz + the default dev group
```

- `uv sync` alone installs just the core (numpy) + dev group.
- The exact resolution is pinned in `uv.lock`; the interpreter is pinned to 3.13
  by `.python-version`. uv fetches that Python automatically if it's missing.
- Prefer plain pip? `pip install -e ".[viz]"` still works — it's a standard
  PEP 621 project. Tests also import from the repo root (`pythonpath = ["."]`),
  so no install is strictly required to run them.

---

## Quickstart

Generate a stream with an injected fault and pull a few samples:

```python
import itertools
from fibersail_edge import DampedOscillatorSensor, SensorConfig, FaultConfig

cfg = SensorConfig(
    sample_rate_hz=1000.0, natural_freq_hz=50.0, zeta=0.05,
    channel="acceleration", duration_s=10.0, seed=42,
    fault=FaultConfig(start_s=4.0, duration_s=2.0, omega_n_factor=0.7, zeta_factor=0.4),
)
sensor = DampedOscillatorSensor(cfg)

for sample in itertools.islice(sensor.stream(), 3):
    print(sample)                 # Sample(t=..., value=...)

print("ground-truth fault window:", sensor.fault_window)   # (4.0, 6.0)
```

Swap in the provided dataset behind the identical interface:

```python
from fibersail_edge import CsvReplaySource
src = CsvReplaySource("take_home_exercise/sample_dataset_small.csv")
for sample in itertools.islice(src.stream(), 3):
    print(sample)
```

Run the code and tests through the uv environment (`uv run <cmd>` executes inside
`.venv` without activating it):

```bash
uv run python -c "import fibersail_edge; print(fibersail_edge.__version__)"
uv run pytest                       # 103 tests, ~10 s
uv run python -m fibersail_edge     # Part 2: run the pipeline + print the eval summary
uv run python -m fibersail_edge.edge.benchmark   # Part 2: throughput table (≥1000 samp/s)
uv run python -m fibersail_edge.cloud            # Part 3: cloud sync demo (mock S3 via moto)
```

`uv sync` installs the `dev` group, which includes `moto` — so the cloud tests and
the demo run out of the box. For a production edge build that needs the real boto3
S3 client (but not moto), install the opt-in extra: `uv sync --extra cloud`.

Feed the sensor stream through the edge processor and read off feature frames:

```python
from fibersail_edge import DampedOscillatorSensor, SensorConfig, FaultConfig, EdgeProcessor, evaluate

sensor = DampedOscillatorSensor(SensorConfig(
    duration_s=15.0, seed=42,
    fault=FaultConfig(start_s=7.0, duration_s=3.0, omega_n_factor=0.7, zeta_factor=0.4),
))
processor = EdgeProcessor.for_source(sensor)          # sizes the window from the source rate

for frame in processor.process_stream(sensor):
    print(frame.t, frame.rms, frame.dominant_freq_hz, frame.is_anomaly)

print(evaluate(sensor, processor).format_summary())   # precision/recall vs the fault window
```

Open / re-run the exploration notebooks (already executed with outputs saved):

```bash
uv run jupyter notebook notebooks/02_edge_processing.ipynb
# or headless:
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_edge_processing.ipynb
```

---

## Part 1 — Simulated sensor (physics)

### Model

A single-degree-of-freedom vibrating structure — a decent first approximation of
an accelerometer on a machine — driven by noise:

```
x''(t) + 2·ζ·ωₙ·x'(t) + ωₙ²·x(t) = F(t)
```

- `ωₙ = 2π·natural_freq_hz` — natural frequency, `ζ` — damping ratio.
- `F(t)` — Gaussian forcing (broadband ambient excitation) in normal operation.
- The emitted channel is configurable: `acceleration` (default, what a real
  vibration sensor reports), `velocity`, or `displacement`.

### Numerical method — and why

Fixed-step **RK4**, hand-rolled, integrating the state `[x, v]`:

- **Bounded O(1) memory.** Only the current 2-element state is retained; the
  series is never materialized. `stream()` is a lazy generator — the edge service
  "should not assume it has the whole array up front."
- **Deterministic, constant per-step cost.** Real-time streaming needs
  predictable timing, which adaptive step-size control (`scipy.integrate`) does
  not provide.

### Fault injection & ground truth

A `FaultConfig` multiplies the baseline parameters for a window, then the system
returns to baseline:

- `omega_n_factor < 1` — loss of effective stiffness (**bearing wear**) → lower
  resonant frequency.
- `zeta_factor < 1` — a **loosening mount** damps less → sharper, higher-amplitude
  resonance.

The fault window is exposed as **metadata** (`sensor.fault_window`,
`sensor.is_faulty(t)`) and is **never** part of the sample stream, so a detector
cannot peek at labels. Part 2 uses it offline to compute precision/recall.

The notebook confirms the fault is genuinely detectable: the spectral peak slides
from 50 Hz → 35 Hz and the rolling RMS rises inside the window.

### Simplifying assumptions

- **Forcing is piecewise-constant per step** (one Gaussian draw per RK4 step). A
  literally white-noise-driven ODE is a stochastic differential equation for
  which RK4 is not formally convergent; holding the force constant over a step
  makes it a *band-limited* excitation — physically reasonable (real forcing is
  band-limited) and numerically well-posed.
- **Sample-rate-invariant noise:** per-step force std = `force_std·√fs`
  (Euler–Maruyama-consistent), so changing `fs` doesn't change the physical
  response around `ωₙ`.
- **Warmup:** the integrator runs `warmup_s` before `t=0` so the emitted signal
  is already stationary at the first sample (transient time constant
  `τ = 1/(ζ·ωₙ) ≈ 64 ms` at defaults).
- **`natural_freq_hz` must be below Nyquist** (`fs/2`); enforced in `SensorConfig`.
- **CSV replay** (`sample_dataset_small.csv`, `;`-delimited, ~364 Hz grid with
  values on ~every 4th row): a row is a sample only when `strain` is present;
  blank filler rows are skipped rather than forward-filled (forward-fill would
  fabricate spectral content). Effective rate ≈ 91 Hz, inferred from the median
  spacing of populated rows. `strain` is the emitted value; `temperature` is
  available via `stream_full()`. No known ground truth → `fault_window is None`.

### Bounded-memory footprint (and how to verify)

The generator holds only the 2-float state plus a handful of scalars — **O(1)**,
independent of stream length. The notebook demonstrates this with `tracemalloc`:
consuming 5k vs. 50k samples shows an identical peak (`1,776 → 1,776 bytes`).
`tests/test_sensor.py::test_memory_is_bounded` asserts the same property. On
constrained hardware you'd verify with the same `tracemalloc` probe (or RSS
sampling) over a long run and confirm a flat curve.

---

## Part 2 — Edge processing service

A streaming processor that consumes any `SampleSource` in (simulated) real time,
maintains a bounded rolling window, computes rolling health features with **no
look-ahead**, flags anomalies, and emits compact **feature frames** for Part 3 —
without ever blocking on the cloud.

```
Sample ─▶ RingBuffer (trailing window) ─▶ FeatureExtractor ─▶ Detector ─▶ FeatureFrame ─▶ FrameSink ─▶ (Part 3)
          bounded, O(1)/sample            RMS/mean/std/FFT     z-score      10 Hz, w/ flag   non-blocking
```

The per-sample hot path is just a ring-buffer push; the expensive work (one FFT +
a few reductions) runs only once per **hop** (default every 0.1 s → 10 frames/s).

### Bounded rolling window — a preallocated ring buffer

`RingBuffer` writes into a single `np.empty(capacity)` allocated once at
construction and never again. Chosen over `collections.deque(maxlen=N)` because:

- **Vectorized features for free.** `snapshot()` returns a contiguous `float64`
  array, so `np.fft.rfft` and the reductions run as pure C — a deque of boxed
  Python floats would need an O(N) copy-and-unbox every hop.
- **Provably bounded memory.** No allocation after construction, so a test can
  push 10k vs. 1M samples through the same buffer and see an essentially identical
  peak (`tests/test_ring_buffer.py::test_memory_is_bounded`).

### Rolling features (RMS, mean, std, dominant frequency)

- **Causal / no look-ahead.** The window is *trailing* — `[t − window_s, t]` — and
  the frame timestamp is the newest sample's time. Nothing is emitted until the
  buffer first fills (a ~`window_s` silent start, well before any fault).
- **Recompute per hop, don't accumulate.** Stats are recomputed from the snapshot
  with numpy's pairwise-summation `mean`/`std`. The O(1) running-sum alternative
  suffers **catastrophic cancellation** on a DC-heavy signal (the CSV strain sits
  near 1500, so `E[x²] ≈ E[x]²`) and drifts on an infinite stream; recompute is
  both more robust and, at 10 Hz over a ~1500-sample window, negligibly cheap.
- **Detrend before the FFT.** The window mean is subtracted before the Hann taper,
  so a DC offset can't dominate the peak search (critical for the CSV source).
- **Resolution.** Δf = fs/N ≈ 0.67 Hz at 1.5 s / 1 kHz, so the 50→35 Hz fault
  shift spans ~22 bins — resolved with wide margin. An optional parabolic sub-bin
  refinement sharpens the peak estimate.
- **10 Hz telemetry.** Each frame carries the newest raw sample; the frame stream
  *is* the ~10 Hz decimated telemetry Part 3 uploads. This is **point** decimation
  (not anti-aliased) — acceptable because spectral content is already summarized by
  `dominant_freq_hz`; `scipy.signal.decimate` is the "with more time" upgrade.

### Anomaly detection — a self-calibrating baseline z-score detector

Streaming and O(1) in memory (a handful of scalars). It learns a healthy baseline
over the first `calibration_frames` (~3 s) via **Welford's** online algorithm, then
flags a frame when *either* monitored feature deviates beyond `k = 4σ`:

- **`dominant_freq_hz` is the primary signature** — the fault is a spectral step
  (amplitude-invariant, so robust to benign gain/load changes). **`rms` is
  secondary** (damping drops → the resonance rings harder). `mean`/`std` are
  dropped as uninformative/redundant on the AC channel.
- A two-counter **Schmitt-trigger debounce** (fire after 3 consecutive deviating
  frames, clear after 5 calm ones) trades a little onset latency for far fewer
  false positives, and standard-deviation **floors** stop a near-constant baseline
  (especially a single-bin FFT peak) from producing spurious flags.
- **Honesty on the threshold.** Overlapping windows make consecutive frames heavily
  autocorrelated — *not* IID — so `k` is chosen a priori (a control-chart value),
  **not** justified from Gaussian tails, and the false-positive rate is validated
  empirically on a healthy stream (`test_no_anomalies_on_healthy_stream`).

`Detector` is a `Protocol`, so an EWMA/Kalman alternative drops in without touching
the processor (see "What I'd change").

### Honest precision / recall against the ground-truth fault

`evaluate()` scores the detector in one streaming O(1) pass and reports **four
complementary views** — never just the flattering one — because a *causal* window
creates two unavoidable boundary artifacts: onset lag (the window is still healthy
right after the fault starts → false negatives) and trailing lag (the window is
still faulty right after it ends → false positives).

`python -m fibersail_edge` on the default fault (50→35 Hz, 7–10 s) prints:

```
  event detected : YES
  detect latency : 0.90 s (reference ≈ 1.05 s)

  raw (point labels — pessimistic floor):
    precision= 0.647  recall= 0.733  F1= 0.688  FPR= 0.113
  windowed (majority-faulty, ±0.75 s guard):
    precision= 0.853  recall= 0.967  F1= 0.906  FPR= 0.047
```

1. **Raw** frame-level metrics (point labels) — the pessimistic, honest floor.
2. **Windowed** metrics — a frame is faulty iff its trailing window is *majority*
   faulty. This is a first-principles guard band of `window_s/2`, **printed and
   derived** — not a hand-tuned fudge. (The rejected alternative, silently shifting
   labels by `window_s/2` to "align" with the lag, is disguised tuning.)
3. **Detection latency** vs. a derived reference (`window_s/2` + debounce). Here
   0.90 s beats the 1.05 s reference — a strong frequency drop trips the threshold
   before the window is even majority-faulty.
4. **Event detected** — the operational "did we catch it at all".

### Throughput — ≥1000 samples/s single-core

The per-sample cost is one ring-buffer write; the FFT runs only 10×/s. Measured on
a laptop (`python -m fibersail_edge.edge.benchmark`, single core):

| measurement | samples/s | × real-time |
|-------------|-----------|-------------|
| processing only (isolated) | ~420,000 | ~420× |
| end-to-end (sensor + processing) | ~137,000 | ~137× |

That is ~400× the 1 kHz bar — the loop demonstrably never falls behind.
`tests/test_throughput.py -s` prints and asserts it.

### Bounded-memory footprint (and how to verify)

Everything is fixed-size regardless of stream length: the ring buffer (1.5 s @
1 kHz = 1500 × 8 B ≈ **12 KB**) + the cached Hann window and `rfftfreq` table
(~12 KB each) + bounded FFT scratch + the detector's handful of scalars — **well
under 100 KB**, constant forever. Verified with `tracemalloc` in
`tests/test_processor.py::test_bounded_memory_over_long_run` (peak flat across a
10× longer run); on constrained hardware you'd confirm the same flat curve (or RSS)
over a long soak.

### Decoupling from Part 3 — the `FrameSink` seam

A lazy iterator alone does **not** decouple: a consumer doing blocking S3 I/O just
relocates the stall upstream. So Part 2 defines the non-blocking contract
(`FrameSink.emit` must be O(1)) and ships in-memory doubles (`ListSink`,
`CallbackSink`); `EdgeProcessor.run(source, sink)` drives it. The processor imports
**nothing** from the cloud layer — dependency flow is one-way, so Part 3's durable,
file-queue-backed, thread-safe sink drops in behind this interface and the edge
loop can never block on S3.

---

## Part 3 — Cloud sync

Batches + compresses feature frames, buffers them **durably on local disk** so
nothing is lost across a process restart, and uploads them to S3 with retry/backoff
under intermittent connectivity — **without the edge loop ever blocking on S3**.

```
EdgeProcessor ─▶ DurableCloudSink.emit  (O(1), non-blocking, drop-newest if full)
                     │  [batcher thread]  batch → NDJSON+gzip → fsync
                     ▼
                  DurableQueue (disk spool)  ──[uploader thread]──▶ Uploader ─▶ S3
                   crash-safe, FIFO             peek→upload→ack           (moto |
                                                retry w/ backoff           LocalStack |
                                                                           real AWS)
```

### Delivery guarantee & idempotency

Delivery is **at-least-once**: `peek → upload → ack(delete)`. The one unavoidable
duplicate — a crash between a successful upload and its ack — is made *harmless* by
two properties working together: the S3 object key is a **pure function of the
batch's persisted identity** (`session_id` + `batch_seq`, frozen into the batch
bytes at creation), and the serialized body is **byte-identical** (`gzip` with
`mtime=0`). So a retried upload writes the *same* key with the *same* bytes — an
idempotent overwrite, never a duplicate. A fresh `session_id` per process keeps
recovered batches from an old run from ever clobbering a new run's objects.

### Durable queue — crash-safe by construction

`DurableQueue` is one file per batch in a spool dir. Enqueue is atomic and durable:
write `<seq>.batch.tmp` → `fsync(file)` → `os.replace()` (atomic on POSIX/Windows) →
`fsync(dir)`. A crash therefore never leaves a half-written batch visible, and a
committed batch survives power loss. On construction it recovers: orphan `*.tmp`
files (writes that never published) are deleted, committed batches are the queue,
and the next sequence continues from `max(seq)+1` — no counter file to disagree with
the spool. `tests/test_durable_queue.py::test_survives_restart_recovers_in_fifo_order`
and a hard-crash (`os._exit`) subprocess test prove recovery rides on `fsync`, not on
any clean-shutdown hook.

### Serialization & compression — NDJSON + gzip

Chosen because it adds **zero runtime dependencies** (`json` + `gzip` are stdlib) —
the only new runtime dep in all of Part 3 is `boto3`, for transport — while staying
human-inspectable (`aws s3 cp … - | gunzip | head`), streamable, and Athena/Glue
queryable. Line 0 is a self-describing header; lines 1..N are one `FeatureFrame`
each. Non-finite floats serialize to `null` (never invalid `NaN`/`Infinity`), and
the Part 2 `bool()` cast on `is_anomaly` is what keeps `numpy.bool_` from breaking
JSON. Measured compression on float-heavy telemetry is **~2.9×** (larger batches
compress better; Parquet is the natural *downstream* format — see "at scale" below —
but pyarrow on the edge would dwarf every other dependency, so we land compact
NDJSON.gz and convert downstream where CPU is cheap).

### S3 object layout (and why)

```
telemetry/v1/sensor_id=<id>/date=YYYY-MM-DD/hour=HH/part-<session>-<seq:08d>.ndjson.gz
```

| Property | How the layout delivers it |
|----------|----------------------------|
| Time-range queries | `date=`/`hour=` Hive partitions → Athena partition pruning; cheap prefix listing |
| Per-device isolation | leading `sensor_id=` → per-device lifecycle/retention + IAM prefix scoping |
| No hot prefixes | high-cardinality `sensor_id` first spreads writes (S3 scales per prefix) |
| Idempotent writes | `(session_id, batch_seq)` filename is deterministic → retry overwrites, no dup |
| Layout vs record evolution | `v1` prefix = layout generation; envelope `schema_version` = record schema |

**Timestamp caveat (called out honestly):** a `FeatureFrame`'s `t` is stream-relative
and resets every run, so it can't globally bucket batches. Partitioning uses the
edge **wall-clock at batch close** (`created_at_utc`), with `session_id` + `batch_seq`
for restart-safe ordering. A real deployment would add a per-frame ingest/device
timestamp; the batch-close wall-clock is the honest stand-in here.

### Non-blocking, bounded-memory sink

`emit()` is O(1): it drops the frame onto a bounded in-memory `queue.Queue`
(`put_nowait`; on overflow it drops the newest and counts it — the sole backpressure
valve). A **batcher thread** accumulates to `batch_max_frames` or `batch_max_seconds`
(monotonic clock, so NTP steps can't stall it), serializes, and spools durably. An
**uploader thread** drains the spool with exponential backoff + jitter; a transient
error leaves the batch on disk to retry, a permanent one is dead-lettered so it can't
block the queue head. `close()` flushes the final partial batch, then drains
(bounded by a timeout). Sink RAM is `O(queue_maxsize + batch_max_frames + one batch)`
— **independent of outage length or spool depth** (the durable buffer lives on disk),
verified by a `tracemalloc` backpressure test.

### Backends — moto (default) and LocalStack, via one `endpoint_url` seam

`S3Config.endpoint_url` is the single knob that lets the *identical* `S3Uploader`
code target three backends — only the config changes, and `boto3` is imported
**only** in `cloud/s3.py`:

- **moto** (default, no endpoint): an in-process mock — no Docker, deterministic,
  CI-friendly. The whole automated suite uses it (`mock_aws`). The catch: objects
  live inside the process and vanish when it exits.
- **LocalStack** (`--localstack`): a real S3 API server in a container — the
  closest setup to production. Objects **persist** for the life of the container,
  so you can browse them with the AWS CLI after the run.
- **real AWS**: same code, no endpoint, real credentials/region instead of the
  dummy ones.

### Run it — moto (zero setup)

```bash
uv run python -m fibersail_edge.cloud --outage-start-s 0 --outage-duration-s 2
```

emits ~286 feature frames through a simulated 2 s outage and prints, e.g.:

```
Backend: moto (in-process mock; objects vanish when this process exits)
  simulated upload failures : 7
  batches enqueued / uploaded: 6 / 6
  objects in S3             : 6
  example key               : telemetry/v1/sensor_id=press-042/date=…/hour=…/part-…-00000000.ndjson.gz
  frames emitted            : 286
  frames in S3              : 286   ✓ no loss
  bytes gz / raw            : 20,644 / 59,140   (compression 2.86x)
```

### Run it — LocalStack (production-faithful; browse the real objects)

```bash
docker compose up -d localstack                       # real S3 server on :4566
uv run python -m fibersail_edge.cloud --localstack    # same pipeline, real endpoint

# the objects are really there — browse them (dummy creds are fine for LocalStack):
export AWS_ACCESS_KEY_ID=testing AWS_SECRET_ACCESS_KEY=testing
aws --endpoint-url=http://localhost:4566 s3 ls s3://fibersail-telemetry/ --recursive
aws --endpoint-url=http://localhost:4566 s3 cp s3://fibersail-telemetry/<key> - | gunzip | head
docker compose down
```

The demo prints these browse commands (with a real key filled in) at the end of a
`--localstack` run. `--endpoint-url <url>` targets any other S3-compatible endpoint.

---

## Key design decisions & trade-offs

- **One streaming contract, many sources.** `SampleSource` (a `Protocol`) is the
  seam that decouples Parts 2/3 from data origin — synthetic physics, recorded
  CSV, or a real device later are interchangeable. `Sample` is a `NamedTuple`
  (immutable, cheap to build/unpack) because Part 2 will pull it at ≥1 kHz.
- **Minimal core dependency (numpy only).** Analysis libraries (scipy/matplotlib/
  pandas/jupyter) are an optional extra, keeping the edge runtime small — relevant
  to Part 2's memory budget.
- **RK4 with piecewise-constant force** over an exact linear-SDE discretization:
  simpler and directly matches the brief, at the cost of not being the strictly
  "optimal" stochastic integrator. Documented as an assumption above.
- **Ground truth kept out of the stream** — a deliberate correctness boundary so
  the detector evaluation in Part 2 is honest.
- **Decoupled cadence (Part 2).** Per-sample work is O(1) (a ring push); the FFT
  runs only at the hop rate. This is what buys the ~400× throughput headroom while
  keeping features causal.
- **Pluggable detector behind a `Protocol`.** One baseline z-score detector ships;
  the seam lets an EWMA/Kalman variant drop in without touching the processor.
- **At-least-once + idempotent keys (Part 3)** over exactly-once: exactly-once to S3
  needs distributed transactions; a deterministic key + byte-identical body makes the
  lone crash-window duplicate a harmless overwrite instead.
- **boto3 isolated to one module (Part 3).** The durable queue, batching, backoff,
  and sink are pure stdlib, so the correctness-critical core is testable with no
  cloud extra and no Docker; only `cloud/s3.py` needs boto3.

## What I'd change with more time

- Offer an **exact state-space (matrix-exponential) discretization** as an
  alternative integrator — unconditionally stable and exactly correct for the
  linear system, useful at high `fs` or stiff parameters.
- **Multi-channel emission** in one pass (accel + velocity + displacement, plus a
  temperature channel) to mirror the CSV schema more fully.
- An **EWMA / Kalman detector** (with freeze-on-anomaly so a sustained fault can't
  drag the adaptive baseline and mask itself) as a second `Detector`, and a
  quantitative comparison against the baseline z-score version.
- **Anti-aliased decimation** (`scipy.signal.decimate`) for the 10 Hz telemetry,
  and a **PR curve** by sweeping the detector score threshold.
- **Multi-sensor fusion** (2–3 correlated sensors) and property-based tests
  (Hypothesis) over parameter ranges.
- **Part 3:** a `dead_letter/` spool (forensics over silent drop) for poison batches;
  a per-frame ingest timestamp; downstream Parquet conversion; and a real LocalStack
  end-to-end CI job alongside the moto unit tests.

---

## From prototype to production edge deployment

How this slice would actually get deployed and operated on real edge hardware
talking to AWS:

- **Runtime / orchestration — bare `systemd` first, AWS IoT Greengrass when fleet
  ops dominate.** A `systemd` unit (`Restart=always`, `MemoryMax`/`CPUQuota` cgroup
  limits, `WatchdogSec`) is the lean, dependency-light default for a small
  homogeneous fleet — it just runs the packaged CLI. Greengrass v2 buys managed
  component deploys, OTA, local pub/sub, fleet provisioning, and IoT-Core-issued
  short-lived credentials (via a role alias) with offline spooling — at the cost of
  a heavier Nucleus runtime and tighter AWS coupling. The package supports both
  unchanged: a plain entry point for `systemd`'s `ExecStart`, and config-driven
  `endpoint_url`/credentials that Greengrass can inject.

- **OTA updates.** Ship the versioned, `uv.lock`-pinned wheel as a Greengrass
  component version (or a signed apt/OCI artifact) onto an A/B partition with a
  post-install health check and automatic rollback. The `telemetry/v1` layout
  prefix, the envelope `schema_version`, and versioned config let new and old
  firmware **coexist during a phased rollout** — a half-updated fleet writes objects
  both readers understand.

- **Observability under constrained connectivity.** Emit structured JSON logs to a
  bounded local ring buffer; ship metrics through the *same* durable-queue pattern
  to CloudWatch (EMF) / IoT Core when a link is up. Surface **queue depth**,
  **disk-spill high-water**, **dropped-frame count**, and **time-since-last-successful
  sync**, and alert on that last one **cloud-side as a dead-man's switch** — an
  offline device can't raise its own alarm, so absence-of-heartbeat is the signal.
  Rate-limit anomaly events so a fault storm can't saturate a thin uplink.

- **Sizing throughput & memory for real hardware.** The laptop `benchmark.py` number
  (~400× headroom) is a *ceiling*, not a spec. Re-run it on the target SoC under the
  same cgroup limits as deployment, pin the interpreter, and set
  `OMP/OPENBLAS_NUM_THREADS=1` for predictable single-core timing. Size the disk
  buffer from the outage budget: `buffer_bytes ≈ compressed_batch_size ×
  (max_outage_seconds / batch_period_seconds)` — e.g. a 6-hour outage at one ~3.4 KB
  batch / 5 s ≈ ~15 MB, trivially bounded by the queue's `max_bytes` cap. Verify the
  flat memory curve with `tracemalloc`/RSS over a multi-day soak (leaks, fragmentation).

### Optional: a one-page sketch for AWS ingestion at scale

Batched S3 (what this repo does) is cheap, simple, and durable, at the cost of
minutes-scale latency — the right default for equipment-health telemetry. If some
signals need near-real-time reaction, the fuller path would be:

```
device ──(MQTT / presigned S3 PUT)──▶ S3 raw (this NDJSON.gz Hive layout)
                                         │  S3 event
                                         ▼
                         Firehose / Glue  ──compact + convert──▶ S3 Parquet (same partitions)
                                                                   │
                                                                   ▼  Athena / Redshift Spectrum
      hot path (optional): Kinesis ──▶ Flink ──▶ Timestream + alerting (S3 stays the cold store)
```

The edge deliberately lands compact NDJSON.gz and lets a downstream job convert to
columnar Parquet where CPU is cheap; the Hive partition scheme above is chosen so
that conversion is drop-in. Batched-S3 vs streaming is a latency/cost trade-off, not
a correctness one — both keep S3 as the durable system of record.
