# Fibersail Edge — Sensor Processing & Cloud Sync

A small, realistic slice of an industrial equipment-health pipeline: a synthetic
vibration sensor (physics) → a streaming edge processor → a durable cloud sync
layer. This repository currently implements **Part 1 — the simulated sensor**,
structured so Parts 2 (edge processing) and 3 (cloud sync) drop in on top of it
without rework.

> Full brief: [`take_home_exercise/take_home_exercise.md`](take_home_exercise/take_home_exercise.md).

---

## Status

| Part | Scope | Status |
|------|-------|--------|
| **1** | Simulated sensor — damped harmonic oscillator, streamed, injectable fault | ✅ implemented |
| 2 | Edge processing — bounded rolling window, RMS/std/dominant-freq, anomaly detection | ⏳ next |
| 3 | Cloud sync — batch + compress, durable file queue, LocalStack S3, backoff | ⏳ next |

---

## Repository layout

```
src/fibersail_edge/
  sources.py     # Sample, SampleSource protocol, CsvReplaySource
  sensor.py      # SensorConfig, FaultConfig, DampedOscillatorSensor (RK4 integrator)
  py.typed       # PEP 561 typing marker
tests/
  test_sensor.py # physics, streaming, fault detectability, reproducibility, memory
  test_sources.py# CSV replay parsing + interface conformance
notebooks/
  01_sensor_exploration.ipynb   # visualization + consistency checks (pre-executed)
take_home_exercise/             # the provided brief + sample_dataset_small.csv
pyproject.toml                  # PEP 621 project; uv_build backend; viz extra + dev group
uv.lock                         # pinned, reproducible resolution (committed)
.python-version                 # interpreter pin (3.13)
```

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
uv run pytest                # 20 tests, ~1 s
```

Open / re-run the exploration notebook (already executed with outputs saved):

```bash
uv run jupyter notebook notebooks/01_sensor_exploration.ipynb
# or headless:
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_sensor_exploration.ipynb
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

## What I'd change with more time

- Offer an **exact state-space (matrix-exponential) discretization** as an
  alternative integrator — unconditionally stable and exactly correct for the
  linear system, useful at high `fs` or stiff parameters.
- **Multi-channel emission** in one pass (accel + velocity + displacement, plus a
  temperature channel) to mirror the CSV schema more fully.
- Property-based tests (Hypothesis) over parameter ranges; a small CLI
  (`python -m fibersail_edge.generate ...`) to emit NDJSON/CSV for ad-hoc runs.

---

## Roadmap — Parts 2 & 3

- **Part 2 (edge):** consume `SampleSource.stream()` through a bounded ring
  buffer; compute rolling RMS / mean / std / dominant frequency (FFT) with no
  look-ahead; threshold-detect anomalies and score precision/recall against
  `fault_window`; benchmark ≥1000 samples/s single-core.
- **Part 3 (cloud sync):** batch + compress windowed output; durable file-backed
  queue that survives a process restart; LocalStack-mocked S3 with simulated
  intermittent connectivity and retry-with-backoff; a justified S3 object layout.
  Decoupled from the edge loop so it never blocks on S3 availability.

A "From prototype to production edge deployment" section (edge runtime/OTA/
observability/sizing) will accompany the full submission.
