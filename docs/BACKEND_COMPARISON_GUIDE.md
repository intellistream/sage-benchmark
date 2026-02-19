# End-to-End Backend Comparison Guide

> **Status**: Complete (Issue #8)  
> **Scope**: Workload4 (distributed RAG pipeline) — SAGE vs Ray side-by-side  
> **Architecture note**: Ray is an *external, optional* baseline only.  
> It is **never** imported into SAGE core or sageFlownet.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick reference: CLI flags](#quick-reference-cli-flags)
5. [Step-by-step: Single backend run](#step-by-step-single-backend-run)
6. [Step-by-step: Paired comparison run](#step-by-step-paired-comparison-run)
7. [Reproducibility controls](#reproducibility-controls)
8. [Understanding output artifacts](#understanding-output-artifacts)
9. [Generating a comparison report](#generating-a-comparison-report)
10. [Architecture: why Ray stays in sage-benchmark](#architecture-why-ray-stays-in-sage-benchmark)
11. [Adding a new backend](#adding-a-new-backend)
12. [Troubleshooting](#troubleshooting)

---

## Overview

`sage-benchmark` supports running the same workload against multiple runtime
backends so results are directly comparable.  Two backends ship today:

| Backend name | Module | Runtime |
|---|---|---|
| `sage` *(default)* | `experiments/backends/sage_runner.py` | SAGE `FlownetEnvironment` (backed by sageFlownet) |
| `ray` *(optional)* | `experiments/backends/ray_runner.py` | Ray `@ray.remote` tasks |

The workload business logic (sources, operators, sinks, metrics) is
expressed once via `WorkloadSpec` + `RunResult` and runs unchanged on either
backend.

---

## Prerequisites

- Python 3.10+
- SAGE installed in editable mode (`./quickstart.sh --dev --yes` in the SAGE
  repo), **or** `pip install -e .` in this repo for the SAGE runner
- For the Ray baseline only: `ray>=2.9,<3.0`

Check your active environment:

```bash
python -c "import sage; print('SAGE OK')"
python -c "import ray; print(ray.__version__)"   # only if testing Ray path
```

---

## Installation

```bash
# Clone and enter the repo
git clone https://github.com/intellistream/sage-benchmark.git
cd sage-benchmark

# Default: SAGE backend only (no Ray dependency)
pip install -e .

# Optional: enable the Ray baseline backend
pip install -e .[ray-baseline]
```

> **Important**: the default install does **not** require Ray.
> Calling `--backend ray` without the `ray-baseline` extra raises a clear
> `RuntimeError` with the install command — it does not silently fall back
> or crash with an `ImportError`.

---

## Quick reference: CLI flags

All workload entry points share the same standardised flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--backend {sage,ray}` | `sage` | Runtime backend to use |
| `--nodes N` | `1` | Worker nodes (distributed mode) |
| `--parallelism P` | `2` | Operator-level parallelism hint |
| `--repeat R` | `1` | Independent repetitions (results averaged) |
| `--seed SEED` | `42` | Global RNG seed — **must match across backends** |
| `--output-dir DIR` | `results` | Root directory for all artifacts |
| `--quick` | off | Reduced-scale smoke-test run |
| `--dry-run` | off | Validate configuration, skip execution |
| `--verbose` / `-v` | off | Enable debug-level output |
| `--warmup-items N` | `0` | Items to process before timing begins |
| `--parity-batch-size N` | `16` | Batch size for deterministic input parity |
| `--num-tasks N` | workload-specific | Number of top-level tasks to process |

---

## Step-by-step: Single backend run

### 1. Run Workload4 on SAGE (default)

```bash
cd /path/to/sage-benchmark

python experiments/distributed_workloads/run_workload4.py \
  --backend sage \
  --num-tasks 20 \
  --parallelism 4 \
  --seed 42 \
  --output-dir results/wl4_sage
```

Expected artifacts under `results/wl4_sage/`:

```
unified_results.jsonl      # one JSON record per run
unified_results.csv        # same data in CSV
repro_manifest.json        # seed, config_hash, warmup split, parity batch
```

### 2. Run Workload4 on Ray baseline

```bash
python experiments/distributed_workloads/run_workload4.py \
  --backend ray \
  --num-tasks 20 \
  --parallelism 4 \
  --seed 42 \
  --output-dir results/wl4_ray
```

> Ray is optional. Install with: `pip install -e .[ray-baseline]`

---

## Step-by-step: Paired comparison run

The `run_paired_backends.py` automation script launches both backends, archives
artifacts with a unique `run_id` + `config_hash`, and calls the comparison
report generator in one command:

```bash
python experiments/analysis/run_paired_backends.py \
  --scheduler fifo \
  --items 20 \
  --parallelism 4 \
  --nodes 1 \
  --seed 42
```

Artifact layout under `artifacts/paired_backend_runs/`:

```
artifacts/paired_backend_runs/
└── run_id=<uuid>/
    └── config_hash=<sha256[:12]>/
        ├── backends/
        │   ├── sage/
        │   │   ├── unified_results.jsonl
        │   │   └── unified_results.csv
        │   └── ray/
        │       ├── unified_results.jsonl
        │       └── unified_results.csv
        ├── comparison/
        │   ├── summary.md
        │   └── comparison.csv
        ├── logs/
        │   ├── sage.log
        │   ├── ray.log
        │   └── compare.log
        └── manifest.json
```

### Via GitHub Actions (remote trigger)

A workflow dispatch is available in `.github/workflows/paired-backend-run.yml`:

```bash
gh workflow run paired-backend-run.yml \
  --field nodes=1 \
  --field parallelism=4 \
  --field seed=42 \
  --field items=20 \
  --repo intellistream/sage-benchmark
```

Artifacts are uploaded as a GitHub Actions artifact for download.

---

## Reproducibility controls

For fair cross-backend comparisons the following controls **must** be kept
identical across both runs:

| Control | Flag | Details |
|---------|------|---------|
| RNG seed | `--seed` | Propagated to `random`, `numpy.random`, and all workload generators |
| Warmup split | `--warmup-items` | Items processed before the timed phase; not included in throughput calculation |
| Input parity | `--parity-batch-size` | Deterministic shuffle with fixed batch size via `deterministic_shuffle_v1` |
| Workload shape | `--num-tasks`, `--parallelism`, `--nodes` | Must be identical for both runs |

A `repro_manifest.json` is written next to every result file and records:

```json
{
  "seed": 42,
  "config_hash": "a3f1b2c4...",
  "warmup_items": 5,
  "parity_batch_size": 16,
  "deterministic_shuffle_version": "v1"
}
```

To verify two runs are comparable, check that their `config_hash` values match:

```bash
python -c "
import json
a = json.load(open('results/wl4_sage/repro_manifest.json'))
b = json.load(open('results/wl4_ray/repro_manifest.json'))
assert a['config_hash'] == b['config_hash'], 'Config mismatch!'
print('Configs match:', a['config_hash'])
"
```

---

## Understanding output artifacts

Every result file uses the **unified metrics schema** defined in
`experiments/common/metrics_schema.py`.  Required fields:

| Field | Type | Description |
|-------|------|-------------|
| `backend` | str | `"sage"` or `"ray"` |
| `workload` | str | Workload identifier (e.g. `"wl4"`) |
| `run_id` | str | UUID for this execution |
| `seed` | int | RNG seed used |
| `nodes` | int | Number of worker nodes |
| `parallelism` | int | Operator parallelism |
| `throughput` | float | Items processed per second |
| `latency_p50` | float \| null | Median latency (ms) |
| `latency_p95` | float \| null | 95th-percentile latency (ms) |
| `latency_p99` | float \| null | 99th-percentile latency (ms) |
| `success_rate` | float | Fraction of items completed successfully (0–1) |
| `duration_seconds` | float | Total wall-clock runtime |
| `timestamp` | str | ISO-8601 UTC timestamp |
| `config_hash` | str \| null | SHA-256 fingerprint of run configuration |
| `extra` | dict | Backend-specific additional fields |

The schema is enforced on write; missing required fields raise `ValueError`
immediately rather than silently omitting them.

---

## Generating a comparison report

If you already have result directories from separate runs:

```bash
python experiments/analysis/compare_backends.py \
  results/wl4_sage \
  results/wl4_ray \
  --output-dir artifacts/backend_comparison
```

Generated outputs:

| File | Description |
|------|-------------|
| `summary.md` | Markdown table: per-backend throughput, p95, p99, success_rate |
| `comparison.csv` | Merged normalised records for further analysis |
| `throughput_comparison.png` | Bar chart comparing throughput across backends |
| `latency_p95_p99_comparison.png` | Side-by-side p95/p99 latency chart |

The report also flags configuration mismatches (different seeds, parallelism, etc.)
so you know immediately if results are not directly comparable.

---

## Architecture: why Ray stays in sage-benchmark

> **This is a strict architectural boundary.  Do not move Ray code into SAGE core.**

```
SAGE core repo          sage-benchmark (this repo)
─────────────           ──────────────────────────
sage-kernel             experiments/backends/base.py   ← ABC (no Ray import)
sage-platform           experiments/backends/sage_runner.py
sageFlownet             experiments/backends/ray_runner.py  ← Ray isolated here
```

SAGE uses **sageFlownet** as its distributed execution runtime, replacing the
earlier Ray dependency.  To maintain a clean separation:

- `ray` is **never imported** inside `SAGE/`, `sageFlownet/`, or any other SAGE
  core repository.
- The Ray runner (`ray_runner.py`) lives exclusively in `sage-benchmark` as an
  optional, separately-installed extra (`pip install -e .[ray-baseline]`).
- Workload business logic (sources, operators, sinks) is **backend-agnostic** —
  it calls neither SAGE nor Ray APIs directly.  Only the runner adapts the spec
  to each runtime.

This design lets us publish fair performance comparisons without coupling SAGE's
architecture to a competing runtime.

### Layer diagram

```
WorkloadSpec (backend-agnostic input)
         │
         ├── SageRunner ──► FlownetEnvironment (sageFlownet runtime)
         │
         └── RayRunner  ──► ray.remote tasks   (Ray runtime, optional)
                │
         RunResult (backend-agnostic output → unified metrics schema)
```

---

## Adding a new backend

1. Create `experiments/backends/<name>_runner.py`.
2. Implement `WorkloadRunner`:

   ```python
   from experiments.backends.base import WorkloadRunner, WorkloadSpec, RunResult, register_runner

   @register_runner("my_backend")
   class MyRunner(WorkloadRunner):
       @property
       def backend_name(self) -> str:
           return "my_backend"

       def is_available(self) -> bool:
           """Return False if the backend's package is not installed."""
           try:
               import my_backend_package  # noqa: F401
               return True
           except ImportError:
               return False

       def run(self, spec: WorkloadSpec) -> RunResult:
           # translate spec → my_backend execution
           ...
           return RunResult(
               backend=self.backend_name,
               elapsed_seconds=elapsed,
               items_processed=count,
               metrics={},
           )
   ```

3. Add an optional dependency group in `pyproject.toml`:

   ```toml
   [project.optional-dependencies]
   my-backend = ["my-backend-package>=1.0"]
   ```

4. Add a unit test in `tests/test_backend_selector.py`.
5. Import the module at the workload entry-point before calling `get_runner()`.

---

## Troubleshooting

### `RuntimeError: Backend 'ray' is not available`

Ray is not installed.  Run:

```bash
pip install -e .[ray-baseline]
```

### `AssertionError: Config mismatch` in `repro_manifest.json` comparison

The two runs used different configurations (different `--seed`, `--parallelism`,
or `--num-tasks`).  Re-run both backends with **identical flags**.

### `ValueError: Missing required fields` in metrics writer

A custom runner is not populating all required schema fields.  Check
`experiments/common/metrics_schema.py:REQUIRED_FIELDS` for the full list.

### Ray run hangs on `ray.init()`

By default the Ray runner calls `ray.init(ignore_reinit_error=True)`.  If Ray
is already initialised with incompatible settings, call `ray.shutdown()` first
or run in a fresh process.

### SAGE run fails with `FlownetEnvironment` errors

Ensure SAGE is installed and the `sage` conda environment is active:

```bash
conda activate sage
python -c "from sage.kernel import FlownetEnvironment; print('OK')"
```

If SAGE is not installed, follow [SAGE quickstart](https://github.com/intellistream/SAGE#quickstart).
