# benchmark_sage – SAGE System-Level Benchmarks and ICML Artifacts

`benchmark_sage` is a home for **system-level benchmarks and artifacts** that focus on SAGE as a
complete ML systems platform.

Key points:

- SAGE is **more than an LLM control plane**. The LLM/embedding control plane is one subsystem. SAGE
  also includes components such as `sage.db`, `sage.flow`, `sage.tsdb`, and others, all orchestrated
  via a common **declarative dataflow model**.
- `packages/sage-benchmark` already contains multiple benchmark suites (agents, control-plane
  scheduling, DB, retrieval, memory, schedulers, refiner, libamm, etc.). `benchmark_sage` can
  aggregate **cross-cutting experiments** that involve several SAGE subsystems together.
- This folder may also store **ICML writing prompts and experiment templates** for the SAGE system
  track papers, under `docs/`.

Suggested uses:

- End-to-end experiments that span `sage.flow` pipelines, `sage.db` storage, `sage.tsdb` time-series
  monitoring, and the LLM/embedding control plane.
- Configs (`config/*.yaml`) for system-track experiments described in an ICML paper.
- Notebook or script entry points that reproduce figures/tables.

## Q-style Workload Catalog (TPC-H/TPC-C inspired)

`benchmark_sage` adopts a fixed `Q1..Q8` catalog where each Q denotes a workload family
rather than a one-off script. This keeps paper claims, configs, and run outputs aligned.

| Query | Name | Entry | Workload Family |
|---|---|---|---|
| Q1 | PipelineChain | `e2e_pipeline` | End-to-end RAG pipeline workloads |
| Q2 | ControlMix | `control_plane` | Mixed LLM+embedding scheduling workloads |
| Q3 | NoisyNeighbor | `isolation` | Multi-tenant interference and isolation workloads |
| Q4 | ScaleFrontier | `scalability` | Scale-out throughput/latency workloads |
| Q5 | HeteroResilience | `heterogeneity` | Heterogeneous deployment and recovery workloads |
| Q6 | BurstTown | `burst_priority` | Bursty mixed-priority transactional workloads |
| Q7 | ReconfigDrill | `reconfiguration` | Online reconfiguration drill workloads |
| Q8 | RecoverySoak | `recovery` | Fault-recovery soak workloads |

Examples:

```bash
# Run a single workload against the default SAGE backend
python -m sage.benchmark.benchmark_sage --experiment Q1

# Run all workloads
python -m sage.benchmark.benchmark_sage --all

# Quick smoke-test
python -m sage.benchmark.benchmark_sage --experiment Q3 --quick
python -m sage.benchmark.benchmark_sage --experiment Q7 --quick

# Backend comparison: same workload, two backends, for fair comparison
python -m sage.benchmark.benchmark_sage --experiment Q1 --backend sage --repeat 3 --seed 42
python -m sage.benchmark.benchmark_sage --experiment Q1 --backend ray  --repeat 3 --seed 42

# Distributed run: 4 nodes, 8-way operator parallelism
python -m sage.benchmark.benchmark_sage --experiment Q4 \
    --backend sage --nodes 4 --parallelism 8 --output-dir results/q4_scale

# Validate config without running
python -m sage.benchmark.benchmark_sage --experiment Q2 --dry-run
```

### Standardised CLI flags (Issue #2)

All workload entry points share the same flag contract so backend comparison runs always
produce comparable `run_config` records.

| Flag | Default | Description |
|------|---------|-------------|
| `--backend {sage,ray}` | `sage` | Runtime backend |
| `--nodes N` | `1` | Worker nodes for distributed execution |
| `--parallelism P` | `2` | Operator parallelism hint |
| `--repeat R` | `1` | Independent repetitions (averaged in results) |
| `--seed SEED` | `42` | Global RNG seed for reproducibility |
| `--output-dir DIR` | `results` | Root directory for artefacts |
| `--quick` | off | Reduced-scale smoke-test run |
| `--dry-run` | off | Validate config, skip execution |
| `--verbose` / `-v` | off | Enable debug output |

Individual workloads may add extra flags on top of the shared contract.

At the repo root, `docs/icml-prompts/` contains reusable writing prompts. You can either reference
them directly or copy customized versions into this folder when preparing a specific ICML
submission.
