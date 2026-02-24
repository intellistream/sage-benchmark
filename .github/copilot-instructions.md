# SAGE Benchmark Copilot Instructions

## Scope
- `isage-benchmark` is an L5 benchmarking suite across SAGE subsystems.
- Benchmark modules live under `src/sage/benchmark/...` with pytest tests.

## Critical rules
- Keep dependency direction: benchmark code may depend on L1-L4, never upward.
- Do not create new local virtual environments (`venv`/`.venv`); use the existing configured Python environment.
- Use typed APIs, clear docstrings, and config-driven experiments.
- Follow prepare/run/finalize experiment lifecycle.
- Respect submodule boundaries (`benchmark_amm`, `benchmark_anns`, `sage.data`): treat as external repos.
- Prefer early validation and explicit errors over fallback behavior.

## Workflow
1. Implement minimal changes in target benchmark module.
2. Update config schemas/docs when adding metrics or settings.
3. Add tests mirroring `src/` structure in `tests/`.

## Key paths
- `src/sage/benchmark/`, `tests/`, module `config/` and `evaluation/` folders.
