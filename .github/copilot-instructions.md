# SAGE Benchmark Copilot Instructions

## Scope
- `isage-benchmark` is an independent benchmarking repository that sits above the core SAGE workspace layers and evaluates SAGE subsystems end-to-end.
- Benchmark modules live under `src/sage/benchmark/...` with pytest tests.

## Critical rules
- Keep dependency direction explicit: benchmark code may depend on the consolidated core stack (`sage.foundation`/L1, `sage.runtime`+`sage.stream`/L2, `sage.cli`/L3, `sage-studio`/L4) as needed, but core packages must never depend on benchmark code.
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

## Polyrepo coordination (mandatory)

- This repository is an independent SAGE sub-repository and is developed/released independently.
- Do not assume sibling source directories exist locally in `intellistream/SAGE`.
- For cross-repo rollout, publish this repo/package first, then bump the version pin in `SAGE/packages/sage/pyproject.toml` when applicable.
- Do not add local editable installs of other SAGE sub-packages in setup scripts or docs.

## 🚫 NEVER_CREATE_DOT_VENV_MANDATORY

- 永远不要创建 `.venv` 或 `venv`（无任何例外）。
- NEVER create `.venv`/`venv` in this repository under any circumstance.
- 必须复用当前已配置的非-venv Python 环境（如现有 conda 环境）。
- If any script/task suggests creating a virtualenv, skip that step and continue with the existing environment.
