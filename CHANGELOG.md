# CHANGELOG

All notable changes to this repository are documented in this file.

## PyPI Verified Releases (`isage-benchmark`)

Source: `https://pypi.org/pypi/isage-benchmark/json` (checked on 2026-02-14, UTC).

- `0.2.4` — 2026-01-03T18:17:17Z
- `0.2.3` — 2026-01-03T16:55:53Z

## [Unreleased]

### Changed
- Updated distributed workload documentation links away from `dev-notes` references.
- Consolidated benchmark documentation retention policy: aggressively trim non-critical markdown while preserving `README*.md`, agent metadata (`*.agent.md`), and Copilot instruction files.
- Preserved high-priority benchmark context in this changelog: experiment focus remains on agent, RAG, control-plane, memory, refiner, scheduler, and system-level benchmarking modules.
- Folded key guidance from removed docs into this changelog:
	- Backend comparison baseline remains `sage` (default) vs optional `ray` baseline backend, with reproducibility driven by consistent `seed`/workload shape and `config_hash` checks.
	- Workload design intent remains four progressive distributed CPU-intensive profiles (W1-W4), emphasizing `KeyBy`/`Join`/`Batch` scaling and reduced dependence on large-model generation.
	- Reranking guidance remains three-tiered: hosted reranker service (accuracy-first), real-embedding + CPU scoring (balanced), deterministic pseudo-random vectors (pure CPU stress path).
	- Workload4 quickstart essentials remain dual-stream configuration (`query_qps` + `doc_qps`) with class/YAML/factory config paths and explicit source distribution settings.
	- Tool-use agent architecture remains ReAct-style (`Thought -> Action -> Observation -> Reflection`) with service-backed memory/context/vector retrieval wiring.

### Removed
- Removed `MIGRATION_TO_INDEPENDENT_REPO.md` from this repository.
- Removed selected non-critical markdown docs under `docs/` and `experiments/` to reduce maintenance surface.
- Removed duplicate root changelog file `changelogs.md`.
- Removed deep-dive docs while retaining their core operational constraints in this changelog:
	- `docs/BACKEND_COMPARISON_GUIDE.md`
	- `docs/WORKLOAD_DESIGNS.md`
	- `experiments/common/RERANKER_SERVICE_GUIDE.md`
	- `experiments/distributed_workloads/workload4/QUICK_START.md`
	- `experiments/tool_use_agent/DESIGN.md`
