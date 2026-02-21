# Change Logs (Core Updates)

## 2026-02-21

- Removed temporary workload helper scripts to keep the repo minimal:
  - `experiments/distributed_workloads/test_run_workload4.sh`
  - `experiments/distributed_workloads/workload4_config_debug.yaml`
- Packaging policy simplified (already applied): keep only `dev` extra flag in `pyproject.toml`.
- CI install command aligned to no-extra default install in paired backend workflow.
