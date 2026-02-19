# Cross-backend comparison report (`compare_backends.py`)

> For the full end-to-end walkthrough see [BACKEND_COMPARISON_GUIDE.md](BACKEND_COMPARISON_GUIDE.md).

Issue #6 delivers a unified report generator for SAGE vs Ray benchmark outputs.

## Command

```bash
python experiments/analysis/compare_backends.py \
  /path/to/sage/results \
  /path/to/ray/results \
  --output-dir artifacts/backend_comparison
```

Arguments:

| Argument | Required | Description |
|----------|----------|-------------|
| `sage_dir` | Yes | Directory (or file) with SAGE backend results |
| `ray_dir` | Yes | Directory (or file) with Ray backend results |
| `--output-dir DIR` | No (default: `artifacts/backend_comparison`) | Output directory for generated files |
| `--verbose` / `-v` | No | Print per-record debug output |

## Supported input formats

- **Directories**: recursively scanned for `unified_results.jsonl` and
  `unified_results.csv` files
- **Files**: read directly

All input records must conform to the unified metrics schema
(`experiments/common/metrics_schema.py`).  Records with unknown backends are
skipped with a warning.

## Generated artifacts

| File | Description |
|------|-------------|
| `summary.md` | Per-backend table (throughput, p95, p99, success_rate) + config mismatch warnings |
| `comparison.csv` | Merged, normalised records suitable for further analysis |
| `throughput_comparison.png` | Bar chart: throughput by backend |
| `latency_p95_p99_comparison.png` | Side-by-side p95/p99 latency bar chart |

## Config mismatch detection

The report compares `config_hash` values across backends and prints a warning
section in `summary.md` when they differ.  A mismatch means the two runs used
different seeds, parallelism, or workload parameters and results may not be
directly comparable.

## Metrics schema fields used

See `experiments/common/metrics_schema.py` for the full list.  Fields used in
the comparison report:

- `backend`, `workload`, `run_id`, `seed`, `nodes`, `parallelism`
- `throughput`, `latency_p50`, `latency_p95`, `latency_p99`
- `success_rate`, `duration_seconds`
- `config_hash` (for mismatch detection)

## Example output (`summary.md`)

```markdown
# Backend Comparison Summary

## Results

| backend | workload | throughput | latency_p95 | latency_p99 | success_rate |
|---------|----------|------------|-------------|-------------|---------------|
| sage    | wl4      | 182.4      | 12.3        | 18.7        | 1.00          |
| ray     | wl4      | 141.1      | 19.8        | 27.4        | 0.98          |

## Configuration

All runs used matching config_hash: `a3f1b2c4d5e6`
```
