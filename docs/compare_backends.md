# Cross-backend comparison report

Issue #6 delivers a unified report generator for SAGE vs Ray benchmark outputs.

## Command

```bash
cd /home/shuhao/sage-benchmark
python experiments/analysis/compare_backends.py \
  /path/to/sage/results \
  /path/to/ray/results \
  --output-dir artifacts/backend_comparison
```

## Supported inputs

- Directories (recursive scan)
- Files
- Artifact names:
  - `unified_results.csv`
  - `unified_results.jsonl`

## Generated artifacts

- `summary.md` (markdown summary with backend table + config mismatch section)
- `comparison.csv` (merged normalized records)
- `throughput_comparison.png`
- `latency_p95_p99_comparison.png`
