# Q1–Q8 single-node benchmark matrix

The Q1–Q8 catalog is a TPC-inspired workload matrix for SAGE system behavior.
It is not an implementation of TPC-H or TPC-C. Each workload writes the same
machine-readable metrics schema so that results can be compared and published
without workload-specific conversion.

## Run the matrix

Install the benchmark package and start the LLM and embedding endpoints required
by the selected workloads. Then run:

```bash
python __main__.py \
  --all \
  --backend sage \
  --nodes 1 \
  --parallelism 2 \
  --seed 42 \
  --repeat 1 \
  --output-dir results/q-matrix
```

Use `--quick` for a reduced smoke run. The default failure policy finishes the
remaining workloads and exits non-zero if any repetition failed. Use
`--fail-fast` to stop at the first failure. `matrix_status.json` records every
attempt, including partial runs.

Run one workload with the same contract:

```bash
python __main__.py \
  --experiment q3 \
  --backend sage \
  --nodes 1 \
  --parallelism 2 \
  --seed 42 \
  --quick \
  --output-dir results/q3-smoke
```

## Output layout

```text
results/q-matrix/
├── matrix_status.json
├── q1/
│   ├── config.json
│   ├── results.json
│   ├── unified_results.jsonl
│   └── unified_results.csv
└── ... q2 through q8
```

Q-matrix records use lower-case workload identifiers (`q1` through `q8`).
Every unified record includes the backend, run ID, seed, node count,
parallelism, configuration hash, metrics, experiment name, component versions,
model identifiers, and a system profile.

## Aggregate and publish

Create the publication bundle locally:

```bash
python scripts/aggregate_for_hf.py
```

This scans `results/**/unified_results.jsonl`, rejects malformed local records,
deduplicates records by benchmark configuration, and writes:

```text
hf_data/
├── benchmark_results.json
└── benchmark_summary.json
```

The summary contains record counts by workload, backend, and seed. Publication
to `intellistream/sage-benchmark-results` is performed by the protected
`upload-to-hf.yml` workflow after an `hf_data/**/*.json` change reaches
`main`. Do not commit tokens or place an HF token in a benchmark result.

## GitHub Actions

Open **Actions → Benchmark CI → Run workflow**. The workflow accepts:

- an experiment list (`all` or comma-separated Q IDs);
- backend;
- seed;
- single-node parallelism;
- quick or full scale.

It archives the partial status and result files even when a workload fails.
A green workflow is required before treating the matrix or HF publication path
as operational.

## Troubleshooting

- **Gateway never becomes healthy:** inspect the uploaded `sagellm` log. Confirm
  that the model can be downloaded from the configured Hugging Face endpoint
  and that the selected sagellm version supports the workflow's serve flags.
- **Embedding smoke test fails:** confirm the registered model and the model in
  the `/v1/embeddings` request are both
  `sentence-transformers/all-MiniLM-L6-v2`.
- **Matrix exits non-zero:** inspect `matrix_status.json` and the corresponding
  `q*/` directory. A partial artifact is diagnostic evidence, not a successful
  benchmark matrix.
- **Aggregation refuses a file:** repair the named malformed JSONL record. The
  publication path intentionally fails closed instead of silently dropping bad
  input.
- **Ray backend is unavailable:** install the documented Ray baseline extra
  before selecting `--backend ray`.
