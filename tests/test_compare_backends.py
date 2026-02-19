"""Tests for backend comparison report generator (Issue #6)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from analysis.compare_backends import (
    detect_config_mismatches,
    discover_result_files,
    run_comparison,
)


def _write_unified_csv(path: Path, rows: list[dict[str, object]]) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_discover_result_files_finds_mixed_inputs(tmp_path: Path) -> None:
    sage_csv = tmp_path / "sage_run" / "unified_results.csv"
    ray_csv = tmp_path / "ray_run" / "nested" / "unified_results.csv"

    _write_unified_csv(sage_csv, [{"backend": "sage", "workload": "q1", "run_id": "r1"}])
    _write_unified_csv(ray_csv, [{"backend": "ray", "workload": "q1", "run_id": "r1"}])

    files = discover_result_files([tmp_path / "sage_run", tmp_path / "ray_run", sage_csv])

    assert sage_csv.resolve() in files
    assert ray_csv.resolve() in files


def test_detect_config_mismatches_flags_cross_backend_differences() -> None:
    df = pd.DataFrame(
        [
            {
                "backend": "sage",
                "workload": "q1",
                "run_id": "run-1",
                "seed": 42,
                "nodes": 2,
                "parallelism": 8,
                "config_hash": "a",
            },
            {
                "backend": "ray",
                "workload": "q1",
                "run_id": "run-1",
                "seed": 42,
                "nodes": 3,
                "parallelism": 8,
                "config_hash": "b",
            },
        ]
    )

    mismatches = detect_config_mismatches(df)

    assert len(mismatches) == 1
    assert mismatches.iloc[0]["workload"] == "q1"
    assert mismatches.iloc[0]["run_id"] == "run-1"


def test_run_comparison_generates_required_artifacts(tmp_path: Path) -> None:
    sage_csv = tmp_path / "results" / "sage" / "unified_results.csv"
    ray_csv = tmp_path / "results" / "ray" / "unified_results.csv"

    _write_unified_csv(
        sage_csv,
        [
            {
                "backend": "sage",
                "workload": "q2",
                "run_id": "same-run",
                "seed": 7,
                "nodes": 2,
                "parallelism": 8,
                "throughput": 120.5,
                "latency_p50": 18.0,
                "latency_p95": 40.0,
                "latency_p99": 66.0,
            }
        ],
    )
    _write_unified_csv(
        ray_csv,
        [
            {
                "backend": "ray",
                "workload": "q2",
                "run_id": "same-run",
                "seed": 7,
                "nodes": 3,
                "parallelism": 8,
                "throughput": 98.4,
                "latency_p50": 22.0,
                "latency_p95": 55.0,
                "latency_p99": 77.0,
            }
        ],
    )

    output_dir = tmp_path / "artifacts"
    artifacts = run_comparison([tmp_path / "results"], output_dir)

    assert artifacts["summary"].exists()
    assert artifacts["comparison_csv"].exists()
    assert artifacts["throughput_plot"].exists()
    assert artifacts["latency_plot"].exists()

    summary_text = artifacts["summary"].read_text(encoding="utf-8")
    assert "Configuration Mismatches" in summary_text
    assert "same-run" in summary_text
