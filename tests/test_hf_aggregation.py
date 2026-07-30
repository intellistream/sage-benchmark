"""Tests for Q-matrix aggregation and publication guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_for_hf import (
    build_coverage_summary,
    get_config_key,
    load_local_results,
)


def _valid_record(workload: str = "Q1") -> dict:
    return {
        "backend": "sage",
        "workload": workload,
        "run_id": "run-1",
        "seed": 42,
        "nodes": 1,
        "parallelism": 2,
        "config_hash": "abc",
        "throughput": 1.0,
        "latency_p99": 2.0,
        "success_rate": 1.0,
    }


def test_loader_normalizes_q_workloads_and_coverage(tmp_path: Path) -> None:
    result_file = tmp_path / "q1" / "unified_results.jsonl"
    result_file.parent.mkdir()
    result_file.write_text(json.dumps(_valid_record()) + "\n", encoding="utf-8")

    records = load_local_results(tmp_path)

    assert records[0]["workload"] == "q1"
    assert build_coverage_summary(records)["by_workload"] == {"q1": 1}
    assert get_config_key(_valid_record("Q1")) == get_config_key(_valid_record("q1"))


def test_loader_fails_closed_on_malformed_local_record(tmp_path: Path) -> None:
    result_file = tmp_path / "q1" / "unified_results.jsonl"
    result_file.parent.mkdir()
    malformed = _valid_record()
    del malformed["config_hash"]
    result_file.write_text(json.dumps(malformed) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="拒绝聚合"):
        load_local_results(tmp_path)
