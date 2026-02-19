"""Tests for unified cross-backend metrics schema and writers (Issue #9)."""

from __future__ import annotations

import json
from pathlib import Path

from common.metrics_schema import (
    REQUIRED_FIELDS,
    UnifiedMetricsRecord,
    compute_backend_hash,
    compute_config_hash,
    normalize_metrics_record,
)
from common.result_writer import CSV_FIELD_ORDER, append_jsonl_record, export_jsonl_to_csv


def test_config_hash_is_deterministic_for_same_dict_content():
    cfg_a = {"backend": "sage", "nodes": 2, "parallelism": 4}
    cfg_b = {"parallelism": 4, "backend": "sage", "nodes": 2}
    assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)


def test_backend_hash_is_deterministic():
    assert compute_backend_hash("sage") == compute_backend_hash("sage")
    assert compute_backend_hash("sage") != compute_backend_hash("ray")


def test_unified_record_contains_all_required_fields():
    record = UnifiedMetricsRecord(
        backend="sage",
        workload="Q1",
        run_id="run-1",
        seed=42,
        nodes=1,
        parallelism=2,
    ).to_dict()

    for key in REQUIRED_FIELDS:
        assert key in record

    assert record["latency_p50"] is None
    assert record["latency_p95"] is None
    assert record["latency_p99"] is None


def test_normalize_metrics_record_fills_missing_with_none():
    normalized = normalize_metrics_record({"backend": "ray", "workload": "Q2"})
    for key in REQUIRED_FIELDS:
        assert key in normalized
    assert normalized["run_id"] is None
    assert normalized["throughput"] is None


def test_jsonl_and_csv_writer_roundtrip(tmp_path: Path):
    jsonl_path = tmp_path / "unified_results.jsonl"
    csv_path = tmp_path / "unified_results.csv"

    record = UnifiedMetricsRecord(
        backend="sage",
        workload="scheduler_comparison",
        run_id="run-abc",
        seed=7,
        nodes=3,
        parallelism=8,
        throughput=99.5,
        duration_seconds=2.0,
    ).to_dict()

    append_jsonl_record(jsonl_path, record)
    export_jsonl_to_csv(jsonl_path, csv_path)

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["run_id"] == "run-abc"
    assert loaded["latency_p50"] is None

    csv_text = csv_path.read_text(encoding="utf-8")
    header = csv_text.splitlines()[0].split(",")
    assert header == list(CSV_FIELD_ORDER)
    assert "run-abc" in csv_text
