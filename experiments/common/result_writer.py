"""Shared result writers for unified benchmark metrics records."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .metrics_schema import normalize_metrics_record


CSV_FIELD_ORDER: tuple[str, ...] = (
    "backend",
    "workload",
    "run_id",
    "seed",
    "nodes",
    "parallelism",
    "throughput",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "success_rate",
    "duration_seconds",
    "timestamp",
    "config_hash",
    "backend_hash",
)


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> Path:
    """Append one normalized metrics record to a JSONL file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_metrics_record(record)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    return target


def export_jsonl_to_csv(jsonl_path: str | Path, csv_path: str | Path) -> Path:
    """Export normalized JSONL records to CSV with stable column order."""
    source = Path(jsonl_path)
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            normalized = normalize_metrics_record(payload)
            row = {key: normalized.get(key) for key in CSV_FIELD_ORDER}
            rows.append(row)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELD_ORDER))
        writer.writeheader()
        writer.writerows(rows)

    return target
