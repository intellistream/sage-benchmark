"""Unified cross-backend benchmark metrics schema.

This module defines a canonical record format for SAGE vs Ray benchmark
comparisons. All records include the same key set; missing metrics are stored
as ``None`` (serialized as ``null`` in JSON) rather than omitted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = (
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
)


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp string."""
    return datetime.now(UTC).isoformat()


def compute_config_hash(config: dict[str, Any]) -> str:
    """Compute deterministic hash of config dict (stable key order)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_backend_hash(backend: str) -> str:
    """Compute deterministic hash for backend label."""
    return hashlib.sha256(backend.encode("utf-8")).hexdigest()


@dataclass
class UnifiedMetricsRecord:
    """Canonical cross-backend metrics record.

    Required fields are always present in :meth:`to_dict`.
    """

    backend: str
    workload: str
    run_id: str
    seed: int
    nodes: int
    parallelism: int
    throughput: float | None = None
    latency_p50: float | None = None
    latency_p95: float | None = None
    latency_p99: float | None = None
    success_rate: float | None = None
    duration_seconds: float | None = None
    timestamp: str = field(default_factory=utc_timestamp)
    config_hash: str | None = None
    backend_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a JSON-serialisable dict with fixed key set."""
        payload: dict[str, Any] = {
            "backend": self.backend,
            "workload": self.workload,
            "run_id": self.run_id,
            "seed": int(self.seed),
            "nodes": int(self.nodes),
            "parallelism": int(self.parallelism),
            "throughput": self.throughput,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "latency_p99": self.latency_p99,
            "success_rate": self.success_rate,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "backend_hash": self.backend_hash,
            "metadata": self.metadata,
        }
        return payload


def normalize_metrics_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize an input record to the unified schema key set.

    Missing required fields are set to ``None``.
    """
    normalized = {field: record.get(field, None) for field in REQUIRED_FIELDS}
    normalized["config_hash"] = record.get("config_hash")
    normalized["backend_hash"] = record.get("backend_hash")
    normalized["metadata"] = record.get("metadata", {})
    return normalized
