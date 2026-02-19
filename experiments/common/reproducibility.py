"""Reproducibility utilities for cross-backend benchmark runs.

This module centralizes deterministic controls used by benchmark entrypoints:
- global seed propagation (``random`` + ``numpy``)
- fixed warmup/benchmark split for input parity
- deterministic input batch generation
- canonical configuration fingerprint
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any


def set_global_seed(seed: int) -> None:
    """Propagate *seed* to supported RNG backends.

    Parameters
    ----------
    seed:
        Non-negative global seed.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        return


def compute_config_fingerprint(config: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 hash for *config*."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParityPlan:
    """Deterministic input parity plan shared by compared backends."""

    seed: int
    sampling_strategy: str
    warmup_items: list[str]
    benchmark_items: list[str]
    batch_size: int

    @property
    def benchmark_batches(self) -> list[list[str]]:
        """Return benchmark items split into fixed-size batches."""
        if not self.benchmark_items:
            return []
        return [
            self.benchmark_items[i : i + self.batch_size]
            for i in range(0, len(self.benchmark_items), self.batch_size)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize plan as a JSON-friendly dictionary."""
        return {
            "seed": self.seed,
            "sampling_strategy": self.sampling_strategy,
            "warmup_items": self.warmup_items,
            "benchmark_items": self.benchmark_items,
            "batch_size": self.batch_size,
            "benchmark_batches": self.benchmark_batches,
        }


def build_input_parity_plan(
    *,
    total_items: int,
    seed: int,
    warmup_count: int,
    batch_size: int,
    item_prefix: str = "item",
) -> ParityPlan:
    """Build a deterministic warmup/benchmark split with fixed batches.

    Strategy:
    1. Generate ``item_prefix_<index>`` tokens.
    2. Apply deterministic shuffle with ``seed``.
    3. Take first ``warmup_count`` as warmup set.
    4. Remaining items are benchmark set and split by ``batch_size``.
    """
    if total_items <= 0:
        raise ValueError("total_items must be positive")
    if warmup_count < 0:
        raise ValueError("warmup_count must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    pool = [f"{item_prefix}_{i}" for i in range(total_items)]
    rng = random.Random(seed)
    rng.shuffle(pool)

    bounded_warmup = min(warmup_count, total_items)
    warmup_items = pool[:bounded_warmup]
    benchmark_items = pool[bounded_warmup:]

    return ParityPlan(
        seed=seed,
        sampling_strategy="deterministic_shuffle_v1",
        warmup_items=warmup_items,
        benchmark_items=benchmark_items,
        batch_size=batch_size,
    )
