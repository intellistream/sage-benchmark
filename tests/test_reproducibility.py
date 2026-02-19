"""Tests for reproducibility helpers used by backend comparison flows."""

from __future__ import annotations

import random

from common.reproducibility import (
    build_input_parity_plan,
    compute_config_fingerprint,
    set_global_seed,
)


def test_set_global_seed_controls_python_random():
    set_global_seed(123)
    first = [random.random() for _ in range(3)]
    set_global_seed(123)
    second = [random.random() for _ in range(3)]
    assert first == second


def test_set_global_seed_controls_numpy_random_when_available():
    import numpy as np

    set_global_seed(456)
    a = np.random.rand(4).tolist()
    set_global_seed(456)
    b = np.random.rand(4).tolist()
    assert a == b


def test_config_fingerprint_is_order_invariant():
    a = {"seed": 42, "backend": "sage", "nodes": 2}
    b = {"nodes": 2, "backend": "sage", "seed": 42}
    assert compute_config_fingerprint(a) == compute_config_fingerprint(b)


def test_input_parity_plan_is_deterministic_and_split_correctly():
    plan_a = build_input_parity_plan(
        total_items=10,
        seed=7,
        warmup_count=3,
        batch_size=4,
    )
    plan_b = build_input_parity_plan(
        total_items=10,
        seed=7,
        warmup_count=3,
        batch_size=4,
    )

    assert plan_a.warmup_items == plan_b.warmup_items
    assert plan_a.benchmark_items == plan_b.benchmark_items
    assert len(plan_a.warmup_items) == 3
    assert len(plan_a.benchmark_items) == 7
    assert sum(len(batch) for batch in plan_a.benchmark_batches) == 7
