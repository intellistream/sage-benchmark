"""Shared argparse helpers for sage-benchmark entry points.

All benchmark workload scripts should use :func:`add_common_benchmark_args` to
register the canonical flag set, then call :func:`validate_benchmark_args` before
running to catch incompatible argument combinations early.

Standardised flags (Issue #2)
------------------------------
- ``--backend {sage,ray}``   – runtime backend (default: ``sage``)
- ``--nodes``                – number of worker nodes (distributed mode)
- ``--parallelism``          – operator parallelism hint
- ``--repeat``               – how many independent repetitions to run
- ``--seed``                 – global RNG seed for reproducibility
- ``--output-dir``           – root directory for result artefacts

Run configuration recording
----------------------------
:func:`build_run_config` serialises the parsed args into a plain ``dict`` that is
identical across backends and workloads, so downstream analysis can perform
fair comparison without re-parsing each file's header.

Usage example
-------------
.. code-block:: python

    import argparse
    from experiments.common.cli_args import (
        add_common_benchmark_args,
        validate_benchmark_args,
        build_run_config,
    )

    def main():
        parser = argparse.ArgumentParser(description="My workload")
        add_common_benchmark_args(parser)
        # … add workload-specific args …
        args = parser.parse_args()
        validate_benchmark_args(args)
        run_cfg = build_run_config(args)
        # run_cfg is a dict with standardised keys

"""

from __future__ import annotations

import argparse
import importlib.util
from typing import Any

# ---------------------------------------------------------------------------
# Canonical backend list
# ---------------------------------------------------------------------------

SUPPORTED_BACKENDS: tuple[str, ...] = ("sage", "ray")
DEFAULT_BACKEND: str = "sage"

RAY_BASELINE_INSTALL_CMD = "python -m pip install -e .[ray-baseline]"


def _module_available(module_name: str) -> bool:
    """Return True when *module_name* can be imported by Python."""
    return importlib.util.find_spec(module_name) is not None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def add_common_benchmark_args(
    parser: argparse.ArgumentParser,
    *,
    include_quick: bool = True,
    include_dry_run: bool = True,
) -> argparse.ArgumentParser:
    """Add the standardised benchmark flag set to *parser*.

    Call this once per workload's ``ArgumentParser`` **before** adding any
    workload-specific flags.  This guarantees that ``--help`` output is
    consistent across all migrated workloads.

    Parameters
    ----------
    parser:
        The :class:`argparse.ArgumentParser` to mutate in-place.
    include_quick:
        Whether to add the ``--quick`` shortcut flag (default: ``True``).
    include_dry_run:
        Whether to add the ``--dry-run`` flag (default: ``True``).

    Returns
    -------
    argparse.ArgumentParser
        The same *parser* object (for method-chaining if desired).
    """
    grp = parser.add_argument_group(
        "common benchmark arguments",
        description=(
            "Flags shared across all benchmark workloads. "
            "Using these consistently enables fair backend comparisons."
        ),
    )

    # ── Backend ─────────────────────────────────────────────────────────────
    grp.add_argument(
        "--backend",
        "-b",
        type=str,
        default=DEFAULT_BACKEND,
        choices=SUPPORTED_BACKENDS,
        metavar="BACKEND",
        help=(
            f"Runtime backend to target. Choices: {', '.join(SUPPORTED_BACKENDS)}. "
            f"(default: {DEFAULT_BACKEND})"
        ),
    )

    # ── Cluster / resource ─────────────────────────────────────────────────
    grp.add_argument(
        "--nodes",
        type=int,
        default=1,
        metavar="N",
        help="Number of worker nodes for distributed execution (default: 1).",
    )
    grp.add_argument(
        "--parallelism",
        type=int,
        default=2,
        metavar="P",
        help="Operator parallelism hint (default: 2; backend may override).",
    )

    # ── Repeatability ───────────────────────────────────────────────────────
    grp.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="R",
        help="Independent repetitions to run and average over (default: 1).",
    )
    grp.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Global RNG seed for reproducibility (default: 42).",
    )

    # ── Output ──────────────────────────────────────────────────────────────
    grp.add_argument(
        "--output-dir",
        type=str,
        default="results",
        metavar="DIR",
        help="Root directory for result artefacts (default: results).",
    )

    # ── Modifiers ───────────────────────────────────────────────────────────
    grp.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose / debug output.",
    )

    if include_quick:
        grp.add_argument(
            "--quick",
            action="store_true",
            default=False,
            help="Run a reduced-scale version suitable for smoke-testing.",
        )

    if include_dry_run:
        grp.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Validate configuration without executing any workload.",
        )

    return parser


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_benchmark_args(args: argparse.Namespace) -> None:
    """Validate *args* for incompatible combinations.

    Raises a human-readable :class:`SystemExit` so callers do not need to
    wrap this call; argparse's own error mechanism is used for consistency.

    Checks performed
    ----------------
    * ``--nodes > 1`` requires a backend that supports distributed execution;
      using ``--backend ray`` with ``--nodes 1`` is silently accepted.
    * ``--repeat`` must be ≥ 1.
    * ``--parallelism`` must be ≥ 1.
    * ``--seed`` must be ≥ 0.

    Parameters
    ----------
    args:
        The :class:`argparse.Namespace` produced by ``parser.parse_args()``.

    Raises
    ------
    SystemExit
        With a non-zero code and an actionable error message when validation
        fails.
    """
    errors: list[str] = []

    # Nodes / backend parity check
    nodes: int = getattr(args, "nodes", 1)
    backend: str = getattr(args, "backend", DEFAULT_BACKEND)
    if nodes > 1 and backend not in ("sage", "ray"):
        errors.append(
            f"--nodes {nodes} requires a distributed-capable backend "
            f"(sage or ray); got '{backend}'."
        )

    # Non-negativity / positivity checks
    repeat: int = getattr(args, "repeat", 1)
    if repeat < 1:
        errors.append(f"--repeat must be ≥ 1; got {repeat}.")

    parallelism: int = getattr(args, "parallelism", 2)
    if parallelism < 1:
        errors.append(f"--parallelism must be ≥ 1; got {parallelism}.")

    seed: int = getattr(args, "seed", 42)
    if seed < 0:
        errors.append(f"--seed must be ≥ 0; got {seed}.")

    # Optional backend dependency guard (fail fast with actionable guidance)
    if backend == "ray" and not _module_available("ray"):
        errors.append(
            "Ray backend selected but 'ray' is not installed. "
            "From the repository root run: "
            f"{RAY_BASELINE_INSTALL_CMD}"
        )

    if errors:
        # Use argparse-style messaging so users get a consistent look
        _fake_parser = argparse.ArgumentParser()
        _fake_parser.error("argument validation failed:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# Run config serialisation
# ---------------------------------------------------------------------------


def build_run_config(args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    """Build a standardised run-configuration dict from *args*.

    The returned dict has identical keys regardless of the workload or backend,
    making it easy to load and compare result JSON files programmatically.

    Parameters
    ----------
    args:
        Parsed :class:`argparse.Namespace` containing (at minimum) the flags
        added by :func:`add_common_benchmark_args`.
    **extra:
        Workload-specific key/value pairs to merge into the config record.

    Returns
    -------
    dict[str, Any]
        A serialisable run configuration with at least the standardised keys.
    """
    cfg: dict[str, Any] = {
        # ── Standardised keys ─────────────────────────────────────────────
        "backend": getattr(args, "backend", DEFAULT_BACKEND),
        "nodes": getattr(args, "nodes", 1),
        "parallelism": getattr(args, "parallelism", 2),
        "repeat": getattr(args, "repeat", 1),
        "seed": getattr(args, "seed", 42),
        "output_dir": getattr(args, "output_dir", "results"),
        # ── Modifiers ─────────────────────────────────────────────────────
        "quick": getattr(args, "quick", False),
        "dry_run": getattr(args, "dry_run", False),
        "verbose": getattr(args, "verbose", False),
    }
    cfg.update(extra)
    return cfg
