"""CLI entry point for running SAGE system-level benchmark experiments.

This module runs benchmark workloads using a TPC-H/TPC-C-inspired Q1..Q8 catalog.

Usage examples:

    python -m sage.benchmark.benchmark_sage --experiment Q1
    python -m sage.benchmark.benchmark_sage --all
    python -m sage.benchmark.benchmark_sage --experiment Q2 --config config/q2.yaml

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKLOAD_CATALOG = {
    "Q1": {
        "name": "PipelineChain",
        "entry": "e2e_pipeline",
        "family": "end-to-end RAG pipeline workloads",
    },
    "Q2": {
        "name": "ControlMix",
        "entry": "control_plane",
        "family": "mixed LLM+embedding scheduling workloads",
    },
    "Q3": {
        "name": "NoisyNeighbor",
        "entry": "isolation",
        "family": "multi-tenant interference and isolation workloads",
    },
    "Q4": {
        "name": "ScaleFrontier",
        "entry": "scalability",
        "family": "scale-out throughput/latency workloads",
    },
    "Q5": {
        "name": "HeteroResilience",
        "entry": "heterogeneity",
        "family": "heterogeneous deployment and recovery workloads",
    },
    "Q6": {
        "name": "BurstTown",
        "entry": "burst_priority",
        "family": "bursty mixed-priority transactional workloads",
    },
    "Q7": {
        "name": "ReconfigDrill",
        "entry": "reconfiguration",
        "family": "online reconfiguration drill workloads",
    },
    "Q8": {
        "name": "RecoverySoak",
        "entry": "recovery",
        "family": "fault-recovery soak workloads",
    },
}

VALID_EXPERIMENTS = tuple(WORKLOAD_CATALOG.keys())


def _workload_label(exp_q: str) -> str:
    meta = WORKLOAD_CATALOG[exp_q]
    return f"{exp_q} ({meta['name']}, {meta['entry']})"


def _normalize_experiment_id(exp_id: str) -> str:
    candidate = exp_id.upper()
    if candidate in VALID_EXPERIMENTS:
        return candidate

    valid = ", ".join(VALID_EXPERIMENTS)
    raise ValueError(f"Invalid experiment id: {exp_id}. Supported values: {valid}")


def _resolve_default_config_path(base_dir: Path, canonical_q: str) -> Path | None:
    q_path = base_dir / f"{canonical_q.lower()}.yaml"
    if q_path.exists():
        return q_path
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SAGE system benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run Q1 workload
    python -m sage.benchmark.benchmark_sage --experiment Q1

    # Run all workloads (Q1..Q8 catalog)
    python -m sage.benchmark.benchmark_sage --all

    # Run with custom config
    python -m sage.benchmark.benchmark_sage --experiment Q2 --config my_config.yaml

    # Dry run (validate only)
    python -m sage.benchmark.benchmark_sage --experiment Q1 --dry-run
        """,
    )

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Workload to run (Q1..Q8).",
    )
    parser.add_argument("--all", "-a", action="store_true", help="Run all workloads in catalog")
    parser.add_argument("--config", "-c", type=str, help="Path to custom config file (YAML)")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick version with reduced samples",
    )

    args = parser.parse_args()

    if not args.experiment and not args.all:
        parser.print_help()
        return 1

    # Import here to avoid slow startup for --help
    from sage.benchmark.benchmark_sage.config.config_loader import ConfigLoader
    from sage.benchmark.benchmark_sage.experiments.q1_pipelinechain import E2EPipelineExperiment
    from sage.benchmark.benchmark_sage.experiments.q2_controlmix import ControlPlaneExperiment
    from sage.benchmark.benchmark_sage.experiments.q3_noisyneighbor import IsolationExperiment
    from sage.benchmark.benchmark_sage.experiments.q4_scalefrontier import ScalabilityExperiment
    from sage.benchmark.benchmark_sage.experiments.q5_heteroresilience import (
        HeterogeneityExperiment,
    )
    from sage.benchmark.benchmark_sage.experiments.q6_bursttown import BurstTownExperiment
    from sage.benchmark.benchmark_sage.experiments.q7_reconfigdrill import (
        ReconfigDrillExperiment,
    )
    from sage.benchmark.benchmark_sage.experiments.q8_recoverysoak import RecoverySoakExperiment

    experiment_map = {
        "Q1": E2EPipelineExperiment,
        "Q2": ControlPlaneExperiment,
        "Q3": IsolationExperiment,
        "Q4": ScalabilityExperiment,
        "Q5": HeterogeneityExperiment,
        "Q6": BurstTownExperiment,
        "Q7": ReconfigDrillExperiment,
        "Q8": RecoverySoakExperiment,
    }

    if args.all:
        experiments_to_run = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    else:
        experiments_to_run = [_normalize_experiment_id(args.experiment)]

    config_loader = ConfigLoader()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}

    for exp_q in experiments_to_run:
        print(f"\n{'=' * 60}")
        print(f"Running Workload {_workload_label(exp_q)}")
        print(f"{'=' * 60}\n")

        if args.config:
            config = config_loader.load(args.config)
        else:
            config_dir = Path(__file__).parent / "config"
            default_config = _resolve_default_config_path(config_dir, exp_q)
            if default_config is not None:
                config = config_loader.load(str(default_config))
            else:
                config = config_loader.get_default_config(exp_q)

        # Keep result metadata aligned with Q-style workload ids.
        config.experiment_section = exp_q

        if args.quick:
            config = config_loader.apply_quick_mode(config)

        exp_class = experiment_map[exp_q]
        experiment = exp_class(
            config=config,
            output_dir=output_dir / exp_q.lower(),
            verbose=args.verbose,
        )

        if args.dry_run:
            print(f"[DRY RUN] Validating config for workload {_workload_label(exp_q)}...")
            experiment.validate()
            print("[DRY RUN] Config validation passed.")
            continue

        try:
            experiment.setup()
            result = experiment.run()
            experiment.teardown()
            results[exp_q] = result
            print(
                f"\nWorkload {_workload_label(exp_q)} completed. "
                f"Results saved to {experiment.output_dir}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error running workload {_workload_label(exp_q)}: {exc}")
            if args.verbose:
                import traceback

                traceback.print_exc()
            results[exp_q] = {"error": str(exc)}

    if not args.dry_run:
        print(f"\n{'=' * 60}")
        print("Experiment Summary")
        print(f"{'=' * 60}")
        for exp_q, result in results.items():
            if isinstance(result, dict) and "error" in result:
                print(f"  {_workload_label(exp_q)}: FAILED - {result['error']}")
            else:
                print(f"  {_workload_label(exp_q)}: COMPLETED")
        print(f"\nResults saved to: {output_dir.absolute()}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
