"""CLI entry point for running SAGE system-level benchmark experiments.

This module runs benchmark workloads using a TPC-H/TPC-C-inspired Q1..Q8 catalog.

Usage examples:

    python -m sage.benchmark.benchmark_sage --experiment Q1
    python -m sage.benchmark.benchmark_sage --all
    python -m sage.benchmark.benchmark_sage --experiment Q2 --config config/q2.yaml
    python -m sage.benchmark.benchmark_sage --experiment Q1 --backend ray --nodes 4 --parallelism 8
    python -m sage.benchmark.benchmark_sage --experiment Q1 --repeat 3 --seed 0

"""

from __future__ import annotations

import argparse
import sys
import uuid
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
    # Import here so --help is fast even without heavy deps installed.
    from experiments.common.cli_args import (
        add_common_benchmark_args,
        build_run_config,
        validate_benchmark_args,
    )

    parser = argparse.ArgumentParser(
        description="SAGE system benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Run Q1 workload against the default SAGE backend
    python -m sage.benchmark.benchmark_sage --experiment Q1

    # Run all workloads (Q1..Q8 catalog)
    python -m sage.benchmark.benchmark_sage --all

    # Compare against Ray with 4 nodes, 8-way parallelism, 3 repetitions
    python -m sage.benchmark.benchmark_sage --experiment Q2 \\
        --backend ray --nodes 4 --parallelism 8 --repeat 3

    # Reproducible run with explicit seed
    python -m sage.benchmark.benchmark_sage --experiment Q1 --seed 0

    # Dry run (validate only)
    python -m sage.benchmark.benchmark_sage --experiment Q1 --dry-run
""",
    )

    # ── Workload selection (suite-level) ────────────────────────────────────
    selection_grp = parser.add_argument_group("workload selection")
    selection_grp.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Workload to run (Q1..Q8).",
    )
    selection_grp.add_argument(
        "--all", "-a", action="store_true", help="Run all workloads in catalog."
    )
    selection_grp.add_argument(
        "--config", "-c", type=str, help="Path to a custom config YAML file."
    )

    # ── Standardised benchmark flags (shared across all workloads) ──────────
    add_common_benchmark_args(parser, include_quick=True, include_dry_run=True)

    args = parser.parse_args()

    if not args.experiment and not args.all:
        parser.print_help()
        return 1

    validate_benchmark_args(args)

    # Import here to avoid slow startup for --help
    from config.config_loader import ConfigLoader
    from experiments.q1_pipelinechain import E2EPipelineExperiment
    from experiments.q2_controlmix import ControlPlaneExperiment
    from experiments.q3_noisyneighbor import IsolationExperiment
    from experiments.q4_scalefrontier import ScalabilityExperiment
    from experiments.q5_heteroresilience import (
        HeterogeneityExperiment,
    )
    from experiments.q6_bursttown import BurstTownExperiment
    from experiments.q7_reconfigdrill import (
        ReconfigDrillExperiment,
    )
    from experiments.q8_recoverysoak import RecoverySoakExperiment

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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {}

    for exp_q in experiments_to_run:
        run_cfg = build_run_config(args, workload=exp_q)
        print(f"\n{'=' * 60}")
        print(f"Running Workload {_workload_label(exp_q)}")
        print(
            f"  backend={run_cfg['backend']}  nodes={run_cfg['nodes']}  "
            f"parallelism={run_cfg['parallelism']}  repeat={run_cfg['repeat']}  "
            f"seed={run_cfg['seed']}"
        )
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

        # Propagate shared CLI args into experiment config
        if hasattr(config, "workload"):
            config.workload.seed = args.seed
            if args.parallelism > 1:
                config.hardware.cpu_nodes = args.nodes - 1 if args.nodes > 1 else 0

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

        for rep in range(1, args.repeat + 1):
            rep_label = f" (rep {rep}/{args.repeat})" if args.repeat > 1 else ""
            print(f"Starting{rep_label} …")
            rep_output = output_dir / exp_q.lower() / (f"rep{rep}" if args.repeat > 1 else "")
            experiment.output_dir = rep_output
            experiment.output_dir.mkdir(parents=True, exist_ok=True)
            experiment.backend = args.backend
            experiment.nodes = int(args.nodes)
            experiment.parallelism = int(args.parallelism)
            experiment.run_id = f"{exp_q.lower()}-{args.backend}-{rep}-{uuid.uuid4().hex[:8]}"
            try:
                experiment.setup()
                result = experiment.run()
                experiment.teardown()
                results.setdefault(exp_q, []).append(result)  # type: ignore[union-attr]
                print(
                    f"  Workload {_workload_label(exp_q)}{rep_label} completed. "
                    f"Results saved to {rep_output}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Error running workload {_workload_label(exp_q)}{rep_label}: {exc}")
                if args.verbose:
                    import traceback

                    traceback.print_exc()
                results.setdefault(exp_q, []).append({"error": str(exc)})  # type: ignore[union-attr]

    if not args.dry_run:
        print(f"\n{'=' * 60}")
        print("Experiment Summary")
        print(f"{'=' * 60}")
        for exp_q, reps in results.items():
            if isinstance(reps, list):
                errors = [r for r in reps if isinstance(r, dict) and "error" in r]
                if errors:
                    print(f"  {_workload_label(exp_q)}: {len(errors)}/{len(reps)} FAILED")
                else:
                    print(f"  {_workload_label(exp_q)}: COMPLETED ({len(reps)} rep(s))")
            elif isinstance(reps, dict) and "error" in reps:
                print(f"  {_workload_label(exp_q)}: FAILED - {reps['error']}")
            else:
                print(f"  {_workload_label(exp_q)}: COMPLETED")
        print(f"\nResults saved to: {output_dir.absolute()}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
