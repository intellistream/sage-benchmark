import argparse
import uuid

from sage.benchmark.benchmark_sage.experiments.common.cli_args import (
    add_common_benchmark_args,
    build_run_config,
    validate_benchmark_args,
)
from sage.benchmark.benchmark_sage.experiments.q3_noisyneighbor import IsolationExperiment


class RecoverySoakExperiment(IsolationExperiment):
    """
    Q8 RecoverySoak workload.

    Long-running interference and failure-recovery stress soak.
    """

    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Q8 – RecoverySoak: benchmark long-running interference and "
            "failure-recovery stress scenarios."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_benchmark_args(parser)
    parser.add_argument("--rate", type=float, default=10.0, metavar="REQ_S")
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Experiment duration in seconds (default: 30).",
    )
    parser.add_argument("--gateway", type=str, default="http://localhost:8888", metavar="URL")
    return parser


def main() -> None:  # pragma: no cover
    import sys
    from pathlib import Path

    from sage.benchmark.benchmark_sage.config.config_loader import ConfigLoader

    parser = _build_parser()
    args = parser.parse_args()
    validate_benchmark_args(args)
    run_cfg = build_run_config(
        args,
        workload="Q8",
        rate=args.rate,
        duration=args.duration,
        gateway=args.gateway,
    )

    if args.verbose:
        import json

        print("Run config:")
        print(json.dumps(run_cfg, indent=2))

    if args.dry_run:
        print("[dry-run] Q8 RecoverySoak – configuration validated. Exiting without running.")
        sys.exit(0)

    loader = ConfigLoader()
    config = loader.get_default_config("Q8")
    config.experiment_section = "Q8"
    if args.quick:
        config = loader.apply_quick_mode(config)

    config.workload.request_rate = args.rate
    config.workload.seed = args.seed
    config.gateway_url = args.gateway

    output_dir = Path(args.output_dir) / "q8"
    for rep in range(1, args.repeat + 1):
        rep_label = f" (repetition {rep}/{args.repeat})" if args.repeat > 1 else ""
        print(f"Running Q8 RecoverySoak{rep_label} …")
        rep_output = output_dir / (f"rep{rep}" if args.repeat > 1 else "")
        experiment = RecoverySoakExperiment(
            config=config, output_dir=rep_output, verbose=args.verbose
        )
        experiment.duration_seconds = args.duration
        experiment.backend = args.backend
        experiment.nodes = int(args.nodes)
        experiment.parallelism = int(args.parallelism)
        experiment.run_id = f"q8-{args.backend}-{rep}-{uuid.uuid4().hex[:8]}"
        experiment.setup()
        experiment.run()
        experiment.teardown()
        print(f"  Results saved to {rep_output.absolute()}")


if __name__ == "__main__":  # pragma: no cover
    main()
