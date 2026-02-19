import argparse
import uuid

from sage.benchmark.benchmark_sage.experiments.common.cli_args import (
    add_common_benchmark_args,
    build_run_config,
    validate_benchmark_args,
)
from sage.benchmark.benchmark_sage.experiments.q5_heteroresilience import HeterogeneityExperiment


class ReconfigDrillExperiment(HeterogeneityExperiment):
    """
    Q7 ReconfigDrill workload.

    Repeated online reconfiguration drill over heterogeneous deployment settings.
    """

    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Q7 – ReconfigDrill: benchmark online reconfiguration drills under "
            "heterogeneous deployment settings."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_benchmark_args(parser)
    parser.add_argument("--rate", type=float, default=10.0, metavar="REQ_S")
    parser.add_argument("--total-requests", type=int, default=100, metavar="N")
    parser.add_argument("--llm-ratio", type=float, default=0.7, metavar="RATIO")
    parser.add_argument("--gateway", type=str, default="http://localhost:8888", metavar="URL")
    return parser


def main() -> None:  # pragma: no cover
    import sys
    from pathlib import Path

    from sage.benchmark.benchmark_sage.config.config_loader import ConfigLoader
    from sage.benchmark.benchmark_sage.experiments.config import WorkloadConfig

    parser = _build_parser()
    args = parser.parse_args()
    validate_benchmark_args(args)
    run_cfg = build_run_config(
        args,
        workload="Q7",
        rate=args.rate,
        total_requests=args.total_requests,
        llm_ratio=args.llm_ratio,
        gateway=args.gateway,
    )

    if args.verbose:
        import json

        print("Run config:")
        print(json.dumps(run_cfg, indent=2))

    if args.dry_run:
        print("[dry-run] Q7 ReconfigDrill – configuration validated. Exiting without running.")
        sys.exit(0)

    loader = ConfigLoader()
    config = loader.get_default_config("Q7")
    config.experiment_section = "Q7"
    if args.quick:
        config = loader.apply_quick_mode(config)

    config.workload = WorkloadConfig(
        total_requests=args.total_requests,
        request_rate=args.rate,
        seed=args.seed,
        llm_ratio=args.llm_ratio,
        warmup_requests=config.workload.warmup_requests,
        input_tokens_min=config.workload.input_tokens_min,
        input_tokens_max=config.workload.input_tokens_max,
        output_tokens_min=config.workload.output_tokens_min,
        output_tokens_max=config.workload.output_tokens_max,
        arrival_pattern=config.workload.arrival_pattern,
    )
    config.gateway_url = args.gateway

    output_dir = Path(args.output_dir) / "q7"
    for rep in range(1, args.repeat + 1):
        rep_label = f" (repetition {rep}/{args.repeat})" if args.repeat > 1 else ""
        print(f"Running Q7 ReconfigDrill{rep_label} …")
        rep_output = output_dir / (f"rep{rep}" if args.repeat > 1 else "")
        experiment = ReconfigDrillExperiment(config=config, output_dir=rep_output, verbose=args.verbose)
        experiment.backend = args.backend
        experiment.nodes = int(args.nodes)
        experiment.parallelism = int(args.parallelism)
        experiment.run_id = f"q7-{args.backend}-{rep}-{uuid.uuid4().hex[:8]}"
        experiment.setup()
        experiment.run()
        experiment.teardown()
        print(f"  Results saved to {rep_output.absolute()}")


if __name__ == "__main__":  # pragma: no cover
    main()
