import argparse
import asyncio
import time
import uuid

from sage.benchmark.benchmark_sage.experiments.base_experiment import BaseExperiment
from sage.benchmark.benchmark_sage.experiments.common import BenchmarkClient, WorkloadGenerator
from sage.benchmark.benchmark_sage.experiments.common.cli_args import (
    add_common_benchmark_args,
    build_run_config,
    validate_benchmark_args,
)


class HeterogeneityExperiment(BaseExperiment):
    """
    Q5 HeteroResilience workload.

    Demonstrates heterogeneous hardware usage with mixed request classes.
    """

    def _setup_impl(self) -> None:
        self.workload_generator = WorkloadGenerator(
            llm_ratio=self.config.workload.llm_ratio, seed=self.config.workload.seed
        )

    def _run_impl(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        config = self.config
        workload = self.workload_generator

        requests_data = []
        for i in range(config.workload.total_requests):
            req_type, params = workload.generate_request(f"req-{i}")
            requests_data.append((req_type, params))

        arrival_times = workload.generate_arrival_times(
            config.workload.total_requests,
            config.workload.request_rate,
            config.workload.arrival_pattern,
        )

        async with BenchmarkClient(config.gateway_url) as client:
            tasks = []
            start_time = time.perf_counter()

            for i, (req_type, params) in enumerate(requests_data):
                target_time = start_time + arrival_times[i]
                now = time.perf_counter()
                if target_time > now:
                    await asyncio.sleep(target_time - now)

                req_id = f"req-{i}"
                if req_type == "llm":
                    task = asyncio.create_task(
                        client.send_llm_request(req_id, params["prompt"], config.llm_model.name)
                    )
                else:
                    task = asyncio.create_task(
                        client.send_embedding_request(
                            req_id, params["texts"], config.embedding_model.name
                        )
                    )
                tasks.append(task)

            self.results = await asyncio.gather(*tasks)

    def _teardown_impl(self) -> None:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Q5 – HeteroResilience: benchmark heterogeneous hardware behavior "
            "under mixed request classes."
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
        workload="Q5",
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
        print("[dry-run] Q5 HeteroResilience – configuration validated. Exiting without running.")
        sys.exit(0)

    loader = ConfigLoader()
    config = loader.get_default_config("Q5")
    config.experiment_section = "Q5"
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

    output_dir = Path(args.output_dir) / "q5"
    for rep in range(1, args.repeat + 1):
        rep_label = f" (repetition {rep}/{args.repeat})" if args.repeat > 1 else ""
        print(f"Running Q5 HeteroResilience{rep_label} …")
        rep_output = output_dir / (f"rep{rep}" if args.repeat > 1 else "")
        experiment = HeterogeneityExperiment(config=config, output_dir=rep_output, verbose=args.verbose)
        experiment.backend = args.backend
        experiment.nodes = int(args.nodes)
        experiment.parallelism = int(args.parallelism)
        experiment.run_id = f"q5-{args.backend}-{rep}-{uuid.uuid4().hex[:8]}"
        experiment.setup()
        experiment.run()
        experiment.teardown()
        print(f"  Results saved to {rep_output.absolute()}")


if __name__ == "__main__":  # pragma: no cover
    main()
