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


class IsolationExperiment(BaseExperiment):
    """
    Q3 NoisyNeighbor workload.

    Simulates multi-tenant isolation pressure with interactive and batch tenants.
    """

    def _setup_impl(self) -> None:
        self.interactive_gen = WorkloadGenerator(llm_ratio=1.0, seed=42)
        self.batch_gen = WorkloadGenerator(llm_ratio=0.5, seed=99)

    def _run_impl(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        config = self.config

        interactive_rate = 5.0
        batch_rate = config.workload.request_rate
        duration = getattr(self, "duration_seconds", 30)

        async with BenchmarkClient(config.gateway_url) as client:
            tasks = []

            tasks.append(
                asyncio.create_task(
                    self._run_user_loop(
                        client, "interactive", interactive_rate, duration, self.interactive_gen
                    )
                )
            )

            tasks.append(
                asyncio.create_task(
                    self._run_user_loop(client, "batch", batch_rate, duration, self.batch_gen)
                )
            )

            results_list = await asyncio.gather(*tasks)

            self.results = []
            for user_results in results_list:
                self.results.extend(user_results)

    async def _run_user_loop(self, client, user_id, rate, duration, generator):
        results = []
        start_time = time.perf_counter()
        req_idx = 0

        while time.perf_counter() - start_time < duration:
            await asyncio.sleep(1.0 / rate)

            req_type, params = generator.generate_request(f"{user_id}-{req_idx}")

            if req_type == "llm":
                res = await client.send_llm_request(
                    f"{user_id}-{req_idx}", params["prompt"], self.config.llm_model.name
                )
            else:
                res = await client.send_embedding_request(
                    f"{user_id}-{req_idx}", params["texts"], self.config.embedding_model.name
                )

            res.metadata["user_id"] = user_id
            results.append(res)
            req_idx += 1

        return results

    def _teardown_impl(self) -> None:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Q3 – NoisyNeighbor: benchmark multi-tenant isolation pressure "
            "(interactive + batch tenants) against different backends."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_benchmark_args(parser)
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        metavar="REQ_S",
        help="Batch tenant request rate in requests/second (default: 10.0).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Experiment duration in seconds (default: 30).",
    )
    parser.add_argument(
        "--gateway",
        type=str,
        default="http://localhost:8888",
        metavar="URL",
        help="Gateway URL used when backend=sage (default: http://localhost:8888).",
    )
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
        workload="Q3",
        rate=args.rate,
        duration=args.duration,
        gateway=args.gateway,
    )

    if args.verbose:
        import json

        print("Run config:")
        print(json.dumps(run_cfg, indent=2))

    if args.dry_run:
        print("[dry-run] Q3 NoisyNeighbor – configuration validated. Exiting without running.")
        sys.exit(0)

    loader = ConfigLoader()
    config = loader.get_default_config("Q3")
    config.experiment_section = "Q3"
    if args.quick:
        config = loader.apply_quick_mode(config)

    config.workload.request_rate = args.rate
    config.workload.seed = args.seed
    config.gateway_url = args.gateway

    output_dir = Path(args.output_dir) / "q3"
    for rep in range(1, args.repeat + 1):
        rep_label = f" (repetition {rep}/{args.repeat})" if args.repeat > 1 else ""
        print(f"Running Q3 NoisyNeighbor{rep_label} …")
        rep_output = output_dir / (f"rep{rep}" if args.repeat > 1 else "")
        experiment = IsolationExperiment(config=config, output_dir=rep_output, verbose=args.verbose)
        experiment.duration_seconds = args.duration
        experiment.backend = args.backend
        experiment.nodes = int(args.nodes)
        experiment.parallelism = int(args.parallelism)
        experiment.run_id = f"q3-{args.backend}-{rep}-{uuid.uuid4().hex[:8]}"
        experiment.setup()
        experiment.run()
        experiment.teardown()
        print(f"  Results saved to {rep_output.absolute()}")


if __name__ == "__main__":  # pragma: no cover
    main()
