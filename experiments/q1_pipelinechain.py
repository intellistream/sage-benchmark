import argparse
import asyncio
import random
import time
import uuid

from sage.benchmark.benchmark_sage.experiments.base_experiment import (
    BaseExperiment,
)
from sage.benchmark.benchmark_sage.experiments.common import BenchmarkClient, RequestResult
from sage.benchmark.benchmark_sage.experiments.common.cli_args import (
    add_common_benchmark_args,
    build_run_config,
    validate_benchmark_args,
)


class E2EPipelineExperiment(BaseExperiment):
    """
    Q1 PipelineChain workload.

    Simulates a RAG pipeline workload (Embed -> Retrieve -> Generate) to evaluate
    system performance under complex, multi-stage traffic patterns.
    """

    def _setup_impl(self) -> None:
        """Setup experiment."""
        pass

    def _run_impl(self) -> None:
        """Run the experiment workload."""
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        """Run async workload."""
        config = self.config

        async with BenchmarkClient(config.gateway_url) as client:
            tasks = []
            start_time = time.perf_counter()

            arrival_times = self._generate_arrival_times(
                config.workload.total_requests, config.workload.request_rate
            )

            for i, arrival_time in enumerate(arrival_times):
                target_time = start_time + arrival_time
                now = time.perf_counter()
                if target_time > now:
                    await asyncio.sleep(target_time - now)

                task = asyncio.create_task(self._run_single_pipeline(client, f"pipe-{i}"))
                tasks.append(task)

            results_list = await asyncio.gather(*tasks)

            self.results = []
            for pipeline_results in results_list:
                self.results.extend(pipeline_results)

    async def _run_single_pipeline(
        self, client: BenchmarkClient, pipeline_id: str
    ) -> list[RequestResult]:
        """Run a single simulated RAG pipeline."""
        results = []

        emb_start = time.perf_counter()
        emb_res = await client.send_embedding_request(
            f"{pipeline_id}-step1-emb",
            ["Simulated user query for RAG pipeline"],
            self.config.embedding_model.name,
        )
        results.append(emb_res)

        if not emb_res.success:
            return results

        retrieve_delay = random.uniform(0.05, 0.2)
        await asyncio.sleep(retrieve_delay)

        prompt = "Context: ...retrieved data...\nQuery: Simulated query\nAnswer:"
        llm_res = await client.send_llm_request(
            f"{pipeline_id}-step3-llm", prompt, self.config.llm_model.name
        )
        results.append(llm_res)

        e2e_latency = (time.perf_counter() - emb_start) * 1000
        e2e_res = RequestResult(
            request_id=f"{pipeline_id}-e2e",
            request_type="pipeline_e2e",
            start_time=emb_start,
            end_time=time.perf_counter(),
            latency_ms=e2e_latency,
            success=llm_res.success,
            tokens_in=emb_res.tokens_in + llm_res.tokens_in,
            tokens_out=llm_res.tokens_out,
        )
        results.append(e2e_res)

        return results

    def _generate_arrival_times(self, n: int, rate: float) -> list[float]:
        import numpy as np

        intervals = np.random.exponential(1.0 / rate, n)
        return list(np.cumsum(intervals))

    def _run_warmup(self) -> None:
        pass

    def _teardown_impl(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Standalone entry point (Q1 / PipelineChain)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the Q1 PipelineChain workload."""
    parser = argparse.ArgumentParser(
        description=(
            "Q1 – PipelineChain: benchmark a multi-stage RAG pipeline "
            "(Embed → Retrieve → Generate) against different backends."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Default run against the SAGE backend
  python -m sage.benchmark.benchmark_sage.experiments.q1_pipelinechain

  # Compare against Ray with 4 parallel workers, 3 repetitions
  python -m sage.benchmark.benchmark_sage.experiments.q1_pipelinechain \\
      --backend ray --parallelism 4 --repeat 3

  # Smoke-test on CI
  python -m sage.benchmark.benchmark_sage.experiments.q1_pipelinechain --quick --dry-run
""",
    )
    # Standard flags shared across all workloads
    add_common_benchmark_args(parser)
    # Q1-specific flags
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        metavar="REQ_S",
        help="Request arrival rate in requests/second (default: 10.0).",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=100,
        metavar="N",
        help="Total number of pipeline requests to issue (default: 100).",
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
    """Standalone entry point for the Q1 PipelineChain workload."""
    import sys
    from pathlib import Path

    from sage.benchmark.benchmark_sage.experiments.config import ExperimentConfig, WorkloadConfig
    from sage.benchmark.benchmark_sage.config.config_loader import ConfigLoader

    parser = _build_parser()
    args = parser.parse_args()
    validate_benchmark_args(args)
    run_cfg = build_run_config(
        args,
        workload="Q1",
        rate=args.rate,
        total_requests=args.total_requests,
        gateway=args.gateway,
    )

    if args.verbose:
        import json
        print("Run config:")
        print(json.dumps(run_cfg, indent=2))

    if args.dry_run:
        print("[dry-run] Q1 PipelineChain – configuration validated. Exiting without running.")
        sys.exit(0)

    loader = ConfigLoader()
    config = loader.get_default_config("Q1")
    config.experiment_section = "Q1"
    if args.quick:
        config = loader.apply_quick_mode(config)

    # Override workload knobs from CLI
    config.workload = WorkloadConfig(
        total_requests=args.total_requests,
        request_rate=args.rate,
        seed=args.seed,
        llm_ratio=config.workload.llm_ratio,
    )
    config.gateway_url = args.gateway

    output_dir = Path(args.output_dir) / "q1"
    for rep in range(1, args.repeat + 1):
        rep_label = f" (repetition {rep}/{args.repeat})" if args.repeat > 1 else ""
        print(f"Running Q1 PipelineChain{rep_label} …")
        rep_output = output_dir / (f"rep{rep}" if args.repeat > 1 else "")
        experiment = E2EPipelineExperiment(
            config=config,
            output_dir=rep_output,
            verbose=args.verbose,
        )
        experiment.backend = args.backend
        experiment.nodes = int(args.nodes)
        experiment.parallelism = int(args.parallelism)
        experiment.run_id = f"q1-{args.backend}-{rep}-{uuid.uuid4().hex[:8]}"
        experiment.setup()
        experiment.run()
        experiment.teardown()
        print(f"  Results saved to {rep_output.absolute()}")


if __name__ == "__main__":  # pragma: no cover
    main()
