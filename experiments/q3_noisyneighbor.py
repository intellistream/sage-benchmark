import asyncio
import time

from sage.benchmark.benchmark_sage.experiments.base_experiment import BaseExperiment
from sage.benchmark.benchmark_sage.experiments.common import BenchmarkClient, WorkloadGenerator


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
        duration = 30

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
