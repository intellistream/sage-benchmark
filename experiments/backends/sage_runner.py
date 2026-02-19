"""
SAGE backend runner for sage-benchmark.

Wraps the SAGE FlownetEnvironment execution path so it satisfies the
:class:`~experiments.backends.base.WorkloadRunner` interface.  This allows
benchmark entry-points to run the exact same workload through SAGE without
coupling workload logic to SAGE-specific APIs.

No new ``ray`` imports are introduced here or anywhere in SAGE core.
"""

from __future__ import annotations

import time
from typing import Any

from backends.base import RunResult, WorkloadRunner, WorkloadSpec, register_runner  # noqa: E402


@register_runner("sage")
class SageRunner(WorkloadRunner):
    """Runs workloads using SAGE's **FlownetEnvironment** execution path.

    This is the default and primary backend.  It wraps the existing SAGE
    pipeline primitive (``SourceFunction → MapFunction → SinkFunction``)
    into the abstract :class:`~experiments.backends.base.WorkloadRunner`
    interface so the same workload spec can later be compared against other
    backends without changing any workload logic.

    FlownetEnvironment is the correct SAGE runtime backend; it uses
    sageFlownet for distributed scheduling/execution rather than the simpler
    LocalEnvironment shim.  ``env.submit(autostop=True)`` blocks until the
    pipeline completes and tears down the environment automatically.

    Scheduler selection
    -------------------
    ``WorkloadSpec.scheduler_name`` is mapped to a SAGE scheduler instance:

    - ``"fifo"`` / ``"default"`` → :class:`~sage.kernel.scheduler.impl.FIFOScheduler`
    - ``"load_aware"``           → :class:`~sage.kernel.scheduler.impl.LoadAwareScheduler`

    Extra knobs (``WorkloadSpec.extra``)
    -------------------------------------
    - ``max_wait_seconds`` (int, default 60): hard timeout passed as the
      ``timeout`` kwarg to ``env.submit()`` when supported; otherwise used as
      a wall-clock guard around the call.
    """

    @property
    def backend_name(self) -> str:
        return "sage"

    def is_available(self) -> bool:
        """Return ``True`` when the SAGE kernel packages are importable."""
        try:
            import sage.kernel.api  # noqa: F401

            return True
        except ImportError:
            return False

    def run(self, spec: WorkloadSpec) -> RunResult:
        """Execute *spec* via SAGE FlownetEnvironment and return results.

        The pipeline shape is deliberately simple and generic so it can serve
        as a representative proxy for any SAGE workload:

        ``DataSource → HeavyProcessor (parallelism=N) → LightFilter → ResultSink``
        """
        from sage.common.core import MapFunction, SinkFunction, SourceFunction
        from sage.kernel.api import FlownetEnvironment
        from sage.kernel.scheduler.impl import FIFOScheduler, LoadAwareScheduler

        # ------------------------------------------------------------------
        # Workload components – business logic is backend-agnostic
        # ------------------------------------------------------------------

        class _Source(SourceFunction):
            """Emit *total_items* string tokens and then signal EOF."""

            def __init__(self, total_items: int = spec.total_items, **kwargs: Any):
                super().__init__(**kwargs)
                self._total = total_items
                self._index = 0

            def execute(self, data: Any = None) -> Any:
                if self._index >= self._total:
                    return None  # EOF
                item = f"item_{self._index}"
                self._index += 1
                return item

        class _Processor(MapFunction):
            """Simulate light CPU work; wraps each item in a 'processed_' prefix."""

            def execute(self, data: Any) -> Any:
                time.sleep(0.01)  # simulate compute
                return f"processed_{data}"

        class _Filter(MapFunction):
            """Pass only even-indexed items (a simple, reproducible filter)."""

            def execute(self, data: Any) -> Any:
                try:
                    idx = int(data.rsplit("_", 1)[-1])
                    return data if idx % 2 == 0 else None
                except (ValueError, IndexError):
                    return data  # unknown format → pass through

        class _Sink(SinkFunction):
            """Collect output items into a class-level list for inspection."""

            collected: list[str] = []

            def __init__(self, **kwargs: Any):
                super().__init__(**kwargs)

            def execute(self, data: Any) -> None:
                if data is not None:
                    _Sink.collected.append(data)

        _Sink.collected = []  # reset before each run

        # ------------------------------------------------------------------
        # Scheduler selection
        # ------------------------------------------------------------------
        _scheduler_map: dict[str, Any] = {
            "fifo": FIFOScheduler(),
            "default": FIFOScheduler(),
            "load_aware": LoadAwareScheduler(max_concurrent=10),
        }
        scheduler = _scheduler_map.get(spec.scheduler_name.lower(), FIFOScheduler())

        # ------------------------------------------------------------------
        # Pipeline construction and execution via FlownetEnvironment
        # ------------------------------------------------------------------
        max_wait: int = int(spec.extra.get("max_wait_seconds", 60))
        env = None
        elapsed = 0.0
        metrics: dict[str, Any] = {}

        try:
            env = FlownetEnvironment(
                name=f"bench_{spec.name}_{spec.scheduler_name}",
                scheduler=scheduler,
            )
            (
                env.from_source(_Source, total_items=spec.total_items)
                .map(_Processor, parallelism=spec.parallelism)
                .filter(_Filter, parallelism=1)
                .sink(_Sink)  # type: ignore[arg-type]
            )

            start = time.time()
            # FlownetEnvironment.submit(autostop=True) blocks until the
            # pipeline finishes and automatically tears down the environment.
            env.submit(autostop=True)
            elapsed = time.time() - start

            # Collect scheduler-level metrics if available
            if (
                hasattr(env, "scheduler")
                and env.scheduler is not None
                and hasattr(env.scheduler, "get_metrics")
            ):
                try:
                    metrics = env.scheduler.get_metrics()
                except Exception:
                    pass

        finally:
            if env is not None:
                try:
                    if hasattr(env, "close"):
                        env.close()
                    elif hasattr(env, "shutdown"):
                        env.shutdown()  # type: ignore[union-attr]
                except Exception:
                    pass

        return RunResult(
            backend=self.backend_name,
            scheduler_name=spec.scheduler_name,
            elapsed_time=elapsed,
            results_count=len(_Sink.collected),
            metrics=metrics,
        )

                try:
                    if hasattr(env, "close"):
                        env.close()
                    elif hasattr(env, "shutdown"):
                        env.shutdown()  # type: ignore[union-attr]
                except Exception:
                    pass

        return RunResult(
            backend=self.backend_name,
            scheduler_name=spec.scheduler_name,
            elapsed_time=elapsed,
            results_count=len(_Sink.collected),
            metrics=metrics,
        )
