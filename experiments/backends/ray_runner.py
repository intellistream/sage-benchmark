"""
Ray backend runner for sage-benchmark (MVP).

Provides an MVP Ray execution path for Workload4 that satisfies the same
:class:`~experiments.backends.base.WorkloadRunner` interface as the SAGE
backend.  This allows side-by-side comparison via ``--backend sage`` vs
``--backend ray`` without changing any workload business logic.

Scope
-----
- Covers Workload4 only in this first iteration.
- Ray is an *optional* dependency.  When it is not installed this module
  imports cleanly; :meth:`RayRunner.is_available` returns ``False`` and
  :func:`~experiments.backends.base.get_runner` raises a helpful
  ``RuntimeError`` with install guidance instead of an ``ImportError``.
- No new SAGE-internal imports; the business logic mirrors the SAGE runner
  so results are directly comparable.

Install guidance (when Ray is missing)
---------------------------------------
    python -m pip install -e .[ray-baseline]

Usage
-----
    from backends.base import WorkloadSpec, get_runner
    import backends.ray_runner          # registers "ray" backend
    import backends.sage_runner         # registers "sage" backend

    spec = WorkloadSpec(name="wl4", total_items=20, parallelism=4)
    for backend in ("sage", "ray"):
        runner = get_runner(backend)
        result = runner.run(spec)
        print(result.summary())
"""

from __future__ import annotations

import time
from typing import Any

from backends.base import RunResult, WorkloadRunner, WorkloadSpec, register_runner

# ---------------------------------------------------------------------------
# Ray availability guard
# ---------------------------------------------------------------------------

_RAY_MIN_VERSION = "2.9"

_RAY_INSTALL_HINT = (
    "Ray is not installed.  Install it with:\n"
    "    python -m pip install -e .[ray-baseline]\n"
    "Or install Ray directly with:\n"
    f'    python -m pip install "ray[default]>={_RAY_MIN_VERSION}"\n'
    "Then re-run with --backend ray."
)


def _ray_available() -> bool:
    """Return ``True`` when Ray is importable (no cluster needed)."""
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@register_runner("ray")
class RayRunner(WorkloadRunner):
    """Runs workloads using Ray distributed task execution.

    Architecture (MVP)
    ------------------
    The pipeline shape mirrors the SAGE runner so results are directly
    comparable:

        ``[item_0, item_1, …] → remote _process_item → remote _filter_item``

    * Each item is dispatched as an independent Ray remote task, giving
      natural task-level parallelism.
    * ``parallelism`` from :class:`~backends.base.WorkloadSpec` is used as the
      Ray ``num_cpus`` resource hint per task and as the size of the submission
      batch (``ray.get`` windows).
    * Ray is initialised lazily inside :meth:`run`; the local process acts as
      the driver.  The cluster / local-mode choice is inferred automatically
      by Ray.
    * Worker functions are decorated inside :meth:`run` so they are serialised
      by cloudpickle (function body) rather than by module reference, which
      means they work correctly in local mode without any extra ``PYTHONPATH``
      configuration on the worker.

    Extra knobs (``WorkloadSpec.extra``)
    --------------------------------------
    - ``ray_address``  (str, default ``None``): Ray cluster address.
      Leave unset to force local mode.
    - ``num_cpus_per_task`` (float, default ``0.5``): CPU resource per task.
    - ``ray_ignore_reinit_error`` (bool, default ``True``): whether to ignore
      a second ``ray.init`` call in the same process.
    - ``sleep_per_item`` (float, default ``0.01``): sleep seconds injected into
      each process step to simulate compute, matching the SAGE runner default.
        - ``input_batches`` (list[list[str]], optional): fixed benchmark input
            batches for backend parity. If set, overrides generated ``item_<n>``
            source tokens.
        - ``warmup_items`` (list[str], optional): deterministic warmup input set;
            warmup is executed before timed benchmark run and excluded from metrics.
    """

    @property
    def backend_name(self) -> str:
        return "ray"

    def is_available(self) -> bool:
        """Return ``True`` when Ray is importable."""
        return _ray_available()

    def run(self, spec: WorkloadSpec) -> RunResult:
        """Execute *spec* via Ray remote tasks and return unified results.

        Parameters
        ----------
        spec:
            Backend-agnostic workload description.  The following
            ``spec.extra`` keys are honoured:

            - ``ray_address`` – Ray cluster address (default: auto / local).
            - ``num_cpus_per_task`` – CPU fraction per task (default: 0.5).
            - ``ray_ignore_reinit_error`` – silently skip re-init (default: True).
            - ``sleep_per_item`` – compute simulation sleep per item (default 0.01s).

        Returns
        -------
        RunResult
            Populated with wall-clock elapsed time, collected-item count and
            Ray-specific metrics (tasks submitted, tasks succeeded, workers).
        """
        if not self.is_available():
            raise RuntimeError(_RAY_INSTALL_HINT)

        import ray

        # ------------------------------------------------------------------
        # Ray initialisation
        # ------------------------------------------------------------------
        ray_address: str | None = spec.extra.get("ray_address", None)
        ignore_reinit: bool = bool(spec.extra.get("ray_ignore_reinit_error", True))
        num_cpus_per_task: float = float(spec.extra.get("num_cpus_per_task", 0.5))
        sleep_per_item: float = float(spec.extra.get("sleep_per_item", 0.01))
        provided_batches: list[list[str]] | None = spec.extra.get("input_batches")
        warmup_items: list[str] = list(spec.extra.get("warmup_items", []))

        if not ray.is_initialized():
            init_kwargs: dict[str, Any] = {"ignore_reinit_error": ignore_reinit}
            if ray_address:
                init_kwargs["address"] = ray_address
            ray.init(**init_kwargs)

        # ------------------------------------------------------------------
        # Define worker functions inline so Ray serialises their bodies via
        # cloudpickle rather than looking them up by module path on workers.
        # This avoids "No module named 'backends'" errors in local mode.
        # ------------------------------------------------------------------
        _sleep = sleep_per_item  # captured by closure

        @ray.remote(num_cpus=num_cpus_per_task)
        def _process(item: str) -> str:
            import time as _time
            _time.sleep(_sleep)
            return f"processed_{item}"

        @ray.remote(num_cpus=num_cpus_per_task)
        def _filter(item: str) -> str | None:
            try:
                idx = int(item.rsplit("_", 1)[-1])
                return item if idx % 2 == 0 else None
            except (ValueError, IndexError):
                return item

        # ------------------------------------------------------------------
        # Source: use fixed parity batches when provided, otherwise generate
        # default item tokens.
        # ------------------------------------------------------------------
        if provided_batches:
            items = [item for batch in provided_batches for item in batch]
        else:
            items = [f"item_{i}" for i in range(spec.total_items)]

        # ------------------------------------------------------------------
        # Pipeline execution via Ray remote tasks
        # Batch size = spec.parallelism to control in-flight concurrency
        # ------------------------------------------------------------------
        batch_size: int = max(1, spec.parallelism)
        collected: list[str] = []
        tasks_submitted: int = 0
        tasks_succeeded: int = 0

        # Warmup pass (not timed / not counted)
        if warmup_items:
            warmup_process_refs = [_process.remote(item) for item in warmup_items]
            warmup_processed: list[str] = ray.get(warmup_process_refs)
            warmup_filter_refs = [_filter.remote(item) for item in warmup_processed]
            _ = ray.get(warmup_filter_refs)

        start = time.time()

        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start : batch_start + batch_size]

            # Stage 1: process
            process_refs = [_process.remote(item) for item in batch]
            processed: list[str] = ray.get(process_refs)
            tasks_submitted += len(process_refs)

            # Stage 2: filter
            filter_refs = [_filter.remote(p_item) for p_item in processed]
            filtered: list[str | None] = ray.get(filter_refs)
            tasks_submitted += len(filter_refs)

            # Sink: collect non-None results
            for result_item in filtered:
                if result_item is not None:
                    collected.append(result_item)
                    tasks_succeeded += 1

        elapsed = time.time() - start

        # ------------------------------------------------------------------
        # Ray-specific metrics
        # ------------------------------------------------------------------
        metrics: dict[str, Any] = {
            "tasks_submitted": tasks_submitted,
            "tasks_succeeded": tasks_succeeded,
            "items_filtered_out": len(items) - tasks_succeeded,
            "warmup_items": len(warmup_items),
            "input_items": len(items),
            "sampling_strategy": spec.extra.get("sampling_strategy", "default_generated"),
        }

        # Attempt to collect cluster-level node info (non-fatal if unavailable)
        try:
            nodes = ray.nodes()
            metrics["ray_nodes"] = len([n for n in nodes if n.get("Alive", False)])
        except Exception:
            pass

        return RunResult(
            backend=self.backend_name,
            scheduler_name=spec.scheduler_name,
            elapsed_time=elapsed,
            results_count=len(collected),
            metrics=metrics,
        )
