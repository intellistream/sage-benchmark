"""Execution guard helpers for benchmark pipelines."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionGuardResult:
    completed: bool
    timed_out: bool
    elapsed_seconds: float


def _is_running(env: Any) -> bool:
    if not hasattr(env, "is_running"):
        return False
    is_running_attr = env.is_running
    if callable(is_running_attr):
        return bool(is_running_attr())
    return bool(is_running_attr)


def run_pipeline_bounded(
    env: Any,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.2,
) -> ExecutionGuardResult:
    """Run pipeline with bounded waiting and timeout stop.

    LocalEnvironment uses non-blocking submit + JobManager polling.
    FlownetEnvironment uses autostop submit (blocking path).
    """
    start_time = time.time()
    platform = getattr(env, "platform", "")

    if platform == "local":
        env.submit(autostop=False)

        while time.time() - start_time < timeout_seconds:
            env_uuid = getattr(env, "env_uuid", None)
            if env_uuid is None:
                time.sleep(poll_interval_seconds)
                continue

            jobs = env.jobmanager.jobs
            if env_uuid not in jobs:
                elapsed = time.time() - start_time
                return ExecutionGuardResult(
                    completed=True, timed_out=False, elapsed_seconds=elapsed
                )

            time.sleep(poll_interval_seconds)

        env.stop()
        elapsed = time.time() - start_time
        return ExecutionGuardResult(completed=False, timed_out=True, elapsed_seconds=elapsed)

    env.submit(autostop=True)
    elapsed = time.time() - start_time

    return ExecutionGuardResult(completed=True, timed_out=False, elapsed_seconds=elapsed)
