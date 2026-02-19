"""
Unit tests for the backend selector / registry in experiments/backends/base.py.

These tests use lightweight mock runners so no real SAGE environment is
started; the goal is to verify dispatch, error handling, and list_backends().
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# We need a clean registry for isolated tests, so we patch it per test.
# ---------------------------------------------------------------------------
# Import via 'backends.*' (experiments/ is in sys.path via conftest)
# to avoid triggering experiments/__init__.py which pulls sage.benchmark.
import backends.base as backend_base
from backends.base import (
    RunResult,
    WorkloadRunner,
    WorkloadSpec,
    get_runner,
    list_backends,
    register_runner,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_runner(name: str, available: bool = True) -> type[WorkloadRunner]:
    """Return a concrete WorkloadRunner subclass usable as a mock."""

    class MockRunner(WorkloadRunner):
        @property
        def backend_name(self) -> str:
            return name

        def is_available(self) -> bool:
            return available

        def run(self, spec: WorkloadSpec) -> RunResult:
            return RunResult(
                backend=name,
                scheduler_name=spec.scheduler_name,
                elapsed_time=0.001,
                results_count=spec.total_items // 2,  # simulate even-only filter
                metrics={"mock": True},
            )

    MockRunner.__qualname__ = f"MockRunner_{name}"
    return MockRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_registry():
    """Snapshot and restore the runner registry around each test."""
    original = dict(backend_base._REGISTRY)
    yield
    backend_base._REGISTRY.clear()
    backend_base._REGISTRY.update(original)


# ---------------------------------------------------------------------------
# Tests: registry / register_runner
# ---------------------------------------------------------------------------


def test_register_runner_adds_to_registry():
    MockRunner = _make_mock_runner("mock_a")
    register_runner("mock_a")(MockRunner)
    assert "mock_a" in backend_base._REGISTRY
    assert backend_base._REGISTRY["mock_a"] is MockRunner


def test_register_runner_lowercases_name():
    MockRunner = _make_mock_runner("mock_b")
    register_runner("Mock_B")(MockRunner)
    assert "mock_b" in backend_base._REGISTRY


def test_register_runner_overwrites():
    """Registering under the same name replaces the previous entry."""
    Runner1 = _make_mock_runner("mock_c")
    Runner2 = _make_mock_runner("mock_c")
    register_runner("mock_c")(Runner1)
    register_runner("mock_c")(Runner2)
    assert backend_base._REGISTRY["mock_c"] is Runner2


# ---------------------------------------------------------------------------
# Tests: list_backends
# ---------------------------------------------------------------------------


def test_list_backends_returns_sorted(isolated_registry):
    backend_base._REGISTRY.clear()
    register_runner("z_back")(_make_mock_runner("z_back"))
    register_runner("a_back")(_make_mock_runner("a_back"))
    register_runner("m_back")(_make_mock_runner("m_back"))
    assert list_backends() == ["a_back", "m_back", "z_back"]


def test_list_backends_empty_when_no_runners(isolated_registry):
    backend_base._REGISTRY.clear()
    assert list_backends() == []


# ---------------------------------------------------------------------------
# Tests: get_runner
# ---------------------------------------------------------------------------


def test_get_runner_returns_correct_instance():
    Runner = _make_mock_runner("my_runner")
    register_runner("my_runner")(Runner)
    runner = get_runner("my_runner")
    assert isinstance(runner, Runner)
    assert runner.backend_name == "my_runner"


def test_get_runner_is_case_insensitive():
    Runner = _make_mock_runner("ci_back")
    register_runner("ci_back")(Runner)
    assert isinstance(get_runner("CI_BACK"), Runner)
    assert isinstance(get_runner("ci_back"), Runner)


def test_get_runner_raises_value_error_unknown_backend(isolated_registry):
    backend_base._REGISTRY.clear()
    with pytest.raises(ValueError, match="Unknown backend"):
        get_runner("nonexistent")


def test_get_runner_raises_value_error_mentions_available():
    Runner = _make_mock_runner("exists")
    register_runner("exists")(Runner)
    with pytest.raises(ValueError, match="exists"):
        get_runner("not_registered")


def test_get_runner_raises_runtime_error_when_unavailable():
    UnavailableRunner = _make_mock_runner("unavail", available=False)
    register_runner("unavail")(UnavailableRunner)
    with pytest.raises(RuntimeError, match="not available"):
        get_runner("unavail")


# ---------------------------------------------------------------------------
# Tests: WorkloadSpec defaults
# ---------------------------------------------------------------------------


def test_workload_spec_defaults():
    spec = WorkloadSpec(name="test")
    assert spec.total_items == 10
    assert spec.parallelism == 2
    assert spec.scheduler_name == "default"
    assert spec.extra == {}


def test_workload_spec_extra_is_independent():
    spec1 = WorkloadSpec(name="a")
    spec2 = WorkloadSpec(name="b")
    spec1.extra["key"] = "val"
    assert "key" not in spec2.extra


# ---------------------------------------------------------------------------
# Tests: RunResult
# ---------------------------------------------------------------------------


def test_run_result_summary_contains_fields():
    r = RunResult(
        backend="mock",
        scheduler_name="fifo",
        elapsed_time=1.234,
        results_count=5,
        metrics={"total_scheduled": 10},
    )
    summary = r.summary()
    assert "mock" in summary
    assert "fifo" in summary
    assert "1.234" in summary
    assert "5" in summary
    assert "total_scheduled" in summary


# ---------------------------------------------------------------------------
# Tests: end-to-end mock dispatch
# ---------------------------------------------------------------------------


def test_dispatch_runs_workload_and_returns_result():
    Runner = _make_mock_runner("dispatch_test")
    register_runner("dispatch_test")(Runner)

    spec = WorkloadSpec(name="e2e", total_items=20, scheduler_name="fifo")
    runner = get_runner("dispatch_test")
    result = runner.run(spec)

    assert result.backend == "dispatch_test"
    assert result.scheduler_name == "fifo"
    assert result.results_count == 10  # total_items // 2
    assert result.metrics == {"mock": True}
