"""Behavioral tests for the Q1--Q8 matrix orchestrator."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_matrix_module():
    spec = importlib.util.spec_from_file_location(
        "sage_benchmark_matrix_main",
        REPO_ROOT / "__main__.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeConfigLoader:
    def load(self, _path: str):
        return self.get_default_config("Q1")

    def get_default_config(self, experiment: str):
        return SimpleNamespace(
            experiment_section=experiment,
            workload=SimpleNamespace(seed=42),
            hardware=SimpleNamespace(cpu_nodes=0),
        )

    def apply_quick_mode(self, config):
        return config


def _install_fake_workloads(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    experiments_package = types.ModuleType("experiments")
    experiments_package.__path__ = []
    monkeypatch.setitem(sys.modules, "experiments", experiments_package)

    common_package = types.ModuleType("experiments.common")
    common_package.__path__ = []
    monkeypatch.setitem(sys.modules, "experiments.common", common_package)
    cli_module = types.ModuleType("experiments.common.cli_args")

    def add_common_benchmark_args(parser, **_kwargs):
        parser.add_argument("--backend", default="sage")
        parser.add_argument("--nodes", type=int, default=1)
        parser.add_argument("--parallelism", type=int, default=2)
        parser.add_argument("--repeat", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--output-dir", default="results")
        parser.add_argument("--verbose", action="store_true")
        parser.add_argument("--quick", action="store_true")
        parser.add_argument("--dry-run", action="store_true")

    cli_module.add_common_benchmark_args = add_common_benchmark_args
    cli_module.validate_benchmark_args = lambda _args: None
    cli_module.build_run_config = lambda args, **extra: {
        "backend": args.backend,
        "nodes": args.nodes,
        "parallelism": args.parallelism,
        "repeat": args.repeat,
        "seed": args.seed,
        **extra,
    }
    monkeypatch.setitem(sys.modules, "experiments.common.cli_args", cli_module)

    config_package = types.ModuleType("config")
    config_package.__path__ = []
    monkeypatch.setitem(sys.modules, "config", config_package)
    config_module = types.ModuleType("config.config_loader")
    config_module.ConfigLoader = _FakeConfigLoader
    monkeypatch.setitem(sys.modules, "config.config_loader", config_module)

    classes = {
        "experiments.q1_pipelinechain": ("E2EPipelineExperiment", "Q1"),
        "experiments.q2_controlmix": ("ControlPlaneExperiment", "Q2"),
        "experiments.q3_noisyneighbor": ("IsolationExperiment", "Q3"),
        "experiments.q4_scalefrontier": ("ScalabilityExperiment", "Q4"),
        "experiments.q5_heteroresilience": ("HeterogeneityExperiment", "Q5"),
        "experiments.q6_bursttown": ("BurstTownExperiment", "Q6"),
        "experiments.q7_reconfigdrill": ("ReconfigDrillExperiment", "Q7"),
        "experiments.q8_recoverysoak": ("RecoverySoakExperiment", "Q8"),
    }

    def make_experiment(workload: str):
        class FakeExperiment:
            def __init__(self, config, output_dir, verbose=False):
                self.config = config
                self.output_dir = output_dir

            def setup(self):
                calls.append(f"{workload}:setup")

            def run(self):
                calls.append(f"{workload}:run")
                if workload == "Q2":
                    raise RuntimeError("intentional Q2 failure")
                return {"workload": workload}

            def teardown(self):
                calls.append(f"{workload}:teardown")

        return FakeExperiment

    for module_name, (class_name, workload) in classes.items():
        module = types.ModuleType(module_name)
        setattr(module, class_name, make_experiment(workload))
        monkeypatch.setitem(sys.modules, module_name, module)


@pytest.mark.parametrize(
    ("failure_flag", "expected_attempts", "last_workload"),
    [
        ("--continue-on-error", 8, "q8"),
        ("--fail-fast", 2, "q2"),
    ],
)
def test_matrix_failure_policy_sets_nonzero_exit_and_writes_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_flag: str,
    expected_attempts: int,
    last_workload: str,
) -> None:
    matrix = _load_matrix_module()
    calls: list[str] = []
    _install_fake_workloads(monkeypatch, calls)
    output_dir = tmp_path / "matrix"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "__main__.py",
            "--all",
            failure_flag,
            "--output-dir",
            str(output_dir),
        ],
    )

    assert matrix.main() == 1

    status = json.loads((output_dir / "matrix_status.json").read_text(encoding="utf-8"))
    assert status["failed"] == 1
    assert status["total"] == expected_attempts
    assert status["attempts"][-1]["workload"] == last_workload
    assert "Q2:teardown" in calls
    assert "intentional Q2 failure" in (output_dir / "q2" / "run.log").read_text(encoding="utf-8")
