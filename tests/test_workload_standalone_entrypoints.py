"""Lightweight guards for Q3-Q8 standalone workload entrypoints.

These tests validate source-level contract (CLI + metadata propagation) without
importing heavyweight runtime dependencies.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINT_FILES: dict[str, Path] = {
    "Q3": REPO_ROOT / "experiments" / "q3_noisyneighbor.py",
    "Q4": REPO_ROOT / "experiments" / "q4_scalefrontier.py",
    "Q5": REPO_ROOT / "experiments" / "q5_heteroresilience.py",
    "Q6": REPO_ROOT / "experiments" / "q6_bursttown.py",
    "Q7": REPO_ROOT / "experiments" / "q7_reconfigdrill.py",
    "Q8": REPO_ROOT / "experiments" / "q8_recoverysoak.py",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_q3_q8_files_exist() -> None:
    for workload, path in ENTRYPOINT_FILES.items():
        assert path.exists(), f"Missing standalone entrypoint for {workload}: {path}"


def test_q3_q8_use_common_cli_helpers() -> None:
    for workload, path in ENTRYPOINT_FILES.items():
        content = _read(path)
        assert "add_common_benchmark_args" in content, f"{workload} missing common CLI args"
        assert "validate_benchmark_args(args)" in content, f"{workload} missing arg validation"
        assert "build_run_config(" in content, f"{workload} missing run_config construction"


def test_q3_q8_support_dry_run() -> None:
    for workload, path in ENTRYPOINT_FILES.items():
        content = _read(path)
        assert "if args.dry_run:" in content, f"{workload} missing dry-run guard"
        assert "configuration validated" in content, f"{workload} missing dry-run message"


def test_q3_q8_propagate_unified_metadata() -> None:
    for workload, path in ENTRYPOINT_FILES.items():
        content = _read(path)
        assert "experiment.backend = args.backend" in content, f"{workload} missing backend metadata"
        assert "experiment.nodes = int(args.nodes)" in content, f"{workload} missing nodes metadata"
        assert (
            "experiment.parallelism = int(args.parallelism)" in content
        ), f"{workload} missing parallelism metadata"
        assert "experiment.run_id =" in content, f"{workload} missing run_id metadata"
