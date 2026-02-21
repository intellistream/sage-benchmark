"""Tests for paired backend automation runner (Issue #7)."""

from __future__ import annotations

from pathlib import Path

from analysis.run_paired_backends import (
    CommandResult,
    build_run_id,
    compute_config_hash,
    run_paired_backends,
)


def test_compute_config_hash_is_stable() -> None:
    payload_a = {"seed": 42, "scheduler": "fifo", "nodes": 1}
    payload_b = {"nodes": 1, "scheduler": "fifo", "seed": 42}
    assert compute_config_hash(payload_a) == compute_config_hash(payload_b)


def test_build_run_id_uses_explicit_value() -> None:
    assert build_run_id(config_hash="abcdef123456", explicit_run_id="manual-run") == "manual-run"


def test_run_paired_backends_success_creates_manifest(tmp_path: Path) -> None:
    def fake_runner(command: list[str], cwd: Path) -> CommandResult:
        _ = cwd

        if "--output-dir" in command:
            output_dir = Path(command[command.index("--output-dir") + 1])
        else:
            output_dir = None

        if command[1].endswith("scheduler_comparison.py"):
            assert output_dir is not None
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "unified_results.csv").write_text(
                "backend,workload,run_id\n"
                f"{command[command.index('--backend') + 1]},scheduler_comparison,{command[command.index('--run-id') + 1]}\n",
                encoding="utf-8",
            )
            (output_dir / "unified_results.jsonl").write_text("{}\n", encoding="utf-8")
            return CommandResult(returncode=0, stdout="ok", stderr="")

        if command[1].endswith("compare_backends.py"):
            assert output_dir is not None
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.md").write_text("# summary\n", encoding="utf-8")
            (output_dir / "comparison.csv").write_text("backend\n", encoding="utf-8")
            return CommandResult(returncode=0, stdout="ok", stderr="")

        raise AssertionError(f"unexpected command: {command}")

    artifacts = run_paired_backends(
        output_root=tmp_path,
        scheduler="fifo",
        items=10,
        parallelism=2,
        nodes=1,
        seed=42,
        python_executable="python",
        run_id="test-run",
        command_runner=fake_runner,
    )

    assert artifacts["summary"].exists()
    assert artifacts["comparison_csv"].exists()
    assert artifacts["manifest"].exists()


def test_run_paired_backends_raises_on_backend_failure(tmp_path: Path) -> None:
    def fake_runner(command: list[str], cwd: Path) -> CommandResult:
        _ = cwd
        if (
            command[1].endswith("scheduler_comparison.py")
            and command[command.index("--backend") + 1] == "ray"
        ):
            return CommandResult(returncode=2, stdout="", stderr="ray failed")

        if "--output-dir" in command:
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "unified_results.csv").write_text(
                "backend,workload,run_id\n", encoding="utf-8"
            )
            (output_dir / "unified_results.jsonl").write_text("{}\n", encoding="utf-8")

        return CommandResult(returncode=0, stdout="ok", stderr="")

    try:
        run_paired_backends(
            output_root=tmp_path,
            scheduler="fifo",
            items=10,
            parallelism=2,
            nodes=1,
            seed=42,
            python_executable="python",
            run_id="fail-run",
            command_runner=fake_runner,
        )
    except RuntimeError as exc:
        assert "ray backend run failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
