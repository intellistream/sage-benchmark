"""Run paired benchmark executions for ``sage`` and ``ray`` backends.

This script provides an automation entrypoint for Issue #7:
- one command triggers paired backend runs
- artifact directories include ``run_id`` and ``config_hash``
- failures produce actionable logs and non-zero exit status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEDULER_COMPARISON_SCRIPT = REPO_ROOT / "experiments" / "scheduler_comparison.py"
COMPARE_BACKENDS_SCRIPT = REPO_ROOT / "experiments" / "analysis" / "compare_backends.py"


@dataclass(frozen=True)
class CommandResult:
    """Minimal command result structure used by the runner abstraction."""

    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], Path], CommandResult]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")


def compute_config_hash(payload: dict[str, object]) -> str:
    """Compute deterministic short hash from config payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def build_run_id(config_hash: str, explicit_run_id: str | None = None) -> str:
    """Build run identifier that includes UTC timestamp and config hash."""
    if explicit_run_id:
        return explicit_run_id.strip()
    return f"paired-{_ts(_utc_now())}-{config_hash}"


def _default_runner(command: list[str], cwd: Path) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _write_command_log(log_path: Path, command: list[str], result: CommandResult) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"$ {' '.join(command)}",
                "",
                "[stdout]",
                result.stdout,
                "",
                "[stderr]",
                result.stderr,
                "",
                f"[returncode] {result.returncode}",
            ]
        ),
        encoding="utf-8",
    )


def run_paired_backends(
    *,
    output_root: Path,
    scheduler: str,
    items: int,
    parallelism: int,
    nodes: int,
    seed: int,
    python_executable: str,
    run_id: str | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Path]:
    """Execute paired ``sage`` and ``ray`` runs and produce comparison artifacts."""
    config_payload = {
        "scheduler": scheduler,
        "items": items,
        "parallelism": parallelism,
        "nodes": nodes,
        "seed": seed,
    }
    config_hash = compute_config_hash(config_payload)
    resolved_run_id = build_run_id(config_hash=config_hash, explicit_run_id=run_id)

    run_root = output_root / f"run_id={resolved_run_id}" / f"config_hash={config_hash}"
    backend_root = run_root / "backends"
    compare_root = run_root / "comparison"
    logs_root = run_root / "logs"
    backend_root.mkdir(parents=True, exist_ok=True)
    compare_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    runner = command_runner or _default_runner
    started_at = _utc_now().isoformat()

    manifest_path = run_root / "manifest.json"
    failure_path = run_root / "failure_report.json"

    artifacts: dict[str, Path] = {}

    try:
        for backend in ("sage", "ray"):
            backend_output = backend_root / backend
            backend_output.mkdir(parents=True, exist_ok=True)
            backend_log = logs_root / f"{backend}.log"

            command = [
                python_executable,
                str(SCHEDULER_COMPARISON_SCRIPT),
                "--backend",
                backend,
                "--scheduler",
                scheduler,
                "--items",
                str(items),
                "--parallelism",
                str(parallelism),
                "--nodes",
                str(nodes),
                "--seed",
                str(seed),
                "--run-id",
                resolved_run_id,
                "--workload",
                "scheduler_comparison",
                "--output-dir",
                str(backend_output),
            ]

            result = runner(command, REPO_ROOT)
            _write_command_log(backend_log, command, result)

            if result.returncode != 0:
                raise RuntimeError(
                    f"{backend} backend run failed (exit={result.returncode}). "
                    f"See log: {backend_log}"
                )

            backend_csv = backend_output / "unified_results.csv"
            backend_jsonl = backend_output / "unified_results.jsonl"
            if not backend_csv.exists() or not backend_jsonl.exists():
                raise FileNotFoundError(
                    f"Missing backend artifacts for '{backend}'. Expected files: "
                    f"{backend_csv} and {backend_jsonl}."
                )

            artifacts[f"{backend}_csv"] = backend_csv
            artifacts[f"{backend}_jsonl"] = backend_jsonl

        compare_log = logs_root / "compare.log"
        compare_command = [
            python_executable,
            str(COMPARE_BACKENDS_SCRIPT),
            str(backend_root / "sage"),
            str(backend_root / "ray"),
            "--output-dir",
            str(compare_root),
        ]
        compare_result = runner(compare_command, REPO_ROOT)
        _write_command_log(compare_log, compare_command, compare_result)

        if compare_result.returncode != 0:
            raise RuntimeError(
                f"Backend comparison failed (exit={compare_result.returncode}). "
                f"See log: {compare_log}"
            )

        summary_path = compare_root / "summary.md"
        comparison_csv_path = compare_root / "comparison.csv"
        if not summary_path.exists() or not comparison_csv_path.exists():
            raise FileNotFoundError(
                "Comparison artifacts missing. Expected summary.md and comparison.csv "
                f"under {compare_root}."
            )

        artifacts["summary"] = summary_path
        artifacts["comparison_csv"] = comparison_csv_path
        artifacts["compare_log"] = compare_log

        manifest = {
            "status": "success",
            "started_at": started_at,
            "finished_at": _utc_now().isoformat(),
            "run_id": resolved_run_id,
            "config_hash": config_hash,
            "config": config_payload,
            "paths": {k: str(v) for k, v in artifacts.items()},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        artifacts["manifest"] = manifest_path
        return artifacts
    except Exception as exc:
        failure_payload = {
            "status": "failed",
            "started_at": started_at,
            "failed_at": _utc_now().isoformat(),
            "run_id": resolved_run_id,
            "config_hash": config_hash,
            "config": config_payload,
            "error": str(exc),
            "logs_dir": str(logs_root),
        }
        failure_path.write_text(json.dumps(failure_payload, indent=2), encoding="utf-8")
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired scheduler benchmark on sage + ray backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python experiments/analysis/run_paired_backends.py
  python experiments/analysis/run_paired_backends.py --scheduler fifo --items 50 --parallelism 4
  python experiments/analysis/run_paired_backends.py --run-id nightly-20260220
""",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/paired_backend_runs",
        help="Root directory where run artifacts will be stored.",
    )
    parser.add_argument(
        "--scheduler",
        default="fifo",
        choices=("fifo", "load_aware", "default"),
        help="Scheduler name passed to scheduler_comparison workload.",
    )
    parser.add_argument("--items", type=int, default=10, help="Total items for workload generation.")
    parser.add_argument(
        "--parallelism", type=int, default=2, help="Operator/task parallelism for both backends."
    )
    parser.add_argument("--nodes", type=int, default=1, help="Node count metadata for both backends.")
    parser.add_argument("--seed", type=int, default=42, help="Global seed metadata for both backends.")
    parser.add_argument("--run-id", default="", help="Optional explicit run_id.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch child benchmark commands.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    artifacts = run_paired_backends(
        output_root=Path(args.output_root),
        scheduler=args.scheduler,
        items=args.items,
        parallelism=args.parallelism,
        nodes=args.nodes,
        seed=args.seed,
        python_executable=args.python,
        run_id=args.run_id.strip() or None,
    )

    print("Paired backend run completed.")
    for key in sorted(artifacts):
        print(f"- {key}: {artifacts[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
