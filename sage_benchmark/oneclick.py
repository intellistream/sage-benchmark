"""One-click orchestration for benchmark run, HF upload, and docs refresh."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run_command(command: list[str], *, cwd: Path) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n$ (cd {cwd} && {printable})")
    subprocess.run(command, cwd=str(cwd), check=True)


def _default_output_dir(repo_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "results" / f"oneclick_{stamp}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run full SAGE benchmark pipeline: all experiments -> aggregate+merge -> "
            "upload HF -> refresh sage-docs leaderboard"
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to sage-benchmark repository root.",
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "sage-docs",
        help="Path to sage-docs repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for benchmark run (defaults to results/oneclick_<timestamp>).",
    )
    parser.add_argument("--quick", action="store_true", help="Run benchmark in quick mode.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate benchmark configs only; skip actual workload execution.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running benchmark experiments.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip HF aggregation/merge/upload steps.",
    )
    parser.add_argument(
        "--skip-docs-refresh",
        action="store_true",
        help="Skip refreshing leaderboard files in sage-docs.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used for all subprocesses.",
    )
    parser.add_argument(
        "--benchmark-command",
        type=str,
        default="",
        help=(
            "Optional custom benchmark command. If set, this command is executed "
            "instead of the default 'python -m sage.benchmark.benchmark_sage --all ...'."
        ),
    )

    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    docs_root = args.docs_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else _default_output_dir(repo_root)

    if not repo_root.exists():
        raise FileNotFoundError(f"sage-benchmark repo root not found: {repo_root}")
    if not args.skip_docs_refresh and not docs_root.exists():
        raise FileNotFoundError(f"sage-docs repo root not found: {docs_root}")

    print("=" * 72)
    print("SAGE Benchmark One-Click Pipeline")
    print("=" * 72)
    print(f"repo_root: {repo_root}")
    print(f"docs_root: {docs_root}")
    print(f"output_dir: {output_dir}")
    print(f"python: {args.python}")

    if not args.skip_run:
        if args.benchmark_command:
            benchmark_cmd = shlex.split(args.benchmark_command)
        else:
            benchmark_cmd = [
                args.python,
                str(repo_root / "__main__.py"),
                "--all",
                "--output-dir",
                str(output_dir),
            ]
            if args.quick:
                benchmark_cmd.append("--quick")
            if args.dry_run:
                benchmark_cmd.append("--dry-run")
        _run_command(benchmark_cmd, cwd=repo_root)
    else:
        print("\n[skip] benchmark run")

    if not args.skip_upload:
        _run_command([args.python, "scripts/aggregate_for_hf.py"], cwd=repo_root)
        _run_command([args.python, "scripts/merge_and_upload.py"], cwd=repo_root)
        _run_command([args.python, "scripts/upload_to_hf.py"], cwd=repo_root)
    else:
        print("\n[skip] HF upload pipeline")

    if not args.skip_docs_refresh:
        _run_command([args.python, "fetch_benchmark_data.py"], cwd=docs_root)
    else:
        print("\n[skip] sage-docs leaderboard refresh")

    print("\n✅ One-click pipeline completed.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
