"""Unified SAGE vs Ray comparison report generator.

This script consumes one or more benchmark result directories/files containing
``unified_results.csv`` or ``unified_results.jsonl`` artifacts, merges records,
checks cross-backend configuration mismatches, and exports:

- summary markdown
- merged comparison CSV
- throughput comparison plot
- latency p95/p99 comparison plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from common.metrics_schema import normalize_metrics_record

SUPPORTED_RESULT_FILENAMES: tuple[str, ...] = (
    "unified_results.csv",
    "unified_results.jsonl",
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "seed",
    "nodes",
    "parallelism",
    "throughput",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "success_rate",
    "duration_seconds",
)


def discover_result_files(input_paths: list[Path]) -> list[Path]:
    """Discover supported result artifacts from directories or file paths."""
    discovered: set[Path] = set()

    for raw_path in input_paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_file():
            if path.name in SUPPORTED_RESULT_FILENAMES:
                discovered.add(path)
            continue

        for file_name in SUPPORTED_RESULT_FILENAMES:
            for candidate in path.rglob(file_name):
                if candidate.is_file():
                    discovered.add(candidate.resolve())

    return sorted(discovered)


def _infer_backend_from_path(path: Path) -> str | None:
    lowered = str(path).lower()
    if "sage" in lowered:
        return "sage"
    if "ray" in lowered:
        return "ray"
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            records.append(payload)
    return records


def _load_csv(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def load_normalized_records(result_files: list[Path]) -> pd.DataFrame:
    """Load and normalize metrics records from supported artifacts."""
    rows: list[dict[str, Any]] = []

    for file_path in result_files:
        if file_path.name.endswith(".jsonl"):
            raw_records = _load_jsonl(file_path)
        else:
            raw_records = _load_csv(file_path)

        inferred_backend = _infer_backend_from_path(file_path)

        for raw in raw_records:
            normalized = normalize_metrics_record(raw)
            if not normalized.get("backend") and inferred_backend:
                normalized["backend"] = inferred_backend
            normalized["source_file"] = str(file_path)
            rows.append(normalized)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "backend" in df.columns:
        df["backend"] = df["backend"].fillna("unknown").astype(str)
    if "workload" in df.columns:
        df["workload"] = df["workload"].fillna("unknown").astype(str)
    if "run_id" in df.columns:
        df["run_id"] = df["run_id"].fillna("unknown").astype(str)

    return df


def detect_config_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    """Detect run groups where backend configs are not aligned."""
    if df.empty:
        return pd.DataFrame()

    mismatch_rows: list[dict[str, Any]] = []

    for (workload, run_id), group in df.groupby(["workload", "run_id"], dropna=False):
        if group["backend"].nunique() < 2:
            continue

        signatures = group[
            ["backend", "seed", "nodes", "parallelism", "config_hash"]
        ].drop_duplicates()
        if (
            signatures[["seed", "nodes", "parallelism", "config_hash"]].drop_duplicates().shape[0]
            <= 1
        ):
            continue

        detail = "; ".join(
            (
                f"{row.backend}: seed={row.seed}, nodes={row.nodes}, "
                f"parallelism={row.parallelism}, config_hash={row.config_hash}"
            )
            for row in signatures.itertuples(index=False)
        )
        mismatch_rows.append(
            {
                "workload": workload,
                "run_id": run_id,
                "backend_count": int(group["backend"].nunique()),
                "detail": detail,
            }
        )

    return pd.DataFrame(mismatch_rows)


def _build_backend_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("backend", dropna=False)
        .agg(
            records=("backend", "count"),
            throughput_mean=("throughput", "mean"),
            latency_p95_mean=("latency_p95", "mean"),
            latency_p99_mean=("latency_p99", "mean"),
        )
        .reset_index()
        .sort_values("backend")
    )
    return summary


def _safe_float(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def write_summary_markdown(
    output_path: Path,
    merged_df: pd.DataFrame,
    mismatch_df: pd.DataFrame,
    csv_path: Path,
    throughput_plot_path: Path,
    latency_plot_path: Path,
) -> Path:
    """Write markdown summary report."""
    backend_summary = _build_backend_summary(merged_df)

    lines: list[str] = []
    lines.append("# Backend Comparison Report")
    lines.append("")
    lines.append(f"- Total records: {len(merged_df)}")
    lines.append(f"- Backends: {', '.join(sorted(merged_df['backend'].unique()))}")
    lines.append(f"- Workloads: {', '.join(sorted(merged_df['workload'].unique()))}")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Comparison CSV: `{csv_path.name}`")
    lines.append(f"- Throughput plot: `{throughput_plot_path.name}`")
    lines.append(f"- Latency plot: `{latency_plot_path.name}`")
    lines.append("")

    lines.append("## Backend Summary")
    lines.append("")
    lines.append("| backend | records | throughput_mean | latency_p95_mean | latency_p99_mean |")
    lines.append("|---|---:|---:|---:|---:|")

    for row in backend_summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.backend} | {row.records} | {_safe_float(row.throughput_mean)} | "
            f"{_safe_float(row.latency_p95_mean)} | {_safe_float(row.latency_p99_mean)} |"
        )

    lines.append("")
    lines.append("## Configuration Mismatches")
    lines.append("")

    if mismatch_df.empty:
        lines.append("No config mismatches detected across compared backends.")
    else:
        lines.append("| workload | run_id | backend_count | detail |")
        lines.append("|---|---|---:|---|")
        for row in mismatch_df.itertuples(index=False):
            detail = str(row.detail).replace("|", "\\|")
            lines.append(f"| {row.workload} | {row.run_id} | {row.backend_count} | {detail} |")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _plot_throughput(df: pd.DataFrame, output_path: Path) -> Path:
    grouped = df.groupby(["workload", "backend"], dropna=False)["throughput"].mean().reset_index()

    pivot = grouped.pivot(index="workload", columns="backend", values="throughput")
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Throughput Comparison")
    ax.set_xlabel("Workload")
    ax.set_ylabel("Throughput")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def _plot_latency(df: pd.DataFrame, output_path: Path) -> Path:
    grouped = (
        df.groupby(["workload", "backend"], dropna=False)[["latency_p95", "latency_p99"]]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    p95_pivot = grouped.pivot(index="workload", columns="backend", values="latency_p95")
    p95_pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Latency p95 Comparison")
    axes[0].set_xlabel("Workload")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].tick_params(axis="x", rotation=30)

    p99_pivot = grouped.pivot(index="workload", columns="backend", values="latency_p99")
    p99_pivot.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Latency p99 Comparison")
    axes[1].set_xlabel("Workload")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def run_comparison(input_paths: list[Path], output_dir: Path) -> dict[str, Path]:
    """Run the backend comparison pipeline and return artifact paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = discover_result_files(input_paths)
    if not result_files:
        raise FileNotFoundError(
            "No unified result artifacts found. Expected unified_results.csv/jsonl in input paths."
        )

    merged_df = load_normalized_records(result_files)
    if merged_df.empty:
        raise ValueError("No records found in discovered artifacts.")

    merged_df = merged_df.sort_values(["workload", "run_id", "backend"]).reset_index(drop=True)

    comparison_csv = output_dir / "comparison.csv"
    merged_df.to_csv(comparison_csv, index=False)

    mismatch_df = detect_config_mismatches(merged_df)

    throughput_plot = output_dir / "throughput_comparison.png"
    latency_plot = output_dir / "latency_p95_p99_comparison.png"
    _plot_throughput(merged_df, throughput_plot)
    _plot_latency(merged_df, latency_plot)

    summary_md = output_dir / "summary.md"
    write_summary_markdown(
        output_path=summary_md,
        merged_df=merged_df,
        mismatch_df=mismatch_df,
        csv_path=comparison_csv,
        throughput_plot_path=throughput_plot,
        latency_plot_path=latency_plot,
    )

    return {
        "summary": summary_md,
        "comparison_csv": comparison_csv,
        "throughput_plot": throughput_plot,
        "latency_plot": latency_plot,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare mixed backend benchmark outputs and generate unified report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input result directories/files (supports recursive discovery).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/backend_comparison",
        help="Output directory for summary markdown, CSV and plots.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    input_paths = [Path(item) for item in args.inputs]
    output_dir = Path(args.output_dir)

    artifacts = run_comparison(input_paths=input_paths, output_dir=output_dir)

    print("Backend comparison report generated:")
    for key, value in artifacts.items():
        print(f"- {key}: {value}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
