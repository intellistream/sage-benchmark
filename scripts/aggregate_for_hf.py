#!/usr/bin/env python3
"""
用户本地聚合命令：从 HF 拉取最新数据并与本地结果合并

这是用户在本地运行的命令，用于准备上传到 GitHub 的数据。

工作流程：
1. 从 HF 下载公开的 benchmark 数据（无需 token）
2. 扫描本地 results/ 目录的新结果（unified_results.jsonl）
3. 智能合并（去重，选性能更好的）
4. 保存到 hf_data/ 目录
5. 用户提交 hf_data/ 到 git（不提交 results/）

运行方式：
    python scripts/aggregate_for_hf.py

HF 仓库（公开访问）：
    https://huggingface.co/datasets/intellistream/sage-benchmark-results
"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter
from pathlib import Path

# HF 配置
HF_REPO = "intellistream/sage-benchmark-results"
HF_BRANCH = "main"
Q_MATRIX_WORKLOADS = frozenset(f"q{index}" for index in range(1, 9))
REQUIRED_IDENTITY_FIELDS = (
    "backend",
    "workload",
    "run_id",
    "seed",
    "nodes",
    "parallelism",
    "config_hash",
)


def normalize_workload_name(value: object) -> object:
    if isinstance(value, str) and value.lower() in Q_MATRIX_WORKLOADS:
        return value.lower()
    return value


def validate_local_record(record: object, *, source: str) -> dict:
    """Validate and canonicalise one locally generated record.

    Remote historical records remain readable for backwards compatibility, but
    malformed new records must not silently enter the publication pipeline.
    """
    if not isinstance(record, dict):
        raise ValueError(f"{source}: expected a JSON object")
    missing = [field for field in REQUIRED_IDENTITY_FIELDS if record.get(field) is None]
    if missing:
        raise ValueError(f"{source}: missing required fields: {', '.join(missing)}")
    normalized = dict(record)
    normalized["workload"] = normalize_workload_name(normalized["workload"])
    return normalized


def download_from_hf(filename: str) -> list[dict]:
    """从 Hugging Face 下载现有数据（公开，无需 token）。"""
    # 优先使用 mirror，避免国内网络超时
    mirror = "https://hf-mirror.com"
    url = f"{mirror}/datasets/{HF_REPO}/resolve/{HF_BRANCH}/{filename}"
    print(f"📥 下载 HF 数据: {url}")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            print(f"  ✓ 下载成功: {len(data)} 条记录")
            return data
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("  ⚠️ 文件不存在（首次上传）")
        else:
            # 尝试主站
            alt_url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_BRANCH}/{filename}"
            print(f"  ⚠️ mirror HTTP {e.code}，尝试主站: {alt_url}")
            try:
                with urllib.request.urlopen(alt_url, timeout=30) as response2:
                    data = json.loads(response2.read().decode("utf-8"))
                    print(f"  ✓ 下载成功: {len(data)} 条记录")
                    return data
            except Exception as e2:
                print(f"  ⚠️ 主站也失败: {e2}")
        return []
    except Exception as e:
        print(f"  ⚠️ 下载失败: {e}")
        return []


def _is_valid_record(record: dict) -> bool:
    """Returns True if the record has at least some successful results.

    Filters out runs where all requests failed (e.g., gateway not running),
    identified by throughput=0 AND latency_p99=0 AND success_rate=0.
    """
    tp = record.get("throughput")
    sr = record.get("success_rate")
    lat = record.get("latency_p99")
    # A scheduler-style entry has no success_rate; only filter when all three are zeroed out
    if (
        tp is not None
        and float(tp) == 0.0
        and sr is not None
        and float(sr) == 0.0
        and (lat is None or lat == 0)
    ):
        return False
    return True


def load_local_results(results_dir: Path) -> list[dict]:
    """递归加载 results/ 目录下的所有 unified_results.jsonl 文件。"""
    all_records: list[dict] = []

    errors: list[str] = []
    for jsonl_file in sorted(results_dir.rglob("unified_results.jsonl")):
        try:
            skipped = 0
            with jsonl_file.open("r", encoding="utf-8") as fh:
                for line_number, line in enumerate(fh, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = validate_local_record(
                        json.loads(stripped),
                        source=f"{jsonl_file}:{line_number}",
                    )
                    if not _is_valid_record(record):
                        skipped += 1
                        continue
                    all_records.append(record)
            label = f"{jsonl_file.relative_to(results_dir)}"
            if skipped:
                print(f"  ✓ 加载: {label} (跳过 {skipped} 条全失败记录)")
            else:
                print(f"  ✓ 加载: {label}")
        except Exception as e:
            print(f"  ✗ 加载失败: {jsonl_file} - {e}")
            errors.append(str(e))

    if errors:
        raise ValueError(
            f"拒绝聚合：发现 {len(errors)} 个格式错误的本地结果文件。请修复上述错误后重试。"
        )
    return all_records


def get_config_key(entry: dict) -> str:
    """生成配置唯一标识 key（用于去重）。"""
    parts = [
        str(entry.get("backend", "")),
        str(normalize_workload_name(entry.get("workload", ""))),
        str(entry.get("seed", "")),
        str(entry.get("nodes", "")),
        str(entry.get("parallelism", "")),
        str(entry.get("config_hash", "")),
    ]
    return "|".join(parts)


def is_better_result(new_entry: dict, existing_entry: dict) -> bool:
    """判断新结果是否比现有结果更好（throughput 优先，其次 latency_p50 更低）。"""
    new_tp = new_entry.get("throughput")
    ext_tp = existing_entry.get("throughput")
    if new_tp is not None and ext_tp is not None:
        if abs(new_tp - ext_tp) > 1e-9:
            return new_tp > ext_tp

    new_lat = new_entry.get("latency_p50")
    ext_lat = existing_entry.get("latency_p50")
    if new_lat is not None and ext_lat is not None:
        return new_lat < ext_lat

    return False


def merge_results(existing: list[dict], new_results: list[dict]) -> list[dict]:
    """合并现有数据和新数据（以 existing 为基准，new_results 追加或更新）。"""
    merged: dict[str, dict] = {}

    # 先加入现有数据
    for entry in existing:
        key = get_config_key(entry)
        merged[key] = entry

    added = updated = skipped = 0

    for entry in new_results:
        key = get_config_key(entry)
        if key not in merged:
            merged[key] = entry
            added += 1
            print(f"    ✓ 新增: {key[:60]}...")
        elif is_better_result(entry, merged[key]):
            merged[key] = entry
            updated += 1
            print(f"    ↑ 更新 (更好): {key[:60]}...")
        else:
            skipped += 1

    print(f"  📊 合并结果: 新增 {added}, 更新 {updated}, 跳过 {skipped}, 总计 {len(merged)}")
    return list(merged.values())


def build_coverage_summary(records: list[dict]) -> dict:
    """Build deterministic coverage counts for logs and publication."""
    by_workload = Counter(str(normalize_workload_name(row.get("workload", ""))) for row in records)
    by_backend = Counter(str(row.get("backend", "")) for row in records)
    by_seed = Counter(str(row.get("seed", "")) for row in records)
    return {
        "total_records": len(records),
        "by_workload": dict(sorted(by_workload.items())),
        "by_backend": dict(sorted(by_backend.items())),
        "by_seed": dict(sorted(by_seed.items())),
    }


def main() -> None:
    print("=" * 70)
    print("📦 SAGE Benchmark - 本地聚合工具")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results"
    hf_output_dir = base_dir / "hf_data"

    hf_output_dir.mkdir(exist_ok=True)

    # Step 1: 从 HF 下载现有数据
    print("\n📥 从 Hugging Face 下载最新数据...")
    print(f"   仓库: https://huggingface.co/datasets/{HF_REPO}")
    existing_results = download_from_hf("benchmark_results.json")

    # Step 2: 加载本地新结果
    print("\n📂 扫描本地 results/ 目录...")
    if not results_dir.exists():
        print("  ⚠️ results/ 目录不存在")
        print("  💡 请先运行 benchmark 生成结果")
        local_records: list[dict] = []
    else:
        local_records = load_local_results(results_dir)
        if not local_records:
            print("  ⚠️ 未找到任何 unified_results.jsonl 文件")
            print("  💡 请先运行 benchmark: python experiments/run_all.sh")
        else:
            print(f"  ✓ 找到 {len(local_records)} 条本地结果")

    # Step 3: 智能合并
    print("\n🔀 智能合并数据...")
    merged = merge_results(existing_results, local_records)

    # Step 4: 保存到 hf_data/
    print("\n💾 保存到 hf_data/ 目录...")
    output_file = hf_output_dir / "benchmark_results.json"
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, ensure_ascii=False)
    print(f"  ✓ {output_file.name} ({len(merged)} 条)")

    coverage = build_coverage_summary(merged)
    summary_file = hf_output_dir / "benchmark_summary.json"
    summary_file.write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("  📊 覆盖统计:")
    for dimension in ("by_workload", "by_backend", "by_seed"):
        print(f"    {dimension}: {coverage[dimension]}")
    print(f"  ✓ {summary_file.name}")

    print("\n" + "=" * 70)
    print("✅ 聚合完成！")
    print("=" * 70)
    print("\n📌 下一步操作：")
    print("  1. 提交聚合数据到 git:")
    print("     git add hf_data/")
    print("     git commit -m 'feat: add benchmark results'")
    print("     git push")
    print("\n  2. GitHub Actions 会自动:")
    print("     - 与 HF 最新数据合并（解决并发冲突）")
    print("     - 上传到 Hugging Face")
    print("     - 清理 hf_data/ 保持仓库轻量")
    print("\n💡 提示: results/ 目录不会被提交（在 .gitignore 中）")
    print("=" * 70)


if __name__ == "__main__":
    main()
