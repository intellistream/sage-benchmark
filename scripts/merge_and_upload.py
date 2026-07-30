#!/usr/bin/env python3
"""
并发安全的合并脚本（由 GitHub Actions 调用）

用于在上传到 HF 前再次合并最新数据，解决多用户同时提交的并发冲突。

工作流程：
1. 读取用户提交的 hf_data/（可能基于旧版本 HF 数据）
2. 从 HF 下载最新数据（可能已被其他用户更新）
3. 三方智能合并（以 HF 最新版本为权威基准）
4. 保存合并结果（供 upload_to_hf.py 使用）
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


def normalize_workload_name(value: object) -> object:
    if isinstance(value, str) and value.lower() in Q_MATRIX_WORKLOADS:
        return value.lower()
    return value


def download_from_hf(filename: str) -> list[dict]:
    """从 HF 下载最新数据（公开，无需 token）。"""
    mirror = "https://hf-mirror.com"
    url = f"{mirror}/datasets/{HF_REPO}/resolve/{HF_BRANCH}/{filename}"
    print(f"  📥 {url}")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            print(f"    ✓ {len(data)} 条记录")
            return data
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("    ⚠️ 文件不存在（首次上传）")
        else:
            # 回退到主站
            alt = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_BRANCH}/{filename}"
            try:
                with urllib.request.urlopen(alt, timeout=30) as r2:
                    data = json.loads(r2.read().decode("utf-8"))
                    print(f"    ✓ {len(data)} 条记录（主站）")
                    return data
            except Exception as e2:
                print(f"    ⚠️ 主站失败: {e2}")
        return []
    except Exception as e:
        print(f"    ⚠️ 下载失败: {e}")
        return []


def get_config_key(entry: dict) -> str:
    """生成配置唯一标识 key。"""
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
    """判断新结果是否优于现有结果。"""
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


def smart_merge(hf_latest: list[dict], user_data: list[dict]) -> list[dict]:
    """三方智能合并。

    关键规则：
    1. HF 最新数据为基准（权威版本）
    2. 用户数据追加或更新
    3. 相同配置时，选择性能更好的
    4. 不同配置则追加

    这样即使用户基于旧版本 HF 数据合并，也能与最新版本正确合并。
    """
    merged: dict[str, dict] = {}

    # 先加入 HF 最新数据（权威版本）
    for entry in hf_latest:
        key = get_config_key(entry)
        merged[key] = entry

    added = updated = skipped = 0

    # 合并用户数据
    for entry in user_data:
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
            print(f"    ○ 跳过 (已有更好): {key[:60]}...")

    print(f"  📊 合并结果: 新增 {added}, 更新 {updated}, 跳过 {skipped}, 总计 {len(merged)}")
    return list(merged.values())


def build_coverage_summary(records: list[dict]) -> dict:
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
    print("=" * 60)
    print("🔀 并发安全合并（GitHub Actions）")
    print("=" * 60)

    hf_data_dir = Path("hf_data")

    if not hf_data_dir.exists():
        print("\n❌ hf_data/ 目录不存在")
        print("💡 用户应该先运行 'python scripts/aggregate_for_hf.py'")
        raise SystemExit(1)

    # 1. 读取用户提交的数据
    print("\n📂 读取用户提交的数据...")
    user_file = hf_data_dir / "benchmark_results.json"

    if not user_file.exists():
        print(f"  ⚠️ 缺少必要文件: {user_file}")
        raise SystemExit(1)

    user_data: list[dict] = json.loads(user_file.read_text(encoding="utf-8"))
    print(f"  ✓ {len(user_data)} 条")

    # 2. 从 HF 下载最新数据（可能已被其他用户更新）
    print("\n📥 从 Hugging Face 下载最新数据...")
    hf_latest = download_from_hf("benchmark_results.json")

    # 3. 智能合并
    print("\n🔀 智能合并（解决并发冲突）...")
    merged = smart_merge(hf_latest, user_data)

    # 4. 保存合并结果（覆盖用户提交的版本，供 upload_to_hf.py 使用）
    print("\n💾 保存合并结果...")
    user_file.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓ {user_file} ({len(merged)} 条)")
    summary_file = hf_data_dir / "benchmark_summary.json"
    summary_file.write_text(
        json.dumps(build_coverage_summary(merged), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓ {summary_file}")

    print("\n✅ 并发安全合并完成！")
    print("💡 下一步: 运行 upload_to_hf.py 上传到 Hugging Face")


if __name__ == "__main__":
    main()
