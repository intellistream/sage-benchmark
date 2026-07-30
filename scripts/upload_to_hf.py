#!/usr/bin/env python3
"""
上传聚合后的 benchmark 结果到 Hugging Face Datasets Hub（由 GitHub Actions 调用）

运行方式（本地手动上传）：
    HF_TOKEN=hf_xxx python scripts/upload_to_hf.py

环境变量：
    HF_TOKEN      - Hugging Face API token（可选，优先使用）
    HF_ENDPOINT   - HF endpoint，默认 https://huggingface.co
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# 配置
HF_REPO = "intellistream/sage-benchmark-results"
HF_DATA_DIR = Path(__file__).parent.parent / "hf_data"


def ensure_repo_exists(api, repo_id: str) -> None:
    """确保 HF dataset repo 存在，不存在则创建。"""
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"✓ Repo exists: {repo_id}")
    except Exception:
        print(f"📦 Creating repo: {repo_id}")
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
        print(f"✓ Created: {repo_id}")


def upload_files(api, repo_id: str, files: list[Path]) -> None:
    """上传文件列表到 HF。"""
    for local_path in files:
        if not local_path.exists():
            print(f"⚠️  File not found: {local_path}")
            continue

        remote_path = local_path.name
        print(f"📤 Uploading: {local_path.name} -> {remote_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update {remote_path} via CI - {datetime.now().isoformat()}",
        )
        print(f"✓ Uploaded: {remote_path}")


def main() -> None:
    # 读取 token（可选）
    token = os.environ.get("HF_TOKEN")

    # 写入默认使用官方 endpoint，避免 mirror 写入权限问题
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"📡 Using HF endpoint: {hf_endpoint}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("❌ 请先安装: pip install huggingface_hub")
        sys.exit(1)

    os.environ["HF_ENDPOINT"] = hf_endpoint

    # 如果提供了 HF_TOKEN，先验证；无效则回退到本机登录态
    resolved_token = token
    if resolved_token:
        try:
            HfApi(endpoint=hf_endpoint, token=resolved_token).whoami()
            print("✓ HF_TOKEN is valid")
        except Exception as exc:
            print(f"⚠️  Provided HF_TOKEN is invalid ({exc}); falling back to local HF auth")
            resolved_token = None
    else:
        print("ℹ️  HF_TOKEN not set; using local HF auth if available")

    api = HfApi(endpoint=hf_endpoint, token=resolved_token)

    # 提前验证鉴权
    try:
        who = api.whoami()
        user_name = who.get("name") if isinstance(who, dict) else None
        print(f"✓ Authenticated as: {user_name or 'unknown'}")
    except Exception:
        print("❌ No valid Hugging Face authentication found.")
        print("\n可选修复方式：")
        print("  1) export HF_TOKEN=hf_xxx")
        print("  2) hf auth login")
        sys.exit(1)

    # 确保 repo 存在
    ensure_repo_exists(api, HF_REPO)

    # 要上传的文件
    files_to_upload = [
        HF_DATA_DIR / "benchmark_results.json",
        HF_DATA_DIR / "benchmark_summary.json",
    ]

    if not HF_DATA_DIR.exists():
        print(f"❌ hf_data 目录不存在: {HF_DATA_DIR}")
        print("💡 请先运行 scripts/aggregate_for_hf.py")
        sys.exit(1)

    print(f"\n📂 Uploading to: {HF_REPO}")
    upload_files(api, HF_REPO, files_to_upload)

    print("\n✅ Upload complete!")
    print(f"🔗 查看: https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
