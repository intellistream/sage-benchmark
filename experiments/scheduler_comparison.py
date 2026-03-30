#!/usr/bin/env python3
"""
调度器对比示例
演示如何使用不同的调度策略并对比性能指标

支持通过 --backend 选择运行后端（默认 sage），保持工作负载逻辑后端无关。

使用示例::

    python experiments/scheduler_comparison.py
    python experiments/scheduler_comparison.py --backend sage --scheduler fifo --items 10

@test:timeout=90
@test:category=scheduler
"""

import argparse
import sys
import time
import uuid
from importlib import metadata
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path guard – make experiments/ importable regardless of CWD.
# Python adds the script's parent directory only when executed directly.
# When the file is discovered via pytest / tox from the repo root, the
# experiments/ package root is absent from sys.path, which causes
#   ModuleNotFoundError: No module named 'backends'
# preventing backend registration and producing "Unknown backend" errors.
# ---------------------------------------------------------------------------
_EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

# Register available backends (import triggers @register_runner decoration)
import backends.sage_runner  # noqa: F401  registers "sage"
from backends.base import WorkloadSpec, get_runner, list_backends
from common.component_versions import collect_component_versions, resolve_first_installed_version
from common.metrics_schema import (
    UnifiedMetricsRecord,
    compute_backend_hash,
    compute_config_hash,
)
from common.result_writer import append_jsonl_record, export_jsonl_to_csv
from common.system_profile import collect_system_profile
from sage.foundation import MapFunction, SinkFunction, SourceFunction

try:
    from _version import __version__ as BENCHMARK_VERSION
except Exception:
    BENCHMARK_VERSION = "unknown"

try:
    SAGE_VERSION = resolve_first_installed_version(
        ["isage", "sage"],
        default="unknown",
    )
except Exception:
    SAGE_VERSION = "unknown"

try:
    SAGELLM_VERSION = resolve_first_installed_version(
        ["isagellm", "sagellm", "sagellm-gateway"],
        default="unknown",
    )
except Exception:
    SAGELLM_VERSION = "unknown"


class DataSource(SourceFunction):
    """简单的数据源，生成一批测试数据"""

    def __init__(self, total_items=20, **kwargs):
        super().__init__(**kwargs)
        self.total_items = total_items
        self.current = 0

    def execute(self, data=None):
        if self.current >= self.total_items:
            return None

        data = f"data_{self.current}"
        self.current += 1
        print(f"📤 Source: {data}")
        return data


class HeavyProcessor(MapFunction):
    """模拟资源密集型处理"""

    def execute(self, data):
        # 模拟耗时计算（减少到0.01秒以加快测试）
        time.sleep(0.01)
        result = f"processed_{data}"
        print(f"⚙️  HeavyProcessor: {data} -> {result}")
        return result


class LightFilter(MapFunction):
    """模拟轻量级过滤"""

    def execute(self, data):
        # 只保留偶数编号的数据
        item_id = int(data.split("_")[-1])
        if item_id % 2 == 0:
            print(f"✅ LightFilter: {data} passed")
            return data
        else:
            print(f"❌ LightFilter: {data} filtered")
            return None


class ResultSink(SinkFunction):
    """收集结果"""

    _all_results: list[str] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = []

    def execute(self, data):
        if data:
            self.results.append(data)
            ResultSink._all_results.append(data)
            print(f"💾 Sink: {data}")

    @classmethod
    def clear_all_results(cls):
        cls._all_results.clear()

    @classmethod
    def result_count(cls) -> int:
        return len(cls._all_results)


def build_unified_record(
    *,
    backend: str,
    scheduler_name: str,
    elapsed_time: float,
    results_count: int,
    raw_metrics: dict,
    workload: str,
    run_id: str,
    seed: int,
    nodes: int,
    parallelism: int,
    config_payload: dict,
) -> UnifiedMetricsRecord:
    """Build a unified metrics record from backend run output."""
    throughput = results_count / elapsed_time if elapsed_time > 0 else None
    component_versions = collect_component_versions()
    resolved_sage_version = (
        SAGE_VERSION if SAGE_VERSION != "unknown" else component_versions.get("isage", "unknown")
    )
    resolved_sagellm_version = (
        SAGELLM_VERSION
        if SAGELLM_VERSION != "unknown"
        else component_versions.get("isagellm", "unknown")
    )

    return UnifiedMetricsRecord(
        backend=backend,
        workload=workload,
        run_id=run_id,
        seed=seed,
        nodes=nodes,
        parallelism=parallelism,
        throughput=throughput,
        latency_p50=raw_metrics.get("latency_p50"),
        latency_p95=raw_metrics.get("latency_p95"),
        latency_p99=raw_metrics.get("latency_p99"),
        success_rate=raw_metrics.get("success_rate"),
        duration_seconds=elapsed_time,
        config_hash=compute_config_hash(config_payload),
        backend_hash=compute_backend_hash(backend),
        metadata={
            "scheduler_name": scheduler_name,
            "sage_version": resolved_sage_version,
            "sagellm_version": resolved_sagellm_version,
            "benchmark_version": BENCHMARK_VERSION,
            "model_name": None,
            "embedding_model_name": None,
            "system_profile": collect_system_profile(),
            "component_versions": component_versions,
            "results_count": results_count,
            "raw_metrics": raw_metrics,
        },
    )


def main():
    """主函数：对比不同调度策略（支持 --backend 选择运行后端）"""

    # ------------------------------------------------------------------
    # CLI argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="SAGE 调度器对比示例 – 支持多后端运行",
    )
    parser.add_argument(
        "--backend",
        default="sage",
        choices=list_backends() or ["sage"],
        help="选择运行后端（默认: sage）",
    )
    parser.add_argument(
        "--scheduler",
        default="fifo",
        choices=["fifo", "load_aware", "default"],
        help="调度策略（默认: fifo）",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=10,
        help="数据源产生的 item 数量（默认: 10）",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=2,
        help="处理算子的并行度（默认: 2）",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="用于记录对比元数据的节点数（默认: 1）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="用于记录对比元数据的随机种子（默认: 42）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/scheduler_comparison",
        help="统一结果输出目录（默认: results/scheduler_comparison）",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="可选运行 ID；为空时自动生成。",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="scheduler_comparison",
        help="工作负载标识（默认: scheduler_comparison）。",
    )
    args = parser.parse_args()

    print(
        """
╔══════════════════════════════════════════════════════════════╗
║           SAGE 调度器对比示例                                  ║
║  演示如何在 Environment 级别配置不同的调度策略                  ║
╚══════════════════════════════════════════════════════════════╝
    """
    )

    # ------------------------------------------------------------------
    # Backend-abstraction path (new)
    # Runs the workload through the selected backend via WorkloadRunner.
    # ------------------------------------------------------------------
    import os

    test_mode = (
        os.environ.get("SAGE_EXAMPLES_MODE") == "test" or os.environ.get("SAGE_TEST_MODE") == "true"
    )

    spec = WorkloadSpec(
        name="scheduler_demo",
        total_items=args.items,
        parallelism=args.parallelism,
        scheduler_name=args.scheduler,
    )

    print(f"\n🔧 后端: {args.backend} | 调度器: {args.scheduler} | items: {args.items}")
    print(f"   可用后端: {', '.join(list_backends())}\n")

    runner = get_runner(args.backend)
    result = runner.run(spec)

    print(f"\n{'=' * 60}")
    print(f"📊 运行结果 ({args.backend})")
    print(f"{'=' * 60}")
    print(result.summary())
    print(f"{'=' * 60}\n")

    config_payload = {
        "backend": args.backend,
        "scheduler": args.scheduler,
        "items": args.items,
        "parallelism": args.parallelism,
        "nodes": args.nodes,
        "seed": args.seed,
        "workload": args.workload,
    }
    run_id = args.run_id.strip() or f"{args.backend}-{uuid.uuid4().hex[:12]}"
    unified_record = build_unified_record(
        backend=result.backend,
        scheduler_name=result.scheduler_name,
        elapsed_time=result.elapsed_time,
        results_count=result.results_count,
        raw_metrics=result.metrics,
        workload=args.workload,
        run_id=run_id,
        seed=args.seed,
        nodes=args.nodes,
        parallelism=args.parallelism,
        config_payload=config_payload,
    )

    output_dir = Path(args.output_dir)
    jsonl_path = output_dir / "unified_results.jsonl"
    csv_path = output_dir / "unified_results.csv"
    append_jsonl_record(jsonl_path, unified_record.to_dict())
    export_jsonl_to_csv(jsonl_path, csv_path)
    print(f"📦 Unified JSONL: {jsonl_path}")
    print(f"📦 Unified CSV:   {csv_path}\n")

    results = [result]

    # ------------------------------------------------------------------
    # Multi-scheduler comparison: complementary scheduler via the same
    # WorkloadRunner abstraction – no new daemon is spawned, no port fight.
    # Skipped when the user already chose load_aware, or in test mode.
    # ------------------------------------------------------------------
    if not test_mode and args.backend == "sage" and args.scheduler != "load_aware":
        time.sleep(0.5)
        print("\n🧪 实验 2: 负载感知调度器 – 对比组")
        spec2 = WorkloadSpec(
            name="scheduler_demo_load_aware",
            total_items=args.items,
            parallelism=args.parallelism,
            scheduler_name="load_aware",
        )
        result2 = runner.run(spec2)
        results.append(result2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📈 调度器性能对比总结")
    print("=" * 80)

    for r in results:
        print(f"\n[{r.backend}/{r.scheduler_name}]")
        print(r.summary())

    print("\n" + "=" * 80)
    print("✅ 所有实验完成！")
    print("=" * 80)

    print(
        """
💡 关键要点：
  1. 通过 --backend 选择运行后端，工作负载逻辑无需修改
     - python scheduler_comparison.py --backend sage
     - python scheduler_comparison.py --backend ray   (需安装 ray_runner)

  2. 用户在创建 Environment 时指定调度策略
     - env = LocalEnvironment(scheduler="fifo")
     - env = FlownetEnvironment(scheduler=LoadAwareScheduler())

  3. 并行度在定义 transformation 时指定
     - .map(HeavyProcessor, parallelism=4)
     - .filter(LightFilter, parallelism=2)
    """
    )


if __name__ == "__main__":
    main()
