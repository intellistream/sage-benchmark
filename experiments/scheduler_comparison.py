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
import time

from sage.common.core import MapFunction, SinkFunction, SourceFunction
from sage.kernel.api import FlownetEnvironment
from sage.kernel.api.local_environment import LocalEnvironment
from sage.kernel.scheduler.impl import FIFOScheduler, LoadAwareScheduler

from common.execution_guard import run_pipeline_bounded

# Register available backends (import triggers @register_runner decoration)
# Use direct 'backends.*' imports – experiments/ is in sys.path when this
# script is executed directly (Python adds the script's directory).
import backends.sage_runner  # noqa: F401  registers "sage"
from backends.base import WorkloadSpec, get_runner, list_backends


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


def run_with_scheduler(scheduler, env_class, scheduler_name):
    """使用指定调度器运行 pipeline"""
    print(f"\n{'=' * 60}")
    print(f"🚀 运行实验: {scheduler_name}")
    print(f"{'=' * 60}\n")

    env = None
    try:
        ResultSink.clear_all_results()

        # 创建环境并指定调度器
        if env_class == LocalEnvironment:
            env = LocalEnvironment(name=f"scheduler_test_{scheduler_name}", scheduler=scheduler)
        else:
            env = FlownetEnvironment(name=f"scheduler_test_{scheduler_name}", scheduler=scheduler)

        # 构建 pipeline
        # 注意：并行度在 operator 级别指定
        (
            env.from_source(DataSource, total_items=10)  # 减少到10个项目以加快测试
            .map(HeavyProcessor, parallelism=2)  # 资源密集型 operator，2 个并行实例
            .filter(LightFilter, parallelism=1)  # 轻量级 operator，1 个并行实例
            .sink(ResultSink)  # type: ignore[arg-type]  # Pass class, not instance
        )

        # 记录开始时间
        start_time = time.time()

        # 提交执行
        print(f"▶️  开始执行 pipeline (调度器: {scheduler_name})...\n")

        # 使用受控超时，避免执行卡住
        max_wait_time = 30  # 最大等待30秒
        try:
            guard_result = run_pipeline_bounded(
                env,
                timeout_seconds=max_wait_time,
                poll_interval_seconds=0.2,
            )

            if guard_result.timed_out:
                print(f"⚠️  {scheduler_name} 执行超时 ({max_wait_time}s)，已停止任务")

        except Exception as e:
            print(f"❌ {scheduler_name} 执行出错: {e}")
            # 不抛出异常，而是记录错误并继续

        # 记录结束时间
        end_time = time.time()
        elapsed = end_time - start_time

        # 获取调度器指标
        try:
            metrics = {}
            if (
                hasattr(env, "scheduler")
                and env.scheduler is not None
                and hasattr(env.scheduler, "get_metrics")
            ):
                metrics = env.scheduler.get_metrics()  # type: ignore[union-attr]
        except Exception as e:
            print(f"⚠️  无法获取调度器指标: {e}")
            metrics = {"error": str(e)}

        print(f"\n{'=' * 60}")
        print(f"📊 {scheduler_name} 执行结果")
        print(f"{'=' * 60}")
        print(f"总耗时: {elapsed:.2f} 秒")
        print(f"处理结果数: {ResultSink.result_count()}")
        print("调度器指标:")
        for key, value in metrics.items():
            print(f"  - {key}: {value}")
        print(f"{'=' * 60}\n")

        return {
            "scheduler": scheduler_name,
            "elapsed_time": elapsed,
            "metrics": metrics,
            "results_count": ResultSink.result_count(),
        }

    except Exception as e:
        print(f"❌ {scheduler_name} 运行失败: {e}")
        return {
            "scheduler": scheduler_name,
            "elapsed_time": 0,
            "metrics": {"error": str(e)},
            "results_count": 0,
        }
    finally:
        # 确保资源清理
        if env:
            try:
                if hasattr(env, "close"):
                    env.close()
                elif hasattr(env, "shutdown"):
                    env.shutdown()  # type: ignore[union-attr]
            except Exception:  # noqa: S110
                pass


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
        os.environ.get("SAGE_EXAMPLES_MODE") == "test"
        or os.environ.get("SAGE_TEST_MODE") == "true"
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

    results = [result]

    # ------------------------------------------------------------------
    # Legacy multi-scheduler comparison (SAGE default path, unchanged)
    # Only runs in non-test mode to keep CI fast.
    # ------------------------------------------------------------------
    if not test_mode and args.backend == "sage":
        time.sleep(1)
        print("\n🧪 实验 2: 负载感知调度器 (Local) – 对比组")
        result2 = run_with_scheduler(
            scheduler=LoadAwareScheduler(max_concurrent=10),
            env_class=LocalEnvironment,
            scheduler_name="LoadAware_Local",
        )
        results.append(result2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📈 调度器性能对比总结")
    print("=" * 80)

    for r in results:
        if hasattr(r, "summary"):
            # RunResult (new abstraction)
            print(f"\n[{r.backend}/{r.scheduler_name}]")
            print(r.summary())
        else:
            # legacy dict from run_with_scheduler
            print(f"\n{r['scheduler']}:")
            print(f"  总耗时: {r['elapsed_time']:.2f} 秒")
            print(f"  调度策略: {r['metrics'].get('scheduler_type', 'N/A')}")
            print(f"  已调度任务数: {r['metrics'].get('total_scheduled', 'N/A')}")
            if "avg_latency_ms" in r["metrics"]:
                print(f"  平均延迟: {r['metrics']['avg_latency_ms']:.2f} ms")

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
