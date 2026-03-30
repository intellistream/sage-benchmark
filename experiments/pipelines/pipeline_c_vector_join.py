"""
Pipeline C: Cross-Source Vector Stream Join (跨源向量流相似度 Join)
===================================================================

拓扑: Source×3 → Map(Embedding)×3 → VectorJoin(Window+TopK) → Filter(ConflictDetect) → Sink

算子:
- Source×N: 多信息源数据加载 (News, Social, Official)
- Map (Embedding): 文本 → 向量
- VectorJoin: 跨流向量相似度 Join + 时间窗口 + TopK
- Filter: 冲突检测
- Sink: 结果输出

特点:
- 时间窗口: 在滑动/滚动窗口内进行跨流匹配
- 向量相似度: 基于 embedding 的语义相似度而非精确键匹配
- TopK 近邻: 找出最相似的 K 个跨源匹配对

注: 这是 SAGE 的向量流相似度 Join，与精确键匹配式 join 不同

数据集: ConflictQA / MemAgentBench
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# 禁用代理，确保内网服务可访问
os.environ.pop("http_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTPS_PROXY", None)

import httpx
from sage.foundation import (
    FilterFunction,
    SagePorts,
    MapFunction,
    SinkFunction,
    SourceFunction,
)

_DEFAULT_EMBEDDING_URL = f"http://localhost:{SagePorts.EMBEDDING_DEFAULT}/v1"
from sage.runtime import FluttyEnvironment as FlownetEnvironment

from ..common.execution_guard import run_pipeline_bounded
from .scheduler import HeadNodeScheduler


class VectorJoinStrategy(str, Enum):
    """向量 Join 策略"""

    IVF = "ivf"  # Inverted File Index - 快速近似匹配
    HNSW = "hnsw"  # Hierarchical Navigable Small World - 高精度匹配
    CLUSTERED = "clustered"  # Clustered Join - 批量窗口匹配


class WindowType(str, Enum):
    """时间窗口类型"""

    SLIDING = "sliding"
    TUMBLING = "tumbling"


@dataclass
class VectorJoinConfig:
    """Vector Join Pipeline 配置"""

    # 数据集
    num_samples: int = 100
    num_sources: int = 3

    # Embedding
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024

    # 时间窗口
    window_type: WindowType = WindowType.SLIDING
    window_size_ms: int = 5000
    window_slide_ms: int = 1000

    # 向量 Join
    join_strategy: VectorJoinStrategy = VectorJoinStrategy.HNSW
    similarity_threshold: float = 0.75
    topk: int = 5

    # 服务端点
    embedding_base_url: str = _DEFAULT_EMBEDDING_URL

    # 运行时
    job_manager_host: str = "localhost"
    job_manager_port: int = 19001
    request_timeout: float = 60.0


@dataclass
class VectorStreamItem:
    """向量流中的单个数据项"""

    item_id: str
    source_id: int
    source_name: str
    text: str
    embedding: list[float] = field(default_factory=list)
    timestamp_ms: int = 0
    window_id: int = 0


@dataclass
class MatchedPair:
    """匹配对"""

    item1: VectorStreamItem
    item2: VectorStreamItem
    similarity: float
    conflict_detected: bool = False


# ============================================================================
# Source: 多源数据加载
# ============================================================================


class MultiSourceFunction(SourceFunction):
    """多源 Source: 从 ConflictQA 数据集加载多源数据

    模拟三个信息源:
    - Source 0 (News): 新闻文本
    - Source 1 (Social): 社交媒体
    - Source 2 (Official): 官方声明
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_sources: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.num_sources = num_sources
        self._data: list[VectorStreamItem] = []
        self._index = 0
        self._loaded = False

    def _load_data(self) -> None:
        """加载数据集"""
        if self._loaded:
            return

        from sage.data.sources.memagentbench.conflict_resolution_loader import (
            ConflictResolutionDataLoader,
        )

        loader = ConflictResolutionDataLoader()
        raw_data = loader.load()

        source_names = ["news", "social", "official"]
        base_time = int(time.time() * 1000)

        # 为每个样本创建多个源的数据项
        for i, sample in enumerate(raw_data[: self.num_samples]):
            for source_id in range(self.num_sources):
                # 从样本中提取不同源的文本
                if source_id == 0:
                    text = sample.get("question", sample.get("query", ""))
                elif source_id == 1:
                    text = sample.get("context", sample.get("passage", ""))[:500]
                else:
                    text = sample.get("answer", sample.get("response", ""))

                if text:
                    self._data.append(
                        VectorStreamItem(
                            item_id=f"item_{i}_{source_id}",
                            source_id=source_id,
                            source_name=source_names[source_id],
                            text=text,
                            timestamp_ms=base_time + i * 100 + source_id * 10,
                        )
                    )

        self._loaded = True
        print(f"📂 Loaded {len(self._data)} items from {self.num_sources} sources")

    def execute(self, data: Any = None) -> Optional[VectorStreamItem]:
        """返回下一个数据项"""
        self._load_data()

        if self._index >= len(self._data):
            return None

        item = self._data[self._index]
        self._index += 1
        return item


# ============================================================================
# Map (Embedding): 向量化
# ============================================================================


class StreamEmbeddingMapFunction(MapFunction):
    """Map (Embedding): 对流数据进行向量化"""

    def __init__(
        self,
        embedding_base_url: str = _DEFAULT_EMBEDDING_URL,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        timeout: float = 60.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self.timeout = timeout

    def execute(self, item: VectorStreamItem) -> VectorStreamItem:
        """执行 embedding"""
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.embedding_base_url}/embeddings",
                json={"input": item.text[:1000], "model": self.embedding_model},
            )
            response.raise_for_status()
            result = response.json()

        item.embedding = result["data"][0]["embedding"]
        return item


# ============================================================================
# Map (VectorJoin): 向量相似度 Join
# ============================================================================


class VectorJoinMapFunction(MapFunction):
    """Map (VectorJoin): 跨源向量相似度 Join

    这个 Map 函数实现向量 Join 的核心逻辑：
    - 收集同一时间窗口内的所有向量
    - 计算跨源的向量相似度
    - 返回 TopK 匹配对

    注: 在真实 SageFlow 中，这会是一个专门的 Join 算子
    """

    def __init__(
        self,
        join_strategy: str = "hnsw",
        similarity_threshold: float = 0.75,
        topk: int = 5,
        window_size_ms: int = 5000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.join_strategy = join_strategy
        self.similarity_threshold = similarity_threshold
        self.topk = topk
        self.window_size_ms = window_size_ms

        # 缓存不同源的向量
        self._source_buffers: dict[int, list[VectorStreamItem]] = {}

    def execute(self, item: VectorStreamItem) -> Optional[list[MatchedPair]]:
        """执行向量 Join"""
        source_id = item.source_id

        # 添加到对应源的缓冲区
        if source_id not in self._source_buffers:
            self._source_buffers[source_id] = []
        self._source_buffers[source_id].append(item)

        # 计算当前窗口
        item.window_id = item.timestamp_ms // self.window_size_ms

        # 尝试与其他源的数据进行匹配
        matched_pairs = []

        for other_source_id, other_items in self._source_buffers.items():
            if other_source_id == source_id:
                continue

            for other_item in other_items:
                # 检查是否在同一时间窗口
                if abs(item.timestamp_ms - other_item.timestamp_ms) > self.window_size_ms:
                    continue

                # 计算相似度
                similarity = self._compute_similarity(item.embedding, other_item.embedding)

                if similarity >= self.similarity_threshold:
                    matched_pairs.append(
                        MatchedPair(
                            item1=item,
                            item2=other_item,
                            similarity=similarity,
                        )
                    )

        # 返回 TopK 匹配对
        if matched_pairs:
            matched_pairs.sort(key=lambda p: p.similarity, reverse=True)
            return matched_pairs[: self.topk]

        return None

    def _compute_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)


# ============================================================================
# Filter (ConflictDetect): 冲突检测
# ============================================================================


class ConflictDetectMapFunction(MapFunction):
    """Map (ConflictDetect): 检测跨源信息冲突并标记

    此 MapFunction 检测匹配对中的冲突并设置 conflict_detected 标志。
    """

    def __init__(self, conflict_threshold: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.conflict_threshold = conflict_threshold

    def execute(self, pairs: Optional[list[MatchedPair]]) -> Optional[list[MatchedPair]]:
        """检测冲突并标记"""
        if not pairs:
            return None

        # 检测语义冲突（简化版本：基于文本差异）
        for pair in pairs:
            text1_terms = set(pair.item1.text.lower().split())
            text2_terms = set(pair.item2.text.lower().split())

            # 计算 Jaccard 距离
            intersection = len(text1_terms & text2_terms)
            union = len(text1_terms | text2_terms)
            jaccard = intersection / union if union > 0 else 0

            # 如果语义相似但文本差异大，可能存在冲突
            if pair.similarity > 0.8 and jaccard < self.conflict_threshold:
                pair.conflict_detected = True

        return pairs


class HasMatchedPairsFilterFunction(FilterFunction):
    """过滤掉 None 或空的匹配结果

    FilterFunction.execute() 返回 bool，表示数据是否应该通过。
    """

    def execute(self, pairs: Optional[list[MatchedPair]]) -> bool:
        """过滤空结果"""
        return pairs is not None and len(pairs) > 0


# ============================================================================
# Sink: 结果输出
# ============================================================================


class VectorJoinSinkFunction(SinkFunction):
    """Vector Join Sink: 输出匹配结果"""

    def __init__(self, output_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.output_path = output_path
        self.results: list[dict] = []
        self.total_pairs = 0
        self.conflicts = 0

    def execute(self, pairs: Optional[list[MatchedPair]]) -> None:
        """输出匹配结果"""
        if not pairs:
            return

        for pair in pairs:
            self.total_pairs += 1
            if pair.conflict_detected:
                self.conflicts += 1

            result = {
                "item1_id": pair.item1.item_id,
                "item1_source": pair.item1.source_name,
                "item2_id": pair.item2.item_id,
                "item2_source": pair.item2.source_name,
                "similarity": pair.similarity,
                "conflict": pair.conflict_detected,
            }
            self.results.append(result)

            status = "⚠️ CONFLICT" if pair.conflict_detected else "✅ MATCH"
            print(
                f"{status} [{pair.item1.source_name}↔{pair.item2.source_name}] sim={pair.similarity:.3f}"
            )

        if self.output_path:
            import json

            with open(self.output_path, "a") as f:
                for r in self.results[-len(pairs) :]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ============================================================================
# Vector Join Pipeline 封装
# ============================================================================


class VectorJoinPipeline:
    """Vector Join Pipeline 封装类"""

    def __init__(self, config: VectorJoinConfig):
        self.config = config
        self.env: Optional[FlownetEnvironment] = None

    def build(self) -> FlownetEnvironment:
        """构建 Vector Join Pipeline"""
        scheduler = HeadNodeScheduler()

        self.env = FlownetEnvironment(
            "vector_join_pipeline",
            config={
                "flownet": {
                    "job_manager_host": self.config.job_manager_host,
                    "job_manager_port": self.config.job_manager_port,
                }
            },
            scheduler=scheduler,
        )

        # 构建 Pipeline: Source → Map(Embed) → Map(Join) → Map(Conflict) → Filter → Sink
        (
            self.env.from_source(
                MultiSourceFunction,
                num_samples=self.config.num_samples,
                num_sources=self.config.num_sources,
            )
            .map(
                StreamEmbeddingMapFunction,
                embedding_base_url=self.config.embedding_base_url,
                embedding_model=self.config.embedding_model,
                timeout=self.config.request_timeout,
            )
            .map(
                VectorJoinMapFunction,
                join_strategy=self.config.join_strategy.value,
                similarity_threshold=self.config.similarity_threshold,
                topk=self.config.topk,
                window_size_ms=self.config.window_size_ms,
            )
            .map(ConflictDetectMapFunction)
            .filter(HasMatchedPairsFilterFunction)
            .sink(VectorJoinSinkFunction)
        )

        return self.env

    def run(self) -> dict:
        """运行 Pipeline"""
        if self.env is None:
            self.build()

        start_time = time.time()
        try:
            run_pipeline_bounded(self.env, timeout_seconds=60.0, poll_interval_seconds=0.2)
        finally:
            self.env.close()

        duration = time.time() - start_time
        return {
            "pipeline": "C (VectorJoin)",
            "duration_seconds": duration,
            "config": {
                "num_samples": self.config.num_samples,
                "num_sources": self.config.num_sources,
                "join_strategy": self.config.join_strategy.value,
            },
        }
