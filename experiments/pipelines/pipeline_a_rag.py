"""
Pipeline A: RAG Pipeline (检索增强生成)
======================================

拓扑: Source → Map(Embedding) → Map(Retrieval) → Filter(Rerank) → Future(LLM) → Sink

算子:
- Source: 加载问答数据集 (qa_base/MMLU/BBH)
- Map (Embedding): 查询向量化
- Map (Retrieval): 向量检索 / BM25 / 混合检索
- Filter (Rerank): 重排序并过滤 top-k
- Future (LLM): 异步 LLM 调用生成答案
- Sink: 输出结果

数据集: qa_base, MMLU, BBH
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx
from sage.foundation import (
    FilterFunction,
    MapFunction,
    SagePorts,
    SinkFunction,
    SourceFunction,
)

_DEFAULT_EMBEDDING_URL = f"http://localhost:{SagePorts.EMBEDDING_DEFAULT}/v1"
_DEFAULT_LLM_URL = f"http://localhost:{SagePorts.LLM_DEFAULT}/v1"
from sage.runtime import FluttyEnvironment as FlownetEnvironment

try:
    from .scheduler import HeadNodeScheduler
except ImportError:
    # 直接运行脚本时使用绝对导入
    from experiments.pipelines.scheduler import (
        HeadNodeScheduler,
    )


class RetrievalMode(str, Enum):
    """检索模式"""

    DENSE = "dense"  # 向量检索
    SPARSE = "sparse"  # BM25
    HYBRID = "hybrid"  # 混合


@dataclass
class RAGConfig:
    """RAG Pipeline 配置"""

    # 数据集
    dataset_name: str = "qa_base"
    num_samples: int = 100

    # 检索
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = 5
    rerank_top_k: int = 3

    # 模型
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # 服务端点
    embedding_base_url: str = _DEFAULT_EMBEDDING_URL
    llm_base_url: str = _DEFAULT_LLM_URL

    # 运行时
    job_manager_host: str = "localhost"
    job_manager_port: int = 19001
    request_timeout: float = 60.0

    # 输出
    output_path: Optional[str] = None


@dataclass
class Document:
    """检索到的文档"""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)


# ============================================================================
# Source: 数据集加载
# ============================================================================


class RAGSourceFunction(SourceFunction):
    """RAG Source: 从数据集加载问答数据

    支持数据集:
    - qa_base: 内置问答数据集
    - mmlu: MMLU 多选题
    - bbh: BIG-Bench Hard
    """

    def __init__(
        self,
        dataset_name: str = "qa_base",
        num_samples: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self._data: list[dict] = []
        self._index = 0
        self._loaded = False

    def _load_data(self) -> None:
        """加载数据集"""
        if self._loaded:
            return

        # 动态导入数据加载器
        if self.dataset_name == "qa_base":
            from sage.data.sources.qa_base.dataloader import QADataLoader

            loader = QADataLoader()
            raw_data = loader.load_queries()  # 使用 load_queries() 方法
        elif self.dataset_name == "mmlu":
            from sage.data.sources.mmlu.dataloader import MMLUDataLoader

            loader = MMLUDataLoader()
            raw_data = loader.load()
        elif self.dataset_name == "bbh":
            from sage.data.sources.bbh.dataloader import BBHDataLoader

            loader = BBHDataLoader()
            raw_data = loader.load()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self._data = raw_data[: self.num_samples]
        self._loaded = True
        print(f"📂 Loaded {len(self._data)} samples from {self.dataset_name}")

    def execute(self, data: Any = None) -> Optional[dict]:
        """返回下一个问答样本"""
        self._load_data()

        if self._index >= len(self._data):
            return None

        sample = self._data[self._index]
        self._index += 1

        # 标准化输出格式
        return {
            "id": self._index,
            "query": sample.get("question", sample.get("query", "")),
            "context": sample.get("context", sample.get("passage", "")),
            "ground_truth": sample.get("answer", sample.get("answers", "")),
        }


# ============================================================================
# Map (Embedding): 查询向量化
# ============================================================================


class EmbeddingMapFunction(MapFunction):
    """Map (Embedding): 调用 embedding 服务将查询转为向量"""

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

    def execute(self, data: dict) -> dict:
        """执行 embedding"""
        query = data["query"]

        with httpx.Client(timeout=self.timeout, proxy=None) as client:
            response = client.post(
                f"{self.embedding_base_url}/embeddings",
                json={"input": query, "model": self.embedding_model},
            )
            response.raise_for_status()
            result = response.json()

        data["query_embedding"] = result["data"][0]["embedding"]
        return data


# ============================================================================
# Map (Retrieval): 检索文档
# ============================================================================


class RetrievalMapFunction(MapFunction):
    """Map (Retrieval): 根据模式检索文档"""

    def __init__(
        self,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        embedding_base_url: str = _DEFAULT_EMBEDDING_URL,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        timeout: float = 60.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.embedding_base_url = embedding_base_url
        self.embedding_model = embedding_model
        self.timeout = timeout

    def execute(self, data: dict) -> dict:
        """执行检索"""
        query = data["query"]
        query_vector = data.get("query_embedding", [])
        context = data.get("context", "")

        # 将 context 分块作为候选文档
        chunk_size = 500
        chunks = [context[i : i + chunk_size] for i in range(0, len(context), chunk_size)]
        if not chunks:
            chunks = [context] if context else ["No context available"]

        if self.retrieval_mode == "dense":
            docs = self._dense_retrieve(query_vector, chunks)
        elif self.retrieval_mode == "sparse":
            docs = self._sparse_retrieve(query, chunks)
        else:  # hybrid
            dense_docs = self._dense_retrieve(query_vector, chunks)
            sparse_docs = self._sparse_retrieve(query, chunks)
            docs = self._merge_results(dense_docs, sparse_docs)

        data["retrieved_docs"] = docs[: self.top_k]
        return data

    def _dense_retrieve(self, query_vector: list[float], chunks: list[str]) -> list[Document]:
        """向量检索"""
        if not query_vector:
            return [Document(text=c, score=0.0) for c in chunks]

        with httpx.Client(timeout=self.timeout, proxy=None) as client:
            response = client.post(
                f"{self.embedding_base_url}/embeddings",
                json={"input": chunks[:20], "model": self.embedding_model},
            )
            response.raise_for_status()
            result = response.json()

        docs = []
        for i, emb_data in enumerate(result["data"]):
            chunk_vector = emb_data["embedding"]
            similarity = self._cosine_similarity(query_vector, chunk_vector)
            docs.append(Document(text=chunks[i], score=similarity))

        docs.sort(key=lambda d: d.score, reverse=True)
        return docs

    def _sparse_retrieve(self, query: str, chunks: list[str]) -> list[Document]:
        """BM25 检索"""
        query_terms = set(query.lower().split())
        docs = []

        for chunk in chunks:
            chunk_terms = set(chunk.lower().split())
            overlap = len(query_terms & chunk_terms)
            score = overlap / (len(query_terms) + 1)
            docs.append(Document(text=chunk, score=score))

        docs.sort(key=lambda d: d.score, reverse=True)
        return docs

    def _merge_results(
        self, dense_docs: list[Document], sparse_docs: list[Document]
    ) -> list[Document]:
        """RRF 融合"""
        k = 60
        scores: dict[str, float] = {}

        for rank, doc in enumerate(dense_docs):
            scores[doc.text] = scores.get(doc.text, 0) + 1 / (k + rank + 1)
        for rank, doc in enumerate(sparse_docs):
            scores[doc.text] = scores.get(doc.text, 0) + 1 / (k + rank + 1)

        merged = [Document(text=t, score=s) for t, s in scores.items()]
        merged.sort(key=lambda d: d.score, reverse=True)
        return merged

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """计算余弦相似度"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


# ============================================================================
# Map (Rerank): 重排序
# ============================================================================


class RerankMapFunction(MapFunction):
    """Map (Rerank): 对检索结果重排序并选取 top-k

    这是一个 MapFunction 因为它转换数据（添加 reranked_docs 字段）。
    """

    def __init__(self, rerank_top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.rerank_top_k = rerank_top_k

    def execute(self, data: dict) -> dict:
        """执行重排序"""
        query = data["query"]
        docs = data.get("retrieved_docs", [])

        if not docs:
            data["reranked_docs"] = []
            return data

        query_terms = set(query.lower().split())

        for doc in docs:
            doc_terms = set(doc.text.lower().split())
            term_overlap = len(query_terms & doc_terms)
            doc.score = 0.7 * doc.score + 0.3 * (term_overlap / (len(query_terms) + 1))

        docs.sort(key=lambda d: d.score, reverse=True)
        data["reranked_docs"] = docs[: self.rerank_top_k]
        return data


class HasDocsFilterFunction(FilterFunction):
    """Filter: 过滤掉没有检索到文档的查询

    FilterFunction.execute() 返回 bool，表示数据是否应该通过。
    """

    def execute(self, data: dict) -> bool:
        """过滤空结果"""
        docs = data.get("reranked_docs", [])
        return len(docs) > 0


# ============================================================================
# Map (LLM Generate): 异步生成
# ============================================================================


class LLMGenerateMapFunction(MapFunction):
    """Map (LLM): 调用 LLM 生成答案"""

    def __init__(
        self,
        llm_base_url: str = _DEFAULT_LLM_URL,
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout: float = 60.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.timeout = timeout

    def execute(self, data: dict) -> dict:
        """执行 LLM 生成"""
        query = data["query"]
        docs = data.get("reranked_docs", [])

        # 构建 prompt
        context = "\n".join([doc.text for doc in docs]) if docs else "No context available"
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        with httpx.Client(timeout=self.timeout, proxy=None) as client:
            response = client.post(
                f"{self.llm_base_url}/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.7,
                },
            )
            response.raise_for_status()
            result = response.json()

        data["answer"] = result["choices"][0]["message"]["content"]
        return data


# ============================================================================
# Sink: 结果输出
# ============================================================================


class RAGSinkFunction(SinkFunction):
    """RAG Sink: 输出结果并收集指标"""

    def __init__(self, output_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.output_path = output_path
        self.results: list[dict] = []

    def execute(self, data: dict) -> None:
        """输出结果"""
        result = {
            "id": data.get("id"),
            "query": data.get("query"),
            "answer": data.get("answer"),
            "ground_truth": data.get("ground_truth"),
        }
        self.results.append(result)

        print(f"✅ [{result['id']}] Q: {result['query'][:50]}... → A: {result['answer'][:50]}...")

        if self.output_path:
            import json

            with open(self.output_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ============================================================================
# RAG Pipeline 封装
# ============================================================================


class RAGPipeline:
    """RAG Pipeline 封装类

    提供:
    - build(): 构建 Pipeline
    - run(): 运行 Pipeline
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.env: Optional[FlownetEnvironment] = None

    def build(self) -> FlownetEnvironment:
        """构建 RAG Pipeline"""
        scheduler = HeadNodeScheduler()

        self.env = FlownetEnvironment(
            "rag_pipeline",
            config={
                "flownet": {
                    "job_manager_host": self.config.job_manager_host,
                    "job_manager_port": self.config.job_manager_port,
                }
            },
            scheduler=scheduler,
        )

        # 构建 Pipeline: Source → Map → Map → Map(Rerank) → Filter → Map → Sink
        (
            self.env.from_source(
                RAGSourceFunction,
                dataset_name=self.config.dataset_name,
                num_samples=self.config.num_samples,
            )
            .map(
                EmbeddingMapFunction,
                embedding_base_url=self.config.embedding_base_url,
                embedding_model=self.config.embedding_model,
                timeout=self.config.request_timeout,
            )
            .map(
                RetrievalMapFunction,
                retrieval_mode=self.config.retrieval_mode.value,
                top_k=self.config.top_k,
                embedding_base_url=self.config.embedding_base_url,
                embedding_model=self.config.embedding_model,
                timeout=self.config.request_timeout,
            )
            .map(RerankMapFunction, rerank_top_k=self.config.rerank_top_k)
            .filter(HasDocsFilterFunction)
            .map(
                LLMGenerateMapFunction,
                llm_base_url=self.config.llm_base_url,
                llm_model=self.config.llm_model,
                timeout=self.config.request_timeout,
            )
            .sink(RAGSinkFunction, output_path=self.config.output_path)
        )

        return self.env

    def run(self) -> dict:
        """运行 Pipeline"""
        if self.env is None:
            self.build()

        start_time = time.time()
        try:
            # autostop=True 会等待 Pipeline 执行完成
            self.env.submit(autostop=True)
        finally:
            self.env.close()

        duration = time.time() - start_time
        return {
            "pipeline": "A (RAG)",
            "duration_seconds": duration,
            "config": {
                "dataset": self.config.dataset_name,
                "num_samples": self.config.num_samples,
                "retrieval_mode": self.config.retrieval_mode.value,
            },
        }


if __name__ == "__main__":
    # 直接运行时的入口
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("Pipeline A: RAG Pipeline")
    print("=" * 60)

    config = RAGConfig(
        num_samples=10,  # 测试用少量样本
        embedding_base_url="http://11.11.11.7:8090/v1",
        llm_base_url="http://11.11.11.7:8903/v1",
        output_path="/tmp/rag_pipeline_output.jsonl",
    )

    pipeline = RAGPipeline(config=config)
    result = pipeline.run()

    print(f"\nResult: {result}")
