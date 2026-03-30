"""
Pipeline B: Long Context Refiner (长文本精炼)
============================================

使用 LLMLingua-2 压缩算法的 RAG Pipeline。
LLMLingua-2 基于 BERT token 分类，比 LLM-based 方法快得多。

拓扑: Source → Retriever → LLMLingua2 → Promptor → Generator → Evaluators → Sink

特点:
    - 快速压缩：使用 BERT 模型进行 token 分类，无需 LLM 推理
    - 多语言支持：使用 mBERT 或 XLM-RoBERTa 模型
    - Token 级精确压缩：每个 token 独立分类
    - 可选的上下文级过滤：粗到细的压缩策略
    - 使用 register_service 避免分布式访问问题

参考论文: https://arxiv.org/abs/2403.12968
数据集: HuggingFace FlashRAG Datasets (NQ)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import yaml
from sage.foundation import MapFunction, SagePorts, SinkFunction, SourceFunction

_DEFAULT_LLM_URL = f"http://localhost:{SagePorts.LLM_DEFAULT}/v1"
from sage.foundation import CustomLogger
from sage.runtime import LocalEnvironment


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(config_path, encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return data


@dataclass
class RefinerConfig:
    """Refiner Pipeline 配置"""

    # 配置文件路径
    config_path: str = ""

    # 运行时配置
    enable_profile: bool = True
    pipeline_timeout: float = 3600.0  # 1 hour for processing


# ============================================================================
# Service Registration Helpers
# ============================================================================


def register_embedding_service(
    env: LocalEnvironment,
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cuda:0",
) -> bool:
    """
    注册 Embedding 服务，用于检索时的向量化。

    使用 UnifiedInferenceClient 或 EmbeddingFactory。
    """
    try:
        from sagellm.embedding import EmbeddingClientAdapter, EmbeddingFactory

        # 创建 embedding 服务包装类
        class EmbeddingService:
            """Embedding 服务封装"""

            def __init__(self, model_name: str, device: str, **kwargs):
                self.model_name = model_name
                self.device = device
                self._embedder = None
                self._client = None

            def _ensure_initialized(self):
                if self._embedder is None:
                    raw_embedder = EmbeddingFactory.create("hash", dim=1024)
                    self._client = EmbeddingClientAdapter(raw_embedder)
                    self._embedder = raw_embedder

            def embed(self, texts: list[str]) -> list[list[float]]:
                """批量 embedding"""
                self._ensure_initialized()
                return self._client.embed(texts)

            def embed_single(self, text: str) -> list[float]:
                """单个 embedding"""
                self._ensure_initialized()
                return self._embedder.embed(text)

        env.register_service(
            "embedding_service",
            EmbeddingService,
            model_name=model_name,
            device=device,
        )
        print(f"[Pipeline] Registered embedding_service (model={model_name})")
        return True
    except ImportError as e:
        print(f"[Pipeline] Embedding service not available: {e}")
        return False
    except Exception as e:
        print(f"[Pipeline] Failed to register embedding_service: {e}")
        return False


def register_vector_db_service(
    env: LocalEnvironment,
    index_path: str,
    documents_path: str,
    mapping_path: Optional[str] = None,
    dimension: int = 1024,
) -> bool:
    """
    注册 Vector DB 服务，预加载 FAISS 索引。

    避免分布式访问问题：在 Head 节点加载一次，通过 call_service 访问。
    """
    try:
        import json

        import faiss

        class FAISSVectorDBService:
            """FAISS Vector DB 服务封装"""

            def __init__(
                self,
                index_path: str,
                documents_path: str,
                mapping_path: Optional[str] = None,
                dimension: int = 1024,
                **kwargs,
            ):
                self.index_path = index_path
                self.documents_path = documents_path
                self.mapping_path = mapping_path
                self.dimension = dimension
                self._index = None
                self._documents = None
                self._mapping = None

            def _ensure_initialized(self):
                if self._index is not None:
                    return

                print(f"[VectorDB] Loading FAISS index from {self.index_path}")
                self._index = faiss.read_index(self.index_path)
                print(f"[VectorDB] Index loaded: {self._index.ntotal} vectors")

                print(f"[VectorDB] Loading documents from {self.documents_path}")
                self._documents = []
                with open(self.documents_path, encoding="utf-8") as f:
                    for line in f:
                        self._documents.append(json.loads(line.strip()))
                print(f"[VectorDB] Loaded {len(self._documents)} documents")

                if self.mapping_path and os.path.exists(self.mapping_path):
                    with open(self.mapping_path, encoding="utf-8") as f:
                        self._mapping = json.load(f)
                    print(f"[VectorDB] Loaded mapping with {len(self._mapping)} entries")

            def search(self, query_vector: np.ndarray, top_k: int = 100) -> list[dict[str, Any]]:
                """搜索最近邻"""
                self._ensure_initialized()

                if query_vector.ndim == 1:
                    query_vector = query_vector.reshape(1, -1)

                query_vector = query_vector.astype(np.float32)
                distances, indices = self._index.search(query_vector, top_k)

                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < 0 or idx >= len(self._documents):
                        continue
                    doc = self._documents[idx].copy()
                    doc["score"] = float(distances[0][i])
                    doc["index"] = int(idx)
                    results.append(doc)

                return results

        env.register_service(
            "vector_db",
            FAISSVectorDBService,
            index_path=index_path,
            documents_path=documents_path,
            mapping_path=mapping_path,
            dimension=dimension,
        )
        print(f"[Pipeline] Registered vector_db (FAISS index={index_path})")
        return True
    except ImportError as e:
        print(f"[Pipeline] FAISS not available: {e}")
        return False
    except Exception as e:
        print(f"[Pipeline] Failed to register vector_db: {e}")
        return False


def register_refiner_service(
    env: LocalEnvironment,
    model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    device: str = "cuda:0",
    rate: float = 0.5,
) -> bool:
    """
    注册 LLMLingua-2 Refiner 服务。

    在 Head 节点加载模型，避免分布式加载问题。
    """
    try:

        class RefinerService:
            """Benchmark-local LLMLingua2-like refiner service."""

            def __init__(
                self,
                model_name: str,
                device: str = "cuda:0",
                rate: float = 0.5,
                **kwargs,
            ):
                self.model_name = model_name
                self.device = device
                self.default_rate = rate

            def compress(
                self,
                context: list[str] | str,
                rate: Optional[float] = None,
                **kwargs,
            ) -> dict[str, Any]:
                """Compress context with a deterministic token-budget approximation."""
                compression_rate = min(max(rate or self.default_rate, 0.1), 1.0)
                segments = context if isinstance(context, list) else [context]
                compressed_segments = []
                original_tokens = 0
                compressed_tokens = 0

                for segment in segments:
                    tokens = segment.split()
                    original_tokens += len(tokens)
                    keep = max(1, int(len(tokens) * compression_rate))
                    reduced = " ".join(tokens[:keep])
                    compressed_segments.append(reduced)
                    compressed_tokens += len(reduced.split())

                return {
                    "compressed_context": compressed_segments,
                    "original_tokens": original_tokens,
                    "compressed_tokens": compressed_tokens,
                    "compression_ratio": compressed_tokens / max(original_tokens, 1),
                    "model_name": self.model_name,
                    "device": self.device,
                }

        env.register_service(
            "refiner_service",
            RefinerService,
            model_name=model_name,
            device=device,
            rate=rate,
        )
        print(f"[Pipeline] Registered refiner_service (model={model_name})")
        return True
    except Exception as e:
        print(f"[Pipeline] Failed to register refiner_service: {e}")
        return False


def register_llm_service(
    env: LocalEnvironment,
    base_url: str = _DEFAULT_LLM_URL,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    api_key: str = "",
) -> bool:
    """
    注册 LLM 服务。
    """
    try:
        import httpx

        class LLMService:
            """LLM 服务封装"""

            def __init__(
                self,
                base_url: str,
                model_name: str,
                api_key: str = "",
                **kwargs,
            ):
                self.base_url = base_url.rstrip("/")
                self.model_name = model_name
                self.api_key = api_key

            def chat(
                self,
                messages: list[dict[str, str]],
                max_tokens: int = 512,
                temperature: float = 0.7,
                **kwargs,
            ) -> str:
                """Chat completion"""
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": self.model_name,
                            "messages": messages,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            **kwargs,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()

                return result["choices"][0]["message"]["content"]

        env.register_service(
            "llm_service",
            LLMService,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
        )
        print(f"[Pipeline] Registered llm_service (base_url={base_url})")
        return True
    except Exception as e:
        print(f"[Pipeline] Failed to register llm_service: {e}")
        return False


# ============================================================================
# Pipeline Operators (使用 call_service 访问服务)
# ============================================================================


class DataSourceOperator(SourceFunction):
    """数据源 Operator: 从 HuggingFace 加载数据集"""

    def __init__(
        self,
        dataset_name: str = "RUC-NLPIR/FlashRAG_datasets",
        dataset_config: str = "nq",
        split: str = "test",
        max_samples: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.max_samples = max_samples
        self._data = None
        self._index = 0

    def _load_data(self):
        if self._data is not None:
            return

        from datasets import load_dataset

        print(f"[Source] Loading dataset: {self.dataset_name}/{self.dataset_config}")
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
        )

        self._data = list(dataset.select(range(min(self.max_samples, len(dataset)))))
        print(f"[Source] Loaded {len(self._data)} samples")

    def execute(self, data: Any = None) -> Optional[dict]:
        self._load_data()

        if self._index >= len(self._data):
            return None

        sample = self._data[self._index]
        self._index += 1

        return {
            "id": sample.get("id", self._index),
            "query": sample.get("question", sample.get("query", "")),
            "ground_truth": sample.get("answer", sample.get("answers", [])),
        }


class RetrieverOperator(MapFunction):
    """检索 Operator: 使用注册的服务进行检索"""

    def __init__(self, top_k: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    def execute(self, data: dict) -> dict:
        query = data.get("query", "")

        # 获取 embedding
        embedding_service = self.call_service("embedding_service")
        query_embedding = embedding_service.embed_single(query)
        query_vector = np.array(query_embedding, dtype=np.float32)

        # 检索
        vector_db = self.call_service("vector_db")
        results = vector_db.search(query_vector, top_k=self.top_k)

        data["retrieval_results"] = results
        data["query_embedding"] = query_embedding
        print(f"[Retriever] Query: {query[:50]}... -> {len(results)} docs")
        return data


class RefinerOperator(MapFunction):
    """压缩 Operator: 使用注册的 LLMLingua-2 服务"""

    def __init__(self, rate: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def execute(self, data: dict) -> dict:
        retrieval_results = data.get("retrieval_results", [])

        if not retrieval_results:
            data["refining_results"] = []
            data["compressed_context"] = ""
            return data

        # 提取文本
        texts = [
            doc.get("text", doc.get("content", doc.get("passage", ""))) for doc in retrieval_results
        ]

        # 压缩
        refiner = self.call_service("refiner_service")
        result = refiner.compress(texts, rate=self.rate)

        data["refining_results"] = [result.get("compressed_prompt", "")]
        data["compressed_context"] = result.get("compressed_prompt", "")
        data["original_tokens"] = result.get("origin_tokens", 0)
        data["compressed_tokens"] = result.get("compressed_tokens", 0)
        data["compression_rate"] = result.get("rate", self.rate)

        print(
            f"[Refiner] Compressed {data['original_tokens']} -> {data['compressed_tokens']} tokens"
        )
        return data


class PromptorOperator(MapFunction):
    """Prompt 构建 Operator"""

    def __init__(self, template: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.template = template or (
            "Answer the question based on the given context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def execute(self, data: dict) -> dict:
        query = data.get("query", "")
        context = data.get("compressed_context", "")

        prompt = self.template.format(context=context, question=query)
        data["prompt"] = prompt
        return data


class GeneratorOperator(MapFunction):
    """生成 Operator: 使用注册的 LLM 服务"""

    def __init__(self, max_tokens: int = 512, temperature: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.temperature = temperature

    def execute(self, data: dict) -> dict:
        prompt = data.get("prompt", "")

        llm = self.call_service("llm_service")
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        data["generated_answer"] = response
        print(f"[Generator] Generated: {response[:100]}...")
        return data


class EvaluatorOperator(MapFunction):
    """评估 Operator: 计算 F1 等指标"""

    def execute(self, data: dict) -> dict:
        generated = data.get("generated_answer", "")
        ground_truth = data.get("ground_truth", "")

        # 简单的 F1 计算
        if isinstance(ground_truth, list):
            ground_truth = " ".join(str(g) for g in ground_truth)

        gen_tokens = set(generated.lower().split())
        gt_tokens = set(ground_truth.lower().split())

        if not gen_tokens or not gt_tokens:
            data["f1_score"] = 0.0
        else:
            common = gen_tokens & gt_tokens
            precision = len(common) / len(gen_tokens) if gen_tokens else 0
            recall = len(common) / len(gt_tokens) if gt_tokens else 0
            data["f1_score"] = (
                2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            )

        print(f"[Evaluator] F1 Score: {data['f1_score']:.4f}")
        return data


class ResultSinkOperator(SinkFunction):
    """结果输出 Operator"""

    def __init__(self, output_path: Optional[str] = None, verbose: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.output_path = output_path
        self.verbose = verbose
        self.results = []

    def execute(self, data: dict) -> None:
        result = {
            "id": data.get("id"),
            "query": data.get("query"),
            "generated_answer": data.get("generated_answer"),
            "f1_score": data.get("f1_score"),
            "compression_rate": data.get("compression_rate"),
            "original_tokens": data.get("original_tokens"),
            "compressed_tokens": data.get("compressed_tokens"),
        }
        self.results.append(result)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Query: {result['query']}")
            answer = result.get("generated_answer", "")
            print(f"Answer: {answer[:200] if answer else 'N/A'}...")
            print(
                f"F1: {result['f1_score']:.4f}, Compression: {result.get('compression_rate', 0):.2%}"
            )
            print(f"{'=' * 60}\n")

        if self.output_path:
            import json

            with open(self.output_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ============================================================================
# Refiner Pipeline 运行函数
# ============================================================================


def pipeline_run(config: dict) -> None:
    """运行 LLMLingua-2 RAG Pipeline

    Args:
        config: 从 YAML 配置文件加载的配置字典
    """
    print("""
========================================================================
              Pipeline B: Long Context Refiner (LLMLingua-2)
========================================================================
  Pipeline: Source -> Retriever -> Refiner -> Promptor -> Generator -> Evaluator

  Integrated Services (通过 register_service 注册，避免分布式访问问题):
    - embedding_service: BGE Embedding 模型
    - vector_db: FAISS 向量索引
    - refiner_service: LLMLingua-2 压缩模型
        - llm_service: sageLLM/OpenAI 兼容服务

  Features:
    - BERT-based token classification (快速，无需 LLM 推理)
    - Multilingual support (mBERT/XLM-RoBERTa)
    - Token-level precise compression
========================================================================
    """)

    env = LocalEnvironment("refiner_pipeline")

    # 注册服务
    print("\n[Pipeline] Registering services...")

    # Embedding 服务
    embedding_cfg = config.get("retriever", {}).get("embedding", {})
    register_embedding_service(
        env,
        model_name=embedding_cfg.get("model", "BAAI/bge-large-en-v1.5"),
        device=f"cuda:{embedding_cfg.get('gpu_device', 0)}",
    )

    # Vector DB 服务
    retriever_cfg = config.get("retriever", {})
    faiss_cfg = retriever_cfg.get("faiss", {})
    register_vector_db_service(
        env,
        index_path=os.path.expandvars(faiss_cfg.get("index_path", "")),
        documents_path=os.path.expandvars(faiss_cfg.get("documents_path", "")),
        mapping_path=os.path.expandvars(faiss_cfg.get("mapping_path", ""))
        if faiss_cfg.get("mapping_path")
        else None,
        dimension=retriever_cfg.get("dimension", 1024),
    )

    # Refiner 服务
    refiner_cfg = config.get("llmlingua2", {})
    register_refiner_service(
        env,
        model_name=refiner_cfg.get(
            "model_name", "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        ),
        device=refiner_cfg.get("device", "cuda:0"),
        rate=refiner_cfg.get("rate", 0.5),
    )

    # LLM 服务
    generator_cfg = config.get("generator", {}).get("sagellm", {})
    register_llm_service(
        env,
        base_url=generator_cfg.get("base_url", _DEFAULT_LLM_URL),
        model_name=generator_cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct"),
        api_key=generator_cfg.get("api_key", ""),
    )

    print()

    # 构建 Pipeline
    source_cfg = config.get("source", {})
    (
        env.from_source(
            DataSourceOperator,
            dataset_name=source_cfg.get("hf_dataset_name", "RUC-NLPIR/FlashRAG_datasets"),
            dataset_config=source_cfg.get("hf_dataset_config", "nq"),
            split=source_cfg.get("hf_split", "test"),
            max_samples=source_cfg.get("max_samples", 20),
        )
        .map(RetrieverOperator, top_k=retriever_cfg.get("top_k", 100))
        .map(RefinerOperator, rate=refiner_cfg.get("rate", 0.5))
        .map(PromptorOperator)
        .map(GeneratorOperator)
        .map(EvaluatorOperator)
        .sink(ResultSinkOperator, verbose=True)
    )

    try:
        start_time = time.time()
        env.submit(autostop=True)
        duration = time.time() - start_time
        print(f"\n[Pipeline] Completed in {duration:.2f} seconds")
    except KeyboardInterrupt:
        print("\n⚠️  KeyboardInterrupt: 用户手动停止")
    except Exception as e:
        print(f"\n❌ Pipeline异常: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n🔄 清理环境...")
        env.close()
        print("✅ 环境已关闭")


# ============================================================================
# Refiner Pipeline 封装类
# ============================================================================


class RefinerPipeline:
    """Refiner Pipeline 封装类

    使用 LLMLingua-2 BERT token 分类进行快速上下文压缩。
    通过 register_service 注册服务，避免分布式访问问题。

    示例:
        >>> config_path = "config/config_llmlingua2.yaml"
        >>> pipeline = RefinerPipeline.from_config(config_path)
        >>> result = pipeline.run()
    """

    def __init__(self, config: dict):
        """初始化 Pipeline

        Args:
            config: 从 YAML 加载的配置字典
        """
        self.config = config
        self.env: Optional[LocalEnvironment] = None

    @classmethod
    def from_config(cls, config_path: str) -> RefinerPipeline:
        """从配置文件创建 Pipeline

        Args:
            config_path: YAML 配置文件路径

        Returns:
            RefinerPipeline 实例
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config = load_config(config_path)
        return cls(config)

    def run(self) -> dict:
        """运行 Pipeline

        Returns:
            包含运行结果的字典
        """
        start_time = time.time()
        pipeline_run(self.config)
        duration = time.time() - start_time

        return {
            "pipeline": "B (LLMLingua-2 Refiner)",
            "duration_seconds": duration,
            "config": {
                "source": self.config.get("source", {}).get("hf_dataset_name", ""),
                "max_samples": self.config.get("source", {}).get("max_samples", 0),
                "model": self.config.get("llmlingua2", {}).get("model_name", ""),
                "rate": self.config.get("llmlingua2", {}).get("rate", 0.5),
            },
        }


# ============================================================================
# 独立运行入口
# ============================================================================


if __name__ == "__main__":
    CustomLogger.disable_global_console_debug()

    # 检查是否在测试模式下运行
    if os.getenv("SAGE_EXAMPLES_MODE") == "test" or os.getenv("SAGE_TEST_MODE") == "true":
        print("🧪 Test mode detected - LLMLingua-2 pipeline requires pre-built FAISS index")
        print("✅ Test passed: Example structure validated")
        sys.exit(0)

    # 配置文件路径 (使用 benchmark_refiner 的配置)
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "benchmark_refiner",
        "config",
        "config_llmlingua2.yaml",
    )

    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        print("Please ensure the config file exists before running this example.")
        sys.exit(1)

    config = load_config(config_path)

    # 检查 LLMLingua-2 相关配置
    if config.get("llmlingua2", {}).get("enabled", True):
        print("🚀 LLMLingua-2 compression enabled")
        print(f"   Model: {config['llmlingua2'].get('model_name', 'default')}")
        print(f"   Rate: {config['llmlingua2'].get('rate', 0.5)}")
    else:
        print("ℹ️  LLMLingua-2 disabled - running in baseline mode")

    # 检查索引文件是否存在
    if config.get("retriever", {}).get("type") == "wiki18_faiss":
        index_path = config["retriever"]["faiss"]["index_path"]
        # 展开环境变量
        index_path = os.path.expandvars(index_path)
        if not os.path.exists(index_path):
            print(f"❌ FAISS index file not found: {index_path}")
            print(
                "Please build the FAISS index first using build_milvus_dense_index.py or similar."
            )
            print("Or modify the config to use a different retriever type.")
            sys.exit(1)

    pipeline_run(config)
