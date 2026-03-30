#!/usr/bin/env python3
"""
模块化 Adaptive-RAG Pipeline 实现

设计原则：
1. 算子解耦: Retriever、Prompter、Generator 是独立的 MapFunction
2. 迭代支持: 通过 "展开循环" 或 "状态机" 模式实现

流处理中的迭代问题：
- 传统 DAG 是无环的，不支持 feedback loop
- 解决方案有三种：
  A. 循环展开 (Unroll): 预设 max_iterations，创建 N 个串联阶段
  B. 状态机 (StateMachine): 单个算子内部维护状态，用条件判断迭代
  C. 递归提交 (ReSubmit): Sink 将需要继续迭代的数据重新提交到 Source

本文件实现方案 A (循环展开)，适合已知最大迭代次数的场景。

架构图:
```
=== 简单查询 (ZERO) ===
Source -> Classifier -> filter(ZERO) -> Generator -> Sink

=== 单次检索 (SINGLE) ===
Source -> Classifier -> filter(SINGLE) -> Retriever -> Prompter -> Generator -> Sink

=== 迭代检索 (MULTI) - 循环展开模式 ===
Source -> Classifier -> filter(MULTI) -> [Stage1] -> [Stage2] -> [Stage3] -> FinalGenerator -> Sink
                                           |           |           |
                                           v           v           v
                                        Retriever   Retriever   Retriever
                                           |           |           |
                                           v           v           v
                                        Reasoner   Reasoner   Reasoner
                                           |           |           |
                                           v           v           v
                                        SubQuery   SubQuery   (直接合成)
```
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sage.foundation import (
    FilterFunction,
    MapFunction,
    SinkFunction,
    SourceFunction,
)
from sage.runtime import FluttyEnvironment as FlownetEnvironment, LocalEnvironment

# 支持直接运行和模块运行
try:
    from .classifier import ClassificationResult, create_classifier
except ImportError:
    from classifier import ClassificationResult, create_classifier


# ============================================================================
# 数据模型
# ============================================================================


@dataclass
class QueryData:
    """查询数据"""

    query: str
    query_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ClassifiedQuery:
    """分类后的查询"""

    query: str
    query_id: str
    complexity: str  # ZERO, SINGLE, MULTI
    confidence: float


@dataclass
class RetrievalState:
    """检索状态 - 用于迭代检索"""

    query: str
    original_query: str
    query_id: str
    complexity: str
    iteration: int = 0
    max_iterations: int = 3
    retrieved_docs: list = field(default_factory=list)
    reasoning_chain: list = field(default_factory=list)
    context: str = ""
    should_continue: bool = True


@dataclass
class PromptData:
    """Prompt 数据"""

    query: str
    query_id: str
    context: str
    complexity: str
    prompt: str = ""


@dataclass
class ResultData:
    """结果数据"""

    query: str
    answer: str
    strategy_used: str
    complexity: str
    retrieval_steps: int = 0
    processing_time_ms: float = 0.0


# ============================================================================
# 独立算子: Classifier
# ============================================================================


class QueryClassifier(MapFunction):
    """查询分类器 - 判断查询复杂度"""

    def __init__(self, classifier_type: str = "rule", **kwargs):
        super().__init__(**kwargs)
        self.classifier_type = classifier_type
        self._classifier = None

    def execute(self, data: QueryData) -> ClassifiedQuery:
        if self._classifier is None:
            self._classifier = create_classifier(self.classifier_type)

        result: ClassificationResult = self._classifier.classify(data.query)

        print(f"📋 Classified: {data.query} -> {result.level.name}")

        return ClassifiedQuery(
            query=data.query,
            query_id=data.query_id,
            complexity=result.level.name,
            confidence=result.confidence,
        )


# ============================================================================
# 独立算子: Retriever
# ============================================================================


class SimpleRetriever(MapFunction):
    """
    检索器 - 独立的检索算子

    输入: RetrievalState (包含当前查询和已检索的文档)
    输出: RetrievalState (更新 retrieved_docs)
    """

    # 简单知识库
    KNOWLEDGE_BASE = [
        {
            "content": "Machine learning is a subset of artificial intelligence that learns from data.",
            "id": "1",
        },
        {"content": "Deep learning uses neural networks with multiple layers.", "id": "2"},
        {"content": "Python is a popular programming language for ML tasks.", "id": "3"},
        {"content": "BERT is a transformer-based model for NLP tasks.", "id": "4"},
        {"content": "RAG combines retrieval with generation for better answers.", "id": "5"},
        {"content": "Transformers use attention mechanisms for sequence modeling.", "id": "6"},
        {"content": "GPT models are autoregressive language models.", "id": "7"},
        {"content": "Vector databases store embeddings for similarity search.", "id": "8"},
    ]

    def __init__(self, top_k: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.top_k = top_k

    def execute(self, state: RetrievalState) -> RetrievalState:
        """执行检索，更新状态"""
        query_words = set(state.query.lower().split())
        scored_docs = []

        for doc in self.KNOWLEDGE_BASE:
            # 去重：跳过已检索的文档
            if doc["id"] in [d.get("id") for d in state.retrieved_docs]:
                continue

            content_words = set(doc["content"].lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored_docs.append({**doc, "score": overlap / len(query_words)})

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        new_docs = scored_docs[: self.top_k]

        # 更新状态
        state.retrieved_docs.extend(new_docs)
        state.context = "\n".join(
            [f"[Doc {i + 1}]: {d['content']}" for i, d in enumerate(state.retrieved_docs)]
        )

        print(f"  🔍 Retrieved {len(new_docs)} new docs (total: {len(state.retrieved_docs)})")

        return state


# ============================================================================
# 独立算子: Prompter
# ============================================================================


class SimplePrompter(MapFunction):
    """
    Prompt 构建器 - 将 context + query 组合成 prompt

    输入: RetrievalState 或 ClassifiedQuery
    输出: PromptData
    """

    def __init__(self, template: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.template = template

    def execute(self, data: RetrievalState | ClassifiedQuery) -> PromptData:
        if isinstance(data, RetrievalState):
            context = data.context or "No relevant documents found."
            query = data.original_query
            query_id = data.query_id
            complexity = data.complexity

            if self.template == "iterative":
                # 迭代检索的最终合成 prompt
                chain_text = "\n".join(
                    [f"Step {i + 1}: {r}" for i, r in enumerate(data.reasoning_chain)]
                )
                prompt = f"""Based on the following reasoning steps and gathered context, provide a comprehensive answer.

Reasoning steps:
{chain_text}

Context:
{context}

Question: {query}

Answer:"""
            else:
                prompt = f"""Answer the question based on the provided context. If no relevant info, say so.

Context:
{context}

Question: {query}

Answer:"""
        else:
            # ClassifiedQuery - 直接生成（无检索）
            query = data.query
            query_id = data.query_id
            complexity = data.complexity
            context = ""
            prompt = f"Answer the following question directly: {query}"

        return PromptData(
            query=query,
            query_id=query_id,
            context=context,
            complexity=complexity,
            prompt=prompt,
        )


# ============================================================================
# 独立算子: Generator
# ============================================================================


class SimpleGenerator(MapFunction):
    """
    生成器 - 调用 LLM 生成回复

    输入: PromptData
    输出: ResultData
    """

    def __init__(
        self,
        llm_base_url: str = "http://11.11.11.7:8903/v1",
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 512,
        strategy_name: str = "unknown",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        self.strategy_name = strategy_name

    def execute(self, data: PromptData) -> ResultData:
        import requests

        start_time = time.time()

        messages = [{"role": "user", "content": data.prompt}]

        try:
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": 0.7,
                },
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
        except Exception as e:
            answer = f"[Generation Error] {str(e)}"

        return ResultData(
            query=data.query,
            answer=answer,
            strategy_used=self.strategy_name,
            complexity=data.complexity,
            retrieval_steps=0,
            processing_time_ms=(time.time() - start_time) * 1000,
        )


# ============================================================================
# 迭代检索专用算子: Reasoner (子查询生成)
# ============================================================================


class IterativeReasoner(MapFunction):
    """
    迭代推理器 - 根据当前上下文决定是否继续检索，生成子查询

    输入: RetrievalState
    输出: RetrievalState (更新 query 和 reasoning_chain)
    """

    def __init__(
        self,
        llm_base_url: str = "http://11.11.11.7:8903/v1",
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model

    def _llm_call(self, messages: list[dict]) -> str:
        import requests

        try:
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.5,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Error] {str(e)}"

    def execute(self, state: RetrievalState) -> RetrievalState:
        state.iteration += 1

        # 记录当前步骤
        state.reasoning_chain.append(
            f"Query: {state.query} -> Found {len(state.retrieved_docs)} docs"
        )

        # 检查是否达到最大迭代或足够的文档
        if state.iteration >= state.max_iterations or len(state.retrieved_docs) >= 6:
            state.should_continue = False
            print(f"  🧠 Reasoner: Iteration {state.iteration} - STOP (enough info)")
            return state

        # 生成下一步子查询
        messages = [
            {
                "role": "system",
                "content": "Generate a follow-up search query. Reply with ONLY the query.",
            },
            {
                "role": "user",
                "content": f"Original: {state.original_query}\nContext so far:\n{state.context}\n\nFollow-up query:",
            },
        ]

        new_query = self._llm_call(messages).strip()
        state.query = new_query

        print(f"  🧠 Reasoner: Iteration {state.iteration} - New query: {new_query[:50]}")

        return state


# ============================================================================
# 辅助算子: 状态转换
# ============================================================================


class ClassifiedToState(MapFunction):
    """将 ClassifiedQuery 转换为 RetrievalState"""

    def __init__(self, max_iterations: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    def execute(self, data: ClassifiedQuery) -> RetrievalState:
        return RetrievalState(
            query=data.query,
            original_query=data.query,
            query_id=data.query_id,
            complexity=data.complexity,
            max_iterations=self.max_iterations,
        )


class StateToPrompt(MapFunction):
    """将 RetrievalState 转换为 PromptData（最终合成）"""

    def execute(self, state: RetrievalState) -> PromptData:
        chain_text = "\n".join([f"Step {i + 1}: {r}" for i, r in enumerate(state.reasoning_chain)])
        context = state.context or "No documents found."

        prompt = f"""Based on the reasoning steps and context, provide a comprehensive answer.

Reasoning:
{chain_text}

Context:
{context}

Question: {state.original_query}

Answer:"""

        return PromptData(
            query=state.original_query,
            query_id=state.query_id,
            context=context,
            complexity=state.complexity,
            prompt=prompt,
        )


# ============================================================================
# 迭代阶段封装 (循环展开模式)
# ============================================================================


class IterationStage(MapFunction):
    """
    迭代阶段 - 封装一轮 Retrieve + Reason

    这是循环展开模式的核心：每个 Stage 是独立的 MapFunction，
    可以串联多个 Stage 实现 N 轮迭代。
    """

    def __init__(
        self,
        stage_id: int,
        top_k: int = 2,
        llm_base_url: str = "http://11.11.11.7:8903/v1",
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stage_id = stage_id
        self.top_k = top_k
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self._retriever = None
        self._reasoner = None

    def _get_retriever(self):
        if self._retriever is None:
            self._retriever = SimpleRetriever(top_k=self.top_k)
        return self._retriever

    def _get_reasoner(self):
        if self._reasoner is None:
            self._reasoner = IterativeReasoner(
                llm_base_url=self.llm_base_url,
                llm_model=self.llm_model,
            )
        return self._reasoner

    def execute(self, state: RetrievalState) -> RetrievalState:
        # 如果前一阶段决定停止，直接透传
        if not state.should_continue:
            return state

        print(f"  📍 Stage {self.stage_id}: Processing...")

        # 1. 检索
        state = self._get_retriever().execute(state)

        # 2. 推理（决定是否继续，生成子查询）
        state = self._get_reasoner().execute(state)

        return state


# ============================================================================
# Filter 算子
# ============================================================================


class ComplexityFilter(FilterFunction):
    """按复杂度过滤"""

    def __init__(self, target_complexity: str, **kwargs):
        super().__init__(**kwargs)
        self.target_complexity = target_complexity

    def execute(self, data: ClassifiedQuery) -> bool:
        return data.complexity == self.target_complexity


# ============================================================================
# Sink 算子
# ============================================================================


class ResultSink(SinkFunction):
    """结果收集器"""

    _all_results: list[ResultData] = []

    def __init__(self, branch_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.branch_name = branch_name
        self.count = 0

    def execute(self, data: ResultData):
        self.count += 1
        ResultSink._all_results.append(data)

        answer_display = data.answer[:200] + "..." if len(data.answer) > 200 else data.answer
        print(
            f"\n🎯 [{self.branch_name}] Result #{self.count}:\n"
            f"   Query: {data.query}\n"
            f"   Strategy: {data.strategy_used}\n"
            f"   Answer: {answer_display}"
        )

        return data

    @classmethod
    def get_all_results(cls) -> list[ResultData]:
        return cls._all_results.copy()

    @classmethod
    def clear_results(cls):
        cls._all_results.clear()


# ============================================================================
# Source 算子
# ============================================================================


class QuerySource(SourceFunction):
    """查询源"""

    def __init__(self, queries: list[str], **kwargs):
        super().__init__(**kwargs)
        self.queries = queries
        self.index = 0

    def execute(self) -> QueryData | None:
        if self.index >= len(self.queries):
            return None

        query = self.queries[self.index]
        query_id = f"q_{self.index}"
        self.index += 1

        return QueryData(query=query, query_id=query_id)


# ============================================================================
# Pipeline 构建函数
# ============================================================================


def build_modular_pipeline(
    env: LocalEnvironment | FlownetEnvironment,
    queries: list[str],
    classifier_type: str = "rule",
    llm_base_url: str = "http://11.11.11.7:8903/v1",
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
    max_iterations: int = 3,
) -> LocalEnvironment | FlownetEnvironment:
    """
    构建模块化 Adaptive-RAG Pipeline

    架构特点:
    1. 算子解耦: Retriever, Prompter, Generator 独立
    2. 迭代展开: MULTI 分支使用 N 个 IterationStage 串联

    Args:
        env: SAGE Environment
        queries: 查询列表
        classifier_type: 分类器类型
        llm_base_url: LLM 服务地址
        llm_model: LLM 模型名称
        max_iterations: 最大迭代次数

    Returns:
        配置好的 Environment
    """

    # 清空之前的结果
    ResultSink.clear_results()

    # Source -> Classifier
    source = env.source(QuerySource(queries=queries))
    classified = source.map(QueryClassifier(classifier_type=classifier_type))

    # ========== 分支 1: ZERO (直接生成) ==========
    zero_branch = classified.filter(ComplexityFilter(target_complexity="ZERO"))
    zero_prompt = zero_branch.map(SimplePrompter(template="direct"))
    zero_result = zero_prompt.map(
        SimpleGenerator(
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            strategy_name="no_retrieval",
        )
    )
    zero_result.sink(ResultSink(branch_name="ZERO"))

    # ========== 分支 2: SINGLE (单次检索) ==========
    single_branch = classified.filter(ComplexityFilter(target_complexity="SINGLE"))
    single_state = single_branch.map(ClassifiedToState(max_iterations=1))
    single_retrieved = single_state.map(SimpleRetriever(top_k=3))
    single_prompt = single_retrieved.map(SimplePrompter(template="rag"))
    single_result = single_prompt.map(
        SimpleGenerator(
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            strategy_name="single_retrieval",
        )
    )
    single_result.sink(ResultSink(branch_name="SINGLE"))

    # ========== 分支 3: MULTI (迭代检索 - 循环展开模式) ==========
    multi_branch = classified.filter(ComplexityFilter(target_complexity="MULTI"))
    multi_state = multi_branch.map(ClassifiedToState(max_iterations=max_iterations))

    # 循环展开: 串联 N 个 IterationStage
    current_state = multi_state
    for i in range(max_iterations):
        current_state = current_state.map(
            IterationStage(
                stage_id=i + 1,
                top_k=2,
                llm_base_url=llm_base_url,
                llm_model=llm_model,
            )
        )

    # 最终合成
    multi_prompt = current_state.map(StateToPrompt())
    multi_result = multi_prompt.map(
        SimpleGenerator(
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            strategy_name="iterative_retrieval",
        )
    )
    multi_result.sink(ResultSink(branch_name="MULTI"))

    return env


# ============================================================================
# 运行示例
# ============================================================================


def main():
    """运行模块化 Adaptive-RAG Pipeline"""

    print("=" * 60)
    print("模块化 Adaptive-RAG Pipeline")
    print("=" * 60)
    print("\n设计特点:")
    print("  1. 算子解耦: Retriever, Prompter, Generator 独立")
    print("  2. 迭代展开: 串联 N 个 IterationStage")
    print("=" * 60)

    # 测试查询
    test_queries = [
        # ZERO - 直接回答
        "What is 2 + 2?",
        # SINGLE - 需要检索
        "What is machine learning?",
        # MULTI - 复杂多步推理
        "Compare the differences between BERT and GPT, and explain how they relate to RAG systems.",
    ]

    # 选择运行模式
    use_remote = False  # 设为 True 使用 Flownet 集群

    if use_remote:
        print("\n🔧 Using RemoteEnvironment (Flownet Cluster)")
        env = FlownetEnvironment(
            "modular-adaptive-rag",
            config={"flownet": {"address": "flownet://sage-node-1"}},
        )
    else:
        print("\n🔧 Using LocalEnvironment")
        env = LocalEnvironment("modular-adaptive-rag")

    # 构建 Pipeline
    env = build_modular_pipeline(
        env=env,
        queries=test_queries,
        classifier_type="rule",
        llm_base_url="http://11.11.11.7:8903/v1",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        max_iterations=3,
    )

    # 执行
    print("\n🚀 Starting Pipeline...")
    start_time = time.time()

    env.submit(autostop=True)

    elapsed = time.time() - start_time
    print(f"\n✅ Pipeline completed in {elapsed:.2f}s")

    # 打印汇总
    results = ResultSink.get_all_results()
    print(f"\n📊 Total results: {len(results)}")

    for r in results:
        print(f"  - [{r.complexity}] {r.query[:40]}... -> {r.strategy_used}")


if __name__ == "__main__":
    main()
