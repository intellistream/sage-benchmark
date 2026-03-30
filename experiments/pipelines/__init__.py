"""
可复用 Pipeline 模块
====================

包含 6 个可复用的 Pipeline 实现 (A-F)，供各实验引用。

Pipeline 定义:
- Pipeline A: RAG (检索增强生成)
- Pipeline B: Long Context Refiner (长文本精炼)
- Pipeline C: Cross-Source Vector Stream Join (跨源向量流相似度 Join)
- Pipeline D: Batch Processing (批处理)
- Pipeline E: Priority Scheduling (优先级调度)
- Pipeline F: SAGE Operators Demo (算子库演示)
- Adaptive-RAG: 自适应 RAG Pipeline (基于问题复杂度动态选择策略)

SAGE 算子类型:
- SourceFunction: 数据源 - execute(data=None) -> Optional[T]
- MapFunction: 一对一映射 - execute(data: T) -> R
- FilterFunction: 过滤 - execute(data: T) -> bool (True=通过, False=过滤)
- FlatMapFunction: 一对多映射 - execute(data: T) -> Iterable[R]
- SinkFunction: 数据汇 - execute(data: T) -> None
- KeyByFunction: 分区键提取 - execute(data: T) -> Hashable
- BatchFunction: 批处理数据源 - execute() -> Optional[T]
- BaseJoinFunction: 多流 Join - execute(payload, key, tag) -> list[R]
- BaseCoMapFunction: 多输入 CoMap - map0(data) -> R, map1(data) -> R

所有 Pipeline 使用:
- 真实 SAGE 算子 (从 sage.foundation 导入)
- FluttyEnvironment 远程执行
- HeadNodeScheduler 限制 Source/Sink 在 head 节点
"""

# Adaptive-RAG Pipeline
try:
    from .adaptive_rag import (
        # Core
        AdaptiveRAGPipeline,
        AdaptiveRouterFunction,
        AdaptiveRouterMapFunction,
        ClassifierMapFunction,
        ComplexityFilterFunction,
        IterativeRetrievalStrategy,
        IterativeRetrieverFunction,
        MultiComplexityFilter,
        # Strategy Functions
        NoRetrievalFunction,
        NoRetrievalStrategy,
        QueryComplexityClassifier,
        QueryComplexityLevel,
        # SAGE Dataflow Components - Router Mode
        QueryData,
        QuerySource,
        ResultData,
        ResultSink,
        SingleComplexityFilter,
        SingleRetrievalStrategy,
        SingleRetrieverFunction,
        # SAGE Dataflow Components - Multi-Branch Mode
        ZeroComplexityFilter,
        build_adaptive_rag_pipeline,
        build_branching_adaptive_rag_pipeline,
    )
except ImportError:  # pragma: no cover - optional compatibility surface
    AdaptiveRAGPipeline = None
    QueryComplexityClassifier = None
    QueryComplexityLevel = None
    NoRetrievalFunction = None
    SingleRetrieverFunction = None
    IterativeRetrieverFunction = None
    AdaptiveRouterFunction = None
    QueryData = None
    ResultData = None
    QuerySource = None
    ClassifierMapFunction = None
    AdaptiveRouterMapFunction = None
    ComplexityFilterFunction = None
    ResultSink = None
    ZeroComplexityFilter = None
    SingleComplexityFilter = None
    MultiComplexityFilter = None
    NoRetrievalStrategy = None
    SingleRetrievalStrategy = None
    IterativeRetrievalStrategy = None
    build_adaptive_rag_pipeline = None
    build_branching_adaptive_rag_pipeline = None
from .pipeline_a_rag import RAGPipeline
from .pipeline_b_refiner import RefinerPipeline
from .pipeline_c_vector_join import VectorJoinPipeline
from .pipeline_d_batch import BatchPipeline
from .pipeline_e_scheduling import SchedulingPipeline

# Pipeline F: Operators Demo
try:
    from .pipeline_f_operators_demo import (
        OPERATOR_SUMMARY,
        CollectorSinkFunction,
        CompositeKeyByFunction,
        # MapFunction 示例
        EnrichUserEventMapFunction,
        # BaseCoMapFunction 示例
        EventOrderCoMapFunction,
        HighValueOrderFilterFunction,
        # BatchFunction 示例
        OrderBatchFunction,
        # SinkFunction 示例
        PrintSinkFunction,
        # FilterFunction 示例
        PurchaseEventFilterFunction,
        # FlatMapFunction 示例
        SplitEventDataFlatMapFunction,
        TokenizeFlatMapFunction,
        # SourceFunction 示例
        UserEventSourceFunction,
        # KeyByFunction 示例
        UserIdKeyByFunction,
        # BaseJoinFunction 示例
        UserOrderJoinFunction,
        # Demo functions
        demo_basic_pipeline,
        demo_flatmap_pipeline,
        demo_keyby_pipeline,
    )
except ImportError:  # pragma: no cover - optional compatibility surface
    OPERATOR_SUMMARY = None
    UserEventSourceFunction = None
    OrderBatchFunction = None
    EnrichUserEventMapFunction = None
    PurchaseEventFilterFunction = None
    HighValueOrderFilterFunction = None
    SplitEventDataFlatMapFunction = None
    TokenizeFlatMapFunction = None
    UserIdKeyByFunction = None
    CompositeKeyByFunction = None
    UserOrderJoinFunction = None
    EventOrderCoMapFunction = None
    PrintSinkFunction = None
    CollectorSinkFunction = None
    demo_basic_pipeline = None
    demo_flatmap_pipeline = None
    demo_keyby_pipeline = None
from .scheduler import HeadNodeScheduler

__all__ = [
    "HeadNodeScheduler",
    "RAGPipeline",
    "RefinerPipeline",
    "VectorJoinPipeline",
    "BatchPipeline",
    "SchedulingPipeline",
    # Pipeline F: Operators Demo
    "OPERATOR_SUMMARY",
    "UserEventSourceFunction",
    "OrderBatchFunction",
    "EnrichUserEventMapFunction",
    "PurchaseEventFilterFunction",
    "HighValueOrderFilterFunction",
    "SplitEventDataFlatMapFunction",
    "TokenizeFlatMapFunction",
    "UserIdKeyByFunction",
    "CompositeKeyByFunction",
    "UserOrderJoinFunction",
    "EventOrderCoMapFunction",
    "PrintSinkFunction",
    "CollectorSinkFunction",
    "demo_basic_pipeline",
    "demo_flatmap_pipeline",
    "demo_keyby_pipeline",
    # Adaptive-RAG
    "AdaptiveRAGPipeline",
    "QueryComplexityClassifier",
    "QueryComplexityLevel",
    "NoRetrievalFunction",
    "SingleRetrieverFunction",
    "IterativeRetrieverFunction",
    "AdaptiveRouterFunction",
    "QueryData",
    "ResultData",
    "QuerySource",
    "ClassifierMapFunction",
    "AdaptiveRouterMapFunction",
    "ComplexityFilterFunction",
    "ResultSink",
    "build_adaptive_rag_pipeline",
    "ZeroComplexityFilter",
    "SingleComplexityFilter",
    "MultiComplexityFilter",
    "NoRetrievalStrategy",
    "SingleRetrievalStrategy",
    "IterativeRetrievalStrategy",
    "build_branching_adaptive_rag_pipeline",
]
