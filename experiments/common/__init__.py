"""
Distributed Scheduling Benchmark - Common Components
=====================================================

:
- models: 数据模型 (State, Config, Metrics)
- operators: Pipeline 算子
- pipeline: Pipeline 工厂
- visualization: 结果可视化
- request_utils: 请求工具类 (BenchmarkClient, RequestResult, WorkloadGenerator)
"""

from .cli_args import (
    DEFAULT_BACKEND,
    SUPPORTED_BACKENDS,
    add_common_benchmark_args,
    build_run_config,
    validate_benchmark_args,
)
from .metrics_schema import (
    REQUIRED_FIELDS,
    UnifiedMetricsRecord,
    compute_backend_hash,
    compute_config_hash,
    normalize_metrics_record,
    utc_timestamp,
)
from .models import (
    AdaptiveRAGQueryData,
    AdaptiveRAGResultData,
    BenchmarkConfig,
    BenchmarkMetrics,
    ClassificationResult,
    IterativeState,
    QueryComplexityLevel,
    TaskState,
)
from .operators import (
    # Adaptive-RAG operators
    AdaptiveRAGQuerySource,
    AdaptiveRAGResultSink,
    # General operators
    ComputeOperator,
    FinalSynthesizer,
    # FiQA operators
    FiQADataLoader,
    FiQAFAISSRetriever,
    FiQATaskSource,
    IterativeReasoner,
    IterativeRetrievalInit,
    IterativeRetriever,
    LLMOperator,
    MetricsSink,
    MultiComplexityFilter,
    NoRetrievalStrategy,
    QueryClassifier,
    RAGOperator,
    SingleComplexityFilter,
    SingleRetrievalStrategy,
    TaskSource,
    ZeroComplexityFilter,
)
from .pipeline import SchedulingBenchmarkPipeline, register_fiqa_vdb_service
from .reproducibility import (
    ParityPlan,
    build_input_parity_plan,
    compute_config_fingerprint,
    set_global_seed,
)
from .request_utils import (
    BenchmarkClient,
    RequestResult,
    WorkloadGenerator,
)
from .result_writer import (
    CSV_FIELD_ORDER,
    append_jsonl_record,
    export_jsonl_to_csv,
)

__all__ = [
    # cli helpers
    "SUPPORTED_BACKENDS",
    "DEFAULT_BACKEND",
    "add_common_benchmark_args",
    "validate_benchmark_args",
    "build_run_config",
    "REQUIRED_FIELDS",
    "UnifiedMetricsRecord",
    "compute_backend_hash",
    "compute_config_hash",
    "normalize_metrics_record",
    "utc_timestamp",
    "CSV_FIELD_ORDER",
    "append_jsonl_record",
    "export_jsonl_to_csv",
    # models - general
    "BenchmarkConfig",
    "BenchmarkMetrics",
    "TaskState",
    # models - adaptive-rag
    "QueryComplexityLevel",
    "ClassificationResult",
    "AdaptiveRAGQueryData",
    "AdaptiveRAGResultData",
    "IterativeState",
    # operators - general
    "TaskSource",
    "ComputeOperator",
    "LLMOperator",
    "RAGOperator",
    "MetricsSink",
    # operators - FiQA
    "FiQADataLoader",
    "FiQATaskSource",
    "FiQAFAISSRetriever",
    # operators - adaptive-rag
    "AdaptiveRAGQuerySource",
    "QueryClassifier",
    "ZeroComplexityFilter",
    "SingleComplexityFilter",
    "MultiComplexityFilter",
    "NoRetrievalStrategy",
    "SingleRetrievalStrategy",
    "IterativeRetrievalInit",
    "IterativeRetriever",
    "IterativeReasoner",
    "FinalSynthesizer",
    "AdaptiveRAGResultSink",
    # pipeline
    "SchedulingBenchmarkPipeline",
    "register_fiqa_vdb_service",
    # request_utils
    "BenchmarkClient",
    "RequestResult",
    "WorkloadGenerator",
    # reproducibility
    "ParityPlan",
    "set_global_seed",
    "compute_config_fingerprint",
    "build_input_parity_plan",
]
