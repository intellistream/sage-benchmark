#!/usr/bin/env python3
"""
Tool Use Agent Pipeline
=======================

Main pipeline implementation integrating:
- sage-mem: HierarchicalMemoryService for agent memory (STM/MTM/LTM)
- sage-refiner: ContextService for context compression
- sage-db: SageDBService for vector search (RAG)
- ReAct planning: Thought-Action-Observation-Reflection loop

Pipeline Architecture:
    UserQuerySource -> ToolSelector -> ToolExecutor -> ResponseGenerator -> ResponseSink

Services are registered with env.register_service() and accessed via self.call_service().

Usage:
    # As module
    from examples.tutorials.L3_libs.agents.tool_use_agent import run_tool_use_demo
    run_tool_use_demo()

    # Command line
    python -m examples.tutorials.L3_libs.agents.tool_use_agent.pipeline
    python pipeline.py --interactive
    python pipeline.py --query "What is SAGE?"

# test_tags: category=agent, timeout=180, requires_llm=optional
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from experiments.common.inference import create_unified_inference_client, embeddings_to_list

# Ensure package is importable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]  # experiments/tool_use_agent -> SAGE
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sage.foundation import CustomLogger
from sage.runtime import LocalEnvironment

try:
    from .models import AgentState
except ImportError:
    from models import AgentState
try:
    from .operators import (
        ResponseGenerator,
        ResponseSink,
        ToolExecutor,
        ToolSelector,
        UserQuerySource,
    )
except ImportError:
    from operators import (
        ResponseGenerator,
        ResponseSink,
        ToolExecutor,
        ToolSelector,
        UserQuerySource,
    )
try:
    from .agent_tools import create_default_registry
except ImportError:
    from agent_tools import create_default_registry


# =============================================================================
# Service Registration Helpers
# =============================================================================


def register_memory_service(env: LocalEnvironment, collection_name: str = "agent_memory") -> bool:
    """
    Register HierarchicalMemoryService for agent memory.

    Uses three-tier memory: STM (short-term), MTM (medium-term), LTM (long-term).
    Falls back gracefully if sage-mem is not available.
    """
    try:

        class InMemoryHierarchicalMemoryService:
            """Benchmark-local memory service with simple relevance retrieval."""

            def __init__(self, collection_name: str, **kwargs):
                self.collection_name = collection_name
                self._entries: list[dict[str, Any]] = []

            def insert(self, entry: str | dict[str, Any], metadata: dict[str, Any] | None = None):
                payload = json.loads(entry) if isinstance(entry, str) else entry
                self._entries.append(
                    {
                        "entry": payload,
                        "metadata": metadata or {},
                    }
                )
                return {"success": True, "count": len(self._entries)}

            def retrieve(self, query: str, top_k: int = 3):
                query_terms = {term for term in query.lower().split() if len(term) > 2}
                ranked = []
                for item in self._entries:
                    text = json.dumps(item["entry"], ensure_ascii=False)
                    lowered = text.lower()
                    score = sum(1 for term in query_terms if term in lowered)
                    ranked.append(
                        {
                            "content": text,
                            "score": score,
                            **item["metadata"],
                        }
                    )

                ranked.sort(key=lambda value: value.get("score", 0), reverse=True)
                return ranked[:top_k]

        env.register_service(
            "memory_service",
            InMemoryHierarchicalMemoryService,
            collection_name=collection_name,
        )
        print("[Pipeline] Registered memory_service (benchmark-local)")
        return True
    except Exception as e:
        print(f"[Pipeline] Failed to register memory_service: {e}")
        return False


def register_context_service(env: LocalEnvironment, max_length: int = 8192) -> bool:
    """
    Register ContextService for context compression.

    Uses sage-refiner to automatically compress long contexts.
    Falls back gracefully if sage-refiner is not available.
    """
    try:

        class SimpleContextService:
            """Benchmark-local context compression and history service."""

            def __init__(self, max_context_length: int, **kwargs):
                self.max_context_length = max_context_length
                self._history: list[dict[str, str]] = []

            def add_to_history(self, role: str, content: str):
                self._history.append({"role": role, "content": content})
                return {"success": True, "history_size": len(self._history)}

            def manage_context(self, query: str, history: list[dict[str, str]] | None = None):
                items = history or self._history
                compressed = []
                total = 0
                for item in reversed(items):
                    content = item.get("content", "")
                    if total + len(content) > self.max_context_length:
                        remaining = self.max_context_length - total
                        if remaining <= 0:
                            break
                        content = content[:remaining]
                    compressed.append({"role": item.get("role", "user"), "content": content})
                    total += len(content)
                    if total >= self.max_context_length:
                        break
                compressed.reverse()
                return {"compressed_context": compressed, "query": query}

        env.register_service(
            "context_service",
            SimpleContextService,
            max_context_length=max_length,
        )
        print("[Pipeline] Registered context_service (benchmark-local)")
        return True
    except Exception as e:
        print(f"[Pipeline] Failed to register context_service: {e}")
        return False


def register_vector_db_service(
    env: LocalEnvironment,
    dimension: int | None = None,
    knowledge_base: list[dict[str, Any]] | None = None,
) -> bool:
    """
    Register SageDBService for vector search (RAG).

    Pre-populates with SAGE documentation knowledge base using real embeddings.
    Falls back gracefully if sage-db is not available.

    Args:
        env: LocalEnvironment to register service
        dimension: Vector dimension (auto-detected from embedding if None)
        knowledge_base: List of documents to pre-load
    """
    try:
        import numpy as np

        # Default SAGE knowledge base
        if knowledge_base is None:
            knowledge_base = [
                {
                    "title": "SAGE Framework Overview",
                    "text": "SAGE is a Python framework for building AI/LLM data processing pipelines with declarative dataflow. It consists of 6 layers from L1-Common to L6-Interface.",
                    "tags": "overview,architecture",
                },
                {
                    "title": "SAGE Installation",
                    "text": "To install SAGE, run ./quickstart.sh --dev --yes for development. Prerequisites: Python 3.10+, build-essential, cmake.",
                    "tags": "installation,setup",
                },
                {
                    "title": "Pipeline Operators",
                    "text": "SAGE uses SourceFunction, MapFunction, SinkFunction operators. Connect via LocalEnvironment and access services with self.call_service().",
                    "tags": "pipeline,operators",
                },
                {
                    "title": "Memory Services",
                    "text": "sage-mem provides HierarchicalMemoryService with STM/MTM/LTM tiers. Use MemoryServiceFactory.create_instance() to create services.",
                    "tags": "memory,sage-mem",
                },
                {
                    "title": "Context Compression",
                    "text": "sage-refiner ContextService provides automatic context compression. Supports simple, llmlingua2, provence, reform algorithms.",
                    "tags": "refiner,compression",
                },
            ]

        # Try to get embeddings for knowledge base
        def get_embeddings(texts: list[str]) -> tuple[list[list[float]], int] | None:
            """Get embeddings using UnifiedInferenceClient"""
            try:
                client = create_unified_inference_client()
                embeddings = embeddings_to_list(client.embed(texts))

                if embeddings and len(embeddings) > 0:
                    dim = len(embeddings[0])
                    return embeddings, dim
            except Exception as e:
                print(f"[Pipeline] Embedding error: {e}")
            return None

        # Create bootstrapped service with real embeddings
        class BootstrappedVectorDBService:
            """Benchmark-local vector DB with pre-loaded knowledge base."""

            def __init__(self, *, initial_data: list[dict], dimension: int, **kwargs):
                self.dimension = dimension
                self._vectors = None
                self._metadata: list[dict[str, Any]] = []

                texts = [item.get("text", item.get("content", "")) for item in initial_data]

                # Try real embeddings first
                embed_result = get_embeddings(texts)

                if embed_result is not None:
                    embeddings, _ = embed_result
                    vectors = np.array(embeddings, dtype=np.float32)
                    print(f"[Pipeline] Using real embeddings (dim={dimension})")
                else:
                    # Fallback to simple hash-based mock embeddings
                    print(f"[Pipeline] Using mock embeddings (dim={dimension})")
                    vectors = []
                    for text in texts:
                        vec = np.zeros(dimension, dtype=np.float32)
                        for i, char in enumerate(text[:dimension]):
                            vec[i % dimension] += ord(char) / 1000.0
                        vec = vec / (np.linalg.norm(vec) + 1e-8)
                        vectors.append(vec)
                    vectors = np.array(vectors, dtype=np.float32)

                # Build metadata list
                metadata_list = []
                for item in initial_data:
                    metadata_list.append(
                        {
                            "id": item.get("id", item.get("title", "")),
                            "title": item.get("title", ""),
                            "text": item.get("text", item.get("content", "")),
                            "tags": item.get("tags", ""),
                        }
                    )

                self._vectors = np.array(vectors, dtype=np.float32)
                self._metadata = metadata_list
                print(f"[Pipeline] Loaded {len(vectors)} documents into vector_db")

            def search(self, query, k: int = 5):
                query_vec = np.array(query, dtype=np.float32).reshape(-1)
                if self._vectors is None or self._vectors.size == 0:
                    return []

                query_norm = np.linalg.norm(query_vec) + 1e-8
                vector_norms = np.linalg.norm(self._vectors, axis=1) + 1e-8
                scores = (self._vectors @ query_vec) / (vector_norms * query_norm)
                ranked_indices = np.argsort(scores)[::-1][:k]

                results = []
                for idx in ranked_indices:
                    results.append(
                        {
                            "id": self._metadata[idx].get("id"),
                            "metadata": self._metadata[idx],
                            "score": float(scores[idx]),
                        }
                    )
                return results

        # Auto-detect dimension from embedding service
        if dimension is None:
            texts = [kb[0].get("text", "") for kb in [knowledge_base] if kb]
            embed_result = get_embeddings(texts[:1]) if texts else None
            if embed_result:
                _, dimension = embed_result
                print(f"[Pipeline] Auto-detected embedding dimension: {dimension}")
            else:
                dimension = 1024  # Default for bge-large models
                print(f"[Pipeline] Using default dimension: {dimension}")

        env.register_service(
            "vector_db",
            BootstrappedVectorDBService,
            initial_data=knowledge_base,
            dimension=dimension,
        )
        print("[Pipeline] Registered vector_db (benchmark-local)")
        return True

    except Exception as e:
        print(f"[Pipeline] Failed to register vector_db: {e}")
        return False


# =============================================================================
# Pipeline Entry Points
# =============================================================================


def run_tool_use_demo(
    queries: list[str] | None = None,
    verbose: bool = True,
    register_services: bool = True,
) -> None:
    """
    Run the Tool Use Agent Pipeline demo.

    Args:
        queries: List of queries to process. Uses defaults if None.
        verbose: Enable verbose output in ResponseSink.
        register_services: Whether to register middleware services.

    Example:
        >>> from examples.tutorials.L3_libs.agents.tool_use_agent import run_tool_use_demo
        >>> run_tool_use_demo(["What is SAGE?", "Calculate 2 + 2"])
    """
    # Suppress debug logging unless verbose
    if not verbose:
        CustomLogger.disable_global_console_debug()

    print("""
========================================================================
                    Tool Use Agent Pipeline Demo
========================================================================
  Pipeline: UserQuery -> ToolSelector -> ToolExecutor -> ResponseGenerator

  Integrated Services:
    - memory_service: HierarchicalMemoryService (sage-mem)
    - context_service: ContextService (sage-refiner)
    - vector_db: SageDBService (sage-db)

  Features:
    - ReAct reasoning: Thought-Action-Observation-Reflection
    - Keyword fallback when LLM unavailable
    - Automatic context compression
    - Persistent memory across queries
========================================================================
    """)

    # Create environment
    env = LocalEnvironment("tool_use_agent")

    # Register services
    if register_services:
        print("\n[Pipeline] Registering services...")
        register_memory_service(env)
        register_context_service(env)
        register_vector_db_service(env)
        print()

    # Create tool registry
    tool_registry = create_default_registry()
    print(f"[Pipeline] Available tools: {tool_registry.list_tools()}\n")

    # Default queries
    if queries is None:
        queries = [
            "What is SAGE framework and how to install it?",
            "Calculate 15 * 23 + 47",
            "Search for information about memory services in SAGE",
        ]

    # In test mode, limit queries
    test_mode = os.getenv("SAGE_TEST_MODE") == "true"
    if test_mode:
        queries = queries[:1]
        print("[Pipeline] Test mode: processing 1 query only\n")

    # Build pipeline
    (
        env.from_source(UserQuerySource, queries=queries)
        .map(ToolSelector, tool_registry=tool_registry)
        .map(ToolExecutor, tool_registry=tool_registry)
        .map(ResponseGenerator)
        .sink(ResponseSink, verbose=verbose)
    )

    # Execute
    start_time = time.time()
    env.submit(autostop=True)
    total_time = time.time() - start_time

    print(f"\n[Pipeline] Completed in {total_time:.2f} seconds")
    env.close()


def run_interactive_mode() -> None:
    """
    Run the agent in interactive mode.

    User can input queries one at a time with persistent memory.
    """
    print("""
========================================================================
              Tool Use Agent - Interactive Mode
  Type your query and press Enter.
  Type 'quit', 'exit', or 'q' to stop.
  Type 'clear' to clear memory.
========================================================================
    """)

    # Create tool registry with no services initially
    tool_registry = create_default_registry()
    print(f"Available tools: {tool_registry.list_tools()}\n")

    # Create operator instances for reuse
    selector = ToolSelector(tool_registry=tool_registry)
    executor = ToolExecutor(tool_registry=tool_registry)
    generator = ResponseGenerator()
    sink = ResponseSink(verbose=True)

    session_id = None

    while True:
        try:
            query = input("\n> Your query: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye.")
                break

            if query.lower() == "clear":
                session_id = None
                print("Memory cleared. Starting new session.")
                continue

            # Create state
            state = AgentState(query=query)
            if session_id:
                state.session_id = session_id
            else:
                session_id = state.session_id

            # Process through operators
            state = selector.execute(state)
            state = executor.execute(state)
            state = generator.execute(state)
            sink.execute(state)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Tool Use Agent Pipeline - SAGE demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default queries
  python pipeline.py

  # Interactive mode
  python pipeline.py --interactive

  # Custom queries
  python pipeline.py --query "What is SAGE?" --query "Calculate 2+2"

  # Quiet mode (less output)
  python pipeline.py --quiet
        """,
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )

    parser.add_argument(
        "--query",
        "-q",
        action="append",
        help="Query to process (can be specified multiple times)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    parser.add_argument(
        "--no-services",
        action="store_true",
        help="Skip registering middleware services (faster startup)",
    )

    args = parser.parse_args()

    # Suppress debug logging
    CustomLogger.disable_global_console_debug()

    if args.interactive:
        run_interactive_mode()
    else:
        run_tool_use_demo(
            queries=args.query,
            verbose=not args.quiet,
            register_services=not args.no_services,
        )


if __name__ == "__main__":
    main()
