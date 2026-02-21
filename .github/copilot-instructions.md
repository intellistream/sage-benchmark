# SAGE Benchmark - GitHub Copilot Instructions

## Project Overview

**SAGE Benchmark** is a comprehensive benchmarking suite for the SAGE framework, providing performance evaluation tools for various AI/ML components including RAG, agent systems, vector databases, approximate matrix multiplication, and more.

**Architecture**: Layer L5 (Applications - Benchmarking)
**Dependencies**: sage.middleware (L4), sage.libs (L3), sage.kernel (L3), sage.platform (L2), sage.common (L1)

## Repository Structure

```
sage-benchmark/
├── src/sage/
│   ├── benchmark/          # Main benchmark package
│   │   ├── benchmark_agent/      # Agent benchmarking (tool selection, planning, timing)
│   │   ├── benchmark_amm/        # AMM benchmarking (Git submodule: LibAMM)
│   │   ├── benchmark_anns/       # ANNS benchmarking (Git submodule: SAGE-DB-Bench)
│   │   ├── benchmark_control_plane/  # Control plane scheduling benchmarks
│   │   ├── benchmark_memory/     # Memory system benchmarks
│   │   ├── benchmark_rag/        # RAG pipeline benchmarks
│   │   ├── benchmark_refiner/    # Context refinement benchmarks
│   │   ├── benchmark_sage/       # System-level SAGE benchmarks
│   │   └── benchmark_scheduler/  # Scheduler benchmarks
│   └── data/               # Dataset management (sageData submodule)
├── tests/                  # Test suite
├── docs/                   # Documentation
└── .github/
    ├── agents/            # Custom Copilot agents
    └── copilot-instructions.md  # This file
```

## Git Submodules

This repository uses Git submodules for external dependencies. **Important submodules**:

1. **benchmark_amm** (`src/sage/benchmark/benchmark_amm/`)
   - Repository: https://github.com/intellistream/LibAMM.git
   - Branch: main-dev
   - Purpose: Approximate Matrix Multiplication benchmarking

2. **benchmark_anns** (`src/sage/benchmark/benchmark_anns/`)
   - Repository: https://github.com/intellistream/SAGE-DB-Bench.git
   - Branch: main-dev
   - Purpose: Approximate Nearest Neighbor Search benchmarking

3. **sage.data** (`src/sage/data/`)
   - Repository: https://github.com/intellistream/sageData.git
   - Branch: main-dev
   - Purpose: Unified dataset management

### Working with Submodules

When suggesting changes to `benchmark_amm` or `benchmark_anns`:
- These are separate repositories tracked as submodules
- Changes must be committed within the submodule first
- Then update the submodule reference in the parent repository
- Example workflow:
  ```bash
  # 1. Make changes in submodule
  cd src/sage/benchmark/benchmark_amm
  git add .
  git commit -m "feat: add new benchmark"
  git push origin main-dev

  # 2. Update reference in parent repo
  cd ../../../..
  git add src/sage/benchmark/benchmark_amm
  git commit -m "chore: update benchmark_amm submodule"
  git push origin main
  ```

## Key Components

### 1. Benchmark Modules

#### benchmark_agent
- **Purpose**: Agent capability evaluation
- **Experiments**: Tool selection, planning, timing detection
- **Key Files**:
  - `experiments/`: Base experiment classes and implementations
  - `config/`: YAML configurations
  - `evaluation/`: Metrics and evaluators
- **Entry Point**: `python -m sage.benchmark.benchmark_agent`

#### benchmark_rag
- **Purpose**: RAG pipeline performance evaluation
- **Features**: Dense, sparse, hybrid, multimodal retrieval
- **Vector DBs**: Milvus, ChromaDB, FAISS
- **Key Files**:
  - `implementations/pipelines/`: RAG pipeline implementations
  - `evaluation/`: Experiment framework
  - `config/`: Pipeline configurations
- **Entry Point**: `python -m sage.benchmark.benchmark_rag`

#### benchmark_refiner
- **Purpose**: Context compression algorithm evaluation
- **Algorithms**: LongLLMLingua, LLMLingua-2, selective context, etc.
- **Metrics**: MNR (Mean Normalized Recall), compression ratio, latency
- **Entry Point**: `sage-refiner-bench` CLI

#### benchmark_control_plane
- **Purpose**: LLM scheduling policy evaluation
- **Schedulers**: Hybrid, LLM-based, rule-based
- **Key Files**:
  - `schedulers/`: Scheduler implementations
  - `experiments/`: Experiment configurations
  - `visualization/`: Result visualization

#### benchmark_memory
- **Purpose**: Memory system performance evaluation
- **Systems**: MemGPT, Mem0, LangMem, MemoRAG
- **Key Files**:
  - `systems/`: Memory system implementations
  - `experiments/`: Benchmark experiments
  - `statistics/`: Statistical analysis tools

#### benchmark_sage
- **Purpose**: System-level SAGE benchmarks
- **Focus**: Cross-cutting experiments across SAGE subsystems
- **Content**: ICML paper artifacts, system benchmarks
- **Key Files**:
  - `docs/`: Writing prompts for SAGE system papers
  - `experiments/`: System-level experiment implementations

### 2. Data Management (sage.data)

- **Architecture**: Two-layer design (Sources + Usages)
- **Datasets**: qa_base, bbh, mmlu, gpqa, locomo, orca_dpo
- **Usage**: `from sage.data import DataManager`
- **Documentation**: See `src/sage/data/README.md`

## Coding Guidelines

### 1. Layer Architecture

SAGE follows a strict 6-layer architecture:
- **L1 (Common)**: Foundation utilities, configuration
- **L2 (Platform)**: Platform services, storage
- **L3 (Kernel/Libs)**: Core algorithms, execution engines
- **L4 (Middleware)**: Operators, components
- **L5 (Apps/Benchmark)**: Applications and benchmarks ← **This package**
- **L6 (Interface)**: CLI, Studio, Gateway

**Important**: No upward dependencies! Benchmark code should only depend on L1-L4.

### 2. Python Style

- **Python Version**: 3.10+
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style for all modules, classes, functions
- **Formatting**: Black (line length 100)
- **Linting**: Ruff
- **Testing**: pytest with good coverage

Example:
```python
"""Module docstring describing purpose."""

from __future__ import annotations

from pathlib import Path
from typing import Any

def benchmark_function(config: dict[str, Any], output_dir: Path) -> dict[str, float]:
    """Run benchmark with given configuration.

    Args:
        config: Configuration dictionary with experiment parameters
        output_dir: Directory to save results

    Returns:
        Dictionary of metric name to value

    Raises:
        ValueError: If configuration is invalid
    """
    # Implementation
    pass
```

### 3. Configuration Management

- **Format**: YAML for configurations
- **Location**: `{module}/config/` directories
- **Pattern**: Use Pydantic models for validation
- **Environment**: Support environment variable substitution

Example:
```yaml
experiment:
  name: "tool_selection_benchmark"
  output_dir: "${SAGE_OUTPUT_DIR:./output}"

model:
  name: "gpt-4"
  temperature: 0.0

evaluation:
  metrics:
    - "precision"
    - "recall"
    - "f1_score"
```

### 4. Experiment Pattern

All benchmarks should follow this pattern:

```python
class BaseExperiment:
    def prepare(self) -> None:
        """Prepare experiment (load data, initialize models)."""
        pass

    def run(self) -> ExperimentResult:
        """Run experiment and return results."""
        pass

    def finalize(self) -> None:
        """Clean up resources."""
        pass
```

### 5. Testing

- **Location**: Mirror `src/` structure in `tests/`
- **Naming**: `test_{module_name}.py`
- **Fixtures**: Use `conftest.py` for shared fixtures
- **Coverage**: Aim for >80% coverage
- **CI**: Tests run on PR submission

Example test:
```python
import pytest
from sage.benchmark.benchmark_agent import ToolSelectionExperiment

def test_tool_selection_basic(sample_config):
    """Test basic tool selection functionality."""
    exp = ToolSelectionExperiment(sample_config)
    exp.prepare()
    result = exp.run()

    assert result.metrics["precision"] > 0
    assert len(result.predictions) == len(result.ground_truth)
```

## CLI Patterns

Most benchmark modules provide CLI entry points:

```bash
# Agent benchmarks
python -m sage.benchmark.benchmark_agent tool-selection --config config.yaml

# Refiner benchmarks
sage-refiner-bench run --config config/config_longrefiner.yaml

# RAG benchmarks
python -m sage.benchmark.benchmark_rag --pipeline dense --db milvus

# System benchmarks
python -m sage.benchmark.benchmark_sage --experiment 5.1
```

## Documentation Standards

### README Structure

Each benchmark module should have:

1. **Overview**: Brief description and purpose
2. **Quick Start**: Minimal example to get started
3. **Architecture**: Component structure
4. **Configuration**: Config file documentation
5. **Examples**: Usage examples
6. **API Reference**: Key classes and functions
7. **Development**: Setup and testing instructions

### Code Comments

- Use docstrings for public APIs
- Inline comments for complex logic
- TODO/FIXME/NOTE markers for tracked items
- Reference issue numbers when relevant

```python
# TODO(#123): Implement parallel processing for large batches
# FIXME: Handle edge case where model returns None
# NOTE: This approach is temporary until we migrate to new API
```

## Common Tasks

### Adding a New Benchmark

1. Create module directory: `src/sage/benchmark/benchmark_X/`
2. Add `__init__.py` with module docstring
3. Create subdirectories: `config/`, `experiments/`, `evaluation/`
4. Implement experiment classes following `BaseExperiment` pattern
5. Add CLI entry point in `__main__.py`
6. Write tests in `tests/benchmark_X/`
7. Update main `README.md` and module `README.md`
8. Add dependencies to `pyproject.toml`

### Adding a New Dataset

1. Add dataset to `sage.data` submodule (sageData repository)
2. Or create dataset loader in benchmark module
3. Document dataset in appropriate README
4. Add usage examples
5. Update DataManager if using centralized management

### Adding New Metrics

1. Implement metric in `{module}/evaluation/metrics.py`
2. Add metric to evaluator classes
3. Document metric calculation and interpretation
4. Add tests for metric calculation
5. Update configuration schemas to include new metric

## Performance Considerations

- **Lazy Loading**: Load models/data only when needed
- **Caching**: Cache expensive computations (embeddings, model outputs)
- **Batching**: Process data in batches for efficiency
- **Parallelization**: Use multiprocessing for CPU-bound tasks
- **GPU Management**: Proper CUDA memory management for GPU tasks
- **Resource Cleanup**: Always clean up in `finalize()` methods

## Error Handling

- **Validation**: Validate configurations early (Pydantic models)
- **Logging**: Use proper logging levels (DEBUG, INFO, WARNING, ERROR)
- **Exceptions**: Raise specific exceptions with clear messages
- **Recovery**: Handle transient failures with retries where appropriate

```python
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> dict:
    """Load and validate configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        # Load and validate
        config = yaml.safe_load(config_path.read_text())
        logger.info(f"Loaded config from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in {config_path}: {e}")
        raise ValueError(f"Failed to parse config: {e}") from e
```

## Integration Points

### SAGE Framework Integration

- **sage.common**: Configuration, paths, utilities
- **sage.platform**: Service management, storage
- **sage.kernel**: Job execution, scheduling
- **sage.libs**: RAG components, algorithms
- **sage.middleware**: Vector DBs, operators

Example:
```python
from sage.common.config import get_config
from sage.libs.rag import DenseRetriever
from sage.middleware.milvus import MilvusClient

# Use SAGE components in benchmarks
config = get_config()
retriever = DenseRetriever(model=config.embedding_model)
client = MilvusClient(host=config.milvus_host)
```

## CI/CD

- **GitHub Actions**: Automated testing on PR
- **Pre-commit**: Black, Ruff, type checking
- **Coverage**: Codecov integration
- **Release**: Automated PyPI publishing on tag

## Support and Resources

- **Main SAGE Repo**: https://github.com/intellistream/SAGE
- **Issues**: Report bugs in respective repositories
- **Documentation**: See module-specific READMEs
- **Papers**: See `docs/benchmark_memory/` for research artifacts

## Tips for Copilot

When helping with sage-benchmark:

1. **Check submodules**: Remember `benchmark_amm` and `benchmark_anns` are submodules
2. **Layer dependencies**: Only suggest dependencies on L1-L4
3. **Configuration-driven**: Prefer YAML configs over hardcoded values
4. **Testing**: Always suggest tests for new functionality
5. **Documentation**: Include docstrings and README updates
6. **Type hints**: All public APIs should have type annotations
7. **Error handling**: Validate inputs and provide clear error messages
8. **Experiment pattern**: Follow prepare/run/finalize lifecycle
9. **CLI consistency**: Follow existing CLI patterns
10. **Performance**: Consider batching and caching for expensive operations

## Version Information

- **Python**: 3.10+
- **Package**: isage-benchmark
- **License**: MIT
- **Maintainer**: IntelliStream Team

---

*Last Updated: January 2026*
