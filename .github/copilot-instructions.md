# sage-benchmark - GitHub Copilot Instructions

## Project Overview

**sage-benchmark** is SAGE framework-specific system-level benchmarking repository, focused on end-to-end experiments and performance validation of the SAGE platform.

**Repository**: https://github.com/intellistream/sage-benchmark
**PyPI**: `isage-sage-benchmark`
**Layer**: L5 (Applications - Benchmarking, independent repository)

**Important Distinction**:
- **sage-benchmark** (this repository): SAGE framework-specific system benchmarks
- **OmniBenchmark**: Organization-level comprehensive benchmark collection

## Architecture & Dependencies

### Layer Position

sage-benchmark is an **independent L5 application** that depends on SAGE core components:

```
sage-benchmark (L5, independent)
    ↓ depends on (via PyPI)
L4: isage-middleware     # Operators and components
L3: isage-kernel         # Dataflow engine
L3: isage-libs           # RAG, agents, algorithms
L2: isage-platform       # Platform services
L1: isage-common         # Foundation
```

### Package Structure

```
sage-benchmark/
├── __init__.py                # Package initialization with version
├── _version.py                # Version information
├── __main__.py                # CLI entry point
├── pyproject.toml             # Package configuration
├── quickstart.sh              # Development setup script
├── config/                    # Experiment configurations
│   ├── exp_5_1.yaml          # E2E pipeline config
│   ├── exp_5_2.yaml          # Control Plane config
│   └── ...
├── experiments/               # Experiment implementations
│   ├── exp_5_1_e2e_pipeline.py
│   ├── exp_5_2_control_plane.py
│   └── ...
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
└── tests/                     # Test suite
```

## 🚨 CRITICAL Coding Principles

### ❌ NEVER MANUAL PIP INSTALL - ALWAYS USE pyproject.toml

**ALL dependencies MUST be declared in pyproject.toml. NEVER use manual `pip install` commands.**

```bash
# ❌ FORBIDDEN
pip install transformers

# ✅ CORRECT - Add to pyproject.toml then:
pip install -e ".[dev]"
```

### ❌ NO FALLBACK LOGIC - PROJECT-WIDE RULE

**NEVER use try-except fallback patterns anywhere in the codebase.**

```python
# ❌ BAD
try:
    from sage.kernel import JobManager
except ImportError:
    JobManager = MockManager

# ✅ GOOD
from sage.kernel import JobManager  # Fails fast if not installed
```

## Development Workflow

### Installation

```bash
# Clone repository
git clone https://github.com/intellistream/sage-benchmark.git
cd sage-benchmark

# Install SAGE dependencies (if not already installed)
pip install isage-common isage-kernel isage-libs isage-middleware

# Install sage-benchmark
./quickstart.sh --dev
```

### Running Experiments

```bash
# E2E Pipeline benchmark
python -m sage_benchmark.experiments.exp_5_1_e2e_pipeline

# Control Plane benchmark
python -m sage_benchmark.experiments.exp_5_2_control_plane

# With custom config
python -m sage_benchmark.experiments.exp_5_1_e2e_pipeline --config config/custom.yaml
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sage_benchmark

# Run specific test
pytest tests/test_exp_5_1.py
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type check
mypy .
```

## Experiment Structure

### Experiment Template

All experiments should follow this pattern:

```python
"""
Experiment X.Y: Brief description

Measures: [what it measures]
Config: config/exp_X_Y.yaml
"""

from pathlib import Path
from typing import Any
import yaml

class ExperimentXY:
    """Brief description."""

    def __init__(self, config_path: Path):
        """Initialize with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def setup(self) -> None:
        """Set up experiment."""
        pass

    def run(self) -> dict[str, Any]:
        """Run experiment and return results."""
        pass

    def teardown(self) -> None:
        """Clean up resources."""
        pass
```

### Configuration Files

Use YAML for all configurations in `config/`:

```yaml
# config/exp_X_Y.yaml
experiment:
  name: "Experiment X.Y"
  description: "Brief description"

parameters:
  # Experiment-specific parameters
  key: value

output:
  dir: "results/exp_X_Y"
  format: "json"
```

## Existing Experiments

### Experiment 5.1: End-to-End Pipeline

**File**: `experiments/exp_5_1_e2e_pipeline.py`
**Config**: `config/exp_5_1.yaml`
**Purpose**: Benchmark complete SAGE dataflow pipeline performance

### Experiment 5.2: Control Plane Scheduling

**File**: `experiments/exp_5_2_control_plane.py`
**Config**: `config/exp_5_2.yaml`
**Purpose**: Evaluate LLM/embedding Control Plane scheduling policies

### Experiment 5.3: Isolation

**File**: `experiments/exp_5_3_isolation.py`
**Config**: `config/exp_5_3.yaml`
**Purpose**: Measure resource isolation between components

### Experiment 5.4: Scalability

**File**: `experiments/exp_5_4_scalability.py`
**Config**: `config/exp_5_4.yaml`
**Purpose**: Test horizontal and vertical scaling

### Experiment 5.5: Heterogeneity

**File**: `experiments/exp_5_5_heterogeneity.py`
**Config**: `config/exp_5_5.yaml`
**Purpose**: Validate cross-platform deployment

## SAGE Dependency Reference

### Common Imports

```python
# L1: Common
from sage.common.config import get_config
from sage.common.components.sage_embedding import EmbeddingFactory

# L3: Kernel
from sage.kernel import JobManager, NodeSelector
from sage.kernel.operators import MapOperator, FilterOperator

# L3: Libs
from sage.libs.rag import SimpleRAG, DenseRetriever
from sage.libs.agents import ReactAgent

# L4: Middleware
from sage.middleware.components.sage_db import SageDB
from sage.middleware.components.sage_flow import SageFlow
```

### Control Plane Usage

**MUST use Control Plane for ALL LLM operations:**

```python
# ✅ CORRECT - Through Control Plane
from isagellm import UnifiedInferenceClient

client = UnifiedInferenceClient.create()
response = client.chat([{"role": "user", "content": "Hello"}])

# ❌ WRONG - Direct engine access
# vllm.entrypoints.openai.api_server
```

## Common Issues

### Import Errors

```python
# If SAGE not installed
from sage.common import ...  # ImportError

# Solution: Install SAGE dependencies
# pip install isage-common isage-kernel isage-libs isage-middleware
```

### Configuration Errors

```python
# Missing config file
config = yaml.safe_load(open("nonexistent.yaml"))  # FileNotFoundError

# Solution: Check config path, use Path.exists()
```

### Resource Cleanup

```python
# Always clean up in experiments
class MyExperiment:
    def teardown(self):
        # Clean up resources
        if hasattr(self, 'client'):
            self.client.close()
```

## Documentation Standards

### Code Comments

- Docstrings for all public APIs
- Inline comments for complex logic
- TODO/FIXME with issue references

```python
def benchmark_function(param: int) -> float:
    """Brief description.

    Args:
        param: Parameter description

    Returns:
        Result description

    Raises:
        ValueError: When condition
    """
    # TODO(#123): Optimize this section
    pass
```

### Experiment Documentation

Each experiment should have:

1. **Python docstring**: Purpose, config, usage
2. **Config file**: YAML with parameters
3. **Design doc** (optional): `docs/exp_X_Y_design.md`

## Version Management & Publishing

Follows SAGE's 4-digit versioning: `MAJOR.MINOR.PATCH.BUILD`

- **MAJOR** (0): Breaking changes
- **MINOR** (1): Feature additions
- **PATCH** (0): Bug fixes
- **BUILD** (0): Per-commit increments

### Publishing Workflow (Manual, One-by-One)

**🚨 CRITICAL: NEVER use bash scripts for publishing. ALWAYS use sage-pypi-publisher CLI tool directly.**

1. **Update version**: Edit `_version.py`
   ```bash
   # _sage_benchmark/_version.py
   __version__ = "X.Y.Z.W"
   ```

2. **Commit and tag**:
   ```bash
   git commit -m "chore: bump version to X.Y.Z.W"
   git tag -a vX.Y.Z.W -m "Release sage-benchmark X.Y.Z.W"
   git push origin vX.Y.Z.W
   ```

3. **Publish to TestPyPI** (test first):
   ```bash
   cd /path/to/sage-benchmark
   sage-pypi-publisher build . --upload -r testpypi --no-dry-run
   ```

4. **Publish to Production PyPI** (same command, change repository):
   ```bash
   cd /path/to/sage-benchmark
   sage-pypi-publisher build . --upload -r pypi --no-dry-run
   ```

### Key Commands

```bash
# ✅ CORRECT: Manual one-by-one using sage-pypi-publisher
cd /path/to/sage-benchmark && sage-pypi-publisher build . --upload -r testpypi --no-dry-run

# ❌ WRONG: Using bash scripts
# ./publish.sh sage-benchmark  # Use CLI directly instead

# ❌ WRONG: Using bash loops
# for pkg in ...; do sage-pypi-publisher ...; done
```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- PR process

## Resources

- **SAGE Main Repo**: https://github.com/intellistream/SAGE
- **SAGE Documentation**: https://sage.intellistream.com
- **OmniBenchmark**: https://github.com/intellistream/OmniBenchmark
- **Issue Tracker**: https://github.com/intellistream/sage-benchmark/issues

## Key Principles for Copilot

When working with sage-benchmark:

1. **Documentation-first**: Check existing experiments and configs before implementing
2. **Fail-fast**: No silent fallbacks, clear error messages
3. **Layer-aware**: Only depend on L1-L4 SAGE components
4. **Config-driven**: Use YAML for all experiment parameters
5. **Clean structure**: Follow experiment template pattern
6. **Test thoroughly**: Add tests for new experiments
7. **Document well**: Clear docstrings and comments

**Remember**: This is an independent repository depending on SAGE via PyPI packages.

---

*Last Updated: February 2026*
