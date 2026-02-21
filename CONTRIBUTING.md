# Contributing to SAGE Benchmark

Thank you for your interest in contributing to SAGE Benchmark! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/intellistream/sage-benchmark.git
cd sage-benchmark
./quickstart.sh  # Initialize git submodules
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 3. Install Pre-commit Hooks

**IMPORTANT**: Always install pre-commit hooks before making commits:

```bash
pre-commit install
```

This ensures your code is automatically checked for style issues before each commit.

#### Why isn't this automatic?

**Q: Why doesn't CI/CD automatically install pre-commit hooks?**

A: CI/CD runs in a temporary environment and uses pre-commit hooks to validate code. However, on your local machine, you must manually run `pre-commit install` because:

1. **Security**: Git hooks can execute arbitrary code, so they require explicit user consent
2. **User control**: Developers may have different workflows or prefer to run checks manually
3. **Git design**: Git hooks are not part of the repository itself (they live in `.git/hooks/`)

**The workflow is:**
- **CI/CD**: Runs `pre-commit run --all-files` to validate your PR
- **Local**: You run `pre-commit install` once, then hooks run automatically on `git commit`

If you forget to install pre-commit locally, CI/CD will catch issues, but it's faster to catch them before pushing!

## Code Quality Standards

SAGE Benchmark follows strict code quality standards:

- **Python Version**: 3.10+
- **Formatter**: Black (line length: 100)
- **Import Sorting**: isort (black profile)
- **Linter**: Ruff
- **Type Hints**: Required for all public APIs
- **Docstrings**: Google style

### Code Style

```python
"""Module docstring describing purpose."""

from __future__ import annotations

from pathlib import Path
from typing import Any

def example_function(config: dict[str, Any], output_dir: Path) -> dict[str, float]:
    """Brief description.

    Args:
        config: Configuration dictionary
        output_dir: Directory to save results

    Returns:
        Dictionary of results

    Raises:
        ValueError: If configuration is invalid
    """
    # Implementation
    pass
```

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to ensure code quality. The hooks run automatically before each commit and include:

1. **trailing-whitespace**: Remove trailing whitespace
2. **end-of-file-fixer**: Ensure files end with newline
3. **check-yaml**: Validate YAML syntax
4. **check-json**: Validate JSON syntax
5. **check-added-large-files**: Prevent large files
6. **check-merge-conflict**: Detect merge conflict markers
7. **detect-private-key**: Prevent committing private keys
8. **black**: Format Python code
9. **isort**: Sort imports
10. **ruff-check**: Lint and auto-fix Python code

**Important**: When ruff auto-fixes files, pre-commit will fail with "files were modified by this hook". This is **expected behavior** - simply re-run the commit command and it will succeed:

```bash
git commit -m "your message"
# If ruff fixes files, you'll see: "ruff-check...Failed - files were modified"
# Just commit again:
git commit -m "your message"
# Now it will pass
```

### Manual Pre-commit Checks

Run pre-commit on all files:

```bash
pre-commit run --all-files
```

Run pre-commit on specific files:

```bash
pre-commit run --files path/to/file.py
```

Update pre-commit hooks to latest versions:

```bash
pre-commit autoupdate
```

### Fixing Common Issues

#### Ruff Auto-Fix Workflow

When ruff auto-fixes files during commit, follow this workflow:

```bash
# First commit attempt - ruff fixes files
git commit -m "your message"
# Output: "ruff-check...Failed - files were modified by this hook"

# Ruff has fixed the files, now commit again
git commit -m "your message"
# Output: All checks pass! ✅
```

This is normal pre-commit behavior and ensures all fixes are included in your commit.

#### Import Sorting Errors

If CI reports isort errors:

```bash
isort --profile black .
```

Or let pre-commit fix them:

```bash
pre-commit run isort --all-files
```

#### Formatting Errors

If CI reports Black formatting errors:

```bash
black --line-length 100 .
```

#### Linting Errors

If CI reports Ruff errors that can't be auto-fixed:

```bash
# Auto-fix what's possible
ruff check --fix .

# See remaining issues
ruff check .
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_example.py

# Run with coverage
pytest --cov=sage.benchmark --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory mirroring `src/` structure
- Name test files `test_*.py`
- Use descriptive test names: `test_<functionality>_<expected_behavior>`
- Aim for >80% code coverage

Example:

```python
import pytest
from sage.benchmark.example import ExampleClass

def test_example_basic_functionality():
    """Test basic functionality of ExampleClass."""
    obj = ExampleClass(config={"key": "value"})
    result = obj.run()

    assert result.status == "success"
    assert len(result.data) > 0
```

## Submitting Changes

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Changes

- Follow code quality standards
- Add tests for new functionality
- Update documentation as needed
- Ensure pre-commit hooks pass

### 3. Commit Changes

Pre-commit hooks will run automatically:

```bash
git add .
git commit -m "feat: add new benchmark for X

- Implement X benchmark
- Add tests for X
- Update documentation"
```

Use conventional commit messages:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

### 5. CI/CD Checks

Your PR will be automatically checked by CI/CD:

- ✅ Tests pass on Python 3.10, 3.11, 3.12
- ✅ Code formatting (Black)
- ✅ Import sorting (isort)
- ✅ Linting (Ruff)
- ✅ Code coverage >80%

## Git Submodules

SAGE Benchmark uses git submodules for external dependencies:

- `src/sage/benchmark/benchmark_amm` → [LibAMM](https://github.com/intellistream/LibAMM.git)
- `src/sage/benchmark/benchmark_anns` → [SAGE-DB-Bench](https://github.com/intellistream/SAGE-DB-Bench.git)
- `src/sage/data` → [sageData](https://github.com/intellistream/sageData.git)

### Working with Submodules

If you need to modify submodule content:

```bash
# 1. Navigate to submodule
cd src/sage/data

# 2. Make changes and commit within submodule
git add .
git commit -m "feat: update data module"
git push origin main-dev

# 3. Return to parent repo and update reference
cd ../../..
git add src/sage/data
git commit -m "chore: update data submodule"
git push origin main-dev
```

## Questions?

If you have questions or need help:

- Open an issue on GitHub
- Check existing documentation
- Reach out to the maintainers

Thank you for contributing to SAGE Benchmark! 🚀
