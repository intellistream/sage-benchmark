# Contributing to sage-benchmark

Thank you for your interest in contributing to sage-benchmark! This document provides guidelines for contributing to the SAGE framework-specific benchmark repository.

## 🎯 Project Scope

**sage-benchmark** is dedicated to SAGE framework-specific system-level benchmarks and experiments, complementary to OmniBenchmark (organization-level benchmark collection).

### What Goes Here

- ✅ End-to-end SAGE system experiments
- ✅ Control Plane scheduling benchmarks
- ✅ Multi-component pipeline benchmarks
- ✅ ICML artifacts and experiment configs
- ✅ SAGE framework validation benchmarks

### What Goes Elsewhere

- ❌ Generic benchmarks → [OmniBenchmark](https://github.com/intellistream/OmniBenchmark)
- ❌ Agent-specific benchmarks → [sage-agent-benchmark](https://github.com/intellistream/sage-agent-benchmark)
- ❌ Core SAGE components → [SAGE](https://github.com/intellistream/SAGE)
- ❌ Application examples → [sage-examples](https://github.com/intellistream/sage-examples)

## 📋 Prerequisites

Before contributing, ensure:

1. **Python 3.10+** installed
2. **SAGE framework** installed from PyPI:
   ```bash
   pip install isage-common isage-kernel isage-libs isage-middleware
   ```
3. **Git** configured with your name and email

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/sage-benchmark.git
cd sage-benchmark
```

### 2. Set Up Development Environment

```bash
# Run quickstart script
./quickstart.sh --dev

# Or manually install
pip install -e ".[dev]"
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## 🔧 Development Workflow

### Code Style

We follow SAGE's unified code style:

- **Formatter**: Ruff (line length 100)
- **Linter**: Ruff
- **Type checker**: Mypy (warning mode)

```bash
# Auto-format code
ruff format .

# Check code
ruff check .

# Type check
mypy .
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sage_benchmark
```

### Git Hooks

Git hooks are automatically installed by `quickstart.sh` to enforce:
- Code formatting (pre-commit)
- Linting (pre-commit)
- Test passing (pre-push)

## 📝 Adding New Benchmarks

### Structure

```
experiments/
├── exp_X_Y_<name>.py        # Experiment implementation
├── <category>/              # Category-specific experiments
│   ├── __init__.py
│   └── experiment.py
config/
├── exp_X_Y.yaml             # Experiment configuration
docs/
├── exp_X_Y_design.md        # Design document
```

### Experiment Template

```python
"""
Experiment X.Y: Brief description

This experiment measures/validates [what it does].
"""

from typing import Any
import yaml
from pathlib import Path

class ExperimentXY:
    """Brief description of experiment."""
    
    def __init__(self, config_path: Path):
        """Initialize experiment with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
    
    def setup(self) -> None:
        """Set up experiment environment."""
        pass
    
    def run(self) -> dict[str, Any]:
        """Run experiment and return results."""
        pass
    
    def teardown(self) -> None:
        """Clean up experiment resources."""
        pass

def main():
    """CLI entry point."""
    import typer
    app = typer.Typer()
    
    @app.command()
    def run(config: Path = Path("config/exp_X_Y.yaml")):
        """Run experiment."""
        exp = ExperimentXY(config)
        exp.setup()
        results = exp.run()
        exp.teardown()
        print(results)
    
    app()

if __name__ == "__main__":
    main()
```

## 📤 Submitting Changes

### Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <summary>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `ci`, `chore`

**Examples**:
```bash
feat(exp5.1): Add end-to-end pipeline benchmark
fix(config): Correct YAML parsing error
docs(README): Update installation instructions
```

### Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Update CHANGELOG.md** with your changes
4. **Ensure all checks pass**:
   - Code formatting
   - Linting
   - Type checking
   - Tests
5. **Create pull request** with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/results if applicable

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow convention
- [ ] All CI checks pass

## 🐛 Reporting Bugs

Use GitHub Issues with the bug template:

1. **Clear title**: Brief description of the bug
2. **Environment**: OS, Python version, SAGE version
3. **Steps to reproduce**: Minimal example
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Logs**: Error messages and stack traces

## 💡 Requesting Features

Use GitHub Issues with the feature template:

1. **Clear title**: Brief feature description
2. **Problem**: What problem does it solve?
3. **Proposed solution**: How should it work?
4. **Alternatives**: Other approaches considered
5. **Use case**: Example scenario

## 🎓 Resources

- **SAGE Documentation**: https://sage.intellistream.com
- **SAGE Repository**: https://github.com/intellistream/SAGE
- **OmniBenchmark**: https://github.com/intellistream/OmniBenchmark
- **Copilot Instructions**: `.github/copilot-instructions.md`

## 📜 Code of Conduct

- Be respectful and inclusive
- Follow project guidelines
- Accept constructive feedback
- Focus on what's best for the community

## 🔒 Security

Report security vulnerabilities privately to: shuhao_zhang@hust.edu.cn

## 📧 Contact

- **Maintainer**: IntelliStream Team
- **Email**: shuhao_zhang@hust.edu.cn
- **Issues**: https://github.com/intellistream/sage-benchmark/issues

---

## Quick Reference

### Common Commands

```bash
# Install development environment
./quickstart.sh --dev

# Format code
ruff format .

# Lint code
ruff check .

# Run tests
pytest

# Run specific experiment
python -m sage_benchmark.experiments.exp_5_1_e2e_pipeline
```

### File Locations

- Experiments: `experiments/`
- Configurations: `config/`
- Documentation: `docs/`
- Tests: `tests/`

Thank you for contributing! 🚀
