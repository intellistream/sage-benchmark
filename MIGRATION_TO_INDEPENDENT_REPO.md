# sage-benchmark 独立化迁移指南

## 概述

本文档描述如何将 `SAGE/benchmark/` 目录迁移到独立的 `sage-benchmark` 仓库。

**重要说明**:
- **OmniBenchmark**: 组织级的综合 benchmark 集合（`intellistream/OmniBenchmark`），包含所有类型的benchmark
- **sage-benchmark**: SAGE 框架特定的 benchmark 独立仓库（将从 `SAGE/benchmark/` 创建）

## 迁移原因

1. **解耦依赖**: Benchmark 作为评估框架，应该独立于 SAGE 核心
2. **独立发布**: 可以独立发布到 PyPI (`isage-benchmark`)
3. **清晰架构**: 遵循 SAGE 的独立仓库策略 (SageVDB, SageFlow, etc.)
4. **易于维护**: 独立的版本管理和发布周期

## 当前状态

**位置**: `/home/shuhao/SAGE/benchmark/`

**结构**:
```
benchmark/
├── __init__.py                          # 包初始化
├── __main__.py                          # CLI 入口
├── README.md                            # 说明文档
├── config/                              # 配置文件
├── docs/                                # 文档
├── experiments/                         # 实验代码
│   ├── exp_5_1_e2e_pipeline.py         # E2E 管道实验
│   ├── exp_5_2_control_plane.py        # Control Plane 实验
│   ├── exp_5_3_isolation.py            # 隔离性实验
│   ├── exp_5_4_scalability.py          # 可扩展性实验
│   ├── exp_5_5_heterogeneity.py        # 异构性实验
│   ├── tool_use_agent/                 # Agent 工具使用
│   └── ...
├── latex/                               # LaTeX 文件
└── scripts/                             # 脚本工具
```

**依赖的 SAGE 组件**:
- `sage.common` (L1: Foundation)
- `sage.kernel` (L3: Dataflow Engine)
- `sage.middleware` (L4: Operators)
- `sage.libs` (L3: Algorithms)

## 目标状态

**独立仓库**: `https://github.com/intellistream/sage-benchmark`（新创建，SAGE特定）

**PyPI 包名**: `isage-benchmark`

**Python 导入名**: `sage_benchmark` (使用 `sage_libs` 命名空间包的模式)

**注意**: 不要与 OmniBenchmark 混淆（组织级综合benchmark集合）

## 迁移步骤

### 1. 创建独立仓库结构

```bash
# 在本地创建新仓库目录
mkdir -p ~/sage-benchmark
cd ~/sage-benchmark

# 初始化 git 仓库
git init
```

### 2. 复制 benchmark 内容

```bash
# 从 SAGE 复制 benchmark 目录内容（保留 git 历史）
cd ~/SAGE
git subtree split --prefix=benchmark --branch benchmark-split

# 在新仓库中拉取
cd ~/sage-benchmark
git pull ~/SAGE benchmark-split
```

### 3. 创建包结构

创建标准的 Python 包结构：

```
sage-benchmark/
├── .github/
│   ├── copilot-instructions.md         # Copilot 指令
│   └── workflows/                      # CI/CD 配置
├── src/
│   └── sage_benchmark/                 # 源代码
│       ├── __init__.py
│       ├── __main__.py
│       ├── experiments/
│       ├── config/
│       └── ...
├── tests/                              # 测试
├── docs/                               # 文档
├── pyproject.toml                      # 包配置
├── README.md
├── LICENSE
├── CHANGELOG.md
└── .gitignore
```

### 4. 创建 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=70.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "isage-benchmark"
version = "0.2.4.0"
description = "Comprehensive evaluation framework for SAGE AI data processing pipelines"
readme = "README.md"
license = {text = "Apache-2.0"}
authors = [
    {name = "IntelliStream Team", email = "shuhao_zhang@hust.edu.cn"}
]
requires-python = ">=3.10"
dependencies = [
    # SAGE 核心包
    "isage-common>=0.2.4",
    "isage-kernel>=0.2.4",
    "isage-libs>=0.2.4",
    "isage-middleware>=0.2.4",
    
    # 科学计算
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    
    # 测试和评估
    "pytest>=8.0.0",
    "pytest-benchmark>=4.0.0",
    "pytest-cov>=6.0.0",
    
    # 工具
    "pyyaml>=6.0",
    "typer>=0.9.0",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.9.1",
    "mypy>=1.8.0",
    "pre-commit>=3.0.0",
]

[project.scripts]
sage-benchmark = "sage_benchmark.__main__:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"
```

### 5. 更新导入路径

将所有导入路径从：
```python
from sage.benchmark.benchmark_sage.experiments import ...
```

改为：
```python
from sage_benchmark.experiments import ...
```

### 6. 创建 Copilot 指令

在 `.github/copilot-instructions.md` 中添加：

```markdown
# sage-benchmark Copilot Instructions

## Overview

**sage-benchmark** is the comprehensive evaluation framework for SAGE AI data processing pipelines.

## 🚨 CRITICAL Principles

### ❌ NEVER MANUAL PIP INSTALL
All dependencies must be declared in pyproject.toml.

### ❌ NO FALLBACK LOGIC
Follow SAGE's fail-fast principle.

### SAGE Dependency

sage-benchmark depends on SAGE core packages:
- isage-common (L1: Foundation)
- isage-kernel (L3: Dataflow Engine)  
- isage-libs (L3: Algorithms)
- isage-middleware (L4: Operators)

## Benchmark Categories

- **benchmark_agent**: Agent capability evaluation
- **benchmark_control_plane**: Control Plane scheduling evaluation
- **benchmark_memory**: Memory system evaluation
- **benchmark_rag**: RAG pipeline evaluation
- **benchmark_refiner**: Context compression evaluation
- **benchmark_anns**: ANNS algorithm evaluation
- **benchmark_amm**: Approximate matrix multiplication evaluation

## Installation

```bash
pip install isage-benchmark
```

## Usage

```bash
# Run specific experiment
sage-benchmark --experiment 5.1

# Run all experiments
sage-benchmark --all

# With custom config
sage-benchmark --experiment 5.2 --config my_config.yaml
```
```

### 7. 创建 README.md

更新 README 包含：
- 独立仓库说明
- 安装指南
- 使用示例
- 贡献指南
- 与 SAGE 的关系

### 8. 设置 CI/CD

创建 `.github/workflows/` 配置：
- `build-test.yml` - 构建和测试
- `publish-pypi.yml` - PyPI 发布
- `code-quality.yml` - 代码质量检查

### 9. 发布到 GitHub

```bash
cd ~/sage-benchmark
git remote add origin git@github.com:intellistream/sage-benchmark.git
git push -u origin main
```

### 10. 发布到 PyPI

使用 sage-pypi-publisher:

```bash
cd ~/sage-pypi-publisher
./publish.sh sage-benchmark --auto-bump patch
```

## SAGE 主仓库更新

### 1. 更新 Copilot 指令

在 `SAGE/.github/copilot-instructions.md` 中更新 benchmark 相关内容：

```markdown
## sage-benchmark (独立仓库)

**sage-benchmark 已独立为独立仓库**: https://github.com/intellistream/sage-benchmark

Comprehensive evaluation framework for AI data processing pipelines.

To use sage-benchmark:
```bash
pip install isage-benchmark
```

For detailed documentation, see the [sage-benchmark repository](https://github.com/intellistream/sage-benchmark).
```

### 2. 添加迁移说明

在 `SAGE/benchmark/` 目录添加 `MOVED_TO_INDEPENDENT_REPO.md`:

```markdown
# Benchmark 已迁移到独立仓库

**sage-benchmark 已迁移到独立仓库**: https://github.com/intellistream/sage-benchmark

请使用：
```bash
pip install isage-benchmark
```

本目录将在未来版本中移除。
```

### 3. 更新文档

更新 `docs-public/docs_src/dev-notes/` 相关文档，说明 benchmark 已独立。

### 4. 清理主仓库

在确认独立仓库完全可用后：

```bash
cd ~/SAGE
git rm -r benchmark/
git commit -m "chore: remove benchmark directory (moved to sage-benchmark repo)"
```

## 验证清单

- [ ] sage-benchmark 仓库已创建并可访问
- [ ] PyPI 包 `isage-benchmark` 已发布
- [ ] 所有导入路径已更新
- [ ] CI/CD 配置正常工作
- [ ] 测试全部通过
- [ ] 文档完整且准确
- [ ] SAGE 主仓库文档已更新
- [ ] Copilot 指令已更新
- [ ] 可以通过 `pip install isage-benchmark` 安装使用

## 注意事项

1. **版本管理**: sage-benchmark 使用独立的版本号，遵循 SAGE 的四段式版本格式 `0.2.4.0`
2. **依赖版本**: 确保依赖的 SAGE 包版本兼容
3. **Git 历史**: 使用 `git subtree` 保留 commit 历史
4. **文档同步**: 保持独立仓库文档与 SAGE 主文档一致
5. **PyPI 命名**: 使用 `isage-benchmark` (带 'i' 前缀)，导入名为 `sage_benchmark`

## 参考

- **SageVDB 独立化**: `docs-public/docs_src/dev-notes/cross-layer/sagedb-independence-migration.md`
- **PyPI 发布工具**: `https://github.com/intellistream/sage-pypi-publisher`
- **SAGE 架构文档**: `docs-public/docs_src/dev-notes/package-architecture.md`
