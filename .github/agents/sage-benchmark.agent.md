---
name: sage-benchmark
description: Repository agent for SAGE benchmark modules, experiment configs, and evaluations.
argument-hint: Provide benchmark module path, expected metric/output change, and test scope.
tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'todo', 'vscode.mermaid-chat-features/renderMermaidDiagram', 'github.vscode-pull-request-github/issue_fetch', 'github.vscode-pull-request-github/suggest-fix', 'github.vscode-pull-request-github/searchSyntax', 'github.vscode-pull-request-github/doSearch', 'github.vscode-pull-request-github/renderIssues', 'github.vscode-pull-request-github/activePullRequest', 'github.vscode-pull-request-github/openPullRequest', 'ms-azuretools.vscode-containers/containerToolsConfig', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-toolsai.jupyter/configureNotebook', 'ms-toolsai.jupyter/listNotebookPackages', 'ms-toolsai.jupyter/installNotebookPackages', 'ms-vscode.cpp-devtools/Build_CMakeTools', 'ms-vscode.cpp-devtools/RunCtest_CMakeTools', 'ms-vscode.cpp-devtools/ListBuildTargets_CMakeTools', 'ms-vscode.cpp-devtools/ListTests_CMakeTools']
---

# SAGE Benchmark Agent

## Scope
- `src/sage/benchmark/**`, `config/`, `experiments/`, `tests/`.

## Rules
- Treat repo as L5 benchmark layer; do not introduce upward dependency violations.
- Do not create new local virtual environments (`venv`/`.venv`); use the existing configured Python environment.
- Keep config-driven experiment behavior and typed APIs.
- Respect submodule boundaries (`benchmark_amm`, `benchmark_anns`, `sage.data`).
- Fail fast with explicit errors.

## Workflow
1. Read module README/config first.
2. Make smallest root-cause fix.
3. Validate with focused tests before broad runs.
