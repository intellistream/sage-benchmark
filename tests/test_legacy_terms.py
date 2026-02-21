"""Guard test: no direct Ray imports or legacy Ray identifiers in source code.

All distributed-execution work must go through sageFlownet / SAGE facade APIs.
"""

from __future__ import annotations

import re
from pathlib import Path

# Repository root is two levels up from this file (tests/ → repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Files and dirs that are explicitly allowed to contain legacy Ray tokens
# (e.g. migration notes in a comment).  Paths relative to REPO_ROOT.
ALLOWLIST: set[Path] = {
    Path("tests/test_legacy_terms.py"),
    Path("experiments/backends/ray_runner.py"),
}

FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "import ray": re.compile(r"\bimport\s+ray\b"),
    "from ray": re.compile(r"\bfrom\s+ray\b"),
    "ray.init()": re.compile(r"\bray\.init\s*\("),
    "ray.remote": re.compile(r"\bray\.remote\b"),
    "RayQueueDescriptor": re.compile(r"\bRayQueueDescriptor\b"),
    "RayQueueManager": re.compile(r"\bRayQueueManager\b"),
    "RayServiceTask": re.compile(r"\bRayServiceTask\b"),
    "ray_task.py": re.compile(r"\bray_task\.py\b"),
    "use_ray=True": re.compile(r"\buse_ray\s*=\s*True\b"),
}


def test_no_legacy_ray_terms_in_source() -> None:
    """No Ray-specific identifiers should appear in Python source files."""
    violations: list[str] = []

    for path in REPO_ROOT.rglob("*.py"):
        # Skip hidden dirs, __pycache__, and explicitly allowlisted paths
        rel = path.relative_to(REPO_ROOT)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if "__pycache__" in rel.parts:
            continue
        if rel in ALLOWLIST:
            continue

        content = path.read_text(encoding="utf-8")
        for term_name, pattern in FORBIDDEN_PATTERNS.items():
            for match in pattern.finditer(content):
                lineno = content.count("\n", 0, match.start()) + 1
                violations.append(f"{rel}:{lineno}: {term_name}")

    assert not violations, (
        "Legacy Ray terms found in source code.  "
        "Use sageFlownet / SAGE facade APIs instead:\n" + "\n".join(sorted(violations))
    )
