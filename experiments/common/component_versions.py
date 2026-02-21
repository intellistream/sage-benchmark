"""Collect SAGE ecosystem component versions for benchmark metadata."""

from __future__ import annotations

from importlib import metadata


CORE_PACKAGES = [
    "isage",
    "isage-common",
    "isage-platform",
    "isage-kernel",
    "isage-libs",
    "isage-middleware",
    "isagellm",
    "isage-flownet",
    "isage-vdb",
    "isage-flow",
    "isage-tsdb",
    "isage-neuromem",
    "isage-refiner",
    "isage-agentic",
    "isage-eval",
    "isage-rag",
]


def collect_component_versions() -> dict[str, str]:
    """Collect versions of core SAGE-related Python packages."""
    installed = {}
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip().lower()
        if name:
            installed[name] = dist.version

    versions: dict[str, str] = {}
    for package in CORE_PACKAGES:
        versions[package] = installed.get(package, "not-installed")

    return versions
