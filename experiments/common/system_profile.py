"""System profile helpers for benchmark result metadata."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path


def _read_cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        text = cpuinfo.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
    return platform.processor() or "unknown"


def _read_memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    text = meminfo.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", text, flags=re.MULTILINE)
    if not match:
        return None
    kb = int(match.group(1))
    return round(kb / 1024 / 1024, 2)


def _read_gpu_name() -> str | None:
    if not Path("/usr/bin/nvidia-smi").exists() and not Path("/bin/nvidia-smi").exists():
        return None
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not names:
        return None
    first = names[0]
    return first if len(names) == 1 else f"{first} (+{len(names)-1} more)"


def collect_system_profile() -> dict[str, object]:
    """Collect lightweight machine profile for benchmark metadata."""
    return {
        "hostname": platform.node() or os.environ.get("HOSTNAME", "unknown"),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine() or "unknown",
        "cpu_model": _read_cpu_model(),
        "cpu_physical_cores": os.cpu_count(),
        "memory_gb": _read_memory_gb(),
        "gpu": _read_gpu_name(),
        "python": platform.python_version(),
    }
