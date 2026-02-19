"""
Backend runner interface for sage-benchmark.

Defines the core abstraction that lets the same workload specification be
executed against different runtime backends (SAGE, Ray, …).

Architecture
------------
- :class:`WorkloadSpec`   – backend-agnostic description of what to run.
- :class:`RunResult`      – backend-agnostic container for execution output.
- :class:`WorkloadRunner` – ABC that each backend must implement.
- :func:`register_runner` – decorator to register a runner class by name.
- :func:`get_runner`      – factory that returns a runner instance by name.
- :func:`list_backends`   – enumerate all registered backend names.

Adding a new backend
--------------------
1. Create ``experiments/backends/<name>_runner.py``.
2. Implement :class:`WorkloadRunner` and decorate with ``@register_runner("<name>")``.
3. Import the module wherever you need it (or call :func:`load_all_backends`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data classes (backend-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class WorkloadSpec:
    """Backend-agnostic workload specification.

    All fields are intentionally generic so that any backend can map them
    to its own execution primitives.

    Attributes
    ----------
    name:
        Human-readable workload name; used as a label in results.
    total_items:
        Number of items the workload source should emit.
    parallelism:
        Desired operator parallelism (hint; backend may ignore it).
    scheduler_name:
        Scheduler strategy hint (e.g. ``"fifo"``, ``"load_aware"``).
    extra:
        Extension dict for backend-specific knobs that do not belong in the
        common interface (e.g. ``{"max_wait_seconds": 60}``).
    """

    name: str
    total_items: int = 10
    parallelism: int = 2
    scheduler_name: str = "default"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result from running a workload with a backend.

    Attributes
    ----------
    backend:
        Name of the backend that produced this result.
    scheduler_name:
        Scheduler strategy that was used.
    elapsed_time:
        Wall-clock execution time in seconds.
    results_count:
        Number of output items collected by the sink.
    metrics:
        Backend-specific metrics dictionary (e.g. scheduler statistics).
    """

    backend: str
    scheduler_name: str
    elapsed_time: float
    results_count: int
    metrics: dict[str, Any]

    def summary(self) -> str:
        """Return a human-readable one-block summary string."""
        lines = [
            f"Backend     : {self.backend}",
            f"Scheduler   : {self.scheduler_name}",
            f"Elapsed (s) : {self.elapsed_time:.3f}",
            f"Output items: {self.results_count}",
        ]
        if self.metrics:
            lines.append("Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner interface
# ---------------------------------------------------------------------------


class WorkloadRunner(ABC):
    """Abstract base class for workload runners.

    Each runtime backend (SAGE, Ray, …) must provide a concrete subclass and
    register it with :func:`register_runner`.  Benchmark entry-points use
    :func:`get_runner` to obtain the right instance based on a ``--backend``
    CLI flag, keeping workload logic completely backend-agnostic.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Canonical lowercase name for this backend (e.g. ``"sage"``)."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this backend is installed and usable right now.

        Implementations should do a lightweight import check; they should
        *not* start any cluster or heavy initialisation here.
        """
        ...

    @abstractmethod
    def run(self, spec: WorkloadSpec) -> RunResult:
        """Execute the workload described by *spec* and return results.

        Parameters
        ----------
        spec:
            Backend-agnostic description of the workload to run.

        Returns
        -------
        RunResult
            Collected metrics from the completed run.
        """
        ...


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[WorkloadRunner]] = {}


def register_runner(name: str):
    """Class decorator: register a :class:`WorkloadRunner` subclass by name.

    Example
    -------
    .. code-block:: python

        @register_runner("my_backend")
        class MyRunner(WorkloadRunner):
            ...
    """

    def decorator(cls: type[WorkloadRunner]) -> type[WorkloadRunner]:
        _REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_runner(backend: str) -> WorkloadRunner:
    """Instantiate and return a runner for the given *backend* name.

    The runner's :meth:`WorkloadRunner.is_available` is checked before
    returning; an error is raised when the backend packages are missing.

    Parameters
    ----------
    backend:
        Case-insensitive backend name (e.g. ``"sage"``).

    Raises
    ------
    ValueError
        If *backend* has not been registered.
    RuntimeError
        If the backend is registered but its packages are not installed.
    """
    key = backend.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise ValueError(
            f"Unknown backend '{backend}'. Available backends: {available}. "
            f"Make sure the corresponding runner module is imported."
        )
    runner = _REGISTRY[key]()
    if not runner.is_available():
        raise RuntimeError(
            f"Backend '{backend}' is registered but not available in this environment. "
            f"Ensure the required packages are installed and try again."
        )
    return runner


def list_backends() -> list[str]:
    """Return sorted list of all currently registered backend names."""
    return sorted(_REGISTRY)
