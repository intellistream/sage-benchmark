"""Benchmark and testbed utilities focused on the SAGE system as a whole.

This package is intended for experiments and benchmarks that treat SAGE as
an end-to-end dataflow-based ML systems platform, not just an LLM
control plane. It can host ICML-oriented experiment configs, runners,
analysis code, and writing prompts.
"""

import importlib.util as _util
import os as _os

# Load _version.py by absolute path so the import works regardless of whether
# this __init__.py is executed as part of the ``sage_benchmark`` package or
# as a bare module discovered by pytest's sys.path traversal.
_VERSION_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_version.py")
_spec = _util.spec_from_file_location("_sage_benchmark_version", _VERSION_FILE)
_version_mod = _util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_version_mod)  # type: ignore[union-attr]

__version__: str = _version_mod.__version__
__author__: str = _version_mod.__author__
__email__: str = _version_mod.__email__

__all__ = ["__version__", "__author__", "__email__"]

