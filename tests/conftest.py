"""Pytest configuration for sage-benchmark tests."""

import sys
from pathlib import Path

# Add sage_benchmark to path
sage_benchmark_root = Path(__file__).parent.parent
sys.path.insert(0, str(sage_benchmark_root))

# Add experiments/ to path so that `experiments.backends.*` can be imported
# without triggering experiments/__init__.py (which pulls sage.benchmark
# packages that may not be installed in a lightweight test environment).
experiments_dir = sage_benchmark_root / "experiments"
if str(experiments_dir) not in sys.path:
    sys.path.insert(0, str(experiments_dir))
