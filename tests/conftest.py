"""Pytest configuration for sage-benchmark tests."""

import sys
from pathlib import Path

# Add sage_benchmark to path
sage_benchmark_root = Path(__file__).parent.parent
sys.path.insert(0, str(sage_benchmark_root))
