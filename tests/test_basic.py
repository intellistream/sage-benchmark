"""Tests for sage_benchmark package."""

import pytest

from sage_benchmark import __version__


def test_version():
    """Test that version is defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__.split(".")) == 4  # MAJOR.MINOR.PATCH.BUILD


def test_imports():
    """Test that basic imports work."""
    import sage_benchmark
    assert hasattr(sage_benchmark, "__version__")
    assert hasattr(sage_benchmark, "__author__")
    assert hasattr(sage_benchmark, "__email__")
