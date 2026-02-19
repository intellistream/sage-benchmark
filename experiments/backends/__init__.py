"""
Backend abstraction layer for sage-benchmark.

See backends/base.py for the WorkloadRunner ABC and registry,
and backends/sage_runner.py for the SAGE implementation.

Import individual sub-modules directly to avoid import-time side-effects:

    from backends.base import WorkloadSpec, get_runner
    import backends.sage_runner  # registers the "sage" backend
"""
