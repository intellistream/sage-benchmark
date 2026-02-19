"""
Backend abstraction layer for sage-benchmark.

See backends/base.py for the WorkloadRunner ABC and registry,
and backends/sage_runner.py for the SAGE implementation.
and backends/ray_runner.py for the Ray MVP baseline (Workload4).

Import individual sub-modules directly to avoid import-time side-effects:

    from backends.base import WorkloadSpec, get_runner
    import backends.sage_runner  # registers the "sage" backend
    import backends.ray_runner   # registers the "ray" backend (requires Ray)
"""
