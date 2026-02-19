# Backend Abstraction Layer

## Why this exists

`sage-benchmark` compares SAGE against other runtime backends.  The naive
approach—hard-coding `LocalEnvironment` calls in every experiment—couples
benchmark logic to SAGE internals and makes it impossible to add a second
baseline (e.g. Ray) without editing every workload file.

The backend abstraction solves this: **workload logic lives in one place;
backend wiring lives in another.**

## Architecture

```
experiments/
└── backends/
    ├── __init__.py        # public re-exports
    ├── base.py            # WorkloadRunner ABC + registry + factory
    └── sage_runner.py     # concrete SAGE implementation
```

### Key types

| Type | Role |
|------|------|
| `WorkloadSpec` | Backend-agnostic workload description (items, parallelism, scheduler hint, extra knobs) |
| `RunResult` | Backend-agnostic result container (elapsed time, item count, metric dict) |
| `WorkloadRunner` | ABC every backend must implement (`backend_name`, `is_available`, `run`) |

### Registry pattern

Runners register themselves via a decorator:

```python
# experiments/backends/my_runner.py
from experiments.backends.base import WorkloadRunner, register_runner

@register_runner("my_backend")
class MyRunner(WorkloadRunner):
    ...
```

The registry is populated at import time.  An experiment entry-point imports
the relevant runner modules and then calls `get_runner("my_backend")`.

## Decision: why NOT add Ray to SAGE core

Per the SAGE architectural direction, `ray` must **not** be imported inside
any SAGE core repository (`SAGE/`, `sageFlownet/`, …).  A Ray runner would
live in `sage-benchmark` only—as another file under `experiments/backends/`.

The workload business logic (sources, operators, sinks) is expressed in
terms of the `WorkloadSpec` dataclass and is fully independent of whichever
backend runs it.  The SAGE runner wraps `FlownetEnvironment` — the correct distributed runtime
backed by sageFlownet — into the `WorkloadSpec` → `RunResult` contract.
`env.submit(autostop=True)` blocks until the pipeline completes and tears down
the environment automatically, so no polling loop is needed in the runner.

A hypothetical Ray runner would translate the same spec into `ray.remote`
calls.

## Adding a new backend

1. Create `experiments/backends/<name>_runner.py`.
2. Implement the three abstract methods:
   - `backend_name` – return a lowercase string (used as the `--backend` value).
   - `is_available` – do a lightweight import check, return `bool`.
   - `run(spec)` – execute the workload, return `RunResult`.
3. Decorate the class with `@register_runner("<name>")`.
4. Import the module at the experiment entry-point **before** calling
   `get_runner()`, e.g.:

   ```python
   import experiments.backends.sage_runner   # registers "sage"
   import experiments.backends.ray_runner    # registers "ray" (future)
   ```

5. Add a unit test in `tests/test_backend_selector.py` covering dispatch.

## CLI wiring example

`scheduler_comparison.py` demonstrates how to expose the backend choice:

```
python experiments/scheduler_comparison.py --backend sage
```

The `--backend` argument defaults to `"sage"` so existing integrations are
unaffected.  Passing `--backend ray` (once `ray_runner.py` exists) would run
the identical workload through Ray without changing any other code.

## Testing strategy

- **Unit tests** (`tests/test_backend_selector.py`): mock `WorkloadRunner`
  subclasses to verify registry dispatch and factory error handling without
  touching any real environment.
- **Integration tests**: run `scheduler_comparison.py --backend sage` inside
  CI to verify the SAGE path produces the expected output count.
