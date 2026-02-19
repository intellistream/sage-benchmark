"""Root conftest – prevents pytest from treating __init__.py as a test module.

The root __init__.py uses a relative import (``from ._version import …``) which
only works when the package is imported through its parent directory.  Pytest's
default collection algorithm would otherwise try to import it as a standalone
module, causing an ``ImportError`` for every test.
"""

from __future__ import annotations

collect_ignore = ["__init__.py"]
