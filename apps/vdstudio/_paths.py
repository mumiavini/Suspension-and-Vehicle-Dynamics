"""Put the repository root and ``scripts/`` on ``sys.path``.

The design scripts (``sla_geometry.py``, ``steering_geometry.py``) live at the
repository root and ``geometry_summary.py`` / the exporters live in
``scripts/``. They are imported as top-level modules, exactly as the test suite
and ``scripts/geometry_summary.py`` do, so the app runs the same code the
summary document is generated from.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"


def ensure_paths() -> None:
    for p in (REPO, SCRIPTS):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


ensure_paths()
