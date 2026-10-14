"""VD Studio -- the PUCPR Racing desktop workbench for suspension and steering.

One application that carries the car from the first concept numbers to the
validation of telemetry: concept parameters, SLA hardpoint synthesis, steering
synthesis, 3D kinematic simulation, trade-off sweeps, quasi-static maneuvers,
telemetry checks, reports and CAD exports.

Layering:
    apps/vdstudio/engine/   Qt-free application logic. Imports vdcore and the
                            repository design scripts (sla_geometry.py,
                            steering_geometry.py, scripts/geometry_summary.py).
                            Testable from a bare REPL.
    apps/vdstudio/ui/       PySide6 + pyqtgraph presentation only.

Design principle (CLAUDE.md): the tool computes trade-offs; the designer picks
the geometry. Nothing in this package presents a recommended value.
"""

from apps.vdstudio import _paths  # noqa: F401  (puts the repo root and scripts/ on sys.path)

__version__ = "0.1.0"
APP_NAME = "VD Studio"
