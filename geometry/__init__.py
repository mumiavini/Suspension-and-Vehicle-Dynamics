"""
geometry
========
Geometry package of the FSAE suspension engine.

Modules:
    primitives  : basic types (Point3D, Vector3D, Point2D) + intersections
    solver_2d   : four-bar mechanism in the Y-Z front view
    model_3d    : OOP classes (ControlArm, SuspensionCorner, Vehicle)
    solver_3d   : 3D kinematic solver (3-sphere intersection + LM)
"""

from geometry.primitives import Point3D, Point2D, Vector3D
from geometry.solver_2d import SuspensionGeometry2D, analyze_heave, CamberAnalysis
from geometry.model_3d import ControlArm, KingpinGeometry, SuspensionCorner, Vehicle
from geometry.solver_3d import TieRod, KinematicSolver3D, KinematicState3D

__all__ = [
    # primitives
    "Point3D", "Point2D", "Vector3D",
    # solver_2d
    "SuspensionGeometry2D", "analyze_heave", "CamberAnalysis",
    # model_3d
    "ControlArm", "KingpinGeometry", "SuspensionCorner", "Vehicle",
    # solver_3d
    "TieRod", "KinematicSolver3D", "KinematicState3D",
]
