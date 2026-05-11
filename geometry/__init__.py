"""
fsae_suspension
===============
Motor de cálculo para geometria de suspensão de Fórmula SAE.

Módulos
-------
geometry.primitives  : tipos geométricos base (Point3D, Vector3D, Point2D)
geometry.solver_2d   : solucionador cinemático 2D (vista frontal)
geometry.model_3d    : modelo OOP 3D (ControlArm, SuspensionCorner, Vehicle)
"""

from geometry.primitives import Point3D, Point2D, Vector3D
from geometry.solver_2d import SuspensionGeometry2D, analyze_heave
from geometry.model_3d import ControlArm, KingpinGeometry, SuspensionCorner, Vehicle

__all__ = [
    "Point3D", "Point2D", "Vector3D",
    "SuspensionGeometry2D", "analyze_heave",
    "ControlArm", "KingpinGeometry", "SuspensionCorner", "Vehicle",
]