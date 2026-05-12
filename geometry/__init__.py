"""
geometry
========
Pacote de geometria do motor de suspensão FSAE.

Módulos:
    primitives  : tipos básicos (Point3D, Vector3D, Point2D) + interseções
    solver_2d   : mecanismo de 4 barras na vista frontal Y-Z
    model_3d    : classes OOP (ControlArm, SuspensionCorner, Vehicle)
    solver_3d   : solver cinemático 3D (interseção de 3 esferas + LM)
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