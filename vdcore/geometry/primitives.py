"""3D geometric primitives for suspension kinematics.

Coordinate system: ISO 8855 — X+ forward, Y+ LEFT, Z+ up.
Right-handed: X × Y = Z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

X_HAT = np.array([1.0, 0.0, 0.0])
Y_HAT = np.array([0.0, 1.0, 0.0])
Z_HAT = np.array([0.0, 0.0, 1.0])
assert np.allclose(np.cross(X_HAT, Y_HAT), Z_HAT), "Frame is not right-handed"


@dataclass(frozen=True, slots=True)
class Point3D:
    """A 3D point in ISO 8855: X+ forward, Y+ LEFT, Z+ up. Units: mm."""

    x_mm: float
    y_mm: float
    z_mm: float

    def to_array(self) -> npt.NDArray[np.float64]:
        return np.array([self.x_mm, self.y_mm, self.z_mm])

    def distance_to(self, other: Point3D) -> float:
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def __sub__(self, other: Point3D) -> Vector3D:
        return Vector3D(
            x=self.x_mm - other.x_mm,
            y=self.y_mm - other.y_mm,
            z=self.z_mm - other.z_mm,
        )

    @classmethod
    def from_array(cls, arr: npt.NDArray[np.float64]) -> Point3D:
        return cls(x_mm=float(arr[0]), y_mm=float(arr[1]), z_mm=float(arr[2]))


@dataclass(frozen=True, slots=True)
class Vector3D:
    """A 3D vector in ISO 8855: X+ forward, Y+ LEFT, Z+ up. Unitless or mm."""

    x: float
    y: float
    z: float

    def to_array(self) -> npt.NDArray[np.float64]:
        return np.array([self.x, self.y, self.z])

    def magnitude(self) -> float:
        return float(np.linalg.norm(self.to_array()))

    def normalize(self) -> Vector3D:
        mag = self.magnitude()
        if mag < 1e-15:
            raise ValueError("Cannot normalize a zero-length vector")
        return Vector3D(x=self.x / mag, y=self.y / mag, z=self.z / mag)

    def dot(self, other: Vector3D) -> float:
        return float(np.dot(self.to_array(), other.to_array()))

    def cross(self, other: Vector3D) -> Vector3D:
        result = np.cross(self.to_array(), other.to_array())
        return Vector3D(x=float(result[0]), y=float(result[1]), z=float(result[2]))

    def angle_to_deg(self, other: Vector3D) -> float:
        cos_angle = self.dot(other) / (self.magnitude() * other.magnitude())
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    @classmethod
    def from_points(cls, start: Point3D, end: Point3D) -> Vector3D:
        return end - start

    @classmethod
    def from_array(cls, arr: npt.NDArray[np.float64]) -> Vector3D:
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))
