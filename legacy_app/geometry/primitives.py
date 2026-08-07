"""
geometry/primitives.py
======================
Basic geometric types of the suspension engine.

AXIS CONVENTION (SAE J670):
    X : points toward the FRONT of the vehicle
    Y : points to the LEFT of the vehicle
    Z : points UP

UNITS:
    - Lengths in millimeters (mm)
    - Angles in degrees (°)

This module contains three main classes:
    - Point3D      : point in 3D space (X, Y, Z)
    - Point2D      : 2D point, used for front-view analyses (Y-Z)
    - Vector3D     : free vector in 3D space, with linear-algebra operations
And two utility functions:
    - circle_circle_intersection : intersection of two circles in the plane
    - line_intersection_2d       : intersection of two lines in the plane
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Point3D — Point in 3D space
# =============================================================================

@dataclass
class Point3D:
    """
    3D Cartesian point (X, Y, Z), in millimeters.

    Example:
        >>> p = Point3D(100.0, 50.0, 200.0)
        >>> p.x, p.y, p.z
        (100.0, 50.0, 200.0)
    """
    x: float
    y: float
    z: float

    # -------------------------------------------------------------------------
    # Conversion to/from NumPy (required for linear algebra)
    # -------------------------------------------------------------------------

    def to_array(self) -> NDArray[np.float64]:
        """Convert the point to a numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Point3D":
        """Create a Point3D from a numpy array [x, y, z]."""
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    # -------------------------------------------------------------------------
    # Arithmetic operators
    # -------------------------------------------------------------------------

    def __sub__(self, other: "Point3D") -> "Vector3D":
        """P1 - P2 returns the vector from P2 to P1."""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, vec: "Vector3D") -> "Point3D":
        """P + V translates the point by the vector."""
        return Point3D(self.x + vec.x, self.y + vec.y, self.z + vec.z)

    def __repr__(self) -> str:
        return f"Point3D(x={self.x:7.2f}, y={self.y:7.2f}, z={self.z:7.2f})"

    # -------------------------------------------------------------------------
    # Geometric methods
    # -------------------------------------------------------------------------

    def distance_to(self, other: "Point3D") -> float:
        """Euclidean distance between two points (mm)."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def midpoint(self, other: "Point3D") -> "Point3D":
        """Midpoint between two points."""
        return Point3D.from_array((self.to_array() + other.to_array()) / 2.0)

    def project_yz(self) -> "Point2D":
        """Projection onto the front plane Y-Z (drops X)."""
        return Point2D(self.y, self.z)

    def project_xz(self) -> "Point2D":
        """Projection onto the side plane X-Z (drops Y)."""
        return Point2D(self.x, self.z)


# =============================================================================
# Point2D — Point in the plane (used in the front view)
# =============================================================================

@dataclass
class Point2D:
    """
    2D Cartesian point, in millimeters.

    Generic axes (u, v) to avoid coupling to a specific plane:
        - In the front view (Y-Z): u = Y, v = Z
        - In the side view (X-Z):  u = X, v = Z
    """
    u: float   # horizontal coordinate of the plane
    v: float   # vertical coordinate of the plane

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.u, self.v], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Point2D":
        return cls(float(arr[0]), float(arr[1]))

    def distance_to(self, other: "Point2D") -> float:
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def __repr__(self) -> str:
        return f"Point2D(u={self.u:7.2f}, v={self.v:7.2f})"


# =============================================================================
# Vector3D — Free vector in 3D space
# =============================================================================

@dataclass
class Vector3D:
    """
    Free vector in 3D space. Supports the usual linear-algebra operations:
    addition, scalar multiplication, dot product, cross product,
    normalization and angle computation.
    """
    x: float
    y: float
    z: float

    # -------------------------------------------------------------------------
    # Constructors and conversions
    # -------------------------------------------------------------------------

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Vector3D":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    @classmethod
    def from_points(cls, origin: Point3D, tip: Point3D) -> "Vector3D":
        """Vector pointing from `origin` to `tip`."""
        return tip - origin

    # -------------------------------------------------------------------------
    # Algebra operations
    # -------------------------------------------------------------------------

    def magnitude(self) -> float:
        """Euclidean norm (length) of the vector."""
        return float(np.linalg.norm(self.to_array()))

    def normalize(self) -> "Vector3D":
        """Return a unit vector with the same direction."""
        mag = self.magnitude()
        if mag < 1e-12:
            raise ValueError("Cannot normalize a null vector.")
        return Vector3D.from_array(self.to_array() / mag)

    def dot(self, other: "Vector3D") -> float:
        """Dot product."""
        return float(np.dot(self.to_array(), other.to_array()))

    def cross(self, other: "Vector3D") -> "Vector3D":
        """Cross product (follows the right-hand rule)."""
        return Vector3D.from_array(np.cross(self.to_array(), other.to_array()))

    def angle_to_deg(self, other: "Vector3D") -> float:
        """
        Angle between two vectors, in degrees, in the range [0°, 180°].

        Computed via: cos(θ) = (v1 · v2) / (|v1| × |v2|)
        """
        cos_theta = self.dot(other) / (self.magnitude() * other.magnitude())
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))   # numerical guard
        return math.degrees(math.acos(cos_theta))

    # -------------------------------------------------------------------------
    # Operators
    # -------------------------------------------------------------------------

    def __add__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D.from_array(self.to_array() + other.to_array())

    def __sub__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D.from_array(self.to_array() - other.to_array())

    def __mul__(self, scalar: float) -> "Vector3D":
        return Vector3D.from_array(self.to_array() * scalar)

    def __repr__(self) -> str:
        return f"Vector3D(x={self.x:+.4f}, y={self.y:+.4f}, z={self.z:+.4f})"


# =============================================================================
# Intersections in the 2D plane
# =============================================================================

def circle_circle_intersection(
    c1: Point2D, r1: float,
    c2: Point2D, r2: float,
    prefer_positive_v: bool = True,
) -> Point2D:
    """
    Intersection between two circles in the 2D plane.

    Two circles may have 0, 1 or 2 intersection points. When there are two,
    we pick the one with the larger `v` coordinate (if `prefer_positive_v=True`)
    or the smaller one (if False). This is needed in the four-bar mechanism to
    select the physically correct solution (upright "facing up").

    Algorithm (classic):
        d = distance between centers
        a = (r1² - r2² + d²) / (2·d)        → projection of the chord midpoint
        h = √(r1² - a²)                       → half of the chord
        mid = c1 + a·(c2-c1)/d                → chord midpoint
        perp = unit vector perpendicular to line c1-c2
        solutions = mid ± h·perp

    Raises ValueError if the circles do not intersect or are concentric.
    """
    p1 = c1.to_array()
    p2 = c2.to_array()

    d = float(np.linalg.norm(p2 - p1))

    # --- Degenerate cases ---
    if d < 1e-12:
        raise ValueError("Coincident circle centers.")
    if d > r1 + r2 + 1e-9:
        raise ValueError(
            f"Circles do not intersect: d={d:.2f}, r1+r2={r1+r2:.2f}"
        )
    if d < abs(r1 - r2) - 1e-9:
        raise ValueError("One circle is contained within the other.")

    # --- Standard computation ---
    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h = math.sqrt(max(r1**2 - a**2, 0.0))   # max() guards against numerical error

    mid = p1 + a * (p2 - p1) / d
    perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / d   # unit perpendicular

    sol_a = Point2D.from_array(mid + h * perp)
    sol_b = Point2D.from_array(mid - h * perp)

    # Solution selection
    if prefer_positive_v:
        return sol_a if sol_a.v >= sol_b.v else sol_b
    else:
        return sol_a if sol_a.v < sol_b.v else sol_b


def line_intersection_2d(
    p1: Point2D, p2: Point2D,
    p3: Point2D, p4: Point2D,
) -> Point2D:
    """
    Intersection between two lines in the 2D plane, each defined by two points.

    Uses the determinant formula for line intersection.
    Raises ValueError if the lines are parallel or coincident.
    """
    x1, y1 = p1.u, p1.v
    x2, y2 = p2.u, p2.v
    x3, y3 = p3.u, p3.v
    x4, y4 = p4.u, p4.v

    # Common denominator (cross product of the directions)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        raise ValueError("The lines are parallel or coincident.")

    # Parameter t on line 1
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    # Intersection point = P1 + t · (P2 - P1)
    return Point2D(
        u=x1 + t * (x2 - x1),
        v=y1 + t * (y2 - y1),
    )
