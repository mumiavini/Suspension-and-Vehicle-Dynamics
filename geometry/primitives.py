"""
primitives.py
=============
Tipos geométricos fundamentais para o motor de suspensão FSAE.
Todos os eixos seguem a convenção SAE:
  X → frente do veículo
  Y → esquerda do veículo
  Z → para cima
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Point3D
# ---------------------------------------------------------------------------

@dataclass
class Point3D:
    """Ponto no espaço 3D com coordenadas cartesianas (mm)."""

    x: float
    y: float
    z: float

    # ------------------------------------------------------------------
    # Conversões / helpers numpy
    # ------------------------------------------------------------------

    def to_array(self) -> NDArray[np.float64]:
        """Retorna coordenadas como vetor numpy (3,)."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Point3D":
        """Cria Point3D (Tensor) a partir de array numpy."""
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    # ------------------------------------------------------------------
    # Operadores aritméticos
    # ------------------------------------------------------------------

    def __sub__(self, other: "Point3D") -> "Vector3D":
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: "Vector3D") -> "Point3D":
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __repr__(self) -> str:
        return f"Point3D(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def distance_to(self, other: "Point3D") -> float:
        """Distância euclidiana entre dois pontos (mm)."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def midpoint(self, other: "Point3D") -> "Point3D":
        return Point3D.from_array((self.to_array() + other.to_array()) / 2.0)

    def project_yz(self) -> "Point2D":
        """Projeta o ponto no plano YZ (vista frontal)."""
        return Point2D(self.y, self.z)

    def project_xz(self) -> "Point2D":
        """Projeta o ponto no plano XZ (vista lateral)."""
        return Point2D(self.x, self.z)


# ---------------------------------------------------------------------------
# Point2D  (usado nos cálculos 2D da vista frontal)
# ---------------------------------------------------------------------------

@dataclass
class Point2D:
    """Ponto no plano 2D (coordenadas em mm)."""

    u: float   # eixo horizontal (Y na vista frontal, X na vista lateral)
    v: float   # eixo vertical   (Z em ambas as vistas)

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.u, self.v], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Point2D":
        return cls(float(arr[0]), float(arr[1]))

    def distance_to(self, other: "Point2D") -> float:
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def __repr__(self) -> str:
        return f"Point2D(u={self.u:.3f}, v={self.v:.3f})"


# ---------------------------------------------------------------------------
# Vector3D
# ---------------------------------------------------------------------------

@dataclass
class Vector3D:
    """Vetor livre no espaço 3D."""

    x: float
    y: float
    z: float

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Vector3D":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    @classmethod
    def from_points(cls, origin: Point3D, tip: Point3D) -> "Vector3D":
        """Vetor de `origin` para `tip`."""
        return tip - origin

    # ------------------------------------------------------------------
    # Álgebra
    # ------------------------------------------------------------------

    def magnitude(self) -> float:
        return float(np.linalg.norm(self.to_array()))

    def normalize(self) -> "Vector3D":
        mag = self.magnitude()
        if mag < 1e-12:
            raise ValueError("Não é possível normalizar vetor nulo.")
        return Vector3D.from_array(self.to_array() / mag)

    def dot(self, other: "Vector3D") -> float:
        return float(np.dot(self.to_array(), other.to_array()))

    def cross(self, other: "Vector3D") -> "Vector3D":
        return Vector3D.from_array(np.cross(self.to_array(), other.to_array()))

    def angle_to_deg(self, other: "Vector3D") -> float:
        """Ângulo entre dois vetores em graus [0, 180]."""
        cos_a = np.clip(
            self.dot(other) / (self.magnitude() * other.magnitude()), -1.0, 1.0
        )
        return float(math.degrees(math.acos(cos_a)))

    def __add__(self, other: "Vector3D") -> "Vector3D":
        return Vector3D.from_array(self.to_array() + other.to_array())

    def __mul__(self, scalar: float) -> "Vector3D":
        return Vector3D.from_array(self.to_array() * scalar)

    def __repr__(self) -> str:
        return f"Vector3D(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


# ---------------------------------------------------------------------------
# Utilitários de interseção (usados pelo solucionador 2D)
# ---------------------------------------------------------------------------

def circle_circle_intersection(
    c1: Point2D, r1: float,
    c2: Point2D, r2: float,
    prefer_positive_v: bool = True
) -> Point2D:
    """
    Calcula o(s) ponto(s) de interseção entre dois círculos no plano 2D.

    Parâmetros
    ----------
    c1, r1 : centro e raio do primeiro círculo
    c2, r2 : centro e raio do segundo círculo
    prefer_positive_v : se True, retorna o ponto com maior coordenada v
                        (útil para resolver a posição da manga de eixo)

    Retorna
    -------
    Point2D com a solução escolhida.

    Levanta
    -------
    ValueError se os círculos não se intersectam ou são concêntricos.
    """
    p1 = c1.to_array()
    p2 = c2.to_array()

    d = float(np.linalg.norm(p2 - p1))

    if d < 1e-12:
        raise ValueError("Os centros dos círculos são coincidentes.")
    if d > r1 + r2 + 1e-9:
        raise ValueError(
            f"Círculos não se intersectam: d={d:.2f}, r1+r2={r1+r2:.2f}"
        )
    if d < abs(r1 - r2) - 1e-9:
        raise ValueError("Um círculo está contido no outro.")

    # Distância do centro 1 ao ponto médio da corda de interseção
    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h_sq = r1**2 - a**2
    h = math.sqrt(max(h_sq, 0.0))

    # Ponto médio da corda
    mid = p1 + a * (p2 - p1) / d

    # Vetor perpendicular à linha que une os centros
    perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / d

    sol_a = Point2D.from_array(mid + h * perp)
    sol_b = Point2D.from_array(mid - h * perp)

    if prefer_positive_v:
        return sol_a if sol_a.v >= sol_b.v else sol_b
    else:
        return sol_a if sol_a.v < sol_b.v else sol_b


def line_intersection_2d(
    p1: Point2D, p2: Point2D,
    p3: Point2D, p4: Point2D
) -> Point2D:
    """
    Interseção entre duas retas (definidas por dois pontos cada) no plano 2D.

    Retorna
    -------
    Point2D de interseção.

    Levanta
    -------
    ValueError se as retas são paralelas.
    """
    x1, y1 = p1.u, p1.v
    x2, y2 = p2.u, p2.v
    x3, y3 = p3.u, p3.v
    x4, y4 = p4.u, p4.v

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        raise ValueError("As retas são paralelas ou coincidentes.")

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return Point2D(ix, iy)