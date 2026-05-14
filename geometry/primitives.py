"""
geometry/primitives.py
======================
Tipos geométricos básicos do motor de suspensão.

CONVENÇÃO DE EIXOS (SAE J670):
    X : aponta para a FRENTE do veículo
    Y : aponta para a ESQUERDA do veículo
    Z : aponta para CIMA

UNIDADES:
    - Comprimentos em milímetros (mm)
    - Ângulos em graus (°)

Este módulo contém três classes principais:
    - Point3D      : ponto no espaço 3D (X, Y, Z)
    - Point2D      : ponto em 2D, usado para análises na vista frontal (Y-Z)
    - Vector3D     : vetor livre no espaço 3D, com operações de álgebra linear
E duas funções utilitárias:
    - circle_circle_intersection : interseção de dois círculos no plano
    - line_intersection_2d       : interseção de duas retas no plano
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Point3D — Ponto no espaço 3D
# =============================================================================

@dataclass
class Point3D:
    """
    Ponto cartesiano 3D (X, Y, Z), em milímetros.

    Exemplo:
        >>> p = Point3D(100.0, 50.0, 200.0)
        >>> p.x, p.y, p.z
        (100.0, 50.0, 200.0)
    """
    x: float
    y: float
    z: float

    # -------------------------------------------------------------------------
    # Conversão para/de NumPy (necessário para álgebra linear)
    # -------------------------------------------------------------------------

    def to_array(self) -> NDArray[np.float64]:
        """Converte o ponto para um numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Point3D":
        """Cria Point3D a partir de um array numpy [x, y, z]."""
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    # -------------------------------------------------------------------------
    # Operadores aritméticos
    # -------------------------------------------------------------------------

    def __sub__(self, other: "Point3D") -> "Vector3D":
        """P1 - P2 retorna o vetor de P2 para P1."""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, vec: "Vector3D") -> "Point3D":
        """P + V translada o ponto pelo vetor."""
        return Point3D(self.x + vec.x, self.y + vec.y, self.z + vec.z)

    def __repr__(self) -> str:
        return f"Point3D(x={self.x:7.2f}, y={self.y:7.2f}, z={self.z:7.2f})"

    # -------------------------------------------------------------------------
    # Métodos geométricos
    # -------------------------------------------------------------------------

    def distance_to(self, other: "Point3D") -> float:
        """Distância euclidiana entre dois pontos (mm)."""
        return float(np.linalg.norm(self.to_array() - other.to_array()))

    def midpoint(self, other: "Point3D") -> "Point3D":
        """Ponto médio entre dois pontos."""
        return Point3D.from_array((self.to_array() + other.to_array()) / 2.0)

    def project_yz(self) -> "Point2D":
        """Projeção no plano frontal Y-Z (descarta X)."""
        return Point2D(self.y, self.z)

    def project_xz(self) -> "Point2D":
        """Projeção no plano lateral X-Z (descarta Y)."""
        return Point2D(self.x, self.z)


# =============================================================================
# Point2D — Ponto no plano (usado na vista frontal)
# =============================================================================

@dataclass
class Point2D:
    """
    Ponto cartesiano 2D, em milímetros.

    Eixos genéricos (u, v) para evitar acoplar a um plano específico:
        - Na vista frontal (Y-Z): u = Y, v = Z
        - Na vista lateral (X-Z): u = X, v = Z
    """
    u: float   # coordenada horizontal do plano
    v: float   # coordenada vertical do plano

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
# Vector3D — Vetor livre no espaço 3D
# =============================================================================

@dataclass
class Vector3D:
    """
    Vetor livre no espaço 3D. Suporta as operações usuais de álgebra linear:
    soma, multiplicação por escalar, produto escalar (dot), produto vetorial
    (cross), normalização e cálculo de ângulo.
    """
    x: float
    y: float
    z: float

    # -------------------------------------------------------------------------
    # Construtores e conversões
    # -------------------------------------------------------------------------

    def to_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> "Vector3D":
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    @classmethod
    def from_points(cls, origin: Point3D, tip: Point3D) -> "Vector3D":
        """Vetor que aponta de `origin` para `tip`."""
        return tip - origin

    # -------------------------------------------------------------------------
    # Operações de álgebra
    # -------------------------------------------------------------------------

    def magnitude(self) -> float:
        """Norma euclidiana (comprimento) do vetor."""
        return float(np.linalg.norm(self.to_array()))

    def normalize(self) -> "Vector3D":
        """Retorna um vetor unitário com a mesma direção."""
        mag = self.magnitude()
        if mag < 1e-12:
            raise ValueError("Não é possível normalizar um vetor nulo.")
        return Vector3D.from_array(self.to_array() / mag)

    def dot(self, other: "Vector3D") -> float:
        """Produto escalar."""
        return float(np.dot(self.to_array(), other.to_array()))

    def cross(self, other: "Vector3D") -> "Vector3D":
        """Produto vetorial (segue a regra da mão direita)."""
        return Vector3D.from_array(np.cross(self.to_array(), other.to_array()))

    def angle_to_deg(self, other: "Vector3D") -> float:
        """
        Ângulo entre dois vetores, em graus, no intervalo [0°, 180°].

        Calculado via: cos(θ) = (v1 · v2) / (|v1| × |v2|)
        """
        cos_theta = self.dot(other) / (self.magnitude() * other.magnitude())
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))   # proteção numérica
        return math.degrees(math.acos(cos_theta))

    # -------------------------------------------------------------------------
    # Operadores
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
# Interseções no plano 2D
# =============================================================================

def circle_circle_intersection(
    c1: Point2D, r1: float,
    c2: Point2D, r2: float,
    prefer_positive_v: bool = True,
) -> Point2D:
    """
    Interseção entre dois círculos no plano 2D.

    Dois círculos podem ter 0, 1 ou 2 pontos de interseção. Quando há dois,
    escolhemos o de maior coordenada `v` (se `prefer_positive_v=True`) ou o
    de menor (se False). Isso é necessário no mecanismo de 4 barras para
    selecionar a solução fisicamente correta (manga "para cima").

    Algoritmo (clássico):
        d = distância entre os centros
        a = (r1² - r2² + d²) / (2·d)        → projeção do ponto médio da corda
        h = √(r1² - a²)                       → metade da corda
        mid = c1 + a·(c2-c1)/d                → ponto médio da corda
        perp = vetor perpendicular à linha c1-c2, unitário
        soluções = mid ± h·perp

    Levanta ValueError se os círculos não se intersectam ou são concêntricos.
    """
    p1 = c1.to_array()
    p2 = c2.to_array()

    d = float(np.linalg.norm(p2 - p1))

    # --- Casos degenerados ---
    if d < 1e-12:
        raise ValueError("Centros dos círculos coincidentes.")
    if d > r1 + r2 + 1e-9:
        raise ValueError(
            f"Círculos não se intersectam: d={d:.2f}, r1+r2={r1+r2:.2f}"
        )
    if d < abs(r1 - r2) - 1e-9:
        raise ValueError("Um círculo está contido no outro.")

    # --- Cálculo padrão ---
    a = (r1**2 - r2**2 + d**2) / (2.0 * d)
    h = math.sqrt(max(r1**2 - a**2, 0.0))   # max() protege de erro numérico

    mid = p1 + a * (p2 - p1) / d
    perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / d   # perpendicular unitário

    sol_a = Point2D.from_array(mid + h * perp)
    sol_b = Point2D.from_array(mid - h * perp)

    # Escolha da solução
    if prefer_positive_v:
        return sol_a if sol_a.v >= sol_b.v else sol_b
    else:
        return sol_a if sol_a.v < sol_b.v else sol_b


def line_intersection_2d(
    p1: Point2D, p2: Point2D,
    p3: Point2D, p4: Point2D,
) -> Point2D:
    """
    Interseção entre duas retas no plano 2D, cada uma definida por dois pontos.

    Usa a fórmula determinantal da interseção de retas.
    Levanta ValueError se as retas forem paralelas ou coincidentes.
    """
    x1, y1 = p1.u, p1.v
    x2, y2 = p2.u, p2.v
    x3, y3 = p3.u, p3.v
    x4, y4 = p4.u, p4.v

    # Denominador comum (cross product das direções)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        raise ValueError("As retas são paralelas ou coincidentes.")

    # Parâmetro t na reta 1
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    # Ponto de interseção = P1 + t · (P2 - P1)
    return Point2D(
        u=x1 + t * (x2 - x1),
        v=y1 + t * (y2 - y1),
    )
