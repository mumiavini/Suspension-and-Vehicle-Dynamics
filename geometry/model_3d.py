"""
model_3d.py
===========
Modelo orientado a objetos para a suspensão completa em 3D.

Arquitetura das classes
-----------------------
Point3D            ← primitives.py
Vector3D           ← primitives.py
ControlArm         ← representa um braço A ou L (dois pontos inboard + um outboard)
SteeringGeometry   ← pino mestre e geometria de direção
SuspensionCorner   ← uma ponta completa (UCA, LCA, steering, pushrod/pullrod)
Vehicle            ← carro completo (4 pontos = 4 SuspensionCorner)

Todos os ângulos são retornados em GRAUS.
Convenção SAE: X→frente, Y→esquerda, Z→cima.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from geometry.primitives import Point3D, Vector3D


# ---------------------------------------------------------------------------
# Braço de controle (Control Arm)
# ---------------------------------------------------------------------------

@dataclass
class ControlArm:
    """
    Braço de controle (A-arm ou L-arm) com dois pontos de ancoragem no chassi
    e um ponto de ancoragem na manga de eixo (outboard).

    Para braços simples (barra de rastreamento, etc.) use inboard_front == inboard_rear.

    Parâmetros
    ----------
    inboard_front : ponto de ancoragem inboard frontal no chassi (mm)
    inboard_rear  : ponto de ancoragem inboard traseiro no chassi (mm)
    outboard      : ponto de ancoragem na manga de eixo (mm)
    name          : identificador (ex: "UCA_FL", "LCA_FR")
    """

    inboard_front: Point3D
    inboard_rear:  Point3D
    outboard:      Point3D
    name:          str = "ControlArm"

    # ------------------------------------------------------------------
    # Vetores e comprimentos
    # ------------------------------------------------------------------

    @property
    def effective_inboard(self) -> Point3D:
        """
        Ponto inboard efetivo: midpoint dos dois pontos inboard.
        Usado para cálculos de vista frontal e lateral.
        """
        return self.inboard_front.midpoint(self.inboard_rear)

    def arm_vector(self) -> Vector3D:
        """Vetor do inboard efetivo ao outboard."""
        return Vector3D.from_points(self.effective_inboard, self.outboard)

    def arm_length(self) -> float:
        """Comprimento efetivo do braço (mm)."""
        return self.arm_vector().magnitude()

    def axis_vector(self) -> Vector3D:
        """
        Vetor do eixo de rotação do braço
        (de inboard_front para inboard_rear).
        """
        return Vector3D.from_points(self.inboard_front, self.inboard_rear)

    def plane_normal(self) -> Vector3D:
        """
        Normal ao plano do braço A (vetor perpendicular ao plano
        definido pelos três pontos de ancoragem).
        """
        v1 = self.axis_vector()
        v2 = self.arm_vector()
        normal = v1.cross(v2)
        return normal.normalize()

    def __repr__(self) -> str:
        return (
            f"ControlArm('{self.name}', "
            f"inboard_eff={self.effective_inboard}, "
            f"outboard={self.outboard}, "
            f"length={self.arm_length():.1f} mm)"
        )


# ---------------------------------------------------------------------------
# Pino mestre e geometria de direção
# ---------------------------------------------------------------------------

@dataclass
class KingpinGeometry:
    """
    Define o pino mestre (eixo de esterçamento) a partir dos pontos
    de ancoragem da manga de eixo.

    upper_ball_joint : centro da junta esférica superior (ponto outboard do UCA)
    lower_ball_joint : centro da junta esférica inferior (ponto outboard do LCA)
    wheel_center     : centro de roda (posição estática)
    contact_patch    : ponto de contato do pneu com o solo
    """

    upper_ball_joint: Point3D
    lower_ball_joint: Point3D
    wheel_center:     Point3D
    contact_patch:    Point3D

    # ------------------------------------------------------------------
    # Eixo do pino mestre
    # ------------------------------------------------------------------

    def kingpin_axis(self) -> Vector3D:
        """
        Vetor unitário ao longo do eixo do pino mestre
        (de lower_ball_joint → upper_ball_joint).
        """
        return Vector3D.from_points(
            self.lower_ball_joint, self.upper_ball_joint
        ).normalize()

    # ------------------------------------------------------------------
    # Kingpin Inclination (KPI)
    # ------------------------------------------------------------------

    def kingpin_inclination_deg(self) -> float:
        """
        Inclinação do pino mestre (KPI) em graus.

        Definição SAE: ângulo entre o eixo do pino mestre projetado no
        plano frontal (Y-Z) e o eixo vertical Z.

        O eixo do pino mestre une o lower_ball_joint ao upper_ball_joint.
        KPI positivo: topo do eixo inclinado para dentro (em direção ao plano
        de simetria do veículo). Para o lado esquerdo (Y > 0), isso significa
        que upper_ball_joint.y < lower_ball_joint.y.
        """
        kp = self.kingpin_axis()  # vetor unitário LBJ → UBJ

        # Projeção no plano YZ (zeramos X)
        yz = np.array([0.0, kp.y, kp.z])
        norm = np.linalg.norm(yz)
        if norm < 1e-12:
            return 0.0
        yz_norm = yz / norm

        # Ângulo com Z
        cos_a = np.clip(float(yz_norm[2]), -1.0, 1.0)
        angle = math.degrees(math.acos(cos_a))

        # Sinal: KPI positivo quando o UBJ está mais para dentro que o LBJ.
        # "Mais para dentro" = |UBJ.y| < |LBJ.y| (mais perto do plano de simetria).
        ubj_y_abs = abs(self.upper_ball_joint.y)
        lbj_y_abs = abs(self.lower_ball_joint.y)
        if ubj_y_abs < lbj_y_abs:
            return angle    # inclinado para dentro → KPI positivo
        else:
            return -angle

    # ------------------------------------------------------------------
    # Caster
    # ------------------------------------------------------------------

    def caster_deg(self) -> float:
        """
        Ângulo de caster em graus.

        Definição SAE: ângulo entre o eixo do pino mestre projetado no
        plano lateral (X-Z) e o eixo vertical Z.

        Caster positivo: a parte superior do pino mestre está deslocada para
        trás (−X) em relação à parte inferior. Convenção de carro de corrida.

        Para o lado esquerdo: UBJ.x < LBJ.x → caster positivo.
        """
        kp = self.kingpin_axis()  # vetor unitário LBJ → UBJ

        # Projeção no plano XZ (zeramos Y)
        xz = np.array([kp.x, 0.0, kp.z])
        norm = np.linalg.norm(xz)
        if norm < 1e-12:
            return 0.0
        xz_norm = xz / norm

        cos_a = np.clip(float(xz_norm[2]), -1.0, 1.0)
        angle = math.degrees(math.acos(cos_a))

        # Sinal: caster positivo quando UBJ está ATRÁS do LBJ (UBJ.x < LBJ.x)
        # kp.x = UBJ.x − LBJ.x; se negativo → UBJ está atrás → caster positivo
        if kp.x < 0:
            return angle
        else:
            return -angle

    # ------------------------------------------------------------------
    # Scrub Radius
    # ------------------------------------------------------------------

    def scrub_radius_mm(self) -> float:
        """
        Raio de scrub (raio de pivô) em mm.

        Definição: distância lateral (Y) entre:
          - O ponto onde o eixo do pino mestre intercepta o solo (Z=0)
          - O centro do contato do pneu com o solo

        Positivo → eixo do pino cruza o solo para DENTRO do contato
                   (configuração típica — ajuda na estabilidade).
        Negativo → eixo cruza para fora (scrub negativo).
        """
        kp_axis = self.kingpin_axis()

        # Ponto na linha do pino mestre em Z=0
        # Partindo do lower_ball_joint: P = LBJ + t * kp_axis, com P.z = 0
        lbj = self.lower_ball_joint.to_array()
        kp  = kp_axis.to_array()

        if abs(kp[2]) < 1e-12:
            # Eixo horizontal — não intercepta Z=0 de forma útil
            return float("inf")

        t = -lbj[2] / kp[2]
        ground_intercept = lbj + t * kp

        # Raio de scrub = distância Y entre intercepto e contato
        scrub = self.contact_patch.y - ground_intercept[1]
        return float(scrub)

    # ------------------------------------------------------------------
    # Mechanical Trail (Caster Trail)
    # ------------------------------------------------------------------

    def mechanical_trail_mm(self) -> float:
        """
        Trail mecânico (caster trail) em mm.

        Distância longitudinal (X) entre:
          - O ponto onde o eixo do pino intercepta o solo
          - O centro do contato do pneu

        Positivo → intercepto está à frente do contato (trail convencional).
        """
        kp_axis = self.kingpin_axis()
        lbj = self.lower_ball_joint.to_array()
        kp  = kp_axis.to_array()

        if abs(kp[2]) < 1e-12:
            return float("inf")

        t = -lbj[2] / kp[2]
        ground_intercept = lbj + t * kp

        trail = self.contact_patch.x - ground_intercept[0]
        return float(trail)

    def __repr__(self) -> str:
        return (
            f"KingpinGeometry(\n"
            f"  KPI     = {self.kingpin_inclination_deg():.3f}°\n"
            f"  Caster  = {self.caster_deg():.3f}°\n"
            f"  Scrub   = {self.scrub_radius_mm():.2f} mm\n"
            f"  Trail   = {self.mechanical_trail_mm():.2f} mm\n"
            f")"
        )


# ---------------------------------------------------------------------------
# SuspensionCorner — ponta completa de suspensão
# ---------------------------------------------------------------------------

@dataclass
class SuspensionCorner:
    """
    Representa uma ponta completa de suspensão (UMA roda).

    Campos obrigatórios
    -------------------
    upper_arm    : braço de controle superior (UCA)
    lower_arm    : braço de controle inferior (LCA)
    wheel_center : centro de roda (estático)
    contact_patch: ponto de contato com o solo
    corner_id    : identificador ("FL", "FR", "RL", "RR")

    Campos opcionais
    ----------------
    toe_link     : barra de toe (rear suspension)
    pushrod      : pushrod (se actuation inboard)
    pullrod      : pullrod (se actuation inboard)
    """

    upper_arm:    ControlArm
    lower_arm:    ControlArm
    wheel_center: Point3D
    contact_patch: Point3D
    corner_id:    str = "FL"

    toe_link:  Optional[ControlArm] = field(default=None)
    pushrod:   Optional[tuple[Point3D, Point3D]] = field(default=None)
    pullrod:   Optional[tuple[Point3D, Point3D]] = field(default=None)

    # ------------------------------------------------------------------
    # Derivados automáticos
    # ------------------------------------------------------------------

    @property
    def kingpin(self) -> KingpinGeometry:
        """Geometria do pino mestre derivada dos outboards do UCA e LCA."""
        return KingpinGeometry(
            upper_ball_joint=self.upper_arm.outboard,
            lower_ball_joint=self.lower_arm.outboard,
            wheel_center=self.wheel_center,
            contact_patch=self.contact_patch,
        )

    # ------------------------------------------------------------------
    # Parâmetros estáticos 3D
    # ------------------------------------------------------------------

    def static_caster_deg(self) -> float:
        """Caster estático (graus)."""
        return self.kingpin.caster_deg()

    def static_kpi_deg(self) -> float:
        """Kingpin Inclination estático (graus)."""
        return self.kingpin.kingpin_inclination_deg()

    def static_scrub_radius_mm(self) -> float:
        """Raio de scrub estático (mm)."""
        return self.kingpin.scrub_radius_mm()

    def static_mechanical_trail_mm(self) -> float:
        """Trail mecânico estático (mm)."""
        return self.kingpin.mechanical_trail_mm()

    def static_camber_deg(self) -> float:
        """
        Cambagem estática em graus, calculada a partir dos outboards 3D.

        O eixo da roda é perpendicular ao eixo do pino mestre e ao eixo X.
        Aqui simplificamos: usamos a inclinação do pino no plano YZ
        como proxy da cambagem (válido para suspensões tipo double wishbone).
        """
        ubj = self.upper_arm.outboard.to_array()
        lbj = self.lower_arm.outboard.to_array()

        upright_vec = ubj - lbj
        # Normalizar no plano YZ
        upright_yz = np.array([0.0, upright_vec[1], upright_vec[2]])
        norm = np.linalg.norm(upright_yz)
        if norm < 1e-12:
            return 0.0
        upright_yz /= norm

        cos_a = np.clip(upright_yz[2], -1.0, 1.0)
        angle = math.degrees(math.acos(cos_a))

        # Sinal: positivo quando topo da manga está para fora (+Y, lado esquerdo)
        if upright_yz[1] > 0:
            angle = -angle

        return angle

    # ------------------------------------------------------------------
    # Eixo de rolagem instantâneo (Roll Axis contribution)
    # ------------------------------------------------------------------

    def roll_center_3d_height(self) -> float:
        """
        Altura aproximada do centro de rolagem 3D (mm) usando projeção 2D (YZ).
        Retorna NaN se o cálculo falhar.
        """
        from geometry.solver_2d import SuspensionGeometry2D, Point2D

        uca_in_2d  = self.upper_arm.effective_inboard.project_yz()
        uca_out_2d = self.upper_arm.outboard.project_yz()
        lca_in_2d  = self.lower_arm.effective_inboard.project_yz()
        lca_out_2d = self.lower_arm.outboard.project_yz()
        wc_2d      = self.wheel_center.project_yz()
        cp_2d      = self.contact_patch.project_yz()

        geom_2d = SuspensionGeometry2D(
            uca_inboard=uca_in_2d,
            uca_outboard=uca_out_2d,
            lca_inboard=lca_in_2d,
            lca_outboard=lca_out_2d,
            wheel_center=wc_2d,
            contact_patch=cp_2d,
        )

        state = geom_2d.solve_heave(0.0)
        height = state.roll_center_height
        return height if height is not None else float("nan")

    def summary(self) -> str:
        """Resumo dos parâmetros estáticos da ponta de suspensão."""
        lines = [
            f"═══ SuspensionCorner [{self.corner_id}] ═══",
            f"  Caster              : {self.static_caster_deg():.3f}°",
            f"  KPI                 : {self.static_kpi_deg():.3f}°",
            f"  Camber (estático)   : {self.static_camber_deg():.3f}°",
            f"  Scrub Radius        : {self.static_scrub_radius_mm():.2f} mm",
            f"  Mechanical Trail    : {self.static_mechanical_trail_mm():.2f} mm",
            f"  Roll Center Height  : {self.roll_center_3d_height():.2f} mm",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Veículo completo
# ---------------------------------------------------------------------------

@dataclass
class Vehicle:
    """
    Modelo completo do veículo com as quatro pontas de suspensão.

    Parâmetros
    ----------
    front_left   : ponta dianteira esquerda
    front_right  : ponta dianteira direita
    rear_left    : ponta traseira esquerda
    rear_right   : ponta traseira direita
    wheelbase_mm : entre-eixos (mm)
    track_front_mm : bitola dianteira (mm)
    track_rear_mm  : bitola traseira (mm)
    """

    front_left:  SuspensionCorner
    front_right: SuspensionCorner
    rear_left:   SuspensionCorner
    rear_right:  SuspensionCorner

    wheelbase_mm:   float = 1600.0
    track_front_mm: float = 1200.0
    track_rear_mm:  float = 1150.0

    def roll_axis(self) -> tuple[float, float]:
        """
        Calcula o eixo de rolagem como a linha que une os centros de rolagem
        dianteiro e traseiro.

        Retorna
        -------
        (rc_front_height_mm, rc_rear_height_mm) — alturas dos centros de rolagem
        """
        rc_front = (
            self.front_left.roll_center_3d_height()
            + self.front_right.roll_center_3d_height()
        ) / 2.0

        rc_rear = (
            self.rear_left.roll_center_3d_height()
            + self.rear_right.roll_center_3d_height()
        ) / 2.0

        return rc_front, rc_rear

    def roll_axis_inclination_deg(self) -> float:
        """
        Inclinação do eixo de rolagem em relação ao solo (graus).
        Positivo → parte traseira do eixo mais alta.
        """
        rc_front, rc_rear = self.roll_axis()
        dz = rc_rear - rc_front
        dx = self.wheelbase_mm
        return math.degrees(math.atan2(dz, dx))

    def summary(self) -> str:
        rc_f, rc_r = self.roll_axis()
        lines = [
            "╔══════════════════════════════════════════╗",
            "║          VEHICLE SUSPENSION SUMMARY      ║",
            "╚══════════════════════════════════════════╝",
            "",
            self.front_left.summary(),
            "",
            self.front_right.summary(),
            "",
            self.rear_left.summary(),
            "",
            self.rear_right.summary(),
            "",
            "─── Eixo de Rolagem (Roll Axis) ───",
            f"  RC Dianteiro (médio) : {rc_f:.2f} mm",
            f"  RC Traseiro  (médio) : {rc_r:.2f} mm",
            f"  Inclinação Roll Axis : {self.roll_axis_inclination_deg():.4f}°",
        ]
        return "\n".join(lines)