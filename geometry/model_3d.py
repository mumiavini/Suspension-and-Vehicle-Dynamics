"""Last commit without claude interfering"""

"""
geometry/model_3d.py
====================
Modelo orientado a objetos da suspensão completa em 3D.

HIERARQUIA DE CLASSES:

    Point3D, Vector3D       (geometry/primitives.py)
            ↓
    ControlArm              braço de controle (A-arm com 2 inboards + 1 outboard)
            ↓
    KingpinGeometry         eixo do pino mestre + métodos de cálculo
            ↓
    SuspensionCorner        uma ponta (UCA + LCA + WC + CP)
            ↓
    Vehicle                 4 cantos (FL, FR, RL, RR) + dados gerais

OBJETIVO DESTE MÓDULO:
    Calcular parâmetros ESTÁTICOS 3D a partir das coordenadas dos hardpoints:
        - Caster
        - KPI (Kingpin Inclination)
        - Camber estático
        - Scrub Radius
        - Mechanical Trail
        - Roll Center / Roll Axis

NOTA: este módulo trabalha apenas com a POSIÇÃO ESTÁTICA. Para movimentos
(bump, roll, steer), use `geometry/solver_3d.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from geometry.primitives import Point3D, Vector3D


# =============================================================================
# ControlArm — Braço de controle (UCA ou LCA)
# =============================================================================

@dataclass
class ControlArm:
    """
    Braço de controle em "A" ou "L".

    Possui DOIS pontos de ancoragem no chassi (inboard_front, inboard_rear)
    e UM ponto na manga de eixo (outboard).

    Esses dois pontos inboard definem o EIXO DE ROTAÇÃO do braço — o braço
    pivota em torno dessa linha quando a suspensão se move.

    Atributos:
        inboard_front : ancoragem inboard frontal (chassi)
        inboard_rear  : ancoragem inboard traseira (chassi)
        outboard      : ancoragem na manga de eixo
        name          : identificador (ex: "UCA_FL")
    """
    inboard_front: Point3D
    inboard_rear:  Point3D
    outboard:      Point3D
    name:          str = "ControlArm"

    # -------------------------------------------------------------------------
    # Propriedades derivadas
    # -------------------------------------------------------------------------

    @property
    def effective_inboard(self) -> Point3D:
        """
        Ponto inboard EFETIVO: midpoint entre inboard_front e inboard_rear.

        Para cálculos 2D na vista frontal, projetamos o braço A num único
        elo equivalente que vai do midpoint até o outboard.
        """
        return self.inboard_front.midpoint(self.inboard_rear)

    def arm_vector(self) -> Vector3D:
        """Vetor do inboard efetivo até o outboard."""
        return Vector3D.from_points(self.effective_inboard, self.outboard)

    def arm_length(self) -> float:
        """Comprimento efetivo do braço (mm)."""
        return self.arm_vector().magnitude()

    def pivot_axis(self) -> Vector3D:
        """
        Eixo em torno do qual o braço pivota (linha inboard_front → inboard_rear).
        Não-unitário.
        """
        return Vector3D.from_points(self.inboard_front, self.inboard_rear)

    def __repr__(self) -> str:
        return (
            f"ControlArm('{self.name}', "
            f"length={self.arm_length():.1f} mm)"
        )


# =============================================================================
# KingpinGeometry — Eixo do pino mestre e ângulos derivados
# =============================================================================

@dataclass
class KingpinGeometry:
    """
    Eixo do pino mestre (steering axis) e métricas associadas.

    O PINO MESTRE é a linha imaginária que une os dois ball joints (LBJ e UBJ).
    A roda esterça em torno dessa linha. Suas inclinações em relação aos eixos
    verticais geram quatro parâmetros fundamentais:

        - Caster    : inclinação no plano lateral X-Z (afeta auto-centragem)
        - KPI       : inclinação no plano frontal Y-Z (afeta efeitos de steer)
        - Scrub     : distância lateral entre eixo e contato (afeta esforço)
        - Trail     : distância longitudinal entre eixo e contato (afeta retorno)

    Atributos:
        upper_ball_joint : centro da junta esférica superior (outboard do UCA)
        lower_ball_joint : centro da junta esférica inferior (outboard do LCA)
        wheel_center     : centro de roda (estático)
        contact_patch    : contato pneu-solo
    """
    upper_ball_joint: Point3D
    lower_ball_joint: Point3D
    wheel_center:     Point3D
    contact_patch:    Point3D

    # -------------------------------------------------------------------------
    # Eixo do pino mestre (vetor unitário)
    # -------------------------------------------------------------------------

    def kingpin_axis(self) -> Vector3D:
        """
        Vetor UNITÁRIO ao longo do pino mestre, apontando de baixo para cima
        (LBJ → UBJ).
        """
        return Vector3D.from_points(
            self.lower_ball_joint, self.upper_ball_joint
        ).normalize()

    # -------------------------------------------------------------------------
    # KPI (Kingpin Inclination)
    # -------------------------------------------------------------------------

    def kingpin_inclination_deg(self) -> float:
        """
        Inclinação do pino mestre no plano frontal Y-Z, em graus.

        CONVENÇÃO:
            KPI POSITIVO quando o topo do pino mestre está mais PARA DENTRO
            do veículo (mais perto do plano de simetria) do que a base.

        TYPICAL FSAE: 5° a 10°
        """
        kp = self.kingpin_axis()

        # Projeção no plano Y-Z (descarta componente X)
        yz = np.array([0.0, kp.y, kp.z])
        norm = float(np.linalg.norm(yz))
        if norm < 1e-12:
            return 0.0
        yz_unit = yz / norm

        # Ângulo com o eixo vertical Z (intervalo 0°..180°)
        cos_theta = float(np.clip(yz_unit[2], -1.0, 1.0))
        angle = math.degrees(math.acos(cos_theta))

        # Sinal: positivo se UBJ está mais perto do plano de simetria que LBJ
        # "Mais perto do plano de simetria" = |Y| menor
        ubj_inner = abs(self.upper_ball_joint.y) < abs(self.lower_ball_joint.y)
        return angle if ubj_inner else -angle

    # -------------------------------------------------------------------------
    # Caster
    # -------------------------------------------------------------------------

    def caster_deg(self) -> float:
        """
        Inclinação do pino mestre no plano lateral X-Z, em graus.

        CONVENÇÃO:
            Caster POSITIVO quando o topo do pino está deslocado PARA TRÁS
            em relação à base (configuração que gera auto-centragem do volante).

        TYPICAL FSAE: 3° a 7°
        """
        kp = self.kingpin_axis()

        # Projeção no plano X-Z (descarta componente Y)
        xz = np.array([kp.x, 0.0, kp.z])
        norm = float(np.linalg.norm(xz))
        if norm < 1e-12:
            return 0.0
        xz_unit = xz / norm

        cos_theta = float(np.clip(xz_unit[2], -1.0, 1.0))
        angle = math.degrees(math.acos(cos_theta))

        # Sinal: positivo se UBJ está ATRÁS de LBJ (UBJ.x < LBJ.x, ou seja, kp.x < 0)
        return angle if kp.x < 0 else -angle

    # -------------------------------------------------------------------------
    # Scrub Radius (raio de scrub)
    # -------------------------------------------------------------------------

    def scrub_radius_mm(self) -> float:
        """
        Distância LATERAL (Y) entre o ponto onde o pino mestre intercepta o
        solo e o centro de contato do pneu.

        CONVENÇÃO:
            POSITIVO: pino cruza o solo PARA DENTRO do contato (típico)
            NEGATIVO: pino cruza o solo PARA FORA do contato

        TYPICAL FSAE: −10 a +30 mm
        """
        intercept = self._kingpin_ground_intercept()
        if intercept is None:
            return float("inf")  # pino horizontal: não intercepta o solo
        return float(self.contact_patch.y - intercept[1])

    # -------------------------------------------------------------------------
    # Mechanical Trail (trail mecânico)
    # -------------------------------------------------------------------------

    def mechanical_trail_mm(self) -> float:
        """
        Distância LONGITUDINAL (X) entre o ponto onde o pino mestre intercepta
        o solo e o centro de contato do pneu.

        CONVENÇÃO:
            POSITIVO: intercepto à frente do contato (trail convencional)

        TYPICAL FSAE: 5 a 25 mm (depende muito do caster)
        """
        intercept = self._kingpin_ground_intercept()
        if intercept is None:
            return float("inf")
        return float(self.contact_patch.x - intercept[0])

    # -------------------------------------------------------------------------
    # Helper privado: onde o pino mestre cruza o solo (Z=0)?
    # -------------------------------------------------------------------------

    def _kingpin_ground_intercept(self) -> Optional[NDArray[np.float64]]:
        """
        Encontra o ponto em que a linha do pino mestre cruza o plano Z=0.

        Parametrização: P(t) = LBJ + t · kp_unit
        Queremos t tal que P.z = 0:
            t = -LBJ.z / kp_unit.z

        Retorna None se o pino for horizontal (kp.z ≈ 0).
        """
        kp = self.kingpin_axis().to_array()
        lbj = self.lower_ball_joint.to_array()

        if abs(kp[2]) < 1e-12:
            return None

        t = -lbj[2] / kp[2]
        return lbj + t * kp

    # -------------------------------------------------------------------------
    # Kingpin Offset @ Wheel Center (distância perpendicular do WC ao eixo)
    # -------------------------------------------------------------------------

    def kingpin_offset_at_wheel_center_mm(self) -> float:
        """
        Offset do eixo do pino mestre no nível do CENTRO DE RODA.

        DEFINIÇÃO: distância LATERAL (Y) entre o ponto onde o pino mestre
        passa na altura do wheel center, e o próprio wheel center.

        DIFERENÇA para Scrub Radius:
            - Scrub Radius é a distância no NÍVEL DO SOLO
            - Kingpin Offset (este) é no NÍVEL DO CENTRO DE RODA
            Estes dois valores se relacionam pelo KPI:
                offset_wc - scrub_radius = WC.z · tan(KPI)

        TÍPICO FSAE: 30-80 mm (positivo)
        """
        kp_unit = self.kingpin_axis().to_array()
        lbj = self.lower_ball_joint.to_array()

        if abs(kp_unit[2]) < 1e-12:
            return float("inf")

        # Parametriza a linha do pino e encontra o ponto na altura do WC
        t_wc = (self.wheel_center.z - lbj[2]) / kp_unit[2]
        point_on_axis = lbj + t_wc * kp_unit

        # Distância lateral (Y) em relação ao WC
        return float(self.wheel_center.y - point_on_axis[1])


# =============================================================================
# SuspensionCorner — Uma ponta completa da suspensão
# =============================================================================

@dataclass
class SuspensionCorner:
    """
    Uma ponta de suspensão (uma roda): UCA + LCA + manga + roda.

    Atributos obrigatórios:
        upper_arm     : braço superior (UCA)
        lower_arm     : braço inferior (LCA)
        wheel_center  : centro da roda (estático)
        contact_patch : contato pneu-solo (estático)
        corner_id     : "FL" | "FR" | "RL" | "RR"

    Atributos opcionais (não usados nesta fase do projeto):
        toe_link / pushrod / pullrod
    """
    upper_arm:     ControlArm
    lower_arm:     ControlArm
    wheel_center:  Point3D
    contact_patch: Point3D
    corner_id:     str = "FL"

    toe_link: Optional[ControlArm] = field(default=None)
    pushrod:  Optional[tuple[Point3D, Point3D]] = field(default=None)
    pullrod:  Optional[tuple[Point3D, Point3D]] = field(default=None)

    # -------------------------------------------------------------------------
    # Geometria do pino mestre (computada sob demanda)
    # -------------------------------------------------------------------------

    @property
    def kingpin(self) -> KingpinGeometry:
        """
        Eixo do pino mestre desta ponta.
        Construído a partir dos outboards do UCA (UBJ) e LCA (LBJ).
        """
        return KingpinGeometry(
            upper_ball_joint=self.upper_arm.outboard,
            lower_ball_joint=self.lower_arm.outboard,
            wheel_center=self.wheel_center,
            contact_patch=self.contact_patch,
        )

    # -------------------------------------------------------------------------
    # Parâmetros estáticos (delegam para KingpinGeometry)
    # -------------------------------------------------------------------------

    def static_caster_deg(self)           -> float: return self.kingpin.caster_deg()
    def static_kpi_deg(self)              -> float: return self.kingpin.kingpin_inclination_deg()
    def static_scrub_radius_mm(self)      -> float: return self.kingpin.scrub_radius_mm()
    def static_mechanical_trail_mm(self)  -> float: return self.kingpin.mechanical_trail_mm()
    def static_kingpin_offset_mm(self)    -> float: return self.kingpin.kingpin_offset_at_wheel_center_mm()

    # -------------------------------------------------------------------------
    # Steer Arm Length — comprimento do braço de direção efetivo
    # -------------------------------------------------------------------------

    def steer_arm_length_mm(self, tie_rod_outboard: Point3D) -> float:
        """
        Comprimento do braço de direção (steer arm).

        DEFINIÇÃO: distância PERPENDICULAR do ponto outboard do tie-rod
        ao eixo do pino mestre. Esse é o "braço de alavanca" através do
        qual a força do tie-rod gera torque de esterçamento na roda.

        FÓRMULA:
            steer_arm = |(TRO - LBJ) × kp_unit|

        TÍPICO FSAE: 50-100 mm
        """
        kp_unit = self.kingpin.kingpin_axis().to_array()
        v = tie_rod_outboard.to_array() - self.lower_arm.outboard.to_array()
        return float(np.linalg.norm(np.cross(v, kp_unit)))

    # -------------------------------------------------------------------------
    # Anti-dive / Anti-lift (vista lateral X-Z)
    # -------------------------------------------------------------------------

    def anti_dive_percent(
        self,
        brake_bias: float = 0.6,
        wheelbase_mm: float = 1550.0,
        cg_height_mm: float = 280.0,
    ) -> float:
        """
        Anti-dive (%) — fração da força de frenagem absorvida pela geometria.

        FÓRMULA (Milliken & Milliken, "Race Car Vehicle Dynamics" eq. 17.21):

            anti_dive_% = brake_bias × tan(θ) × (wheelbase / h_CG) × 100%

        onde:
            θ      = ângulo entre a horizontal e a linha do CP ao IC_lateral
            IC_lat = interseção das prolongações dos braços no plano X-Z
            h_CG   = altura do centro de gravidade do veículo (mm)

        Por que precisa de wheelbase e h_CG:
            O termo wheelbase/h_CG vem do equilíbrio de torques na transferência
            longitudinal de carga. Sem esses parâmetros, a fórmula geométrica
            isolada não é interpretável como "%".

        Parâmetros default são típicos FSAE (wheelbase 1550, CG ~280mm).
        AJUSTE conforme seu veículo para um valor preciso.

        TÍPICO FSAE: 0% a 30% (anti-dive); negativo = pro-dive
        """
        from geometry.primitives import line_intersection_2d

        uca_in_2d  = self.upper_arm.effective_inboard.project_xz()
        uca_out_2d = self.upper_arm.outboard.project_xz()
        lca_in_2d  = self.lower_arm.effective_inboard.project_xz()
        lca_out_2d = self.lower_arm.outboard.project_xz()

        try:
            ic_lat = line_intersection_2d(
                uca_in_2d, uca_out_2d, lca_in_2d, lca_out_2d,
            )
        except ValueError:
            return 0.0   # braços paralelos: IC no infinito → 0%

        cp_2d = self.contact_patch.project_xz()
        dx = ic_lat.u - cp_2d.u
        dz = ic_lat.v - cp_2d.v

        if abs(dx) < 1e-6 or cg_height_mm < 1e-6:
            return 0.0

        # tan(θ) = dz / |dx|, com sinal vindo de dz
        tan_theta = dz / abs(dx)

        # Anti-dive % considerando a transferência longitudinal
        anti_dive = brake_bias * tan_theta * (wheelbase_mm / cg_height_mm) * 100.0

        # Saturação física razoável
        return float(max(-200.0, min(200.0, anti_dive)))

    def anti_squat_percent(self, drive_fraction: float = 1.0) -> float:
        """
        Anti-squat (%) — equivalente ao anti-dive mas para aceleração
        (relevante apenas para a suspensão TRASEIRA num carro de tração traseira).

        Para o eixo dianteiro: anti_squat = 0 (não recebe torque de tração).
        Para o eixo traseiro (com diff fechado): drive_fraction = 1.0

        ALGORITMO: idêntico ao anti-dive, mas conta o IC do lado oposto
        ao da frenagem. Aqui simplificamos retornando a mesma geometria.
        """
        return self.anti_dive_percent(brake_bias=drive_fraction)

    # -------------------------------------------------------------------------
    # Camber estático 3D — calculado pela projeção da manga no plano Y-Z
    # -------------------------------------------------------------------------

    def static_camber_deg(self) -> float:
        """
        Cambagem estática construtiva em graus.

        IMPORTANTE: o camber estático NÃO pode ser inferido apenas dos
        hardpoints. Ele depende de como a MANGA foi fabricada (posição
        relativa dos furos do mancal de roda em relação aos furos das
        ball joints).

        Para a maioria das mangas FSAE, a inclinação construtiva é planejada
        para dar o camber estático desejado (ex: -1.5°). Esse valor é o
        OFFSET que precisa ser adicionado ao camber dinâmico calculado
        pelo solver.

        Este método retorna o offset construtivo gravado em
        `self.static_camber_offset_deg`. Se não definido, retorna 0.

        CONVENÇÃO SAE:
            − = topo da roda inclinado PARA DENTRO do veículo
            + = topo da roda inclinado PARA FORA
        """
        return getattr(self, "static_camber_offset_deg", 0.0)

    # -------------------------------------------------------------------------
    # Roll Center estático 3D — usa o solver 2D na projeção Y-Z
    # -------------------------------------------------------------------------

    def roll_center_height_mm(self) -> float:
        """
        Altura do Roll Center desta ponta (mm), calculada na vista frontal.

        Projeta todos os pontos no plano Y-Z e usa o solver 2D para resolver
        o estado estático (heave = 0). Retorna a coordenada Z do RC.
        Retorna NaN se indeterminado.
        """
        # Import local para evitar ciclo de imports
        from geometry.solver_2d import SuspensionGeometry2D

        geom_2d = SuspensionGeometry2D(
            uca_inboard  = self.upper_arm.effective_inboard.project_yz(),
            uca_outboard = self.upper_arm.outboard.project_yz(),
            lca_inboard  = self.lower_arm.effective_inboard.project_yz(),
            lca_outboard = self.lower_arm.outboard.project_yz(),
            wheel_center = self.wheel_center.project_yz(),
            contact_patch= self.contact_patch.project_yz(),
        )
        state = geom_2d.solve_heave(0.0)
        h = state.roll_center_height
        return h if h is not None else float("nan")

    # -------------------------------------------------------------------------
    # Resumo formatado
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """Retorna um resumo formatado dos parâmetros estáticos."""
        return "\n".join([
            f"═══ SuspensionCorner [{self.corner_id}] ═══",
            f"  Caster              : {self.static_caster_deg():+.3f}°",
            f"  KPI                 : {self.static_kpi_deg():+.3f}°",
            f"  Camber (estático)   : {self.static_camber_deg():+.3f}°",
            f"  Scrub Radius        : {self.static_scrub_radius_mm():+.2f} mm",
            f"  Mechanical Trail    : {self.static_mechanical_trail_mm():+.2f} mm",
            f"  Roll Center Height  : {self.roll_center_height_mm():+.2f} mm",
        ])


# =============================================================================
# Vehicle — Carro completo (4 cantos + dimensões gerais)
# =============================================================================

@dataclass
class Vehicle:
    """
    Veículo completo: quatro pontas de suspensão + parâmetros gerais.

    Atributos:
        front_left   : ponta FL
        front_right  : ponta FR
        rear_left    : ponta RL
        rear_right   : ponta RR
        wheelbase_mm : entre-eixos (mm)
        track_front_mm / track_rear_mm : bitolas (mm)
    """
    front_left:  SuspensionCorner
    front_right: SuspensionCorner
    rear_left:   SuspensionCorner
    rear_right:  SuspensionCorner

    wheelbase_mm:   float = 1600.0
    track_front_mm: float = 1200.0
    track_rear_mm:  float = 1180.0

    # -------------------------------------------------------------------------
    # Eixo de rolagem (Roll Axis)
    # -------------------------------------------------------------------------

    def roll_axis(self) -> tuple[float, float]:
        """
        Retorna (rc_front, rc_rear): alturas médias do RC dianteiro e traseiro.

        O EIXO DE ROLAGEM do veículo é a linha que une esses dois RCs.
        """
        rc_front = 0.5 * (self.front_left.roll_center_height_mm()
                        + self.front_right.roll_center_height_mm())
        rc_rear  = 0.5 * (self.rear_left.roll_center_height_mm()
                        + self.rear_right.roll_center_height_mm())
        return rc_front, rc_rear

    def roll_axis_inclination_deg(self) -> float:
        """
        Inclinação do eixo de rolagem em relação ao solo (graus).
        Positivo = parte traseira do eixo mais alta que a dianteira.
        """
        rc_front, rc_rear = self.roll_axis()
        return math.degrees(math.atan2(rc_rear - rc_front, self.wheelbase_mm))

    # -------------------------------------------------------------------------
    # Ackermann estático (geométrico)
    # -------------------------------------------------------------------------

    def static_ackermann_percent(
        self,
        tie_rod_fl_outboard: Point3D,
        tie_rod_fr_outboard: Point3D,
    ) -> float:
        """
        Ackermann estático (%) — quanto a geometria de direção SE APROXIMA
        do Ackermann perfeito (100%).

        ALGORITMO:
            O Ackermann perfeito (100%) ocorre quando as linhas que partem
            de cada king pin axis, passando pelo respectivo tie-rod outboard,
            convergem para um único ponto no eixo traseiro.

            Aproximação prática (Steer Arm method):
                tan(α_perfeito) = (track / 2) / wheelbase
                α_real = atan(steer_arm_offset / wheel_center_to_pin_dist)

            Razão (real / perfeito) × 100% = Ackermann %

        SIMPLIFICAÇÃO USADA:
            Mede o ângulo entre o steer arm (vetor da projeção do pino mestre
            ao TRO, no plano XY) e o eixo Y do veículo. Compara ao ângulo
            Ackermann perfeito do veículo.

        Retorna porcentagem (100% = perfeito, 0% = paralelo).
        """
        # Ângulo Ackermann perfeito do veículo (geometria simples)
        if self.wheelbase_mm < 1e-6 or self.track_front_mm < 1e-6:
            return 0.0
        perfect_angle_rad = math.atan2(self.track_front_mm / 2.0, self.wheelbase_mm)

        # Ângulo real do steer arm da ponta FL
        kp_fl = self.front_left.kingpin.kingpin_axis().to_array()
        wc_fl = self.front_left.wheel_center.to_array()
        tro_fl = tie_rod_fl_outboard.to_array()

        # Vetor TRO → projeção no plano XY (vista superior)
        steer_arm_vec_fl = tro_fl - wc_fl
        # Remove componente paralela ao kingpin axis
        steer_arm_perp_fl = steer_arm_vec_fl - np.dot(steer_arm_vec_fl, kp_fl) * kp_fl
        sa_xy_fl = np.array([steer_arm_perp_fl[0], steer_arm_perp_fl[1]])
        if float(np.linalg.norm(sa_xy_fl)) < 1e-9:
            return 0.0
        # O ângulo do steer arm com o eixo Y (lateral)
        real_angle_rad = abs(math.atan2(sa_xy_fl[0], abs(sa_xy_fl[1]) + 1e-12))

        # Ackermann % = razão entre o ângulo real e o ideal
        if perfect_angle_rad < 1e-9:
            return 0.0
        return float(100.0 * real_angle_rad / perfect_angle_rad)

    # -------------------------------------------------------------------------
    # Resumo formatado
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        rc_f, rc_r = self.roll_axis()
        return "\n".join([
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
            "─── Roll Axis ───",
            f"  RC Dianteiro (médio)  : {rc_f:+.2f} mm",
            f"  RC Traseiro  (médio)  : {rc_r:+.2f} mm",
            f"  Inclinação Roll Axis  : {self.roll_axis_inclination_deg():+.4f}°",
        ])
