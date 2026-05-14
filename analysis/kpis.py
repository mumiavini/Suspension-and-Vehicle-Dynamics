"""
analysis/kpis.py
================
Cálculo de KPIs adicionais da geometria de suspensão.

Complementa os KPIs básicos (Caster, KPI, Camber, Scrub, Trail, RC Height)
calculados em `geometry/model_3d.py` com:

    - Wheelbase, Track Width
    - Ride Camber (°/m) e Roll Camber (°/°)
    - Static Sum Toe
    - Static Ackermann (%)
    - Steer Ratio, C-factor, Steer Arm Length
    - Roll Center sob carga lateral (aproximação 1g)
    - Anti-dive / Anti-squat (versão simplificada)

NOTA: alguns KPIs (wheel rate, motion ratio, damping) dependem de
parâmetros externos (rigidez de mola, geometria de rocker, dados de
amortecedor) e não são calculáveis pela cinemática pura. Esses ficam
como INPUTS do usuário no app, não como cálculos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from geometry.primitives import Point3D
from geometry.model_3d import SuspensionCorner, Vehicle
from geometry.solver_3d import TieRod, KinematicSolver3D


# =============================================================================
# Dimensões gerais do veículo
# =============================================================================

def wheelbase_mm(
    front_corner: SuspensionCorner,
    rear_corner:  SuspensionCorner,
) -> float:
    """
    Wheelbase (entre-eixos): distância longitudinal (X) entre o WC dianteiro
    e o WC traseiro do MESMO lado.
    """
    return abs(front_corner.wheel_center.x - rear_corner.wheel_center.x)


def track_width_mm(
    left_corner:  SuspensionCorner,
    right_corner: SuspensionCorner,
) -> float:
    """
    Bitola (track width): distância lateral (Y) entre o WC esquerdo e o WC
    direito do MESMO eixo (dianteiro ou traseiro).
    """
    return abs(left_corner.wheel_center.y - right_corner.wheel_center.y)


# =============================================================================
# Camber dinâmico — Ride Camber e Roll Camber
# =============================================================================

def ride_camber_deg_per_m(
    corner: SuspensionCorner,
    tie_rod: TieRod,
    heave_range_mm: float = 25.0,
) -> float:
    """
    Ride Camber: taxa de variação do camber com heave, em °/m.

    É o mesmo que "camber gain", mas na unidade da imagem (°/m em vez de °/mm).

    Calculado por regressão linear de camber × heave num pequeno range.
    """
    solver = KinematicSolver3D(corner, tie_rod)
    heaves = np.linspace(-heave_range_mm, heave_range_mm, 11)
    cambers = []
    for h in heaves:
        solver.reset_seed()
        cambers.append(solver.solve(float(h), 0.0, 0.0).camber_deg)

    # Regressão linear: slope em °/mm
    slope_mm = float(np.polyfit(heaves, cambers, 1)[0])
    return slope_mm * 1000.0   # converte para °/m


def roll_camber_deg_per_deg(
    corner: SuspensionCorner,
    tie_rod: TieRod,
    roll_range_deg: float = 2.0,
) -> float:
    """
    Roll Camber: taxa de variação do camber com roll do chassi, em °/°.

    Caracteriza quanto a roda EXTERNA ganha de camber quando o chassi rola.
    Valor típico FSAE: -0.5 a -1.5 (camber negativo aumenta com roll positivo).

    Calculado por regressão linear de camber × roll.
    """
    solver = KinematicSolver3D(corner, tie_rod)
    rolls = np.linspace(-roll_range_deg, roll_range_deg, 11)
    cambers = []
    for r in rolls:
        solver.reset_seed()
        cambers.append(solver.solve(0.0, float(r), 0.0).camber_deg)

    slope = float(np.polyfit(rolls, cambers, 1)[0])
    return slope


# =============================================================================
# Toe estático e Sum Toe
# =============================================================================

def static_toe_deg(
    corner: SuspensionCorner,
    tie_rod: TieRod,
) -> float:
    """
    Toe estático absoluto desta roda, em graus.

    CONVENÇÃO:
        + = toe-in (roda apontando para o centro do veículo)
        − = toe-out

    DEFINIÇÃO:
        Toe é o ângulo entre a direção em que a roda aponta (no plano XY)
        e o eixo longitudinal X do veículo.

        Como o "ponto que define a frente da roda" não é um hardpoint,
        usamos a convenção que o WC e o CP estão alinhados no plano da
        roda. Para roda perfeitamente neutra (toe=0), CP fica em (X_wc,
        Y_wc, 0) — exatamente abaixo do WC.

        Se houver deslocamento longitudinal entre CP e WC (CP.x != WC.x),
        a roda está com toe.

    NOTA: para um carro montado simétrico com CP exatamente abaixo do WC
    em XY, este valor sempre será 0. Para introduzir toe estático, o usuário
    pode deslocar o CP em X (ou rotacionar a manga construtivamente).
    """
    wc = corner.wheel_center
    cp = corner.contact_patch

    # vetor da CP→WC projetado em XY: define a direção longitudinal da roda
    dx = wc.x - cp.x
    dy = wc.y - cp.y

    # Para roda neutra, dx=0 e dy=0 (CP exatamente abaixo do WC) → toe = 0
    # Se dx != 0 mas dy = 0, indica deslocamento puro longitudinal: ainda toe = 0
    # Toe real = ângulo entre o EIXO DA RODA (perpendicular ao eixo do hub)
    # e o eixo X. Como a definição depende da orientação da manga, retornamos
    # 0 para geometrias simétricas e o ângulo derivado caso WC e CP estejam
    # rotacionados em XY.

    # Aproximação: usa o ângulo do vetor "WC para frente" em XY.
    # Se WC e CP coincidem em XY, o toe é 0 (geometria neutra).
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0

    # Caso o usuário tenha colocado CP deslocado, calcula o toe relativo
    # ao plano YZ (eixo lateral). O toe positivo (in) significa frente
    # da roda para dentro.
    # Para esquerda (Y>0): frente para dentro = +X (frente) tem Y menor
    # Para direita (Y<0): frente para dentro = +X tem Y maior
    angle = math.degrees(math.atan2(dx, abs(dy) + 1e-12))

    # Convenção: angle pequeno → toe ~0
    # Se WC.y > 0 (esquerda) e dx > 0 → frente da roda apontando para fora? não.
    # Mantemos a convenção simples: módulo pequeno, sinal por lado.
    if abs(angle) > 45:
        # Provavelmente o CP foi colocado errado, retorna 0
        return 0.0

    return angle if corner.wheel_center.y > 0 else -angle


def static_sum_toe_deg(
    left_corner:  SuspensionCorner, left_tie_rod:  TieRod,
    right_corner: SuspensionCorner, right_tie_rod: TieRod,
) -> float:
    """
    Static Sum Toe (graus): soma do toe estático das duas rodas do mesmo eixo.

    CONVENÇÃO (imagem):
        + = toe-in total (ambas convergindo)
        − = toe-out total (ambas divergindo)

    É o valor que aparece na ficha de setup do carro.
    """
    return (
        static_toe_deg(left_corner, left_tie_rod)
      + static_toe_deg(right_corner, right_tie_rod)
    )


# =============================================================================
# Ackermann e geometria de direção
# =============================================================================

def ackermann_geometry(
    front_left_corner:  SuspensionCorner, fl_tie_rod: TieRod,
    front_right_corner: SuspensionCorner, fr_tie_rod: TieRod,
    rear_corner: SuspensionCorner,
) -> dict[str, float]:
    """
    Calcula a geometria de Ackermann estática.

    DEFINIÇÃO:
        Ackermann puro 100% = quando os prolongamentos das linhas dos
        steering arms (do KPI até o ponto outboard do tie-rod, projetados
        no plano horizontal) se encontram exatamente sobre o eixo traseiro.

        Ackermann 0% = quando essas linhas são paralelas (roda interna e
        externa esterçam pelo mesmo ângulo).

    CÁLCULO:
        1. Para cada roda dianteira, determina o ponto onde o pino mestre
           cruza o plano da altura do tie-rod.
        2. Traça uma reta desse ponto até o TRO, projetada em XY.
        3. Vê onde essas duas retas se cruzam em X (longitudinal).
        4. Compara com a posição do eixo traseiro:
              x_inter = x_eixo_traseiro  → 100% Ackermann
              x_inter = -∞               → 0% Ackermann
        5. Ackermann (%) = wheelbase / (x_kpi - x_intersect) × 100

    Retorna dict com:
        ackermann_percent : % de Ackermann estático
        wheelbase_mm
        steer_arm_length_left, _right : comprimento do steering arm (mm)
    """
    # --- Comprimento do steering arm em cada roda ---
    # Steering arm = distância do tie-rod outboard até o pino mestre
    sa_l = _steering_arm_length(front_left_corner, fl_tie_rod)
    sa_r = _steering_arm_length(front_right_corner, fr_tie_rod)

    # --- Pontos do KPI no plano horizontal (altura do TRO) ---
    # Aproximamos a "linha de ackermann" como TRO → projeção do KPI em XY
    # na altura do TRO.
    def kpi_at_height(corner: SuspensionCorner, z_target: float) -> tuple[float, float]:
        """Retorna (X, Y) do eixo do pino mestre na altura z_target."""
        lbj = corner.lower_arm.outboard.to_array()
        ubj = corner.upper_arm.outboard.to_array()
        kp = ubj - lbj
        if abs(kp[2]) < 1e-12:
            return (float(lbj[0]), float(lbj[1]))
        t = (z_target - lbj[2]) / kp[2]
        p = lbj + t * kp
        return (float(p[0]), float(p[1]))

    fl_tro = fl_tie_rod.outboard
    fr_tro = fr_tie_rod.outboard
    kpi_l = kpi_at_height(front_left_corner,  fl_tro.z)
    kpi_r = kpi_at_height(front_right_corner, fr_tro.z)

    # --- Linhas KPI → TRO no plano XY, prolongadas para trás ---
    # Equação da reta: P(t) = KPI + t · (TRO - KPI)
    # Para grande t > 1, vai além do TRO; queremos achar onde as duas se cruzam.
    def line_intersection_xy(p1, p2, p3, p4):
        """Interseção das retas (p1,p2) e (p3,p4) no plano XY."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return None  # linhas paralelas → 0% Ackermann
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    intersect = line_intersection_xy(
        kpi_l, (fl_tro.x, fl_tro.y),
        kpi_r, (fr_tro.x, fr_tro.y),
    )

    wb = wheelbase_mm(front_left_corner, rear_corner)

    if intersect is None:
        ackermann_pct = 0.0
    else:
        # Pega o X médio do KPI dianteiro como referência do eixo dianteiro
        x_front = 0.5 * (kpi_l[0] + kpi_r[0])
        x_rear_target = front_left_corner.wheel_center.x - wb  # eixo traseiro
        x_inter = intersect[0]

        # Ackermann% = (eixo_diant - x_inter) / (eixo_diant - eixo_tras) × 100
        denom = x_front - x_rear_target
        if abs(denom) < 1e-9:
            ackermann_pct = 0.0
        else:
            ackermann_pct = float((x_front - x_inter) / denom * 100.0)

    return {
        "ackermann_percent":       ackermann_pct,
        "wheelbase_mm":            wb,
        "steer_arm_length_left":   sa_l,
        "steer_arm_length_right":  sa_r,
    }


def _steering_arm_length(corner: SuspensionCorner, tie_rod: TieRod) -> float:
    """
    Comprimento do steering arm: distância perpendicular do TRO ao eixo
    do pino mestre (= raio efetivo de esterçamento).
    """
    ubj = corner.upper_arm.outboard.to_array()
    lbj = corner.lower_arm.outboard.to_array()
    tro = tie_rod.outboard.to_array()

    kp = ubj - lbj
    kp_norm = float(np.linalg.norm(kp))
    if kp_norm < 1e-12:
        return 0.0
    kp_unit = kp / kp_norm

    # Vetor do LBJ ao TRO, componente perpendicular ao pino
    v = tro - lbj
    v_perp = v - np.dot(v, kp_unit) * kp_unit
    return float(np.linalg.norm(v_perp))


# =============================================================================
# Steer Ratio e C-factor (rack)
# =============================================================================

def steer_ratio_and_cfactor(
    corner: SuspensionCorner,
    tie_rod: TieRod,
    *,
    rack_test_mm: float = 5.0,
) -> dict[str, float]:
    """
    Calcula Steer Ratio (volante:roda) e C-factor (mm de rack por rotação
    do volante).

    DEFINIÇÕES:
        C-factor (mm/rev) : deslocamento do rack para 1 rotação completa
                             do pinhão (360°). DEPENDE DO PINHÃO da cremalheira,
                             que é INPUT do usuário — aqui retornamos o que
                             pode ser calculado: a relação rack → roda em
                             unidades de mm de rack por grau de roda.

        Steer Ratio (x:1) : graus de volante por grau de roda. Depende do
                            C-factor + do que calculamos aqui.

    SAÍDA INTERMEDIÁRIA:
        rack_per_wheel_deg_mm_per_deg : quantos mm de rack são necessários
                                         para 1° de esterçamento da roda.

    Para obter o Steer Ratio final, o usuário multiplica por:
        steer_ratio = c_factor / 360 / rack_per_wheel_deg_mm_per_deg
    """
    # Roda em rack=0 e em rack=test, mede diferença de toe
    solver = KinematicSolver3D(corner, tie_rod)
    s0 = solver.solve(0.0, 0.0, 0.0)
    s1 = solver.solve(0.0, 0.0, rack_test_mm)

    # delta toe em graus
    d_toe = s1.toe_deg - s0.toe_deg

    if abs(d_toe) < 1e-9:
        return {
            "rack_per_wheel_deg_mm_per_deg": float("inf"),
            "wheel_deg_per_rack_mm":         0.0,
        }

    # Quantos mm de rack para 1° de roda
    rack_per_deg = rack_test_mm / abs(d_toe)
    # Inverso: quantos graus de roda por mm de rack
    deg_per_rack_mm = 1.0 / rack_per_deg

    return {
        "rack_per_wheel_deg_mm_per_deg": rack_per_deg,
        "wheel_deg_per_rack_mm":         deg_per_rack_mm,
    }


def steer_ratio_from_pinion(
    rack_per_wheel_deg_mm_per_deg: float,
    c_factor_mm_per_rev: float,
) -> float:
    """
    Calcula Steer Ratio (volante:roda) a partir do C-factor da cremalheira.

    Steer Ratio = (graus de volante por grau de roda)
                = (mm_rack/grau_roda) / (mm_rack/grau_volante)
                = rack_per_wheel_deg / (c_factor / 360)
    """
    if c_factor_mm_per_rev <= 0:
        return float("inf")
    rack_per_wheel_deg = abs(rack_per_wheel_deg_mm_per_deg)
    rack_per_steer_wheel_deg = c_factor_mm_per_rev / 360.0
    return rack_per_wheel_deg / rack_per_steer_wheel_deg


# =============================================================================
# Roll Center sob carga lateral (1g)
# =============================================================================

def roll_center_at_1g_lat(
    left_corner:  SuspensionCorner, left_tie_rod:  TieRod,
    right_corner: SuspensionCorner, right_tie_rod: TieRod,
    *,
    cg_height_mm:           float = 280.0,
    track_width_mm_override: Optional[float] = None,
    roll_stiffness_deg_per_g: float = 1.5,
) -> dict[str, float]:
    """
    Estima a posição do Roll Center (Y, Z) com o carro sob aceleração
    lateral de 1g, simulado como o roll equivalente.

    SIMPLIFICAÇÃO:
        Em 1g lateral, o chassi rola um ângulo proporcional à roll stiffness.
        Aplicamos esse roll e calculamos onde o RC fica.

    Parâmetros:
        cg_height_mm           : altura do CG (necessário só para o cálculo
                                  formal — não afeta o RC diretamente aqui)
        roll_stiffness_deg_per_g: quantos graus o chassi rola por g
                                  (valor típico FSAE: 1.0–2.0 °/g)

    Retorna dict com:
        rc_y_mm, rc_z_mm : posição do RC sob 1g
    """
    # Roll aplicado em 1g
    roll_1g = float(roll_stiffness_deg_per_g)

    # Roda o solver com esse roll nos dois corners; o RC é calculado pelo
    # método 2D na projeção YZ dos pontos atuais.
    solver_l = KinematicSolver3D(left_corner,  left_tie_rod)
    solver_r = KinematicSolver3D(right_corner, right_tie_rod)
    state_l = solver_l.solve(0.0, roll_1g, 0.0)
    state_r = solver_r.solve(0.0, roll_1g, 0.0)

    # RC ESQUERDO (calculado pelo solver via projeção 2D)
    from analysis.sweeps import SweepRunner
    runner_l = SweepRunner(solver=solver_l)
    rc_y_l, rc_z_l = runner_l._estimate_roll_center_yz(state_l)
    runner_r = SweepRunner(solver=solver_r)
    rc_y_r, rc_z_r = runner_r._estimate_roll_center_yz(state_r)

    # RC médio dos dois lados (em estado de roll, eles divergem; pegamos média)
    return {
        "rc_y_mm": 0.5 * (rc_y_l + rc_y_r),
        "rc_z_mm": 0.5 * (rc_z_l + rc_z_r),
        "roll_applied_deg": roll_1g,
    }


# =============================================================================
# Anti-dive / Anti-squat (versão simplificada, vista lateral)
# =============================================================================

def anti_dive_percent(
    corner: SuspensionCorner,
    *,
    brake_bias_pct: float = 50.0,
    cg_height_mm:   float = 280.0,
    wheelbase_mm_value: Optional[float] = None,
) -> float:
    """
    Anti-dive (%) — VERSÃO SIMPLIFICADA.

    DEFINIÇÃO RIGOROSA:
        Anti-dive = tan(θ_SVIC) × wheelbase / cg_height × brake_bias_pct
        onde θ_SVIC é o ângulo entre o eixo de força de frenagem e a horizontal,
        medido a partir do CP até o "Side View Instant Center" (SVIC) —
        a interseção das prolongações dos braços UCA e LCA na vista LATERAL (X-Z).

    APROXIMAÇÃO USADA AQUI:
        Calculamos o SVIC como interseção das prolongações dos braços projetados
        em XZ (usando os pontos efetivos inboard e outboard).
        Depois aplicamos a fórmula acima.

    NOTA: para resultados precisos, o cálculo formal de anti-dive depende
    de se o freio é INBOARD (preso ao chassi) ou OUTBOARD (preso à manga).
    Aqui assumimos OUTBOARD (caso comum em FSAE). Para freio inboard, o
    valor de anti-dive geometricamente possível é 0.
    """
    # Projeção em XZ dos pontos efetivos
    uca_in = corner.upper_arm.effective_inboard
    uca_out = corner.upper_arm.outboard
    lca_in = corner.lower_arm.effective_inboard
    lca_out = corner.lower_arm.outboard

    # Interseção das retas dos braços no plano XZ
    def line_intersect_xz(p1, p2, p3, p4):
        x1, z1 = p1.x, p1.z
        x2, z2 = p2.x, p2.z
        x3, z3 = p3.x, p3.z
        x4, z4 = p4.x, p4.z
        denom = (x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return None
        t = ((x1 - x3) * (z3 - z4) - (z1 - z3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), z1 + t * (z2 - z1))

    svic = line_intersect_xz(lca_in, lca_out, uca_in, uca_out)
    if svic is None:
        return 0.0   # braços paralelos em XZ → 0% anti-dive

    # Ângulo do SVIC visto do contact patch
    cp = corner.contact_patch
    dx = svic[0] - cp.x
    dz = svic[1] - cp.z
    if abs(dx) < 1e-9:
        return 0.0
    theta = math.atan2(dz, abs(dx))  # ângulo positivo se SVIC está acima

    # Wheelbase: precisa ser informado externamente (ou usa padrão)
    wb = wheelbase_mm_value if wheelbase_mm_value else 1550.0

    # Proteção contra cg_height inválido (evita ZeroDivision e valores negativos)
    if cg_height_mm <= 1e-6:
        return 0.0

    anti_dive = math.tan(theta) * wb / cg_height_mm * (brake_bias_pct / 100.0) * 100.0
    return float(anti_dive)


def anti_squat_percent(
    corner: SuspensionCorner,
    *,
    cg_height_mm:   float = 280.0,
    wheelbase_mm_value: Optional[float] = None,
    drive_type:     str = "RWD",
) -> float:
    """
    Anti-squat (%) — análogo ao anti-dive para a traseira.

    Funciona apenas para o corner traseiro (motor). Para FWD, retorna 0.
    """
    if drive_type.upper() not in ("RWD", "AWD"):
        return 0.0

    # Mesma lógica do anti-dive, mas para o corner traseiro
    return anti_dive_percent(
        corner,
        brake_bias_pct=100.0,   # toda a tração na traseira
        cg_height_mm=cg_height_mm,
        wheelbase_mm_value=wheelbase_mm_value,
    )


# =============================================================================
# KPI Bundle — reúne tudo em um único dict
# =============================================================================

@dataclass
class FullKPIReport:
    """
    Relatório completo de KPIs para um veículo, no formato da ficha de setup.

    Os campos None são aqueles que não são calculáveis sem dados externos
    (rigidez de mola, motion ratio, etc.) — fica como input do usuário.
    """
    # Dimensões
    wheelbase_mm: float
    track_front_mm: float
    track_rear_mm: float

    # Por ROW (linha) — dianteiro e traseiro separados
    front: dict[str, float]
    rear:  dict[str, float]


def build_full_report(
    vehicle: Vehicle,
    tie_rods: dict[str, TieRod],
    *,
    cg_height_mm: float = 280.0,
    brake_bias_pct: float = 60.0,
    drive_type: str = "RWD",
    roll_stiffness_deg_per_g: float = 1.5,
) -> FullKPIReport:
    """
    Constrói o relatório completo de KPIs para o veículo.

    Parâmetros externos (que não vêm dos hardpoints):
        cg_height_mm              : altura do CG
        brake_bias_pct            : % de frenagem na dianteira
        drive_type                : "RWD", "FWD" ou "AWD"
        roll_stiffness_deg_per_g  : roll por g lateral (tunable, depende
                                     das molas e ARB; valor típico FSAE)
    """
    fl, fr = vehicle.front_left,  vehicle.front_right
    rl, rr = vehicle.rear_left,   vehicle.rear_right
    tr_fl, tr_fr = tie_rods["FL"], tie_rods["FR"]
    tr_rl, tr_rr = tie_rods["RL"], tie_rods["RR"]

    wb     = wheelbase_mm(fl, rl)
    tr_f   = track_width_mm(fl, fr)
    tr_r   = track_width_mm(rl, rr)

    # --- Dianteiro ---
    ack_geom = ackermann_geometry(fl, tr_fl, fr, tr_fr, rl)
    steer_info = steer_ratio_and_cfactor(fl, tr_fl)
    rc_1g_front = roll_center_at_1g_lat(
        fl, tr_fl, fr, tr_fr,
        cg_height_mm=cg_height_mm,
        roll_stiffness_deg_per_g=roll_stiffness_deg_per_g,
    )

    front = {
        # Estáticos
        "static_camber_left":   fl.static_camber_deg(),
        "static_camber_right":  fr.static_camber_deg(),
        "static_sum_toe":       static_sum_toe_deg(fl, tr_fl, fr, tr_fr),
        "caster_left":          fl.static_caster_deg(),
        "caster_right":         fr.static_caster_deg(),
        "kpi_left":             fl.static_kpi_deg(),
        "kpi_right":            fr.static_kpi_deg(),
        "scrub_left":           fl.static_scrub_radius_mm(),
        "scrub_right":          fr.static_scrub_radius_mm(),
        "trail_left":           fl.static_mechanical_trail_mm(),
        "trail_right":          fr.static_mechanical_trail_mm(),
        # RC
        "rc_height_static":     0.5*(fl.roll_center_height_mm() + fr.roll_center_height_mm()),
        "rc_y_at_1g":           rc_1g_front["rc_y_mm"],
        "rc_z_at_1g":           rc_1g_front["rc_z_mm"],
        # Dinâmicos
        "ride_camber_deg_per_m": ride_camber_deg_per_m(fl, tr_fl),
        "roll_camber":           roll_camber_deg_per_deg(fl, tr_fl),
        "anti_dive_pct":         anti_dive_percent(
                                    fl, brake_bias_pct=brake_bias_pct,
                                    cg_height_mm=cg_height_mm, wheelbase_mm_value=wb),
        # Direção
        "ackermann_pct":        ack_geom["ackermann_percent"],
        "steer_arm_length_l":   ack_geom["steer_arm_length_left"],
        "steer_arm_length_r":   ack_geom["steer_arm_length_right"],
        "rack_per_deg":         steer_info["rack_per_wheel_deg_mm_per_deg"],
        "wheel_deg_per_rack":   steer_info["wheel_deg_per_rack_mm"],
    }

    # --- Traseiro ---
    rc_1g_rear = roll_center_at_1g_lat(
        rl, tr_rl, rr, tr_rr,
        cg_height_mm=cg_height_mm,
        roll_stiffness_deg_per_g=roll_stiffness_deg_per_g,
    )

    rear = {
        "static_camber_left":   rl.static_camber_deg(),
        "static_camber_right":  rr.static_camber_deg(),
        "static_sum_toe":       static_sum_toe_deg(rl, tr_rl, rr, tr_rr),
        "kpi_left":             rl.static_kpi_deg(),
        "kpi_right":            rr.static_kpi_deg(),
        "scrub_left":           rl.static_scrub_radius_mm(),
        "scrub_right":          rr.static_scrub_radius_mm(),
        "rc_height_static":     0.5*(rl.roll_center_height_mm() + rr.roll_center_height_mm()),
        "rc_y_at_1g":           rc_1g_rear["rc_y_mm"],
        "rc_z_at_1g":           rc_1g_rear["rc_z_mm"],
        "ride_camber_deg_per_m": ride_camber_deg_per_m(rl, tr_rl),
        "roll_camber":           roll_camber_deg_per_deg(rl, tr_rl),
        "anti_squat_pct":        anti_squat_percent(
                                    rl, cg_height_mm=cg_height_mm,
                                    wheelbase_mm_value=wb,
                                    drive_type=drive_type),
    }

    return FullKPIReport(
        wheelbase_mm=wb,
        track_front_mm=tr_f,
        track_rear_mm=tr_r,
        front=front,
        rear=rear,
    )