"""
analysis/viz3d.py
=================
Visualização 3D dos hardpoints e da suspensão usando Plotly.

Funções principais:
    plot_corner_3d       : 1 corner em isolamento (UCA, LCA, TR, manga, roda)
    plot_vehicle_3d      : todos os 4 corners + eixos/silhueta do chassi
    plot_corner_animated : versão animada com frames para sweep
                           (heave/roll/steer mostrado com slider)

CONVENÇÕES:
    - Pontos coloridos por grupo (UCA azul, LCA vermelho, TR verde, Wheel laranja)
    - Linhas finas conectam pontos do mesmo braço
    - Aro do pneu desenhado como círculo no plano da roda
    - Eixos do mundo (X+ frente, Y+ esquerda, Z+ cima) sempre visíveis
"""

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from geometry.primitives import Point3D
from geometry.model_3d import SuspensionCorner, Vehicle
from geometry.solver_3d import TieRod, KinematicSolver3D, KinematicState3D

if TYPE_CHECKING:
    import plotly.graph_objects as go


# =============================================================================
# Constantes de estilo
# =============================================================================

# Cor por grupo de pontos — consistente entre todas as visualizações
GROUP_COLORS: dict[str, str] = {
    "UCA":     "#1f77b4",   # azul
    "LCA":     "#d62728",   # vermelho
    "Tie-rod": "#2ca02c",   # verde
    "Wheel":   "#ff7f0e",   # laranja
    "Manga":   "#9467bd",   # roxo (linhas da manga)
    "Chassi":  "#7f7f7f",   # cinza (silhueta do chassi)
}

# Mapeamento de hardpoint → grupo
POINT_TO_GROUP: dict[str, str] = {
    "UCA_IN_FRONT":  "UCA",
    "UCA_IN_REAR":   "UCA",
    "UCA_OUT":       "UCA",
    "LCA_IN_FRONT":  "LCA",
    "LCA_IN_REAR":   "LCA",
    "LCA_OUT":       "LCA",
    "TIE_ROD_IN":    "Tie-rod",
    "TIE_ROD_OUT":   "Tie-rod",
    "WHEEL_CENTER":  "Wheel",
    "CONTACT_PATCH": "Wheel",
}


# =============================================================================
# Coleta dos pontos de um corner em estado arbitrário
# =============================================================================

def _collect_corner_points(
    corner:  SuspensionCorner,
    tie_rod: TieRod,
    state:   Optional[KinematicState3D] = None,
) -> dict[str, Point3D]:
    """
    Reúne todos os 10 hardpoints de um corner em um dict {nome → Point3D}.

    Se `state` for fornecido, usa as posições dinâmicas (outboards + WC + CP
    da manga rotacionada). Caso contrário, usa as posições estáticas originais.

    Os inboards do chassi vêm SEMPRE do corner original — eles se movem por
    heave/roll do CHASSI, não pela manga, e essa transformação é feita só na
    visualização animada.
    """
    pts: dict[str, Point3D] = {
        "UCA_IN_FRONT": corner.upper_arm.inboard_front,
        "UCA_IN_REAR":  corner.upper_arm.inboard_rear,
        "LCA_IN_FRONT": corner.lower_arm.inboard_front,
        "LCA_IN_REAR":  corner.lower_arm.inboard_rear,
        "TIE_ROD_IN":   tie_rod.inboard,
    }
    if state is None:
        pts["UCA_OUT"]       = corner.upper_arm.outboard
        pts["LCA_OUT"]       = corner.lower_arm.outboard
        pts["TIE_ROD_OUT"]   = tie_rod.outboard
        pts["WHEEL_CENTER"]  = corner.wheel_center
        pts["CONTACT_PATCH"] = corner.contact_patch
    else:
        pts["UCA_OUT"]       = state.uca_outboard
        pts["LCA_OUT"]       = state.lca_outboard
        pts["TIE_ROD_OUT"]   = state.tie_rod_outboard
        pts["WHEEL_CENTER"]  = state.wheel_center
        pts["CONTACT_PATCH"] = state.contact_patch
    return pts


def _move_chassis_points_for_state(
    points:   dict[str, Point3D],
    heave_mm: float,
    roll_deg: float,
) -> dict[str, Point3D]:
    """
    Aplica heave+roll aos pontos do chassi (apenas os inboards).
    Replica a lógica de `KinematicSolver3D._move_chassis_point` para manter
    consistência visual com o solver.
    """
    chassis_keys = {"UCA_IN_FRONT", "UCA_IN_REAR",
                    "LCA_IN_FRONT", "LCA_IN_REAR",
                    "TIE_ROD_IN"}
    if abs(heave_mm) < 1e-12 and abs(roll_deg) < 1e-12:
        return dict(points)

    cos_t = math.cos(math.radians(roll_deg))
    sin_t = math.sin(math.radians(roll_deg))

    result: dict[str, Point3D] = {}
    for name, p in points.items():
        if name in chassis_keys:
            # Rotação de roll em torno de X, depois translação Z (heave)
            y_new = p.y * cos_t - p.z * sin_t
            z_new = p.y * sin_t + p.z * cos_t + heave_mm
            result[name] = Point3D(p.x, y_new, z_new)
        else:
            result[name] = p
    return result


# =============================================================================
# Aro do pneu — círculo no plano da roda
# =============================================================================

def _generate_wheel_outline(
    wheel_center:  Point3D,
    contact_patch: Point3D,
    n_segments: int = 32,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Gera um círculo representando o contorno externo do pneu, no plano
    perpendicular ao eixo do hub.

    O eixo do hub é assumido como sendo PERPENDICULAR ao vetor WC→CP
    (e horizontal, paralelo a Y). Para uma roda com camber, esse eixo
    é ligeiramente inclinado.

    Retorna arrays (x, y, z) com `n_segments + 1` pontos formando um círculo
    fechado (último ponto = primeiro).
    """
    wc = wheel_center.to_array()
    cp = contact_patch.to_array()

    # Raio do pneu = distância WC → CP
    radius = float(np.linalg.norm(wc - cp))
    if radius < 1e-6:
        return (np.array([wc[0]]), np.array([wc[1]]), np.array([wc[2]]))

    # Vetor radial "para baixo" no plano da roda (WC→CP)
    radial_down = (cp - wc) / radius   # módulo = 1

    # Eixo do hub: perpendicular ao radial e aproximadamente paralelo a Y.
    # Aproximação simples: eixo Y projetado e ortonormalizado contra radial.
    y_axis = np.array([0.0, 1.0, 0.0])
    hub_axis = y_axis - np.dot(y_axis, radial_down) * radial_down
    hub_norm = float(np.linalg.norm(hub_axis))
    if hub_norm < 1e-6:
        # Caso degenerado: usa o eixo X
        hub_axis = np.array([1.0, 0.0, 0.0])
        hub_axis = hub_axis - np.dot(hub_axis, radial_down) * radial_down
        hub_norm = float(np.linalg.norm(hub_axis))
    hub_axis = hub_axis / hub_norm

    # Vetor tangencial no plano da roda: radial × hub_axis
    tangent = np.cross(radial_down, hub_axis)
    tangent = tangent / float(np.linalg.norm(tangent))

    # Gera o círculo
    angles = np.linspace(0, 2 * math.pi, n_segments + 1)
    xs = wc[0] + radius * (np.cos(angles) * radial_down[0] + np.sin(angles) * tangent[0])
    ys = wc[1] + radius * (np.cos(angles) * radial_down[1] + np.sin(angles) * tangent[1])
    zs = wc[2] + radius * (np.cos(angles) * radial_down[2] + np.sin(angles) * tangent[2])
    return (xs, ys, zs)


# =============================================================================
# Plot de UM corner
# =============================================================================

def plot_corner_3d(
    corner:    SuspensionCorner,
    tie_rod:   TieRod,
    *,
    state:     Optional[KinematicState3D] = None,
    show_tire: bool = True,
    title:     Optional[str] = None,
) -> "go.Figure":
    """
    Gera figura Plotly 3D de UM corner.

    Mostra:
        - 10 hardpoints como esferas coloridas por grupo
        - Braços UCA (2 linhas: front→out, rear→out)
        - Braços LCA (idem)
        - Tie-rod (1 linha)
        - Manga (triângulo UBJ-LBJ-TRO)
        - Roda (linha vertical WC→CP)
        - Pneu (aro circular no plano da roda)
        - Eixos de referência no canto

    Se `state` for fornecido (do solver), mostra a geometria deslocada.
    """
    import plotly.graph_objects as go

    pts = _collect_corner_points(corner, tie_rod, state)

    fig = go.Figure()

    # ─── Hardpoints (pontos) agrupados por categoria para legenda ────────────
    for group_name in ["UCA", "LCA", "Tie-rod", "Wheel"]:
        group_pts = [(name, p) for name, p in pts.items()
                      if POINT_TO_GROUP[name] == group_name]
        if not group_pts:
            continue
        xs = [p.x for _, p in group_pts]
        ys = [p.y for _, p in group_pts]
        zs = [p.z for _, p in group_pts]
        names = [name for name, _ in group_pts]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            marker=dict(size=6, color=GROUP_COLORS[group_name]),
            text=names,
            textfont=dict(size=9, color=GROUP_COLORS[group_name]),
            textposition="top center",
            name=group_name,
            hovertemplate="<b>%{text}</b><br>X=%{x:.1f}<br>Y=%{y:.1f}<br>Z=%{z:.1f}<extra></extra>",
        ))

    # ─── Linhas dos braços ───────────────────────────────────────────────────
    def add_line(p1: Point3D, p2: Point3D, color: str, name: str,
                  width: float = 4, dash: str = "solid",
                  showlegend: bool = False) -> None:
        fig.add_trace(go.Scatter3d(
            x=[p1.x, p2.x], y=[p1.y, p2.y], z=[p1.z, p2.z],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=name, showlegend=showlegend, hoverinfo="skip",
        ))

    # UCA: 2 linhas (front-out e rear-out)
    add_line(pts["UCA_IN_FRONT"], pts["UCA_OUT"], GROUP_COLORS["UCA"], "UCA")
    add_line(pts["UCA_IN_REAR"],  pts["UCA_OUT"], GROUP_COLORS["UCA"], "UCA")
    # LCA: idem
    add_line(pts["LCA_IN_FRONT"], pts["LCA_OUT"], GROUP_COLORS["LCA"], "LCA")
    add_line(pts["LCA_IN_REAR"],  pts["LCA_OUT"], GROUP_COLORS["LCA"], "LCA")
    # Tie-rod
    add_line(pts["TIE_ROD_IN"], pts["TIE_ROD_OUT"], GROUP_COLORS["Tie-rod"], "Tie-rod")
    # Manga: triângulo UBJ-LBJ-TRO
    add_line(pts["UCA_OUT"], pts["LCA_OUT"], GROUP_COLORS["Manga"], "Manga", width=3)
    add_line(pts["UCA_OUT"], pts["TIE_ROD_OUT"], GROUP_COLORS["Manga"], "Manga", width=2, dash="dot")
    add_line(pts["LCA_OUT"], pts["TIE_ROD_OUT"], GROUP_COLORS["Manga"], "Manga", width=2, dash="dot")
    # WC ↔ CP (eixo vertical da roda)
    add_line(pts["WHEEL_CENTER"], pts["CONTACT_PATCH"],
              GROUP_COLORS["Wheel"], "WC-CP", width=3, dash="dash")

    # ─── Aro do pneu ─────────────────────────────────────────────────────────
    if show_tire:
        xs, ys, zs = _generate_wheel_outline(pts["WHEEL_CENTER"], pts["CONTACT_PATCH"])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=GROUP_COLORS["Wheel"], width=2),
            name="Pneu", showlegend=False, hoverinfo="skip",
        ))

    # ─── Solo (plano Z=0 transparente em volta do contact patch) ─────────────
    cp = pts["CONTACT_PATCH"]
    floor_size = 200.0
    fig.add_trace(go.Mesh3d(
        x=[cp.x - floor_size, cp.x + floor_size, cp.x + floor_size, cp.x - floor_size],
        y=[cp.y - floor_size, cp.y - floor_size, cp.y + floor_size, cp.y + floor_size],
        z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="lightgray", opacity=0.2,
        name="Solo", showlegend=False, hoverinfo="skip",
    ))

    # ─── Layout: eixos com aspecto físico (1:1:1) ───────────────────────────
    fig.update_layout(
        title=title or f"Corner {corner.corner_id}",
        scene=dict(
            xaxis_title="X (mm) — frente",
            yaxis_title="Y (mm) — esquerda",
            zaxis_title="Z (mm) — cima",
            aspectmode="data",   # mantém escala física real entre os 3 eixos
            camera=dict(eye=dict(x=1.3, y=-1.5, z=1.0)),
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
    )

    return fig


# =============================================================================
# Plot do VEÍCULO COMPLETO
# =============================================================================

def plot_vehicle_3d(
    vehicle:  Vehicle,
    tie_rods: dict[str, TieRod],
    *,
    show_tires:  bool = True,
    show_chassis_box: bool = True,
    title: Optional[str] = None,
) -> "go.Figure":
    """
    Gera figura Plotly 3D do veículo completo (4 corners).

    Args:
        vehicle    : objeto Vehicle com os 4 corners
        tie_rods   : dict {"FL", "FR", "RL", "RR"} → TieRod
        show_chassis_box : se True, desenha um wireframe simplificado do
                            chassi conectando os inboards dos 4 cantos
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    corners_map = {
        "FL": vehicle.front_left,
        "FR": vehicle.front_right,
        "RL": vehicle.rear_left,
        "RR": vehicle.rear_right,
    }

    # ─── Coleta de pontos por grupo para a legenda (uma entrada por grupo) ──
    all_pts: dict[str, list[tuple[str, Point3D]]] = {
        "UCA": [], "LCA": [], "Tie-rod": [], "Wheel": [],
    }
    for cid, corner in corners_map.items():
        tr = tie_rods[cid]
        for name, p in _collect_corner_points(corner, tr).items():
            grp = POINT_TO_GROUP[name]
            all_pts[grp].append((f"{cid}_{name}", p))

    # Pontos (legenda única por grupo, mas hover mostra cada ponto)
    for grp, ptlist in all_pts.items():
        xs = [p.x for _, p in ptlist]
        ys = [p.y for _, p in ptlist]
        zs = [p.z for _, p in ptlist]
        names = [name for name, _ in ptlist]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=4, color=GROUP_COLORS[grp]),
            text=names,
            name=grp,
            hovertemplate="<b>%{text}</b><br>X=%{x:.1f}<br>Y=%{y:.1f}<br>Z=%{z:.1f}<extra></extra>",
        ))

    # ─── Linhas (braços, manga, tie-rods) para cada corner ───────────────────
    def add_segment(p1: Point3D, p2: Point3D, color: str, width: int = 3,
                     dash: str = "solid") -> None:
        fig.add_trace(go.Scatter3d(
            x=[p1.x, p2.x], y=[p1.y, p2.y], z=[p1.z, p2.z],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            showlegend=False, hoverinfo="skip",
        ))

    for cid, corner in corners_map.items():
        tr = tie_rods[cid]
        pts = _collect_corner_points(corner, tr)
        # UCA
        add_segment(pts["UCA_IN_FRONT"], pts["UCA_OUT"], GROUP_COLORS["UCA"])
        add_segment(pts["UCA_IN_REAR"],  pts["UCA_OUT"], GROUP_COLORS["UCA"])
        # LCA
        add_segment(pts["LCA_IN_FRONT"], pts["LCA_OUT"], GROUP_COLORS["LCA"])
        add_segment(pts["LCA_IN_REAR"],  pts["LCA_OUT"], GROUP_COLORS["LCA"])
        # Tie-rod
        add_segment(pts["TIE_ROD_IN"], pts["TIE_ROD_OUT"], GROUP_COLORS["Tie-rod"])
        # Manga
        add_segment(pts["UCA_OUT"], pts["LCA_OUT"], GROUP_COLORS["Manga"], width=3)
        add_segment(pts["UCA_OUT"], pts["TIE_ROD_OUT"], GROUP_COLORS["Manga"], width=2, dash="dot")
        add_segment(pts["LCA_OUT"], pts["TIE_ROD_OUT"], GROUP_COLORS["Manga"], width=2, dash="dot")
        # WC-CP
        add_segment(pts["WHEEL_CENTER"], pts["CONTACT_PATCH"],
                     GROUP_COLORS["Wheel"], width=2, dash="dash")

        # Pneu
        if show_tires:
            xs, ys, zs = _generate_wheel_outline(pts["WHEEL_CENTER"], pts["CONTACT_PATCH"])
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines",
                line=dict(color=GROUP_COLORS["Wheel"], width=2),
                showlegend=False, hoverinfo="skip",
            ))

    # ─── Wireframe do chassi (conectando inboards dos 4 cantos) ──────────────
    if show_chassis_box:
        chassis_pts = {}
        for cid, corner in corners_map.items():
            # Usa o centroide dos 2 inboards UCA como "ponto do chassi superior"
            uca_in = corner.upper_arm.effective_inboard
            lca_in = corner.lower_arm.effective_inboard
            chassis_pts[f"{cid}_top"]    = uca_in
            chassis_pts[f"{cid}_bottom"] = lca_in

        # Conexões: arestas do "box" do chassi
        box_edges = [
            # Topo (UCA inboards)
            ("FL_top", "FR_top"), ("RL_top", "RR_top"),
            ("FL_top", "RL_top"), ("FR_top", "RR_top"),
            # Fundo (LCA inboards)
            ("FL_bottom", "FR_bottom"), ("RL_bottom", "RR_bottom"),
            ("FL_bottom", "RL_bottom"), ("FR_bottom", "RR_bottom"),
            # Verticais
            ("FL_top", "FL_bottom"), ("FR_top", "FR_bottom"),
            ("RL_top", "RL_bottom"), ("RR_top", "RR_bottom"),
        ]
        for a, b in box_edges:
            add_segment(chassis_pts[a], chassis_pts[b],
                         GROUP_COLORS["Chassi"], width=1, dash="dot")

    # ─── Solo ────────────────────────────────────────────────────────────────
    xs_floor = [vehicle.front_left.contact_patch.x + 200,
                vehicle.rear_left.contact_patch.x  - 200]
    ys_floor = [vehicle.front_left.contact_patch.y + 200,
                vehicle.front_right.contact_patch.y - 200]
    fig.add_trace(go.Mesh3d(
        x=[xs_floor[0], xs_floor[0], xs_floor[1], xs_floor[1]],
        y=[ys_floor[0], ys_floor[1], ys_floor[1], ys_floor[0]],
        z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="lightgray", opacity=0.15,
        name="Solo", showlegend=False, hoverinfo="skip",
    ))

    # ─── Layout ──────────────────────────────────────────────────────────────
    fig.update_layout(
        title=title or "Veículo completo",
        scene=dict(
            xaxis_title="X (mm) — frente",
            yaxis_title="Y (mm) — esquerda",
            zaxis_title="Z (mm) — cima",
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
        ),
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
    )

    return fig


# =============================================================================
# Plot ANIMADO de um corner (sweep)
# =============================================================================

def plot_corner_animated(
    corner:    SuspensionCorner,
    tie_rod:   TieRod,
    *,
    sweep_axis: str = "heave",      # "heave", "roll" ou "steer"
    sweep_min:  float = -25.0,
    sweep_max:  float =  25.0,
    n_frames:   int   = 15,
    show_tire:  bool  = True,
) -> "go.Figure":
    """
    Versão animada da visualização 3D de um corner.

    Adiciona um slider que percorre `n_frames` posições no eixo escolhido
    (heave em mm, roll em °, ou rack em mm) e mostra a geometria se movendo.

    Útil para ver visualmente como camber, scrub, etc. mudam durante o
    movimento da suspensão.
    """
    import plotly.graph_objects as go

    if sweep_axis not in ("heave", "roll", "steer"):
        raise ValueError(f"sweep_axis inválido: {sweep_axis}")

    # Gera frames resolvendo o solver em cada posição
    solver = KinematicSolver3D(corner, tie_rod)
    values = np.linspace(sweep_min, sweep_max, n_frames)

    # Resolve o estado em cada ponto (mantém continuidade)
    solver.reset_seed()
    frame_states: list[tuple[float, KinematicState3D, float, float]] = []
    # tuple: (slider_value, state, heave, roll)
    for v in values:
        if sweep_axis == "heave":
            state = solver.solve(float(v), 0.0, 0.0)
            frame_states.append((float(v), state, float(v), 0.0))
        elif sweep_axis == "roll":
            state = solver.solve(0.0, float(v), 0.0)
            frame_states.append((float(v), state, 0.0, float(v)))
        else:  # steer
            state = solver.solve(0.0, 0.0, float(v))
            frame_states.append((float(v), state, 0.0, 0.0))

    # Pega o estado neutro (mais próximo de zero) para o frame inicial
    idx_zero = int(np.argmin(np.abs(values)))
    initial_state = frame_states[idx_zero][1]
    initial_heave = frame_states[idx_zero][2]
    initial_roll  = frame_states[idx_zero][3]

    # ─── Frame inicial: monta a figura base ─────────────────────────────────
    fig = plot_corner_3d(corner, tie_rod, state=initial_state, show_tire=show_tire,
                          title=f"Corner {corner.corner_id} — {sweep_axis} sweep")

    # ─── Gera os frames de animação ─────────────────────────────────────────
    # Para cada frame, recalcula as posições incluindo os inboards do chassi
    # (que se movem em heave/roll, mas não em steer puro).
    frames: list[go.Frame] = []
    slider_steps: list[dict] = []

    for slider_val, state, heave, roll in frame_states:
        pts_static = _collect_corner_points(corner, tie_rod, state)
        pts = _move_chassis_points_for_state(pts_static, heave, roll)

        # Re-gera todas as traces (na mesma ordem da figura base)
        frame_data: list = []

        # 1. Markers por grupo
        for grp in ["UCA", "LCA", "Tie-rod", "Wheel"]:
            grp_pts = [(n, pts[n]) for n in pts if POINT_TO_GROUP[n] == grp]
            xs = [p.x for _, p in grp_pts]
            ys = [p.y for _, p in grp_pts]
            zs = [p.z for _, p in grp_pts]
            frame_data.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers+text",
                marker=dict(size=6, color=GROUP_COLORS[grp]),
                text=[n for n, _ in grp_pts],
                textfont=dict(size=9, color=GROUP_COLORS[grp]),
                textposition="top center",
            ))

        # 2. Linhas (mesma ordem do plot_corner_3d)
        def line_data(p1, p2):
            return go.Scatter3d(x=[p1.x, p2.x], y=[p1.y, p2.y], z=[p1.z, p2.z],
                                 mode="lines")

        line_specs = [
            (pts["UCA_IN_FRONT"], pts["UCA_OUT"]),
            (pts["UCA_IN_REAR"],  pts["UCA_OUT"]),
            (pts["LCA_IN_FRONT"], pts["LCA_OUT"]),
            (pts["LCA_IN_REAR"],  pts["LCA_OUT"]),
            (pts["TIE_ROD_IN"],   pts["TIE_ROD_OUT"]),
            (pts["UCA_OUT"],      pts["LCA_OUT"]),
            (pts["UCA_OUT"],      pts["TIE_ROD_OUT"]),
            (pts["LCA_OUT"],      pts["TIE_ROD_OUT"]),
            (pts["WHEEL_CENTER"], pts["CONTACT_PATCH"]),
        ]
        for p1, p2 in line_specs:
            frame_data.append(line_data(p1, p2))

        # 3. Aro do pneu
        if show_tire:
            xs_t, ys_t, zs_t = _generate_wheel_outline(
                pts["WHEEL_CENTER"], pts["CONTACT_PATCH"]
            )
            frame_data.append(go.Scatter3d(x=xs_t, y=ys_t, z=zs_t, mode="lines"))

        # 4. Solo (não muda)
        cp_static = pts["CONTACT_PATCH"]
        floor_size = 200.0
        frame_data.append(go.Mesh3d(
            x=[cp_static.x - floor_size, cp_static.x + floor_size,
               cp_static.x + floor_size, cp_static.x - floor_size],
            y=[cp_static.y - floor_size, cp_static.y - floor_size,
               cp_static.y + floor_size, cp_static.y + floor_size],
            z=[0, 0, 0, 0],
            i=[0, 0], j=[1, 2], k=[2, 3],
        ))

        # Adiciona o frame com label "{sweep_axis}={value:.1f}"
        unit = {"heave": "mm", "roll": "°", "steer": "mm"}[sweep_axis]
        frame_name = f"{slider_val:+.2f}"
        frames.append(go.Frame(data=frame_data, name=frame_name))

        slider_steps.append({
            "args": [[frame_name], {"frame": {"duration": 50, "redraw": True},
                                     "mode": "immediate"}],
            "label": f"{slider_val:+.1f} {unit}",
            "method": "animate",
        })

    fig.frames = frames

    # ─── Slider e botão Play ─────────────────────────────────────────────────
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.05, "y": -0.05,
            "buttons": [
                {"label": "▶ Play",
                 "method": "animate",
                 "args": [None, {"frame": {"duration": 100, "redraw": True},
                                  "fromcurrent": True}]},
                {"label": "⏸ Pause",
                 "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": idx_zero,
            "y": 0,  "x": 0.15,
            "len": 0.8,
            "currentvalue": {"prefix": f"{sweep_axis}: ", "visible": True},
            "steps": slider_steps,
        }],
    )

    return fig
