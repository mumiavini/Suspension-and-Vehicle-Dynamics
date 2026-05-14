"""
analysis/sweeps.py
==================
Varreduras paramétricas (SWEEPS) da cinemática da suspensão.

Um SWEEP é uma sequência de configurações (heave, roll, rack) executadas
em ordem, onde cada ponto usa o anterior como seed para o solver — isso
garante continuidade física do movimento.

TIPOS DE SWEEP:
    - Heave Sweep : varia o heave, com roll=0 e rack=0
    - Roll Sweep  : varia o roll,  com heave=0 e rack=0
    - Steer Sweep : varia o rack,  com heave=0 e roll=0

SAÍDA:
    np.ndarray com dtype estruturado (definido em SWEEP_DTYPE). Acesso por
    nome de coluna: `sweep["camber_deg"]`, `sweep["heave_mm"]`, etc.

MÉTRICAS DERIVADAS:
    - camber_gain_per_mm  : taxa de variação da cambagem com heave
    - bump_steer_per_mm   : taxa de variação do toe com heave
    - rc_migration_range  : amplitude de migração do Roll Center

PLOTAGEM (Plotly):
    Funções `plot_*` retornam objetos `plotly.graph_objects.Figure` para
    renderização no Streamlit ou Jupyter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from geometry.solver_3d import KinematicSolver3D, KinematicState3D

if TYPE_CHECKING:
    # plotly é importado APENAS para type hints — em runtime, é lazy import
    import plotly.graph_objects as go


# =============================================================================
# Dtype estruturado para o resultado dos sweeps
# =============================================================================

SWEEP_DTYPE: np.dtype = np.dtype([
    # Inputs aplicados
    ("heave_mm",   "f8"),
    ("roll_deg",   "f8"),
    ("rack_mm",    "f8"),
    # Ângulos derivados
    ("camber_deg", "f8"),
    ("toe_deg",    "f8"),
    ("caster_deg", "f8"),
    ("kpi_deg",    "f8"),
    # Posição do Roll Center na vista frontal
    ("rc_y_mm",    "f8"),
    ("rc_z_mm",    "f8"),
    # Posição do centro de roda
    ("wc_x_mm",    "f8"),
    ("wc_y_mm",    "f8"),
    ("wc_z_mm",    "f8"),
    # Diagnóstico
    ("residual",   "f8"),
    ("converged",  "?"),     # bool
])


# =============================================================================
# SweepRunner — Executor de varreduras
# =============================================================================

@dataclass
class SweepRunner:
    """
    Executa varreduras paramétricas usando um KinematicSolver3D.

    Uso:
        runner = SweepRunner(solver=my_solver)
        heave_data = runner.heave_sweep(-25.0, 25.0, 1.0)
        roll_data  = runner.roll_sweep(-3.0, 3.0, 0.2)
        steer_data = runner.steer_sweep(-30.0, 30.0, 1.0)

    Atributos:
        solver       : solver 3D já inicializado
        static_state : estado estático (cache, computado no __post_init__)
    """
    solver:       KinematicSolver3D
    static_state: Optional[KinematicState3D] = field(default=None)

    def __post_init__(self) -> None:
        """Computa o estado estático como referência."""
        self.solver.reset_seed()
        self.static_state = self.solver.solve(0.0, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # Sweeps padrão
    # -------------------------------------------------------------------------

    def heave_sweep(
        self,
        heave_min_mm: float = -25.0,
        heave_max_mm: float =  25.0,
        step_mm:      float =   1.0,
    ) -> NDArray:
        """
        Varredura de heave puro (bump/rebound), com roll=0 e rack=0.
        """
        values = np.arange(heave_min_mm, heave_max_mm + step_mm * 0.5, step_mm)
        configurations = [(float(h), 0.0, 0.0) for h in values]
        return self._run_sweep(configurations)

    def roll_sweep(
        self,
        roll_min_deg: float = -3.0,
        roll_max_deg: float =  3.0,
        step_deg:     float =  0.1,
    ) -> NDArray:
        """
        Varredura de rolagem do chassi, com heave=0 e rack=0.
        """
        values = np.arange(roll_min_deg, roll_max_deg + step_deg * 0.5, step_deg)
        configurations = [(0.0, float(r), 0.0) for r in values]
        return self._run_sweep(configurations)

    def steer_sweep(
        self,
        rack_min_mm: float = -30.0,
        rack_max_mm: float =  30.0,
        step_mm:     float =   1.0,
    ) -> NDArray:
        """
        Varredura de esterçamento (deslocamento do rack), com heave=0 e roll=0.
        """
        values = np.arange(rack_min_mm, rack_max_mm + step_mm * 0.5, step_mm)
        configurations = [(0.0, 0.0, float(r)) for r in values]
        return self._run_sweep(configurations)

    def combined_sweep(
        self,
        configurations: list[tuple[float, float, float]],
    ) -> NDArray:
        """
        Varredura arbitrária. Use para combinações tipo heave+roll simultâneo.

        IMPORTANTE: ordene as configurações para que pontos adjacentes estejam
        próximos no espaço de fase (o solver usa o anterior como seed).
        """
        return self._run_sweep(configurations)

    # -------------------------------------------------------------------------
    # Loop principal de execução
    # -------------------------------------------------------------------------

    def _run_sweep(
        self,
        configurations: list[tuple[float, float, float]],
    ) -> NDArray:
        """
        Executa o solver para cada configuração e preenche o array de resultado.
        """
        n = len(configurations)
        result = np.empty(n, dtype=SWEEP_DTYPE)

        # Reset do seed: o sweep começa da posição estática
        self.solver.reset_seed()

        for i, (heave, roll, rack) in enumerate(configurations):
            try:
                state = self.solver.solve(heave_mm=heave, roll_deg=roll, rack_mm=rack)
                self._fill_record(result, i, state)
            except Exception:
                # Solver falhou: marca o registro como não-convergido
                result[i] = self._make_failed_record(heave, roll, rack)

        return result

    # -------------------------------------------------------------------------
    # Preenche um registro do array com os dados do estado
    # -------------------------------------------------------------------------

    def _fill_record(
        self,
        arr:   NDArray,
        idx:   int,
        state: KinematicState3D,
    ) -> None:
        """Copia os campos do KinematicState3D para o registro do array."""
        rc_y, rc_z = self._estimate_roll_center_yz(state)

        arr[idx]["heave_mm"]   = state.heave_mm
        arr[idx]["roll_deg"]   = state.roll_deg
        arr[idx]["rack_mm"]    = state.rack_mm
        arr[idx]["camber_deg"] = state.camber_deg
        arr[idx]["toe_deg"]    = state.toe_deg
        arr[idx]["caster_deg"] = state.caster_deg
        arr[idx]["kpi_deg"]    = state.kpi_deg
        arr[idx]["rc_y_mm"]    = rc_y
        arr[idx]["rc_z_mm"]    = rc_z
        arr[idx]["wc_x_mm"]    = state.wheel_center.x
        arr[idx]["wc_y_mm"]    = state.wheel_center.y
        arr[idx]["wc_z_mm"]    = state.wheel_center.z
        arr[idx]["residual"]   = state.residual_norm
        arr[idx]["converged"]  = state.converged

    @staticmethod
    def _make_failed_record(heave: float, roll: float, rack: float) -> NDArray:
        """Registro padrão para falha do solver."""
        rec = np.zeros(1, dtype=SWEEP_DTYPE)[0]
        rec["heave_mm"]  = heave
        rec["roll_deg"]  = roll
        rec["rack_mm"]   = rack
        rec["residual"]  = np.nan
        rec["converged"] = False
        return rec

    # -------------------------------------------------------------------------
    # Estimativa do Roll Center a partir do estado 3D
    # -------------------------------------------------------------------------

    def _estimate_roll_center_yz(
        self,
        state: KinematicState3D,
    ) -> tuple[float, float]:
        """
        Estima o Roll Center na vista frontal Y-Z usando o método 2D padrão
        (Centro Instantâneo → linha até contact patch → plano de simetria).

        Usa os inboards efetivos do corner original, deslocados pela mesma
        transformação (heave + roll) que o solver aplicou.
        """
        from geometry.primitives import Point2D, line_intersection_2d

        corner = self.solver.corner
        uca_in_eff = corner.upper_arm.effective_inboard
        lca_in_eff = corner.lower_arm.effective_inboard

        # Aplica o movimento do chassi nos inboards efetivos (cópia da lógica
        # do solver para consistência)
        uca_in_arr = self.solver._move_chassis_point(
            uca_in_eff, state.heave_mm, state.roll_deg
        )
        lca_in_arr = self.solver._move_chassis_point(
            lca_in_eff, state.heave_mm, state.roll_deg
        )

        # Projeções no plano Y-Z (u=Y, v=Z)
        uca_in_2d  = Point2D(float(uca_in_arr[1]),  float(uca_in_arr[2]))
        uca_out_2d = Point2D(state.uca_outboard.y,  state.uca_outboard.z)
        lca_in_2d  = Point2D(float(lca_in_arr[1]),  float(lca_in_arr[2]))
        lca_out_2d = Point2D(state.lca_outboard.y,  state.lca_outboard.z)

        # Centro Instantâneo: interseção das prolongações dos braços
        try:
            ic = line_intersection_2d(lca_in_2d, lca_out_2d, uca_in_2d, uca_out_2d)
        except ValueError:
            return (0.0, 0.0)   # braços paralelos: RC no nível do solo

        # Linha IC → contact patch, intersectada com Y=0 (plano de simetria)
        cp = Point2D(state.contact_patch.y, state.contact_patch.z)
        du = cp.u - ic.u
        if abs(du) < 1e-12:
            return (0.0, ic.v)

        t = -ic.u / du
        v_rc = ic.v + t * (cp.v - ic.v)
        return (0.0, float(v_rc))


# =============================================================================
# Métricas derivadas dos sweeps
# =============================================================================

def camber_gain_per_mm(sweep: NDArray) -> float:
    """
    Camber gain (°/mm) — regressão linear de camber vs heave.

    Para um heave sweep, retorna a INCLINAÇÃO da reta que melhor ajusta
    os pontos (camber_deg) em função de (heave_mm).

    TÍPICO FSAE: −0.005 a −0.025 °/mm
    """
    mask = sweep["converged"]
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(sweep["heave_mm"][mask], sweep["camber_deg"][mask], 1)
    return float(slope)


def bump_steer_per_mm(sweep: NDArray) -> float:
    """
    Bump steer (°/mm) — regressão linear de toe vs heave.

    Quanto a roda esterça (involuntariamente) quando faz bump/rebound.
    Deve ser MINIMIZADO (idealmente < 0.005°/mm).
    """
    mask = sweep["converged"]
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(sweep["heave_mm"][mask], sweep["toe_deg"][mask], 1)
    return float(slope)


def rc_migration_range(sweep: NDArray) -> tuple[float, float]:
    """
    Amplitude de migração do Roll Center: (ΔY, ΔZ) em mm.

    ΔY = quanto o RC migra lateralmente
    ΔZ = quanto o RC migra verticalmente

    Idealmente próximos de zero (RC estável).
    """
    mask = sweep["converged"]
    if mask.sum() < 2:
        return (float("nan"), float("nan"))
    dy = float(sweep["rc_y_mm"][mask].max() - sweep["rc_y_mm"][mask].min())
    dz = float(sweep["rc_z_mm"][mask].max() - sweep["rc_z_mm"][mask].min())
    return (dy, dz)


# =============================================================================
# Plotagem (Plotly) — imports lazy para não exigir plotly fora dos plots
# =============================================================================

def plot_camber_vs_heave(sweep: NDArray, title: str = "Camber vs Heave") -> "go.Figure":
    """Gráfico: Camber (°) versus Heave (mm)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["heave_mm"],
        y=sweep["camber_deg"],
        mode="lines+markers",
        name="Camber",
        line=dict(width=2),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Heave (mm)  [+ = bump]",
        yaxis_title="Camber (°)  [− = topo p/ dentro]",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_bump_steer(sweep: NDArray, title: str = "Bump Steer") -> "go.Figure":
    """
    Gráfico: variação de toe (°) versus Heave (mm).

    NOTA: o solver já retorna toe como DELTA relativo ao estado estático,
    então pode ser plotado diretamente.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["heave_mm"],
        y=sweep["toe_deg"],
        mode="lines+markers",
        name="Δ Toe",
        line=dict(width=2, color="darkorange"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Heave (mm)",
        yaxis_title="Δ Toe (°)  [+ = toe-in]",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_rc_migration(sweep: NDArray, title: str = "Roll Center Migration") -> "go.Figure":
    """
    Gráfico: trajetória do Roll Center no plano Y-Z, colorida pelo heave.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["rc_y_mm"],
        y=sweep["rc_z_mm"],
        mode="lines+markers",
        marker=dict(
            size=6,
            color=sweep["heave_mm"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Heave (mm)"),
        ),
        line=dict(width=1, color="gray"),
        name="RC trajectory",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="RC Y (mm)",
        yaxis_title="RC Z (mm)",
        template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)   # eixos isométricos
    return fig


def plot_caster_kpi_vs_steer(
    sweep: NDArray,
    title: str = "Caster & KPI vs Steer",
) -> "go.Figure":
    """Gráfico: Caster e KPI (°) versus deslocamento do rack (mm)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["rack_mm"], y=sweep["caster_deg"],
        mode="lines+markers", name="Caster (°)",
        line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=sweep["rack_mm"], y=sweep["kpi_deg"],
        mode="lines+markers", name="KPI (°)",
        line=dict(width=2, dash="dash"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Rack (mm)",
        yaxis_title="Ângulo (°)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig
