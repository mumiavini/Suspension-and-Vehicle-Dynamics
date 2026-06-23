"""
geometry/solver_3d.py
=====================
Solver cinemático 3D — movimento da manga de eixo como CORPO RÍGIDO.

CONCEITO FÍSICO
---------------
A manga de eixo tem TRÊS pontos de ancoragem (ball joints):
    - UBJ (Upper Ball Joint)  : outboard do UCA
    - LBJ (Lower Ball Joint)  : outboard do LCA
    - TRO (Tie-Rod Outboard)  : outboard do tie-rod

Cada um deve manter distância FIXA do respectivo ponto inboard (3 esferas
no espaço 3D). Adicionalmente, as três distâncias INTERNAS da manga
(UBJ-LBJ, UBJ-TRO, LBJ-TRO) também devem ser preservadas (corpo rígido).

SISTEMA NÃO-LINEAR A RESOLVER (9 incógnitas, 6 equações):
    Para cada ball joint i:
        (X_i - x_i_in)² + (Y_i - y_i_in)² + (Z_i - z_i_in)² = L_i²

    Mais 3 equações de corpo rígido:
        |UBJ - LBJ| = const     (dist. estática)
        |UBJ - TRO| = const
        |LBJ - TRO| = const

Como temos 9 DOF (3 pontos × 3 coords) e 6 constraints, restam 3 DOF.
Adicionamos REGULARIZAÇÃO SUAVE (ancorar perto da posição anterior)
para que o sistema seja bem-condicionado.

ALGORITMO: scipy.optimize.least_squares com Levenberg-Marquardt.

ENTRADAS DO SOLVER:
    - heave_mm : deslocamento vertical do chassi
    - roll_deg : ângulo de rolagem do chassi (em torno do eixo X)
    - rack_mm  : deslocamento do rack de direção (em Y)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from geometry.primitives import Point3D, Vector3D
from geometry.model_3d import SuspensionCorner, ControlArm


# =============================================================================
# Tie-Rod (terminal de direção)
# =============================================================================

@dataclass
class TieRod:
    """
    Tie-rod: barra que liga o rack/braço pitman à manga.

    Atributos:
        inboard  : ponto fixo no rack (move-se com o chassi + offset do rack)
        outboard : ponto na manga (rotaciona com ela)
    """
    inboard:  Point3D
    outboard: Point3D
    name:     str = "TieRod"

    @property
    def length(self) -> float:
        """Comprimento do tie-rod (mm), invariante durante o movimento."""
        return self.inboard.distance_to(self.outboard)

    def __repr__(self) -> str:
        return f"TieRod('{self.name}', length={self.length:.2f} mm)"


# =============================================================================
# Estado cinemático 3D (resultado de uma resolução)
# =============================================================================

@dataclass
class KinematicState3D:
    """
    Estado da suspensão para uma dada configuração (heave, roll, rack).

    Inputs:
        heave_mm, roll_deg, rack_mm

    Posições resolvidas:
        uca_outboard, lca_outboard, tie_rod_outboard, wheel_center, contact_patch

    Ângulos derivados:
        camber_deg, toe_deg, caster_deg, kpi_deg

    Diagnóstico do solver:
        converged, residual_norm, iterations
    """
    heave_mm: float = 0.0
    roll_deg: float = 0.0
    rack_mm:  float = 0.0

    uca_outboard:     Point3D = field(default_factory=lambda: Point3D(0, 0, 0))
    lca_outboard:     Point3D = field(default_factory=lambda: Point3D(0, 0, 0))
    tie_rod_outboard: Point3D = field(default_factory=lambda: Point3D(0, 0, 0))
    wheel_center:     Point3D = field(default_factory=lambda: Point3D(0, 0, 0))
    contact_patch:    Point3D = field(default_factory=lambda: Point3D(0, 0, 0))

    camber_deg: float = 0.0
    toe_deg:    float = 0.0
    caster_deg: float = 0.0
    kpi_deg:    float = 0.0

    converged:     bool  = True
    residual_norm: float = 0.0
    iterations:    int   = 0


# =============================================================================
# Solver 3D
# =============================================================================

class KinematicSolver3D:
    """
    Resolve a cinemática 3D da manga de eixo via least_squares.

    Uso típico:
        solver = KinematicSolver3D(corner, tie_rod)
        state  = solver.solve(heave_mm=10.0, roll_deg=0.5, rack_mm=0.0)

    Para sweeps (varreduras), o solver mantém um cache do último estado
    como SEED para o próximo, garantindo continuidade física. Use
    `solver.reset_seed()` ao iniciar um novo sweep.
    """

    # Peso da regularização suave (usado em _residuals e _residuals_jac)
    _REG_WEIGHT: float = 1e-4

    def __init__(
        self,
        corner: SuspensionCorner,
        tie_rod: TieRod,
        *,
        tolerance: float = 1e-9,
        max_iter:  int   = 100,
    ) -> None:
        """
        Inicializa o solver pré-calculando todas as distâncias invariantes.
        """
        self.corner: SuspensionCorner = corner
        self.tie_rod: TieRod           = tie_rod
        self.tolerance: float          = tolerance
        self.max_iter: int             = max_iter

        # ─── Comprimentos dos elos (invariantes) ──────────────────────────────
        self._L_uca: float = corner.upper_arm.arm_length()
        self._L_lca: float = corner.lower_arm.arm_length()
        self._L_tr:  float = tie_rod.length

        # ─── Distâncias internas da manga (corpo rígido) ──────────────────────
        ubj = corner.upper_arm.outboard.to_array()
        lbj = corner.lower_arm.outboard.to_array()
        tro = tie_rod.outboard.to_array()
        self._d_ubj_lbj: float = float(np.linalg.norm(ubj - lbj))
        self._d_ubj_tro: float = float(np.linalg.norm(ubj - tro))
        self._d_lbj_tro: float = float(np.linalg.norm(lbj - tro))

        # ─── Offsets locais do WC e CP em relação ao referencial da manga ────
        # Precisamos disso para reconstruir as posições do WC/CP após a manga
        # girar. Calculados UMA vez aqui no init.
        self._wc_local_offset: NDArray[np.float64] = self._compute_local_offset(
            corner.wheel_center.to_array(), ubj, lbj, tro
        )
        self._cp_local_offset: NDArray[np.float64] = self._compute_local_offset(
            corner.contact_patch.to_array(), ubj, lbj, tro
        )

        # ─── Toe DE REFERÊNCIA (estado estático) ──────────────────────────────
        # Pré-calcula o toe na posição estática para que reportemos sempre
        # o DELTA em relação a este zero. Isso elimina o offset arbitrário
        # do toe absoluto.
        self._toe_static: float = self._compute_toe_absolute(
            corner.upper_arm.outboard, corner.lower_arm.outboard,
            tie_rod.outboard, corner.wheel_center,
        )

        # Cache do último estado (para usar como seed do próximo)
        self._last_state: Optional[KinematicState3D] = None

    # =========================================================================
    # MÉTODO PRINCIPAL: resolver
    # =========================================================================

    def solve(
        self,
        heave_mm: float = 0.0,
        roll_deg: float = 0.0,
        rack_mm:  float = 0.0,
    ) -> KinematicState3D:
        """
        Resolve a cinemática 3D para uma configuração (heave, roll, rack).

        Parâmetros:
            heave_mm : deslocamento vertical do chassi (+ = chassi sobe)
            roll_deg : rolagem do chassi em torno do eixo X
                       (+ = chassi rola para a direita; lado esquerdo desce)
            rack_mm  : deslocamento lateral do rack (+ = para a esquerda)
        """
        # ─── 1. Mover os pontos inboard (chassi + rack) ───────────────────────
        uca_in_eff = self.corner.upper_arm.effective_inboard
        lca_in_eff = self.corner.lower_arm.effective_inboard
        tr_in      = self.tie_rod.inboard

        uca_in_moved = self._move_chassis_point(uca_in_eff, heave_mm, roll_deg)
        lca_in_moved = self._move_chassis_point(lca_in_eff, heave_mm, roll_deg)
        tr_in_moved  = self._move_chassis_point(tr_in,      heave_mm, roll_deg)
        tr_in_moved[1] += rack_mm   # rack desloca-se lateralmente em Y

        # ─── 2. Construir seed inicial (último estado ou estático) ────────────
        if self._last_state is not None:
            seed = np.concatenate([
                self._last_state.uca_outboard.to_array(),
                self._last_state.lca_outboard.to_array(),
                self._last_state.tie_rod_outboard.to_array(),
            ])
        else:
            seed = np.concatenate([
                self.corner.upper_arm.outboard.to_array(),
                self.corner.lower_arm.outboard.to_array(),
                self.tie_rod.outboard.to_array(),
            ])

        # ─── 3. Resolver o sistema não-linear ────────────────────────────────
        # jac analítico: evita ~10 avaliações numéricas do residual por passo
        # do LM (diferenças finitas em 9 variáveis) → solver ~3-5× mais rápido.
        result = least_squares(
            fun=self._residuals,
            jac=self._residuals_jac,
            x0=seed,
            args=(uca_in_moved, lca_in_moved, tr_in_moved, seed),
            method="lm",                   # Levenberg-Marquardt
            xtol=self.tolerance,
            ftol=self.tolerance,
            max_nfev=self.max_iter * 10,
        )

        # ─── 4. Extrair posições resolvidas ──────────────────────────────────
        x = result.x
        ubj = x[0:3]; lbj = x[3:6]; tro = x[6:9]
        uca_out = Point3D.from_array(ubj)
        lca_out = Point3D.from_array(lbj)
        tr_out  = Point3D.from_array(tro)

        # ─── 5. Reconstruir WC e CP a partir do referencial local da manga ────
        wc_arr = self._reconstruct_from_local(self._wc_local_offset, ubj, lbj, tro)
        cp_arr = self._reconstruct_from_local(self._cp_local_offset, ubj, lbj, tro)
        wheel_center  = Point3D.from_array(wc_arr)
        contact_patch = Point3D.from_array(cp_arr)

        # ─── 6. Calcular ângulos derivados ───────────────────────────────────
        camber = self._compute_camber(uca_out, lca_out, wheel_center, contact_patch)
        caster = self._compute_caster(uca_out, lca_out)
        kpi    = self._compute_kpi(uca_out, lca_out)

        # Toe RELATIVO ao estado estático (= bump steer + steer angle)
        toe_abs = self._compute_toe_absolute(uca_out, lca_out, tr_out, wheel_center)
        toe = toe_abs - self._toe_static

        state = KinematicState3D(
            heave_mm=heave_mm, roll_deg=roll_deg, rack_mm=rack_mm,
            uca_outboard=uca_out,
            lca_outboard=lca_out,
            tie_rod_outboard=tr_out,
            wheel_center=wheel_center,
            contact_patch=contact_patch,
            camber_deg=camber,
            toe_deg=toe,
            caster_deg=caster,
            kpi_deg=kpi,
            converged=result.success,
            residual_norm=float(np.linalg.norm(result.fun)),
            iterations=int(result.nfev),
        )

        # Cache para a próxima chamada (continuidade no sweep)
        self._last_state = state
        return state

    def reset_seed(self) -> None:
        """Limpa o cache. Use ao iniciar um novo sweep."""
        self._last_state = None

    # =========================================================================
    # Função residual (least_squares minimiza ||residuals||²)
    # =========================================================================

    def _residuals(
        self,
        x:            NDArray[np.float64],
        uca_in_moved: NDArray[np.float64],
        lca_in_moved: NDArray[np.float64],
        tr_in_moved:  NDArray[np.float64],
        seed:         NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Vetor de resíduos. O solver minimiza a soma dos quadrados destes.

        Composição:
            r[0..2] : 3 constraints de distância para os inboards (esferas)
            r[3..5] : 3 constraints de distância interna da manga (rigid body)
            r[6..14]: 9 termos de regularização suave (ancora perto do seed)
        """
        ubj = x[0:3]; lbj = x[3:6]; tro = x[6:9]

        # 1. Distâncias para os inboards (esferas)
        r_ubj = np.linalg.norm(ubj - uca_in_moved) - self._L_uca
        r_lbj = np.linalg.norm(lbj - lca_in_moved) - self._L_lca
        r_tro = np.linalg.norm(tro - tr_in_moved)  - self._L_tr

        # 2. Distâncias internas da manga (corpo rígido)
        r_d1 = np.linalg.norm(ubj - lbj) - self._d_ubj_lbj
        r_d2 = np.linalg.norm(ubj - tro) - self._d_ubj_tro
        r_d3 = np.linalg.norm(lbj - tro) - self._d_lbj_tro

        # 3. Regularização suave (peso muito pequeno para não dominar)
        reg = (x - seed) * self._REG_WEIGHT

        return np.concatenate([
            np.array([r_ubj, r_lbj, r_tro, r_d1, r_d2, r_d3]),
            reg,
        ])

    def _residuals_jac(
        self,
        x:            NDArray[np.float64],
        uca_in_moved: NDArray[np.float64],
        lca_in_moved: NDArray[np.float64],
        tr_in_moved:  NDArray[np.float64],
        seed:         NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Jacobiano ANALÍTICO de `_residuals` (15 resíduos × 9 variáveis).

        Para um resíduo de distância r = ||a − b|| − L:
            ∂r/∂a = (a − b) / ||a − b||      (e ∂r/∂b = −∂r/∂a)

        As linhas de regularização são simplesmente I₉ × _REG_WEIGHT.
        """
        ubj = x[0:3]; lbj = x[3:6]; tro = x[6:9]

        def unit(d: NDArray[np.float64]) -> NDArray[np.float64]:
            n = float(np.linalg.norm(d))
            return d / n if n > 1e-12 else np.zeros(3)

        J = np.zeros((15, 9))

        # 1. Distâncias para os inboards (só dependem do próprio ponto)
        J[0, 0:3] = unit(ubj - uca_in_moved)
        J[1, 3:6] = unit(lbj - lca_in_moved)
        J[2, 6:9] = unit(tro - tr_in_moved)

        # 2. Distâncias internas da manga (par de pontos, sinais opostos)
        u = unit(ubj - lbj); J[3, 0:3] = u; J[3, 3:6] = -u
        u = unit(ubj - tro); J[4, 0:3] = u; J[4, 6:9] = -u
        u = unit(lbj - tro); J[5, 3:6] = u; J[5, 6:9] = -u

        # 3. Regularização
        J[6:15, :] = np.eye(9) * self._REG_WEIGHT
        return J

    # =========================================================================
    # Movimento dos pontos do chassi (heave + roll)
    # =========================================================================

    @staticmethod
    def _move_chassis_point(
        point:    Point3D,
        heave_mm: float,
        roll_deg: float,
    ) -> NDArray[np.float64]:
        """
        Aplica heave (translação Z) e roll (rotação em X) a um ponto do chassi.

        ORDEM DAS TRANSFORMAÇÕES:
            1. Roll: rotação em torno do eixo X (longitudinal), origem em Y=Z=0
            2. Heave: translação em Z

        Para o eixo de rolagem real (que não passa pela origem), a rigor
        deveríamos transladar para o RC, rotacionar, e desfazer. Aqui usamos
        a aproximação simples (rotação na origem), que é válida para
        ângulos de roll pequenos típicos de FSAE (< 3°).
        """
        p = point.to_array().copy()

        # Roll em torno do eixo X (Y e Z são rotacionados)
        if abs(roll_deg) > 1e-12:
            theta = math.radians(roll_deg)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            y_new = p[1] * cos_t - p[2] * sin_t
            z_new = p[1] * sin_t + p[2] * cos_t
            p[1] = y_new
            p[2] = z_new

        # Heave
        p[2] += heave_mm
        return p

    # =========================================================================
    # Referencial local da manga (para reconstruir WC e CP)
    # =========================================================================

    @staticmethod
    def _build_local_frame(
        ubj: NDArray[np.float64],
        lbj: NDArray[np.float64],
        tro: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Constrói uma base ORTONORMAL fixa na manga, ancorada em LBJ.

        Procedimento (Gram-Schmidt):
            e1 = (UBJ - LBJ) / |UBJ - LBJ|         direção do pino mestre
            e2 = (TRO - LBJ) - (projetado em e1)    ortogonal a e1
            e3 = e1 × e2                            completa a base destro

        Essa base ROTACIONA junto com a manga, mantendo as coordenadas
        locais de qualquer ponto SOLIDÁRIO à manga constantes.
        """
        v1 = ubj - lbj
        e1 = v1 / np.linalg.norm(v1)

        v2 = tro - lbj
        v2_perp = v2 - np.dot(v2, e1) * e1
        e2 = v2_perp / np.linalg.norm(v2_perp)

        e3 = np.cross(e1, e2)
        return e1, e2, e3

    @classmethod
    def _compute_local_offset(
        cls,
        point: NDArray[np.float64],
        ubj:   NDArray[np.float64],
        lbj:   NDArray[np.float64],
        tro:   NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Calcula as coordenadas de `point` no referencial local da manga.
        Estes valores são INVARIANTES durante o movimento (manga rígida).
        """
        e1, e2, e3 = cls._build_local_frame(ubj, lbj, tro)
        delta = point - lbj
        return np.array([
            float(np.dot(delta, e1)),
            float(np.dot(delta, e2)),
            float(np.dot(delta, e3)),
        ])

    @classmethod
    def _reconstruct_from_local(
        cls,
        local_offset: NDArray[np.float64],
        ubj: NDArray[np.float64],
        lbj: NDArray[np.float64],
        tro: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Reconstrói a posição GLOBAL de um ponto a partir de seu offset local.
        Usa o referencial local da manga na configuração ATUAL.
        """
        e1, e2, e3 = cls._build_local_frame(ubj, lbj, tro)
        return lbj + local_offset[0] * e1 + local_offset[1] * e2 + local_offset[2] * e3

    # =========================================================================
    # Cálculo dos ângulos derivados
    # =========================================================================

    @staticmethod
    def _compute_camber(uca_out: Point3D, lca_out: Point3D,
                         wheel_center: Point3D, contact_patch: Point3D) -> float:
        """
        Camber dinâmico: inclinação do plano da roda em relação à vertical,
        na vista frontal (plano Y-Z).

        DEFINIÇÃO:
            Usa o vetor CP→WC projetado em Y-Z. Para roda vertical (camber=0),
            esse vetor é (0, 0, +R). Se a manga rotaciona em torno de X,
            o vetor ganha componente em Y.

        CONVENÇÃO SAE:
            − = topo da roda inclinado PARA DENTRO do veículo
            + = topo da roda inclinado PARA FORA
        """
        wc = wheel_center.to_array()
        cp = contact_patch.to_array()

        dy = wc[1] - cp[1]
        dz = wc[2] - cp[2]

        if abs(dz) < 1e-9:
            return 0.0

        # Ângulo entre (CP→WC) e o eixo Z vertical
        angle = math.degrees(math.atan2(dy, dz))

        # Sinal: para esquerda (WC.y > 0), camber negativo = WC mais para dentro
        # que CP = dy < 0 → angle < 0 → camber = +angle (mantém negativo)
        # Wait: dy < 0 dá angle < 0; queremos camber = -|angle| (negativo)
        # → camber = angle quando lado esquerdo
        # Para direita (WC.y < 0), camber negativo = WC mais para dentro = dy > 0
        # → camber = -angle quando lado direito
        if wc[1] > 0:   # esquerdo
            return angle
        else:           # direito
            return -angle

    @staticmethod
    def _compute_caster(uca_out: Point3D, lca_out: Point3D) -> float:
        """
        Caster: inclinação do pino mestre no plano X-Z.
        Positivo = topo do pino atrás da base.
        """
        ubj = uca_out.to_array()
        lbj = lca_out.to_array()
        kp = ubj - lbj
        kp_xz = np.array([kp[0], 0.0, kp[2]])
        n = float(np.linalg.norm(kp_xz))
        if n < 1e-12:
            return 0.0
        cos_t = float(np.clip(kp_xz[2] / n, -1.0, 1.0))
        angle = math.degrees(math.acos(cos_t))
        return angle if kp[0] < 0 else -angle

    @staticmethod
    def _compute_kpi(uca_out: Point3D, lca_out: Point3D) -> float:
        """
        KPI: inclinação do pino mestre no plano Y-Z.
        Positivo = topo do pino para dentro (mais perto do plano de simetria).
        """
        ubj = uca_out.to_array()
        lbj = lca_out.to_array()
        kp = ubj - lbj
        kp_yz = np.array([0.0, kp[1], kp[2]])
        n = float(np.linalg.norm(kp_yz))
        if n < 1e-12:
            return 0.0
        cos_t = float(np.clip(kp_yz[2] / n, -1.0, 1.0))
        angle = math.degrees(math.acos(cos_t))
        return angle if abs(ubj[1]) < abs(lbj[1]) else -angle

    @staticmethod
    def _compute_toe_absolute(
        uca_out:      Point3D,
        lca_out:      Point3D,
        tr_out:       Point3D,
        wheel_center: Point3D,
    ) -> float:
        """
        Toe ABSOLUTO em graus.

        ALGORITMO:
            1. Eixo do pino mestre (LBJ → UBJ), normalizado
            2. Steering arm = (TRO - WC), projetado PERPENDICULARMENTE ao pino
            3. Toe = atan2(componente X, componente Y) deste vetor projetado

        IMPORTANTE: este valor sozinho não tem significado direto — ele
        depende de uma escolha arbitrária de orientação do tie-rod. O que
        importa é a VARIAÇÃO em relação ao estado estático (delta toe),
        que é o que o solver retorna em `state.toe_deg`.
        """
        ubj = uca_out.to_array()
        lbj = lca_out.to_array()
        tro = tr_out.to_array()
        wc  = wheel_center.to_array()

        # Eixo unitário do pino mestre
        kp = ubj - lbj
        kp_norm = float(np.linalg.norm(kp))
        if kp_norm < 1e-12:
            return 0.0
        kp_unit = kp / kp_norm

        # Steering arm: vetor do WC ao TRO, projetado perpendicularmente ao pino
        steer_arm = tro - wc
        steer_perp = steer_arm - np.dot(steer_arm, kp_unit) * kp_unit

        # Projeção no plano XY (vista superior)
        sa_xy = np.array([steer_perp[0], steer_perp[1]])
        if float(np.linalg.norm(sa_xy)) < 1e-9:
            return 0.0

        # Toe = ângulo do steering arm em relação ao eixo Y, no plano XY
        return math.degrees(math.atan2(sa_xy[0], abs(sa_xy[1]) + 1e-12))
