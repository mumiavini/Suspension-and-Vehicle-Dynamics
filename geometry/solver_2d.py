"""
geometry/solver_2d.py
=====================
Solver cinemático 2D — análise da suspensão na VISTA FRONTAL (plano Y-Z).

CONCEITO FÍSICO
---------------
Vista de frente, a suspensão é um MECANISMO DE QUATRO BARRAS:

       UCA_in ●──────────● UCA_out          (braço superior)
                          │
                          │ manga (upright)
                          │
       LCA_in ●──────────● LCA_out          (braço inferior)

    Elo fixo  : chassi (segmento UCA_in → LCA_in)
    Elo 1     : braço superior (UCA_in → UCA_out)
    Elo 2     : braço inferior (LCA_in → LCA_out)
    Acoplador : manga de eixo (UCA_out → LCA_out)

Em movimento de HEAVE (deslocamento vertical relativo entre chassi e roda),
os pontos inboard sobem/descem rigidamente. Os outboards (na manga) devem
satisfazer simultaneamente:
    |UCA_out − UCA_in| = L_UCA           (UCA rígido)
    |LCA_out − LCA_in| = L_LCA           (LCA rígido)
    |UCA_out − LCA_out| = L_upright      (manga rígida)

Resolvemos isso por INTERSEÇÃO DE DOIS CÍRCULOS (uma vez para UCA_out,
uma para LCA_out), iterando até as três restrições serem satisfeitas.

SAÍDAS PRINCIPAIS
-----------------
    - Posição da manga em uma dada configuração
    - Cambagem (camber)
    - Centro de Rolagem (Roll Center)
    - Camber Gain (°/mm de heave)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from geometry.primitives import (
    Point2D,
    circle_circle_intersection,
    line_intersection_2d,
)


# =============================================================================
# Estado cinemático (resultado de uma resolução)
# =============================================================================

@dataclass
class KinematicState2D:
    """
    Resultado da resolução do mecanismo de 4 barras para um dado heave.

    Atributos:
        heave_mm       : deslocamento vertical aplicado (mm); + = bump
        wheel_center   : posição do centro de roda no plano Y-Z (mm)
        upright_upper  : posição do outboard do UCA (mm)
        upright_lower  : posição do outboard do LCA (mm)
        camber_deg     : ângulo de cambagem (graus); − = topo p/ dentro
        roll_center    : posição do Roll Center (mm). None se indeterminado.
    """
    heave_mm:      float
    wheel_center:  Point2D
    upright_upper: Point2D
    upright_lower: Point2D
    camber_deg:    float
    roll_center:   Optional[Point2D]

    @property
    def roll_center_height(self) -> Optional[float]:
        """Altura do Roll Center (Z, em mm). None se indeterminado."""
        return self.roll_center.v if self.roll_center else None


# =============================================================================
# Geometria 2D da suspensão (uma ponta, na vista frontal)
# =============================================================================

@dataclass
class SuspensionGeometry2D:
    """
    Define a geometria 2D (vista frontal) de UMA ponta de suspensão.

    Convenção: para o LADO ESQUERDO use Y > 0; para o DIREITO, Y < 0.
    Como esta classe usa (u, v) genéricos: u = Y, v = Z.

    Atributos:
        uca_inboard   : ancoragem inboard do braço superior (no chassi)
        uca_outboard  : ancoragem outboard do braço superior (na manga)
        lca_inboard   : ancoragem inboard do braço inferior (no chassi)
        lca_outboard  : ancoragem outboard do braço inferior (na manga)
        wheel_center  : centro da roda (posição estática)
        contact_patch : ponto de contato do pneu com o solo
    """
    uca_inboard:   Point2D
    uca_outboard:  Point2D
    lca_inboard:   Point2D
    lca_outboard:  Point2D
    wheel_center:  Point2D
    contact_patch: Point2D

    def __post_init__(self) -> None:
        """
        Pré-calcula comprimentos invariantes (rígidos) usados no solver.
        Esses valores não mudam durante o movimento da suspensão.
        """
        # Comprimentos dos braços (medidos no estado estático e fixos)
        self._L_uca:     float = self.uca_inboard.distance_to(self.uca_outboard)
        self._L_lca:     float = self.lca_inboard.distance_to(self.lca_outboard)
        self._L_upright: float = self.uca_outboard.distance_to(self.lca_outboard)

        # Offset (delta) do centro de roda em relação ao LCA outboard, no
        # referencial GLOBAL ESTÁTICO. Usaremos isso para reconstruir o WC
        # após a manga se mover — assumindo manga rígida.
        self._wc_offset_u: float = self.wheel_center.u - self.lca_outboard.u
        self._wc_offset_v: float = self.wheel_center.v - self.lca_outboard.v

    # -------------------------------------------------------------------------
    # Propriedades convenientes
    # -------------------------------------------------------------------------

    @property
    def uca_length(self)     -> float: return self._L_uca
    @property
    def lca_length(self)     -> float: return self._L_lca
    @property
    def upright_length(self) -> float: return self._L_upright

    def static_camber_deg(self) -> float:
        """Cambagem estática (com a suspensão na posição de referência)."""
        return self._compute_camber(self.uca_outboard, self.lca_outboard)

    # =========================================================================
    # MÉTODO PRINCIPAL: resolver a suspensão para um dado heave
    # =========================================================================

    def solve_heave(self, heave_mm: float) -> KinematicState2D:
        """
        Resolve a posição da manga para um deslocamento de heave.

        MODELO DE HEAVE:
            Quando a roda sobe (bump) relativo ao chassi, equivale a mover
            o chassi PARA CIMA enquanto a roda fica fixa. Por isso somamos
            +heave_mm às coordenadas Z dos pontos inboard.

        Parâmetros:
            heave_mm : deslocamento vertical (mm). + = bump, − = rebound.

        Retorna:
            KinematicState2D com posições e ângulos resolvidos.
        """
        # ─── 1. Movimentar os pontos inboard (chassi) ─────────────────────────
        # Em heave puro, só Z muda (subida/descida vertical)
        uca_in_moved = Point2D(self.uca_inboard.u, self.uca_inboard.v + heave_mm)
        lca_in_moved = Point2D(self.lca_inboard.u, self.lca_inboard.v + heave_mm)

        # ─── 2. Resolver o mecanismo de 4 barras ──────────────────────────────
        # Acha as novas posições dos outboards (na manga)
        lca_out, uca_out = self._solve_four_bar(lca_in_moved, uca_in_moved)

        # ─── 3. Reconstruir o centro de roda a partir da manga ────────────────
        # O WC é solidário à manga; quando a manga gira, o WC acompanha
        wc_new = self._reconstruct_wheel_center(lca_out, uca_out)

        # ─── 4. Calcular ângulos derivados ────────────────────────────────────
        camber = self._compute_camber(uca_out, lca_out)
        rc     = self._compute_roll_center(lca_in_moved, lca_out,
                                            uca_in_moved, uca_out,
                                            wc_new)

        return KinematicState2D(
            heave_mm=heave_mm,
            wheel_center=wc_new,
            upright_upper=uca_out,
            upright_lower=lca_out,
            camber_deg=camber,
            roll_center=rc,
        )

    # =========================================================================
    # PASSO 2: Solver do mecanismo de 4 barras (iterativo)
    # =========================================================================

    def _solve_four_bar(
        self,
        lca_in_moved: Point2D,
        uca_in_moved: Point2D,
    ) -> tuple[Point2D, Point2D]:
        """
        Encontra (LCA_out, UCA_out) que satisfazem os três comprimentos rígidos.

        ALGORITMO (iteração de ponto fixo):
            1. Inicia com lca_out e uca_out na posição estática (seed)
            2. Atualiza lca_out = interseção de
                   círculo(lca_in, r=L_lca) ∩ círculo(uca_out, r=L_upright)
            3. Atualiza uca_out = interseção de
                   círculo(uca_in, r=L_uca) ∩ círculo(lca_out, r=L_upright)
            4. Repete até convergência (delta < tolerância)

        IMPORTANTE — escolha da solução de interseção:
            Cada interseção de círculos tem 2 soluções. Para garantir
            CONTINUIDADE FÍSICA, escolhemos sempre a que está mais próxima
            da posição anterior (tracking).
        """
        L_lca     = self._L_lca
        L_uca     = self._L_uca
        L_upright = self._L_upright

        # SEED: posição estática conhecida (esta É a solução para heave=0,
        # e para heave pequeno é uma boa estimativa inicial)
        uca_out = self.uca_outboard
        lca_out = self.lca_outboard

        # Iteração de ponto fixo
        for _ in range(30):
            # Atualiza LCA_out (mantém distâncias para lca_in_moved e uca_out)
            lca_out_new = self._closest_intersection(
                c1=lca_in_moved, r1=L_lca,
                c2=uca_out,      r2=L_upright,
                reference=lca_out,
            )

            # Atualiza UCA_out (mantém distâncias para uca_in_moved e lca_out_new)
            uca_out_new = self._closest_intersection(
                c1=uca_in_moved, r1=L_uca,
                c2=lca_out_new,  r2=L_upright,
                reference=uca_out,
            )

            # Critério de convergência
            d_lca = lca_out_new.distance_to(lca_out)
            d_uca = uca_out_new.distance_to(uca_out)

            lca_out, uca_out = lca_out_new, uca_out_new

            if d_lca < 1e-8 and d_uca < 1e-8:
                break

        return lca_out, uca_out

    @staticmethod
    def _closest_intersection(
        c1: Point2D, r1: float,
        c2: Point2D, r2: float,
        reference: Point2D,
    ) -> Point2D:
        """
        Interseção de dois círculos, retornando a solução mais próxima de
        `reference`. Usado para tracking de continuidade do mecanismo.
        """
        p1 = c1.to_array()
        p2 = c2.to_array()
        ref = reference.to_array()

        d = float(np.linalg.norm(p2 - p1))
        if d < 1e-12:
            raise ValueError("Centros coincidentes.")
        if d > r1 + r2 + 1e-6:
            raise ValueError(
                f"Círculos não se intersectam: d={d:.2f}, r1+r2={r1+r2:.2f}"
            )
        if d < abs(r1 - r2) - 1e-6:
            raise ValueError("Um círculo contém o outro.")

        # Geometria padrão de interseção de círculos
        a   = (r1**2 - r2**2 + d**2) / (2.0 * d)
        h   = math.sqrt(max(r1**2 - a**2, 0.0))
        mid = p1 + a * (p2 - p1) / d
        perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / d

        sol_a = mid + h * perp
        sol_b = mid - h * perp

        # Escolhe a solução mais próxima da referência (continuidade)
        if np.linalg.norm(sol_a - ref) <= np.linalg.norm(sol_b - ref):
            return Point2D.from_array(sol_a)
        else:
            return Point2D.from_array(sol_b)

    # =========================================================================
    # PASSO 3: Reconstrução do centro de roda
    # =========================================================================

    def _reconstruct_wheel_center(
        self,
        lca_out: Point2D,
        uca_out: Point2D,
    ) -> Point2D:
        """
        Reconstrói o centro de roda na nova posição da manga.

        A manga é um corpo rígido: o vetor (LCA_out → WC) tem comprimento e
        orientação RELATIVA À MANGA constantes. Mas a manga rotacionou, então
        precisamos rotacionar esse vetor pelo mesmo ângulo que a manga girou.

        ALGORITMO:
            1. Computa o offset estático do WC em relação ao LCA_out, no
               referencial LOCAL da manga (eixos axial/perpendicular).
            2. Reconstrói o WC aplicando esse offset no referencial atual
               (que rotaciona junto com a manga).
        """
        # --- Referencial LOCAL ATUAL da manga (eixos axial e perpendicular) ---
        dx_now = uca_out.u - lca_out.u
        dz_now = uca_out.v - lca_out.v
        L_now  = math.hypot(dx_now, dz_now)
        if L_now < 1e-12:
            raise ValueError("Manga de comprimento zero.")
        e_axial_now = np.array([dx_now / L_now, dz_now / L_now])  # ao longo da manga
        e_perp_now  = np.array([-e_axial_now[1], e_axial_now[0]])  # 90° anti-horário

        # --- Referencial LOCAL ESTÁTICO da manga (para descobrir o offset) ---
        dx_0 = self.uca_outboard.u - self.lca_outboard.u
        dz_0 = self.uca_outboard.v - self.lca_outboard.v
        L_0  = self._L_upright
        e_axial_0 = np.array([dx_0 / L_0, dz_0 / L_0])
        e_perp_0  = np.array([-e_axial_0[1], e_axial_0[0]])

        # Decomposição do offset estático nos eixos locais
        offset = np.array([self._wc_offset_u, self._wc_offset_v])
        s_axial = float(np.dot(offset, e_axial_0))  # componente axial
        s_perp  = float(np.dot(offset, e_perp_0))   # componente perpendicular

        # Recomposição usando os eixos locais ATUAIS
        wc_new_arr = (
            lca_out.to_array()
            + s_axial * e_axial_now
            + s_perp  * e_perp_now
        )
        return Point2D.from_array(wc_new_arr)

    # =========================================================================
    # PASSO 4a: Cálculo da Cambagem
    # =========================================================================

    @staticmethod
    def _compute_camber(uca_out: Point2D, lca_out: Point2D) -> float:
        """
        Cambagem: ângulo da MANGA em relação ao eixo Z (vertical), no plano Y-Z.

        CONVENÇÃO SAE:
            - Camber NEGATIVO: topo da roda inclinado PARA DENTRO do veículo
            - Camber POSITIVO: topo da roda inclinado PARA FORA

        Para o LADO ESQUERDO (Y > 0):
            Se UCA_out.u < LCA_out.u → topo para dentro → camber NEGATIVO
            (porque "dentro" = Y menor para o lado esquerdo)
        """
        dy = uca_out.u - lca_out.u   # componente lateral
        dz = uca_out.v - lca_out.v   # componente vertical

        # atan2(dy, dz) é o ângulo entre o eixo da manga e o eixo Z
        # Com sinal invertido para seguir a convenção SAE (− = para dentro)
        return -math.degrees(math.atan2(dy, dz))

    # =========================================================================
    # PASSO 4b: Cálculo do Roll Center (Centro de Rolagem)
    # =========================================================================

    @staticmethod
    def _compute_roll_center(
        lca_in:  Point2D,
        lca_out: Point2D,
        uca_in:  Point2D,
        uca_out: Point2D,
        wheel_center: Point2D,
    ) -> Optional[Point2D]:
        """
        Calcula o Roll Center pelo método do CENTRO INSTANTÂNEO (IC).

        REFERÊNCIA: Milliken & Milliken, "Race Car Vehicle Dynamics", Cap. 17.

        ALGORITMO:
            1. IC = interseção das prolongações das retas dos braços
               (linha LCA_in→LCA_out × linha UCA_in→UCA_out)
            2. Liga IC ao centro de contato do pneu (CP)
            3. RC = interseção dessa reta com o plano de simetria (u = 0)

        Casos degenerados:
            - Braços paralelos: IC vai para o infinito; RC fica no solo (Z=0)
            - Linha IC→CP vertical: RC é o próprio CP refletido no eixo
        """
        # CP: ponto de contato no plano de simetria (u=Y do WC, v=0)
        contact_patch = Point2D(wheel_center.u, 0.0)

        # PASSO 1: Centro Instantâneo das prolongações dos braços
        try:
            ic = line_intersection_2d(lca_in, lca_out, uca_in, uca_out)
        except ValueError:
            # Braços paralelos: IC no infinito → RC no nível do solo
            return Point2D(0.0, 0.0)

        # PASSO 2-3: linha IC→CP, intersectada com u=0 (plano de simetria)
        # Parametrização: P(t) = IC + t · (CP - IC)
        # Queremos t tal que P.u = 0 → t = -IC.u / (CP.u - IC.u)
        delta_u = contact_patch.u - ic.u
        if abs(delta_u) < 1e-12:
            # Linha vertical → RC à altura do IC, projetado em u=0
            return Point2D(0.0, ic.v)

        t = -ic.u / delta_u
        v_rc = ic.v + t * (contact_patch.v - ic.v)
        return Point2D(0.0, v_rc)


# =============================================================================
# Função utilitária: análise de camber gain por varredura de heave
# =============================================================================

@dataclass
class CamberAnalysis:
    """
    Resultado de uma varredura paramétrica de heave.

    Atributos:
        heave_range_mm        : lista dos valores de heave testados
        camber_deg            : cambagem em cada ponto
        roll_center_height_mm : altura do Roll Center em cada ponto
    """
    heave_range_mm:        list[float]
    camber_deg:            list[float]
    roll_center_height_mm: list[Optional[float]]

    def camber_gain_deg_per_mm(self) -> float:
        """
        Taxa de variação da cambagem com o heave (°/mm), via regressão linear.
        """
        if len(self.heave_range_mm) < 2:
            return 0.0
        # polyfit grau 1: coef. angular = camber gain
        coef = np.polyfit(self.heave_range_mm, self.camber_deg, 1)
        return float(coef[0])


def analyze_heave(
    geometry: SuspensionGeometry2D,
    heave_range_mm: float = 50.0,
    steps: int = 21,
) -> CamberAnalysis:
    """
    Executa varredura simétrica de heave (de −range/2 a +range/2).

    Parâmetros:
        geometry       : geometria 2D da suspensão
        heave_range_mm : amplitude total da varredura (mm)
        steps          : número de pontos amostrados (preferir ímpar p/ incluir 0)

    Retorna:
        CamberAnalysis com os arrays de resultado.
    """
    half = heave_range_mm / 2.0
    heave_values = list(np.linspace(-half, half, steps))

    cambers: list[float] = []
    rc_heights: list[Optional[float]] = []

    for h in heave_values:
        state = geometry.solve_heave(h)
        cambers.append(state.camber_deg)
        rc_heights.append(state.roll_center_height)

    return CamberAnalysis(
        heave_range_mm=heave_values,
        camber_deg=cambers,
        roll_center_height_mm=rc_heights,
    )
