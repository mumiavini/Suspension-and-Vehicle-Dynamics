"""
solver_2d.py
============
Motor de cálculo 2D para análise de suspensão na vista frontal (plano Y-Z).

Modela a suspensão como um mecanismo de quatro barras:
  - Elo fixo  : chassi (pontos de ancoragem inboard)
  - Elo 1     : braço superior (Upper Control Arm)
  - Elo 2     : braço inferior (Lower Control Arm)
  - Elo acoplador: manga de eixo (upright)

Referência de sinais:
  +Y → esquerda do veículo
  +Z → para cima
  Cambagem positiva: topo da roda inclinado para fora (veículo em estado ERETO)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np
from geometry.primitives import Point2D, circle_circle_intersection, line_intersection_2d


# ---------------------------------------------------------------------------
# Resultado do estado cinemático 2D
# ---------------------------------------------------------------------------

@dataclass
class KinematicState2D:
    """Estado cinemático da suspensão em uma posição de heave."""

    heave_mm: float            # deslocamento vertical da roda (+ = bump)

    # Posições dos pontos principais
    wheel_center: Point2D      # centro de roda no plano YZ
    upright_upper: Point2D     # ponto superior da manga de eixo
    upright_lower: Point2D     # ponto inferior da manga de eixo

    # Ângulos
    camber_deg: float          # cambagem (+ = positiva)

    # Centro de rolagem
    roll_center: Optional[Point2D]  # None se as linhas forem paralelas

    @property
    def roll_center_height(self) -> Optional[float]:
        """Altura do centro de rolagem (mm acima do solo)."""
        return self.roll_center.v if self.roll_center else None


# ---------------------------------------------------------------------------
# Geometria estática da suspensão 2D (vista frontal)
# ---------------------------------------------------------------------------

@dataclass
class SuspensionGeometry2D:
    """
    Define a geometria de UMA ponta de suspensão na vista frontal (plano YZ).

    Parâmetros (todos em mm, referenciados ao plano de simetria do veículo)
    -----------------------------------------------------------------------
    uca_inboard  : ponto de ancoragem inboard do braço superior no chassi
    uca_outboard : ponto de ancoragem outboard do braço superior na manga
    lca_inboard  : ponto de ancoragem inboard do braço inferior no chassi
    lca_outboard : ponto de ancoragem outboard do braço inferior na manga
    wheel_center : centro de roda (posição estática)
    contact_patch: ponto de contato do pneu com o solo

    Nota: Para o lado ESQUERDO, Y deve ser positivo.
          Para o lado DIREITO, use Y negativo.
    """

    uca_inboard:   Point2D
    uca_outboard:  Point2D
    lca_inboard:   Point2D
    lca_outboard:  Point2D
    wheel_center:  Point2D
    contact_patch: Point2D

    def __post_init__(self) -> None:
        # Comprimentos dos braços (rígidos — invariantes durante o movimento)
        self._uca_length: float = self.uca_inboard.distance_to(self.uca_outboard)
        self._lca_length: float = self.lca_inboard.distance_to(self.lca_outboard)

        # Distância do outboard superior ao outboard inferior (manga de eixo)
        self._upright_length: float = self.uca_outboard.distance_to(self.lca_outboard)

        # Offset do centro de roda em relação ao ponto outboard inferior
        self._wc_offset_u: float = self.wheel_center.u - self.lca_outboard.u
        self._wc_offset_v: float = self.wheel_center.v - self.lca_outboard.v

    # ------------------------------------------------------------------
    # Propriedades estáticas
    # ------------------------------------------------------------------

    @property
    def uca_length(self) -> float:
        return self._uca_length

    @property
    def lca_length(self) -> float:
        return self._lca_length

    @property
    def upright_length(self) -> float:
        return self._upright_length

    def static_camber_deg(self) -> float:
        """Cambagem estática em graus."""
        return self._camber_from_outboard_points(self.uca_outboard, self.lca_outboard)

    # ------------------------------------------------------------------
    # Solucionador cinemático
    # ------------------------------------------------------------------

    def solve_heave(self, heave_mm: float) -> KinematicState2D:
        """
        Resolve a posição da manga de eixo para um dado deslocamento de heave.

        O heave é aplicado movendo o chassi verticalmente em relação à roda
        (equivalente a mover os pontos inboard verticalmente).

        Parâmetros
        ----------
        heave_mm : deslocamento vertical em mm
                   +heave_mm → bump (roda sobe em relação ao chassi)

        Retorna
        -------
        KinematicState2D com toda a cinemática calculada.
        """
        # Deslocamento dos pontos inboard (chassi sobe quando a roda está em bump)
        uca_in = Point2D(self.uca_inboard.u, self.uca_inboard.v + heave_mm)
        lca_in = Point2D(self.lca_inboard.u, self.lca_inboard.v + heave_mm)

        # --- Passo 1: encontrar posição do outboard inferior (LCA) ---
        # Centro = lca_inboard (deslocado), raio = comprimento do LCA
        # O outboard deve manter distância fixa ao outboard superior (upright)
        # Mas primeiro resolvemos LCA outboard com apenas o constraint do LCA.
        # Depois o UCA outboard é encontrado por interseção com o upright.

        # Interseção: círculo centrado em lca_in (r=lca_length) ∩
        #             círculo centrado em uca_in  (r=uca_length + upright_length)
        # — abordagem direta: resolve sistema de quatro barras iterativamente
        # usando dois círculos: um para cada braço.

        # --- Quatro barras: resolução analítica ---
        # Ponto LCA outboard: interseção de
        #   C1 = lca_inboard_novo, r1 = lca_length
        #   C2 = uca_inboard_novo, r2 = uca_length + upright_length  (limite superior)
        # Mas isso não fecha corretamente o quadrilátero.
        # Abordagem correta: resolver UCA outboard e LCA outboard simultaneamente.

        lca_out, uca_out = self._solve_four_bar(lca_in, uca_in)

        # --- Passo 2: reconstruir o centro de roda ---
        # O centro de roda é fixo na manga de eixo.
        # Encontramos o ângulo atual da manga e aplicamos o offset.
        wc = self._reconstruct_wheel_center(lca_out, uca_out)

        # --- Passo 3: câmbagem ---
        camber = self._camber_from_outboard_points(uca_out, lca_out)

        # --- Passo 4: centro de rolagem ---
        rc = self._compute_roll_center(lca_in, lca_out, uca_in, uca_out, wc)

        return KinematicState2D(
            heave_mm=heave_mm,
            wheel_center=wc,
            upright_upper=uca_out,
            upright_lower=lca_out,
            camber_deg=camber,
            roll_center=rc,
        )

    # ------------------------------------------------------------------
    # Métodos privados de suporte
    # ------------------------------------------------------------------

    def _solve_four_bar(
        self,
        lca_in_moved: Point2D,
        uca_in_moved: Point2D,
    ) -> tuple[Point2D, Point2D]:
        """
        Resolve o mecanismo de quatro barras para encontrar as posições
        dos pontos outboard (LCA e UCA) dado o movimento dos inboards.

        Constraints:
            |lca_out - lca_in_moved| = L_lca          (braço inferior rígido)
            |uca_out - uca_in_moved| = L_uca           (braço superior rígido)
            |uca_out - lca_out|      = L_upright       (manga rígida)

        Estratégia:
          - Seed inicial: posições estáticas dos outboards.
          - Iteração: atualiza lca_out e uca_out alternadamente por
            interseção de dois círculos, escolhendo sempre a solução
            mais próxima da posição anterior (tracking de continuidade).
          - Converge em ≤ 5 iterações para heave < ±60 mm.
        """
        L_lca     = self._lca_length
        L_uca     = self._uca_length
        L_upright = self._upright_length

        # Seed: posição estática (garante solução correta para heave pequeno)
        uca_out = self.uca_outboard
        lca_out = self.lca_outboard

        for iteration in range(30):
            # --- Atualiza lca_out ---
            # lca_out deve estar a L_lca de lca_in_moved E a L_upright de uca_out
            try:
                lca_out_new = self._closest_circle_intersection(
                    c1=lca_in_moved, r1=L_lca,
                    c2=uca_out,      r2=L_upright,
                    reference=lca_out,
                )
            except ValueError as e:
                raise ValueError(
                    f"Mecanismo fora de alcance na iteração {iteration}: {e}"
                )

            # --- Atualiza uca_out ---
            try:
                uca_out_new = self._closest_circle_intersection(
                    c1=uca_in_moved,  r1=L_uca,
                    c2=lca_out_new,   r2=L_upright,
                    reference=uca_out,
                )
            except ValueError as e:
                raise ValueError(
                    f"Mecanismo fora de alcance na iteração {iteration}: {e}"
                )

            # Verifica convergência
            delta_lca = lca_out_new.distance_to(lca_out)
            delta_uca = uca_out_new.distance_to(uca_out)

            lca_out = lca_out_new
            uca_out = uca_out_new

            if delta_lca < 1e-8 and delta_uca < 1e-8:
                break

        return lca_out, uca_out

    @staticmethod
    def _closest_circle_intersection(
        c1: Point2D, r1: float,
        c2: Point2D, r2: float,
        reference: Point2D,
    ) -> Point2D:
        """
        Retorna o ponto de interseção dos dois círculos que está mais próximo
        do ponto de referência.

        Isso implementa tracking de continuidade: ao longo de uma sequência
        de posições de heave, o mecanismo segue a solução fisicamente correta.
        """
        p1 = c1.to_array()
        p2 = c2.to_array()
        ref = reference.to_array()

        d = float(np.linalg.norm(p2 - p1))

        if d < 1e-12:
            raise ValueError("Centros coincidentes.")
        if d > r1 + r2 + 1e-6:
            raise ValueError(f"Círculos não se intersectam (d={d:.2f} > r1+r2={r1+r2:.2f}).")
        if d < abs(r1 - r2) - 1e-6:
            raise ValueError("Um círculo contém o outro.")

        a = (r1**2 - r2**2 + d**2) / (2.0 * d)
        h_sq = r1**2 - a**2
        h = math.sqrt(max(h_sq, 0.0))

        mid = p1 + a * (p2 - p1) / d
        perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / d

        sol_a = mid + h * perp
        sol_b = mid - h * perp

        # Escolhe a solução mais próxima da referência (continuidade)
        dist_a = float(np.linalg.norm(sol_a - ref))
        dist_b = float(np.linalg.norm(sol_b - ref))

        if dist_a <= dist_b:
            return Point2D.from_array(sol_a)
        else:
            return Point2D.from_array(sol_b)

    def _reconstruct_wheel_center(
        self,
        lca_out: Point2D,
        uca_out: Point2D,
    ) -> Point2D:
        """
        Reconstrói o centro de roda com base na orientação atual da manga de eixo.
        O offset do WC em relação ao LCA outboard é mantido no referencial da manga.
        """
        # Vetor ao longo da manga de eixo (de baixo para cima)
        dx = uca_out.u - lca_out.u
        dz = uca_out.v - lca_out.v
        upright_len = math.hypot(dx, dz)

        if upright_len < 1e-12:
            raise ValueError("Comprimento da manga de eixo é nulo.")

        # Versores da manga (axial e perpendicular)
        ex = dx / upright_len   # ao longo da manga
        ez = dz / upright_len
        nx = -ez                # normal (perpendicular à manga)
        nz = ex

        # Offset original em coordenadas da manga (estática)
        dx0 = self.uca_outboard.u - self.lca_outboard.u
        dz0 = self.uca_outboard.v - self.lca_outboard.v
        up0 = self._upright_length

        ex0 = dx0 / up0
        ez0 = dz0 / up0
        nx0 = -ez0
        nz0 = ex0

        # Offset do WC no referencial estático
        dwc_u = self._wc_offset_u
        dwc_v = self._wc_offset_v

        # Projeção no referencial local estático
        s_axial = dwc_u * ex0 + dwc_v * ez0
        s_perp  = dwc_u * nx0 + dwc_v * nz0

        # Reconstrução no referencial atual
        wc_u = lca_out.u + s_axial * ex + s_perp * nx
        wc_v = lca_out.v + s_axial * ez + s_perp * nz

        return Point2D(wc_u, wc_v)

    @staticmethod
    def _camber_from_outboard_points(
        uca_out: Point2D,
        lca_out: Point2D,
    ) -> float:
        """
        Calcula a cambagem como o ângulo do eixo da manga de eixo em relação
        à vertical (eixo Z), no plano YZ.

        Convenção SAE:
          Negativo = topo da roda inclinado para dentro (câmbagem negativa).
          Positivo = topo da roda inclinado para fora.

        Para o lado esquerdo (Y > 0), se uca_out.u < lca_out.u o topo
        está mais para dentro → câmbagem negativa.
        """
        dy = uca_out.u - lca_out.u  # componente lateral (Y)
        dz = uca_out.v - lca_out.v  # componente vertical (Z)

        # atan2(dy, dz): ângulo da manga em relação ao eixo Z
        # Ângulo positivo → manga inclinada para +Y (câmbagem positiva, topo para fora)
        camber_rad = math.atan2(dy, dz)
        # Inverter sinal para convenção SAE: negativo = topo para dentro
        return -math.degrees(camber_rad)

    @staticmethod
    def _compute_roll_center(
        lca_in: Point2D,
        lca_out: Point2D,
        uca_in: Point2D,
        uca_out: Point2D,
        wheel_center: Point2D,
    ) -> Optional[Point2D]:
        """
        Calcula o Centro de Rolagem pelo método do Centro Instantâneo (IC).

        Algoritmo (Reuleaux / suspensão double wishbone):
          1. Encontra o Centro Instantâneo (IC) da manga de eixo:
             interseção das prolongações das retas do UCA e do LCA.
          2. Liga o IC ao centro de contato do pneu com o solo
             (contact patch, u = wheel_center.u, v = 0).
          3. O Roll Center é a interseção dessa reta com o plano de
             simetria do veículo (u = 0, ou seja, com o eixo Z central).

        Referência: Milliken & Milliken, "Race Car Vehicle Dynamics", Cap. 17.
        """
        contact_patch = Point2D(wheel_center.u, 0.0)

        try:
            # Passo 1: Centro Instantâneo = interseção das linhas dos braços
            instant_center = line_intersection_2d(lca_in, lca_out, uca_in, uca_out)
        except ValueError:
            # Braços paralelos → IC no infinito → RC no solo (altura = 0)
            # Para braços paralelos horizontais, RC está na intersecção
            # das duas retas verticais com o plano de simetria → Z do solo
            return Point2D(0.0, 0.0)

        try:
            # Passo 2: reta IC → contact_patch
            # Passo 3: interseção com u = 0 (plano de simetria)
            roll_center = _line_at_u(instant_center, contact_patch, 0.0)
            return roll_center
        except ZeroDivisionError:
            # Linha IC→contact é vertical (IC e contact têm mesmo u)
            # RC é o ponto do eixo vertical em Z = contact_patch.v
            return Point2D(0.0, contact_patch.v)


# ---------------------------------------------------------------------------
# Análise de variação de cambagem (Camber Gain)
# ---------------------------------------------------------------------------

@dataclass
class CamberAnalysis:
    """Resultado da análise de ganho de cambagem em heave."""

    heave_range_mm: list[float]
    camber_deg:     list[float]
    roll_center_height_mm: list[Optional[float]]

    def camber_gain_deg_per_mm(self) -> float:
        """
        Taxa de variação de cambagem (graus por mm de heave).
        Calculada por regressão linear simples sobre o range fornecido.
        """
        import numpy as np
        h = [x for x in self.heave_range_mm]
        c = self.camber_deg
        if len(h) < 2:
            return 0.0
        coeffs = np.polyfit(h, c, 1)
        return float(coeffs[0])


def analyze_heave(
    geometry: SuspensionGeometry2D,
    heave_range_mm: float = 50.0,
    steps: int = 21,
) -> CamberAnalysis:
    """
    Executa análise paramétrica de heave simétrico (bump e rebound).

    Parâmetros
    ----------
    geometry      : geometria estática da suspensão
    heave_range_mm: amplitude total (±heave_range_mm/2)
    steps         : número de pontos de simulação (ímpar recomendado)

    Retorna
    -------
    CamberAnalysis com os arrays de resultados.
    """
    import numpy as np

    half = heave_range_mm / 2.0
    heave_values = list(np.linspace(-half, half, steps))

    camber_list: list[float] = []
    rc_height_list: list[Optional[float]] = []

    for h in heave_values:
        state = geometry.solve_heave(h)
        camber_list.append(state.camber_deg)
        rc_height_list.append(state.roll_center_height)

    return CamberAnalysis(
        heave_range_mm=heave_values,
        camber_deg=camber_list,
        roll_center_height_mm=rc_height_list,
    )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _line_at_u(p1: Point2D, p2: Point2D, u_target: float) -> Point2D:
    """
    Encontra o ponto na reta definida por p1→p2 com coordenada u = u_target.
    Levanta ZeroDivisionError se a linha for vertical (du ≈ 0).
    """
    du = p2.u - p1.u
    if abs(du) < 1e-12:
        raise ZeroDivisionError("Linha vertical — não intercepta u_target de forma única.")
    t = (u_target - p1.u) / du
    v_target = p1.v + t * (p2.v - p1.v)
    return Point2D(u_target, v_target)


# Para manter compatibilidade com importações opcionais
from typing import Optional