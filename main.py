"""
main.py
=======
Script de demonstração do motor de cálculo de suspensão FSAE.

Coordenadas fictícias, mas representativas de um carro de Fórmula SAE
convencional com suspensão double wishbone dianteira e traseira.

Convenção SAE:
  X → frente do veículo  (positivo para frente)
  Y → esquerda           (positivo para a esquerda)
  Z → para cima          (positivo para cima)

Todas as unidades em MILÍMETROS e GRAUS.
"""

from __future__ import annotations

import sys
import os

# Permite importar os módulos sem instalar o pacote
sys.path.insert(0, os.path.dirname(__file__))

from geometry.primitives import Point2D, Point3D, Vector3D
from geometry.solver_2d  import SuspensionGeometry2D, analyze_heave
from geometry.model_3d   import ControlArm, SuspensionCorner, Vehicle


# ═══════════════════════════════════════════════════════════════════════════
# 1. DEGRAU 1 — Análise 2D (Vista Frontal) · Suspensão Dianteira Esquerda
# ═══════════════════════════════════════════════════════════════════════════

def demo_2d_front_left() -> None:
    print("\n" + "═" * 60)
    print("  DEGRAU 1 — ANÁLISE 2D · VISTA FRONTAL (Dianteira Esq.)")
    print("═" * 60)

    # -------------------------------------------------------------------
    # Coordenadas YZ típicas de FSAE (bitola ~1200 mm, altura CG ~280 mm)
    # Objetivo: RC ≈ 25–80 mm, cambagem estática ≈ −1° a −2°
    #
    # Convenção: inboard mais perto do centro (Y pequeno), outboard mais
    # afastado (Y ≈ 600 mm = metade da bitola de 1200 mm).
    # Z: inboards na altura do chassi, outboards na altura da manga.
    # -------------------------------------------------------------------

    # Braço superior (UCA)
    # Inboard: Y=150, Z=295  |  Outboard: Y=590, Z=280
    # → braço quase horizontal, ligeiramente inclinado para cima para fora
    uca_inboard  = Point2D(u=150.0, v=295.0)
    uca_outboard = Point2D(u=590.0, v=280.0)

    # Braço inferior (LCA)
    # Inboard: Y=130, Z=165  |  Outboard: Y=590, Z=155
    # → braço mais longo e quase horizontal (padrão FSAE)
    lca_inboard  = Point2D(u=130.0, v=165.0)
    lca_outboard = Point2D(u=590.0, v=155.0)

    # Centro de roda e contato (raio do pneu ≈ 220 mm)
    wheel_center  = Point2D(u=600.0, v=220.0)
    contact_patch = Point2D(u=600.0, v=0.0)

    geom_2d = SuspensionGeometry2D(
        uca_inboard=uca_inboard,
        uca_outboard=uca_outboard,
        lca_inboard=lca_inboard,
        lca_outboard=lca_outboard,
        wheel_center=wheel_center,
        contact_patch=contact_patch,
    )

    print(f"\n  Comprimento UCA  : {geom_2d.uca_length:.2f} mm")
    print(f"  Comprimento LCA  : {geom_2d.lca_length:.2f} mm")
    print(f"  Comprimento Manga: {geom_2d.upright_length:.2f} mm")
    print(f"  Cambagem estática: {geom_2d.static_camber_deg():.3f}°")

    # -------------------------------------------------------------------
    # Estado estático (heave = 0)
    # -------------------------------------------------------------------
    print("\n  ── Estado Estático (heave = 0 mm) ──")
    state_0 = geom_2d.solve_heave(0.0)
    rc = state_0.roll_center
    print(f"  Cambagem        : {state_0.camber_deg:.4f}°")
    if rc:
        print(f"  Roll Center     : Y={rc.u:.2f} mm, Z={rc.v:.2f} mm")
        print(f"  RC Height       : {state_0.roll_center_height:.2f} mm")
    else:
        print("  Roll Center     : indefinido (braços paralelos)")

    # -------------------------------------------------------------------
    # Estado em bump (heave = +25 mm)
    # -------------------------------------------------------------------
    print("\n  ── Estado em Bump (heave = +25 mm) ──")
    state_bump = geom_2d.solve_heave(25.0)
    rc_b = state_bump.roll_center
    print(f"  Cambagem        : {state_bump.camber_deg:.4f}°")
    if rc_b:
        print(f"  Roll Center     : Y={rc_b.u:.2f} mm, Z={rc_b.v:.2f} mm")
        print(f"  RC Height       : {state_bump.roll_center_height:.2f} mm")

    # -------------------------------------------------------------------
    # Estado em rebound (heave = -25 mm)
    # -------------------------------------------------------------------
    print("\n  ── Estado em Rebound (heave = -25 mm) ──")
    state_rbd = geom_2d.solve_heave(-25.0)
    rc_r = state_rbd.roll_center
    print(f"  Cambagem        : {state_rbd.camber_deg:.4f}°")
    if rc_r:
        print(f"  Roll Center     : Y={rc_r.u:.2f} mm, Z={rc_r.v:.2f} mm")
        print(f"  RC Height       : {state_rbd.roll_center_height:.2f} mm")

    # -------------------------------------------------------------------
    # Análise paramétrica de heave — Camber Gain
    # -------------------------------------------------------------------
    print("\n  ── Análise de Camber Gain (±40 mm, 9 pontos) ──")
    analysis = analyze_heave(geom_2d, heave_range_mm=80.0, steps=9)

    print(f"  {'Heave (mm)':>12} {'Cambagem (°)':>14} {'RC Height (mm)':>16}")
    print("  " + "-" * 44)
    for h, c, rch in zip(
        analysis.heave_range_mm,
        analysis.camber_deg,
        analysis.roll_center_height_mm
    ):
        rc_str = f"{rch:.2f}" if rch is not None else "  n/a"
        print(f"  {h:>12.1f} {c:>14.4f} {rc_str:>16}")

    gain = analysis.camber_gain_deg_per_mm()
    print(f"\n  Camber Gain (regressão linear): {gain:.5f} °/mm")
    print(f"  ≈ {gain * 25.4:.4f} °/polegada")


# ═══════════════════════════════════════════════════════════════════════════
# 2. DEGRAU 2 — Modelo 3D · Caster, KPI, Scrub, Eixo de Rolagem
# ═══════════════════════════════════════════════════════════════════════════

def _make_front_left_corner() -> SuspensionCorner:
    """
    Constrói a ponta dianteira ESQUERDA com coordenadas 3D realistas.

    Parâmetros alvo:
      Caster  ≈ 4–5° (inboard traseiro mais alto que frontal)
      KPI     ≈ 5–7° (outboard inclinado para dentro)
      Camber  ≈ −1° (câmbagem negativa estática)
      Scrub   ≈ 20–40 mm positivo
    """
    # UCA: os dois pontos inboard definem o eixo de rotação do braço.
    # Para gerar caster: ponto inboard traseiro ligeiramente mais alto em Z
    # Para gerar KPI: outboard com Y maior que inboard efetivo
    uca = ControlArm(
        inboard_front=Point3D(x= 60.0, y=150.0, z=292.0),   # inboard frontal
        inboard_rear =Point3D(x=-70.0, y=150.0, z=298.0),   # inboard traseiro (mais alto → caster)
        outboard     =Point3D(x= -5.0, y=590.0, z=280.0),   # UBJ: X menor que LBJ → caster positivo
        name="UCA_FL",
    )

    lca = ControlArm(
        inboard_front=Point3D(x= 70.0, y=130.0, z=162.0),
        inboard_rear =Point3D(x=-80.0, y=130.0, z=168.0),   # traseiro mais alto → caster
        outboard     =Point3D(x= 15.0, y=600.0, z=152.0),   # LBJ: Y maior → KPI positivo
        name="LCA_FL",
    )

    # Centro de roda e ponto de contato (raio do pneu = 220 mm)
    wheel_center  = Point3D(x=5.0, y=610.0, z=220.0)
    contact_patch = Point3D(x=5.0, y=610.0, z=0.0)

    return SuspensionCorner(
        upper_arm=uca,
        lower_arm=lca,
        wheel_center=wheel_center,
        contact_patch=contact_patch,
        corner_id="FL",
    )


def _make_front_right_corner() -> SuspensionCorner:
    """
    Constrói a ponta dianteira DIREITA (espelho em Y).
    """
    uca = ControlArm(
        inboard_front=Point3D(x= 60.0, y=-150.0, z=292.0),
        inboard_rear =Point3D(x=-70.0, y=-150.0, z=298.0),
        outboard     =Point3D(x= -5.0, y=-590.0, z=280.0),
        name="UCA_FR",
    )

    lca = ControlArm(
        inboard_front=Point3D(x= 70.0, y=-130.0, z=162.0),
        inboard_rear =Point3D(x=-80.0, y=-130.0, z=168.0),
        outboard     =Point3D(x= 15.0, y=-600.0, z=152.0),
        name="LCA_FR",
    )

    wheel_center  = Point3D(x=5.0, y=-610.0, z=220.0)
    contact_patch = Point3D(x=5.0, y=-610.0, z=0.0)

    return SuspensionCorner(
        upper_arm=uca,
        lower_arm=lca,
        wheel_center=wheel_center,
        contact_patch=contact_patch,
        corner_id="FR",
    )


def _make_rear_left_corner() -> SuspensionCorner:
    """
    Constrói a ponta traseira ESQUERDA.
    Caster traseiro menor (suspensão não esterça).
    """
    uca = ControlArm(
        inboard_front=Point3D(x=-1430.0, y=145.0, z=290.0),
        inboard_rear =Point3D(x=-1570.0, y=145.0, z=294.0),
        outboard     =Point3D(x=-1500.0, y=585.0, z=278.0),
        name="UCA_RL",
    )

    lca = ControlArm(
        inboard_front=Point3D(x=-1430.0, y=125.0, z=160.0),
        inboard_rear =Point3D(x=-1570.0, y=125.0, z=163.0),
        outboard     =Point3D(x=-1500.0, y=585.0, z=150.0),
        name="LCA_RL",
    )

    wheel_center  = Point3D(x=-1500.0, y=600.0, z=220.0)
    contact_patch = Point3D(x=-1500.0, y=600.0, z=0.0)

    return SuspensionCorner(
        upper_arm=uca,
        lower_arm=lca,
        wheel_center=wheel_center,
        contact_patch=contact_patch,
        corner_id="RL",
    )


def _make_rear_right_corner() -> SuspensionCorner:
    """Ponta traseira DIREITA (espelho em Y da traseira esquerda)."""
    uca = ControlArm(
        inboard_front=Point3D(x=-1430.0, y=-145.0, z=290.0),
        inboard_rear =Point3D(x=-1570.0, y=-145.0, z=294.0),
        outboard     =Point3D(x=-1500.0, y=-585.0, z=278.0),
        name="UCA_RR",
    )

    lca = ControlArm(
        inboard_front=Point3D(x=-1430.0, y=-125.0, z=160.0),
        inboard_rear =Point3D(x=-1570.0, y=-125.0, z=163.0),
        outboard     =Point3D(x=-1500.0, y=-585.0, z=150.0),
        name="LCA_RR",
    )

    wheel_center  = Point3D(x=-1500.0, y=-600.0, z=220.0)
    contact_patch = Point3D(x=-1500.0, y=-600.0, z=0.0)

    return SuspensionCorner(
        upper_arm=uca,
        lower_arm=lca,
        wheel_center=wheel_center,
        contact_patch=contact_patch,
        corner_id="RR",
    )


def demo_3d_vehicle() -> None:
    print("\n" + "═" * 60)
    print("  DEGRAU 2 — MODELO 3D · PARÂMETROS ESTÁTICOS")
    print("═" * 60)

    fl = _make_front_left_corner()
    fr = _make_front_right_corner()
    rl = _make_rear_left_corner()
    rr = _make_rear_right_corner()

    # -------------------------------------------------------------------
    # Parâmetros individuais das pontas
    # -------------------------------------------------------------------
    for corner in [fl, fr, rl, rr]:
        print(f"\n{corner.summary()}")

    # -------------------------------------------------------------------
    # Veículo completo — Roll Axis
    # -------------------------------------------------------------------
    vehicle = Vehicle(
        front_left=fl,
        front_right=fr,
        rear_left=rl,
        rear_right=rr,
        wheelbase_mm=1600.0,
        track_front_mm=1200.0,
        track_rear_mm=1180.0,
    )

    rc_front, rc_rear = vehicle.roll_axis()
    print("\n" + "─" * 60)
    print("  EIXO DE ROLAGEM (Roll Axis) — Veículo Completo")
    print("─" * 60)
    print(f"  RC Dianteiro (médio)       : {rc_front:.2f} mm")
    print(f"  RC Traseiro  (médio)       : {rc_rear:.2f} mm")
    print(f"  Inclinação do Roll Axis    : {vehicle.roll_axis_inclination_deg():.5f}°")

    if rc_rear > rc_front:
        print("  → Roll Axis inclinado para CIMA na traseira (convencional FSAE)")
    elif rc_rear < rc_front:
        print("  → Roll Axis inclinado para BAIXO na traseira")
    else:
        print("  → Roll Axis paralelo ao solo")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Verificações de sanidade dos primitivos matemáticos
# ═══════════════════════════════════════════════════════════════════════════

def demo_primitives() -> None:
    print("\n" + "═" * 60)
    print("  VALIDAÇÃO DOS PRIMITIVOS MATEMÁTICOS")
    print("═" * 60)

    # Distância entre dois pontos
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(3, 4, 0)
    print(f"\n  Distância P1→P2 (esperado 5.000): {p1.distance_to(p2):.3f}")

    # Produto vetorial
    v1 = Vector3D(1, 0, 0)
    v2 = Vector3D(0, 1, 0)
    v3 = v1.cross(v2)
    print(f"  X × Y (esperado Z = [0,0,1])   : {v3}")

    # Ângulo entre vetores
    angle = v1.angle_to_deg(v2)
    print(f"  Ângulo X→Y (esperado 90.000°)  : {angle:.3f}°")

    # Normalização
    v4 = Vector3D(3, 4, 0)
    v4n = v4.normalize()
    print(f"  |[3,4,0]| normalizado           : {v4n} (mag={v4n.magnitude():.6f})")

    # Interseção de círculos
    from geometry.primitives import circle_circle_intersection
    c1 = Point2D(0, 0)
    c2 = Point2D(10, 0)
    ip = circle_circle_intersection(c1, 6.0, c2, 6.0, prefer_positive_v=True)
    print(f"  Interseção círculos (esperado ~[5, 5.196]): {ip}")


# ═══════════════════════════════════════════════════════════════════════════
# Ponto de entrada
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║     FSAE SUSPENSION GEOMETRY ENGINE — DEMO v0.1       ║")
    print("╚" + "═" * 58 + "╝")

    demo_primitives()
    demo_2d_front_left()
    demo_3d_vehicle()

    print("\n" + "═" * 60)
    print("  Demo concluída com sucesso.")
    print("═" * 60 + "\n")