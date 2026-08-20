"""
sla_geometry.py — Solver parametrico de geometria SLA em vista frontal (FSAE).

Sistema de coordenadas (vista frontal, lado direito do carro):
    y : positivo para FORA (linha de centro do carro em y = 0)
    z : positivo para CIMA (solo em z = 0)
Todas as dimensoes em mm, angulos em graus, salvo indicacao.

Convencao de camber: negativo = topo da roda inclinado para DENTRO (SAE).

Fluxo:
    1. CornerSpec  -> define os alvos (bitola, RC, FVSA, KPI, pivo inferior...)
    2. solve_pickups() -> resolve os 4 pontos internos a partir dos alvos
    3. CornerKinematics -> quadrilatero articulado, varredura de bump e rolagem
    4. report() -> tabela de pontos + verificacoes de sanidade

Autor: gerado para PUCPR Racing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


# ----------------------------------------------------------------------------
# Utilitarios geometricos
# ----------------------------------------------------------------------------

def line_intersection(p1, p2, p3, p4):
    """Intersecao das retas (p1,p2) e (p3,p4). Retorna None se paralelas."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py])


def circle_intersection(c1, r1, c2, r2, reference):
    """Intersecao de dois circulos. Retorna o ramo mais proximo de `reference`."""
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)
    if d > r1 + r2 or d < abs(r1 - r2) or d < 1e-12:
        raise ValueError(
            f"Quadrilatero nao fecha (d={d:.2f}, r1={r1:.2f}, r2={r2:.2f}). "
            "Curso pedido excede o limite cinematico."
        )
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h_sq = r1 ** 2 - a ** 2
    h = np.sqrt(max(h_sq, 0.0))
    base = c1 + a * d_vec / d
    perp = np.array([-d_vec[1], d_vec[0]]) / d
    s1 = base + h * perp
    s2 = base - h * perp
    ref = np.asarray(reference, float)
    return s1 if np.linalg.norm(s1 - ref) < np.linalg.norm(s2 - ref) else s2


def rot2(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------------
# Especificacao do canto
# ----------------------------------------------------------------------------

@dataclass
class CornerSpec:
    """Alvos de projeto de um canto (dianteiro ou traseiro)."""

    name: str = "Dianteira"

    # --- alvos primarios ---
    track: float = 1200.0            # bitola [mm]
    rc_height: float = 35.0          # altura do centro de rolagem em projeto [mm]
    fvsa_length: float = 1500.0      # front view swing arm: contato -> FVIC [mm]

    # --- pacote roda / upright ---
    loaded_radius: float = 240.0     # raio carregado do pneu [mm]
    rim_diameter_in: float = 13.0    # diametro do aro [pol]
    lbj_y: float = 565.0             # pivo inferior externo, y [mm]
    lbj_z: float = 125.0             # pivo inferior externo, z [mm]
    kpi: float = 10.0                # inclinacao do pino mestre [deg]
    spindle_length: float = 235.0    # distancia LBJ -> UBJ [mm]

    # --- chassi ---
    rail_y: float = 175.0            # semi-largura da longarina (fixacoes internas) [mm]

    # --- setup ---
    static_camber: float = -1.5      # camber estatico [deg]
    travel_bump: float = 25.0        # curso em compressao [mm]
    travel_droop: float = 25.0       # curso em extensao [mm]

    # --- bandas de verificacao (a traseira nao esterca: KPI pode ser menor) ---
    kpi_band: tuple = (6.0, 14.0)
    roll_ref: float = 1.5            # rolagem de referencia p/ camber vs pista [deg]

    # ------------------------------------------------------------------
    @property
    def half_track(self) -> float:
        return self.track / 2.0

    @property
    def contact_patch(self) -> np.ndarray:
        return np.array([self.half_track, 0.0])

    @property
    def wheel_center(self) -> np.ndarray:
        return np.array([self.half_track, self.loaded_radius])

    @property
    def lbj(self) -> np.ndarray:
        return np.array([self.lbj_y, self.lbj_z])

    @property
    def ubj(self) -> np.ndarray:
        """Pivo superior externo: sobre o eixo KPI, acima do LBJ."""
        return self.lbj + self.spindle_length * np.array(
            [-np.sin(self.kpi * D2R), np.cos(self.kpi * D2R)]
        )

    @property
    def fvic(self) -> np.ndarray:
        """Centro instantaneo em vista frontal, derivado de RC e FVSA."""
        y = self.half_track - self.fvsa_length
        z = 2.0 * self.rc_height * self.fvsa_length / self.track
        return np.array([y, z])

    @property
    def scrub_radius(self) -> float:
        """Positivo = eixo KPI encontra o solo por DENTRO do contato do pneu."""
        y_ground = self.lbj_y + self.lbj_z * np.tan(self.kpi * D2R)
        return self.half_track - y_ground

    @property
    def camber_gain_rate(self) -> float:
        """Taxa teorica de ganho de camber [deg/mm] = 57.3 / FVSA."""
        return R2D / self.fvsa_length

    @property
    def rc_height_for_flat_lca(self) -> float:
        """Altura de RC que deixa o LCA exatamente horizontal."""
        return self.lbj_z * self.half_track / self.fvsa_length


def solve_pickups(spec: CornerSpec) -> dict:
    """Resolve os pontos internos: intersecao das retas dos bracos com a longarina."""
    fvic = spec.fvic
    lbj, ubj = spec.lbj, spec.ubj

    def point_on_line_at_y(p_out, p_ic, y_target):
        t = (y_target - p_out[0]) / (p_ic[0] - p_out[0])
        return p_out + t * (p_ic - p_out)

    lca_in = point_on_line_at_y(lbj, fvic, spec.rail_y)
    uca_in = point_on_line_at_y(ubj, fvic, spec.rail_y)

    return {
        "lbj": lbj,
        "ubj": ubj,
        "lca_in": lca_in,
        "uca_in": uca_in,
        "fvic": fvic,
        "lca_length": float(np.linalg.norm(lbj - lca_in)),
        "uca_length": float(np.linalg.norm(ubj - uca_in)),
        "lca_angle_deg": float(np.degrees(np.arctan2(lbj[1] - lca_in[1], lbj[0] - lca_in[0]))),
        "uca_angle_deg": float(np.degrees(np.arctan2(ubj[1] - uca_in[1], ubj[0] - uca_in[0]))),
        "sep_outboard": float(ubj[1] - lbj[1]),
        "sep_inboard": float(uca_in[1] - lca_in[1]),
    }


# ----------------------------------------------------------------------------
# Cinematica do quadrilatero
# ----------------------------------------------------------------------------

class CornerKinematics:
    """Quadrilatero articulado em vista frontal: chassi - LCA - upright - UCA."""

    def __init__(self, spec: CornerSpec):
        self.spec = spec
        g = solve_pickups(spec)
        self.g = g

        self.lca_in = g["lca_in"]
        self.uca_in = g["uca_in"]
        self.L_lca = g["lca_length"]
        self.L_uca = g["uca_length"]
        self.L_spindle = spec.spindle_length

        self.lbj0 = g["lbj"]
        self.ubj0 = g["ubj"]
        self.theta0 = np.arctan2(
            self.lbj0[1] - self.lca_in[1], self.lbj0[0] - self.lca_in[0]
        )
        self.psi0 = np.arctan2(
            self.ubj0[0] - self.lbj0[0], self.ubj0[1] - self.lbj0[1]
        )
        # vetores rigidos do upright, referenciados ao LBJ
        self.r_cp = spec.contact_patch - self.lbj0
        self.r_wc = spec.wheel_center - self.lbj0

    # ------------------------------------------------------------------
    def solve_theta(self, theta: float) -> dict:
        """Estado do canto para um dado angulo do LCA [rad]."""
        lbj = self.lca_in + self.L_lca * np.array([np.cos(theta), np.sin(theta)])
        ubj = circle_intersection(
            self.uca_in, self.L_uca, lbj, self.L_spindle, reference=self.ubj0
        )

        psi = np.arctan2(ubj[0] - lbj[0], ubj[1] - lbj[1])
        d_psi = psi - self.psi0                      # rotacao do upright
        R = rot2(d_psi)

        cp = lbj + R @ self.r_cp
        wc = lbj + R @ self.r_wc

        ic = line_intersection(self.lca_in, lbj, self.uca_in, ubj)
        if ic is None:                                # bracos paralelos
            ic = np.array([np.sign(self.spec.fvic[0]) * 1e7, lbj[1]])

        rc = line_intersection(cp, ic, np.array([0.0, 0.0]), np.array([0.0, 1.0]))

        return {
            "theta": theta,
            "lbj": lbj,
            "ubj": ubj,
            "cp": cp,
            "wc": wc,
            "ic": ic,
            "rc_height": float(rc[1]) if rc is not None else np.nan,
            "camber": float(self.spec.static_camber + np.degrees(d_psi)),
            "bump": float(cp[1] - self.spec.contact_patch[1]),
            "half_track": float(cp[0]),
            "fvsa": float(cp[0] - ic[0]),
        }

    def solve_bump(self, bump: float) -> dict:
        """Estado do canto para um dado curso vertical do contato [mm]."""
        span = 0.75  # rad
        f = lambda th: self.solve_theta(th)["bump"] - bump
        lo, hi = self.theta0 - span, self.theta0 + span
        # encolhe o intervalo ate o quadrilatero fechar nos dois extremos
        for _ in range(60):
            try:
                f(lo); f(hi)
                break
            except ValueError:
                span *= 0.9
                lo, hi = self.theta0 - span, self.theta0 + span
        theta = brentq(f, lo, hi, xtol=1e-12)
        return self.solve_theta(theta)

    # ------------------------------------------------------------------
    def bump_sweep(self, n: int = 41) -> dict:
        """Varredura de curso (referencial chassi)."""
        bumps = np.linspace(-self.spec.travel_droop, self.spec.travel_bump, n)
        keys = ["camber", "rc_height", "half_track", "fvsa"]
        out = {k: [] for k in keys}
        out["bump"] = []
        for b in bumps:
            s = self.solve_bump(b)
            out["bump"].append(b)
            for k in keys:
                out[k].append(s[k])
        res = {k: np.asarray(v) for k, v in out.items()}
        res["scrub"] = res["half_track"] - self.spec.half_track
        return res

    def roll_sweep(self, phi_max: float = 2.5, n: int = 21) -> dict:
        """Varredura de rolagem com os dois contatos no solo (referencial pista).

        Rolagem positiva = carroceria gira com o lado direito para baixo,
        logo a roda direita comprime e a esquerda estende.
        """
        phis = np.linspace(0.0, phi_max, n)
        rc_z, rc_y, cam_out, cam_in = [], [], [], []

        for phi_deg in phis:
            phi = phi_deg * D2R

            def z_mismatch(b):
                sR = self.solve_bump(+b)
                sL = self.solve_bump(-b)
                cpR = sR["cp"]
                cpL = np.array([-sL["cp"][0], sL["cp"][1]])  # espelha para a esquerda
                R = rot2(-phi)                                # chassi -> pista
                return (R @ cpR)[1] - (R @ cpL)[1]

            b_guess = self.spec.half_track * np.tan(phi)
            if phi_deg == 0.0:
                b = 0.0
            else:
                lo, hi = 0.2 * b_guess, 2.5 * b_guess
                b = brentq(z_mismatch, lo, hi, xtol=1e-10)

            sR, sL = self.solve_bump(+b), self.solve_bump(-b)
            R = rot2(-phi)
            mirror = np.array([-1.0, 1.0])

            cpR, icR = R @ sR["cp"], R @ sR["ic"]
            cpL, icL = R @ (mirror * sL["cp"]), R @ (mirror * sL["ic"])
            shift = np.array([0.0, -cpR[1]])
            cpR, icR, cpL, icL = cpR + shift, icR + shift, cpL + shift, icL + shift

            rc = line_intersection(cpR, icR, cpL, icL)
            rc_z.append(rc[1] if rc is not None else np.nan)
            rc_y.append(rc[0] if rc is not None else np.nan)
            # Camber relativo a PISTA. Rolagem positiva = lado direito para baixo,
            # logo o topo da carroceria inclina para +y. Para a roda EXTERNA
            # (direita) isso e para fora -> camber mais positivo (+phi).
            # Para a roda INTERNA (esquerda), +y aponta para dentro -> mais
            # negativo (-phi).
            cam_out.append(sR["camber"] + phi_deg)
            cam_in.append(sL["camber"] - phi_deg)

        return {
            "roll": phis,
            "rc_height": np.asarray(rc_z),
            "rc_lateral": np.asarray(rc_y),
            "camber_outside": np.asarray(cam_out),
            "camber_inside": np.asarray(cam_in),
        }


# ----------------------------------------------------------------------------
# Massa, rolagem e rigidez
# ----------------------------------------------------------------------------

@dataclass
class VehicleSpec:
    wheelbase: float = 1535.0
    mass_total: float = 280.0        # carro + piloto [kg]
    mass_unsprung: float = 48.0      # soma dos 4 cantos [kg]
    cg_height: float = 300.0         # com piloto [mm]
    front_mass_frac: float = 0.45    # 45/55
    roll_gradient_target: float = 1.0  # [deg/g]

    @property
    def mass_sprung(self) -> float:
        return self.mass_total - self.mass_unsprung


def roll_stiffness_report(veh: VehicleSpec, front: CornerSpec, rear: CornerSpec) -> dict:
    """Rigidez em rolagem exigida e alvo de rigidez torcional do chassi."""
    x_cg = veh.front_mass_frac * veh.wheelbase          # do eixo dianteiro
    h_axis = front.rc_height + (rear.rc_height - front.rc_height) * (x_cg / veh.wheelbase)
    H = veh.cg_height - h_axis                           # braco de momento [mm]

    W_s = veh.mass_sprung * 9.81                         # [N]
    K_roll = W_s * (H / 1000.0) / veh.roll_gradient_target  # [N.m/deg]

    tilt_min_track = 3.4641 * veh.cg_height              # tan(60) * h_cg * 2
    return {
        "roll_axis_at_cg": h_axis,
        "roll_moment_arm": H,
        "sprung_weight_N": W_s,
        "roll_stiffness_Nm_deg": K_roll,
        "chassis_torsional_min": 3.0 * K_roll,
        "chassis_torsional_target": 5.0 * K_roll,
        "tilt_min_track": tilt_min_track,
        "tilt_ok": min(front.track, rear.track) >= tilt_min_track,
    }


# ----------------------------------------------------------------------------
# Relatorio
# ----------------------------------------------------------------------------

def check(label: str, value: float, lo: float, hi: float, unit: str = "mm") -> str:
    flag = "OK " if lo <= value <= hi else "!! "
    return f"  {flag}{label:<34s} {value:8.4f} {unit:<7s} (alvo {lo:g} a {hi:g})"


def report(spec: CornerSpec) -> str:
    g = solve_pickups(spec)
    kin = CornerKinematics(spec)
    sw = kin.bump_sweep()

    i0 = int(np.argmin(np.abs(sw["bump"])))
    dcam = np.gradient(sw["camber"], sw["bump"])[i0]
    drc = np.gradient(sw["rc_height"], sw["bump"])[i0]
    dscrub = np.gradient(sw["scrub"], sw["bump"])[i0]

    rim_r = spec.rim_diameter_in * 25.4 / 2.0
    rim_lo = spec.loaded_radius - rim_r
    rim_hi = spec.loaded_radius + rim_r

    L = [f"\n{'=' * 74}", f" {spec.name.upper()}", "=" * 74]
    L.append("\n PONTOS  (y para fora, z para cima, origem no solo/linha de centro)")
    L.append(f"   {'ponto':<26s} {'y [mm]':>10s} {'z [mm]':>10s}")
    for key, nm in [("lbj", "Pivo inferior externo"), ("ubj", "Pivo superior externo"),
                    ("lca_in", "LCA interno"), ("uca_in", "UCA interno"),
                    ("fvic", "FVIC (construcao)")]:
        p = g[key]
        L.append(f"   {nm:<26s} {p[0]:10.1f} {p[1]:10.1f}")

    L.append("\n BRACOS")
    L.append(f"   Comprimento LCA / UCA          {g['lca_length']:7.1f} / {g['uca_length']:.1f} mm"
             f"   (razao {g['uca_length'] / g['lca_length']:.3f})")
    L.append(f"   Separacao vertical externa     {g['sep_outboard']:7.1f} mm")
    L.append(f"   Separacao vertical interna     {g['sep_inboard']:7.1f} mm")
    lca_dir = "sobe" if g["lca_angle_deg"] < 0 else "desce"
    L.append(f"   Inclinacao LCA                 {g['lca_angle_deg']:7.2f} deg"
             f"   ({lca_dir} indo da roda para o chassi)")
    L.append(f"   Inclinacao UCA                 {g['uca_angle_deg']:7.2f} deg"
             f"   ({'desce' if g['uca_angle_deg'] > 0 else 'SOBE -> ERRO'} indo para o chassi)")
    L.append(f"   RC que deixaria o LCA plano    {spec.rc_height_for_flat_lca:7.1f} mm")

    L.append("\n VERIFICACOES")
    L.append(check("Altura do centro de rolagem", spec.rc_height, 20, 70))
    L.append(check("Comprimento FVSA", spec.fvsa_length, 1300, 1700))
    L.append(check("Raio de scrub", spec.scrub_radius, 5, 25))
    L.append(check("KPI", spec.kpi, spec.kpi_band[0], spec.kpi_band[1], "deg"))
    L.append(check("Comprimento de mangueta", spec.spindle_length, 200, 260))
    L.append(check("Comprimento LCA", g["lca_length"], 320, 430))
    L.append(check("Razao UCA/LCA", g["uca_length"] / g["lca_length"], 0.55, 0.98, "-"))
    L.append(check("Ganho de camber", abs(dcam), 0.030, 0.050, "deg/mm"))
    ok_side = "OK " if g["fvic"][0] < 0 else "!! "
    L.append(f"  {ok_side}{'FVIC no lado oposto do carro':<34s} {g['fvic'][0]:8.1f} mm")
    ok_rim = "OK " if (rim_lo + 15 <= spec.lbj_z and spec.ubj[1] <= rim_hi - 15) else "!! "
    L.append(f"  {ok_rim}{'Pivos dentro do aro':<34s} "
             f"{rim_lo:.0f} a {rim_hi:.0f} mm (LBJ {spec.lbj_z:.0f}, UBJ {spec.ubj[1]:.0f})")

    L.append("\n TAXAS EM TORNO DO ESTATICO")
    L.append(f"   Ganho de camber                {dcam:7.4f} deg/mm"
             f"   ({dcam * spec.travel_bump:+.2f} deg em {spec.travel_bump:.0f} mm)")
    L.append(f"   Migracao do centro de rolagem  {drc:7.4f} mm/mm")
    L.append(f"   Variacao de meia-bitola        {dscrub:7.4f} mm/mm")
    L.append(f"   Camber em compressao total     {sw['camber'][-1]:7.2f} deg")
    L.append(f"   Camber em extensao total       {sw['camber'][0]:7.2f} deg")
    L.append(f"   RC entre extremos de curso     {sw['rc_height'].min():7.1f} a "
             f"{sw['rc_height'].max():.1f} mm")

    rs = kin.roll_sweep(phi_max=spec.roll_ref, n=7)
    cam_out, cam_in = rs["camber_outside"][-1], rs["camber_inside"][-1]
    L.append(f"\n EM {spec.roll_ref:.1f} deg DE ROLAGEM  (camber relativo a PISTA)")
    L.append(f"   Roda externa                   {cam_out:7.2f} deg"
             f"   (estatico {spec.static_camber:+.2f})")
    L.append(f"   Roda interna                   {cam_in:7.2f} deg")
    L.append(f"   Altura do RC                   {rs['rc_height'][-1]:7.1f} mm"
             f"   (projeto {spec.rc_height:.1f})")
    L.append(f"   Migracao lateral do RC         {rs['rc_lateral'][-1]:7.1f} mm")
    flag = "OK " if -2.5 <= cam_out <= 0.0 else "!! "
    L.append(f"  {flag}Camber da roda externa na faixa util (-2.5 a 0.0 deg)")
    if cam_out > 0.0:
        L.append("      -> roda externa POSITIVA: aumente camber estatico "
                 "negativo ou encurte o FVSA")
    return "\n".join(L)


def vehicle_report(veh: VehicleSpec, front: CornerSpec, rear: CornerSpec) -> str:
    r = roll_stiffness_report(veh, front, rear)
    L = [f"\n{'=' * 74}", " VEICULO / RIGIDEZ", "=" * 74]
    L.append(f"   Massa total / suspensa         {veh.mass_total:7.1f} / {veh.mass_sprung:.1f} kg")
    L.append(f"   Eixo de rolagem na altura do CG{r['roll_axis_at_cg']:7.1f} mm")
    L.append(f"   Braco de momento de rolagem    {r['roll_moment_arm']:7.1f} mm")
    L.append(f"   Roll gradient alvo             {veh.roll_gradient_target:7.2f} deg/g")
    L.append(f"   -> Rigidez em rolagem exigida  {r['roll_stiffness_Nm_deg']:7.1f} N.m/deg")
    L.append(f"   -> Rigidez torcional do chassi {r['chassis_torsional_min']:7.0f} min / "
             f"{r['chassis_torsional_target']:.0f} alvo  N.m/deg")
    tilt = "OK " if r["tilt_ok"] else "!! "
    L.append(f"  {tilt} Teste de inclinacao 60 deg: bitola minima "
             f"{r['tilt_min_track']:.0f} mm "
             f"(atual {min(front.track, rear.track):.0f} mm)")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------------

def plot_all(front: CornerSpec, rear: CornerSpec, path: str = "geometria.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Geometria de suspensao — vista frontal", fontsize=13)

    for spec, color in [(front, "tab:blue"), (rear, "tab:red")]:
        kin = CornerKinematics(spec)
        g = kin.g
        sw = kin.bump_sweep()
        rs = kin.roll_sweep()

        a = ax[0, 0]
        a.plot([g["lbj"][0], g["lca_in"][0]], [g["lbj"][1], g["lca_in"][1]],
               "-o", color=color, lw=2, label=spec.name)
        a.plot([g["ubj"][0], g["uca_in"][0]], [g["ubj"][1], g["uca_in"][1]],
               "-o", color=color, lw=2)
        a.plot([g["lbj"][0], g["ubj"][0]], [g["lbj"][1], g["ubj"][1]],
               "-", color=color, lw=3, alpha=0.5)
        a.plot([spec.half_track, g["fvic"][0]], [0, g["fvic"][1]],
               "--", color=color, lw=0.8, alpha=0.6)
        a.plot(g["fvic"][0], g["fvic"][1], "x", color=color, ms=9)
        a.plot(0, spec.rc_height, "s", color=color, ms=7)

        ax[0, 1].plot(sw["camber"], sw["bump"], color=color, label=spec.name)
        ax[0, 2].plot(sw["rc_height"], sw["bump"], color=color, label=spec.name)
        ax[1, 0].plot(sw["scrub"], sw["bump"], color=color, label=spec.name)
        ax[1, 1].plot(rs["roll"], rs["rc_height"], color=color, label=f"{spec.name} altura")
        ax[1, 1].plot(rs["roll"], rs["rc_lateral"], "--", color=color,
                      label=f"{spec.name} lateral")
        ax[1, 2].plot(rs["roll"], rs["camber_outside"], color=color,
                      label=f"{spec.name} externa")
        ax[1, 2].plot(rs["roll"], rs["camber_inside"], "--", color=color,
                      label=f"{spec.name} interna")

    ax[0, 0].axhline(0, color="k", lw=1)
    ax[0, 0].axvline(0, color="k", lw=0.5, ls=":")
    ax[0, 0].set(title="Construcao (vista frontal)", xlabel="y [mm]", ylabel="z [mm]")
    ax[0, 0].set_aspect("equal"); ax[0, 0].legend(fontsize=8)

    ax[0, 1].set(title="Camber vs curso", xlabel="camber [deg]", ylabel="bump [mm]")
    ax[0, 2].set(title="Centro de rolagem vs curso", xlabel="RC [mm]", ylabel="bump [mm]")
    ax[1, 0].set(title="Variacao de meia-bitola", xlabel="scrub [mm]", ylabel="bump [mm]")
    ax[1, 1].set(title="Migracao do RC em rolagem", xlabel="rolagem [deg]", ylabel="RC [mm]")
    ax[1, 2].set(title="Camber vs pista em rolagem", xlabel="rolagem [deg]",
                 ylabel="camber [deg]")

    for a in ax.flat:
        a.grid(alpha=0.3)
        if a is not ax[0, 0]:
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return path


# ----------------------------------------------------------------------------
# Configuracao PUCPR Racing
# ----------------------------------------------------------------------------

FRONT = CornerSpec(
    name="Dianteira",
    track=1240.0, rc_height=35.0, fvsa_length=1500.0,
    loaded_radius=245.0, rim_diameter_in=13.0,
    lbj_y=582.0, lbj_z=130.0, kpi=10.0, spindle_length=259.34,
    rail_y=175.0, static_camber=-1.5,
)

REAR = CornerSpec(
    name="Traseira",
    track=1200.0, rc_height=55.0, fvsa_length=1400.0,
    loaded_radius=245.0, rim_diameter_in=13.0,
    lbj_y=558.6, lbj_z=130.0, kpi=8.5, spindle_length=263.23,
    rail_y=175.0, static_camber=-1.5, kpi_band=(3.0, 10.0),
)

VEHICLE = VehicleSpec(
    wheelbase=1540.0, mass_total=315.0, mass_unsprung=45.0,
    cg_height=320.0, front_mass_frac=0.45, roll_gradient_target=1.0,
)





# ============================================================================
# PARTE 2 — VISTA LATERAL E PLANTA: COORDENADAS X
# ============================================================================
#
# Convencao 3D:
#     x : positivo para TRAS   (origem no eixo dianteiro)
#     y : positivo para FORA   (origem na linha de centro)
#     z : positivo para CIMA   (origem no solo, altura de rodagem de projeto)
#
# Cada bracao tem DUAS fixacoes no chassi, separadas em x pela "base".
# O eixo de pivo e a reta que une essas duas fixacoes.
#
# Dois efeitos independentes:
#   1) VARREDURA EM PLANTA (sweep): desloca o pivo externo em x em relacao ao
#      meio da base. NAO altera a vista frontal (o eixo continua paralelo a x),
#      mas concentra carga numa das pernas.
#   2) INCLINACAO EM VISTA LATERAL (dz): fixacoes dianteira e traseira em z
#      diferentes. E ISSO que gera anti-dive / anti-squat.
# ============================================================================


@dataclass
class AxleXSpec:
    """Layout longitudinal de um canto."""

    name: str = "Dianteira"
    axle_x: float = 0.0              # x do centro de roda / pivos externos [mm]
    is_front: bool = True

    # --- base entre as duas fixacoes de cada bracao ---
    base_lca: float = 260.0          # [mm]
    base_uca: float = 240.0          # [mm]

    # --- varredura: e = x_pivo_externo - x_meio_da_base ---
    #     e > 0  -> pivo externo ATRAS do meio da base (bracao "arrastado")
    sweep_lca: float = 0.0           # [mm]
    sweep_uca: float = 0.0           # [mm]

    # --- inclinacao em vista lateral: dz = z(fix. traseira) - z(fix. dianteira)
    dz_lca: float = 0.0              # [mm]
    dz_uca: float = 0.0              # [mm]

    # ------------------------------------------------------------------
    def _arm_x(self, base, sweep):
        """x das fixacoes dianteira e traseira."""
        x_mid = self.axle_x - sweep
        return x_mid - base / 2.0, x_mid + base / 2.0

    @property
    def lca_x(self):
        return self._arm_x(self.base_lca, self.sweep_lca)

    @property
    def uca_x(self):
        return self._arm_x(self.base_uca, self.sweep_uca)

    def ratio_ea(self, arm="lca"):
        """|e|/a — mede o quanto a carga migra para uma perna so."""
        if arm == "lca":
            return abs(self.sweep_lca) / (self.base_lca / 2.0)
        return abs(self.sweep_uca) / (self.base_uca / 2.0)


def build_3d(corner: CornerSpec, xspec: AxleXSpec) -> dict:
    """Combina a vista frontal (y,z) com o layout longitudinal (x).

    Retorna os 6 pontos do canto em 3D.
    """
    g = solve_pickups(corner)
    xl_f, xl_r = xspec.lca_x
    xu_f, xu_r = xspec.uca_x

    z_lca, z_uca = g["lca_in"][1], g["uca_in"][1]
    ry = corner.rail_y

    return {
        "LBJ  (pivo inf. externo)": np.array([xspec.axle_x, *g["lbj"]]),
        "UBJ  (pivo sup. externo)": np.array([xspec.axle_x, *g["ubj"]]),
        "LCA fix. dianteira":       np.array([xl_f, ry, z_lca - xspec.dz_lca / 2.0]),
        "LCA fix. traseira":        np.array([xl_r, ry, z_lca + xspec.dz_lca / 2.0]),
        "UCA fix. dianteira":       np.array([xu_f, ry, z_uca - xspec.dz_uca / 2.0]),
        "UCA fix. traseira":        np.array([xu_r, ry, z_uca + xspec.dz_uca / 2.0]),
    }


# ----------------------------------------------------------------------------
# Vista lateral: centro instantaneo e anti-geometria
# ----------------------------------------------------------------------------

def side_view_ic(corner: CornerSpec, xspec: AxleXSpec):
    """Centro instantaneo em vista lateral (SVIC), em (x, z).

    Construcao: projete cada eixo de pivo no plano xz; o SVIC e a intersecao
    das duas projecoes. Eixos horizontais (dz=0) -> retas paralelas -> SVIC no
    infinito -> anti = 0.
    """
    g = solve_pickups(corner)
    xl_f, xl_r = xspec.lca_x
    xu_f, xu_r = xspec.uca_x
    zl, zu = g["lca_in"][1], g["uca_in"][1]

    p1 = np.array([xl_f, zl - xspec.dz_lca / 2.0])
    p2 = np.array([xl_r, zl + xspec.dz_lca / 2.0])
    p3 = np.array([xu_f, zu - xspec.dz_uca / 2.0])
    p4 = np.array([xu_r, zu + xspec.dz_uca / 2.0])

    return line_intersection(p1, p2, p3, p4)   # None se paralelas


def anti_percent(corner: CornerSpec, xspec: AxleXSpec, veh: VehicleSpec,
                 brake_bias_front: float = 0.70) -> dict:
    """Anti-dive (frenagem) e anti-squat (aceleracao).

    Freios OUTBOARD e diferencial montado no CHASSI: em ambos os casos a
    construcao parte do CONTATO DO PNEU ate o SVIC.

        %anti-dive_diant = eps     * tan(theta) * L/h
        %anti-lift_tras  = (1-eps) * tan(theta) * L/h
        %anti-squat_tras =           tan(theta) * L/h
    """
    ic = side_view_ic(corner, xspec)
    L_over_h = veh.wheelbase / veh.cg_height

    if ic is None or not np.isfinite(ic).all():
        return {"svic": None, "svsa": np.inf, "tan_theta": 0.0,
                "anti_dive": 0.0, "anti_lift": 0.0, "anti_squat": 0.0}

    dx = ic[0] - xspec.axle_x            # SVIC em relacao ao contato do pneu
    if xspec.is_front:
        tan_t = ic[1] / dx if abs(dx) > 1e-9 else np.inf
    else:
        tan_t = ic[1] / (-dx) if abs(dx) > 1e-9 else np.inf

    return {
        "svic": ic,
        "svsa": abs(dx),
        "tan_theta": tan_t,
        "anti_dive": 100.0 * brake_bias_front * tan_t * L_over_h,
        "anti_lift": 100.0 * (1 - brake_bias_front) * tan_t * L_over_h,
        "anti_squat": 100.0 * tan_t * L_over_h,
    }


def dz_uca_for_anti(corner: CornerSpec, xspec: AxleXSpec, veh: VehicleSpec,
                    target_pct: float, mode: str = "dive",
                    brake_bias_front: float = 0.70) -> float:
    """Resolve o dz do UCA que entrega o anti desejado (mantendo dz_lca).

    mode: 'dive' (dianteira), 'lift' (traseira em frenagem) ou 'squat'.
    """
    from dataclasses import replace as _replace

    def err(dz):
        xs = _replace(xspec, dz_uca=dz)
        a = anti_percent(corner, xs, veh, brake_bias_front)
        key = {"dive": "anti_dive", "lift": "anti_lift", "squat": "anti_squat"}[mode]
        return a[key] - target_pct

    if abs(target_pct) < 1e-9:
        return xspec.dz_lca
    lo, hi = xspec.dz_lca - 120.0, xspec.dz_lca + 120.0
    # evita a singularidade em dz_uca == dz_lca (retas paralelas)
    eps = 1e-3
    for a_, b_ in [(lo, xspec.dz_lca - eps), (xspec.dz_lca + eps, hi)]:
        try:
            if err(a_) * err(b_) < 0:
                return brentq(err, a_, b_, xtol=1e-9)
        except (ValueError, ZeroDivisionError):
            continue
    return np.nan


# ----------------------------------------------------------------------------
# Divisao de carga entre as duas pernas do bracao
# ----------------------------------------------------------------------------

def leg_forces(bj, p_front, p_rear, load_at_bj):
    """Divide a carga da rotula entre as duas pernas do bracao.

    O bracao so reage carga no SEU PLANO; a componente perpendicular vai para
    o pushrod / amortecedor. Por isso a carga e projetada no plano do bracao.
    Positivo = tracao.
    """
    bj, p1, p2 = map(lambda v: np.asarray(v, float), (bj, p_front, p_rear))
    u1, u2 = p1 - bj, p2 - bj
    n = np.cross(u1, u2)
    n /= np.linalg.norm(n)

    F = np.asarray(load_at_bj, float)
    F_plane = F - np.dot(F, n) * n

    e1 = u1 / np.linalg.norm(u1)
    e2 = u2 / np.linalg.norm(u2)
    b2 = np.cross(n, e1)
    A = np.array([[1.0, np.dot(e2, e1)],
                  [0.0, np.dot(e2, b2)]])
    rhs = -np.array([np.dot(F_plane, e1), np.dot(F_plane, b2)])
    return np.linalg.solve(A, rhs)


def upright_reactions(corner: CornerSpec, xspec: AxleXSpec, load_cp):
    """Equilibrio do upright -> forcas nas duas rotulas.

    Modelo (vista frontal, pushrod fixado no LCA):
      - a forca no pivo SUPERIOR atua ao longo do plano do UCA (1 incognita),
        obtida por momento em torno do pivo inferior;
      - o pivo INFERIOR fecha o equilibrio de forcas (2 incognitas);
      - o pushrod absorve o que o LCA nao consegue reagir no proprio plano.

    load_cp = (Fx, Fy, Fz) no contato do pneu.
    Retorna as forcas aplicadas PELO UPRIGHT nos bracoes (acao, nao reacao).
    """
    g = solve_pickups(corner)
    lbj = np.array([xspec.axle_x, *g["lbj"]])
    ubj = np.array([xspec.axle_x, *g["ubj"]])
    cp = np.array([xspec.axle_x, corner.half_track, 0.0])

    d = g["uca_in"] - g["ubj"]                 # direcao do UCA em vista frontal
    d = np.array([0.0, d[0], d[1]])
    d /= np.linalg.norm(d)

    F = np.asarray(load_cp, float)
    r_cp = cp - lbj
    r_u = ubj - lbj

    # momento em torno do eixo x que passa pelo pivo inferior
    M = np.cross(r_cp, F)[0]
    denom = np.cross(r_u, d)[0]
    T_u = -M / denom
    F_ubj = T_u * d
    F_lbj = -(F + F_ubj)

    # forcas que o UPRIGHT aplica nos bracoes = reacao do que sofre
    return -F_lbj, -F_ubj, lbj, ubj


def load_cases(corner: CornerSpec, veh: VehicleSpec, mu_y=1.5, mu_x=1.4,
               lltd=0.55, ay=1.5) -> dict:
    """Casos de carga aproximados NO CONTATO do pneu externo carregado.

    Estimativa de primeira ordem para dimensionamento. Substitua por dados do
    simulador / TTC quando disponiveis.
    """
    W = veh.mass_total * 9.81
    is_front = corner.name.lower().startswith("d")
    frac = veh.front_mass_frac if is_front else 1 - veh.front_mass_frac
    share = lltd if is_front else 1 - lltd

    Fz_static = W * frac / 2.0
    dW = share * (W * ay * veh.cg_height / 1000.0) / (corner.track / 1000.0)
    Fz = Fz_static + dW
    return {
        "Curva pura": np.array([0.0, -mu_y * Fz, Fz]),
        "Frenagem":   np.array([mu_x * Fz, 0.0, Fz]),
        "Combinado":  np.array([0.7 * mu_x * Fz, -0.7 * mu_y * Fz, Fz]),
        "Fz": Fz,
    }


# ----------------------------------------------------------------------------
# Relatorio 3D
# ----------------------------------------------------------------------------

def report_3d(corner: CornerSpec, xspec: AxleXSpec, veh: VehicleSpec,
              brake_bias_front: float = 0.70) -> str:
    P = build_3d(corner, xspec)
    a = anti_percent(corner, xspec, veh, brake_bias_front)
    lc = load_cases(corner, veh)

    L = [f"\n{'=' * 78}", f" {corner.name.upper()} — PONTOS 3D", "=" * 78]
    L.append("\n (x para tras desde o eixo dianteiro; y para fora; z para cima desde o solo)")
    L.append(f"   {'ponto':<28s} {'x [mm]':>9s} {'y [mm]':>9s} {'z [mm]':>9s}")
    for k, v in P.items():
        L.append(f"   {k:<28s} {v[0]:9.1f} {v[1]:9.1f} {v[2]:9.1f}")

    L.append("\n LAYOUT LONGITUDINAL")
    L.append(f"   Base LCA / UCA                 {xspec.base_lca:6.0f} / {xspec.base_uca:.0f} mm")
    L.append(f"   Varredura LCA / UCA            {xspec.sweep_lca:6.0f} / {xspec.sweep_uca:.0f} mm")
    for arm in ("lca", "uca"):
        r = xspec.ratio_ea(arm)
        flag = "OK " if r <= 1.5 else "!! "
        L.append(f"  {flag}Razao e/a {arm.upper():<21s} {r:6.2f}       (alvo <= 1.5)")

    L.append("\n VISTA LATERAL / ANTI-GEOMETRIA")
    if a["svic"] is None:
        L.append("   Eixos de pivo horizontais -> SVIC no infinito, anti = 0%")
    else:
        L.append(f"   SVIC (x, z)                    {a['svic'][0]:9.1f} {a['svic'][1]:9.1f} mm")
        L.append(f"   SVSA                           {a['svsa']:9.1f} mm")
    if xspec.is_front:
        L.append(check("Anti-dive", a["anti_dive"], 0, 30, "%"))
    else:
        L.append(check("Anti-squat", a["anti_squat"], 0, 30, "%"))
        L.append(check("Anti-lift (frenagem)", a["anti_lift"], 0, 30, "%"))

    L.append(f"\n FORCAS NAS PERNAS  (Fz externo estimado {lc['Fz']:.0f} N)")
    L.append(f"   {'caso':<14s} {'LCA diant':>11s} {'LCA tras':>11s} "
             f"{'UCA diant':>11s} {'UCA tras':>11s}")
    for case in ("Curva pura", "Frenagem", "Combinado"):
        Flbj, Fubj, lbj, ubj = upright_reactions(corner, xspec, lc[case])
        fl = leg_forces(lbj, P["LCA fix. dianteira"], P["LCA fix. traseira"], Flbj)
        fu = leg_forces(ubj, P["UCA fix. dianteira"], P["UCA fix. traseira"], Fubj)
        L.append(f"   {case:<14s} {fl[0]:11.0f} {fl[1]:11.0f} {fu[0]:11.0f} {fu[1]:11.0f}")
    L.append("   (positivo = tracao, negativo = compressao)")
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Layout PUCPR Racing
# ----------------------------------------------------------------------------

FRONT_X = AxleXSpec(
    name="Dianteira", axle_x=0.0, is_front=True,
    base_lca=260.0, base_uca=240.0,
    sweep_lca=0.0, sweep_uca=0.0,
    dz_lca=0.0, dz_uca=0.0,
)

REAR_X = AxleXSpec(
    name="Traseira", axle_x=1540.0, is_front=False,
    base_lca=340.0, base_uca=320.0,
    sweep_lca=230.0, sweep_uca=220.0,     # arrastada: fixacoes a frente do pivo
    dz_lca=0.0, dz_uca=0.0,
)


if __name__ == "__main__":
    # --- Parte 1: vista frontal ---
    print(report(FRONT))
    print(report(REAR))
    print(vehicle_report(VEHICLE, FRONT, REAR))

    # --- Parte 2: coordenadas X, anti-geometria, forcas ---
    print(report_3d(FRONT, FRONT_X, VEHICLE))
    print(report_3d(REAR, REAR_X, VEHICLE))

    out = plot_all(FRONT, REAR)
    print(f"\nGrafico salvo em: {out}\n")
