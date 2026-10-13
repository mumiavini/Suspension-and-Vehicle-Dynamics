"""VD Studio audit fixes, group physics: maneuver, concept, setup sheet, exports.

Every test pins a fixed behaviour with the audit's repro numbers where they
exist: the symptom is gone and a known answer (computed independently in the
test) holds. Default car: m = 315 kg, m_u = 45 kg, h = 320 mm, CG 847 mm aft
of the front axle on L = 1540 mm, loaded radius 245 mm, solved RCs 35.50 /
55.00 mm, contact-patch tracks 1240 / 1200 mm.

Frames: ISO 8855 (X fwd, Y LEFT, Z up). Positive ay = left turn: right wheels
are outer.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest

from apps.vdstudio.engine import concept, maneuver, mass
from apps.vdstudio.engine import exports as X
from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine.project import MassItem, Project

g = K.GRAVITY


@pytest.fixture(scope="module")
def geo() -> G.CarGeometry:
    return G.resolve(Project())


@pytest.fixture(scope="module")
def static(geo) -> K.KpiSet:
    return K.compute_static(geo)


@pytest.fixture(scope="module")
def rc(static) -> dict[str, float]:
    return {"front": static.value("front.rc_static"), "rear": static.value("rear.rc_static")}


def _sprung_by_hand(v, r_f: float = 245.0, r_r: float = 245.0):
    """Independent sprung CG: remove half the unsprung mass at each axle (wheel-centre height)."""
    m, mu = v.total_mass_kg, v.unsprung_mass_kg
    ms = m - mu
    hs = (m * v.cg_height_mm - 0.5 * mu * r_f - 0.5 * mu * r_r) / ms
    ms_front = m * (1.0 - v.cg_from_front_axle_mm / v.wheelbase_mm) - 0.5 * mu
    a_s = v.wheelbase_mm * (ms - ms_front) / ms
    return ms, hs, a_s, ms_front / ms


def _moment_Nm(geo, r) -> float:
    return sum(r.transfer[ax]["total"] * geo.track_mm(ax) for ax in ("front", "rear")) / 1000.0


# --------------------------------------------------------------------------- #
# physics-01: sprung CG in the lateral transfer; statics closes
# --------------------------------------------------------------------------- #

def test_p01_sprung_cg_known_answer(geo, rc) -> None:
    v = geo.vehicle_data
    ms, hs, a_s, _ = _sprung_by_hand(v)
    assert hs == pytest.approx(332.5, abs=1e-9)
    assert a_s == pytest.approx(859.833, abs=1e-3)
    r = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(ay_g=1.5, roll_gradient_deg_per_g=0.0),
                              rc)
    assert r.sprung.cg_height_mm == pytest.approx(hs, abs=1e-9)
    assert r.roll_axis_at_cg_mm == pytest.approx(46.387, abs=1e-3)
    assert r.roll_moment_arm_mm == pytest.approx(286.113, abs=1e-3)
    # verifier's independent decomposition (no pendulum at gradient 0)
    assert r.transfer["front"]["total"] == pytest.approx(574.02, abs=0.01)
    assert r.transfer["rear"]["total"] == pytest.approx(642.91, abs=0.01)
    assert r.wheels["FL"].fz_N == pytest.approx(121.3, abs=0.05)
    assert r.wheels["RL"].fz_N == pytest.approx(206.9, abs=0.05)
    # the symptom: 1433.609 N·m (-3.35 %) before; m·g·ay·h now
    assert _moment_Nm(geo, r) == pytest.approx(1483.272, abs=1e-3)


def test_p01_moment_balance_holds_for_random_inputs(geo) -> None:
    rng = np.random.default_rng(20261012)
    for _ in range(12):
        v = replace(geo.vehicle_data,
                    total_mass_kg=float(rng.uniform(220, 400)),
                    unsprung_mass_kg=float(rng.uniform(25, 70)),
                    cg_height_mm=float(rng.uniform(250, 380)),
                    cg_from_front_axle_mm=float(rng.uniform(0.4, 0.65)) * 1540.0)
        gg = replace(geo, vehicle_data=v)
        inp = maneuver.ManeuverInputs(ay_g=float(rng.uniform(-2.0, 2.0)),
                                      ax_g=float(rng.uniform(-1.0, 1.0)),
                                      roll_gradient_deg_per_g=float(rng.choice([0.0, 0.7, 1.6])),
                                      roll_stiffness_front_share=float(rng.uniform(0, 1)))
        rcs = {"front": float(rng.uniform(0, 80)), "rear": float(rng.uniform(0, 100))}
        r = maneuver.run_maneuver(gg, inp, rcs)
        lhs = _moment_Nm(gg, r) - r.pendulum_moment_Nm
        rhs = v.total_mass_kg * g * inp.ay_g * v.cg_height_mm / 1000.0
        assert abs(lhs - rhs) < 1e-6
        # the pendulum moment is W_s·d·φ (small angle)
        ws_d = r.sprung.mass_kg * g * r.roll_moment_arm_mm / 1000.0
        assert r.pendulum_moment_Nm == pytest.approx(ws_d * math.radians(r.roll_deg), abs=1e-9)
        # total vertical load is conserved
        assert sum(w.fz_N for w in r.wheels.values()) == pytest.approx(
            v.total_mass_kg * g, rel=1e-12)


def test_p01_concept_roll_budget_uses_the_sprung_cg() -> None:
    import sla_geometry as sla

    p = Project()
    v = p.vehicle
    ck = concept.concept_kpis(p)
    ms, hs, a_s, _ = _sprung_by_hand(v, p.front.loaded_radius_mm, p.rear.loaded_radius_mm)
    rcf, rcr = p.front.rc_height_mm, p.rear.rc_height_mm
    d = hs - (rcf + (rcr - rcf) * a_s / v.wheelbase_mm)
    grad = v.target_roll_gradient_deg_per_g
    assert ck.value("concept.roll_arm") == pytest.approx(d, abs=1e-9)
    assert ck.value("concept.k_roll") == pytest.approx(ms * g * d / 1000.0 / grad, rel=1e-12)
    assert ck.value("concept.k_roll_pend") == pytest.approx(
        ms * g * d / 1000.0 * (1.0 / grad + math.pi / 180.0), rel=1e-12)
    # the whole-car value is quoted, not used
    whole = sla.solve_vehicle(v, rcf, rcr, 1200.0)
    assert f"{whole.required_roll_stiffness_Nm_per_deg:.1f}" in ck.get("concept.k_roll").note
    assert mass.UNSPRUNG_ASSUMPTION in ck.get("concept.roll_arm").note


# --------------------------------------------------------------------------- #
# physics-02: a failed static RC is an explicit failure, never NaN loads
# --------------------------------------------------------------------------- #

def _parallel_front_arms() -> Project:
    p = Project()
    hs = G.freeze_to_hardpoints(p)
    p.hardpoints = hs
    p.geometry_mode = "hardpoints"
    for c in ("FL", "FR"):
        pts = hs.points[c]
        zl, zu = pts["LCA_OUT"][2], pts["UCA_OUT"][2]
        for n in ("LCA_IN_FRONT", "LCA_IN_REAR"):
            x, y, _ = pts[n]
            pts[n] = (x, y, zl)
        for n in ("UCA_IN_FRONT", "UCA_IN_REAR"):
            x, y, _ = pts[n]
            pts[n] = (x, y, zu)
    return p


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_p02_nan_static_rc_raises(geo) -> None:
    with pytest.raises(ValueError, match="static roll centre unavailable"):
        maneuver.run_maneuver(geo, maneuver.ManeuverInputs(), {"front": math.nan, "rear": 55.0})
    p = _parallel_front_arms()
    pg = G.resolve(p)
    st = K.compute_static(pg)
    assert not st.get("front.rc_static").ok
    rcs = {"front": st.value("front.rc_static"), "rear": st.value("rear.rc_static")}
    with pytest.raises(ValueError, match=r"front\.rc_static"):
        maneuver.run_maneuver(pg, maneuver.ManeuverInputs(), rcs)
    # the design RCs remain a valid query
    r = maneuver.run_maneuver(pg, maneuver.ManeuverInputs(rc_source="design"), rcs)
    assert all(math.isfinite(w.fz_N) for w in r.wheels.values())


def test_p02_lift_is_reported(geo, rc) -> None:
    r = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(ay_g=2.0, roll_gradient_deg_per_g=0.5),
                              rc)
    assert r.wheels["FL"].fz_N < 0
    assert any(w.startswith("FL has lifted") for w in r.warnings)


# --------------------------------------------------------------------------- #
# physics-04: a failed roll solve keeps the loads, flags the kinematics
# --------------------------------------------------------------------------- #

def test_p04_roll_solve_failure_keeps_the_load_transfer(geo, rc) -> None:
    inp = maneuver.ManeuverInputs(ay_g=2.5, roll_gradient_deg_per_g=5.0)   # 12.5 deg of roll
    r = maneuver.run_maneuver(geo, inp, rc)                                # used to raise
    assert any("front roll solve failed" in w for w in r.warnings)
    for cid in ("FL", "FR"):
        w = r.wheels[cid]
        assert not w.converged and math.isnan(w.camber_road_deg) and math.isnan(w.travel_mm)
    assert sum(w.fz_N for w in r.wheels.values()) == pytest.approx(
        geo.vehicle_data.total_mass_kg * g, rel=1e-12)
    assert _moment_Nm(geo, r) - r.pendulum_moment_Nm == pytest.approx(
        315.0 * g * 2.5 * 320.0 / 1000.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# physics-05: one weight distribution
# --------------------------------------------------------------------------- #

def test_p05_maneuver_uses_the_vehicle_mass_fraction(geo, rc) -> None:
    v = replace(geo.vehicle_data, wheelbase_mm=1600.0)        # 47.06 % front
    gg = replace(geo, vehicle_data=v)
    r = maneuver.run_maneuver(gg, maneuver.ManeuverInputs(ay_g=0.0), rc)
    W = v.total_mass_kg * g
    assert r.wheels["FL"].fz_N == pytest.approx(W * v.front_mass_fraction / 2.0, rel=1e-12)
    assert r.wheels["FL"].fz_N == pytest.approx(727.15, abs=0.01)  # was 695.3 (45.00 %)


# --------------------------------------------------------------------------- #
# physics-06: roll axis above the sprung CG
# --------------------------------------------------------------------------- #

def test_p06_roll_axis_above_cg_is_flagged(geo) -> None:
    r = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(ay_g=1.0), {"front": 400.0,
                                                                     "rear": 400.0})
    assert r.roll_moment_arm_mm == pytest.approx(332.5 - 400.0, abs=1e-9)
    assert math.isnan(r.k_roll_total_Nm_per_deg)       # was -211.9 N·m/deg
    assert any("at/above the sprung CG" in w for w in r.warnings)


# --------------------------------------------------------------------------- #
# physics-07, -12, -13: concept note wording
# --------------------------------------------------------------------------- #

def test_p07_p12_p13_concept_notes() -> None:
    p = Project()
    ck = concept.concept_kpis(p)
    ssf = ck.get("concept.ssf")
    assert ssf.value == pytest.approx(1200.0 / 640.0)
    assert "sliding before rollover needs T/(2h) > μ" in ssf.note
    assert "rollover before sliding" not in ssf.note
    assert "Rigid body" in ssf.note and "cos θ" in ssf.note
    # SSF 1.875 > μ 1.5: this car slides before it rolls -- the note now says so
    assert ssf.value > p.vehicle.mu_lateral
    tl = ck.get("concept.t_over_l").note
    assert "higher T/L (shorter wheelbase) raises the yaw gain" in tl
    assert "higher = calmer" not in tl


def test_p08_tilt_formulas_recover_the_cg_height() -> None:
    """The two printed formulas are exact for a rigid car (h 320, T 1200, W 3090.15 N)."""
    W, T, h = 315.0 * g, 1200.0, 320.0
    for deg in (20.0, 30.0, 60.0):
        th = math.radians(deg)
        # level scales, one side raised: vertical reactions
        n_high = W * (0.5 * T * math.cos(th) - h * math.sin(th)) / (T * math.cos(th))
        assert (W / 2 - n_high) * T / (W * math.tan(th)) == pytest.approx(h, rel=1e-12)
        # tilt table, platform scales: reactions normal to the platform
        n_low = W * math.cos(th) / 2 + W * h * math.sin(th) / T
        n_hi = W * math.cos(th) / 2 - W * h * math.sin(th) / T
        assert (n_low - n_hi) * T / (2 * W * math.sin(th)) == pytest.approx(h, rel=1e-12)


# --------------------------------------------------------------------------- #
# physics-10, -16, -20: pendulum term, LLTD at unit ay, input domain
# --------------------------------------------------------------------------- #

def test_p10_stiffness_matches_concept_pendulum_definition(geo) -> None:
    p = Project()
    inp = maneuver.ManeuverInputs(ay_g=1.5, rc_source="design")
    r = maneuver.run_maneuver(geo, inp)
    ck = concept.concept_kpis(p)
    assert r.k_roll_total_Nm_per_deg == pytest.approx(ck.value("concept.k_roll_pend"),
                                                      rel=1e-12)
    # elastic transfer carries W_s·d·(ay + φ)
    ws = r.sprung.mass_kg * g
    phi = math.radians(r.roll_deg)
    el = 0.5 * ws * (1.5 + phi) * r.roll_moment_arm_mm / geo.track_mm("front")
    assert r.transfer["front"]["elastic"] == pytest.approx(el, rel=1e-12)


def test_p10_default_solved_numbers(geo, rc) -> None:
    r = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(), rc)
    # 757.82 N·m per g of W_s·d times (1 + π/180); was 725.15 (whole-car h, no pendulum)
    assert r.k_roll_total_Nm_per_deg == pytest.approx(771.05, abs=0.01)
    assert r.transfer["front"]["total"] == pytest.approx(582.02, abs=0.01)
    assert r.transfer["rear"]["total"] == pytest.approx(651.18, abs=0.01)


def test_p16_lltd_defined_at_zero_ay_and_rigid_k(geo, rc) -> None:
    a = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(ay_g=0.0), rc)
    b = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(ay_g=1.5), rc)
    c = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(ay_g=-0.7), rc)
    assert math.isfinite(a.lltd_front)                     # was nan (0/0)
    assert a.lltd_front == pytest.approx(b.lltd_front, rel=1e-12)
    assert c.lltd_front == pytest.approx(b.lltd_front, rel=1e-12)
    tf = b.transfer
    assert b.lltd_front == pytest.approx(
        tf["front"]["total"] / (tf["front"]["total"] + tf["rear"]["total"]), rel=1e-12)
    rigid = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(roll_gradient_deg_per_g=0.0), rc)
    assert rigid.k_roll_total_Nm_per_deg == math.inf


def test_p20_engine_input_domain(geo, rc) -> None:
    with pytest.raises(ValueError, match="front share"):
        maneuver.run_maneuver(geo, maneuver.ManeuverInputs(roll_stiffness_front_share=1.2), rc)
    with pytest.raises(ValueError, match="roll gradient"):
        maneuver.run_maneuver(geo, maneuver.ManeuverInputs(roll_gradient_deg_per_g=-1.0), rc)
    p = Project()
    p.vehicle = replace(p.vehicle, design_ay_g=0.0)
    ck = concept.concept_kpis(p)                          # used to raise ZeroDivisionError
    for key in ("concept.lltd_lo", "concept.lltd_hi"):
        k = ck.get(key)
        assert k.status == "fail" and "ay" in k.error


# --------------------------------------------------------------------------- #
# physics-15: travel beyond the design range
# --------------------------------------------------------------------------- #

def test_p15_travel_outside_design_range_is_flagged(geo, rc) -> None:
    ok = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(), rc)       # ±16.2 mm
    assert all(w.in_design_travel for w in ok.wheels.values())
    r = maneuver.run_maneuver(geo, maneuver.ManeuverInputs(roll_gradient_deg_per_g=5.0), rc)
    assert r.wheels["FR"].travel_mm == pytest.approx(80.78, abs=0.01)
    assert not any(w.in_design_travel for w in r.wheels.values())
    assert any("FR travel +80.8 mm is outside the front design range" in w for w in r.warnings)


# --------------------------------------------------------------------------- #
# physics-19: CG estimator
# --------------------------------------------------------------------------- #

def test_p19_negative_mass_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be ≥ 0"):
        concept.cg_estimate([MassItem("a", 10.0, 300.0), MassItem("b", -5.0, 100.0)], 1540.0)
    est = concept.cg_estimate([MassItem("a", 10.0, 300.0), MassItem("b", 0.0, 100.0)], 1540.0)
    assert est.cg_height_mm == pytest.approx(300.0)


# --------------------------------------------------------------------------- #
# Setup sheet: physics-09, -17, -18 and the roll-camber label
# --------------------------------------------------------------------------- #

def _row(rows, name):
    return next(r for r in rows if r[1] == name)


def _sheet(geo, static, **fields):
    p = Project()
    p.setup_sheet.update(fields)
    return X.setup_sheet_rows(p, geo, static)


def test_p09_frequency_is_labelled_and_ride_rate_is_optional(geo, static) -> None:
    rows = _sheet(geo, static, spring_f="35", spring_r="40", mr_f="0.9", mr_r="0.9")
    fn = _row(rows, "Sprung natural frequency")
    assert "tyre compliance NOT included" in fn[5] and "upper bound" in fn[5]
    assert not any(r[1].startswith("Ride rate") for r in rows)
    # sprung corner mass from the sprung axle load: (315·0.45 − 22.5) / 2 = 59.625 kg
    m_f, m_r = (315 * 0.45 - 22.5) / 2, (315 * 0.55 - 22.5) / 2
    assert float(fn[3]) == pytest.approx(math.sqrt(28.35e3 / m_f) / (2 * math.pi), abs=0.005)
    assert _row(rows, "Sprung mass per corner")[3:5] == [f"{m_f:.1f}", f"{m_r:.1f}"]
    rows = _sheet(geo, static, spring_f="35", spring_r="40", mr_f="0.9", mr_r="0.9",
                  tyre_rate_f="100", tyre_rate_r="100")
    tyre = _row(rows, "Tyre vertical rate (estimate)")
    assert tyre[3:6] == ["100", "100", "input (estimate: no TTC data; optional)"]
    ride = _row(rows, "Ride rate (wheel rate + tyre in series)")
    kr_f, kr_r = 28.35 * 100 / 128.35, 32.4 * 100 / 132.4      # 22.09 / 24.47 N/mm
    assert float(ride[3]) == pytest.approx(kr_f, abs=0.005)
    assert float(ride[4]) == pytest.approx(kr_r, abs=0.005)
    rf = _row(rows, "Ride frequency (with tyre)")
    assert float(rf[3]) == pytest.approx(math.sqrt(kr_f * 1e3 / m_f) / (2 * math.pi), abs=0.005)
    assert float(rf[4]) == pytest.approx(math.sqrt(kr_r * 1e3 / m_r) / (2 * math.pi), abs=0.005)


@pytest.mark.parametrize(("text", "value"), [("35", 35.0), ("35,0", 35.0), ("35 N/mm", 35.0),
                                             ("0,9", 0.9), (" .9 ", 0.9), ("1e2", 100.0)])
def test_p17_parser_accepts_decimal_comma_and_units(text, value) -> None:
    assert X._f(text) == pytest.approx(value)


def test_p17_setup_sheet_reports_unreadable_and_flags_mr(geo, static) -> None:
    rows = _sheet(geo, static, spring_f="35,0", mr_f="0,9")
    assert _row(rows, "Wheel rate")[3] == "28.35"             # was '—'
    rows = _sheet(geo, static, spring_f="abc", mr_f="0.9")
    wr = _row(rows, "Wheel rate")
    assert wr[3] == "—" and "could not read front spring rate 'abc'" in wr[5]
    rows = _sheet(geo, static, spring_f="35", mr_f="1.6")
    wr = _row(rows, "Wheel rate")
    assert float(wr[3]) == pytest.approx(35 * 1.6 ** 2)       # flagged, never rejected
    assert "check MR definition (spring/wheel)" in wr[5]


def test_p18_springs_only_roll_rows(geo, static, rc) -> None:
    rows = _sheet(geo, static, spring_f="35", spring_r="40", mr_f="0.9", mr_r="0.9")
    rr_f = 0.5 * 28.35e3 * 1.24 ** 2 * math.pi / 180
    rr_r = 0.5 * 32.4e3 * 1.20 ** 2 * math.pi / 180
    assert float(_row(rows, "Roll stiffness front share (springs only)")[3]) == pytest.approx(
        rr_f / (rr_f + rr_r), abs=5e-4)                       # 0.483
    ms, hs, a_s, _ = _sprung_by_hand(geo.vehicle_data)
    wsd = ms * g * (hs - (rc["front"] + (rc["rear"] - rc["front"]) * a_s / 1540.0)) / 1000
    grad = wsd / (rr_f + rr_r - wsd * math.pi / 180)
    row = _row(rows, "Roll gradient (springs only; no ARB, no tyre)")
    assert float(row[3]) == pytest.approx(grad, abs=5e-4)    # 0.979 deg/g
    assert "ARB and tyre not included" in row[5]


def test_roll_camber_row_names_its_frame(geo, static) -> None:
    rows = X.setup_sheet_rows(Project(), geo, static)
    assert any(r[1] == "Roll camber (chassis-ref.)" for r in rows)
    assert not any(r[1] == "Roll camber" for r in rows)


# --------------------------------------------------------------------------- #
# Exports: whatif_ui-05, -12, -21
# --------------------------------------------------------------------------- #

def test_whatif05_report_prints_each_axles_band(geo, static) -> None:
    md = X.markdown_report(Project(), geo, static)
    lines = md.splitlines()
    kpi = next(ln for ln in lines if ln.startswith("| KPI (L) |"))
    assert "| F 6 … 14 / R 3 … 10 |" in kpi                   # was '| 6 … 14 |'
    trail = next(ln for ln in lines if ln.startswith("| Mechanical trail (L) |"))
    assert "| F 10 … 35 / R — |" in trail and "R no band" in trail
    rc_row = next(ln for ln in lines if ln.startswith("| Roll centre height (static) |"))
    assert "| 20 … 70 |" in rc_row                            # shared band: unchanged


def test_whatif12_optimumk_alignment_header_precision() -> None:
    p = Project()
    p.steering = replace(p.steering, static_toe_deg_per_side=0.25)
    p.front = replace(p.front, static_camber_deg=-1.75)
    txt = X.optimumk_text(G.resolve(p))
    assert "# Static camber: -1.750 deg" in txt                # was -1.8
    assert "# Static toe (per side): 0.250 deg" in txt         # was 0.2


def test_whatif21_kpi_csv_carries_full_precision(static) -> None:
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(X.kpi_csv(static))))
    head = list(rows[0])
    assert head[head.index("value") + 1] == "value_raw"
    by_key = {r["key"]: r for r in rows}
    for key, k in static.items.items():
        if k.ok:
            assert float(by_key[key]["value_raw"]) == k.value     # exact round trip
        else:
            assert by_key[key]["value_raw"] == ""
    assert by_key["front.rc_static"]["value"] == "35.50"


# --------------------------------------------------------------------------- #
# UI: maneuver page (physics-02/03/04/11/14/15/16), concept page (-08/-19), reports (-20)
# --------------------------------------------------------------------------- #

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VDSTUDIO_NO_GL", "1")


@pytest.fixture(scope="module")
def qapp():
    qtw = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    return qtw.QApplication.instance() or qtw.QApplication([])


def _ctrl(project: Project | None = None):
    from apps.vdstudio.ui.controller import ProjectController

    ctrl = ProjectController(project)
    ctrl.recompute()
    ctrl._debounce.stop()               # no swept-KPI job: these pages need the static set
    return ctrl


def _tab(t) -> list[list[str]]:
    return [[t.item(i, j).text() if t.item(i, j) else "" for j in range(t.columnCount())]
            for i in range(t.rowCount())]


@pytest.fixture()
def man_page(qapp):
    from apps.vdstudio.ui.pages.maneuver import ManeuverPage

    ctrl = _ctrl()
    page = ManeuverPage(ctrl)
    page.job.request = lambda *a, **k: None   # the ay sweep is not under test here
    page.on_geometry(ctrl.geometry)
    yield page
    ctrl.shutdown()


def test_ui_p14_legs_follow_the_state_and_share_its_outer_load(man_page) -> None:
    import sla_geometry as sla

    page = man_page
    wheels = {r[0]: r for r in _tab(page.wheels)}
    legs = _tab(page.legs)
    assert legs[0][0] == "Front / Cornering" and legs[3][0] == "Rear / Cornering"
    assert legs[0][5] == wheels["FR"][1] and legs[3][5] == wheels["RR"][1]   # one outer load
    # cornering legs scale with Fz from sla's own case (LCA -1337 at Fz 1233.6)
    geo = page.ctrl.geometry
    ag, veh = geo.design.front, geo.vehicle_data
    lc = sla.load_cases(ag, veh)
    ref = sla.upright_reactions(ag)
    F_lbj, _ = sla.solve_ball_joint_forces(ag, lc["Cornering"])
    y_l = float(ag.lca_in[0])
    fl = sla.leg_forces(ref["lbj"], np.array([ag.lca_in_front_x_mm, y_l, ag.lca_in_front_z_mm]),
                        np.array([ag.lca_in_rear_x_mm, y_l, ag.lca_in_rear_z_mm]), F_lbj)
    fz = maneuver.run_maneuver(geo, page._inputs(), page._rc()).wheels["FR"].fz_N
    assert float(legs[0][1]) == pytest.approx(fl[0] * fz / lc["Fz"], abs=0.5)
    assert "ay +1.50 g" in page.legs_note.text() and "LLTD 0.472" in page.legs_note.text()
    page.inp["ay_g"].setValue(0.5)
    legs2 = _tab(page.legs)
    wheels2 = {r[0]: r for r in _tab(page.wheels)}
    assert legs2[0][5] == wheels2["FR"][1] != legs[0][5]      # was fixed at 1234
    assert "ay +0.50 g" in page.legs_note.text()


def test_ui_p03_unbalanced_cases_are_not_printed(man_page) -> None:
    legs = {r[0]: r for r in _tab(man_page.legs)}
    for name in ("Front", "Rear"):
        assert legs[f"{name} / Cornering"][1].lstrip("-").isdigit()
        for case in ("Braking", "Combined"):
            row = legs[f"{name} / {case}"]
            assert row[1].startswith("not computed: two-force UCA cannot react the Fx moment")
            assert row[2:5] == ["", "", ""]
    assert "out-of-plane LBJ load (≈ Fz) assumed taken by the pushrod; not shown" in \
        man_page.legs_note.text()


def test_ui_p11_assumptions_are_stated(man_page) -> None:
    from PySide6 import QtWidgets

    texts = " ".join(lab.text() for lab in man_page.findChildren(QtWidgets.QLabel))
    assert mass.UNSPRUNG_ASSUMPTION in texts
    assert "pitch rotates about the CG station" in texts


def test_ui_p16_rendering_of_zero_ay_and_rigid(man_page) -> None:
    page = man_page
    page.inp["ay_g"].setValue(0.0)
    s = dict((r[0], r[1]) for r in _tab(page.summary))
    assert s["Resulting LLTD (front share, any ay)"] == "0.472"    # was 'nan'
    page.inp["ay_g"].setValue(1.5)
    page.inp["roll_gradient_deg_per_g"].setValue(0.0)
    s = dict((r[0], r[1]) for r in _tab(page.summary))
    k = next(v for key, v in s.items() if key.startswith("Total roll stiffness implied"))
    assert k == "rigid (gradient 0)"                                # was 'inf N·m/deg'


def test_ui_p15_rows_beyond_design_travel_are_greyed(man_page) -> None:
    from apps.vdstudio.ui import theme

    page = man_page
    page.inp["roll_gradient_deg_per_g"].setValue(5.0)
    muted = theme.current().text_muted.lower()
    for i in range(page.wheels.rowCount()):
        assert page.wheels.item(i, 3).foreground().color().name() == muted
    assert "outside the front design range" in page.warn.label.text()


def test_ui_p04_failure_clears_every_table(man_page, monkeypatch) -> None:
    from apps.vdstudio.ui.pages import maneuver as page_mod

    page = man_page
    assert page.wheels.rowCount() == 4

    def boom(*_a, **_k):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(page_mod, "run_maneuver", boom)
    page.inp["ay_g"].setValue(2.0)
    assert page.warn.label.text().startswith("Failed: synthetic failure")
    for t in (page.summary, page.wheels, page.transfer, page.legs):
        assert t.rowCount() == 0                                    # no stale ay=1.5 numbers


@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
def test_ui_p02_failed_static_rc_shows_the_reason(qapp) -> None:
    from apps.vdstudio.ui.pages.maneuver import ManeuverPage

    ctrl = _ctrl(_parallel_front_arms())
    page = ManeuverPage(ctrl)
    page.job.request = lambda *a, **k: None
    page.on_geometry(ctrl.geometry)
    txt = page.warn.label.text()
    assert txt.startswith("Failed: static roll centre unavailable")
    assert "<br>front.rc_static: " in txt and "FVIC is at infinity" in txt
    assert "All four wheels loaded" not in txt
    assert page.wheels.rowCount() == 0
    ctrl.shutdown()


def test_ui_p08_p19_concept_page_text(qapp) -> None:
    from apps.vdstudio.ui.pages.concept import ConceptPage

    ctrl = _ctrl()
    page = ConceptPage(ctrl)
    page.on_project()
    txt = page.cg_label.text()
    # default table 230 kg / 288.87 mm against vehicle data 315 kg / 320 mm
    assert "Δ vs vehicle data: -85.0 kg, CG height -31.1 mm" in txt
    assert "h = ΔW·T / (W·tan θ)" in txt and "(N_low − N_high)·T / (2·W·sin θ)" in txt
    assert "sin θ)." in txt and "h = ΔW·T / (W·sin θ)" not in txt
    ctrl.project.mass_items.append(MassItem("bad", -5.0, 100.0))
    page._update()
    assert "CG estimate not computed" in page.cg_label.text()
    ctrl.shutdown()


def test_ui_p20_report_survives_a_concept_failure(qapp, monkeypatch) -> None:
    from apps.vdstudio.ui.pages import reports as rep_mod

    ctrl = _ctrl()
    page = rep_mod.ReportsPage(ctrl)

    def boom(_p):
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(rep_mod, "concept_kpis", boom)
    page._refresh_report()                                          # used to raise in the slot
    assert "concept-stage numbers could not be computed" in page._md
    assert "Concept-stage numbers" not in page._md          # the section is left out
    ctrl.shutdown()
