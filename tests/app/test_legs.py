"""Link forces at a maneuver state (engine.legs): load cases, load points, provenance."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine import maneuver
from apps.vdstudio.engine.legs import LINK_ORDER, leg_cases, pushrod_line
from apps.vdstudio.engine.project import Project, project_from_dict, project_to_dict
from vdcore.analysis.upright_loads import upright_link_forces


def _state(p: Project, **kw):
    geo = G.resolve(p)
    st = K.compute_static(geo)
    rc = {a: st.value(f"{a}.rc_static") for a in ("front", "rear")}
    return geo, maneuver.run_maneuver(geo, maneuver.ManeuverInputs(**kw), rc)


def _by(cases):
    return {(c.axle, c.case): c for c in cases}


def test_every_case_is_solved_on_the_outer_wheel() -> None:
    p = Project()
    geo, r = _state(p)
    cases = _by(leg_cases(geo, p, r))
    assert set(cases) == {("front", "Cornering"), ("front", "Braking"), ("front", "Combined"),
                          ("rear", "Cornering"), ("rear", "Braking"), ("rear", "Combined"),
                          ("rear", "Traction")}
    for c in cases.values():
        assert not c.error and set(c.forces) == set(LINK_ORDER)
    fr = cases[("front", "Cornering")]
    assert fr.corner == "FR" and fr.fz_N == pytest.approx(r.wheels["FR"].fz_N)
    # left turn: the road pushes the outer (right) tyre toward +Y; braking is -X
    assert fr.load_N[1] > 0 and cases[("front", "Braking")].load_N[0] < 0
    assert cases[("rear", "Traction")].load_N[0] > 0
    # a right turn mirrors the outer wheel and the lateral force
    geo_r, r_r = _state(p, ay_g=-1.5)
    fl = _by(leg_cases(geo_r, p, r_r))[("front", "Cornering")]
    assert fl.corner == "FL" and fl.load_N[1] < 0
    for k in LINK_ORDER:
        assert fl.forces[k] == pytest.approx(fr.forces[k], abs=1e-6)


def test_inboard_brakes_move_fx_to_the_wheel_centre() -> None:
    p = Project()
    q = replace(p, layout=replace(p.layout, front_brakes_inboard=True))
    geo, r = _state(q)
    c = _by(leg_cases(geo, q, r))[("front", "Braking")]
    assert c.fx_point == "wheel centre"
    pr_o, pr_i = pushrod_line(geo, c.corner, q.pushrod_front)
    # Fx at the wheel centre, Fz at the patch: superposition of the linear solve
    a = upright_link_forces(geo.corner(c.corner), (c.load_N[0], 0, 0),
                            geo.p(c.corner, "WHEEL_CENTER"), pr_o, pr_i).as_dict()
    b = upright_link_forces(geo.corner(c.corner), (0, 0, c.load_N[2]),
                            geo.p(c.corner, "CONTACT_PATCH"), pr_o, pr_i).as_dict()
    for k in LINK_ORDER:
        assert c.forces[k] == pytest.approx(a[k] + b[k], abs=1e-9)
    outboard = _by(leg_cases(*_state(p)[:1], p, r))[("front", "Braking")]
    assert outboard.forces["pushrod"] != pytest.approx(c.forces["pushrod"], abs=1.0)


def test_hardpoint_mode_is_supported() -> None:
    p = Project()
    p.hardpoints = G.freeze_to_hardpoints(p)
    p.geometry_mode = "hardpoints"
    geo, r = _state(p)
    p0 = Project()
    g0, r0 = _state(p0)
    synth = _by(leg_cases(g0, p0, r0))
    for key, c in _by(leg_cases(geo, p, r)).items():
        for k in LINK_ORDER:
            assert c.forces[k] == pytest.approx(synth[key].forces[k], abs=1e-6)


def test_pushrod_is_an_estimate_that_follows_the_lbj_and_round_trips() -> None:
    p = Project()
    assert p.pushrod_front.source == "estimate"
    geo = G.resolve(p)
    o, i = pushrod_line(geo, "FR", p.pushrod_front)
    lbj = geo.p("FR", "LCA_OUT")
    assert np.allclose(o, lbj) and np.allclose(i, lbj + [0.0, 250.0, 300.0])   # mirrored Y
    q = replace(p, pushrod_rear=replace(p.pushrod_rear, inboard_from_lbj_mm=(10.0, -200.0, 350.0)))
    back = project_from_dict(project_to_dict(q))
    assert back.pushrod_rear.inboard_from_lbj_mm == (10.0, -200.0, 350.0)
    d = project_to_dict(p)
    del d["pushrod_front"]
    assert project_from_dict(d).pushrod_front == p.pushrod_front     # old files load


def test_an_unloaded_outer_wheel_is_not_computed() -> None:
    p = Project()
    geo, r = _state(p, ay_g=0.0, ax_g=0.0)
    # both front wheels unloaded (lifted): nothing to solve on that axle
    r.wheels["FR"] = replace(r.wheels["FR"], fz_N=-5.0)
    r.wheels["FL"] = replace(r.wheels["FL"], fz_N=-10.0)
    c = _by(leg_cases(geo, p, r))[("front", "Braking")]
    assert c.error == "not computed: outer wheel not loaded" and not c.forces
