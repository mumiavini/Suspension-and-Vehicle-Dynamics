"""Rocker / damper / springs on the project (engine/suspension.py)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine import suspension as S
from apps.vdstudio.engine.legs import leg_cases
from apps.vdstudio.engine.maneuver import ManeuverInputs, run_maneuver
from apps.vdstudio.engine.project import (
    DamperSpec,
    Project,
    project_from_dict,
    project_to_dict,
)

FRONT = dict(mount=(9.0, 542.0, 145.0), pivot=(20.0, 180.0, 535.0), toward=(560.0, 175.0, 500.0))


@pytest.fixture(scope="module")
def setup():
    p = Project()
    g = G.resolve(p)
    st = K.compute_static(g)
    rc = {"front": st.value("front.rc_static"), "rear": st.value("rear.rc_static")}
    base = S.build_rocker(FRONT["mount"], FRONT["pivot"], FRONT["toward"], 50, 60, 195)
    spec = S.tune_rocker(g, p, "front", base, 0.80, 50.0, FRONT["toward"])
    rear = S.tune_rocker(g, p, "rear", S.build_rocker((-1530.0, 523.0, 145.0),
                                                       (-1500.0, 160.0, 420.0),
                                                       (-1250.0, 120.0, 440.0), 50, 60, 195),
                         0.80, 50.0, (-1250.0, 120.0, 440.0))
    p = replace(p, rocker_front=spec, rocker_rear=rear)
    return p, g, rc, S.suspension_report(g, p, rc)


def test_builder_places_tangential_arms() -> None:
    sp = S.build_rocker(FRONT["mount"], FRONT["pivot"], FRONT["toward"], 50, 60, 195)
    o, a = np.array(sp.rocker_pivot), np.array(sp.rocker_axis)
    rr, rd, m = np.array(sp.rocker_rod), np.array(sp.rocker_damper), np.array(sp.rod_mount)
    assert np.linalg.norm(rr - o) == pytest.approx(50, abs=1e-3)
    assert np.linalg.norm(rd - o) == pytest.approx(60, abs=1e-3)
    assert abs((rr - o) @ (m - rr)) < 1e-2 * np.linalg.norm(m - rr)   # arm ⟂ rod
    assert abs(a @ (rr - o)) < 5e-3 and abs(a @ (rd - o)) < 5e-3     # axis stored to 1e-4
    assert np.linalg.norm(np.array(sp.damper_chassis) - rd) == pytest.approx(195, abs=1e-3)


def test_tuned_seed_hits_its_targets(setup) -> None:
    _p, _g, _rc, rep = setup
    for a in (rep.front, rep.rear):
        assert a.ok, a.error
        assert a.mr_ride == pytest.approx(0.80, abs=1e-6)
        assert a.droop_margin_mm == pytest.approx(a.bump_margin_mm, abs=0.05)
        assert a.stroke_used_mm == pytest.approx(a.damper_droop_mm - a.damper_bump_mm)
        assert a.mr_droop < a.mr_ride < a.mr_bump                    # this seed: progressive
        assert a.clearance_mm > 20.0


def test_rates_and_static_loads(setup) -> None:
    p, g, _rc, rep = setup
    a = rep.front
    fitted = [o for o in a.springs if o.fitted][0]
    k = S.n_mm(p.springs.front_lb_in)
    assert fitted.wheel_rate_n_mm == pytest.approx(k * a.mr_ride ** 2)
    assert fitted.frequency_hz == pytest.approx(
        math.sqrt(fitted.wheel_rate_n_mm * 1000 / a.sprung_corner_kg) / (2 * math.pi))
    t = g.track_mm("front") / 1000
    assert fitted.roll_rate_nm_deg == pytest.approx(
        0.5 * fitted.wheel_rate_n_mm * 1000 * t * t * math.pi / 180)
    # the spring carries the sprung corner weight through the MR (virtual work)
    assert a.spring_force_ride_N * a.mr_ride == pytest.approx(a.sprung_corner_kg * K.GRAVITY)
    assert a.spring_compression_ride_mm == pytest.approx(a.spring_force_ride_N / k)
    assert {o.lb_in for o in a.springs} == {220.0, 550.0}
    assert S.n_mm(220.0) == pytest.approx(38.528, abs=1e-3)


def test_roll_budget_closes(setup) -> None:
    p, _g, rc, rep = setup
    b = rep.budget
    share = p.springs.roll_stiffness_front_share
    assert b.springs_nm_deg["front"] + b.arb_needed_nm_deg["front"] == pytest.approx(
        share * b.required_nm_deg)
    assert b.springs_nm_deg["rear"] + b.arb_needed_nm_deg["rear"] == pytest.approx(
        (1 - share) * b.required_nm_deg)
    # springs only: the gradient that the springs' total gives, by the same definition
    k = b.springs_nm_deg["front"] + b.springs_nm_deg["rear"]
    wsd = b.required_nm_deg / (1 / b.gradient_deg_per_g + math.pi / 180)
    assert b.gradient_springs_only == pytest.approx(wsd / (k - wsd * math.pi / 180))
    assert b.gradient_springs_only > b.gradient_deg_per_g     # these springs alone roll more


def test_damper_that_does_not_fit_is_reported(setup) -> None:
    p, g, rc, _rep = setup
    short = replace(p, damper=DamperSpec(stroke_mm=30.0))
    a = S.suspension_report(g, short, rc).front
    assert a.ok and a.stroke_used_mm > a.stroke_usable_mm
    assert min(a.droop_margin_mm, a.bump_margin_mm) < 0


def test_disabled_and_failing_layouts(setup) -> None:
    p, g, rc, _rep = setup
    rep = S.suspension_report(g, Project(), rc)
    assert not rep.front.ok and "no rocker" in rep.front.error and rep.budget is None
    locked = replace(p.rocker_front, rocker_rod=(20.0, 145.0, 573.0),
                     rocker_axis=(0.0, 0.73, 0.68), rocker_damper=(80.0, 180.0, 535.0))
    a = S.suspension_report(g, replace(p, rocker_front=locked), rc).front
    assert not a.ok and "locks or separates" in a.error


def test_mirror_layout_gives_the_same_axle(setup) -> None:
    p, g, _rc, _rep = setup
    from vdcore.analysis.rocker import rocker_sweep

    zs = [-20.0, 0.0, 20.0]
    left = rocker_sweep(g.corner("FL"), S.layout_for(p.rocker_front, "FL"), zs)
    right = rocker_sweep(g.corner("FR"), S.layout_for(p.rocker_front, "FR"), zs)
    assert right.motion_ratio == pytest.approx(left.motion_ratio, abs=1e-6)


def test_link_forces_use_the_rocker_pushrod(setup) -> None:
    p, g, rc, _rep = setup
    r = run_maneuver(g, ManeuverInputs(), rc)
    lbj = g.p("FL", "LCA_OUT")
    rod = S.pushrod_for_legs(g, p, "front")
    assert np.allclose(lbj + np.array(rod.inboard_from_lbj_mm), p.rocker_front.rocker_rod)
    old = {(c.corner, c.case): c for c in leg_cases(g, Project(), r)}
    new = {(c.corner, c.case): c for c in leg_cases(g, p, r)}
    assert old.keys() == new.keys()
    assert any(abs(new[k].forces["pushrod"] - old[k].forces["pushrod"]) > 1.0 for k in new)


def test_project_round_trip(setup) -> None:
    p, *_ = setup
    back = project_from_dict(project_to_dict(p))
    assert back.rocker_front == p.rocker_front and back.springs == p.springs
    d = project_to_dict(Project())
    for key in ("rocker_front", "rocker_rear", "damper", "springs"):
        del d[key]
    old = project_from_dict(d)
    assert not old.rocker_front.enabled and old.damper.eye_to_eye_extended_mm == 220.0


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("VDSTUDIO_NO_GL", "1")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_rocker_page(qapp, setup) -> None:
    import time

    from PySide6 import QtWidgets

    from apps.vdstudio.ui.controller import ProjectController
    from apps.vdstudio.ui.main_window import NAV
    from apps.vdstudio.ui.pages.rocker import RockerPage

    assert ("Design", "rocker", "Pushrod & rocker", "rocker", "RockerPage") in NAV
    p, *_ = setup
    ctrl = ProjectController(p)
    ctrl.recompute()
    page = RockerPage(ctrl)

    def wait(cond) -> None:
        t = time.time()
        while time.time() - t < 30 and not cond():
            QtWidgets.QApplication.processEvents()
            time.sleep(0.01)

    try:
        page.show()
        page.on_project()
        wait(lambda: page.report is not None)
        rows = {page.table.item(i, 0).text(): (page.table.item(i, 1).text(),
                                               page.table.item(i, 2).text())
                for i in range(page.table.rowCount())}
        assert rows["Motion ratio at ride (damper / wheel)"] == ("0.800", "0.800")
        assert page.rates.rowCount() == 2 and page.budget.rowCount() == 6
        assert "ESTIMATE" in page.warn.label.text()
        # the construction helper: a new target MR, saved and undoable
        page.report = None
        page.seed["front"]["mr"].setValue(0.9)
        page._rebuild("front")
        page.on_project()
        wait(lambda: page.report is not None)
        assert page.report.front.mr_ride == pytest.approx(0.9, abs=1e-4)
        ctrl.undo()
        assert ctrl.project.rocker_front == p.rocker_front
        # available springs typed as text
        page.avail.setText("220, 400, 550")
        page._set_available()
        assert ctrl.project.springs.available_lb_in == (220.0, 400.0, 550.0)
    finally:
        ctrl.shutdown()
        page.close()
