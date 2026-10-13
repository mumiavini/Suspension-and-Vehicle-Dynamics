"""VD Studio: what-if engine, trade-off page, controller and widget fixes.

Each test pins one behaviour from the what-if / UI audit:

* a target query finds both crossings of a V-shaped output, reports failed
  scan points and failed refinements, and never calls a target unreachable;
* a root is applied exactly as the query evaluated it (mirrored hardpoints
  move on both sides), one Apply button per root, nothing pre-selected;
* the registry carries the hardpoint-mode alignment inputs and the rear toe;
* a front/rear row never shows the front band beside the rear value;
* sweep inputs keep their precision on screen and in the CSV;
* a swept-KPI failure replaces the old values instead of leaving them on
  screen as current, and the status bar never says 'up to date' early;
* the Altair cross-check builds each axle from its own settings;
* the sensitivity matrix does not average a kink away;
* the main window fits a 1366x768 screen.
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apps.vdstudio.engine import whatif as W
from apps.vdstudio.engine.geometry import freeze_to_hardpoints, resolve
from apps.vdstudio.engine.kpis import compute_static
from apps.vdstudio.engine.project import Project

REPO = Path(__file__).resolve().parents[2]
NAN = float("nan")


# --------------------------------------------------------------------------- #
# Engine: target query (no Qt)
# --------------------------------------------------------------------------- #

def _fake_output(monkeypatch, fn, base: float = 0.0) -> None:
    """Replace the solver behind the what-if engine with ``output = fn(input)``;
    ``base`` is the input of the unmodified design."""

    def ev(project, params, values, outputs, groups):
        y = float(fn(float(values[0]) if values else base))
        ok = math.isfinite(y)
        errs = {} if ok else {o: "forced failure" for o in outputs}
        return W.PointResult(tuple(values), {o: y for o in outputs},
                             {o: ("info" if ok else "fail") for o in outputs},
                             "; ".join(f"{o}: {e}" for o, e in errs.items()), errs)

    monkeypatch.setattr(W, "evaluate", ev)
    monkeypatch.setattr(W, "groups_for", lambda keys: set())


def test_target_query_finds_both_crossings_inside_one_scan_interval(monkeypatch) -> None:
    # |x - 5| scanned at 0, 4.5, 9: 5, 0.5, 4 -- no sign change for a target of 0.25
    _fake_output(monkeypatch, lambda x: abs(x - 5.0))
    r = W.target_query(Project(), "p", "y", 0.25, 0.0, 9.0, n_scan=3)
    assert r.roots == pytest.approx([4.75, 5.25], abs=2e-3)
    assert "not reachable" not in r.message


def test_target_query_reports_a_tangent_touch_as_a_root(monkeypatch) -> None:
    _fake_output(monkeypatch, lambda x: (x - 5.0) ** 2)
    r = W.target_query(Project(), "p", "y", 0.0, 0.0, 9.0, n_scan=3)
    assert r.roots == pytest.approx([5.0], abs=5e-3)


def test_target_query_without_a_root_states_what_was_searched(monkeypatch) -> None:
    _fake_output(monkeypatch, lambda x: abs(x - 5.0) + 1.0)
    r = W.target_query(Project(), "p", "y", 0.5, 0.0, 9.0, n_scan=3)
    assert r.roots == []
    assert ("No sign change of output − target between the 3 scan points (spacing 4.5)"
            in r.message)
    assert "a pair of crossings closer than the spacing would be missed" in r.message
    # the refined extremum, as information (minimum 1 at 5)
    assert re.search(r"closest: output 1\.000\d* at 5\.000", r.message)
    assert "not reachable" not in r.message


def test_target_query_reports_failed_scan_points(monkeypatch) -> None:
    _fake_output(monkeypatch, lambda x: x - 5.0 if x < 7.0 else NAN)
    r = W.target_query(Project(), "p", "y", 0.0, 0.0, 9.0, n_scan=4)
    assert r.roots == pytest.approx([5.0], abs=2e-3)
    assert r.failed_x == [9.0]
    assert "1 scan point(s) did not solve (9)" in r.message
    assert r.scan_errors[3] == "y: forced failure"
    assert r.scan_errors[0] == ""


def test_target_query_reports_a_bracket_it_could_not_refine(monkeypatch) -> None:
    # solves at every scan point (0, 4.5, 9) but not near the crossing at 5
    _fake_output(monkeypatch, lambda x: NAN if 4.9 < x < 5.1 else x - 5.0)
    r = W.target_query(Project(), "p", "y", 0.0, 0.0, 9.0, n_scan=3)
    assert r.roots == []
    assert r.failed_x == []
    assert len(r.unrefined_brackets) == 1
    a, b, why = r.unrefined_brackets[0]
    assert (a, b) == (4.5, 9.0)
    assert "refinement failed" in why and "did not solve" in why
    assert "could not be refined" in r.message
    assert "not reachable" not in r.message


def test_target_query_finds_both_bump_steer_peak_crossings_on_the_real_car() -> None:
    # peak |toe| over the bump travel has its minimum (~0.16 deg) near the default rack
    # height; both 0.25 deg crossings sit inside one 7.5 mm scan interval
    r = W.target_query(Project(), "steering.rack_z_mm", "front.bump_steer_peak", 0.25,
                       150.0, 165.0, n_scan=3)
    assert r.roots == pytest.approx([155.875, 158.854], abs=0.01)


def test_evaluate_names_each_failed_output() -> None:
    outs = ["front.camber_gain", "front.rc_static"]
    pt = W.evaluate(Project(), ["front.travel_bump_mm"], [400.0], outs,
                    sorted(W.groups_for(outs)))
    assert not math.isfinite(pt.values["front.camber_gain"])
    assert math.isfinite(pt.values["front.rc_static"])
    assert set(pt.errors) == {"front.camber_gain"}
    assert pt.error.startswith("front.camber_gain: ")
    assert pt.errors["front.camber_gain"] in pt.error
    assert not pt.fully_solved


# --------------------------------------------------------------------------- #
# Engine: registry and sensitivity
# --------------------------------------------------------------------------- #

def test_registry_labels_the_wheelbase_and_adds_the_rear_toe() -> None:
    wb = W.param_by_path("vehicle.wheelbase_mm")
    assert wb.label == "Wheelbase (moves the rear axle; CG station kept)"
    assert wb.mode == "synthesis"
    paths = {p.path for p in W.params_for(Project())}
    assert "rear_static_toe_deg_per_side" in paths
    assert "rear.axle_x_mm" not in {p.path for p in W.all_params()}
    q = W.param_by_path("rear_static_toe_deg_per_side").apply(Project(), 0.15)
    assert resolve(q).axles["rear"].static_toe_deg_per_side == pytest.approx(0.15)


@pytest.mark.parametrize(("field", "step", "value"), [
    ("static_camber_deg", 0.25, -2.0),
    ("static_toe_deg_per_side", 0.05, 0.1),
    ("loaded_radius_mm", 2.0, 240.0),
])
def test_hardpoint_mode_sweeps_alignment_and_tyre_radius(field: str, step: float,
                                                        value: float) -> None:
    p = Project()
    p.hardpoints = freeze_to_hardpoints(p)
    p.geometry_mode = "hardpoints"
    params = {x.path: x for x in W.params_for(p)}
    for axle in ("front", "rear"):
        par = params[f"hardpoints.{axle}.{field}"]
        assert par.step == step and par.mode == "hardpoints" and not par.mirror
        before = par.get(p)
        q = par.apply(p, value)
        assert par.get(q) == value
        assert par.get(p) == before                    # the queried project is a copy
        assert getattr(resolve(q).axles[axle], field) == pytest.approx(value)
    assert "vehicle.wheelbase_mm" not in params       # the hardpoints carry the wheelbase


def test_sensitivity_keeps_both_one_sided_values(monkeypatch) -> None:
    _fake_output(monkeypatch, lambda x: abs(x - 1.0), base=1.0)
    monkeypatch.setattr(W, "param_by_path", lambda path: SimpleNamespace(
        step=0.5, get=lambda project: 1.0, apply=lambda project, v: project))
    s = W.sensitivity(Project(), ["p"], ["y"])
    assert s.up["y"]["p"] == pytest.approx(0.5) and s.dn["y"]["p"] == pytest.approx(0.5)
    assert s.per_step("y", "p") == pytest.approx(0.0)  # the central value hides the kink
    assert s.one_sided("y", "p") == pytest.approx((0.5, 0.5))
    assert s.is_kink("y", "p")


def test_sensitivity_kink_flag() -> None:
    def res(base, up, dn):
        return W.SensitivityResult(["p"], ["y"], {"y": base}, {"p": 2.0},
                                   {"y": {"p": (up - dn) / 4.0}}, {"y": {"p": up}},
                                   {"y": {"p": dn}})

    # the audit's case: peak |toe| about the default rack height
    assert res(0.159830, 0.283135, 0.279125).is_kink("y", "p")
    assert not res(35.5, 40.5, 30.5).is_kink("y", "p")          # smooth
    assert res(10.0, 10.9, 9.99).is_kink("y", "p")              # very different sizes
    assert not res(1.0, NAN, 0.9).is_kink("y", "p")


# --------------------------------------------------------------------------- #
# Docs and the Altair runner (no Qt)
# --------------------------------------------------------------------------- #

def test_frames_reference_calls_the_solidworks_fsae_frame_right_handed() -> None:
    text = (REPO / ".claude/skills/vd-conventions/references/frames.md").read_text("utf-8")
    assert "| **SolidWorks (typical FSAE)** | Rearward | Right | Up | Right |" in text
    assert "(right-handed: a 180° rotation about Z" in text
    assert "Z+ up (left-handed). Negate X and Y." not in text
    assert np.linalg.det(np.diag([-1.0, -1.0, 1.0])) == pytest.approx(1.0)


def _spy_altair(monkeypatch):
    import altair_model.kpi_runner as KR

    built: list[tuple[str, float, float | None, float | None]] = []
    runs: dict[str, tuple[float, float]] = {}
    real_build = KR.build_corner

    def build(cid, points, static_toe_deg=0.0, *, static_camber_deg=None,
              loaded_radius_mm=None):
        built.append((cid, static_toe_deg, static_camber_deg, loaded_radius_mm))
        return real_build(cid, points, static_toe_deg, static_camber_deg=static_camber_deg,
                          loaded_radius_mm=loaded_radius_mm)

    def axle_result(axle, label, csv_path, work, *, roll_deg, travel_mm, sweep_steps,
                    roll_travels):
        runs[label] = (roll_deg, travel_mm)
        return SimpleNamespace(values={})

    monkeypatch.setattr(KR, "build_corner", build)
    monkeypatch.setattr(KR, "axle_roll", lambda axle, deg: SimpleNamespace(wheel_travel_mm=0.0))
    monkeypatch.setattr(KR, "_axle_result", axle_result)
    return KR, built, runs


def test_altair_runner_builds_each_axle_from_its_own_settings(monkeypatch, tmp_path) -> None:
    from apps.vdstudio.engine import exports as X

    KR, built, runs = _spy_altair(monkeypatch)
    path = tmp_path / "hp.csv"
    path.write_text(X.hardpoints_csv(resolve(Project()), "iso"), encoding="utf-8")
    per_axle = {
        "front": {"static_toe_deg": 0.0, "static_camber_deg": -1.5, "loaded_radius_mm": 245.0,
                  "roll_deg": 1.5, "travel_mm": 25.0},
        "rear": {"static_toe_deg": 0.15, "static_camber_deg": -1.0, "loaded_radius_mm": 250.0,
                 "roll_deg": 1.0, "travel_mm": 30.0},
    }
    KR.altair_kpis_from_csv(path, per_axle=per_axle, workdir=tmp_path)
    by_corner = {c: rest for c, *rest in built}
    assert by_corner["FL"] == by_corner["FR"] == [0.0, -1.5, 245.0]
    assert by_corner["RL"] == by_corner["RR"] == [0.15, -1.0, 250.0]
    assert runs == {"Front": (1.5, 25.0), "Rear": (1.0, 30.0)}

    # the old keyword signature still builds both axles from one setting
    built.clear()
    runs.clear()
    KR.altair_kpis_from_csv(path, roll_deg=2.0, travel_mm=20.0, static_toe_deg=0.1,
                            static_camber_deg=-2.0, loaded_radius_mm=240.0, workdir=tmp_path)
    assert {tuple(rest) for _, *rest in built} == {(0.1, -2.0, 240.0)}
    assert runs == {"Front": (2.0, 20.0), "Rear": (2.0, 20.0)}
    with pytest.raises(ValueError, match="unknown axle or setting"):
        KR.altair_kpis_from_csv(path, per_axle={"rear": {"toe": 0.1}}, workdir=tmp_path)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VDSTUDIO_NO_GL", "1")


@pytest.fixture(scope="module")
def app():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    pytest.importorskip("pyqtgraph")
    from apps.vdstudio.ui import theme

    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(a, theme.DARK)
    return a


def _pump(app, seconds: float, until=None) -> None:
    t = time.time()
    while time.time() - t < seconds:
        app.processEvents()
        time.sleep(0.01)
        if until is not None and until():
            return


@pytest.fixture(scope="module")
def window(app):
    from apps.vdstudio.ui.controller import ProjectController
    from apps.vdstudio.ui.main_window import MainWindow

    ctrl = ProjectController()
    win = MainWindow(ctrl)
    win.resize(1366, 768)
    win.show()
    ctrl.recompute()
    _pump(app, 60, lambda: not ctrl.dynamic_stale)
    yield win
    for p in win.pages.values():
        if hasattr(p, "shutdown"):
            p.shutdown()
    ctrl.shutdown()
    ctrl._set_dirty(False)
    win.close()


@pytest.fixture()
def page(app):
    """A trade-off page on its own controller (synthesis, default car)."""
    from apps.vdstudio.ui.controller import ProjectController
    from apps.vdstudio.ui.pages.tradeoff import TradeoffPage

    ctrl = ProjectController()
    ctrl.new(Project())
    pg_ = TradeoffPage(ctrl)
    pg_.resize(1300, 800)
    pg_.show()
    _pump(app, 0.2)
    yield pg_
    pg_.shutdown()
    ctrl.shutdown()
    pg_.close()


def _click_in_next_modal(app, startswith: str, seen: list[str]) -> None:
    from PySide6 import QtCore, QtWidgets

    def poll() -> None:
        box = QtWidgets.QApplication.activeModalWidget()
        if box is None:
            QtCore.QTimer.singleShot(50, poll)
            return
        seen.extend(b.text() for b in box.buttons())
        for b in box.buttons():
            if b.text().startswith(startswith):
                b.click()
                return

    QtCore.QTimer.singleShot(50, poll)


def _stamped(page, query: str = "test query") -> None:
    from apps.vdstudio.ui.pages import tradeoff as TO

    page._run_stamp = TO.project_stamp(page.ctrl.project)
    page._pending_sha = TO.project_sha1(page.ctrl.project)
    page._run_query = query


def _table(page) -> list[list[str]]:
    t = page.table
    return [[t.item(i, j).text() for j in range(t.columnCount())] for i in range(t.rowCount())]


def test_main_window_fits_a_1366x768_screen(app, window) -> None:
    w, h = window.minimumSizeHint().toTuple()
    assert w <= 1366 and h <= 768, (w, h)
    for key in window.pages:                 # no page forces the window larger
        window.goto(key)
        _pump(app, 0.05)
        w, h = window.minimumSizeHint().toTuple()
        assert w <= 1366 and h <= 768, (key, w, h)
    assert window.size().toTuple() == (1366, 768)
    window.goto("dashboard")


def test_dashboard_cards_keep_their_unit_and_say_per_side(app, window) -> None:
    from apps.vdstudio.ui.pages.dashboard import CARDS

    labels = dict((k, lab) for k, lab, _ in CARDS)
    assert labels["front.bump_steer_peak"] == "Front bump steer (peak, per side)"
    assert labels["rear.bump_steer_peak"] == "Rear bump steer (peak, per side)"
    card = window.pages["dashboard"].cards["front.camber_gain"]
    assert card.lab.wordWrap()
    assert card.lab.text().endswith("[deg/mm]")


def test_kpi_table_shows_each_axle_band(app) -> None:
    from apps.vdstudio.ui.widgets import KpiTable, pair_band_text

    ks = compute_static(resolve(Project()))
    text, tip = pair_band_text(ks.get("front.kpi_L"), ks.get("rear.kpi_L"))
    assert text == "F 6 … 14 / R 3 … 10"
    assert "front: 6 … 14 (sla_geometry.CheckLimits" in tip
    assert "rear: 3 … 10 (sla_geometry.CheckLimits" in tip
    text, tip = pair_band_text(ks.get("front.trail_L"), ks.get("rear.trail_L"))
    assert text == "F 10 … 35 / R —"
    assert "rear: no band" in tip
    text, _ = pair_band_text(ks.get("front.rc_static"), ks.get("rear.rc_static"))
    assert text == "20 … 70"                         # one band when both axles share it

    table = KpiTable()
    table.set_kpis(ks)
    rows = {table.item(i, 0).text().strip(): i for i in range(table.rowCount())}
    assert table.item(rows["KPI (L)"], 4).text() == "F 6 … 14 / R 3 … 10"
    assert table.item(rows["Mechanical trail (L)"], 4).text() == "F 10 … 35 / R —"


def test_stale_swept_values_are_greyed_centrally(app) -> None:
    from apps.vdstudio.ui.controller import LiveKpiSet
    from apps.vdstudio.ui.widgets import Card, KpiTable

    ks = compute_static(resolve(Project()))
    live = LiveKpiSet(dict(ks.items), frozenset({"front.rc_static"}))
    table = KpiTable()
    table.set_kpis(live)
    rows = {table.item(i, 0).text().strip(): i for i in range(table.rowCount())}
    i = rows[ks.get("front.rc_static").label]
    assert table.item(i, 1).font().italic()           # stale front value
    assert not table.item(i, 2).font().italic()       # rear value is current
    assert "stale" in table.item(i, 1).toolTip()
    card = Card("Front RC", "mm")
    card.set_kpi(ks.get("front.rc_static"), stale=True)
    assert "(stale)" in card.sub.text()
    assert "italic" in card.value.styleSheet()


def test_swept_kpi_failure_replaces_the_old_values(app, window, monkeypatch) -> None:
    import apps.vdstudio.engine.kpis as K

    ctrl = window.ctrl
    assert ctrl.kpis.get("steering.ackermann").ok
    assert window.calc_label.text() == "swept KPIs up to date"

    def boom(*_a, **_k):
        raise ZeroDivisionError("forced")

    monkeypatch.setattr(K, "compute_dynamic", boom)
    ctrl.set_value("steering.steer_arm_angle_deg", ctrl.project.steering.steer_arm_angle_deg + 6)
    # right after the edit: the swept values are the previous geometry's, and say so
    assert "up to date" not in window.calc_label.text()
    assert "stale" in window.calc_label.text()
    assert "steering.ackermann" in ctrl.kpis.stale_keys
    _pump(app, 30, lambda: bool(ctrl.dynamic_error))
    assert ctrl.dynamic_error.startswith("swept KPIs failed: ZeroDivisionError: forced")
    ack = ctrl.kpis.get("steering.ackermann")
    assert not ack.ok and math.isnan(ack.value)
    assert ack.error == ctrl.dynamic_error
    assert ctrl.kpis.get("front.rc_static").ok        # static KPIs are this car's
    assert "swept KPIs FAILED" in window.calc_label.text()
    window.goto("dashboard")
    _pump(app, 0.2)
    dash = window.pages["dashboard"]
    assert dash.cards["steering.ackermann"].value.text() == "—"
    assert "Swept KPIs FAILED" in dash.summary.label.text()
    monkeypatch.undo()
    ctrl.undo()
    _pump(app, 60, lambda: not ctrl.dynamic_stale)
    assert ctrl.kpis.get("steering.ackermann").ok
    assert ctrl.dynamic_error == ""
    assert window.calc_label.text() == "swept KPIs up to date"


def test_altair_cross_check_passes_each_axle_its_own_settings(app, window,
                                                              monkeypatch) -> None:
    import altair_model.kpi_runner as KR

    calls: list[dict] = []

    def fake(path, **kw):
        calls.append(kw)
        return {}

    monkeypatch.setattr(KR, "altair_kpis_from_csv", fake)
    ctrl = window.ctrl
    ctrl.set_value("rear.static_camber_deg", -1.0)
    ctrl.set_value("rear_static_toe_deg_per_side", 0.15)
    try:
        window.pages["integrations"]._altair()
        _pump(app, 20, lambda: bool(calls))
    finally:
        ctrl.undo()
        ctrl.undo()
    pa = calls[0]["per_axle"]
    assert pa["front"]["static_camber_deg"] == pytest.approx(-1.5)
    assert pa["front"]["static_toe_deg"] == pytest.approx(ctrl.project.steering
                                                          .static_toe_deg_per_side)
    assert pa["rear"]["static_camber_deg"] == pytest.approx(-1.0)
    assert pa["rear"]["static_toe_deg"] == pytest.approx(0.15)
    for a in ("front", "rear"):
        assert set(pa[a]) == {"static_toe_deg", "static_camber_deg", "loaded_radius_mm",
                              "roll_deg", "travel_mm"}
    _pump(app, 60, lambda: not ctrl.dynamic_stale)


def test_grid_keeps_input_precision_on_screen_and_in_the_csv(page) -> None:
    from apps.vdstudio.ui.pages.tradeoff import axis_decimals

    assert axis_decimals([526.966, 531.966, 536.966]) == 3
    assert axis_decimals([1540.0, 1540.25, 1540.5, 1540.75, 1541.0]) == 2
    axis = np.linspace(1540.0, 1541.0, 5)
    pts = [W.PointResult((float(x),), {"front.anti": 7.4831257735914}, {"front.anti": "in"})
           for x in axis]
    g = W.GridResult(["vehicle.wheelbase_mm"], ["front.anti"], [axis], pts)
    _stamped(page)
    page._done(g)
    shown = [r[0] for r in _table(page)]
    assert shown == ["1540.00", "1540.25", "1540.50", "1540.75", "1541.00"]
    header, *data = [ln for ln in page._csv.splitlines() if not ln.startswith("#")]
    assert [ln.split(",")[0] for ln in data] == ["1540.0", "1540.25", "1540.5", "1540.75",
                                                 "1541.0"]
    assert data[0].split(",")[1] == "7.4831257735914"   # outputs at full precision too
    # the on-screen header wraps; the tooltip and the CSV keep the whole label
    full = "Wheelbase (moves the rear axle; CG station kept) [mm]"
    assert header.startswith(full + ",")
    assert "\n" in page.table.horizontalHeaderItem(0).text()
    assert page.table.horizontalHeaderItem(0).toolTip() == full
    # a constant output is drawn on a readable range, not with 15-digit ticks
    (anti,) = [it for it in page.lines.ci.items if hasattr(it, "viewRange")]
    lo, hi = anti.viewRange()[1]
    assert hi - lo == pytest.approx(2e-3 * 7.4831257735914, rel=1e-6)


def test_grid_reports_per_output_failures_and_stamps_the_csv(page) -> None:
    outs = ["front.rc_static", "front.camber_gain", "front.rc_span"]
    axis = np.array([30.0, 1030.0, 2030.0])
    pts = [
        W.PointResult((30.0,), {"front.rc_static": 38.519, "front.camber_gain": NAN,
                                "front.rc_span": NAN},
                      {"front.rc_static": "in", "front.camber_gain": "fail",
                       "front.rc_span": "fail"},
                      "front.camber_gain: FL did not converge; front.rc_span: no IC",
                      {"front.camber_gain": "FL did not converge", "front.rc_span": "no IC"}),
        W.PointResult((1030.0,), {"front.rc_static": 35.476, "front.camber_gain": -0.04,
                                  "front.rc_span": NAN},
                      {"front.rc_static": "in", "front.camber_gain": "in",
                       "front.rc_span": "fail"},
                      "front.rc_span: no IC", {"front.rc_span": "no IC"}),
        W.PointResult((2030.0,), {"front.rc_static": 35.514, "front.camber_gain": -0.03,
                                  "front.rc_span": NAN},
                      {"front.rc_static": "in", "front.camber_gain": "in",
                       "front.rc_span": "fail"},
                      "front.rc_span: no IC", {"front.rc_span": "no IC"}),
    ]
    g = W.GridResult(["front.fvsa_length_mm"], outs, [axis], pts)
    _stamped(page, "grid sweep: front.fvsa_length_mm")
    page._done(g)
    assert page.status.text().startswith(
        "0 of 3 combinations fully solved; 3 with failed outputs (—)")
    rows = _table(page)
    assert rows[0][2] == "—" and rows[0][-1] == ("front.camber_gain: FL did not converge; "
                                                 "front.rc_span: no IC")
    assert page.table.item(0, 2).toolTip() == "failed: FL did not converge"
    lines = page._csv.splitlines()
    assert lines[0].startswith(f"# project {page.ctrl.project.name}, geometry mode synthesis, "
                               "sha1 ")
    assert "(project JSON), computed 20" in lines[0]
    assert lines[1] == "# grid sweep: front.fvsa_length_mm"
    # an all-failed output is not drawn (no RuntimeWarning, no empty line) but named
    plots = [it for it in page.lines.ci.items if hasattr(it, "listDataItems")]
    span = [p for p in plots if "every point failed" in p.titleLabel.text.replace("<br>", " ")]
    assert len(span) == 1 and span[0].listDataItems() == []


def test_query_shows_roots_in_the_table_and_csv_and_clears_old_plots(app, page) -> None:
    # a grid result first, so its curves exist
    axis = np.array([1.0, 2.0])
    page._done(W.GridResult(["front.rc_height_mm"], ["front.rc_static"], [axis],
                            [W.PointResult((x,), {"front.rc_static": x}, {"front.rc_static":
                                                                          "info"})
                             for x in axis]))
    assert len(page.lines.ci.items) == 1
    r = W.TargetQueryResult(
        "steering.rack_z_mm", "front.bump_steer_peak", 0.25,
        np.array([150.0, 157.5, 165.0]), np.array([0.3, NAN, 0.4]), [155.8747, 158.8544],
        "2 value(s) ...", [157.5], [(157.5, 165.0, "refinement failed: forced")],
        ["", "front.bump_steer_peak: forced", ""], 1.5e-3)
    seen: list[str] = []
    _click_in_next_modal(app, "Close", seen)
    _stamped(page)
    page._done(r)
    assert seen == ["Apply 155.875 (my decision)", "Apply 158.854 (my decision)", "Close"]
    assert len(page.lines.ci.items) == 0                # the grid's curves are gone
    rows = _table(page)
    assert rows[0][:3] == ["root 1", "155.875", "0.25 (target)"]
    assert rows[1][:3] == ["root 2", "158.854", "0.25 (target)"]
    assert rows[2][0] == "unrefined" and "refinement failed" in rows[2][3]
    assert rows[4][2] == "—" and rows[4][3] == "front.bump_steer_peak: forced"
    assert "\nroot,155.8747,0.25," in page._csv
    assert "\nscan,157.5,nan,front.bump_steer_peak: forced" in page._csv


def test_outputs_box_is_hidden_in_query_mode(page) -> None:
    page.mode.setCurrentIndex(2)
    assert page.out_box.isHidden()
    page.mode.setCurrentIndex(0)
    assert not page.out_box.isHidden()


def test_heat_map_text_is_readable_on_light_and_dark_cells(page) -> None:
    import pyqtgraph as pg

    a, b = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    vals = {(0.0, 0.0): 0.0, (0.0, 1.0): 1.0, (1.0, 0.0): 2.0, (1.0, 1.0): 3.0}
    pts = [W.PointResult((x, y), {"front.rc_static": vals[(x, y)]}, {"front.rc_static": "info"})
           for x in a for y in b]
    page._done(W.GridResult(["front.rc_height_mm", "front.fvsa_length_mm"],
                            ["front.rc_static"], [a, b], pts))
    texts = {t.textItem.toPlainText(): t.color.name() for t in page.heat.getPlotItem().items
             if isinstance(t, pg.TextItem)}
    assert texts["3"] == "#000000"                      # top of viridis: yellow
    assert texts["0"] == "#ffffff"                      # bottom of viridis: purple
    # a result of another kind removes the map, title included
    page._done(W.SensitivityResult(["front.rc_height_mm"], ["front.rc_static"],
                                   {"front.rc_static": 35.5}, {"front.rc_height_mm": 5.0},
                                   {"front.rc_static": {"front.rc_height_mm": 1.0}}))
    pi = page.heat.getPlotItem()
    assert not [t for t in pi.items if isinstance(t, (pg.TextItem, pg.ImageItem))]
    assert not pi.titleLabel.isVisible()
    assert page.heat_out.count() == 0


def test_sensitivity_cell_shows_both_sides_at_a_kink(page) -> None:
    s = W.SensitivityResult(
        ["steering.rack_z_mm", "front.kpi_deg"], ["front.bump_steer_peak"],
        {"front.bump_steer_peak": 0.159830}, {"steering.rack_z_mm": 2.0, "front.kpi_deg": 1.0},
        {"front.bump_steer_peak": {"steering.rack_z_mm": (0.283135 - 0.279125) / 4.0,
                                   "front.kpi_deg": 0.01}},
        {"front.bump_steer_peak": {"steering.rack_z_mm": 0.283135, "front.kpi_deg": 0.169830}},
        {"front.bump_steer_peak": {"steering.rack_z_mm": 0.279125, "front.kpi_deg": 0.149830}})
    page._done(s)
    kink = page.table.item(0, 1)
    assert kink.text() == "+0.123 / +0.119 (kink)"
    assert "Δ+ (input +2): +0.123305" in kink.toolTip()
    assert "Δ− (input −2): +0.119295" in kink.toolTip()
    assert page.table.item(0, 2).text() == "+0.01"       # smooth: the central value
    assert "Δ+" in page.table.item(0, 2).toolTip()
    assert "1 cell(s) marked (kink)" in page.status.text()


def test_each_root_has_its_own_apply_button_and_nothing_is_preselected(app, page) -> None:
    r = W.TargetQueryResult("front.rc_height_mm", "front.rc_static", 36.0,
                            np.array([20.0, 50.0]), np.array([20.0, 50.0]), [30.0, 40.0],
                            "2 value(s) ...")
    before = page.ctrl.project.front.rc_height_mm
    seen: list[str] = []
    _click_in_next_modal(app, "Apply 40.000", seen)
    page._offer_roots(r, 3)
    assert seen == ["Apply 30.000 (my decision)", "Apply 40.000 (my decision)", "Close"]
    assert page.ctrl.project.front.rc_height_mm == 40.0
    status = page.status.text()
    assert status.startswith("Applied Front roll centre height target [mm] = 40.000 (your "
                             "decision; Ctrl+Z undoes).")
    assert "Other roots: 30.000." in status
    page.ctrl.undo()
    assert page.ctrl.project.front.rc_height_mm == before
    # closing the dialog applies nothing
    seen.clear()
    _click_in_next_modal(app, "Close", seen)
    page._offer_roots(r, 3)
    assert page.ctrl.project.front.rc_height_mm == before


def test_applying_a_mirrored_hardpoint_root_keeps_the_car_symmetric(app) -> None:
    from apps.vdstudio.ui.controller import ProjectController
    from apps.vdstudio.ui.pages.tradeoff import TradeoffPage

    p = Project()
    p.hardpoints = freeze_to_hardpoints(p)
    p.geometry_mode = "hardpoints"
    path = "hardpoints.points.FL.UCA_OUT.1"
    base_y = p.hardpoints.points["FL"]["UCA_OUT"][1]
    target = compute_static(resolve(p)).value("front.rc_static") + 0.5
    r = W.target_query(p, path, "front.rc_static", target, base_y - 6.0, base_y + 6.0, n_scan=5)
    assert len(r.roots) == 1
    ctrl = ProjectController()
    ctrl.new(p)
    page = TradeoffPage(ctrl)
    try:
        page.apply_root(r, r.roots[0])
        pts = ctrl.project.hardpoints.points
        assert pts["FL"]["UCA_OUT"][1] == pytest.approx(r.roots[0])
        assert pts["FR"]["UCA_OUT"][1] == pytest.approx(-r.roots[0])
        st = ctrl.static
        assert st.value("front.kpi_L") == pytest.approx(st.value("front.kpi_R"), abs=1e-9)
        assert st.value("front.scrub_L") == pytest.approx(st.value("front.scrub_R"), abs=1e-9)
        assert st.value("front.rc_static") == pytest.approx(target, abs=0.01)  # as queried
        assert page.status.text().startswith("Applied ")
        ctrl.undo()
        assert ctrl.project.hardpoints.points["FR"]["UCA_OUT"][1] == pytest.approx(-base_y)
    finally:
        page.shutdown()
        ctrl.shutdown()
