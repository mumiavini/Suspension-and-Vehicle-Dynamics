"""Hairpin check with the outermost point; plots restyled on a theme toggle; table eliding."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest

from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine.project import Project, project_from_dict, project_to_dict
from vdcore.geometry.solver import DWSolver

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VDSTUDIO_NO_GL", "1")


def test_radii_formulas_reduce_to_the_patch_path() -> None:
    r = K.turning_radii_m(1540.0, 27.06, 245.0, 0.0, 0.0, 0.0)
    patch = 1540.0 / math.sin(math.radians(27.06)) / 1000.0
    assert float(r["patch"]) == pytest.approx(patch, rel=1e-12)
    assert float(r["body"]) == pytest.approx(patch, rel=1e-12)      # no offset = the patch
    assert float(r["tyre"]) == pytest.approx(math.hypot(patch, 0.245), rel=1e-12)
    r = K.turning_radii_m(1540.0, 27.06, 245.0, 190.0, 50.0, 700.0)
    assert float(r["tyre"]) == pytest.approx(math.hypot(patch + 0.095, 0.245), rel=1e-12)
    cot = 1.0 / math.tan(math.radians(27.06))
    assert float(r["body"]) == pytest.approx(math.hypot(2.240, 0.050 + 1.540 * cot), rel=1e-12)
    assert float(r["outermost"]) == max(float(r[k]) for k in ("patch", "tyre", "body"))
    assert np.isnan(K.turning_radii_m(1540.0, 0.0, 245.0, 190.0, 0.0, 0.0)["patch"])


def test_radii_against_the_solved_steered_wheel() -> None:
    """Independent: the turn centre where the solved outer wheel's axle meets the rear-axle
    line, the tyre outline built from the solved heading. Agrees with the kinematic
    formulas to the ~1.5 % the steered patch moves at lock."""
    geo = G.resolve(Project())
    rack = geo.steering_inputs.max_rack_travel_mm
    fr = DWSolver(geo.corner("FR")).solve(rack_mm=rack)        # outer wheel of a left turn
    toe = math.radians(fr.toe_deg_per_side)        # + = toe-in: a right wheel steering left
    axle = np.array([math.sin(toe), -math.cos(toe)])   # outboard spin axis in plan view (DWSolver)
    cp = np.array([fr.contact_patch.x_mm, fr.contact_patch.y_mm])
    x_rear = geo.p("RR", "CONTACT_PATCH")[0]
    t = (cp[0] - x_rear) / axle[0]                    # walk inboard along the axle to x_rear
    centre = cp - t * axle
    r_patch = float(np.linalg.norm(cp - centre))
    head = np.array([axle[1], -axle[0]])
    w2, r = 95.0, geo.axles["front"].loaded_radius_mm
    corners = [cp + w2 * axle + s * r * head for s in (-1.0, 1.0)]
    r_tyre = max(float(np.linalg.norm(c - centre)) for c in corners)
    ks = K.compute_dynamic(geo, ["steer"])
    outer = ks.value("steering.max_steer_outer")
    f = K.turning_radii_m(geo.wheelbase_mm, outer, r, 190.0, 0.0, 0.0)
    assert r_patch / 1000.0 == pytest.approx(float(f["patch"]), rel=0.025)
    assert r_tyre / 1000.0 == pytest.approx(float(f["tyre"]), rel=0.025)


def test_kpi_states_which_point_sets_the_radius_and_compares_to_the_hairpin() -> None:
    ks = K.compute_dynamic(G.resolve(Project()), ["steer"])
    k = ks.get("steering.turn_radius_outermost")
    assert k.value == pytest.approx(3.49, abs=0.01) and k.band == (0.0, 4.5)
    assert "outer front tyre" in k.note and "'estimate'" in k.note
    p = replace(Project(), turning_envelope=replace(
        Project().turning_envelope, body_outboard_of_patch_mm=50.0,
        body_ahead_of_front_axle_mm=700.0))
    kb = K.compute_dynamic(G.resolve(p), ["steer"]).get("steering.turn_radius_outermost")
    assert kb.value == pytest.approx(3.80, abs=0.01) and "body point" in kb.note
    back = project_from_dict(project_to_dict(p))
    assert back.turning_envelope.body_ahead_of_front_axle_mm == 700.0
    d = project_to_dict(Project())
    del d["turning_envelope"]
    assert project_from_dict(d).turning_envelope == Project().turning_envelope


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_theme_toggle_restyles_existing_plots(qapp) -> None:
    from PySide6 import QtGui

    from apps.vdstudio.ui import theme
    from apps.vdstudio.ui.widgets import plot, restyle_plots

    theme.apply(qapp, theme.DARK)
    from PySide6 import QtWidgets

    host = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(host)
    pw = plot("t", "x", "y")
    lay.addWidget(pw)
    theme.apply(qapp, theme.LIGHT)
    assert pw.backgroundBrush().color().name() != QtGui.QColor(theme.LIGHT.plot_bg).name()
    assert restyle_plots(host) == 1
    assert pw.backgroundBrush().color().name() == QtGui.QColor(theme.LIGHT.plot_bg).name()
    assert pw.getPlotItem().getAxis("left").textPen().color().name() == \
        QtGui.QColor(theme.LIGHT.plot_fg).name()
    theme.apply(qapp, theme.DARK)


def test_simple_table_elides_long_text_with_a_tooltip(qapp) -> None:
    from PySide6 import QtCore, QtWidgets

    from apps.vdstudio.ui.widgets import SimpleTable

    t = SimpleTable(["a", "source"])
    long = "derived from the inputs above (spring × MR²), not from the linkage; " * 2
    t.set_rows([["x", long], ["y", "short"]])
    assert not t.wordWrap() and t.textElideMode() == QtCore.Qt.ElideRight
    assert t.item(0, 1).toolTip() == long and t.item(1, 1).toolTip() == ""
    assert t.horizontalHeader().sectionResizeMode(1) == QtWidgets.QHeaderView.Stretch
