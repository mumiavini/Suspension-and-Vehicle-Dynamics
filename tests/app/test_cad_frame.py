"""CAD placement for the SolidWorks skeleton: forward/left axes and the ISO origin in CAD."""

from __future__ import annotations

import importlib
import itertools
import os
from dataclasses import replace

import numpy as np
import pytest

from apps.vdstudio.engine import exports as X
from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine.project import (
    CAD_AXES,
    CadFrame,
    Project,
    project_from_dict,
    project_to_dict,
)

swx = X.swx                     # scripts/ is on sys.path once apps.vdstudio is imported
swi = importlib.import_module("import_solidworks_hardpoints")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VDSTUDIO_NO_GL", "1")


@pytest.fixture(scope="module")
def geo():
    return G.resolve(Project())


def test_axis_frames_are_proper_rotations() -> None:
    n = 0
    for fwd, left in itertools.product(CAD_AXES, CAD_AXES):
        if fwd[1] == left[1]:
            with pytest.raises(ValueError, match="different CAD axes"):
                swx.axis_frame(fwd, left)
            continue
        m = np.array(swx.axis_frame(fwd, left).matrix)
        assert np.allclose(m @ m.T, np.eye(3)) and np.linalg.det(m) == pytest.approx(1.0)
        assert tuple(m[:, 0]) == swx.AXES[fwd] and tuple(m[:, 1]) == swx.AXES[left]
        n += 1
    assert n == 24                                   # every right-handed axis assignment
    assert swx.axis_frame("+X", "+Y").matrix == swx.FRAMES["iso"]
    assert swx.axis_frame("-X", "-Y").matrix == swx.FRAMES["sw_fsae"]
    with pytest.raises(ValueError, match="axes are one of"):
        swx.axis_frame("+W", "+Y")


def test_car_lengthwise_along_cad_y_with_an_offset_origin() -> None:
    """Front = CAD -Y, left = CAD +X (so up = +Z), ISO origin at CAD (0, -350, -80)."""
    fr = swx.axis_frame("-Y", "+X", (0.0, -350.0, -80.0))
    assert "up +Z" in fr.label
    # 1 m ahead of the front axle lands 1 m further along CAD -Y from the origin point
    assert X.to_cad((1000.0, 0.0, 0.0), fr) == pytest.approx((0.0, -1350.0, -80.0))
    assert X.to_cad((0.0, 600.0, 0.0), fr) == pytest.approx((600.0, -350.0, -80.0))   # left
    assert X.to_cad((0.0, 0.0, 300.0), fr) == pytest.approx((0.0, -350.0, 220.0))     # up


@pytest.mark.parametrize("forward,left,origin", [
    ("+X", "+Y", (0.0, 0.0, 0.0)),
    ("-X", "-Y", (1200.0, 0.0, -45.0)),
    ("-Y", "+X", (0.0, -350.0, -80.0)),
    ("+Z", "+X", (12.5, 900.0, -3.0)),             # a Y-up model
])
def test_draw_then_read_back_recovers_every_hardpoint(geo, forward, left, origin) -> None:
    fr = swx.axis_frame(forward, left, origin)
    table = X._sw_table(geo)
    groups = swx.build_groups(table, fr, "corner")
    segments = [(name, tuple(v * 1000.0 for v in a), tuple(v * 1000.0 for v in b))
                for name, segs in groups for _label, a, b in segs]
    back = swi.decode(segments, fr)
    assert set(back) == set(table)
    worst = max(float(np.linalg.norm(np.subtract(back[k], table[k]))) for k in table)
    assert worst < 1e-9
    if origin != (0.0, 0.0, 0.0):                  # read without the origin: shifted car
        moved = swi.decode(segments, swx.axis_frame(forward, left))
        assert max(abs(moved[k][2] - table[k][2]) for k in table) > 1.0 or \
            max(abs(moved[k][0] - table[k][0]) for k in table) > 1.0


def test_macro_states_the_placement(geo) -> None:
    fr = swx.axis_frame("-Y", "+X", (0.0, -350.0, -80.0))
    m = X.solidworks_macro(geo, fr)
    assert "forward -Y, left +X, up +Z; ISO origin at CAD (0, -350, -80) mm" in m
    # the first line drawn is the placed point, in metres
    start, end, _label = swx.MEMBERS[0][2][0]
    first = next(line for line in m.splitlines() if "CreateLine" in line)
    nums = [float(v) for v in first.split("CreateLine", 1)[1].split(",")]
    want = [v / 1000.0 for v in (*X.to_cad(geo.points["FL"][start], fr),
                                 *X.to_cad(geo.points["FL"][end], fr))]
    assert nums == pytest.approx(want, abs=1e-9)
    with pytest.raises(ValueError, match="supports the frames"):
        X.solidworks_macro(geo, "optimumk")


def test_project_keeps_the_placement() -> None:
    p = replace(Project(), cad_frame=CadFrame("-Y", "+X", 0.0, -350.0, -80.0))
    back = project_from_dict(project_to_dict(p))
    assert back.cad_frame == p.cad_frame
    assert X.cad_frame(back).origin_mm == (0.0, -350.0, -80.0)
    d = project_to_dict(Project())
    del d["cad_frame"]
    assert project_from_dict(d).cad_frame == CadFrame()          # old files: ISO, no offset
    assert X.cad_frame(Project()).matrix == swx.FRAMES["iso"]


def test_integrations_page_preview_follows_the_placement() -> None:
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from apps.vdstudio.ui.controller import ProjectController
    from apps.vdstudio.ui.pages.integrations import IntegrationsPage

    ctrl = ProjectController()
    ctrl.recompute()
    page = IntegrationsPage(ctrl)
    try:
        page.show()
        page._preset("-X", "-Y")
        assert ctrl.project.cad_frame.forward == "-X"
        page._set_cad("cad_frame.origin_z_mm", -80.0)
        page.on_project()
        assert "forward -X, left -Y, up +Z" in page.sw_frame_label.text()
        row = next(r for r in range(page.sw_preview.rowCount())
                   if page.sw_preview.item(r, 0).text() == "FL"
                   and page.sw_preview.item(r, 1).text() == "CONTACT_PATCH")
        want = X.to_cad(ctrl.geometry.points["FL"]["CONTACT_PATCH"],
                        swx.axis_frame("-X", "-Y", (0.0, 0.0, -80.0)))
        got = [float(page.sw_preview.item(row, c).text()) for c in (2, 3, 4)]
        assert got == pytest.approx(want, abs=1e-3)
        assert page.btn_macro.isEnabled()
        page._set_cad("cad_frame.left", "-X")                       # parallel to forward
        page.on_project()
        assert "Invalid placement" in page.sw_frame_label.text()
        assert not page.btn_macro.isEnabled() and page.sw_preview.rowCount() == 0
        ctrl.undo()                                                  # placement edits undo
        page.on_project()
        assert ctrl.project.cad_frame.left == "-Y" and page.btn_macro.isEnabled()
    finally:
        ctrl.shutdown()
        page.close()


def test_hardpoints_in_cad_coordinates(geo) -> None:
    fr = swx.axis_frame("-Y", "-X", (0.0, -120.4, 75.0))
    rows = X.hardpoints_cad_csv(geo, fr).splitlines()
    assert rows[0].split(",")[:6] == ["corner", "point", "mounts_to", "cad_x_mm", "cad_y_mm",
                                     "cad_z_mm"]
    by = {(r.split(",")[0], r.split(",")[1]): r.split(",") for r in rows[1:]}
    assert len(by) == 40
    fl = by[("FL", "LCA_IN_FRONT")]
    assert fl[2] == "chassis"
    cad = [float(v) for v in fl[3:6]]
    assert cad == pytest.approx(X.to_cad(geo.points["FL"]["LCA_IN_FRONT"], fr), abs=1e-3)
    assert [float(v) for v in fl[6:9]] == pytest.approx(geo.points["FL"]["LCA_IN_FRONT"],
                                                        abs=1e-3)
    # the contact patch sits on the chassis model's ground plane (CAD Z = +75)
    assert float(by[("FL", "CONTACT_PATCH")][5]) == pytest.approx(75.0)
