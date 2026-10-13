"""Steering: caster/trail, steering arm, rack, Ackermann, ratio, bump steer, effort."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...engine import curves as C
from ...engine import whatif as W
from ...engine.geometry import CarGeometry
from ...engine.kpis import KpiSet
from .. import fields, theme
from ..asyncjob import Debounced
from ..widgets import KpiTable, ParamForm, band_region, banner, button, pen, plot, scroll
from .base import Page

GROUPS = ["Steering geometry", "Steering kinematics", "Steering effort (parking)", "Bump steer"]


def rack_axis_label(left_for_positive: bool | None) -> str:
    """Rack axis label from the +1 mm probe (``SteerCurves.left_turn_for_positive_rack``)."""
    if left_for_positive is None:
        return "rack [mm] (turn direction unknown: +1 mm probe did not solve)"
    return "rack [mm] (+ = left turn)" if left_for_positive else \
        "rack [mm] (+ = right turn: steering arms behind the axle)"


class SteeringPage(Page):
    title = "Steering"
    subtitle = ("Caster and mechanical trail, the steering arm, rack position and hardware. "
                "Ackermann and ratio are solved on the real linkage (DWSolver rack sweep), not "
                "read off the plan-view construction.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        self.note = banner("")
        self.root.addWidget(self.note)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.root.addWidget(split, 1)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        tabs = QtWidgets.QTabWidget()
        self.form = ParamForm(ctrl, fields.steering_fields())
        self.bands = ParamForm(ctrl, fields.steering_band_fields())
        tabs.addTab(scroll(self.form), "Inputs")
        tabs.addTab(scroll(self.bands), "Bands")
        ll.addWidget(tabs, 1)
        q = QtWidgets.QGroupBox("Target queries (the answer is shown, nothing is applied)")
        ql = QtWidgets.QVBoxLayout(q)
        ql.addWidget(button("Rack z that nulls the front linear bump steer…",
                            lambda: self._query("steering.rack_z_mm", "front.bump_steer", 0.0,
                                                -25, 25)))
        ql.addWidget(button("Rack half-length that nulls the front linear bump steer…",
                            lambda: self._query("steering.rack_half_length_mm",
                                                "front.bump_steer", 0.0, -60, 60)))
        row = QtWidgets.QHBoxLayout()
        self.ack_target = QtWidgets.QDoubleSpinBox()
        self.ack_target.setRange(-50, 200)
        self.ack_target.setValue(100.0)
        self.ack_target.setSuffix(" %")
        row.addWidget(button("Arm angle for Ackermann =",
                             lambda: self._query("steering.steer_arm_angle_deg",
                                                 "steering.ackermann", self.ack_target.value(),
                                                 -25, 25)))
        row.addWidget(self.ack_target)
        ql.addLayout(row)
        ql.addWidget(button("Rear toe-link inboard Z that nulls the rear bump steer…",
                            lambda: self._query("rear_toe_link_inboard.2", "rear.bump_steer",
                                                0.0, -25, 25)))
        self.q_status = QtWidgets.QLabel("")
        self.q_status.setWordWrap(True)
        ql.addWidget(self.q_status)
        ll.addWidget(q)
        left.setMinimumWidth(430)
        split.addWidget(left)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        pt = QtWidgets.QTabWidget()
        self.p_ack = plot("Ackermann: inner vs outer wheel", "outer wheel steer [deg]",
                          "inner wheel steer [deg]")
        self.p_pct = plot("Ackermann % along the stroke", "outer wheel steer [deg]", "% Ackermann")
        self.p_rack = plot("Steer angle vs rack travel", rack_axis_label(True),
                           "toe change per side [deg] (+ in)")
        self.p_ratio = plot("Steering ratio vs rack (local, SW : road wheel)",
                            rack_axis_label(True), "ratio [:1]")
        self.p_bump = plot("Bump steer (toe change vs wheel travel)", "toe per side [deg] (+ in)",
                           "wheel travel [mm] (+ bump)")
        self.p_cam = plot("Inclination (wheel plane to vertical) and caster vs steer (FL)",
                          "FL steer [deg]", "[deg]")
        self.p_rad = plot("Kinematic turning radius", "outer wheel steer [deg]",
                          "outer front patch path radius [m]")
        for w, name in ((self.p_ack, "Ackermann"), (self.p_pct, "Ackermann %"),
                        (self.p_rack, "Steer vs rack"), (self.p_ratio, "Ratio vs rack"),
                        (self.p_bump, "Bump steer"), (self.p_cam, "Inclination / caster"),
                        (self.p_rad, "Turning radius")):
            pt.addTab(w, name)
        right.addWidget(pt)
        self.table = KpiTable(GROUPS)
        right.addWidget(self.table)
        right.setSizes([480, 420])
        split.addWidget(right)
        split.setSizes([460, 1000])
        self.job = Debounced(250, self)
        self.qjob = Debounced(0, self)
        self.qjob.progress.connect(lambda f, m: self.q_status.setText(f"querying… {100*f:.0f} %"))
        self.qjob.failed.connect(lambda m: self.q_status.setText("query failed: " + m[:200]))

    def on_project(self) -> None:
        self.form.refresh()
        self.bands.refresh()
        synth = self.ctrl.project.geometry_mode == "synthesis"
        for f in self.form.findChildren(QtWidgets.QGroupBox):
            if f.title().startswith(("Kingpin", "Steering arm", "Rack (")):
                f.setEnabled(synth)
        self.note.label.setText(
            "Synthesis inputs drive caster, the steering arm and the rack." if synth else
            "<b>Hardpoint mode:</b> caster, arm and rack come from the point table; the "
            "hardware, alignment and band inputs still apply.")

    def on_geometry(self, geo: CarGeometry) -> None:
        def work(progress, cancelled):
            s = C.steer_curves(geo, 41, cancelled)
            h = C.heave_curves(geo, -geo.front_inputs.travel_droop_mm,
                               geo.front_inputs.travel_bump_mm, 21, cancelled)
            return s, h

        self.job.request(work, self._draw)

    def on_kpis(self, ks: KpiSet) -> None:
        self.table.set_kpis(ks)

    def _draw(self, res) -> None:
        s, h = res
        t = theme.current()
        for w in (self.p_ack, self.p_pct, self.p_rack, self.p_ratio, self.p_bump, self.p_cam,
                  self.p_rad):
            w.clear()
        o = s.ack_outer
        self.p_ack.plot(o, s.ack_inner, pen=pen(0, 2.5), name="actual inner (solved)")
        self.p_ack.plot(o, s.ack_ideal, pen=pen(1, 2, QtCore.Qt.DashLine), name="100 % Ackermann")
        self.p_ack.plot(o, o, pen=pen(7, 1, QtCore.Qt.DotLine), name="parallel steer")
        ref = self.ctrl.project.steering.ackermann_at_steer_deg
        self.p_ack.addItem(pg.InfiniteLine(ref, angle=90, pen=pg.mkPen(t.text_muted, width=1)))
        lim = self.ctrl.project.steering.limits.ackermann_pct
        band_region(self.p_pct, lim[0], lim[1])
        m = np.isfinite(s.ack_pct) & (o > 0.5)
        self.p_pct.plot(o[m], s.ack_pct[m], pen=pen(0, 2.5), name="solved")
        self.p_rack.plot(s.rack, s.steer_fl, pen=pen(0), name="FL")
        self.p_rack.plot(s.rack, s.steer_fr, pen=pen(1), name="FR")
        for w in (self.p_rack, self.p_ratio):
            w.getPlotItem().setLabel("bottom", rack_axis_label(s.left_turn_for_positive_rack))
        self.p_ratio.plot(s.rack, s.ratio_fl, pen=pen(0), name="FL", connect="finite")
        self.p_ratio.plot(s.rack, s.ratio_fr, pen=pen(1), name="FR", connect="finite")
        for cid, i in (("FL", 0), ("RL", 2)):
            self.p_bump.plot(h.corners[cid]["toe"], h.travel, pen=pen(i), name=cid)
        self.p_cam.plot(s.steer_fl, s.camber_fl, pen=pen(0),
                        name="inclination FL (wheel plane to vertical)")
        self.p_cam.plot(s.steer_fl, s.caster_fl, pen=pen(1), name="caster FL")
        self.p_cam.plot(s.steer_fl, s.kpi_fl, pen=pen(2), name="KPI FL")
        m = np.isfinite(s.turn_radius_m) & (o > 2.0)
        self.p_rad.plot(o[m], s.turn_radius_m[m], pen=pen(0, 2.5),
                        name="patch-centre path (L / sin δo)")
        self.p_rad.addItem(pg.InfiniteLine(
            4.5, angle=0, label="FSAE hairpin outside edge (9 m Ø): the car's OUTERMOST point "
            "must stay inside", labelOpts={"color": t.warn},
            pen=pg.mkPen(t.warn, width=1, style=QtCore.Qt.DashLine)))
        _ = math

    # -- queries ------------------------------------------------------------------------
    def _query(self, param: str, output: str, target: float, dlo: float, dhi: float) -> None:
        p = self.ctrl.snapshot()
        if param.startswith("steering.") and p.geometry_mode != "synthesis" and param not in (
                "steering.pinion_radius_mm",):
            self.q_status.setText("This query varies a synthesis input; switch the geometry "
                                  "source to synthesis first.")
            return
        base = W.param_by_path(param).get(p)
        lo, hi = base + dlo, base + dhi

        def work(progress, cancelled):
            return W.target_query(p, param, output, target, lo, hi, n_scan=9,
                                  progress=progress, cancelled=cancelled)

        self.q_status.setText("querying…")
        self.qjob.request(work, lambda r: self._show_query(r, base), immediate=True)

    def _show_query(self, r: W.TargetQueryResult, base: float) -> None:
        par = W.param_by_path(r.param)
        if not r.roots:
            self.q_status.setText(r.message)
            return
        k = self.ctrl.kpis.get(r.output) if self.ctrl.kpis is not None else None
        out = f"{k.scope} {k.label} [{k.unit}]" if k is not None else r.output
        roots = ", ".join(f"{x:.4f}" for x in r.roots)
        self.q_status.setText(f"<b>{par.label}</b> = {roots} {par.unit} puts <i>{out}</i> "
                              f"at {r.target:g} (current {base:.4f}). {r.message}")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Target query")
        box.setText(f"{par.label} = {roots} {par.unit}\n(current value {base:.4f})\n\n"
                    f"{r.message}\n\nApply a value? (Your decision -- it is recorded and "
                    "undoable.)")
        # every root gets its own button: the query fills the table, it never picks a row
        buttons = {box.addButton(f"Apply {x:.4f}", QtWidgets.QMessageBox.AcceptRole): float(x)
                   for x in r.roots}
        box.addButton("Keep current", QtWidgets.QMessageBox.RejectRole)
        box.exec()
        x = buttons.get(box.clickedButton())
        if x is not None:
            self.ctrl.apply(lambda p, par=par, v=x: par.apply(p, v))
            others = ", ".join(f"{y:.4f}" for y in r.roots if y != x)
            self.q_status.setText(f"Applied <b>{par.label}</b> = {x:.4f} {par.unit} (your "
                                  "decision; Ctrl+Z undoes)."
                                  + (f" Other roots: {others} {par.unit}." if others else ""))
