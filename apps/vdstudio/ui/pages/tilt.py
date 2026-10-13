"""Tilt table: does the car stay on its wheels at the test angle, and by how much."""

from __future__ import annotations

import html
import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...engine.geometry import CarGeometry
from ...engine.tilt import TiltInputs, TiltResult, tilt_table
from .. import fields, theme
from ..widgets import ParamForm, SimpleTable, banner, pen, plot, scroll
from .base import Page

SCOPE = ("Quasi-static. Tips about the line through the two downhill contact patches, taken "
         "at the CG station. Body roll uses the roll gradient (design_intent) on the sin θ "
         "lateral load and moves the sprung CG d·φ downhill. NOT modelled (no wheel rates, no "
         "tyre data): the springs extending under the cos θ normal load and the tyres' side "
         "deflection -- set them below to see their effect. One uphill wheel lifting is not "
         "the tip: the car stands on three wheels until the tip angle.")
CORNERS = ("FL", "FR", "RL", "RR")


class TiltPage(Page):
    title = "Tilt table"
    subtitle = ("The car on the tilt table (downhill on its right): the angle at which it "
                "tips, the wheel loads at the test angle, and what moves the answer. Reported, "
                "never judged.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.root.addWidget(split, 1)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.form = ParamForm(ctrl, fields.tilt_fields())
        ll.addWidget(self.form)
        box = QtWidgets.QGroupBox("What-if (not saved): effects the model leaves out")
        form = QtWidgets.QFormLayout(box)
        self.inp: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key, lab, lo, hi, step, val, unit, tip in (
                ("roll_stiffness_front_share", "Roll stiffness, front share", 0, 1, 0.05, 0.5,
                 "-", "Springs + anti-roll bars, front / total (design_intent). Decides which "
                      "uphill wheel lifts first; not the tip angle."),
                ("cg_rise_mm", "CG rise on the table", 0, 100, 1, 0, "mm",
                 "The normal load drops to cos θ, so the springs extend and the body rises. "
                 "Needs the wheel rates to compute."),
                ("lateral_shift_mm", "Extra CG shift downhill", 0, 100, 1, 0, "mm",
                 "Tyre side deflection, lateral compliance. Needs tyre data to compute.")):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(2 if unit == "-" else 1)
            sb.setValue(val)
            sb.setSuffix(f" {unit}" if unit != "-" else "")
            sb.setKeyboardTracking(False)
            sb.setToolTip(tip)
            sb.valueChanged.connect(lambda _v: self._run())
            lab_w = QtWidgets.QLabel(lab)
            lab_w.setToolTip(tip)
            form.addRow(lab_w, sb)
            self.inp[key] = sb
        ll.addWidget(box)
        self.verdict = QtWidgets.QLabel("")
        self.verdict.setWordWrap(True)
        self.verdict.setTextFormat(QtCore.Qt.RichText)
        ll.addWidget(self.verdict)
        self.summary = SimpleTable(["quantity", "value"])
        self.summary.setMinimumHeight(345)
        ll.addWidget(self.summary, 1)
        self.wheels = SimpleTable(["wheel", "side", "Fz on the table [N]", "share of level [%]"])
        self.wheels.setMinimumHeight(150)
        ll.addWidget(self.wheels)
        note = QtWidgets.QLabel(SCOPE)
        note.setObjectName("muted")
        note.setWordWrap(True)
        ll.addWidget(note)
        self.warn = banner("")
        self.warn.setVisible(False)
        ll.addWidget(self.warn)
        left.setMinimumWidth(440)
        left_scroll = scroll(left)           # the inputs and tables scroll on short screens
        left_scroll.setMinimumWidth(470)
        split.addWidget(left_scroll)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.p_view = plot("Seen from behind, at the table angle", "", "", legend=False)
        self.p_view.setAspectLocked(True)
        self.p_view.getPlotItem().hideAxis("left")
        self.p_view.getPlotItem().hideAxis("bottom")
        rl.addWidget(self.p_view, 3)
        row = QtWidgets.QHBoxLayout()
        self.p_fz = plot("Wheel loads vs table angle", "table angle [deg]", "Fz [N]")
        self.p_h = plot("Tip angle vs CG height", "CG height [mm]", "tip angle [deg]")
        row.addWidget(self.p_fz)
        row.addWidget(self.p_h)
        rl.addLayout(row, 2)
        split.addWidget(right)
        split.setSizes([460, 980])

    # -- inputs ------------------------------------------------------------------------
    def on_project(self) -> None:
        self.form.refresh()

    def on_geometry(self, geo: CarGeometry) -> None:
        self._run()

    def on_kpis(self, ks) -> None:
        self._run()

    def _rc(self) -> dict[str, float]:
        ks = self.ctrl.static
        return {"front": ks.value("front.rc_static"), "rear": ks.value("rear.rc_static")}

    def _inputs(self, **over) -> TiltInputs:
        veh = self.ctrl.project.vehicle
        kw = {"angle_deg": veh.tilt_test_angle_deg,
              "roll_gradient_deg_per_g": veh.target_roll_gradient_deg_per_g,
              **{k: sb.value() for k, sb in self.inp.items()}}
        kw.update(over)
        return TiltInputs(**kw)

    # -- compute and draw -------------------------------------------------------------
    def _run(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        rc = self._rc()
        try:
            r = tilt_table(geo, rc, self._inputs())
        except (ValueError, RuntimeError) as exc:
            self._clear(f"Failed: {html.escape(str(exc))}")
            return
        self.warn.setVisible(bool(r.notes))
        self.warn.label.setText("<br>".join(html.escape(n) for n in r.notes))
        self._show(r)
        self._draw_view(geo, r)
        self._draw_fz(geo, rc, r)
        self._draw_h(geo, rc, r)

    def _clear(self, msg: str) -> None:
        for t in (self.summary, self.wheels):
            t.set_rows([])
        for p in (self.p_view, self.p_fz, self.p_h):
            p.clear()
        self.verdict.setText("")
        self.warn.setVisible(True)
        self.warn.label.setText(msg)

    def _show(self, r: TiltResult) -> None:
        t = theme.current()
        ang = r.inputs.angle_deg
        m = r.margin_deg
        if not math.isfinite(r.tip_angle_deg):
            colour, words = t.bad, "tip angle not found"
        elif m >= 0:
            colour, words = t.ok, f"stays on its wheels at {ang:g}°: tips at " \
                                  f"{r.tip_angle_deg:.2f}° ({m:+.2f}°)"
        else:
            colour, words = t.bad, f"tips before {ang:g}°: at {r.tip_angle_deg:.2f}° ({m:+.2f}°)"
        self.verdict.setText(f"<span style='font-size:13pt; color:{colour}'><b>{words}</b></span>"
                             f"<br><span style='color:{t.text_muted}'>with the inputs and "
                             "assumptions on this page</span>")
        lift = ("none below 89°" if r.first_lift is None else
                f"{r.first_lift[0]} at {r.first_lift[1]:.2f}°")
        self.summary.set_rows([
            ["Tip angle (with body roll)", f"{r.tip_angle_deg:.2f}°"],
            ["Tip angle, rigid car", f"{r.tip_angle_rigid_deg:.2f}°"],
            [f"Margin to {ang:g}°", f"{m:+.2f}°"],
            ["First uphill wheel lift", lift],
            [f"Body roll at {ang:g}°", f"{r.roll_deg:.2f}°"],
            [f"CG shift downhill at {ang:g}°", f"{r.cg_shift_mm:.1f} mm"],
            ["CG height used", f"{r.cg_height_mm:.1f} mm"],
            [f"CG height that tips exactly at {ang:g}°", f"{r.max_cg_height_mm:.1f} mm"],
            ["Tipping line from the centreline (CG station)", f"{r.tip_line_mm:.1f} mm"],
            ["Roll arm d / sprung fraction", f"{r.roll_arm_mm:.1f} mm / "
                                             f"{100 * r.sprung_fraction:.1f} %"],
            ["Roll centres used (solved, static)",
             f"F {r.rc_mm['front']:.1f} / R {r.rc_mm['rear']:.1f} mm"],
        ])
        level = tilt_table(self.ctrl.geometry, self._rc(), self._inputs(angle_deg=1e-9))
        rows = []
        for cid in CORNERS:
            fz, f0 = r.wheel_fz_N[cid], level.wheel_fz_N[cid]
            rows.append([cid, "uphill" if cid in ("FL", "RL") else "downhill",
                         f"{fz:.0f}" if fz > 0 else f"{fz:.0f} (lifted)", f"{100 * fz / f0:.0f}"])
        self.wheels.set_rows(rows, right_cols=(2, 3))

    def _draw_view(self, geo: CarGeometry, r: TiltResult) -> None:
        """Rear view (car's left on screen left), table rotated so the right side is low."""
        p = self.p_view
        p.clear()
        t = theme.current()
        th = math.radians(r.inputs.angle_deg)
        c, s = math.cos(th), math.sin(th)

        def w(xs, zs):           # table frame (screen x = −y_car, z normal) -> world
            xs, zs = np.asarray(xs, float), np.asarray(zs, float)
            return xs * c + zs * s, -xs * s + zs * c

        half = r.tip_line_mm
        span = half + 500.0
        p.plot(*w([-span, span], [0, 0]), pen=pg.mkPen(t.text_muted, width=3))   # table
        p.plot([-span * c, span * c], [-span * s, -span * s],
               pen=pg.mkPen(t.text_muted, width=1, style=QtCore.Qt.DashLine))       # floor
        rad = geo.axles["front"].loaded_radius_mm
        wid = 190.0
        for side in (-1, 1):                                                     # tyres
            x0 = side * half
            xs = [x0 - wid / 2, x0 + wid / 2, x0 + wid / 2, x0 - wid / 2, x0 - wid / 2]
            zs = [0, 0, 2 * rad, 2 * rad, 0]
            p.plot(*w(xs, zs), pen=pg.mkPen(t.text, width=2))
        # body: a box between the wheels, rolled about the CG by φ (illustrative)
        phi = math.radians(r.roll_deg)
        bx = np.array([-half + 150, half - 150, half - 150, -half + 150, -half + 150])
        bz = np.array([120, 120, 560, 560, 120], float)
        cx, cz = r.cg_shift_mm, r.cg_height_mm
        rx = cx + (bx - cx) * math.cos(phi) + (bz - cz) * math.sin(phi)
        rz = cz - (bx - cx) * math.sin(phi) + (bz - cz) * math.cos(phi)
        p.plot(*w(rx, rz), pen=pen(0, 1.5))
        gx, gz = w([cx], [cz])
        tipx, tipz = w([half], [0])
        stable = math.isfinite(r.tip_angle_deg) and r.margin_deg >= 0
        col = t.ok if stable else t.bad
        p.plot([gx[0], gx[0]], [gz[0], -span * s - 50], pen=pg.mkPen(col, width=2))   # gravity
        p.plot([tipx[0], tipx[0]], [tipz[0], gz[0] + 100],
               pen=pg.mkPen(t.warn, width=1.5, style=QtCore.Qt.DashLine))           # tip line
        p.plot(gx, gz, pen=None, symbol="o", symbolSize=12, symbolBrush=col)
        p.plot(tipx, tipz, pen=None, symbol="t", symbolSize=12, symbolBrush=t.warn)
        lab = pg.TextItem(f"CG — {abs(tipx[0] - gx[0]):.0f} mm inside the tipping point"
                          if gx[0] <= tipx[0] else "CG outside the tipping point: tips",
                          color=col, anchor=(0, 1))
        lab.setPos(gx[0] + 30, gz[0] + 30)
        p.addItem(lab)
        ang = pg.TextItem(f"{r.inputs.angle_deg:g}°", color=t.text_muted, anchor=(1, 0))
        ang.setPos(span * c, -span * s)
        p.addItem(ang)

    def _draw_fz(self, geo: CarGeometry, rc: dict[str, float], r: TiltResult) -> None:
        p = self.p_fz
        p.clear()
        t = theme.current()
        angles = np.linspace(0.0, 75.0, 76)
        res = [tilt_table(geo, rc, self._inputs(angle_deg=max(float(a), 1e-9))) for a in angles]
        for i, cid in enumerate(CORNERS):
            fz = np.array([x.wheel_fz_N[cid] for x in res])
            fz[fz < 0] = np.nan                     # lifted: three-wheel stance not modelled
            p.plot(angles, fz, pen=pen(i), name=cid, connect="finite")
        p.addItem(pg.InfiniteLine(r.inputs.angle_deg, angle=90,
                                  pen=pg.mkPen(t.text_muted, style=QtCore.Qt.DashLine)))
        if math.isfinite(r.tip_angle_deg):
            p.addItem(pg.InfiniteLine(r.tip_angle_deg, angle=90, pen=pg.mkPen(t.bad, width=2),
                                      label=f"tips {r.tip_angle_deg:.1f}°",
                                      labelOpts={"position": 0.9, "color": t.bad}))
        p.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(t.text_muted)))

    def _draw_h(self, geo: CarGeometry, rc: dict[str, float], r: TiltResult) -> None:
        p = self.p_h
        p.clear()
        t = theme.current()
        h0 = geo.vehicle_data.cg_height_mm
        hs = np.linspace(0.75 * h0, 1.3 * h0, 45)
        rise_now = self.inp["cg_rise_mm"].value()
        for i, rise in enumerate(sorted({0.0, 10.0, 20.0, rise_now})):
            ys = [tilt_table(geo, rc, self._inputs(cg_height_mm=float(h), cg_rise_mm=rise))
                  .tip_angle_deg for h in hs]
            p.plot(hs, ys, pen=pen(i, 2.5 if rise == rise_now else 1.5),
                   name=f"CG rise {rise:g} mm" + (" (on page)" if rise == rise_now else ""))
        p.addItem(pg.InfiniteLine(r.inputs.angle_deg, angle=0,
                                  pen=pg.mkPen(t.bad, style=QtCore.Qt.DashLine)))
        p.addItem(pg.InfiniteLine(h0, angle=90, pen=pg.mkPen(t.text_muted),
                                  label=f"project {h0:g} mm",
                                  labelOpts={"position": 0.92, "color": t.text_muted}))
