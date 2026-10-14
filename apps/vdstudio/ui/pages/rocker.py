"""Pushrod & rocker: the linkage between the wheel and the team's damper and springs."""

from __future__ import annotations

import html
import math
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...engine import suspension as S
from ...engine.geometry import CarGeometry
from ...engine.project import Project
from .. import fields, theme
from ..asyncjob import Debounced
from ..widgets import ParamForm, SimpleTable, banner, button, pen, plot, scroll
from .base import Page

SCOPE = ("Kinematics on DWSolver (vdcore.analysis.rocker): the rod keeps its length, the rocker "
         "turns about its axis, MR = −dL_damper/d(wheel travel). Wheel rate = k·MR² (the "
         "spring-force × dMR/dz term is left out); frequency from the sprung corner mass, tyre "
         "not in series. Clearances are centreline distances: subtract the tube radii. "
         "Information only: the tool does not pick the MR, the spring or the bar.")
AXLES = ("front", "rear")
VIEWS = {"Front view (Y–Z)": (1, 2), "Side view (X–Z)": (0, 2), "Top view (X–Y)": (0, 1)}


def _f(x: float, fmt: str = ".1f", unit: str = "") -> str:
    return "—" if not (isinstance(x, (int, float)) and math.isfinite(x)) else \
        f"{x:{fmt}}{(' ' + unit) if unit else ''}"


class RockerPage(Page):
    title = "Pushrod & rocker"
    subtitle = ("Lay out the rod, the rocker and the damper on the chassis, and see what the "
                "team's damper and springs do through it: motion ratio, damper travel, rates, "
                "roll stiffness and the anti-roll bar that closes the gap. Reported, never "
                "chosen.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.root.addWidget(split, 1)

        # ---- left: inputs ---------------------------------------------------------------
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.forms: list[ParamForm] = [ParamForm(ctrl, fields.damper_spring_fields())]
        ll.addWidget(self.forms[0])
        avail = QtWidgets.QHBoxLayout()
        avail.addWidget(QtWidgets.QLabel("Springs available (lb/in):"))
        self.avail = QtWidgets.QLineEdit()
        self.avail.setToolTip("Comma-separated ratings; each one is shown in the rates table.")
        self.avail.editingFinished.connect(self._set_available)
        avail.addWidget(self.avail, 1)
        ll.addLayout(avail)
        tabs = QtWidgets.QTabWidget()
        self.seed: dict[str, dict[str, QtWidgets.QDoubleSpinBox]] = {}
        for axle in AXLES:
            w = QtWidgets.QWidget()
            wl = QtWidgets.QVBoxLayout(w)
            form = ParamForm(ctrl, fields.rocker_fields(axle), columns=2)
            self.forms.append(form)
            wl.addWidget(form)
            wl.addWidget(self._seed_box(axle))
            wl.addStretch(1)
            tabs.addTab(w, f"{axle.capitalize()} rocker")
        ll.addWidget(tabs)
        note = QtWidgets.QLabel(SCOPE)
        note.setObjectName("muted")
        note.setWordWrap(True)
        ll.addWidget(note)
        left_scroll = scroll(left)
        left_scroll.setMinimumWidth(560)
        split.addWidget(left_scroll)

        # ---- right: results -------------------------------------------------------------
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.warn = banner("")
        rl.addWidget(self.warn)
        rtabs = QtWidgets.QTabWidget()
        rl.addWidget(rtabs, 1)

        res = QtWidgets.QWidget()
        resl = QtWidgets.QVBoxLayout(res)
        self.table = SimpleTable(["quantity", "front", "rear"])
        self.table.setMinimumHeight(420)
        resl.addWidget(self.table, 3)
        self.rates = SimpleTable(["spring", "wheel rate F / R [N/mm]", "frequency F / R [Hz]",
                                  "roll stiffness F / R [N·m/deg]"])
        self.rates.setMinimumHeight(110)
        resl.addWidget(QtWidgets.QLabel("Every spring available, through the ride-height MR:"))
        resl.addWidget(self.rates, 1)
        self.budget = SimpleTable(["roll budget", "value"])
        self.budget.setMinimumHeight(190)
        resl.addWidget(self.budget, 2)
        rtabs.addTab(scroll(res), "Results")

        curves = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(curves)
        self.p_mr = plot("Motion ratio vs wheel travel", "wheel travel [mm] (+ bump)",
                         "MR (damper / wheel)")
        self.p_len = plot("Damper length vs wheel travel", "wheel travel [mm] (+ bump)",
                          "damper eye to eye [mm]")
        cl.addWidget(self.p_mr)
        cl.addWidget(self.p_len)
        rtabs.addTab(curves, "Curves")

        view = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(view)
        row = QtWidgets.QHBoxLayout()
        self.view_axle = QtWidgets.QComboBox()
        self.view_axle.addItems(["front", "rear"])
        self.view_kind = QtWidgets.QComboBox()
        self.view_kind.addItems(list(VIEWS))
        for cb in (self.view_axle, self.view_kind):
            cb.currentTextChanged.connect(lambda _t: self._draw_view())
            row.addWidget(cb)
        row.addWidget(QtWidgets.QLabel("droop / ride / bump; left corner"))
        row.addStretch(1)
        vl.addLayout(row)
        self.p_view = plot("", "", "", legend=True)
        self.p_view.setAspectLocked(True)
        vl.addWidget(self.p_view, 1)
        rtabs.addTab(view, "Linkage view")
        split.addWidget(right)
        split.setSizes([600, 840])

        self.report: S.SuspensionReport | None = None
        self.job = Debounced(250, self)
        self.job.failed.connect(lambda m: self._message(f"Failed: {html.escape(m[:400])}"))

    # -- seed helper ---------------------------------------------------------------------
    def _seed_box(self, axle: str) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Rebuild the rocker arms for a target (construction helper)")
        box.setToolTip("Keeps the rod's outer end and the rocker pivot above, places both "
                       "arms so the rod and the damper act along their pickups' motion at ride "
                       "height, sizes the damper arm for the target MR and sets the damper's "
                       "ride length for equal droop and bump margins. The target is your "
                       "query argument; the result is a starting point you then own.")
        form = QtWidgets.QFormLayout(box)
        spins: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key, lab, lo, hi, step, val, dec in (
                ("mr", "Target MR at ride", 0.2, 2.0, 0.05, 0.80, 3),
                ("rod_arm", "Rod arm length", 15, 200, 5, 50, 1),
                ("tx", "Damper points toward X", -5000, 5000, 10, 0, 1),
                ("ty", "Damper points toward Y", -5000, 5000, 10, 0, 1),
                ("tz", "Damper points toward Z", -5000, 5000, 10, 0, 1)):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(dec)
            sb.setValue(val)
            form.addRow(lab, sb)
            spins[key] = sb
        self.seed[axle] = spins
        form.addRow(button("Rebuild arms", lambda a=axle: self._rebuild(a), primary=True))
        return box

    def _sync_seed(self) -> None:
        """'Damper points toward' follows the current damper line unless edited."""
        for axle in AXLES:
            spec = getattr(self.ctrl.project, f"rocker_{axle}")
            if not spec.enabled:
                continue
            d0, d1 = np.array(spec.rocker_damper), np.array(spec.damper_chassis)
            far = d1 + 2.0 * (d1 - d0)
            for k, v in zip(("tx", "ty", "tz"), far, strict=True):
                sb = self.seed[axle][k]
                sb.blockSignals(True)
                sb.setValue(float(v))
                sb.blockSignals(False)

    def _rebuild(self, axle: str) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        spec = getattr(self.ctrl.project, f"rocker_{axle}")
        sp = self.seed[axle]
        toward = (sp["tx"].value(), sp["ty"].value(), sp["tz"].value())
        try:
            base = S.build_rocker(spec.rod_mount, spec.rocker_pivot, toward, sp["rod_arm"].value(),
                                  60.0, 195.0, spec.mount_body, spec.rod_type)
            new = S.tune_rocker(geo, self.ctrl.project, axle, base, sp["mr"].value(),
                                sp["rod_arm"].value(), toward)
        except Exception as exc:          # noqa: BLE001 -- shown to the designer, not swallowed
            self._message(f"Rebuild failed: {html.escape(str(exc))}")
            return

        def edit(p: Project) -> None:
            setattr(p, f"rocker_{axle}", replace(new, rod_type=spec.rod_type))

        self.ctrl.apply(edit, geometry=False)

    def _set_available(self) -> None:
        try:
            vals = tuple(sorted({float(v) for v in self.avail.text().replace(";", ",").split(",")
                                 if v.strip()}))
        except ValueError:
            self._message("Springs available: give numbers separated by commas.")
            return
        if not vals or vals == self.ctrl.project.springs.available_lb_in:
            return

        def edit(p: Project) -> None:
            p.springs = replace(p.springs, available_lb_in=vals)

        self.ctrl.apply(edit, geometry=False)

    # -- compute ---------------------------------------------------------------------------
    def on_project(self) -> None:
        for f in self.forms:
            f.refresh()
        self.avail.setText(", ".join(f"{v:g}" for v in self.ctrl.project.springs.available_lb_in))
        self._sync_seed()
        self._run()

    def on_geometry(self, geo: CarGeometry) -> None:
        self._run()

    def _rc(self) -> dict[str, float]:
        ks = self.ctrl.static
        return {"front": ks.value("front.rc_static"), "rear": ks.value("rear.rc_static")}

    def _run(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        project, rc = self.ctrl.project, self._rc()

        def work(progress, cancelled):
            return S.suspension_report(geo, project, rc)

        self.job.request(work, self._show)

    def _message(self, text: str) -> None:
        self.warn.label.setText(text)
        self.warn.setVisible(bool(text))

    # -- show ------------------------------------------------------------------------------
    def _show(self, rep: S.SuspensionReport) -> None:
        self.report = rep
        d = self.ctrl.project.damper
        msgs = [f"{a.axle}: {html.escape(a.error)}" for a in (rep.front, rep.rear) if a.error]
        for a in (rep.front, rep.rear):
            if a.ok and min(a.droop_margin_mm, a.bump_margin_mm) < 0:
                msgs.append(f"{a.axle}: the damper runs out of travel "
                            f"(droop margin {a.droop_margin_mm:+.1f} mm, bump margin "
                            f"{a.bump_margin_mm:+.1f} mm)")
            if a.ok and a.clearance_mm < 10.0:
                msgs.append(f"{a.axle}: rod passes {a.clearance_mm:.1f} mm (centrelines) from "
                            f"the {html.escape(a.clearance_with)}")
        if d.stroke_source == "estimate":
            msgs.append(f"Damper stroke {d.stroke_mm:g} mm is an ESTIMATE: measure the exposed "
                        "shaft; the travel margins depend on it.")
        self._message("<br>".join(msgs))

        def col(fn):
            return [fn(a) if a.ok else "—" for a in (rep.front, rep.rear)]

        rows = [
            ["Motion ratio at ride (damper / wheel)", *col(lambda a: _f(a.mr_ride, ".3f"))],
            ["MR at full droop → full bump", *col(lambda a: f"{a.mr_droop:.3f} → "
                                                             f"{a.mr_bump:.3f}")],
            ["Progression over the travel", *col(lambda a: _f(a.progression_pct, "+.1f", "%"))],
            ["Wheel travel used (droop / bump)", *col(lambda a: f"{a.droop_mm:+.0f} / "
                                                              f"{a.bump_mm:+.0f} mm")],
            ["Damper at ride", *col(lambda a: _f(a.damper_ride_mm, ".1f", "mm"))],
            ["Damper at full droop / full bump", *col(lambda a: f"{a.damper_droop_mm:.1f} / "
                                                               f"{a.damper_bump_mm:.1f} mm")],
            [f"Margin to extended ({d.eye_to_eye_extended_mm:g} mm)",
             *col(lambda a: _f(a.droop_margin_mm, "+.1f", "mm"))],
            [f"Margin to compressed ({d.eye_to_eye_extended_mm - d.stroke_mm + d.bump_rubber_mm:g}"
             " mm)", *col(lambda a: _f(a.bump_margin_mm, "+.1f", "mm"))],
            ["Damper stroke used / usable", *col(lambda a: f"{a.stroke_used_mm:.1f} / "
                                                          f"{a.stroke_usable_mm:.1f} mm")],
            ["Rocker rotation (droop → bump)", *col(lambda a: f"{a.psi_range_deg[0]:+.1f}° → "
                                                             f"{a.psi_range_deg[1]:+.1f}°")],
            ["Worst transmission angle rod / damper", *col(
                lambda a: f"{a.rod_angle_max_deg:.0f}° / {a.damper_angle_max_deg:.0f}°")],
            ["Rod length", *col(lambda a: _f(a.rod_length_mm, ".1f", "mm"))],
            ["Rod force per N of wheel load", *col(lambda a: _f(a.rod_per_wheel, ".3f"))],
            ["Sprung corner mass", *col(lambda a: _f(a.sprung_corner_kg, ".1f", "kg"))],
            ["Spring force at ride (fitted spring)", *col(lambda a: _f(a.spring_force_ride_N,
                                                                       ".0f", "N"))],
            ["Spring compression at ride", *col(lambda a: _f(a.spring_compression_ride_mm,
                                                             ".1f", "mm"))],
            ["Rod force at ride (compression +)", *col(lambda a: _f(a.rod_force_ride_N, ".0f",
                                                                    "N"))],
            ["Closest member to the rod", *col(lambda a: f"{a.clearance_mm:.1f} mm, "
                                                        f"{a.clearance_with}")],
        ]
        self.table.set_rows(rows, right_cols=(1, 2))

        if rep.front.ok and rep.rear.ok:
            fr = {o.lb_in: o for o in rep.front.springs}
            rr = {o.lb_in: o for o in rep.rear.springs}
            srows = []
            for lb in sorted(set(fr) | set(rr)):
                f, r = fr.get(lb), rr.get(lb)
                fit = " (fitted " + "/".join(x for x, o in (("F", f), ("R", r))
                                             if o is not None and o.fitted) + ")" \
                    if (f and f.fitted) or (r and r.fitted) else ""
                srows.append([f"{lb:g} lb/in = {S.n_mm(lb):.1f} N/mm{fit}",
                              f"{_f(f.wheel_rate_n_mm if f else math.nan)} / "
                              f"{_f(r.wheel_rate_n_mm if r else math.nan)}",
                              f"{_f(f.frequency_hz if f else math.nan, '.2f')} / "
                              f"{_f(r.frequency_hz if r else math.nan, '.2f')}",
                              f"{_f(f.roll_rate_nm_deg if f else math.nan, '.0f')} / "
                              f"{_f(r.roll_rate_nm_deg if r else math.nan, '.0f')}"])
            self.rates.set_rows(srows, right_cols=(1, 2, 3))
        else:
            self.rates.set_rows([])
        b = rep.budget
        if b is None:
            self.budget.set_rows([["Roll budget", "needs both rockers and the static roll "
                                                  "centres"]])
        else:
            def arb(x: float) -> str:
                return f"{x:.0f} N·m/deg" if x >= 0 else f"none (springs exceed by {-x:.0f})"
            self.budget.set_rows([
                ["Target roll gradient (vehicle data, design_intent)",
                 f"{b.gradient_deg_per_g:.2f} deg/g"],
                ["Roll stiffness that gradient needs (incl. pendulum)",
                 f"{b.required_nm_deg:.0f} N·m/deg"],
                ["From the fitted springs, front / rear",
                 f"{b.springs_nm_deg['front']:.0f} / {b.springs_nm_deg['rear']:.0f} N·m/deg"],
                ["Gradient with the springs alone", f"{_f(b.gradient_springs_only, '.2f')} deg/g"],
                [f"Anti-roll bar needed at a {100 * b.front_share:.0f} % front split, front",
                 arb(b.arb_needed_nm_deg["front"])],
                ["… rear", arb(b.arb_needed_nm_deg["rear"])],
            ])
        self._draw_curves()
        self._draw_view()

    def _draw_curves(self) -> None:
        t = theme.current()
        d = self.ctrl.project.damper
        self.p_mr.clear()
        self.p_len.clear()
        if self.report is None:
            return
        for i, a in enumerate((self.report.front, self.report.rear)):
            if not a.ok or a.sweep is None:
                continue
            self.p_mr.plot(a.sweep.travel_mm, a.sweep.motion_ratio, pen=pen(i), name=a.axle)
            self.p_len.plot(a.sweep.travel_mm, a.sweep.damper_length_mm, pen=pen(i),
                            name=a.axle)
        for y, lab in ((d.eye_to_eye_extended_mm, "extended"),
                       (d.eye_to_eye_extended_mm - d.stroke_mm + d.bump_rubber_mm,
                        "compressed")):
            self.p_len.addItem(pg.InfiniteLine(y, angle=0, pen=pg.mkPen(t.bad, width=1.5),
                                               label=lab, labelOpts={"color": t.bad,
                                                                     "position": 0.5}))

    def _draw_view(self) -> None:
        p = self.p_view
        p.clear()
        geo = self.ctrl.geometry
        if self.report is None or geo is None:
            return
        a = self.report.front if self.view_axle.currentText() == "front" else self.report.rear
        if not a.ok or a.sweep is None:
            return
        hi, vi = VIEWS[self.view_kind.currentText()]
        t = theme.current()
        corner = geo.corner(S.AXLE_CORNER[a.axle])
        from vdcore.geometry.solver import DWSolver

        s = DWSolver(corner)
        zs = a.sweep.travel_mm
        picks = [(0, "droop", 1), (int(np.argmin(np.abs(zs))), "ride", 0), (len(zs) - 1, "bump",
                                                                             2)]

        def pt(h) -> np.ndarray:
            return np.array([h.x_mm, h.y_mm, h.z_mm])

        def line(p1, p2, pn, name=None):
            p.plot([p1[hi], p2[hi]], [p1[vi], p2[vi]], pen=pn, name=name)

        for k, (i, label, ci) in enumerate(picks):
            st = a.sweep.states[i]
            res = s.solve(wheel_travel_mm=float(zs[i]))
            lift = np.array([0.0, 0.0, float(zs[i])])
            ubj, lbj, tro = (pt(x) + lift for x in (res.ubj, res.lbj, res.tro))
            w = 2.5 if label == "ride" else 1.2
            grey = pg.mkPen(t.text_muted, width=w)
            for pin in (corner.uca_inboard_front, corner.uca_inboard_rear):
                line(pt(pin), ubj, grey)
            for pin in (corner.lca_inboard_front, corner.lca_inboard_rear):
                line(pt(pin), lbj, grey)
            line(ubj, lbj, grey)
            line(pt(corner.tie_rod_inboard), tro, grey)
            line(st.rod_mount, st.rocker_rod, pen(ci, w), name=f"rod, {label}")
            o = np.array(a.spec.rocker_pivot)
            for q in (st.rocker_rod, st.rocker_damper):
                line(o, q, pen(ci, w))
            line(st.rocker_damper, np.array(a.spec.damper_chassis),
                 pg.mkPen(t.warn, width=w), name="damper" if k == 1 else None)
        o = np.array(a.spec.rocker_pivot)
        p.plot([o[hi]], [o[vi]], pen=None, symbol="o", symbolSize=8, symbolBrush=t.text)
