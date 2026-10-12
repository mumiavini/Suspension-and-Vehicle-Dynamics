"""Quasi-static maneuver: steady ay / ax, load transfer split, wheel loads and road cambers."""

from __future__ import annotations

import html
import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

import sla_geometry as sla

from ...engine.geometry import CarGeometry
from ...engine.maneuver import ASSUMPTIONS, ManeuverInputs, ManeuverResult, run_maneuver
from .. import theme
from ..asyncjob import Debounced
from ..widgets import SimpleTable, banner, button, pen, plot
from .base import Page

LEG_CASES = ("Cornering", "Braking", "Combined")
NOT_IN_EQUILIBRIUM = ("not computed: two-force UCA cannot react the Fx moment (equilibrium "
                      "not satisfied)")
LEGS_SCOPE = ("Screening only: zero-caster upright, UCA as two-force member, tyre μ assumed; "
              "out-of-plane LBJ load (≈ Fz) assumed taken by the pushrod; not shown.")
BEYOND_TRAVEL = ("travel outside the axle's design bump/droop range: bump stop and droop "
                 "limit are not modelled")


def _num(x: float, fmt: str) -> str:
    return format(x, fmt) if math.isfinite(x) else "—"


class ManeuverPage(Page):
    title = "Quasi-static maneuver"
    subtitle = ("One steady state of lateral and longitudinal g. Not a dynamics simulation: no "
                "transients, no tyre forces. The roll/pitch gradients and the elastic roll-"
                "stiffness split are design-intent inputs; the LLTD that results is reported, "
                "never chosen.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.root.addWidget(split, 1)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        box = QtWidgets.QGroupBox("Steady state")
        form = QtWidgets.QFormLayout(box)
        self.inp: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key, lab, lo, hi, step, val, unit in (
                ("ay_g", "Lateral acceleration ay (+ = left turn)", -3, 3, 0.1, 1.5, "g"),
                ("ax_g", "Longitudinal acceleration ax (− = braking)", -3, 3, 0.1, 0.0, "g"),
                ("roll_gradient_deg_per_g", "Roll gradient (design_intent)", 0, 5, 0.05, 1.0,
                 "deg/g"),
                ("pitch_gradient_deg_per_g", "Pitch gradient (design_intent)", 0, 5, 0.05, 0.5,
                 "deg/g"),
                ("roll_stiffness_front_share", "Elastic roll stiffness, front share", 0, 1, 0.02,
                 0.5, "-")):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setSingleStep(step)
            sb.setDecimals(3)
            sb.setValue(val)
            sb.setSuffix(f" {unit}")
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda _v: (self._run(), self._sweep()))
            form.addRow(lab, sb)
            self.inp[key] = sb
        self.rc_src = QtWidgets.QComboBox()
        self.rc_src.addItems(["solved", "design"])
        self.rc_src.currentTextChanged.connect(lambda _t: (self._run(), self._sweep()))
        form.addRow("Roll centre heights from", self.rc_src)
        form.addRow(button("Use the vehicle's roll-gradient target", self._use_target))
        ll.addWidget(box)
        assumptions = QtWidgets.QLabel(
            f"Assumptions: {ASSUMPTIONS}. Lateral transfer = geometric + elastic (incl. the "
            "pendulum term W_s·d·φ) + unsprung, on the sprung CG.")
        assumptions.setObjectName("muted")
        assumptions.setWordWrap(True)
        ll.addWidget(assumptions)
        self.summary = SimpleTable(["quantity", "value"])
        ll.addWidget(self.summary, 1)
        self.warn = banner("")
        ll.addWidget(self.warn)
        left.setMinimumWidth(430)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.wheels = SimpleTable(["wheel", "Fz [N]", "travel [mm]", "camber vs road [deg]",
                                   "camber vs chassis [deg]", "toe/side [deg]"])
        self.wheels.setMaximumHeight(160)
        rl.addWidget(self.wheels)
        self.transfer = SimpleTable(["axle", "geometric [N]", "elastic incl. pendulum [N]",
                                     "unsprung [N]", "total ΔFz [N]"])
        self.transfer.setMaximumHeight(110)
        rl.addWidget(self.transfer)
        sweep_row = QtWidgets.QHBoxLayout()
        sweep_row.addWidget(QtWidgets.QLabel("Sweep ay from 0 to the value above:"))
        sweep_row.addWidget(button("Run sweep", self._sweep))
        sweep_row.addStretch(1)
        rl.addLayout(sweep_row)
        g = QtWidgets.QHBoxLayout()
        self.p_fz = plot("Wheel loads vs ay", "ay [g]", "Fz [N]")
        self.p_cam = plot("Camber vs road vs ay", "ay [g]", "camber [deg]")
        g.addWidget(self.p_fz)
        g.addWidget(self.p_cam)
        rl.addLayout(g, 1)
        self.legs = SimpleTable(["axle / case", "LCA front", "LCA rear", "UCA front", "UCA rear",
                                 "Fz outer [N]"])
        self.legs.setMaximumHeight(170)
        self.legs_note = QtWidgets.QLabel("Wishbone leg forces from sla_geometry load cases "
                                          "(N, + tension). " + LEGS_SCOPE)
        self.legs_note.setWordWrap(True)
        rl.addWidget(self.legs_note)
        rl.addWidget(self.legs)
        split.addWidget(right)
        split.setSizes([440, 1000])
        self.job = Debounced(300, self)
        self.job.failed.connect(self._sweep_failed)

    def _inputs(self) -> ManeuverInputs:
        v = {k: sb.value() for k, sb in self.inp.items()}
        return ManeuverInputs(rc_source=self.rc_src.currentText(), **v)

    def _use_target(self) -> None:
        self.inp["roll_gradient_deg_per_g"].setValue(
            self.ctrl.project.vehicle.target_roll_gradient_deg_per_g)

    def on_geometry(self, geo: CarGeometry) -> None:
        self._run()
        self._sweep()

    def _rc(self) -> dict[str, float]:
        ks = self.ctrl.static
        return {"front": ks.value("front.rc_static"), "rear": ks.value("rear.rc_static")}

    def _run(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        inp = self._inputs()
        try:
            r = run_maneuver(geo, inp, self._rc())
        except (RuntimeError, ValueError) as exc:
            # no table may keep numbers from a state the inputs no longer describe
            self._clear()
            msg = f"Failed: {html.escape(str(exc))}"
            if inp.rc_source == "solved":
                for axle in ("front", "rear"):
                    k = self.ctrl.static.get(f"{axle}.rc_static")
                    if k is not None and k.error:
                        msg += f"<br>{axle}.rc_static: {html.escape(k.error)}"
            self.warn.label.setText(msg)
            return
        self._show(r)
        self._legs(geo, r)

    def _clear(self) -> None:
        for t in (self.summary, self.wheels, self.transfer, self.legs):
            t.clearSpans()
            t.set_rows([])

    def _show(self, r: ManeuverResult) -> None:
        rows, beyond = [], []
        for i, (cid, w) in enumerate(r.wheels.items()):
            rows.append([cid, _num(w.fz_N, ".0f"), _num(w.travel_mm, "+.2f"),
                         "—" if not w.converged else f"{w.camber_road_deg:+.3f}",
                         "—" if not w.converged else f"{w.camber_chassis_deg:+.3f}",
                         "—" if not w.converged else f"{w.toe_deg_per_side:+.4f}"])
            if not w.in_design_travel:
                beyond.append(i)
        self.wheels.set_rows(rows, right_cols=range(1, 6))
        muted = QtGui.QColor(theme.current().text_muted)
        for i in beyond:
            for j in range(self.wheels.columnCount()):
                it = self.wheels.item(i, j)
                it.setForeground(muted)
                it.setToolTip(BEYOND_TRAVEL)
        self.transfer.set_rows([[a, f"{d['geometric']:.0f}", f"{d['elastic']:.0f}",
                                 f"{d['unsprung']:.0f}", f"{d['total']:.0f}"]
                                for a, d in r.transfer.items()], right_cols=range(1, 5))
        k = r.k_roll_total_Nm_per_deg
        k_txt = (f"{k:.0f} N·m/deg" if math.isfinite(k) else
                 "rigid (gradient 0)" if k == math.inf else "undefined (see warning)")
        sprung = [] if r.sprung is None else [
            ["Sprung CG height / station", f"{r.sprung.cg_height_mm:.1f} mm / "
                                           f"{r.sprung.cg_from_front_axle_mm:.1f} mm aft"]]
        self.summary.set_rows([
            ["Roll angle", f"{r.roll_deg:+.2f} deg"],
            ["Pitch angle (+ nose down)", f"{r.pitch_deg:+.2f} deg"],
            ["Resulting LLTD (front share, any ay)", _num(r.lltd_front, ".3f")],
            *sprung,
            ["Roll axis height at sprung CG", f"{r.roll_axis_at_cg_mm:.1f} mm"],
            ["Roll moment arm d (sprung CG)", f"{r.roll_moment_arm_mm:.1f} mm"],
            ["Total roll stiffness implied (W_s·d·(1/gradient + π/180), incl. pendulum)",
             k_txt],
            ["Longitudinal transfer", f"{r.longitudinal_transfer_N:+.0f} N (to rear +)"],
            ["RC heights used", f"F {r.rc_mm['front']:.1f} / R {r.rc_mm['rear']:.1f} mm "
                                f"({self.rc_src.currentText()})"],
        ])
        self.warn.label.setText("<br>".join(html.escape(w) for w in r.warnings) if r.warnings
                                else "All four wheels loaded at this state.")

    def _sweep(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        base = self._inputs()
        rc = self._rc()
        ays = np.linspace(0.0, base.ay_g, 11)

        def work(progress, cancelled):
            out = []
            for i, ay in enumerate(ays):
                if cancelled():
                    raise InterruptedError
                b = ManeuverInputs(**{**base.__dict__, "ay_g": float(ay)})
                out.append(run_maneuver(geo, b, rc))
                progress((i + 1) / len(ays), "")
            return out

        self.job.request(work, lambda res: self._draw_sweep(ays, res))

    def _sweep_failed(self, msg: str) -> None:
        self.p_fz.clear()
        self.p_cam.clear()
        if not self.warn.label.text().startswith("Failed:"):
            self.warn.label.setText("sweep failed: " + html.escape(msg[:200]))

    def _draw_sweep(self, ays: np.ndarray, res: list[ManeuverResult]) -> None:
        self.p_fz.clear()
        self.p_cam.clear()
        for i, cid in enumerate(("FL", "FR", "RL", "RR")):
            self.p_fz.plot(ays, [r.wheels[cid].fz_N for r in res], pen=pen(i), name=cid)
            self.p_cam.plot(ays, [r.wheels[cid].camber_road_deg for r in res], pen=pen(i),
                            name=cid)
        self.p_fz.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(theme.current().bad)))

    def _legs(self, geo: CarGeometry, r: ManeuverResult) -> None:
        """Leg forces at the outer-wheel loads of the state on screen."""
        self.legs.clearSpans()
        if geo.design is None:
            self.legs.set_rows([["available in synthesis mode only", "", "", "", "", ""]])
            self.legs_note.setText("Wishbone leg forces from sla_geometry load cases (N, + "
                                   "tension). " + LEGS_SCOPE)
            return
        rows: list[list[str]] = []
        spans: list[int] = []
        outer_txt = []
        veh = geo.vehicle_data
        for name, ag, cids in (("Front", geo.design.front, ("FL", "FR")),
                               ("Rear", geo.design.rear, ("RL", "RR"))):
            outer = max(cids, key=lambda c: r.wheels[c].fz_N)
            fz = r.wheels[outer].fz_N
            outer_txt.append(f"{outer} {fz:.0f} N")
            if not fz > 0:
                rows += [[f"{name} / {case}", "not computed: outer wheel not loaded", "", "", "",
                          _num(fz, ".0f")] for case in LEG_CASES]
                spans += [len(rows) - 3, len(rows) - 2, len(rows) - 1]
                continue
            # sla_geometry's load cases are linear in the outer-wheel Fz. Take their
            # shape (μ from vehicle data; at ay = 0 its Fz is the static load, > 0) and
            # evaluate it at the Fz of the state above, not at sla's own lumped
            # W·ay·h/T transfer, so both tables on this page use one wheel load.
            lc = sla.load_cases(ag, veh, ay=0.0)
            ref = sla.upright_reactions(ag)
            lbj, ubj, cp = ref["lbj"], ref["ubj"], ref["cp"]
            y_l, y_u = float(ag.lca_in[0]), float(ag.uca_in[0])
            lf = np.array([ag.lca_in_front_x_mm, y_l, ag.lca_in_front_z_mm])
            lr = np.array([ag.lca_in_rear_x_mm, y_l, ag.lca_in_rear_z_mm])
            uf = np.array([ag.uca_in_front_x_mm, y_u, ag.uca_in_front_z_mm])
            ur = np.array([ag.uca_in_rear_x_mm, y_u, ag.uca_in_rear_z_mm])
            for case in LEG_CASES:
                load = lc[case] * (fz / lc["Fz"])
                F_lbj, F_ubj = sla.solve_ball_joint_forces(ag, load)
                # moment of the loads on the upright about the LBJ: the two-force UCA
                # reacts no x-force, so any Fx leaves it unbalanced
                m = np.cross(cp - lbj, load) + np.cross(ubj - lbj, -F_ubj)
                if float(np.linalg.norm(m)) > 1.0:          # N·mm
                    rows.append([f"{name} / {case}", NOT_IN_EQUILIBRIUM, "", "", "",
                                 f"{fz:.0f}"])
                    spans.append(len(rows) - 1)
                    continue
                fl = sla.leg_forces(lbj, lf, lr, F_lbj)
                fu = sla.leg_forces(ubj, uf, ur, F_ubj)
                rows.append([f"{name} / {case}", f"{fl[0]:.0f}", f"{fl[1]:.0f}", f"{fu[0]:.0f}",
                             f"{fu[1]:.0f}", f"{fz:.0f}"])
        self.legs.set_rows(rows, right_cols=range(1, 6))
        for i in spans:
            self.legs.setSpan(i, 1, 1, 4)
        inp = r.inputs
        self.legs_note.setText(
            f"Wishbone leg forces (N, + tension) for the sla_geometry load cases (μ lateral "
            f"{veh.mu_lateral:g}, longitudinal {veh.mu_longitudinal:g}: vehicle data) at the "
            f"outer-wheel loads of the state above (ay {inp.ay_g:+.2f} g, ax {inp.ax_g:+.2f} g, "
            f"LLTD {_num(r.lltd_front, '.3f')}: {', '.join(outer_txt)}). " + LEGS_SCOPE)
