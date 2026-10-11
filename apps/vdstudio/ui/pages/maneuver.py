"""Quasi-static maneuver: steady ay / ax, load transfer split, wheel loads and road cambers."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

import sla_geometry as sla

from ...engine.geometry import CarGeometry
from ...engine.maneuver import ManeuverInputs, ManeuverResult, run_maneuver
from .. import theme
from ..asyncjob import Debounced
from ..widgets import SimpleTable, banner, button, pen, plot
from .base import Page


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
        self.transfer = SimpleTable(["axle", "geometric [N]", "elastic [N]", "unsprung [N]",
                                     "total ΔFz [N]"])
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
        rl.addWidget(QtWidgets.QLabel(
            "Wishbone leg forces from sla_geometry load cases (N, + tension). Screening only: "
            "zero-caster upright, UCA as two-force member, no pushrod, tyre μ assumed."))
        rl.addWidget(self.legs)
        split.addWidget(right)
        split.setSizes([440, 1000])
        self.job = Debounced(300, self)
        self.job.failed.connect(lambda m: self.warn.label.setText("sweep failed: " + m[:200]))

    def _inputs(self) -> ManeuverInputs:
        v = {k: sb.value() for k, sb in self.inp.items()}
        return ManeuverInputs(rc_source=self.rc_src.currentText(), **v)

    def _use_target(self) -> None:
        self.inp["roll_gradient_deg_per_g"].setValue(
            self.ctrl.project.vehicle.target_roll_gradient_deg_per_g)

    def on_geometry(self, geo: CarGeometry) -> None:
        self._run()
        self._legs(geo)
        self._sweep()

    def _rc(self) -> dict[str, float]:
        ks = self.ctrl.static
        return {"front": ks.value("front.rc_static"), "rear": ks.value("rear.rc_static")}

    def _run(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        try:
            r = run_maneuver(geo, self._inputs(), self._rc())
        except (RuntimeError, ValueError) as exc:
            self.warn.label.setText(f"Failed: {exc}")
            return
        self._show(r)

    def _show(self, r: ManeuverResult) -> None:
        rows = []
        for cid, w in r.wheels.items():
            rows.append([cid, f"{w.fz_N:.0f}", f"{w.travel_mm:+.2f}",
                         "—" if not w.converged else f"{w.camber_road_deg:+.3f}",
                         "—" if not w.converged else f"{w.camber_chassis_deg:+.3f}",
                         "—" if not w.converged else f"{w.toe_deg_per_side:+.4f}"])
        self.wheels.set_rows(rows, right_cols=range(1, 6))
        self.transfer.set_rows([[a, f"{d['geometric']:.0f}", f"{d['elastic']:.0f}",
                                 f"{d['unsprung']:.0f}", f"{d['total']:.0f}"]
                                for a, d in r.transfer.items()], right_cols=range(1, 5))
        self.summary.set_rows([
            ["Roll angle", f"{r.roll_deg:+.2f} deg"],
            ["Pitch angle (+ nose down)", f"{r.pitch_deg:+.2f} deg"],
            ["Resulting LLTD (front share)", f"{r.lltd_front:.3f}"],
            ["Roll axis height at CG", f"{r.roll_axis_at_cg_mm:.1f} mm"],
            ["Roll moment arm", f"{r.roll_moment_arm_mm:.1f} mm"],
            ["Total roll stiffness implied", f"{r.k_roll_total_Nm_per_deg:.0f} N·m/deg"],
            ["Longitudinal transfer", f"{r.longitudinal_transfer_N:+.0f} N (to rear +)"],
            ["RC heights used", f"F {r.rc_mm['front']:.1f} / R {r.rc_mm['rear']:.1f} mm "
                                f"({self.rc_src.currentText()})"],
        ])
        self.warn.label.setText("<br>".join(r.warnings) if r.warnings else
                                "All four wheels loaded at this state.")

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

    def _draw_sweep(self, ays: np.ndarray, res: list[ManeuverResult]) -> None:
        self.p_fz.clear()
        self.p_cam.clear()
        for i, cid in enumerate(("FL", "FR", "RL", "RR")):
            self.p_fz.plot(ays, [r.wheels[cid].fz_N for r in res], pen=pen(i), name=cid)
            self.p_cam.plot(ays, [r.wheels[cid].camber_road_deg for r in res], pen=pen(i),
                            name=cid)
        self.p_fz.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(theme.current().bad)))

    def _legs(self, geo: CarGeometry) -> None:
        if geo.design is None:
            self.legs.set_rows([["available in synthesis mode only", "", "", "", "", ""]])
            return
        rows = []
        veh = geo.vehicle_data
        for name, ag in (("Front", geo.design.front), ("Rear", geo.design.rear)):
            inp = ag.inputs
            lc = sla.load_cases(ag, veh)
            lbj = np.array([inp.axle_x_mm, float(ag.lbj[0]), float(ag.lbj[1])])
            ubj = np.array([inp.axle_x_mm, float(ag.ubj[0]), float(ag.ubj[1])])
            y_l, y_u = float(ag.lca_in[0]), float(ag.uca_in[0])
            lf = np.array([ag.lca_in_front_x_mm, y_l, ag.lca_in_front_z_mm])
            lr = np.array([ag.lca_in_rear_x_mm, y_l, ag.lca_in_rear_z_mm])
            uf = np.array([ag.uca_in_front_x_mm, y_u, ag.uca_in_front_z_mm])
            ur = np.array([ag.uca_in_rear_x_mm, y_u, ag.uca_in_rear_z_mm])
            for case in ("Cornering", "Braking", "Combined"):
                F_lbj, F_ubj = sla.solve_ball_joint_forces(ag, lc[case])
                fl = sla.leg_forces(lbj, lf, lr, F_lbj)
                fu = sla.leg_forces(ubj, uf, ur, F_ubj)
                rows.append([f"{name} / {case}", f"{fl[0]:.0f}", f"{fl[1]:.0f}", f"{fu[0]:.0f}",
                             f"{fu[1]:.0f}", f"{lc['Fz']:.0f}"])
        self.legs.set_rows(rows, right_cols=range(1, 6))
