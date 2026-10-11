"""Kinematic sweeps: every quantity vs wheel travel (each corner) and vs roll (each axle)."""

from __future__ import annotations

import csv
import io

from PySide6 import QtWidgets

from ...engine import curves as C
from ...engine.geometry import CarGeometry
from ...engine.kpis import KpiSet
from ..asyncjob import Debounced
from ..widgets import KpiTable, band_region, banner, button, pen, plot, save_text
from .base import Page

CORNER_PEN = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}


class KinematicsPage(Page):
    title = "Kinematic sweeps"
    subtitle = ("Heave: each corner solved on DWSolver over its travel (Z values chassis-"
                "referenced). Roll: both wheels on the road at each chassis roll angle; camber "
                "relative to the ROAD. Failed solves are gaps, never interpolated.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        bar = QtWidgets.QHBoxLayout()
        self.t_lo = QtWidgets.QDoubleSpinBox()
        self.t_hi = QtWidgets.QDoubleSpinBox()
        for sb, v in ((self.t_lo, -30.0), (self.t_hi, 30.0)):
            sb.setRange(-120, 120)
            sb.setValue(v)
            sb.setSuffix(" mm")
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda _v: self._request())
        self.n = QtWidgets.QSpinBox()
        self.n.setRange(5, 201)
        self.n.setValue(41)
        self.n.valueChanged.connect(lambda _v: self._request())
        self.roll_max = QtWidgets.QDoubleSpinBox()
        self.roll_max.setRange(0.2, 8)
        self.roll_max.setValue(3.0)
        self.roll_max.setSuffix(" deg")
        self.roll_max.valueChanged.connect(lambda _v: self._request())
        self.corners = {c: QtWidgets.QCheckBox(c) for c in ("FL", "FR", "RL", "RR")}
        for c, cb in self.corners.items():
            cb.setChecked(c in ("FL", "RL"))
            cb.toggled.connect(lambda _b: self._draw())
        bar.addWidget(QtWidgets.QLabel("Travel"))
        bar.addWidget(self.t_lo)
        bar.addWidget(QtWidgets.QLabel("to"))
        bar.addWidget(self.t_hi)
        bar.addWidget(QtWidgets.QLabel("points"))
        bar.addWidget(self.n)
        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel("Roll up to"))
        bar.addWidget(self.roll_max)
        bar.addSpacing(16)
        for cb in self.corners.values():
            bar.addWidget(cb)
        bar.addStretch(1)
        self.status = QtWidgets.QLabel("")
        bar.addWidget(self.status)
        bar.addWidget(button("Export curves CSV…", self._export))
        self.root.addLayout(bar)

        tabs = QtWidgets.QTabWidget()
        heave = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(heave)
        grid.setContentsMargins(4, 4, 4, 4)
        self.hp: dict[str, object] = {}
        for i, (k, (lab, unit)) in enumerate(C.HEAVE_QUANTITIES.items()):
            w = plot(lab, "wheel travel [mm] (+ bump)", unit)
            w.setMinimumHeight(200)
            self.hp[k] = w
            grid.addWidget(w, i // 3, i % 3)
        tabs.addTab(heave, "Heave (bump / droop)")
        roll = QtWidgets.QWidget()
        rg = QtWidgets.QGridLayout(roll)
        self.rp: dict[str, object] = {}
        for i, (k, (lab, unit)) in enumerate(C.ROLL_QUANTITIES.items()):
            w = plot(lab, "chassis roll [deg]", unit)
            self.rp[k] = w
            rg.addWidget(w, i // 3, i % 3)
        rg.addWidget(banner("Outer = wheel in bump. Positive camber relative to the road is "
                            "the tyre leaning out of the corner. Shaded = the outer-camber "
                            "design-intent band."), 1, 2)
        tabs.addTab(roll, "Roll")
        rates = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rates)
        self.table = KpiTable(group_filter=lambda g: g in ("Kinematic rates", "Bump steer")
                              or g.startswith("Roll ("))
        rl.addWidget(self.table)
        tabs.addTab(rates, "Rates table")
        self.root.addWidget(tabs, 1)
        self.job = Debounced(300, self)
        self.job.busy.connect(lambda b: self.status.setText("solving…" if b else ""))
        self.job.failed.connect(lambda m: self.status.setText("failed: " + m.splitlines()[0]))
        self._res: tuple[C.HeaveCurves, C.RollCurves] | None = None
        self._car: CarGeometry | None = None

    def on_project_replaced(self) -> None:
        self._res = None

    def on_geometry(self, geo: CarGeometry) -> None:
        self._car = geo
        self._request()

    def _request(self) -> None:
        geo = self._car
        if geo is None:
            return
        lo, hi, n, rmax = self.t_lo.value(), self.t_hi.value(), self.n.value(), \
            self.roll_max.value()

        def work(progress, cancelled):
            h = C.heave_curves(geo, lo, hi, n, cancelled)
            r = C.roll_curves(geo, rmax, 9, cancelled)
            return h, r

        self.job.request(work, self._done)

    def _done(self, res) -> None:
        self._res = res
        self._draw()

    def _draw(self) -> None:
        if self._res is None:
            return
        h, r = self._res
        for k, w in self.hp.items():
            w.clear()
            for cid, cb in self.corners.items():
                if cb.isChecked():
                    w.plot(h.travel, h.corners[cid][k], pen=pen(CORNER_PEN[cid]), name=cid,
                           connect="finite")
        geo = self._car
        for k, w in self.rp.items():
            w.clear()
            for i, axle in enumerate(("front", "rear")):
                w.plot(r.roll, r.axles[axle][k], pen=pen(2 * i), name=axle, connect="finite")
                if k == "outer_camber" and geo is not None:
                    lo, hi = geo.axle_inputs(axle).limits.outer_camber_in_roll_deg
                    if i == 0:
                        band_region(w, lo, hi)
        if r.errors:
            self.status.setText("roll: " + "; ".join(f"{a}: {e}" for a, e in r.errors.items()))

    def on_kpis(self, ks: KpiSet) -> None:
        self.table.set_kpis(ks)

    def _export(self) -> None:
        if self._res is None:
            return
        h, r = self._res
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        head = ["wheel_travel_mm"] + [f"{c}_{k}" for c in h.corners for k in C.HEAVE_QUANTITIES]
        w.writerow(head)
        for i, t in enumerate(h.travel):
            w.writerow([f"{t:.4f}"] + [f"{h.corners[c][k][i]:.6g}" for c in h.corners
                                       for k in C.HEAVE_QUANTITIES])
        w.writerow([])
        w.writerow(["roll_deg"] + [f"{a}_{k}" for a in r.axles for k in C.ROLL_QUANTITIES])
        for i, phi in enumerate(r.roll):
            w.writerow([f"{phi:.4f}"] + [f"{r.axles[a][k][i]:.6g}" for a in r.axles
                                         for k in C.ROLL_QUANTITIES])
        save_text(self, buf.getvalue(), "kinematic_curves.csv", "CSV (*.csv)")
