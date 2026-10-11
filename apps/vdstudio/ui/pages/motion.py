"""3D motion: drive heave, roll, pitch and steering live; every frame is a DWSolver solve."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from ...engine.geometry import CarGeometry
from ...engine.pose import Pose, PoseSolver
from .. import theme
from ..views import CarGL, FrontView, gl_available
from ..widgets import SimpleTable, banner, button
from .base import Page


class _Axis(QtWidgets.QWidget):
    """Slider + spin box pair; emits float values."""

    changed = QtCore.Signal(float)

    def __init__(self, label: str, unit: str, lo: float, hi: float, dec: int = 1) -> None:
        super().__init__()
        self.scale = 10 ** dec
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lab = QtWidgets.QLabel(label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(dec)
        self.spin.setSuffix(f" {unit}")
        self.spin.setFixedWidth(110)
        self.set_range(lo, hi)
        lay.addWidget(lab, 0, 0, 1, 2)
        lay.addWidget(self.slider, 1, 0)
        lay.addWidget(self.spin, 1, 1)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def set_range(self, lo: float, hi: float) -> None:
        self.slider.setRange(int(lo * self.scale), int(hi * self.scale))
        self.spin.setRange(lo, hi)

    def _from_slider(self, v: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(v / self.scale)
        self.spin.blockSignals(False)
        self.changed.emit(v / self.scale)

    def _from_spin(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * self.scale)))
        self.slider.blockSignals(False)
        self.changed.emit(v)

    def value(self) -> float:
        return self.spin.value()

    def set(self, v: float, emit: bool = False) -> None:
        self.spin.blockSignals(not emit)
        self.slider.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(int(round(v * self.scale)))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)


class MotionPage(Page):
    title = "3D motion"
    subtitle = ("Move the chassis and the steering wheel; each frame solves all four corners on "
                "DWSolver. Wheel travel per corner follows the pose to first order (patches stay "
                "on the road within a fraction of a mm). Quantitative roll numbers come from the "
                "exact solve on the Kinematics / Maneuver pages.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.root.addWidget(split, 1)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        box = QtWidgets.QGroupBox("Chassis && steering")
        bl = QtWidgets.QVBoxLayout(box)
        self.heave = _Axis("Heave (chassis down = bump)", "mm", -40, 40, 1)
        self.roll = _Axis("Roll (+ = left side up, i.e. a left turn)", "deg", -4, 4, 2)
        self.pitch = _Axis("Pitch (+ = nose down)", "deg", -3, 3, 2)
        self.steer = _Axis("Steering-wheel angle (+ = left)", "deg", -180, 180, 0)
        for a in (self.heave, self.roll, self.pitch, self.steer):
            a.changed.connect(lambda _v: self._update())
            bl.addWidget(a)
        self.rack_label = QtWidgets.QLabel("")
        self.rack_label.setObjectName("muted")
        bl.addWidget(self.rack_label)
        bl.addWidget(button("Reset pose", self._reset))
        ll.addWidget(box)

        anim = QtWidgets.QGroupBox("Animate")
        al = QtWidgets.QGridLayout(anim)
        for i, (name, key) in enumerate((("Heave cycle", "heave"), ("Roll cycle", "roll"),
                                         ("Pitch cycle", "pitch"), ("Steer lock-to-lock", "steer"),
                                         ("Slalom (steer + roll)", "slalom"),
                                         ("Stop", "stop"))):
            al.addWidget(button(name, lambda k=key: self._animate(k)), i // 2, i % 2)
        ll.addWidget(anim)

        vbox = QtWidgets.QGroupBox("View")
        vl = QtWidgets.QGridLayout(vbox)
        for i, (name, key) in enumerate((("Iso", "iso"), ("Front", "front"), ("Side", "side"),
                                         ("Top", "top"), ("FL corner", "corner"))):
            vl.addWidget(button(name, lambda k=key: self._preset(k)), 0 if i < 3 else 1, i % 3)
        self.cb_labels = QtWidgets.QCheckBox("corner labels")
        self.cb_chassis = QtWidgets.QCheckBox("chassis rails")
        self.cb_chassis.setChecked(True)
        self.cb_rc = QtWidgets.QCheckBox("roll centres && axis")
        self.cb_rc.setChecked(True)
        for i, cb in enumerate((self.cb_labels, self.cb_chassis, self.cb_rc)):
            cb.toggled.connect(lambda _b: self._update())
            vl.addWidget(cb, 2 + i, 0, 1, 3)
        ll.addWidget(vbox)
        self.readout = SimpleTable(["", "travel mm", "camber° (chassis)", "camber° (road)",
                                    "toe°/side", "caster°", "KPI°"])
        self.readout.setMinimumHeight(170)
        ll.addWidget(self.readout, 1)
        left.setMinimumWidth(400)
        split.addWidget(left)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        ok, why = gl_available()
        self.gl: CarGL | None = None
        if ok:
            try:
                self.gl = CarGL()
                right.addWidget(self.gl)
            except Exception as exc:  # no OpenGL context on this machine
                right.addWidget(banner(f"3D view unavailable: {exc}"))
        else:
            right.addWidget(banner(why))
        fv_box = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(fv_box)
        fl.setContentsMargins(0, 0, 0, 0)
        self.fv_front = FrontView()
        self.fv_rear = FrontView()
        self.fv_front.getPlotItem().setTitle("Front axle at this pose (chassis frame)", size="9pt")
        self.fv_rear.getPlotItem().setTitle("Rear axle at this pose (chassis frame)", size="9pt")
        fl.addWidget(self.fv_front)
        fl.addWidget(self.fv_rear)
        right.addWidget(fv_box)
        right.setSizes([620, 300])
        split.addWidget(right)
        split.setSizes([420, 1100])

        self.solver: PoseSolver | None = None
        self.pose: Pose | None = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self._anim = ""
        self._phase = 0.0

    # -- geometry ------------------------------------------------------------------------
    def on_geometry(self, geo: CarGeometry) -> None:
        self.solver = PoseSolver(geo)
        fi, ri = geo.front_inputs, geo.rear_inputs
        self.heave.set_range(-max(fi.travel_droop_mm, ri.travel_droop_mm) - 10,
                             max(fi.travel_bump_mm, ri.travel_bump_mm) + 10)
        lock = self._sw_lock(geo)
        self.steer.set_range(-lock, lock)
        self._update()

    def _sw_lock(self, geo: CarGeometry) -> float:
        s = geo.steering_inputs
        return 360.0 * s.max_rack_travel_mm / (2 * math.pi * s.pinion_radius_mm) \
            if s.pinion_radius_mm > 0 else 180.0

    def _rack(self) -> float:
        geo = self.ctrl.geometry
        if geo is None:
            return 0.0
        r = geo.steering_inputs.pinion_radius_mm
        return self.steer.value() / 360.0 * 2 * math.pi * r

    def _update(self) -> None:
        geo = self.ctrl.geometry
        if geo is None or self.solver is None or self.solver.geo is not geo:
            if geo is None:
                return
            self.solver = PoseSolver(geo)
        rack = self._rack()
        self.rack_label.setText(f"rack travel {rack:+.2f} mm "
                                f"(limit ±{geo.steering_inputs.max_rack_travel_mm:g} mm)")
        pose = self.solver.solve(self.heave.value(), self.roll.value(), self.pitch.value(), rack)
        self.pose = pose
        if self.gl is not None:
            self.gl.show_labels = self.cb_labels.isChecked()
            self.gl.show_chassis = self.cb_chassis.isChecked()
            self.gl.show_rc = self.cb_rc.isChecked()
            self.gl.draw(geo, pose)
        self.fv_front.draw(geo, pose, "front")
        self.fv_rear.draw(geo, pose, "rear")
        rows = []
        for cid, c in pose.corners.items():
            side = 1.0 if cid in ("FL", "RL") else -1.0
            road = c.camber_deg - side * pose.roll_deg
            if not c.converged:
                rows.append([cid, f"{c.travel_mm:+.1f}", "no solve", "—", "—", "—", "—"])
                continue
            rows.append([cid, f"{c.travel_mm:+.1f}", f"{c.camber_deg:+.3f}", f"{road:+.3f}",
                         f"{c.toe_deg_per_side:+.3f}", f"{c.caster_deg:.2f}",
                         f"{c.kpi_deg:.2f}"])
        self.readout.set_rows(rows, right_cols=range(1, 7))
        _ = theme

    def _reset(self) -> None:
        self._anim = ""
        self._timer.stop()
        for a in (self.heave, self.roll, self.pitch, self.steer):
            a.set(0.0)
        self._update()

    def _preset(self, key: str) -> None:
        if self.gl is not None:
            self.gl.preset(key)

    # -- animation ----------------------------------------------------------------------
    def _animate(self, key: str) -> None:
        if key == "stop":
            self._timer.stop()
            self._anim = ""
            return
        self._anim = key
        self._phase = 0.0
        self._timer.start()

    def _tick(self) -> None:
        geo = self.ctrl.geometry
        if geo is None or not self.isVisible():
            self._timer.stop()
            return
        self._phase += 0.06
        s = math.sin(self._phase)
        k = self._anim
        fi = geo.front_inputs
        if k == "heave":
            self.heave.set(s * min(fi.travel_bump_mm, fi.travel_droop_mm))
        elif k == "roll":
            self.roll.set(2.0 * s)
        elif k == "pitch":
            self.pitch.set(1.5 * s)
        elif k == "steer":
            self.steer.set(s * self._sw_lock(geo))
        elif k == "slalom":
            self.steer.set(0.45 * s * self._sw_lock(geo))
            self.roll.set(1.5 * math.sin(self._phase - 0.35))
        self._update()

    def shutdown(self) -> None:
        self._timer.stop()

    def hideEvent(self, e) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(e)
