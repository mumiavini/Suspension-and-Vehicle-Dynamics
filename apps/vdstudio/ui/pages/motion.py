"""3D motion: drive heave, roll, pitch and steering live; every frame is a DWSolver solve."""

from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ...engine.curves import positive_rack_turns_left
from ...engine.geometry import CarGeometry
from ...engine.pose import Pose, PoseSolver, wheel_ring
from .. import theme
from ..views import CarGL, FrontView, gl_available
from ..widgets import SimpleTable, banner, button
from .base import Page

TYRE_WIDTH_MM = 190.0      # CarGL / FrontView drawing default


def steer_label(left_for_positive: bool | None) -> str:
    """Steering-wheel slider label from the +1 mm rack probe (+SW angle = +rack)."""
    if left_for_positive is None:
        return "Steering-wheel angle (turn direction unknown: +1 mm rack probe did not solve)"
    return "Steering-wheel angle (+ = left turn)" if left_for_positive else \
        "Steering-wheel angle (+ = right turn: steering arms behind the axle)"


def _mark_front_view(view: FrontView, geo: CarGeometry, pose: Pose, axle: str,
                     colour: str) -> None:
    """Outline in ``colour`` the tyres of this axle's corners posed beyond design travel."""
    r = geo.axles[axle].loaded_radius_mm
    for cid in ("FL", "FR") if axle == "front" else ("RL", "RR"):
        c = pose.corners[cid]
        if not c.beyond_travel:
            continue
        cam = math.radians(c.camber_deg)
        side = 1.0 if cid in ("FL", "RL") else -1.0
        up = np.array([side * math.sin(cam), math.cos(cam)])
        out = np.array([side * math.cos(cam), -math.sin(cam)])
        wc = c.points["WHEEL_CENTER"][1:]
        view.line(np.array([wc + a * r * up + b * TYRE_WIDTH_MM / 2 * out
                            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))]), colour, 3)
        view.text(wc + np.array([-200.0, r + 25.0]), "beyond travel", colour)


class _Axis(QtWidgets.QWidget):
    """Slider + spin box pair; emits float values."""

    changed = QtCore.Signal(float)

    def __init__(self, label: str, unit: str, lo: float, hi: float, dec: int = 1) -> None:
        super().__init__()
        self.scale = 10 ** dec
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lab = QtWidgets.QLabel(label)
        self.label = lab
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
        self.steer = _Axis(steer_label(True), "deg", -180, 180, 0)
        for a in (self.heave, self.roll, self.pitch, self.steer):
            a.changed.connect(lambda _v: self._update())
            bl.addWidget(a)
        self.rack_label = QtWidgets.QLabel("")
        self.rack_label.setObjectName("muted")
        self.rack_label.setWordWrap(True)
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
        self.readout = SimpleTable(self._headers(False))
        self.readout.setMinimumHeight(170)
        ll.addWidget(self.readout, 1)
        left.setMinimumWidth(400)
        split.addWidget(left)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        ok, why = gl_available()
        self.gl: CarGL | None = None
        self._beyond_gl = None
        if ok:
            try:
                self.gl = CarGL()
                right.addWidget(self.gl)
                # tint for corners beyond design travel, drawn over the car
                self._beyond_gl = self.gl.gl.GLLinePlotItem(mode="lines", width=4,
                                                            antialias=True)
                self.gl.view.addItem(self._beyond_gl)
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
        self._left_for_positive: bool | None = True
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._tick)
        self._anim = ""
        self._phase = 0.0

    # -- geometry ------------------------------------------------------------------------
    @staticmethod
    def _headers(steered: bool) -> list[str]:
        # a steered wheel's Y-Z projected camber reads ~1/cos(steer) too large: show inclination
        q = "incl.°" if steered else "camber°"
        return ["", "travel mm", f"{q} (chassis)", f"{q} (road)", "toe°/side", "caster°", "KPI°",
                "design travel"]

    def on_geometry(self, geo: CarGeometry) -> None:
        self.solver = PoseSolver(geo)
        self._left_for_positive = positive_rack_turns_left(geo)
        self.steer.label.setText(steer_label(self._left_for_positive))
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
        steered = abs(rack) > 1e-9
        self.rack_label.setText(f"rack travel {rack:+.2f} mm "
                                f"(limit ±{geo.steering_inputs.max_rack_travel_mm:g} mm)"
                                + ("; incl. = inclination (wheel plane to vertical)"
                                   if steered else ""))
        pose = self.solver.solve(self.heave.value(), self.roll.value(), self.pitch.value(), rack)
        self.pose = pose
        if self.gl is not None:
            self.gl.show_labels = self.cb_labels.isChecked()
            self.gl.show_chassis = self.cb_chassis.isChecked()
            self.gl.show_rc = self.cb_rc.isChecked()
            self.gl.draw(geo, pose)
        self.fv_front.draw(geo, pose, "front")
        self.fv_rear.draw(geo, pose, "rear")
        t = theme.current()
        self._mark_beyond_travel(geo, pose, t.bad)
        rows = []
        for cid, c in pose.corners.items():
            inp = geo.axle_inputs("front" if cid[0] == "F" else "rear")
            limit = (f"beyond −{inp.travel_droop_mm:g}…+{inp.travel_bump_mm:g}"
                     if c.beyond_travel else "")
            if not c.converged:
                rows.append([cid, f"{c.travel_mm:+.1f}", "no solve", "—", "—", "—", "—", limit])
                continue
            # inclination relative to the road: the spin axis carried into the world
            s_world = pose.rotation @ c.spin_axis
            road = -math.degrees(math.asin(float(np.clip(s_world[2], -1.0, 1.0))))
            rows.append([cid, f"{c.travel_mm:+.1f}", f"{c.inclination_deg:+.3f}", f"{road:+.3f}",
                         f"{c.toe_deg_per_side:+.3f}", f"{c.caster_deg:.2f}",
                         f"{c.kpi_deg:.2f}", limit])
        self.readout.setHorizontalHeaderLabels(self._headers(steered))
        for j in (2, 3):
            self.readout.horizontalHeaderItem(j).setToolTip(
                "inclination (wheel plane to vertical); DWSolver's camber is projected on the "
                "vehicle Y-Z plane and reads about 1/cos(steer) too large on a steered wheel"
                if steered else "camber (equals the inclination when not steered)")
        self.readout.set_rows(rows, right_cols=range(1, 7))
        tint = QtGui.QColor(theme.status_color("out"))
        tint.setAlpha(90)
        for i, c in enumerate(pose.corners.values()):
            if c.beyond_travel:
                for j in range(self.readout.columnCount()):
                    self.readout.item(i, j).setBackground(tint)

    def _mark_beyond_travel(self, geo: CarGeometry, pose: Pose, colour: str) -> None:
        """Tint the corners posed outside [-droop, +bump]: DWSolver still assembles them."""
        for view, axle in ((self.fv_front, "front"), (self.fv_rear, "rear")):
            _mark_front_view(view, geo, pose, axle, colour)
        if self._beyond_gl is None or self.gl is None:
            return
        seg = []
        for cid, c in pose.corners.items():
            if not c.beyond_travel:
                continue
            r = geo.axles["front" if cid[0] == "F" else "rear"].loaded_radius_mm
            for off in (-TYRE_WIDTH_MM / 2, TYRE_WIDTH_MM / 2):
                ring = wheel_ring(c.points["WHEEL_CENTER"] + off * c.spin_axis, c.spin_axis, r)
                for a, b in zip(ring[:-1], ring[1:], strict=True):
                    seg += [a, b]
        if seg:
            self._beyond_gl.setData(pos=pose.to_world(np.array(seg)),
                                    color=CarGL._color(colour, 1.0))
        self._beyond_gl.setVisible(bool(seg))

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
            # the body rolls out of the turn: a left turn lifts the left side (+roll)
            sign = -1.0 if self._left_for_positive is False else 1.0
            self.roll.set(sign * 1.5 * math.sin(self._phase - 0.35))
        self._update()

    def shutdown(self) -> None:
        self._timer.stop()

    def hideEvent(self, e) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(e)
