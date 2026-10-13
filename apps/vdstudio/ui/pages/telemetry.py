"""Telemetry: load a log, map channels, validate the data, compare the car with the model."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...engine.project import SOURCES, Project, TelemetrySettings
from ...engine.telemetry import analysis as A
from ...engine.telemetry import mapping as M
from ...engine.telemetry.session import Session, load_session
from .. import theme
from ..asyncjob import Debounced
from ..widgets import SimpleTable, button, pen, plot
from .base import Page


class TelemetryPage(Page):
    title = "Telemetry & validation"
    subtitle = ("Two questions kept apart: is the data right (quality and physics checks that "
                "need no model), and does the car do what the design says (gradients against "
                "the design-intent targets, kinematic replay of the measured travel through "
                "DWSolver). MoTeC .ld and CSV logs.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(button("Load log (.ld / .csv)…", self._load, primary=True))
        bar.addWidget(button("Auto-map channels", self._automap))
        self.info = QtWidgets.QLabel("No log loaded.")
        self.info.setObjectName("muted")
        bar.addWidget(self.info, 1)
        self.root.addLayout(bar)
        self.tabs = QtWidgets.QTabWidget()
        self.root.addWidget(self.tabs, 1)
        self.session: Session | None = None
        self.sig: dict[str, np.ndarray] = {}
        self.travel: dict[str, np.ndarray] = {}
        self.maps: A.KinematicMaps | None = None
        self._maps_geo = None
        # SW -> mean road-wheel steer, solved on DWSolver, cached per geometry
        self._steer_fn = None
        self._steer_geo = None
        self._steer_err = ""
        self.steer_zero: dict[str, float] = {}
        self._build_mapping()
        self._build_quality()
        self._build_channels()
        self._build_vehicle()
        self._build_model()
        self._build_replay()
        self.job = Debounced(0, self)
        self.job.busy.connect(lambda b: self.replay_status.setText(
            "solving kinematic maps on DWSolver…" if b else self.replay_status.text()))
        self.job.failed.connect(lambda m: self.replay_status.setText("failed: " + m[:200]))

    # ------------------------------------------------------------------ construction
    def _build_mapping(self) -> None:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        self.map_table = QtWidgets.QTableWidget()
        self.map_table.setColumnCount(6)
        self.map_table.setHorizontalHeaderLabels(["Role", "App unit", "Channel", "× scale",
                                                  "flip sign", "why this scale"])
        self.map_table.verticalHeader().setVisible(False)
        self.map_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView
                                                               .ResizeToContents)
        self.map_table.horizontalHeader().setStretchLastSection(True)
        self.map_table.setRowCount(len(M.ROLES))
        self._combos: dict[str, QtWidgets.QComboBox] = {}
        self._scales: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._flips: dict[str, QtWidgets.QCheckBox] = {}
        for i, r in enumerate(M.ROLES):
            self.map_table.setItem(i, 0, QtWidgets.QTableWidgetItem(r.label))
            self.map_table.setItem(i, 1, QtWidgets.QTableWidgetItem(r.unit))
            co = QtWidgets.QComboBox()
            # side swaps are the classic logging error: the full name must be readable
            co.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
            co.addItem("—", "")
            co.currentIndexChanged.connect(lambda _i, k=r.key: self._map_changed(k))
            self.map_table.setCellWidget(i, 2, co)
            sc = QtWidgets.QDoubleSpinBox()
            sc.setDecimals(6)
            sc.setRange(-1e6, 1e6)
            sc.setValue(1.0)
            sc.setKeyboardTracking(False)
            sc.valueChanged.connect(lambda _v, k=r.key: self._map_changed(k))
            self.map_table.setCellWidget(i, 3, sc)
            fl = QtWidgets.QCheckBox()
            fl.toggled.connect(lambda _b, k=r.key: self._map_changed(k))
            self.map_table.setCellWidget(i, 4, fl)
            self.map_table.setItem(i, 5, QtWidgets.QTableWidgetItem(""))
            self._combos[r.key], self._scales[r.key], self._flips[r.key] = co, sc, fl
        lay.addWidget(self.map_table, 3)
        side = QtWidgets.QGroupBox("Interpretation")
        f = QtWidgets.QFormLayout(side)
        self.mr_f = QtWidgets.QDoubleSpinBox()
        self.mr_r = QtWidgets.QDoubleSpinBox()
        for sb in (self.mr_f, self.mr_r):
            sb.setRange(0.05, 5)
            sb.setDecimals(3)
            sb.setSingleStep(0.05)
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda _v: self._settings_changed())
        self.mr_src = QtWidgets.QComboBox()
        self.mr_src.addItems(SOURCES)
        self.mr_src.currentTextChanged.connect(lambda _t: self._settings_changed())
        self.comp_pos = QtWidgets.QCheckBox("damper channel positive in compression")
        self.comp_pos.toggled.connect(lambda _b: self._settings_changed())
        self.min_v = QtWidgets.QDoubleSpinBox()
        self.min_v.setRange(0, 80)
        self.min_v.setSuffix(" m/s")
        self.min_ay = QtWidgets.QDoubleSpinBox()
        self.min_ay.setRange(0, 3)
        self.min_ay.setSingleStep(0.05)
        self.min_ay.setSuffix(" g")
        for sb in (self.min_v, self.min_ay):
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(lambda _v: self._settings_changed())
        self.zero_mode = QtWidgets.QComboBox()
        self.zero_mode.addItems(["median of straight running", "as logged (sensor zeroed)"])
        self.zero_mode.setToolTip("Dampers: median while straight and steady. Steering: median "
                                  "while straight (|ay| < 0.05 g, |yaw| < 1 deg/s).")
        self.zero_mode.currentIndexChanged.connect(lambda _i: self._recompute())
        self.ratio_override = QtWidgets.QDoubleSpinBox()
        self.ratio_override.setRange(0, 50)
        self.ratio_override.setDecimals(2)
        self.ratio_override.setSpecialValueText("from geometry")
        self.ratio_override.setKeyboardTracking(False)
        self.ratio_override.valueChanged.connect(lambda _v: self._settings_changed())
        f.addRow("Damper / wheel motion ratio, front", self.mr_f)
        f.addRow("Damper / wheel motion ratio, rear", self.mr_r)
        f.addRow("Motion ratio provenance", self.mr_src)
        f.addRow(self.comp_pos)
        f.addRow("Zero reference (dampers, steering)", self.zero_mode)
        f.addRow("Steady-state: min speed", self.min_v)
        f.addRow("Steady-state: min |ay|", self.min_ay)
        f.addRow("Steering ratio", self.ratio_override)
        f.addRow(QtWidgets.QLabel(
            "After scaling and flips every analysis assumes ISO 8855: ay, yaw rate and "
            "steering positive in a LEFT turn. The Data-quality tab checks that claim."))
        lay.addWidget(side, 2)
        self.tabs.addTab(w, "Channel mapping")

    def _build_quality(self) -> None:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(QtWidgets.QLabel("<b>Physics consistency</b> (no model involved)"))
        self.checks = SimpleTable(["check", "status", "value", "expected", "meaning"])
        self.checks.setMinimumHeight(190)       # every check row visible without scrolling
        lay.addWidget(self.checks)
        lay.addWidget(QtWidgets.QLabel("<b>Channel quality</b>"))
        self.quality = SimpleTable(["channel", "unit", "rate Hz", "missing %", "min", "max",
                                    "longest flat s", "spikes", "at limit %", "issues"])
        lay.addWidget(self.quality, 1)
        self.tabs.addTab(w, "Data quality")

    def _build_channels(self) -> None:
        w = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        self.ch_list = QtWidgets.QListWidget()
        self.ch_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.ch_list.itemSelectionChanged.connect(self._draw_channels)
        self.x_axis = QtWidgets.QComboBox()
        self.x_axis.addItems(["time [s]", "distance [m]"])
        self.x_axis.currentIndexChanged.connect(lambda _i: self._draw_channels())
        ll.addWidget(QtWidgets.QLabel("Channels (multi-select):"))
        ll.addWidget(self.ch_list, 1)
        ll.addWidget(self.x_axis)
        w.addWidget(left)
        self.ch_plots = pg.GraphicsLayoutWidget()
        w.addWidget(self.ch_plots)
        w.setSizes([260, 1100])
        self.tabs.addTab(w, "Channels")

    def _build_vehicle(self) -> None:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        self.p_gg = plot("GG diagram", "ay [g] (+ left)", "ax [g] (+ accelerating)", False)
        self.p_gg.setAspectLocked(True)
        self.p_map = plot("Track map (dead reckoning from speed + yaw rate)", "x [m]", "y [m]",
                          False)
        self.p_map.setAspectLocked(True)
        lay.addWidget(self.p_gg)
        lay.addWidget(self.p_map)
        self.tabs.addTab(w, "GG & track")

    def _build_model(self) -> None:
        w = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(w)
        self.p_roll = plot("Roll (from dampers) vs ay", "ay [g]", "roll [deg]")
        self.p_pitch = plot("Pitch (from dampers) vs ax", "ax [g]", "pitch [deg] (+ nose down)")
        self.p_us = plot("Understeer: δ_road − L/R vs |ay|", "|ay| [g]", "[deg]")
        self.p_lltd = plot("Measured LLTD vs |ay| (if wheel loads are logged)", "|ay| [g]",
                           "front share")
        g.addWidget(self.p_roll, 0, 0)
        g.addWidget(self.p_pitch, 0, 1)
        g.addWidget(self.p_us, 1, 0)
        g.addWidget(self.p_lltd, 1, 1)
        self.model_table = SimpleTable(["quantity", "measured", "R² / n", "design intent",
                                        "notes"])
        self.model_table.setMinimumHeight(190)
        self.model_table.setMaximumHeight(230)
        g.addWidget(self.model_table, 2, 0, 1, 2)
        g.setRowStretch(0, 1)
        g.setRowStretch(1, 1)
        self.tabs.addTab(w, "Model vs measurement")

    def _build_replay(self) -> None:
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(button("Run kinematic replay", self._replay, primary=True))
        self.replay_status = QtWidgets.QLabel(
            "Pushes the measured wheel travel (dampers ÷ motion ratio) and rack (steering ÷ "
            "pinion) through camber/toe maps solved on DWSolver for the CURRENT geometry.")
        self.replay_status.setWordWrap(True)
        row.addWidget(self.replay_status, 1)
        lay.addLayout(row)
        g = QtWidgets.QGridLayout()
        self.p_cam_t = plot("Camber vs road over the log", "time [s]", "camber [deg]")
        self.p_hist = plot("Camber vs road while cornering (|ay| above threshold)",
                           "camber [deg]", "share of cornering time [%]")
        self.p_toe_t = plot("Toe per side over the log (steer + bump steer)", "time [s]",
                            "toe [deg] (+ in)")
        g.addWidget(self.p_cam_t, 0, 0)
        g.addWidget(self.p_hist, 0, 1)
        g.addWidget(self.p_toe_t, 1, 0)
        self.replay_table = SimpleTable(["corner", "mean road camber, outer",
                                         "mean road camber, inner", "outer-camber band share",
                                         "outside solved envelope %", "travel range mm"])
        g.addWidget(self.replay_table, 1, 1)
        lay.addLayout(g, 1)
        self.tabs.addTab(w, "Kinematic replay")

    # ------------------------------------------------------------------ project <-> UI
    def on_project(self) -> None:
        tel = self.ctrl.project.telemetry
        for sb, v in ((self.mr_f, tel.damper_motion_ratio_front),
                      (self.mr_r, tel.damper_motion_ratio_rear), (self.min_v, tel.min_speed_mps),
                      (self.min_ay, tel.min_abs_ay_g),
                      (self.ratio_override, tel.steering_ratio_override or 0.0)):
            sb.blockSignals(True)
            sb.setValue(float(v))
            sb.blockSignals(False)
        self.mr_src.blockSignals(True)
        self.mr_src.setCurrentText(tel.motion_ratio_source)
        self.mr_src.blockSignals(False)
        self.comp_pos.blockSignals(True)
        self.comp_pos.setChecked(tel.damper_compression_positive)
        self.comp_pos.blockSignals(False)
        self._fill_mapping()

    def _settings_changed(self) -> None:
        def edit(p: Project) -> None:
            p.telemetry = replace(
                p.telemetry, damper_motion_ratio_front=self.mr_f.value(),
                damper_motion_ratio_rear=self.mr_r.value(),
                motion_ratio_source=self.mr_src.currentText(),
                damper_compression_positive=self.comp_pos.isChecked(),
                min_speed_mps=self.min_v.value(), min_abs_ay_g=self.min_ay.value(),
                steering_ratio_override=self.ratio_override.value() or None)

        self.ctrl.apply(edit, geometry=False)
        self._recompute()

    def _fill_mapping(self) -> None:
        tel = self.ctrl.project.telemetry
        names = self.session.names() if self.session else []
        for r in M.ROLES:
            co = self._combos[r.key]
            co.blockSignals(True)
            co.clear()
            co.addItem("—", "")
            for n in names:
                co.addItem(n, n)
            ch = tel.channel_map.get(r.key, "")
            i = co.findData(ch)
            co.setCurrentIndex(i if i >= 0 else 0)
            co.blockSignals(False)
            sc = self._scales[r.key]
            sc.blockSignals(True)
            sc.setValue(float(tel.scale.get(r.key, 1.0)))
            sc.blockSignals(False)
            fl = self._flips[r.key]
            fl.blockSignals(True)
            fl.setChecked(bool(tel.flip_sign.get(r.key, False)))
            fl.blockSignals(False)
        self.map_table.resizeColumnsToContents()

    def _map_changed(self, role: str) -> None:
        ch = self._combos[role].currentData() or ""
        sc = self._scales[role].value()
        fl = self._flips[role].isChecked()

        def edit(p: Project) -> None:
            t = p.telemetry
            cm, scale, flip = dict(t.channel_map), dict(t.scale), dict(t.flip_sign)
            if ch:
                cm[role] = ch
            else:
                cm.pop(role, None)
            scale[role] = sc
            flip[role] = fl
            p.telemetry = replace(t, channel_map=cm, scale=scale, flip_sign=flip)

        self.ctrl.apply(edit, geometry=False)
        self._recompute()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load log", str(Path.cwd()), "Logs (*.ld *.csv *.tsv *.txt)")
        if path:
            self.load_path(path)

    def load_path(self, path: str) -> None:
        try:
            s = load_session(path)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Telemetry", f"Could not read {path}:\n{exc}")
            return
        self.session = s
        meta = ", ".join(f"{k}: {v}" for k, v in s.meta.items() if k != "source")
        self.info.setText(f"<b>{s.name}</b> — {len(s.channels)} channels, {s.duration_s:.1f} s "
                          f"at {s.rate_hz:.1f} Hz. {meta}")
        self.ch_list.clear()
        self.ch_list.addItems(s.names())
        tel = self.ctrl.project.telemetry
        if not any(ch in s.channels for ch in tel.channel_map.values()):
            self._automap()
        else:
            self._fill_mapping()
            self._recompute()

    def _automap(self) -> None:
        if self.session is None:
            return
        s = self.session
        alts: dict[str, list[str]] = {}
        cm = M.auto_map(s.names(), alts)
        scale, why = {}, {}
        for role, ch in cm.items():
            scale[role], why[role] = M.guess_scale(role, s.units.get(ch, ""), s.get(ch))
            if role in alts:
                why[role] += f"; REVIEW: also matches {', '.join(alts[role][:3])}"

        def edit(p: Project) -> None:
            p.telemetry = replace(p.telemetry, channel_map=cm, scale=scale, flip_sign={})

        self.ctrl.apply(edit, geometry=False)
        self._fill_mapping()
        for i, r in enumerate(M.ROLES):
            self.map_table.item(i, 5).setText(why.get(r.key, ""))
        self._recompute()

    # ------------------------------------------------------------------ analysis
    def _tel(self) -> TelemetrySettings:
        return self.ctrl.project.telemetry

    def _recompute(self) -> None:
        s = self.session
        if s is None:
            return
        tel = self._tel()
        self.sig = M.signals(s, tel.channel_map, tel.scale, tel.flip_sign)
        geo = self.ctrl.geometry
        zero_on = self.zero_mode.currentIndex() == 0 and bool(self.sig)
        zero = A.straight_line_zero(self.sig, tel) if zero_on else {}
        self.travel = A.wheel_travel(self.sig, tel, zero) if self.sig else {}
        # the same combo zeroes the steering (replay, understeer, checks see it)
        self.steer_zero = A.straight_line_steer_zero(self.sig, tel) if zero_on else {}
        for role, z in self.steer_zero.items():
            self.sig[role] = self.sig[role] - z
        rate = s.rate_hz
        q = A.channel_quality(s, self.sig.get("speed"))
        rows = [[c.name, c.unit, f"{c.rate_hz:.1f}", f"{c.nan_pct:.1f}", f"{c.vmin:.4g}",
                 f"{c.vmax:.4g}", f"{c.longest_flat_s:.1f}", str(c.spikes),
                 f"{c.saturated_pct:.1f}", "; ".join(c.issues)] for c in q]
        rows += [[n, "", "", "", "", "", "", "", "", f"not read: {why}"]
                 for n, why in s.skipped]
        self.quality.set_rows(rows, right_cols=range(2, 9))
        for i, r in enumerate(rows):
            if r[9]:
                it = self.quality.item(i, 9)
                it.setForeground(pg.mkColor(theme.current().warn))
        if self.sig:
            ratio = self._ratio()
            checks = A.consistency_checks(self.sig, tel, rate, geo, self.travel, ratio)
            if zero_on and ("steer_sw" in self.sig or "steer_road" in self.sig):
                for role in ("steer_sw", "steer_road"):
                    if role not in self.sig:
                        continue
                    z = self.steer_zero.get(role)
                    rw = f" (≈ {z / ratio:+.3f} deg road wheel)" if z is not None \
                        and role == "steer_sw" and math.isfinite(ratio) and ratio > 0 else ""
                    checks.append(A.Check(
                        f"{role} zero (straight running)", "n/a" if z is None else "ok",
                        "—" if z is None else f"{z:+.3f} deg{rw}", "—",
                        "no straight running found: steering used as logged" if z is None
                        else "subtracted from the channel; choose 'as logged' to see the raw "
                        "offset"))
            self.checks.set_rows([[c.name, c.status, c.value, c.expected, c.detail]
                                  for c in checks], status_col=1)
        self._draw_vehicle()
        self._draw_model()
        self._draw_channels()

    def _draw_channels(self) -> None:
        s = self.session
        self.ch_plots.clear()
        if s is None:
            return
        names = [it.text() for it in self.ch_list.selectedItems()][:8]
        if not names:
            return
        use_dist = self.x_axis.currentIndex() == 1 and "distance" in self.sig
        x = self.sig["distance"] if use_dist else s.time_s
        first = None
        for i, n in enumerate(names):
            p = self.ch_plots.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("left", n, units=s.units.get(n, ""))
            p.plot(x, s.channels[n], pen=pen(i, 1.5), connect="finite")
            if first is None:
                first = p
            else:
                p.setXLink(first)
        if first is not None:
            first.setTitle("x: " + ("distance [m]" if use_dist else "time [s]"), size="8pt")

    def _draw_vehicle(self) -> None:
        self.p_gg.clear()
        self.p_map.clear()
        sig = self.sig
        if "ay" in sig and "ax" in sig:
            m = np.isfinite(sig["ay"]) & np.isfinite(sig["ax"])
            self.p_gg.plot(sig["ay"][m], sig["ax"][m], pen=None, symbol="o", symbolSize=3,
                           symbolBrush=pg.mkBrush(theme.current().series[0]),
                           symbolPen=pg.mkPen(None))
        if self.session is not None:
            xy = A.track_map(sig, self.session.rate_hz)
            if xy is not None:
                self.p_map.plot(xy[0], xy[1], pen=pen(0, 2))
                self.p_map.plot([xy[0][0]], [xy[1][0]], pen=None, symbol="o", symbolSize=10,
                                symbolBrush=pg.mkBrush(theme.current().ok))

    def _ratio(self) -> float:
        tel = self._tel()
        if tel.steering_ratio_override:
            return float(tel.steering_ratio_override)
        return self.ctrl.kpis.value("steering.ratio")

    def _sw_to_delta(self, geo):
        """Solved SW -> road-wheel table for ``geo`` (None + ``_steer_err`` if it failed)."""
        if self._steer_geo is not geo:
            self._steer_geo = geo
            try:
                self._steer_fn, self._steer_err = A.steer_table(geo), ""
            except (RuntimeError, ValueError) as exc:
                self._steer_fn, self._steer_err = None, str(exc)
        return self._steer_fn

    def _draw_model(self) -> None:
        s, geo = self.session, self.ctrl.geometry
        for p in (self.p_roll, self.p_pitch, self.p_us, self.p_lltd):
            p.clear()
        if s is None or geo is None or not self.sig:
            return
        tel = self._tel()
        rate = s.rate_hz
        target = self.ctrl.project.vehicle.target_roll_gradient_deg_per_g
        rows = []

        def scatter(pw, gr: A.GradientResult, i: int) -> None:
            pw.plot(gr.x, gr.y, pen=None, symbol="o", symbolSize=3,
                    symbolBrush=pg.mkBrush(theme.current().series[i]), symbolPen=pg.mkPen(None),
                    name=gr.name)
            if math.isfinite(gr.fit.slope):
                xx = np.array([np.nanmin(gr.x), np.nanmax(gr.x)])
                pw.plot(xx, gr.fit.slope * xx + gr.fit.intercept, pen=pen(i, 2.5))

        if self.travel:
            for i, gr in enumerate(A.roll_gradients(self.sig, self.travel, geo, tel, rate,
                                                    target)):
                scatter(self.p_roll, gr, i)
                rows.append([gr.name, f"{gr.fit.slope:.3f} {gr.unit}",
                             f"{gr.fit.r2:.3f} / {gr.fit.n}", f"{target:g} deg/g (target)",
                             gr.note])
            xx = np.array([-2.5, 2.5])
            self.p_roll.plot(xx, target * xx, pen=pen(7, 1.5, QtCore.Qt.DashLine),
                             name="design-intent gradient")
            for k, pg_ in enumerate(A.pitch_gradient(self.sig, self.travel, geo, tel, rate)):
                scatter(self.p_pitch, pg_, 2 + 3 * k)
                rows.append([pg_.name, f"{pg_.fit.slope:.3f} {pg_.unit}",
                             f"{pg_.fit.r2:.3f} / {pg_.fit.n}", "—", pg_.note])
        # δ from the solved linkage unless the user overrides the ratio (then linear)
        table = None
        if "steer_sw" in self.sig and "steer_road" not in self.sig \
                and not tel.steering_ratio_override:
            table = self._sw_to_delta(geo)
            if table is None:
                rows.append(["Understeer gradient", "failed", "", "—",
                             f"SW -> road-wheel table: {self._steer_err}"])
        if table is not None or "steer_road" in self.sig or tel.steering_ratio_override:
            us = A.understeer_gradient(self.sig, tel, rate, self._ratio(), geo.wheelbase_mm,
                                       sw_to_delta=table, steer_zero=self.steer_zero)
            if us is not None:
                scatter(self.p_us, us, 3)
                rows.append([us.name, f"{us.fit.slope:+.3f} {us.unit} (intercept "
                             f"{us.fit.intercept:+.2f} deg)", f"{us.fit.r2:.3f} / {us.fit.n}",
                             "—", us.note])
        lt = A.measured_lltd(self.sig, tel, rate, (geo.track_mm("front"), geo.track_mm("rear")))
        if lt is not None:
            scatter(self.p_lltd, lt, 4)
            s_m = self.ctrl.project.vehicle.lltd_front
            tf, tr = geo.track_mm("front"), geo.track_mm("rear")
            s_f = (s_m / tf) / (s_m / tf + (1.0 - s_m) / tr)
            rows.append([lt.name, f"{lt.force_share:.3f} force / {lt.moment_share:.3f} moment",
                         f"n {lt.fit.n}", f"{s_f:.3f} force / {s_m:.2f} moment (load-case "
                         "assumption; sla_geometry uses it as a moment share)", lt.note])
        if not rows:
            rows = [["Map ay, speed, dampers (and steering) to compare the car with the model",
                     "", "", "", ""]]
        self.model_table.set_rows(rows)

    # ------------------------------------------------------------------ replay
    def _replay(self) -> None:
        s, geo = self.session, self.ctrl.geometry
        if s is None or geo is None:
            self.replay_status.setText("Load a log first.")
            return
        if not self.travel:
            self.replay_status.setText("Map the four damper channels first.")
            return
        if self.maps is not None and self._maps_geo is geo:
            self._show_replay()
            return

        def work(progress, cancelled):
            return A.build_maps(geo, cancelled=cancelled)

        def done(maps: A.KinematicMaps) -> None:
            self.maps = maps
            self._maps_geo = geo
            self._show_replay()

        self.job.request(work, done, immediate=True)

    def _show_replay(self) -> None:
        s, geo, maps = self.session, self.ctrl.geometry, self.maps
        if s is None or geo is None or maps is None:
            return
        r = A.replay(s, self.sig, self.travel, maps, geo, self._tel())
        for p in (self.p_cam_t, self.p_hist, self.p_toe_t):
            p.clear()
        rows = []
        bands = {ax: geo.axle_inputs(ax).limits.outer_camber_in_roll_deg
                 for ax in ("front", "rear")}
        corner_mask = r.cornering
        ay_sign = np.sign(self.sig.get("ay", np.zeros(len(r.time_s))))
        for i, cid in enumerate(("FL", "FR", "RL", "RR")):
            if cid not in r.camber_road:
                continue
            cr = r.camber_road[cid]
            self.p_cam_t.plot(r.time_s, cr, pen=pen(i, 1.2), name=cid, connect="finite")
            self.p_toe_t.plot(r.time_s, r.toe[cid], pen=pen(i, 1.2), name=cid, connect="finite")
            sel = cr[corner_mask & np.isfinite(cr)]
            if sel.size:
                h, edges = np.histogram(sel, bins=30)
                self.p_hist.plot(edges, 100.0 * h / sel.size, stepMode="center",
                                 pen=pen(i, 1.5), name=cid)
            # "outer" while cornering: the side the load moved to (ay > 0 -> right wheels);
            # a wheel's outer and inner samples are reported apart, never blended
            side = 1.0 if cid in ("FL", "RL") else -1.0
            fin = corner_mask & np.isfinite(cr)
            oc = cr[fin & (ay_sign == -side)]
            ic = cr[fin & (ay_sign == side)]
            band = bands["front" if cid[0] == "F" else "rear"]
            share = 100.0 * np.mean((oc >= band[0]) & (oc <= band[1])) if oc.size else math.nan
            tr = r.travel[cid]
            rows.append([cid, f"{np.mean(oc):+.2f} deg" if oc.size else "—",
                         f"{np.mean(ic):+.2f} deg" if ic.size else "—",
                         f"{share:.0f} %" if math.isfinite(share) else "—",
                         f"{r.outside_pct[cid]:.1f}",
                         f"{np.nanmin(tr):+.1f} … {np.nanmax(tr):+.1f}"])
        self.replay_table.set_rows(rows)
        bf, br = bands["front"], bands["rear"]
        failed = (f" {maps.failed} grid solves did not converge; samples near them are NaN "
                  "(counted as outside the envelope)." if maps.failed else "")
        self.replay_status.setText(
            f"Replayed {len(r.time_s)} samples. 'Outer-camber band share' = share of the time "
            f"that wheel was the outer one while cornering with road camber inside "
            f"{bf[0]:g} … {bf[1]:g} deg front, {br[0]:g} … {br[1]:g} deg rear (design intent). "
            "Front camber is the wheel inclination to the vertical (the wheels are steered). "
            "Samples outside the solved travel envelope are excluded, never clamped." + failed)

    def on_geometry(self, geo) -> None:
        if self.session is not None:
            self._recompute()
