"""CAD & cross-checks: SolidWorks skeleton, OptimumK point table, Altair MotionSolve replay."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from PySide6 import QtWidgets

from ...engine import exports as X
from ...engine.project import (
    CAD_AXES,
    CORNERS,
    INPUT_POINTS,
    CadFrame,
    HardpointSet,
    Project,
    with_path,
)
from ..asyncjob import Debounced
from ..widgets import Field, ParamForm, SimpleTable, banner, button, save_text
from .base import Page

# Altair setup-sheet keys -> this app's KPI keys (per axle)
ALTAIR_KEYS = {
    "camber_gain": "camber_gain", "roll_camber": "roll_camber", "rc_static": "rc_static",
    "rc_1g_z": "roll_rc_height", "rc_1g_y": "roll_rc_lateral", "bump_steer": "bump_steer",
    "bump_steer_peak": "bump_steer_peak", "kpi_l": "kpi_L", "kpi_r": "kpi_R",
    "caster_l": "caster_L", "caster_r": "caster_R", "camber_l": "camber_L",
    "camber_r": "camber_R", "scrub_l": "scrub_L", "scrub_r": "scrub_R", "trail_l": "trail_L",
    "trail_r": "trail_R", "sum_toe": "total_toe",
}


class IntegrationsPage(Page):
    title = "CAD & cross-checks"
    subtitle = ("Get the points into CAD and back, and get a second opinion from tools that "
                "share nothing with vdcore but the hardpoints: OptimumK and Altair MotionSolve.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        tabs = QtWidgets.QTabWidget()
        self.root.addWidget(tabs, 1)

        # SolidWorks
        sw = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(sw)
        ok, why = X.solidworks_available()
        sl.addWidget(banner(
            "Draws the suspension as 3D-sketch lines (scripts/export_solidworks_skeleton.py) and "
            "reads them back by topology (scripts/import_solidworks_hardpoints.py). To land the "
            "wireframe on your chassis model, say how that model is placed: which CAD axis "
            "points to the car's front and which to its left (up follows), and the CAD "
            "coordinates of the ISO origin -- the ground, on the car centreline, under the "
            "front axle. A wrong axis gives a rotated or mirrored car without any error."
            + ("" if ok else f"<br><b>Live link unavailable here:</b> {why} The VBA "
                             "macro works on any machine.")))
        e = "cad_frame."
        self.sw_form = ParamForm(ctrl, [("Chassis CAD placement (saved in the project)", [
            Field(e + "forward", "CAD axis to the car's FRONT", kind="choice", choices=CAD_AXES,
                  tip="In SolidWorks, the triad axis that points from the main hoop towards "
                      "the front bulkhead."),
            Field(e + "left", "CAD axis to the car's LEFT", kind="choice", choices=CAD_AXES,
                  tip="Left as seen by the driver."),
            Field(e + "origin_x_mm", "ISO origin in CAD: X", "mm", -1e5, 1e5, 1, 2,
                  tip="CAD coordinates of the point on the ground, on the car centreline, "
                      "under the front axle. 0, 0, 0 = the CAD origin is that point."),
            Field(e + "origin_y_mm", "ISO origin in CAD: Y", "mm", -1e5, 1e5, 1, 2),
            Field(e + "origin_z_mm", "ISO origin in CAD: Z", "mm", -1e5, 1e5, 1, 2),
        ])], setter=self._set_cad)
        sl.addWidget(self.sw_form)
        presets = QtWidgets.QHBoxLayout()
        presets.addWidget(QtWidgets.QLabel("Presets:"))
        presets.addWidget(button("ISO 8855 (X front, Y left, Z up)",
                                 lambda: self._preset("+X", "+Y")))
        presets.addWidget(button("SolidWorks FSAE (X rear, Y right, Z up)",
                                 lambda: self._preset("-X", "-Y")))
        presets.addStretch(1)
        sl.addLayout(presets)
        self.sw_frame_label = QtWidgets.QLabel("")
        self.sw_frame_label.setWordWrap(True)
        sl.addWidget(self.sw_frame_label)
        form = QtWidgets.QFormLayout()
        self.sw_group = QtWidgets.QComboBox()
        self.sw_group.addItems(["corner", "part", "single"])
        form.addRow("One 3D sketch per", self.sw_group)
        sl.addLayout(form)
        row = QtWidgets.QHBoxLayout()
        self.btn_macro = button("Save VBA macro (.bas)…", self._macro, primary=True)
        row.addWidget(self.btn_macro)
        self.btn_push = button("Push skeleton to running SolidWorks", self._push)
        self.btn_pull = button("Read skeleton back → hardpoint set", self._pull)
        self._sw_live = ok
        row.addWidget(self.btn_push)
        row.addWidget(self.btn_pull)
        row.addStretch(1)
        sl.addLayout(row)
        self.sw_status = QtWidgets.QLabel("")
        self.sw_status.setWordWrap(True)
        sl.addWidget(self.sw_status)
        sl.addWidget(QtWidgets.QLabel("Preview: the points as they will land in the CAD model "
                                      "(mm; the macro writes metres)."))
        self.sw_preview = SimpleTable(["corner", "point", "X", "Y", "Z"])
        sl.addWidget(self.sw_preview, 1)
        tabs.addTab(sw, "SolidWorks")

        # OptimumK
        ok_w = QtWidgets.QWidget()
        ol = QtWidgets.QVBoxLayout(ok_w)
        ol.addWidget(banner("Point table in OptimumK's frame (X fwd, Y RIGHT, Z up) from "
                            "scripts/export_hardpoints_for_optimumk.py. Correlate with "
                            "tests/benchmarks/test_optimumk_correlation.py."))
        self.optk = QtWidgets.QPlainTextEdit()
        self.optk.setReadOnly(True)
        f = self.optk.font()
        f.setFamily("Consolas")
        f.setStyleHint(f.StyleHint.Monospace)
        self.optk.setFont(f)
        ol.addWidget(self.optk, 1)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(button("Copy to clipboard", lambda: QtWidgets.QApplication.clipboard()
                            .setText(self.optk.toPlainText())))
        r2.addWidget(button("Save…", lambda: save_text(self, self.optk.toPlainText(),
                                                       "optimumk_points.txt", "Text (*.txt)")))
        r2.addStretch(1)
        ol.addLayout(r2)
        tabs.addTab(ok_w, "OptimumK")

        # Altair
        al_w = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(al_w)
        try:
            from altair_model.msolve_driver import altair_available, altair_python
            self._altair_ok = altair_available()
            where = str(altair_python())
        except Exception as exc:  # pragma: no cover - environment dependent
            self._altair_ok, where = False, str(exc)
        al.addWidget(banner(
            "Altair MotionSolve assembles real joints and integrates an index-3 DAE; vdcore "
            "drives nine distance residuals with least_squares. Both columns run through "
            "vdcore's KPI formulas, so they differ only in where the upright ended up. A full "
            "run takes ~150 s. " + ("Altair found." if self._altair_ok else
                                    f"<b>Altair 2026.1 not found</b> (expected {where}).")))
        self.btn_altair = button("Run MotionSolve cross-check", self._altair, primary=True)
        self.btn_altair.setEnabled(self._altair_ok)
        al.addWidget(self.btn_altair)
        self.altair_status = QtWidgets.QLabel("")
        al.addWidget(self.altair_status)
        self.altair_table = SimpleTable(["axle", "KPI", "vdcore", "MotionSolve", "Δ"])
        al.addWidget(self.altair_table, 1)
        tabs.addTab(al_w, "Altair MotionSolve")
        self.job = Debounced(0, self)
        self.job.progress.connect(lambda f, m: self.altair_status.setText(f"running… {m}"))
        self.job.failed.connect(lambda m: self.altair_status.setText("failed: " + m[:300]))

    def on_geometry(self, geo) -> None:
        self.optk.setPlainText(X.optimumk_text(geo))
        self._preview()

    def on_project(self) -> None:
        self.sw_form.refresh()
        self._preview()

    def _set_cad(self, path: str, value) -> None:
        """CAD placement edits: undoable and saved, but they cannot change the car."""
        self.ctrl.apply(lambda p: with_path(p, path, value), geometry=False)

    def _preset(self, forward: str, left: str) -> None:
        cur = self.ctrl.project.cad_frame
        if (cur.forward, cur.left) == (forward, left):
            return

        def edit(p: Project) -> None:
            p.cad_frame = CadFrame(forward, left, cur.origin_x_mm, cur.origin_y_mm,
                                   cur.origin_z_mm)

        self.ctrl.apply(edit, geometry=False)

    def _frame(self):
        """The CAD placement, or None (with the reason shown) when the axes clash."""
        try:
            fr = X.cad_frame(self.ctrl.project)
        except ValueError as exc:
            self.sw_frame_label.setText(f"<b>Invalid placement:</b> {exc}.")
            return None
        self.sw_frame_label.setText(f"Placement: {fr.label}.")
        return fr

    def _preview(self) -> None:
        fr = self._frame()
        valid = fr is not None
        self.btn_macro.setEnabled(valid)
        self.btn_push.setEnabled(valid and self._sw_live)
        self.btn_pull.setEnabled(valid and self._sw_live)
        geo = self.ctrl.geometry
        if geo is None or fr is None:
            self.sw_preview.set_rows([])
            return
        rows = []
        for c, n, x, y, z, _s, _t in geo.rows():
            X_, Y_, Z_ = X.to_cad((x, y, z), fr)
            rows.append([c, n, f"{X_:.3f}", f"{Y_:.3f}", f"{Z_:.3f}"])
        self.sw_preview.set_rows(rows, right_cols=(2, 3, 4))

    # -- SolidWorks ---------------------------------------------------------------------
    def _macro(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        fr = self._frame()
        if fr is None:
            return
        txt = X.solidworks_macro(geo, fr, self.sw_group.currentText())
        p = save_text(self, txt, "solidworks_skeleton.bas", "VBA module (*.bas)")
        if p:
            self.sw_status.setText(f"Saved {p}. In SolidWorks: Tools › Macro › New, then in the "
                                   "VBA editor File › Import File… › this .bas, run main().")

    def _push(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        try:
            X.solidworks_push(geo, X.cad_frame(self.ctrl.project), self.sw_group.currentText())
            self.sw_status.setText("Skeleton drawn in the active SolidWorks document (not saved).")
        except Exception as exc:
            self.sw_status.setText(f"SolidWorks push failed: {exc}")

    def _pull(self) -> None:
        try:
            pts = X.solidworks_pull(X.cad_frame(self.ctrl.project))
        except Exception as exc:
            self.sw_status.setText(f"SolidWorks read failed: {exc}")
            return
        missing = [f"{c}/{n}" for c in CORNERS for n in INPUT_POINTS if n not in pts.get(c, {})]
        if missing:
            self.sw_status.setText("Incomplete skeleton: " + ", ".join(missing))
            return
        geo = self.ctrl.geometry
        if geo is not None:
            mine = {c: {n: geo.points[c][n] for n in INPUT_POINTS} for c in CORNERS}
            d = X.diff_points(mine, pts, 0.001)
            self.sw_status.setText(f"CAD read back: {len(d)} point(s) differ from the current "
                                   "car by more than 0.001 mm.")
        r = QtWidgets.QMessageBox.question(self, "SolidWorks", "Use the CAD points as the "
                                           "project's hardpoint set (source 'cad')?")
        if r != QtWidgets.QMessageBox.Yes:
            return
        old = self.ctrl.project.hardpoints

        def edit(p: Project) -> None:
            hs = HardpointSet.from_points(pts, source="cad", tol_mm=0.5,
                                          origin="SolidWorks read-back")
            if old is not None:
                hs.front, hs.rear = old.front, old.rear
            p.hardpoints = hs
            p.geometry_mode = "hardpoints"

        self.ctrl.apply(edit)

    # -- Altair -------------------------------------------------------------------------------
    def _altair(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        csv_text = X.hardpoints_csv(geo, "iso")
        # each axle with its own alignment, tyre radius, roll reference and travel -- the
        # same inputs the vdcore column used, so a difference is a kinematics difference
        per_axle = {a: {"static_toe_deg": geo.axles[a].static_toe_deg_per_side,
                        "static_camber_deg": geo.axles[a].static_camber_deg,
                        "loaded_radius_mm": geo.axles[a].loaded_radius_mm,
                        "roll_deg": geo.axle_inputs(a).roll_reference_deg,
                        "travel_mm": geo.axle_inputs(a).travel_bump_mm}
                    for a in ("front", "rear")}

        def work(progress, cancelled):
            from altair_model.kpi_runner import altair_kpis_from_csv
            with tempfile.TemporaryDirectory(prefix="vdstudio_altair_") as tmp:
                path = Path(tmp) / "hardpoints.csv"
                path.write_text(csv_text, encoding="utf-8")
                progress(0.05, "MotionSolve (~150 s)")
                res = altair_kpis_from_csv(path, per_axle=per_axle, workdir=Path(tmp))
                return {k: dict(v.values) for k, v in res.items()}

        self.altair_status.setText("running MotionSolve…")
        self.job.request(work, self._altair_done, immediate=True)

    def _altair_done(self, values: dict[str, dict[str, float]]) -> None:
        ks = self.ctrl.kpis
        rows = []
        for axle, d in values.items():
            for ak, mine in ALTAIR_KEYS.items():
                if ak not in d:
                    continue
                k = ks.get(f"{axle}.{mine}")
                v = k.value if k is not None and k.ok else math.nan
                a = float(d[ak])
                rows.append([axle, k.label if k else ak, f"{v:.6g}", f"{a:.6g}",
                             f"{v - a:+.3g}" if math.isfinite(v) and math.isfinite(a) else "—"])
        self.altair_table.set_rows(rows, right_cols=(2, 3, 4))
        self.altair_status.setText("MotionSolve cross-check complete.")
