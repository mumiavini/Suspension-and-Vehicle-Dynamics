"""Hardpoints: the point table, geometry source, CSV/JSON import-export and CAD diff."""

from __future__ import annotations

import math
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ...engine import exports as X
from ...engine.geometry import POINT_ORDER, CarGeometry, freeze_to_hardpoints
from ...engine.project import CORNERS, INPUT_POINTS, SOURCES, AxleAlignment, HardpointSet, Project
from .. import theme
from ..widgets import Field, ParamForm, SimpleTable, banner, button, save_text, scroll
from .base import Page

MIRROR = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}


class HardpointsPage(Page):
    title = "Hardpoints"
    subtitle = ("The complete point set every KPI runs on (ISO 8855: X forward, Y LEFT, Z up; "
                "mm). In synthesis mode it is read-only and regenerates on every design edit; "
                "freeze it, import a CAD/measured set, or edit points directly in hardpoint "
                "mode.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        self.info = banner("")
        self.root.addWidget(self.info)

        bar = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["synthesis", "hardpoints"])
        self.mode.currentTextChanged.connect(self._set_mode)
        bar.addWidget(QtWidgets.QLabel("Geometry source:"))
        bar.addWidget(self.mode)
        bar.addWidget(button("Freeze synthesis → hardpoints", self._freeze,
                             tip="Copy the current synthesised points into an editable set "
                                 "(provenance stays design_intent)."))
        bar.addWidget(button("Import CSV / JSON…", self._import))
        bar.addWidget(button("Export CSV…", self._export))
        bar.addWidget(button("Compare with file…", self._compare,
                             tip="Diff against a CAD read-back or measured set."))
        self.sym = QtWidgets.QCheckBox("Symmetric edit (mirror L↔R)")
        self.sym.setChecked(True)
        bar.addWidget(self.sym)
        self.frame = QtWidgets.QComboBox()
        for k, v in X.FRAME_LABELS.items():
            self.frame.addItem(v, k)
        bar.addWidget(QtWidgets.QLabel("File frame:"))
        bar.addWidget(self.frame)
        bar.addStretch(1)
        self.root.addLayout(bar)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["Corner", "Point", "X [mm]", "Y [mm]", "Z [mm]",
                                              "Source", "Tol [mm]"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._cell_edited)
        split.addWidget(self.table)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        groups = []
        for axle in ("front", "rear"):
            a = f"hardpoints.{axle}."
            groups.append((f"{axle.title()} alignment & tyre (hardpoint mode)", [
                Field(a + "static_camber_deg", "Static camber (built into upright)", "deg", -6, 3,
                      0.1, 2),
                Field(a + "static_toe_deg_per_side", "Static toe per side (+ in)", "deg", -3, 3,
                      0.05, 3),
                Field(a + "loaded_radius_mm", "Tyre loaded radius", "mm", 150, 400, 1, 1),
                Field(a + "tire_source", "Tyre radius provenance", kind="choice",
                      choices=SOURCES),
                Field(a + "tire_tol_mm", "Tyre radius tolerance", "mm", 0, 50, 0.5, 1),
            ]))
        self.align = ParamForm(ctrl, groups)
        rl.addWidget(self.align)
        self.align_note = QtWidgets.QLabel(
            "Static camber, static toe and the tyre radius of a hardpoint set are edited here "
            "once a set exists (freeze the synthesis or import a file). In synthesis mode they "
            "come from the axle and steering inputs.")
        self.align_note.setWordWrap(True)
        self.align_note.setObjectName("muted")
        rl.addWidget(self.align_note)
        self.diff_label = QtWidgets.QLabel("")
        self.diff_label.setObjectName("muted")
        rl.addWidget(self.diff_label)
        self.diff = SimpleTable(["Corner", "Point", "|Δ| [mm]", "this project", "file"])
        rl.addWidget(self.diff, 1)
        split.addWidget(scroll(right))
        split.setSizes([900, 520])
        self.root.addWidget(split, 1)
        self._filling = False

    # -- state ---------------------------------------------------------------------------
    def on_project(self) -> None:
        p = self.ctrl.project
        self.mode.blockSignals(True)
        self.mode.setCurrentText(p.geometry_mode)
        self.mode.blockSignals(False)
        hp_mode = p.geometry_mode == "hardpoints"
        self.align.setVisible(p.hardpoints is not None)
        self.align_note.setVisible(p.hardpoints is None)
        self.align.setEnabled(hp_mode and p.hardpoints is not None)
        if p.hardpoints is not None:
            self.align.refresh()
        if hp_mode:
            origin = p.hardpoints.origin if p.hardpoints else ""
            self.info.label.setText(
                f"<b>Hardpoint mode</b> — edits are direct. Set: <i>{origin or 'unnamed'}</i>. "
                "CONTACT_PATCH is derived from the wheel centre, loaded radius and camber.")
        else:
            self.info.label.setText(
                "<b>Synthesis mode</b> — the table shows the points generated from the design "
                "inputs (read-only). Freeze them or import a set to edit points directly.")

    def on_geometry(self, geo: CarGeometry) -> None:
        self._fill(geo)

    def _fill(self, geo: CarGeometry) -> None:
        editable = self.ctrl.project.geometry_mode == "hardpoints"
        self._filling = True
        rows = geo.rows()
        self.table.setRowCount(len(rows))
        muted = QtGui.QColor(theme.current().text_muted)
        for i, (c, n, x, y, z, src, tol) in enumerate(rows):
            derived = n == "CONTACT_PATCH"
            vals = [c, n, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", src,
                    "" if (isinstance(tol, float) and math.isnan(tol)) else f"{tol:g}"]
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                can = editable and not derived and j in (2, 3, 4, 5, 6)
                if not can:
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                if derived:
                    it.setForeground(muted)
                if j in (2, 3, 4, 6):
                    it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                if j == 5 and src == "estimate":
                    col = QtGui.QColor(theme.current().warn)
                    col.setAlpha(70)
                    it.setBackground(col)
                self.table.setItem(i, j, it)
        self._filling = False

    # -- edits --------------------------------------------------------------------------
    def _cell_edited(self, it: QtWidgets.QTableWidgetItem) -> None:
        if self._filling or self.ctrl.project.geometry_mode != "hardpoints":
            return
        r, col = it.row(), it.column()
        c, n = self.table.item(r, 0).text(), self.table.item(r, 1).text()
        txt = it.text().strip()
        sym = self.sym.isChecked()

        def edit(p: Project) -> None:
            hs = p.hardpoints
            assert hs is not None
            if col in (2, 3, 4):
                v = float(txt.replace(",", "."))
                pts = [c, MIRROR[c]] if sym else [c]
                for cc in pts:
                    xyz = list(hs.points[cc][n])
                    val = v if (cc == c or col != 3) else -v
                    xyz[col - 2] = val + 0.0
                    hs.points[cc][n] = tuple(xyz)  # type: ignore[assignment]
            elif col == 5:
                if txt not in SOURCES:
                    raise ValueError(f"source must be one of {SOURCES}")
                for cc in ([c, MIRROR[c]] if sym else [c]):
                    hs.sources.setdefault(cc, {})[n] = txt
            elif col == 6:
                for cc in ([c, MIRROR[c]] if sym else [c]):
                    hs.tols.setdefault(cc, {})[n] = float(txt.replace(",", "."))

        try:
            self.ctrl.apply(edit)
        except (ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(self, "Hardpoints", f"Not applied: {exc}")
            if self.ctrl.geometry is not None:
                self._fill(self.ctrl.geometry)

    def _set_mode(self, mode: str) -> None:
        p = self.ctrl.project
        if mode == p.geometry_mode:
            return
        if mode == "hardpoints" and p.hardpoints is None:
            r = QtWidgets.QMessageBox.question(
                self, "Hardpoints", "There is no hardpoint set yet. Freeze the current "
                "synthesis into one?")
            if r != QtWidgets.QMessageBox.Yes:
                self.mode.setCurrentText(p.geometry_mode)
                return
            self._freeze()
            return
        self.ctrl.set_value("geometry_mode", mode)

    def _freeze(self) -> None:
        if self.ctrl.project.hardpoints is not None and self.ctrl.project.geometry_mode == \
                "hardpoints":
            r = QtWidgets.QMessageBox.question(
                self, "Hardpoints", "Replace the current hardpoint set with the synthesis?")
            if r != QtWidgets.QMessageBox.Yes:
                return
        try:
            hs = freeze_to_hardpoints(self.ctrl.project)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Hardpoints", f"Synthesis failed: {exc}")
            return

        def edit(p: Project) -> None:
            p.hardpoints = hs
            p.geometry_mode = "hardpoints"

        self.ctrl.apply(edit)

    def _import(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import hardpoints", str(Path.cwd()), "Hardpoints (*.csv *.json)")
        if not path:
            return
        frame = self.frame.currentData()
        try:
            pts, src, tol = X.read_hardpoints(path, frame)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Import", str(exc))
            return
        missing = [f"{c}/{n}" for c in CORNERS for n in INPUT_POINTS if n not in pts.get(c, {})]
        if missing:
            QtWidgets.QMessageBox.warning(self, "Import", "Missing points: " + ", ".join(missing))
            return
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Provenance", "Where do these points come from?", list(SOURCES), 0, False)
        if not ok:
            return
        old = self.ctrl.project.hardpoints
        hs = HardpointSet.from_points(pts, source=choice, tol_mm=1.0,  # type: ignore[arg-type]
                                      origin=f"{Path(path).name} ({X.FRAME_LABELS[frame]})",
                                      front=old.front if old else _align(self.ctrl, "front"),
                                      rear=old.rear if old else _align(self.ctrl, "rear"))
        for c, d in src.items():
            hs.sources[c].update(d)
        for c, d in tol.items():
            hs.tols[c].update(d)

        def edit(p: Project) -> None:
            p.hardpoints = hs
            p.geometry_mode = "hardpoints"

        self.ctrl.apply(edit)

    def _export(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        frame = self.frame.currentData()
        save_text(self, X.hardpoints_csv(geo, frame, extended=True),
                  f"hardpoints_{frame}.csv", "CSV (*.csv)")

    def _compare(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Compare with", str(Path.cwd()), "Hardpoints (*.csv *.json)")
        if not path:
            return
        try:
            pts, _, _ = X.read_hardpoints(path, self.frame.currentData())
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Compare", str(exc))
            return
        mine = {c: {n: geo.points[c][n] for n in INPUT_POINTS} for c in CORNERS}
        d = X.diff_points(mine, pts, 0.001)
        self.diff_label.setText(f"{len(d)} point(s) differ by more than 0.001 mm from "
                                f"{Path(path).name}." if d else
                                f"Identical to {Path(path).name} within 0.001 mm.")
        def fmt(p) -> str:
            return "—" if p is None else f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"

        self.diff.set_rows([[c, n, f"{dd:.3f}", fmt(a), fmt(b)] for c, n, dd, a, b in d],
                           right_cols=(2,))


def _align(ctrl, axle: str) -> AxleAlignment:
    geo = ctrl.geometry
    if geo is None:
        return AxleAlignment()
    a = geo.axles[axle]
    return AxleAlignment(a.static_camber_deg, a.static_toe_deg_per_side, a.loaded_radius_mm,
                         "estimate", 1.0)


_ = POINT_ORDER
