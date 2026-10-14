"""Concept & targets: vehicle data, CG estimator, rules, roll budget, camber recovery."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ...engine.concept import apply_cg_estimate, cg_estimate, concept_kpis
from ...engine.project import SOURCES, MassItem, Project
from .. import fields
from ..widgets import KpiTable, ParamForm, banner, button, scroll
from .base import Page


class ConceptPage(Page):
    title = "Concept & targets"
    subtitle = ("What the car must be before any hardpoint exists: mass and CG, rules and tilt "
                "margin, the roll stiffness a roll-gradient target implies, the chassis torsion "
                "that keeps the front/rear split meaningful, and how much roll the FVSA recovers. "
                "Every target here is design_intent from literature; the numbers are its "
                "consequences.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.root.addWidget(split, 1)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.form = ParamForm(ctrl, fields.vehicle_fields())
        ll.addWidget(self.form)

        cg = QtWidgets.QGroupBox("CG estimator (subsystem masses)")
        cl = QtWidgets.QVBoxLayout(cg)
        self.mass = QtWidgets.QTableWidget()
        self.mass.setColumnCount(5)
        self.mass.setHorizontalHeaderLabels(["Subsystem", "Mass [kg]", "CG z [mm]",
                                             "CG x behind front axle [mm]", "Source"])
        self.mass.verticalHeader().setVisible(False)
        mh = self.mass.horizontalHeader()
        mh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        mh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)     # the subsystem name
        self.mass.itemChanged.connect(self._mass_edited)
        cl.addWidget(self.mass)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(button("Add row", self._add))
        row.addWidget(button("Remove selected", self._remove))
        row.addStretch(1)
        row.addWidget(button("Copy estimate into vehicle data", self._apply, primary=True,
                             tip="Your decision: overwrites total mass, CG height (and CG "
                                 "station if every x is given)."))
        cl.addLayout(row)
        self.cg_label = QtWidgets.QLabel("")
        self.cg_label.setWordWrap(True)
        cl.addWidget(self.cg_label)
        ll.addWidget(cg)
        left_scroll = scroll(left)
        # the scroll area, not its content, sets the splitter's minimum: spin boxes keep
        # their value and unit suffix readable instead of being cut by the viewport
        left_scroll.setMinimumWidth(620)
        split.addWidget(left_scroll)

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(banner("Checks use the synthesis axle inputs (track, RC and FVSA targets). "
                            "Solved values for the actual hardpoints are on the Results page."))
        self.table = KpiTable()
        rl.addWidget(self.table, 1)
        split.addWidget(right)
        split.setSizes([660, 840])
        self._filling = False

    def on_project(self) -> None:
        self.form.refresh()
        self._fill_mass()
        self._update()

    def _fill_mass(self) -> None:
        self._filling = True
        items = self.ctrl.project.mass_items
        self.mass.setRowCount(len(items))
        for i, m in enumerate(items):
            for j, v in enumerate((m.name, f"{m.mass_kg:g}", f"{m.z_mm:g}",
                                   "" if m.x_mm is None else f"{m.x_mm:g}", m.source)):
                self.mass.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self._filling = False

    def _read_mass(self) -> list[MassItem]:
        out = []
        for i in range(self.mass.rowCount()):
            def t(j: int, i: int = i) -> str:
                it = self.mass.item(i, j)
                return it.text().strip() if it else ""
            try:
                x = t(3)
                src = t(4) if t(4) in SOURCES else "estimate"
                out.append(MassItem(t(0) or f"item {i + 1}", float(t(1) or 0),
                                    float(t(2) or 0), float(x) if x else None, src))  # type: ignore[arg-type]
            except ValueError:
                continue
        return out

    def _mass_edited(self, _it) -> None:
        if self._filling:
            return
        items = self._read_mass()

        def edit(p: Project) -> None:
            p.mass_items = items

        self.ctrl.apply(edit, geometry=False)

    def _add(self) -> None:
        def edit(p: Project) -> None:
            p.mass_items.append(MassItem("new item", 0.0, 300.0))

        self.ctrl.apply(edit)

    def _remove(self) -> None:
        rows = sorted({i.row() for i in self.mass.selectedIndexes()}, reverse=True)
        if not rows:
            return

        def edit(p: Project) -> None:
            for r in rows:
                if 0 <= r < len(p.mass_items):
                    p.mass_items.pop(r)

        self.ctrl.apply(edit)

    def _apply(self) -> None:
        p = self.ctrl.project
        try:
            est = cg_estimate(p.mass_items, p.vehicle.wheelbase_mm)
        except ValueError as exc:
            self.cg_label.setText(f"Estimate not applied: {exc}")
            return
        if est.total_mass_kg <= 0:
            return
        self.ctrl.apply(lambda q: apply_cg_estimate(q, est))

    def _update(self) -> None:
        p = self.ctrl.project
        v = p.vehicle
        try:
            est = cg_estimate(p.mass_items, v.wheelbase_mm)
        except ValueError as exc:
            txt = f"CG estimate not computed: {exc}."
        else:
            txt = (f"Estimate: <b>{est.total_mass_kg:.1f} kg</b>, CG height "
                   f"<b>{est.cg_height_mm:.1f} mm</b>")
            if est.front_mass_fraction is not None:
                txt += f", {100 * est.front_mass_fraction:.1f} % front"
            else:
                txt += " (weight distribution needs every x)"
            if est.total_mass_kg > 0:
                txt += (f". Δ vs vehicle data: {est.total_mass_kg - v.total_mass_kg:+.1f} kg, "
                        f"CG height {est.cg_height_mm - v.cg_height_mm:+.1f} mm")
                if est.front_mass_fraction is not None:
                    txt += (f", front mass fraction "
                            f"{100 * (est.front_mass_fraction - v.front_mass_fraction):+.1f} "
                            "points")
            txt += f". {100 * est.estimate_share:.0f} % of the mass is tagged estimate."
        txt += (" Validate the CG height (suspension locked, h above the contact plane): raise "
                "one side by θ on level scales, h = ΔW·T / (W·tan θ) with ΔW the drop of the "
                "raised-side reading; on a tilt table with platform scales, "
                "h = (N_low − N_high)·T / (2·W·sin θ).")
        self.cg_label.setText(txt)
        try:
            self.table.set_kpis(concept_kpis(p))
        except (ValueError, ZeroDivisionError) as exc:
            self.cg_label.setText(f"Concept checks failed: {exc}")
