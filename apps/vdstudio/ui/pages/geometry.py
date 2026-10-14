"""Suspension geometry: SLA synthesis inputs on the left, live construction views
and checks on the right. Every edit re-synthesises the car in milliseconds."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ...engine.geometry import CarGeometry, synthesis_rear_axle_x_mm
from ...engine.kpis import KpiSet
from ...engine.pose import PoseSolver
from ...engine.project import Project, with_path
from .. import fields
from ..views import FrontView, SideView, TopView
from ..widgets import KpiTable, ParamForm, banner, button, scroll
from .base import Page

STATIC_GROUPS = ["Front view", "Anti-geometry", "Members (true 3D)", "Packaging",
                 "Static alignment"]


class GeometryPage(Page):
    title = "Suspension geometry — SLA synthesis"
    subtitle = ("Design intent in, hardpoints out: the FVIC is placed by the roll-centre and FVSA "
                "targets, the ball joints by the kingpin, and the chassis pickups fall where the "
                "wishbone lines cross the pickup planes (sla_geometry.py). Caster and the front "
                "tie rod come from the Steering page (steering_geometry.py).")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        self.mode_banner = banner("")
        self.root.addWidget(self.mode_banner)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.root.addWidget(split, 1)

        # left: editors
        tabs = QtWidgets.QTabWidget()
        self.forms: list[ParamForm] = []
        # the brake / drive layout sets the anti reference in BOTH geometry modes, so its
        # form is never disabled with the synthesis inputs
        self.layout_forms: list[ParamForm] = []
        for axle in ("front", "rear"):
            groups = fields.axle_fields(axle)
            cut = next(i for i, (t, _) in enumerate(groups) if t.startswith("Side view")) + 1
            # in synthesis the rear axle station is derived from the vehicle wheelbase:
            # show that value, read-only
            base = (lambda: _with_derived_rear(ctrl.project)) if axle == "rear" else None
            above, below = (ParamForm(ctrl, groups[:cut], base=base),
                            ParamForm(ctrl, groups[cut:], base=base))
            lay = ParamForm(ctrl, fields.layout_fields(axle))
            if axle == "rear":
                _, ed = above._editors["rear.axle_x_mm"]
                ed.setReadOnly(True)
                ed.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            self.forms += [above, below]
            self.layout_forms.append(lay)
            box = QtWidgets.QWidget()
            bl = QtWidgets.QVBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            for w in (above, lay, below):
                bl.addWidget(w)
            tabs.addTab(scroll(box), f"{axle.title()} axle")
        for title, groups in (("Rear toe link", fields.toe_link_fields()),
                              ("Bands", fields.axle_band_fields("front")
                               + fields.axle_band_fields("rear"))):
            f = ParamForm(ctrl, groups)
            self.forms.append(f)
            tabs.addTab(scroll(f), title)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(tabs, 1)
        ll.addWidget(button("Show the sla_geometry text report…", self._report))
        left.setMinimumWidth(420)
        split.addWidget(left)

        # right: views + checks
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vt = QtWidgets.QTabWidget()
        fv_box = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(fv_box)
        fl.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        self.axle_combo = QtWidgets.QComboBox()
        self.axle_combo.addItems(["front", "rear"])
        self.axle_combo.currentTextChanged.connect(lambda _t: self._draw())
        self.constr = QtWidgets.QCheckBox("construction lines (IC, RC)")
        self.constr.setChecked(True)
        self.constr.toggled.connect(lambda _b: self._draw())
        row.addWidget(QtWidgets.QLabel("Axle:"))
        row.addWidget(self.axle_combo)
        row.addWidget(self.constr)
        row.addStretch(1)
        fl.addLayout(row)
        self.front = FrontView()
        fl.addWidget(self.front, 1)
        vt.addTab(fv_box, "Front view")
        self.side = SideView()
        vt.addTab(self.side, "Side view (anti-geometry)")
        self.top = TopView()
        vt.addTab(self.top, "Top view")
        right.addWidget(vt)
        self.table = KpiTable(STATIC_GROUPS)
        right.addWidget(self.table)
        right.setSizes([520, 380])
        split.addWidget(right)
        split.setSizes([460, 1000])
        self._pose = None
        self._car: CarGeometry | None = None

    def on_project(self) -> None:
        for f in self.forms + self.layout_forms:
            f.refresh()
        synth = self.ctrl.project.geometry_mode == "synthesis"
        for f in self.forms:
            f.setEnabled(synth)
        if synth:
            self.mode_banner.label.setText(
                "Geometry source: <b>synthesis</b>. Edits apply immediately; swept KPIs follow "
                "in the background. Bands are design-intent information, not targets.")
        else:
            self.mode_banner.label.setText(
                "Geometry source: <b>hardpoints</b> — these synthesis inputs are inactive "
                "(the brake / drive layout still applies). Switch back on the Hardpoints page. "
                "The views below show the hardpoint car.")

    def on_geometry(self, geo: CarGeometry) -> None:
        self._car = geo
        self._pose = PoseSolver(geo).solve()
        self._draw()

    def on_geometry_failed(self, msg: str) -> None:
        self.mode_banner.label.setText(f"<b>Geometry failed:</b> {msg}")

    def _draw(self) -> None:
        if self._car is None or self._pose is None:
            return
        c = self.constr.isChecked()
        self.front.draw(self._car, self._pose, self.axle_combo.currentText(),
                        show_construction=c)
        self.side.draw(self._car, self._pose, show_construction=c)
        self.top.draw(self._car, self._pose, show_construction=c)

    def on_kpis(self, ks: KpiSet) -> None:
        self.table.set_kpis(ks)

    def _report(self) -> None:
        geo = self.ctrl.geometry
        text = geo.design.text if geo is not None and geo.design is not None else \
            "No synthesis report: the geometry source is hardpoints (or synthesis failed)."
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("sla_geometry report")
        dlg.resize(820, 760)
        lay = QtWidgets.QVBoxLayout(dlg)
        te = QtWidgets.QPlainTextEdit(text)
        te.setReadOnly(True)
        f = te.font()
        f.setFamily("Consolas")
        f.setStyleHint(f.StyleHint.Monospace)
        te.setFont(f)
        lay.addWidget(te)
        dlg.exec()


def _with_derived_rear(project: Project) -> Project:
    """The project as synthesis reads it: rear axle station = front station + wheelbase."""
    return with_path(project, "rear.axle_x_mm", synthesis_rear_axle_x_mm(project))
