"""Dashboard: the car at a glance, what needs attention, and where the project stands."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ...engine.geometry import CarGeometry
from ...engine.kpis import KpiSet
from ...engine.pose import PoseSolver
from ...engine.project import STAGE_STATUSES, STAGES, Project, StageState
from ..views import FrontView, TopView
from ..widgets import Card, SimpleTable, banner, button, scroll
from .base import Page

CARDS: tuple[tuple[str, str, str], ...] = (
    ("front.rc_static", "Front RC height", "mm"),
    ("rear.rc_static", "Rear RC height", "mm"),
    ("front.camber_gain", "Front camber gain", "deg/mm"),
    ("rear.camber_gain", "Rear camber gain", "deg/mm"),
    ("front.roll_outer_camber", "Front outer camber in roll", "deg"),
    ("rear.roll_outer_camber", "Rear outer camber in roll", "deg"),
    ("front.anti", "Anti-dive", "%"),
    ("rear.anti", "Anti-squat", "%"),
    ("front.bump_steer_peak", "Front bump steer (peak)", "deg"),
    ("rear.bump_steer_peak", "Rear bump steer (peak)", "deg"),
    ("steering.ackermann", "Ackermann (solved)", "%"),
    ("steering.ratio", "Steering ratio", ":1"),
    ("steering.max_steer_inner", "Max steer, inner", "deg"),
    ("steering.effort_sw_torque", "Parking SW torque", "N·m"),
    ("vehicle.wheelbase", "Wheelbase", "mm"),
    ("vehicle.tilt_margin", "Tilt-test margin", "mm"),
)
STAGE_PAGE = {"concept": "concept", "geometry": "geometry", "steering": "steering",
              "kinematics": "kinematics", "crosscheck": "integrations", "cad": "integrations",
              "manufacturing": "hardpoints", "testing": "telemetry", "design_event": "reports"}


class DashboardPage(Page):
    title = ""

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        content = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        self.root.addWidget(scroll(content))

        head = QtWidgets.QHBoxLayout()
        idbox = QtWidgets.QGroupBox("Project")
        f = QtWidgets.QFormLayout(idbox)
        self.ed: dict[str, QtWidgets.QLineEdit] = {}
        for key, lab in (("name", "Car / project"), ("team", "Team"), ("author", "Author")):
            le = QtWidgets.QLineEdit()
            le.editingFinished.connect(lambda k=key, e=le: self._set(k, e.text()))
            f.addRow(lab, le)
            self.ed[key] = le
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setMaximumHeight(70)
        self.notes.setPlaceholderText("Project notes…")
        f.addRow("Notes", self.notes)
        f.addRow(button("Save notes", lambda: self._set("notes", self.notes.toPlainText())))
        head.addWidget(idbox, 2)
        self.top = TopView()
        self.top.setMinimumHeight(260)
        self.front = FrontView()
        self.front.setMinimumHeight(260)
        head.addWidget(self.top, 2)
        head.addWidget(self.front, 3)
        lay.addLayout(head)

        self.summary = banner("")
        lay.addWidget(self.summary)
        grid = QtWidgets.QGridLayout()
        self.cards: dict[str, Card] = {}
        for i, (key, lab, unit) in enumerate(CARDS):
            c = Card(lab, unit)
            self.cards[key] = c
            grid.addWidget(c, i // 8, i % 8)
        lay.addLayout(grid)

        low = QtWidgets.QHBoxLayout()
        sbox = QtWidgets.QGroupBox("FSAE workflow")
        sl = QtWidgets.QGridLayout(sbox)
        self.stage_status: dict[str, QtWidgets.QComboBox] = {}
        self.stage_notes: dict[str, QtWidgets.QLineEdit] = {}
        for i, (key, label, desc) in enumerate(STAGES):
            lab = QtWidgets.QLabel(f"<b>{label}</b>")
            lab.setToolTip(desc)
            co = QtWidgets.QComboBox()
            co.addItems(STAGE_STATUSES)
            co.currentTextChanged.connect(lambda t, k=key: self._stage(k, status=t))
            le = QtWidgets.QLineEdit()
            le.setPlaceholderText(desc)
            le.editingFinished.connect(lambda k=key, e=le: self._stage(k, notes=e.text()))
            go = button("Open", lambda k=key: self._goto(STAGE_PAGE.get(k, "dashboard")))
            sl.addWidget(lab, i, 0)
            sl.addWidget(co, i, 1)
            sl.addWidget(le, i, 2)
            sl.addWidget(go, i, 3)
            self.stage_status[key] = co
            self.stage_notes[key] = le
        sl.setColumnStretch(2, 1)
        low.addWidget(sbox, 3)
        abox = QtWidgets.QGroupBox("Needs attention (information, not verdicts)")
        al = QtWidgets.QVBoxLayout(abox)
        self.attention = SimpleTable(["scope", "item", "value", "band / reason"])
        al.addWidget(self.attention)
        low.addWidget(abox, 2)
        lay.addLayout(low)
        self._filling = False

    def _goto(self, key: str) -> None:
        mw = getattr(self, "main_window", None)
        if mw is not None:
            mw.goto(key)

    def _set(self, key: str, value: str) -> None:
        if getattr(self.ctrl.project, key) != value:
            def edit(p: Project) -> None:
                setattr(p, key, value)

            self.ctrl.apply(edit, geometry=False)

    def _stage(self, key: str, status: str | None = None, notes: str | None = None) -> None:
        if self._filling:
            return
        cur = self.ctrl.project.stages.get(key, StageState())
        new = StageState(status if status is not None else cur.status,
                         notes if notes is not None else cur.notes)
        if new == cur:
            return

        def edit(p: Project) -> None:
            p.stages[key] = new

        self.ctrl.apply(edit, geometry=False)

    def on_project(self) -> None:
        self._filling = True
        p = self.ctrl.project
        for key, le in self.ed.items():
            le.setText(getattr(p, key))
        if self.notes.toPlainText() != p.notes:
            self.notes.setPlainText(p.notes)
        for key, _, _ in STAGES:
            st = p.stages.get(key, StageState())
            self.stage_status[key].setCurrentText(st.status)
            self.stage_notes[key].setText(st.notes)
        self._filling = False

    def on_geometry(self, geo: CarGeometry) -> None:
        pose = PoseSolver(geo).solve()
        self.top.draw(geo, pose, show_construction=False)
        self.front.draw(geo, pose, "front")

    def on_geometry_failed(self, msg: str) -> None:
        self.summary.label.setText(f"<b>Geometry failed:</b> {msg}")

    def on_kpis(self, ks: KpiSet) -> None:
        for key, card in self.cards.items():
            card.set_kpi(ks.get(key))
        c = ks.counts()
        geo = self.ctrl.geometry
        est = geo.estimate_count() if geo else 0
        stale = " Swept KPIs are being recomputed…" if self.ctrl.dynamic_stale else ""
        mode = self.ctrl.project.geometry_mode
        self.summary.label.setText(
            f"<b>{self.ctrl.project.name}</b> — geometry source <b>{mode}</b>. "
            f"{c['in']} KPIs inside their design-intent band, <b>{c['out']}</b> outside, "
            f"{c['fail']} failed, {c['info']} without a band. {est} estimate-tagged input(s)."
            + stale)
        rows = []
        if geo is not None:
            for w in geo.warnings:
                rows.append(["geometry", "warning", "", w])
        for k in ks.items.values():
            if k.status == "fail":
                rows.append([k.scope, k.label, "—", k.error])
        for k in ks.items.values():
            if k.status == "out":
                rows.append([k.scope, k.label, f"{k.text()} {k.unit}",
                             f"{k.band_text()} ({k.band_source})"])
        if not rows:
            rows = [["", "Nothing outside its band and nothing failed.", "", ""]]
        self.attention.set_rows(rows)
        _ = QtCore
