"""Design log and design targets: the decision trail the Design Event judges ask for."""

from __future__ import annotations

import datetime as _dt

from PySide6 import QtCore, QtWidgets

from ...engine.project import STAGES, LogEntry, Project, TargetEntry
from ..widgets import banner, button
from .base import Page

LOG_COLS = ("date", "stage", "title", "decision", "rationale", "alternatives", "author")
TARGET_COLS = ("name", "value", "unit", "tolerance", "event", "rationale", "reference", "source")


class LogPage(Page):
    title = "Design log & targets"
    subtitle = ("Write down what was decided, why, and what else was considered — while it is "
                "fresh. Targets carry their rationale and reference: a target you cannot "
                "justify should not exist. Both feed the design report.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        tabs = QtWidgets.QTabWidget()
        self.root.addWidget(tabs, 1)

        lw = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(lw)
        self.log = QtWidgets.QTableWidget()
        self.log.setColumnCount(len(LOG_COLS))
        self.log.setHorizontalHeaderLabels([c.title() for c in LOG_COLS])
        self.log.verticalHeader().setVisible(False)
        self.log.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.log.horizontalHeader().setStretchLastSection(True)
        self.log.setWordWrap(True)
        for i, w in enumerate((95, 150, 200, 320, 320, 220)):
            self.log.setColumnWidth(i, w)
        self.log.itemChanged.connect(lambda _it: self._save_log())
        ll.addWidget(self.log, 1)
        row = QtWidgets.QHBoxLayout()
        self.stage = QtWidgets.QComboBox()
        for k, label, _ in STAGES:
            self.stage.addItem(label, k)
        row.addWidget(QtWidgets.QLabel("New entry at stage"))
        row.addWidget(self.stage)
        row.addWidget(button("Add entry", self._add_log, primary=True))
        row.addWidget(button("Add entry with current KPI snapshot", self._add_snapshot,
                             tip="Records the key numbers at the moment of the decision."))
        row.addWidget(button("Delete selected", self._del_log))
        row.addStretch(1)
        ll.addLayout(row)
        tabs.addTab(lw, "Decision log")

        tw = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tw)
        tl.addWidget(banner("Targets are literature / design-intent values (tag "
                            "<code>design_intent</code>). The tool never chases them: use them "
                            "as arguments to a Trade-off target query, and quote the rationale "
                            "when you defend the pick."))
        self.targets = QtWidgets.QTableWidget()
        self.targets.setColumnCount(len(TARGET_COLS))
        self.targets.setHorizontalHeaderLabels([c.title() for c in TARGET_COLS])
        self.targets.verticalHeader().setVisible(False)
        self.targets.horizontalHeader().setStretchLastSection(True)
        self.targets.itemChanged.connect(lambda _it: self._save_targets())
        tl.addWidget(self.targets, 1)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(button("Add target", self._add_target))
        r2.addWidget(button("Delete selected", self._del_target))
        r2.addStretch(1)
        tl.addLayout(r2)
        tabs.addTab(tw, "Design targets")
        self._filling = False

    def on_project(self) -> None:
        self._filling = True
        p = self.ctrl.project
        self.log.setRowCount(len(p.design_log))
        for i, e in enumerate(p.design_log):
            for j, c in enumerate(LOG_COLS):
                self.log.setItem(i, j, QtWidgets.QTableWidgetItem(str(getattr(e, c))))
        self.log.resizeRowsToContents()
        self.targets.setRowCount(len(p.targets))
        for i, t in enumerate(p.targets):
            for j, c in enumerate(TARGET_COLS):
                v = getattr(t, c)
                self.targets.setItem(i, j, QtWidgets.QTableWidgetItem(
                    f"{v:g}" if isinstance(v, float) else str(v)))
        self.targets.resizeColumnsToContents()
        self._filling = False

    def _cell(self, table: QtWidgets.QTableWidget, i: int, j: int) -> str:
        it = table.item(i, j)
        return it.text() if it is not None else ""

    def _save_log(self) -> None:
        if self._filling:
            return
        entries = [LogEntry(**{c: self._cell(self.log, i, j) for j, c in enumerate(LOG_COLS)})
                   for i in range(self.log.rowCount())]

        def edit(p: Project) -> None:
            p.design_log = entries

        self.ctrl.apply(edit, geometry=False)

    def _add_log(self, decision: str = "") -> None:
        stage = self.stage.currentData()

        def edit(p: Project) -> None:
            p.design_log.append(LogEntry(_dt.date.today().isoformat(), stage, "New decision",
                                         decision=decision, author=p.author))

        self.ctrl.apply(edit, geometry=False)

    def _add_snapshot(self) -> None:
        ks = self.ctrl.kpis
        keys = ("front.rc_static", "rear.rc_static", "front.camber_gain", "rear.camber_gain",
                "front.anti", "front.bump_steer_peak", "steering.ackermann", "steering.ratio")
        parts = []
        for k in keys:
            v = ks.get(k)
            if v is not None:
                parts.append(f"{v.scope} {v.label} = {v.text()} {v.unit}")
        self._add_log("Snapshot: " + "; ".join(parts))

    def _del_log(self) -> None:
        rows = sorted({i.row() for i in self.log.selectedIndexes()}, reverse=True)

        def edit(p: Project) -> None:
            for r in rows:
                p.design_log.pop(r)

        if rows:
            self.ctrl.apply(edit, geometry=False)

    def _save_targets(self) -> None:
        if self._filling:
            return
        out = []
        for i in range(self.targets.rowCount()):
            d = {c: self._cell(self.targets, i, j) for j, c in enumerate(TARGET_COLS)}
            try:
                d["value"] = float(d["value"] or 0)
                d["tolerance"] = float(d["tolerance"] or 0)
            except ValueError:
                continue
            out.append(TargetEntry(**d))  # type: ignore[arg-type]

        def edit(p: Project) -> None:
            p.targets = out

        self.ctrl.apply(edit, geometry=False)

    def _add_target(self) -> None:
        def edit(p: Project) -> None:
            p.targets.append(TargetEntry("New target", 0.0, "", 0.0, "all", "", "",
                                         "design_intent"))

        self.ctrl.apply(edit, geometry=False)

    def _del_target(self) -> None:
        rows = sorted({i.row() for i in self.targets.selectedIndexes()}, reverse=True)

        def edit(p: Project) -> None:
            for r in rows:
                p.targets.pop(r)

        if rows:
            self.ctrl.apply(edit, geometry=False)
        _ = QtCore
