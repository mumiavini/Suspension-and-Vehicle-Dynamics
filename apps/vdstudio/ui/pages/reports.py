"""Results & reports: every KPI, the setup sheet, and the design report (Markdown / PDF)."""

from __future__ import annotations

import json

from PySide6 import QtCore, QtGui, QtWidgets

from ...engine import exports as X
from ...engine.concept import concept_kpis
from ...engine.kpis import KpiSet
from ...engine.project import project_to_dict
from ..widgets import Field, KpiTable, ParamForm, SimpleTable, banner, button, save_text, scroll
from .base import Page


class ReportsPage(Page):
    title = "Results & reports"
    subtitle = ("Everything the app computes in one place, the setup sheet for the team, and a "
                "design report that states provenance and never recommends.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        tabs = QtWidgets.QTabWidget()
        self.root.addWidget(tabs, 1)

        allw = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(allw)
        row = QtWidgets.QHBoxLayout()
        self.status_filter = QtWidgets.QComboBox()
        self.status_filter.addItems(["all", "outside band", "failed", "inside band", "no band"])
        self.status_filter.currentIndexChanged.connect(lambda _i: self._fill_all())
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("search KPI…")
        self.search.textChanged.connect(lambda _t: self._fill_all())
        self.counts = QtWidgets.QLabel("")
        row.addWidget(QtWidgets.QLabel("Show"))
        row.addWidget(self.status_filter)
        row.addWidget(self.search, 1)
        row.addWidget(self.counts)
        row.addWidget(button("Export KPI CSV…", self._export_kpis))
        al.addLayout(row)
        self.all_table = KpiTable()
        al.addWidget(self.all_table, 1)
        tabs.addTab(allw, "All KPIs")

        sheet = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        fields = [Field(f"setup_sheet.{k}", lab, unit, kind="text")
                  for k, lab, unit, _c in X.SETUP_FIELDS]
        self.sheet_form = ParamForm(ctrl, [("Inputs the hardpoints do not carry", fields)],
                                    setter=self._set_sheet)
        sheet.addWidget(scroll(self.sheet_form))
        sw = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(sw)
        sl.setContentsMargins(0, 0, 0, 0)
        self.sheet = SimpleTable(["Category", "Parameter", "Unit", "Front", "Rear", "Source"])
        sl.addWidget(self.sheet, 1)
        sl.addWidget(button("Export setup sheet CSV…", self._export_sheet))
        sheet.addWidget(sw)
        sheet.setSizes([420, 900])
        tabs.addTab(sheet, "Setup sheet")

        rep = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(rep)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(button("Refresh report", self._refresh_report, primary=True))
        row.addWidget(button("Save Markdown…", self._save_md))
        row.addWidget(button("Export PDF…", self._save_pdf))
        row.addWidget(button("Save project JSON…", self._save_json))
        row.addWidget(button("Hardpoints CSV (ISO)…", self._save_hp))
        row.addStretch(1)
        rl.addLayout(row)
        rl.addWidget(banner("The report is built from the KPIs currently on screen. If the swept "
                            "KPIs are still computing, wait for the status bar to say they are "
                            "up to date before exporting."))
        self.report = QtWidgets.QTextBrowser()
        self.report.setOpenExternalLinks(True)
        rl.addWidget(self.report, 1)
        tabs.addTab(rep, "Design report")
        self._md = ""

    def _set_sheet(self, path: str, value) -> None:
        key = path.split(".", 1)[1]

        def edit(p) -> None:
            p.setup_sheet[key] = value

        self.ctrl.apply(edit, geometry=False)

    def on_project(self) -> None:
        self.sheet_form.refresh()
        self._fill_sheet()

    def on_kpis(self, ks: KpiSet) -> None:
        self._fill_all()
        self._fill_sheet()

    def _fill_all(self) -> None:
        ks = self.ctrl.kpis
        want = {"all": None, "outside band": "out", "failed": "fail", "inside band": "in",
                "no band": "info"}[self.status_filter.currentText()]
        q = self.search.text().lower()
        sub = KpiSet({k: v for k, v in ks.items.items()
                      if (want is None or v.status == want)
                      and (not q or q in v.label.lower() or q in v.group.lower())})
        self.all_table.set_kpis(sub)
        c = ks.counts()
        self.counts.setText(f"inside {c['in']} · outside {c['out']} · no band {c['info']} · "
                            f"failed {c['fail']}")

    def _fill_sheet(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        self.sheet.set_rows(X.setup_sheet_rows(self.ctrl.project, geo, self.ctrl.kpis))

    def _export_kpis(self) -> None:
        save_text(self, X.kpi_csv(self.ctrl.kpis), "kpis.csv", "CSV (*.csv)")

    def _export_sheet(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            return
        rows = X.setup_sheet_rows(self.ctrl.project, geo, self.ctrl.kpis)
        save_text(self, X.rows_csv(["category", "parameter", "unit", "front", "rear", "source"],
                                   rows), "setup_sheet.csv", "CSV (*.csv)")

    def _refresh_report(self) -> None:
        geo = self.ctrl.geometry
        if geo is None:
            self.report.setPlainText("Geometry failed: " + self.ctrl.geometry_error)
            return
        p = self.ctrl.project
        try:
            concept = concept_kpis(p)
            concept_err = ""
        except (ValueError, ZeroDivisionError) as exc:
            concept, concept_err = None, str(exc)
        self._md = X.markdown_report(p, geo, self.ctrl.kpis, concept)
        if concept_err:
            self._md = (f"> **Note:** the concept-stage numbers could not be computed "
                        f"({concept_err}); that section is left out.\n\n" + self._md)
        if self.ctrl.dynamic_stale:
            self._md = ("> **Note:** swept KPIs were still being recomputed when this report "
                        "was built; some values may be from the previous design state.\n\n"
                        + self._md)
        self.report.setMarkdown(self._md)

    def _ensure(self) -> bool:
        if not self._md:
            self._refresh_report()
        return bool(self._md)

    def _save_md(self) -> None:
        if self._ensure():
            save_text(self, self._md, "design_report.md", "Markdown (*.md)")

    def _save_pdf(self) -> None:
        if not self._ensure():
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export PDF", "design_report.pdf",
                                                        "PDF (*.pdf)")
        if not path:
            return
        doc = QtGui.QTextDocument()
        doc.setMarkdown(self._md)
        writer = QtGui.QPdfWriter(path)
        writer.setPageSize(QtGui.QPageSize(QtGui.QPageSize.A4))
        writer.setResolution(110)
        writer.setPageMargins(QtCore.QMarginsF(12, 12, 12, 12), QtGui.QPageLayout.Millimeter)
        doc.setPageSize(QtCore.QSizeF(writer.width(), writer.height()))
        doc.print_(writer)

    def _save_json(self) -> None:
        save_text(self, json.dumps(project_to_dict(self.ctrl.project), indent=2,
                                   ensure_ascii=False), "project.vdproj.json", "JSON (*.json)")

    def _save_hp(self) -> None:
        geo = self.ctrl.geometry
        if geo is not None:
            save_text(self, X.hardpoints_csv(geo, "iso"), "hardpoints_iso.csv", "CSV (*.csv)")
