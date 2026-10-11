"""Trade-offs: the sweep / what-if engine. Fills tables of consequences; never picks a row."""

from __future__ import annotations

import csv
import io
import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ...engine import whatif as W
from .. import theme
from ..asyncjob import Debounced
from ..widgets import banner, button, pen, plot, save_text
from .base import Page

DEFAULT_OUTPUTS = ("front.camber_gain", "front.rc_static", "front.rc_span",
                   "front.roll_outer_camber", "front.bump_steer_peak", "front.anti",
                   "steering.ackermann", "steering.ratio")


class _ParamRow(QtWidgets.QWidget):
    def __init__(self, label: str, optional: bool = False) -> None:
        super().__init__()
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 2, 0, 6)
        self.enable = QtWidgets.QCheckBox(label)
        self.enable.setChecked(not optional)
        self.enable.setEnabled(optional)
        self.combo = QtWidgets.QComboBox()
        self.combo.setMinimumWidth(280)
        self.lo = QtWidgets.QDoubleSpinBox()
        self.hi = QtWidgets.QDoubleSpinBox()
        for sb in (self.lo, self.hi):
            sb.setRange(-1e6, 1e6)
            sb.setDecimals(3)
        self.n = QtWidgets.QSpinBox()
        self.n.setRange(2, 41)
        self.n.setValue(5)
        lay.addWidget(self.enable, 0, 0)
        lay.addWidget(self.combo, 0, 1, 1, 6)
        lay.addWidget(QtWidgets.QLabel("from"), 1, 1)
        lay.addWidget(self.lo, 1, 2)
        lay.addWidget(QtWidgets.QLabel("to"), 1, 3)
        lay.addWidget(self.hi, 1, 4)
        lay.addWidget(QtWidgets.QLabel("steps"), 1, 5)
        lay.addWidget(self.n, 1, 6)
        lay.setColumnStretch(2, 1)
        lay.setColumnStretch(4, 1)

    def path(self) -> str:
        return self.combo.currentData()

    def values(self) -> np.ndarray:
        return np.linspace(self.lo.value(), self.hi.value(), self.n.value())


class TradeoffPage(Page):
    title = "Trade-offs (what-if)"
    subtitle = ("Name the inputs to vary and the consequences to report; the engine fills the "
                "table. No row is ranked or highlighted as preferred — the designer reads the "
                "table and decides. Out-of-band cells are coloured as information.")

    def __init__(self, ctrl) -> None:
        super().__init__(ctrl)
        self.runner: W.Runner | None = None
        top = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.root.addWidget(top, 1)

        # ---- setup panel
        setup = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(setup)
        sl.setContentsMargins(0, 0, 0, 0)
        self.mode = QtWidgets.QTabWidget()
        grid_w = QtWidgets.QWidget()
        gl = QtWidgets.QVBoxLayout(grid_w)
        self.pa = _ParamRow("A")
        self.pb = _ParamRow("B", optional=True)
        self.pb.enable.setChecked(False)
        gl.addWidget(self.pa)
        gl.addWidget(self.pb)
        gl.addWidget(QtWidgets.QLabel("Full factorial: every A × B combination is solved."))
        gl.addStretch(1)
        self.mode.addTab(grid_w, "Grid sweep")
        sens_w = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(sens_w)
        sv.addWidget(QtWidgets.QLabel("Inputs (central difference ± their step about the "
                                      "current design):"))
        self.sens_params = QtWidgets.QListWidget()
        sv.addWidget(self.sens_params, 1)
        self.mode.addTab(sens_w, "Sensitivity matrix")
        q_w = QtWidgets.QWidget()
        ql = QtWidgets.QFormLayout(q_w)
        self.q_param = _ParamRow("Input")
        self.q_param.n.setValue(9)
        self.q_out = QtWidgets.QComboBox()
        self.q_target = QtWidgets.QDoubleSpinBox()
        self.q_target.setRange(-1e6, 1e6)
        self.q_target.setDecimals(5)
        ql.addRow(self.q_param)
        ql.addRow("Output", self.q_out)
        ql.addRow("Target value", self.q_target)
        ql.addRow(QtWidgets.QLabel("Every crossing inside the range is returned. The target is an "
                                   "argument to this query, not a property of the design."))
        self.mode.addTab(q_w, "Target query")
        sl.addWidget(self.mode)

        ob = QtWidgets.QGroupBox("Outputs (consequences to report)")
        ol = QtWidgets.QVBoxLayout(ob)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("filter…")
        self.filter.textChanged.connect(self._filter_outputs)
        self.outputs = QtWidgets.QListWidget()
        ol.addWidget(self.filter)
        ol.addWidget(self.outputs, 1)
        sl.addWidget(ob, 1)

        run = QtWidgets.QHBoxLayout()
        self.parallel = QtWidgets.QCheckBox("parallel (process pool)")
        self.parallel.setChecked(True)
        self.btn_run = button("Run", self._run, primary=True)
        self.btn_cancel = button("Cancel", self._cancel)
        self.btn_cancel.setEnabled(False)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        run.addWidget(self.parallel)
        run.addWidget(self.btn_run)
        run.addWidget(self.btn_cancel)
        run.addWidget(self.bar, 1)
        sl.addLayout(run)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        sl.addWidget(self.status)
        setup.setMinimumWidth(560)
        top.addWidget(setup)

        # ---- results
        res = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plots = QtWidgets.QTabWidget()
        self.lines = pg.GraphicsLayoutWidget()
        self.plots.addTab(self.lines, "Curves")
        heat_w = QtWidgets.QWidget()
        hl = QtWidgets.QVBoxLayout(heat_w)
        self.heat_out = QtWidgets.QComboBox()
        self.heat_out.currentIndexChanged.connect(lambda _i: self._draw_heat())
        hl.addWidget(self.heat_out)
        self.heat = pg.PlotWidget()
        hl.addWidget(self.heat, 1)
        self.plots.addTab(heat_w, "Map (2 inputs)")
        self.q_plot = plot("Target query scan", "input", "output")
        self.plots.addTab(self.q_plot, "Query scan")
        res.addWidget(self.plots)
        tbox = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(tbox)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(banner("This table is the result. It is not sorted by any merit and no row "
                            "is preferred; cells outside their design-intent band are tinted."))
        self.table = QtWidgets.QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        tl.addWidget(self.table, 1)
        tl.addWidget(button("Export table CSV…", self._export))
        res.addWidget(tbox)
        res.setSizes([430, 420])
        top.addWidget(res)
        top.setSizes([600, 900])

        self.job = Debounced(0, self)
        self.job.progress.connect(lambda f, m: (self.bar.setValue(int(100 * f)),
                                                self.status.setText(f"solving {m}")))
        self.job.failed.connect(self._failed)
        self._result: object = None
        self._csv = ""
        self._populated_mode = ""
        self._keys: tuple[str, ...] = ()
        self._outputs_touched = False
        self._filling_outputs = False
        self.outputs.itemChanged.connect(self._outputs_changed)
        for row in (self.pa, self.pb, self.q_param):
            row.combo.currentIndexChanged.connect(lambda _i, r=row: self._default_range(r))

    # -- population -------------------------------------------------------------------------
    def on_project(self) -> None:
        mode = self.ctrl.project.geometry_mode
        if mode == self._populated_mode:
            return
        self._populated_mode = mode
        params = W.params_for(self.ctrl.project)
        for row in (self.pa, self.pb, self.q_param):
            row.combo.blockSignals(True)
            row.combo.clear()
            for p in params:
                row.combo.addItem(f"{p.label} [{p.unit}]", p.path)
            row.combo.blockSignals(False)
        self.sens_params.clear()
        for p in params:
            it = QtWidgets.QListWidgetItem(f"{p.label} [{p.unit}]  (step {p.step:g})")
            it.setData(QtCore.Qt.UserRole, p.path)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if p.path in (
                "front.rc_height_mm", "front.fvsa_length_mm", "front.kpi_deg",
                "steering.rack_z_mm", "steering.steer_arm_angle_deg") else QtCore.Qt.Unchecked)
            self.sens_params.addItem(it)
        if mode == "synthesis":
            self._select(self.pa, "front.rc_height_mm")
            self._select(self.pb, "front.fvsa_length_mm")
            self._select(self.q_param, "steering.rack_z_mm")
        for r in (self.pa, self.pb, self.q_param):
            self._default_range(r)

    def _select(self, row: _ParamRow, path: str) -> None:
        i = row.combo.findData(path)
        if i >= 0:
            row.combo.setCurrentIndex(i)

    def _default_range(self, row: _ParamRow) -> None:
        path = row.path()
        if not path:
            return
        try:
            p = W.param_by_path(path)
            v = p.get(self.ctrl.project)
        except (KeyError, AttributeError, TypeError, ValueError):
            return
        row.lo.setValue(v - 2 * p.step)
        row.hi.setValue(v + 2 * p.step)

    def on_kpis(self, ks) -> None:
        keys = tuple(ks.items)
        if keys == self._keys:
            return
        self._keys = keys
        checked = set(self._checked_outputs())
        if not self._outputs_touched:
            checked |= set(DEFAULT_OUTPUTS)
        current_q = self.q_out.currentData()
        self._filling_outputs = True
        self.outputs.clear()
        self.q_out.clear()
        for k in ks.items.values():
            label = f"{k.scope}: {k.label} [{k.unit}]"
            it = QtWidgets.QListWidgetItem(label)
            it.setData(QtCore.Qt.UserRole, k.key)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if k.key in checked else QtCore.Qt.Unchecked)
            self.outputs.addItem(it)
            self.q_out.addItem(label, k.key)
        self._filling_outputs = False
        i = self.q_out.findData(current_q or "front.bump_steer")
        if i >= 0:
            self.q_out.setCurrentIndex(i)
        self._filter_outputs(self.filter.text())

    def _outputs_changed(self, _it) -> None:
        if not self._filling_outputs:
            self._outputs_touched = True

    def _filter_outputs(self, text: str) -> None:
        t = text.lower()
        for i in range(self.outputs.count()):
            it = self.outputs.item(i)
            it.setHidden(bool(t) and t not in it.text().lower())

    def _checked_outputs(self) -> list[str]:
        return [self.outputs.item(i).data(QtCore.Qt.UserRole) for i in range(self.outputs.count())
                if self.outputs.item(i).checkState() == QtCore.Qt.Checked]

    # -- run --------------------------------------------------------------------------------
    def _ensure_runner(self) -> W.Runner:
        want = self.parallel.isChecked()
        if self.runner is None or self.runner.parallel != want:
            if self.runner is not None:
                self.runner.close()
            self.runner = W.Runner(parallel=want)
        return self.runner

    def _run(self) -> None:
        proj = self.ctrl.snapshot()
        runner = self._ensure_runner()
        mode = self.mode.currentIndex()
        outputs = self._checked_outputs()
        if mode in (0, 1) and not outputs:
            self.status.setText("Tick at least one output.")
            return
        if mode == 0:
            params = [self.pa.path()]
            axes = [self.pa.values()]
            if self.pb.enable.isChecked():
                params.append(self.pb.path())
                axes.append(self.pb.values())
            n = int(np.prod([len(a) for a in axes]))
            self.status.setText(f"{n} combinations…")

            def work(progress, cancelled):
                return W.run_grid(proj, params, axes, outputs, runner, progress, cancelled)
        elif mode == 1:
            params = [self.sens_params.item(i).data(QtCore.Qt.UserRole)
                      for i in range(self.sens_params.count())
                      if self.sens_params.item(i).checkState() == QtCore.Qt.Checked]
            if not params:
                self.status.setText("Tick at least one input.")
                return

            def work(progress, cancelled):
                return W.sensitivity(proj, params, outputs, runner=runner, progress=progress,
                                     cancelled=cancelled)
        else:
            par, out = self.q_param.path(), self.q_out.currentData()
            lo, hi, n, tgt = self.q_param.lo.value(), self.q_param.hi.value(), \
                self.q_param.n.value(), self.q_target.value()

            def work(progress, cancelled):
                return W.target_query(proj, par, out, tgt, lo, hi, n, runner, progress, cancelled)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.bar.setValue(0)
        self.job.request(work, self._done, immediate=True)

    def _cancel(self) -> None:
        self.job.cancel()
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status.setText("cancelled")

    def _failed(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status.setText("failed: " + msg.splitlines()[0])

    def _done(self, res) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.bar.setValue(100)
        self._result = res
        if isinstance(res, W.GridResult):
            self._show_grid(res)
        elif isinstance(res, W.SensitivityResult):
            self._show_sens(res)
        elif isinstance(res, W.TargetQueryResult):
            self._show_query(res)

    # -- presentation ------------------------------------------------------------------------
    def _label(self, key: str) -> str:
        k = self.ctrl.kpis.get(key)
        if k is not None:
            return f"{k.scope}: {k.label} [{k.unit}]"
        try:
            p = W.param_by_path(key)
            return f"{p.label} [{p.unit}]"
        except KeyError:
            return key

    def _set_table(self, head: list[str], rows: list[list[str]], status: list[list[str]] | None
                   = None) -> None:
        self.table.clear()
        self.table.setColumnCount(len(head))
        self.table.setHorizontalHeaderLabels(head)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, v in enumerate(r):
                it = QtWidgets.QTableWidgetItem(v)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                st = status[i][j] if status else ""
                if st in ("out", "fail"):
                    c = QtGui.QColor(theme.status_color(st))
                    c.setAlpha(80)
                    it.setBackground(c)
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(head)
        w.writerows(rows)
        self._csv = buf.getvalue()

    def _show_grid(self, g: W.GridResult) -> None:
        head = [self._label(p) for p in g.params] + [self._label(o) for o in g.outputs] + ["error"]
        rows, stat = [], []
        for pt in g.points:
            rows.append([f"{v:.4g}" for v in pt.assignment]
                        + [("—" if not math.isfinite(pt.values[o]) else f"{pt.values[o]:.5g}")
                           for o in g.outputs] + [pt.error])
            stat.append([""] * len(g.params) + [pt.status[o] for o in g.outputs] + [""])
        self._set_table(head, rows, stat)
        fails = sum(1 for pt in g.points if pt.error)
        extra = f"; {fails} could not be built/solved (shown as —)" if fails else ""
        self.status.setText(f"{len(g.points)} combinations solved" + extra)
        # curves: small multiples, one per output, x = A, one line per B value
        self.lines.clear()
        cols = 3
        for i, o in enumerate(g.outputs):
            p = self.lines.addPlot(row=i // cols, col=i % cols)
            p.setTitle(self._label(o), size="8pt")
            p.showGrid(x=True, y=True, alpha=0.25)
            arr = g.array(o)
            x = g.axes[0]
            if arr.ndim == 1:
                p.plot(x, arr, pen=pen(0), symbol="o", symbolSize=5, connect="finite")
            else:
                for j in range(arr.shape[1]):
                    p.plot(x, arr[:, j], pen=pen(j), connect="finite")
            p.setLabel("bottom", self._label(g.params[0]))
        self.heat_out.blockSignals(True)
        self.heat_out.clear()
        for o in g.outputs:
            self.heat_out.addItem(self._label(o), o)
        self.heat_out.blockSignals(False)
        self._draw_heat()
        self.plots.setCurrentIndex(1 if len(g.params) == 2 else 0)

    def _draw_heat(self) -> None:
        g = self._result
        self.heat.clear()
        if not isinstance(g, W.GridResult) or len(g.params) != 2:
            return
        o = self.heat_out.currentData()
        if o is None:
            return
        arr = g.array(o)
        a, b = g.axes
        img = pg.ImageItem(arr)
        da = (a[-1] - a[0]) / max(len(a) - 1, 1)
        db = (b[-1] - b[0]) / max(len(b) - 1, 1)
        img.setRect(QtCore.QRectF(a[0] - da / 2, b[0] - db / 2, (a[-1] - a[0]) + da,
                                  (b[-1] - b[0]) + db))
        finite = arr[np.isfinite(arr)]
        cmap = pg.colormap.get("viridis")
        img.setColorMap(cmap)
        if finite.size:
            img.setLevels((float(finite.min()), float(finite.max())))
        self.heat.addItem(img)
        self.heat.setLabel("bottom", self._label(g.params[0]))
        self.heat.setLabel("left", self._label(g.params[1]))
        self.heat.setTitle(self._label(o), size="9pt")
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                v = arr[i, j]
                t = pg.TextItem("—" if not np.isfinite(v) else f"{v:.4g}", color="w",
                                anchor=(0.5, 0.5))
                t.setPos(x, y)
                self.heat.addItem(t)

    def _show_sens(self, s: W.SensitivityResult) -> None:
        head = ["output (base value)"] + [f"{self._label(p)} +{s.steps[p]:g}" for p in s.params]
        rows = []
        stat = []
        for o in s.outputs:
            vals = [s.per_step(o, p) for p in s.params]
            rows.append([f"{self._label(o)} ({s.base.get(o, float('nan')):.4g})"]
                        + [("—" if not math.isfinite(v) else f"{v:+.4g}") for v in vals])
            stat.append([""] * (1 + len(vals)))
        self._set_table(head, rows, stat)
        # tint by relative magnitude per row (information density, not merit)
        for i, o in enumerate(s.outputs):
            vals = np.array([abs(s.per_step(o, p)) for p in s.params])
            mx = np.nanmax(vals) if np.isfinite(vals).any() else 0.0
            for j, v in enumerate(vals):
                if mx > 0 and np.isfinite(v):
                    c = QtGui.QColor(theme.current().accent)
                    c.setAlpha(int(20 + 120 * v / mx))
                    self.table.item(i, j + 1).setBackground(c)
        self.status.setText("Each cell: change of the output for one step of the input, about "
                            "the current design (central difference).")

    def _show_query(self, r: W.TargetQueryResult) -> None:
        self.q_plot.clear()
        self.q_plot.plot(r.scan_x, r.scan_y, pen=pen(0), symbol="o", symbolSize=6,
                         connect="finite")
        self.q_plot.addItem(pg.InfiniteLine(r.target, angle=0,
                                            pen=pg.mkPen(theme.current().warn, width=1)))
        for x in r.roots:
            self.q_plot.addItem(pg.InfiniteLine(x, angle=90,
                                                pen=pg.mkPen(theme.current().ok, width=1)))
        self.q_plot.setLabel("bottom", self._label(r.param))
        self.q_plot.setLabel("left", self._label(r.output))
        self.plots.setCurrentWidget(self.q_plot)
        self._set_table([self._label(r.param), self._label(r.output)],
                        [[f"{x:.5g}", f"{y:.5g}"] for x, y in zip(r.scan_x, r.scan_y,
                                                                  strict=True)])
        roots = ", ".join(f"{x:.5g}" for x in r.roots) or "none"
        self.status.setText(f"Roots: {roots}. {r.message}")
        if r.roots:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Target query")
            box.setText(f"{self._label(r.param)} = {roots}\n\n{r.message}")
            ap = box.addButton(f"Apply {r.roots[0]:.5g} (my decision)",
                               QtWidgets.QMessageBox.AcceptRole)
            box.addButton("Close", QtWidgets.QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is ap:
                self.ctrl.set_value(r.param, float(r.roots[0]))

    def _export(self) -> None:
        if self._csv:
            save_text(self, self._csv, "tradeoff.csv", "CSV (*.csv)")

    def shutdown(self) -> None:
        self.job.cancel()
        if self.runner is not None:
            self.runner.close()
