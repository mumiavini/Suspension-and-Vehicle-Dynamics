"""Trade-offs: the sweep / what-if engine. Fills tables of consequences; never picks a row."""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import html
import io
import json
import math
import textwrap
from collections.abc import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ...engine import whatif as W
from ...engine.project import Project, project_to_dict
from .. import theme
from ..asyncjob import Debounced
from ..widgets import banner, button, pen, plot, save_text, scroll
from .base import Page

DEFAULT_OUTPUTS = ("front.camber_gain", "front.rc_static", "front.rc_span",
                   "front.roll_outer_camber", "front.bump_steer_peak", "front.anti",
                   "steering.ackermann", "steering.ratio")
ERROR_COL_MAX_PX = 420
CURVE_MIN_W, CURVE_MIN_H = 220, 170     # px per small-multiple plot
HEADER_WRAP = 24                         # characters per table header line


def axis_decimals(values: Sequence[float]) -> int:
    """Decimals that show a sweep axis without collapsing distinct values.

    At least enough for the smallest step, and up to three more so values such
    as 526.966 mm print exactly (``.4g`` printed 527 and 1540.25 as 1540).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 3
    gaps = np.diff(np.unique(v))
    d = max(0, -math.floor(math.log10(float(gaps.min())))) if gaps.size else 0
    scale = max(1.0, float(np.abs(v).max()))
    for dd in range(d, d + 4):
        if np.all(np.abs(np.round(v, dd) - v) <= 1e-9 * scale):
            return dd
    return d + 3


def _fmt_out(y: float) -> str:
    return "—" if not math.isfinite(y) else f"{y:.5g}"


def _raw(v: float) -> str:
    """Full-precision float for the CSV (``repr`` round-trips); NaN as ``nan``."""
    return repr(float(v)) if math.isfinite(v) else "nan"


def project_sha1(project: Project) -> str:
    text = json.dumps(project_to_dict(project), indent=2, ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def project_stamp(project: Project) -> str:
    """Which design a result was computed for: name, geometry mode, sha1 of the
    project JSON and the time."""
    ts = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    return (f"project {project.name}, geometry mode {project.geometry_mode}, "
            f"sha1 {project_sha1(project)} (project JSON), computed {ts}")


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
        q_help = QtWidgets.QLabel(
            "The scan brackets every sign change of output − target and refines it; interior "
            "extrema of the scan are refined too, so a V-shaped output's two crossings are "
            "found. Crossings closer together than the scan spacing can still be missed. The "
            "target is an argument to this query, not a property of the design.")
        q_help.setWordWrap(True)
        ql.addRow(q_help)
        self.mode.addTab(q_w, "Target query")
        sl.addWidget(self.mode)

        ob = QtWidgets.QGroupBox("Outputs (consequences to report)")
        self.out_box = ob
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
        # small multiples keep a readable size and scroll, rather than shrink until unreadable
        self.lines_scroll = scroll(self.lines)
        self.plots.addTab(self.lines_scroll, "Curves")
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
        self.stamp_label = QtWidgets.QLabel("")
        self.stamp_label.setObjectName("muted")
        self.stamp_label.setWordWrap(True)
        tl.addWidget(self.stamp_label)
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
        self._run_stamp = ""          # project_stamp() of the snapshot being computed
        self._run_query = ""          # one line describing the query
        self._result_stamp = ""
        self._result_query = ""
        self._result_sha = ""
        self._pending_sha = ""
        self._populated_mode = ""
        self._keys: tuple[str, ...] = ()
        self._outputs_touched = False
        self._filling_outputs = False
        self.outputs.itemChanged.connect(self._outputs_changed)
        for row in (self.pa, self.pb, self.q_param):
            row.combo.currentIndexChanged.connect(lambda _i, r=row: self._default_range(r))
        self.mode.currentChanged.connect(self._mode_changed)

    def _mode_changed(self, index: int) -> None:
        # a target query reports one output, chosen in its own tab
        self.out_box.setVisible(index != 2)

    # -- population -------------------------------------------------------------------------
    def on_project(self) -> None:
        self._update_stamp_label()
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
            query = "grid sweep: " + " × ".join(
                f"{p} [{float(a[0])!r} … {float(a[-1])!r}, {len(a)} values]"
                for p, a in zip(params, axes, strict=True))

            def work(progress, cancelled):
                return W.run_grid(proj, params, axes, outputs, runner, progress, cancelled)
        elif mode == 1:
            params = [self.sens_params.item(i).data(QtCore.Qt.UserRole)
                      for i in range(self.sens_params.count())
                      if self.sens_params.item(i).checkState() == QtCore.Qt.Checked]
            if not params:
                self.status.setText("Tick at least one input.")
                return
            query = "sensitivity (± one step about the current design): " + ", ".join(params)

            def work(progress, cancelled):
                return W.sensitivity(proj, params, outputs, runner=runner, progress=progress,
                                     cancelled=cancelled)
        else:
            par, out = self.q_param.path(), self.q_out.currentData()
            lo, hi, n, tgt = self.q_param.lo.value(), self.q_param.hi.value(), \
                self.q_param.n.value(), self.q_target.value()
            query = (f"target query: {par} in [{lo!r}, {hi!r}] ({n} scan points) for "
                     f"{out} = {tgt!r}")

            def work(progress, cancelled):
                return W.target_query(proj, par, out, tgt, lo, hi, n, runner, progress, cancelled)

        self._run_stamp = project_stamp(proj)
        self._pending_sha = project_sha1(proj)
        self._run_query = query
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
        self._result_stamp, self._result_query = self._run_stamp, self._run_query
        self._result_sha = self._pending_sha
        self._update_stamp_label()
        self._clear_plots()
        if isinstance(res, W.GridResult):
            self._show_grid(res)
        elif isinstance(res, W.SensitivityResult):
            self._show_sens(res)
        elif isinstance(res, W.TargetQueryResult):
            self._show_query(res)

    def _clear_plots(self) -> None:
        """Remove the plots of an earlier result (of any mode): none may sit beside a
        newer one."""
        self.lines.clear()
        self.lines.setMinimumSize(0, 0)
        self.heat.clear()
        self.heat.setTitle(None)
        self.q_plot.clear()
        for pw, x, y in ((self.heat, "", ""), (self.q_plot, "input", "output")):
            pw.setLabel("bottom", x)
            pw.setLabel("left", y)
        self.heat_out.blockSignals(True)
        self.heat_out.clear()
        self.heat_out.blockSignals(False)

    def _update_stamp_label(self) -> None:
        if not self._result_stamp:
            self.stamp_label.setText("")
            return
        text = f"Result for {self._result_stamp}."
        if self._result_sha and project_sha1(self.ctrl.project) != self._result_sha:
            text += " The design has changed since this result was computed."
        self.stamp_label.setText(text)

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
                   = None, raw: list[list[str]] | None = None, raw_head: list[str] | None = None,
                   tips: list[list[str]] | None = None) -> None:
        """Show ``rows`` (display strings). The CSV is built from ``raw`` (full-precision
        values) when given, under a comment line naming the design it was computed for."""
        self.table.clear()
        self.table.setColumnCount(len(head))
        # long KPI labels wrap (the unit stays visible) instead of widening the column past
        # its right-aligned values; the full label is also the header tooltip
        self.table.setHorizontalHeaderLabels([textwrap.fill(h, HEADER_WRAP) for h in head])
        for j, h in enumerate(head):
            hi = self.table.horizontalHeaderItem(j)
            if hi is not None:
                hi.setToolTip(h)
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
                tip = tips[i][j] if tips else ""
                if tip:
                    it.setToolTip(tip)
                self.table.setItem(i, j, it)
        self.table.resizeColumnsToContents()
        if head and head[-1] in ("error", "error / note"):
            last = len(head) - 1
            if self.table.columnWidth(last) > ERROR_COL_MAX_PX:
                self.table.setColumnWidth(last, ERROR_COL_MAX_PX)
        buf = io.StringIO()
        if self._result_stamp:
            buf.write(f"# {self._result_stamp}\n")
        if self._result_query:
            buf.write(f"# {self._result_query}\n")
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(raw_head if raw_head is not None else head)
        w.writerows(raw if raw is not None else rows)
        self._csv = buf.getvalue()

    def _show_grid(self, g: W.GridResult) -> None:
        head = [self._label(p) for p in g.params] + [self._label(o) for o in g.outputs] + ["error"]
        decs = [axis_decimals(a) for a in g.axes]
        rows, stat, tips, raw = [], [], [], []
        for pt in g.points:
            rows.append([f"{v:.{d}f}" for v, d in zip(pt.assignment, decs, strict=True)]
                        + [_fmt_out(pt.values[o]) for o in g.outputs] + [pt.error])
            stat.append([""] * len(g.params) + [pt.status[o] for o in g.outputs] + [""])
            tips.append([""] * len(g.params)
                        + [(f"failed: {pt.errors[o]}" if o in pt.errors else "")
                           for o in g.outputs]
                        + [pt.error.replace("; ", "\n")])
            raw.append([_raw(v) for v in pt.assignment] + [_raw(pt.values[o]) for o in g.outputs]
                       + [pt.error])
        self._set_table(head, rows, stat, raw, tips=tips)
        n = len(g.points)
        n_ok = sum(1 for pt in g.points if pt.fully_solved)
        text = f"{n_ok} of {n} combinations fully solved"
        if n_ok < n:
            text += (f"; {n - n_ok} with failed outputs (—). Hover a — cell or read the error "
                     "column for the reason.")
        self.status.setText(text)
        # curves: small multiples, one per output, x = A, one line per B value; as many
        # columns as fit (plot titles carry the unit and are never clipped), scrolling down
        avail = (max(self.lines_scroll.viewport().width(), self.plots.width() - 24)
                 - self.lines_scroll.verticalScrollBar().sizeHint().width())
        for cols in range(max(1, min(3, avail // CURVE_MIN_W)), 0, -1):
            self.lines.clear()
            self._draw_curves(g, cols)
            need = self.lines.ci.layout.effectiveSizeHint(QtCore.Qt.MinimumSize).width()
            if need <= avail:
                break
        self.lines.setMinimumSize(max(cols * CURVE_MIN_W, math.ceil(need)),
                                  math.ceil(len(g.outputs) / cols) * CURVE_MIN_H)
        self.heat_out.blockSignals(True)
        self.heat_out.clear()
        for o in g.outputs:
            self.heat_out.addItem(self._label(o), o)
        self.heat_out.blockSignals(False)
        self._draw_heat()
        self.plots.setCurrentIndex(1 if len(g.params) == 2 else 0)

    def _draw_curves(self, g: W.GridResult, cols: int) -> None:
        for i, o in enumerate(g.outputs):
            p = self.lines.addPlot(row=i // cols, col=i % cols)
            p.showGrid(x=True, y=True, alpha=0.25)
            arr = g.array(o)
            x = g.axes[0]
            title = self._label(o)
            if not np.isfinite(arr).any():
                title += " — every point failed"
            elif arr.ndim == 1:
                p.plot(x, arr, pen=pen(0), symbol="o", symbolSize=5, connect="finite")
            else:
                for j in range(arr.shape[1]):
                    if np.isfinite(arr[:, j]).any():     # an all-failed line has nothing to draw
                        p.plot(x, arr[:, j], pen=pen(j), connect="finite")
            fin = arr[np.isfinite(arr)]
            if fin.size and float(np.ptp(fin)) <= 1e-9 * max(1.0, float(np.abs(fin).max())):
                # constant to rounding: a flat line on a readable range, not 15-digit ticks
                m = float(fin.mean())
                pad = max(1e-3 * abs(m), 1e-6)
                p.setYRange(m - pad, m + pad, padding=0)
            # wrapped, so a long KPI label (unit last) does not set the column width
            p.setTitle("<br>".join(html.escape(t) for t in textwrap.wrap(title, HEADER_WRAP + 4)),
                       size="8pt")
            p.setLabel("bottom", self._label(g.params[0]))

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
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        if finite.size:
            img.setLevels((lo, hi))
        self.heat.addItem(img)
        self.heat.setLabel("bottom", self._label(g.params[0]))
        self.heat.setLabel("left", self._label(g.params[1]))
        self.heat.setTitle(self._label(o), size="9pt")
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                v = arr[i, j]
                if np.isfinite(v):
                    t = (v - lo) / (hi - lo) if hi > lo else 0.5
                    bg = cmap.map(float(t), mode="qcolor")
                    # readable on light (yellow) and dark (purple) cells alike
                    lum = 0.2126 * bg.red() + 0.7152 * bg.green() + 0.0722 * bg.blue()
                    color = "#000000" if lum > 140 else "#ffffff"
                else:
                    color = theme.current().text
                t_item = pg.TextItem("—" if not np.isfinite(v) else f"{v:.4g}", color=color,
                                     anchor=(0.5, 0.5))
                t_item.setPos(x, y)
                self.heat.addItem(t_item)

    def _show_sens(self, s: W.SensitivityResult) -> None:
        head = ["output (base value)"] + [f"{self._label(p)} +{s.steps[p]:g}" for p in s.params]
        rows, stat, tips, raw = [], [], [], []
        mags: list[list[float]] = []
        kinks = 0
        for o in s.outputs:
            row = [f"{self._label(o)} ({s.base.get(o, float('nan')):.4g})"]
            trow, mrow = [""], []
            rrow = [self._label(o), _raw(s.base.get(o, float("nan")))]
            for p in s.params:
                v = s.per_step(o, p)
                dp, dm = s.one_sided(o, p)
                if s.is_kink(o, p):
                    kinks += 1
                    row.append(f"{dp:+.3g} / {dm:+.3g} (kink)")
                    mrow.append(max(abs(dp), abs(dm)))
                else:
                    row.append("—" if not math.isfinite(v) else f"{v:+.4g}")
                    mrow.append(abs(v))
                trow.append(f"Δ+ (input +{s.steps[p]:g}): {dp:+.6g}\n"
                            f"Δ− (input −{s.steps[p]:g}): {dm:+.6g}\n"
                            f"central difference per +{s.steps[p]:g}: {v:+.6g}")
                rrow += [_raw(v), _raw(dp), _raw(dm)]
            rows.append(row)
            tips.append(trow)
            raw.append(rrow)
            mags.append(mrow)
            stat.append([""] * (1 + len(s.params)))
        raw_head = ["output", "base"]
        for p in s.params:
            lab = self._label(p)
            raw_head += [f"{lab} central per +{s.steps[p]:g}", f"{lab} Δ+ (+{s.steps[p]:g})",
                         f"{lab} Δ− (−{s.steps[p]:g})"]
        self._set_table(head, rows, stat, raw, raw_head, tips)
        # tint by relative magnitude per row (information density, not merit)
        for i, mrow in enumerate(mags):
            vals = np.array(mrow, dtype=float)
            mx = np.nanmax(vals) if np.isfinite(vals).any() else 0.0
            for j, v in enumerate(vals):
                if mx > 0 and np.isfinite(v):
                    c = QtGui.QColor(theme.current().accent)
                    c.setAlpha(int(20 + 120 * v / mx))
                    self.table.item(i, j + 1).setBackground(c)
        text = ("Each cell: change of the output for one step of the input, about the current "
                "design (central difference). Hover a cell for both one-sided changes.")
        if kinks:
            text += (f" {kinks} cell(s) marked (kink) show Δ+ / Δ− — the change for +step and "
                     "for −step — because they have the same sign or very different sizes; a "
                     "central difference would average them away.")
        self.status.setText(text)

    def _show_query(self, r: W.TargetQueryResult) -> None:
        if np.isfinite(r.scan_y).any():
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
        dx = axis_decimals(r.scan_x)
        dr = max(dx, -math.floor(math.log10(r.xtol)) if r.xtol > 0 else dx)
        errs = r.scan_errors or [""] * len(r.scan_x)
        head = ["row", self._label(r.param), self._label(r.output), "error / note"]
        rows, stat, tips, raw = [], [], [], []
        for k, x in enumerate(r.roots, 1):
            rows.append([f"root {k}", f"{x:.{dr}f}", f"{r.target:.6g} (target)", ""])
            stat.append(["", "", "", ""])
            tips.append(["", f"{x!r}", "", ""])
            raw.append(["root", _raw(x), _raw(r.target), ""])
        for a, b, why in r.unrefined_brackets:
            rows.append(["unrefined", f"{a:.{dx}f} … {b:.{dx}f}", "—", why])
            stat.append(["", "", "fail", ""])
            tips.append(["", "", "", why])
            raw.append(["unrefined bracket", _raw(a), "", f"bracket [{a!r}, {b!r}]: {why}"])
        for x, y, e in zip(r.scan_x, r.scan_y, errs, strict=True):
            ok = math.isfinite(y)
            rows.append(["scan", f"{x:.{dx}f}", _fmt_out(y), e])
            stat.append(["", "", "" if ok else "fail", ""])
            tips.append(["", "", f"failed: {e}" if (e and not ok) else "", e])
            raw.append(["scan", _raw(x), _raw(y), e])
        self._set_table(head, rows, stat, raw, ["kind", r.param, r.output, "note"], tips)
        roots = ", ".join(f"{x:.{dr}f}" for x in r.roots) or "none"
        self.status.setText(f"Roots: {roots}. {r.message}")
        if r.roots:
            self._offer_roots(r, dr)

    def _offer_roots(self, r: W.TargetQueryResult, decimals: int) -> None:
        """One Apply button per root: the designer picks; nothing is pre-selected."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Target query")
        listed = ", ".join(f"{x:.{decimals}f}" for x in r.roots)
        box.setText(f"{self._label(r.param)} = {listed}\n\n{r.message}")
        buttons = {box.addButton(f"Apply {x:.{decimals}f} (my decision)",
                                 QtWidgets.QMessageBox.AcceptRole): x for x in r.roots}
        box.addButton("Close", QtWidgets.QMessageBox.RejectRole)
        box.exec()
        x = buttons.get(box.clickedButton())
        if x is not None:
            self.apply_root(r, float(x), decimals)

    def apply_root(self, r: W.TargetQueryResult, x: float, decimals: int = 6) -> None:
        """Apply one root through its :class:`W.Param`, exactly as the query evaluated it:
        a mirrored hardpoint moves on both sides, so the car stays symmetric."""
        try:
            par = W.param_by_path(r.param)
        except KeyError:
            self.ctrl.set_value(r.param, float(x))
        else:
            self.ctrl.apply(lambda p, par=par, v=float(x): par.apply(p, v))
        others = ", ".join(f"{y:.{decimals}f}" for y in r.roots if y != x) or "none"
        self.status.setText(f"Applied {self._label(r.param)} = {x:.{decimals}f} (your "
                            f"decision; Ctrl+Z undoes). Other roots: {others}.")

    def _export(self) -> None:
        if self._csv:
            save_text(self, self._csv, "tradeoff.csv", "CSV (*.csv)")

    def shutdown(self) -> None:
        self.job.cancel()
        if self.runner is not None:
            self.runner.close()
