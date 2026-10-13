"""Reusable widgets: parameter forms bound to project paths, KPI tables, cards, plots."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..engine.kpis import KPI, KpiSet
from ..engine.project import get_path
from . import theme
from .controller import ProjectController

# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #


def page_header(title: str, subtitle: str = "") -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 6)
    lay.setSpacing(2)
    t = QtWidgets.QLabel(title)
    t.setObjectName("pageTitle")
    lay.addWidget(t)
    if subtitle:
        s = QtWidgets.QLabel(subtitle)
        s.setObjectName("pageSubtitle")
        s.setWordWrap(True)
        lay.addWidget(s)
    return w


def banner(text: str) -> QtWidgets.QFrame:
    f = QtWidgets.QFrame()
    f.setObjectName("banner")
    lay = QtWidgets.QHBoxLayout(f)
    lay.setContentsMargins(10, 6, 10, 6)
    lab = QtWidgets.QLabel(text)
    lab.setWordWrap(True)
    lab.setTextFormat(QtCore.Qt.RichText)
    lab.setOpenExternalLinks(True)
    lay.addWidget(lab)
    f.label = lab  # type: ignore[attr-defined]
    return f


def scroll(widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(widget)
    sa.setFrameShape(QtWidgets.QFrame.NoFrame)
    return sa


def hbox(*widgets: QtWidgets.QWidget | None, stretch: bool = False) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        if x is None:
            lay.addStretch(1)
        else:
            lay.addWidget(x)
    if stretch:
        lay.addStretch(1)
    return w


def button(text: str, slot: Callable[[], Any] | None = None, primary: bool = False,
           tip: str = "") -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    if primary:
        b.setObjectName("primary")
    if tip:
        b.setToolTip(tip)
    if slot is not None:
        b.clicked.connect(lambda _=False: slot())
    return b


# --------------------------------------------------------------------------- #
# Parameter forms
# --------------------------------------------------------------------------- #

@dataclass
class Field:
    path: str
    label: str
    unit: str = ""
    lo: float = -1e6
    hi: float = 1e6
    step: float = 1.0
    dec: int = 2
    kind: str = "float"        # float | int | optfloat | choice | text | bool
    tip: str = ""
    choices: Sequence[str] = field(default_factory=tuple)


class ParamForm(QtWidgets.QWidget):
    """Groups of fields bound to dotted project paths through the controller.

    Spin boxes commit on arrow / wheel / Enter (not on every keystroke), and
    each commit is a recorded, undoable edit that recomputes the car at once.
    """

    edited = QtCore.Signal(str)

    def __init__(self, ctrl: ProjectController, groups: Sequence[tuple[str, Sequence[Field]]],
                 columns: int = 1, base: Callable[[], Any] | None = None,
                 setter: Callable[[str, Any], None] | None = None) -> None:
        super().__init__()
        self.ctrl = ctrl
        self._base = base or (lambda: ctrl.project)
        self._setter = setter or (lambda p, v: ctrl.set_value(p, v))
        self._editors: dict[str, tuple[Field, Any]] = {}
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        for i, (title, fields) in enumerate(groups):
            box = QtWidgets.QGroupBox(title.replace("&", "&&"))
            form = QtWidgets.QFormLayout(box)
            form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            form.setHorizontalSpacing(8)
            form.setVerticalSpacing(5)
            for f in fields:
                ed = self._make_editor(f)
                lab = QtWidgets.QLabel(f.label)
                if f.tip:
                    lab.setToolTip(f.tip)
                    ed.setToolTip(f.tip)
                form.addRow(lab, ed)
            grid.addWidget(box, i // columns, i % columns)
        self.refresh()

    def _make_editor(self, f: Field) -> QtWidgets.QWidget:
        if f.kind in ("float", "int"):
            sb: QtWidgets.QAbstractSpinBox
            if f.kind == "int":
                sb = QtWidgets.QSpinBox()
                sb.setRange(int(f.lo), int(f.hi))
                sb.setSingleStep(int(f.step) or 1)
            else:
                sb = QtWidgets.QDoubleSpinBox()
                sb.setRange(f.lo, f.hi)
                sb.setDecimals(f.dec)
                sb.setSingleStep(f.step)
            sb.setKeyboardTracking(False)
            sb.setSuffix(f"  {f.unit}" if f.unit else "")
            sb.setMinimumWidth(130)
            sb.setAlignment(QtCore.Qt.AlignRight)
            sb.valueChanged.connect(lambda v, p=f.path, k=f.kind: self._commit(
                p, int(v) if k == "int" else float(v)))
            self._editors[f.path] = (f, sb)
            return sb
        if f.kind == "optfloat":
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            cb = QtWidgets.QCheckBox()
            cb.setToolTip("Override (unchecked = use the shared value)")
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(f.lo, f.hi)
            sb.setDecimals(f.dec)
            sb.setSingleStep(f.step)
            sb.setKeyboardTracking(False)
            sb.setSuffix(f"  {f.unit}" if f.unit else "")
            sb.setMinimumWidth(110)
            lay.addWidget(cb)
            lay.addWidget(sb, 1)

            def commit(*_a: Any, p: str = f.path) -> None:
                self._commit(p, float(sb.value()) if cb.isChecked() else None)
                sb.setEnabled(cb.isChecked())

            cb.toggled.connect(commit)
            sb.valueChanged.connect(commit)
            self._editors[f.path] = (f, (cb, sb))
            return w
        if f.kind == "choice":
            co = QtWidgets.QComboBox()
            co.addItems(list(f.choices))
            co.currentTextChanged.connect(lambda v, p=f.path: self._commit(p, v))
            self._editors[f.path] = (f, co)
            return co
        if f.kind == "bool":
            cb = QtWidgets.QCheckBox()
            cb.toggled.connect(lambda v, p=f.path: self._commit(p, bool(v)))
            self._editors[f.path] = (f, cb)
            return cb
        le = QtWidgets.QLineEdit()
        le.editingFinished.connect(lambda p=f.path, e=le: self._commit(p, e.text()))
        self._editors[f.path] = (f, le)
        return le

    def _commit(self, path: str, value: Any) -> None:
        try:
            cur = get_path(self._base(), path)
        except (AttributeError, KeyError, IndexError):
            cur = object()
        if cur == value:
            return
        self._setter(path, value)
        self.edited.emit(path)

    def refresh(self) -> None:
        base = self._base()
        for path, (f, ed) in self._editors.items():
            try:
                v = get_path(base, path)
            except (AttributeError, KeyError, IndexError):
                continue
            if f.kind == "optfloat":
                cb, sb = ed
                cb.blockSignals(True)
                sb.blockSignals(True)
                cb.setChecked(v is not None)
                sb.setEnabled(v is not None)
                if v is not None:
                    sb.setValue(float(v))
                cb.blockSignals(False)
                sb.blockSignals(False)
                continue
            ed.blockSignals(True)
            if f.kind == "int":
                ed.setValue(int(v))
            elif f.kind == "float":
                ed.setValue(float(v))
            elif f.kind == "choice":
                ed.setCurrentText(str(v))
            elif f.kind == "bool":
                ed.setChecked(bool(v))
            else:
                ed.setText("" if v is None else str(v))
            ed.blockSignals(False)


# --------------------------------------------------------------------------- #
# KPI display
# --------------------------------------------------------------------------- #

def _item(text: str, status: str | None = None, align_right: bool = False,
          tip: str = "", bold: bool = False) -> QtWidgets.QTableWidgetItem:
    it = QtWidgets.QTableWidgetItem(text)
    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
    if align_right:
        it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    if status in ("in", "out", "fail"):
        c = QtGui.QColor(theme.status_color(status))
        c.setAlpha(60 if status == "in" else 90)
        it.setBackground(c)
    if tip:
        it.setToolTip(tip)
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    return it


STALE_TIP = ("stale: computed for the previous geometry; the swept KPIs are being "
             "recomputed")


def kpi_tip(k: KPI, stale: bool = False) -> str:
    lines = [f"<b>{k.label}</b>", f"value: {k.text()} {k.unit}"]
    if stale:
        lines.append(f"<i>{STALE_TIP}</i>")
    if k.band:
        lines.append(f"band: {k.band_text()} ({k.band_source}) → {theme.status_text(k.status)}")
    if k.source:
        lines.append(f"source: {k.source}")
    if k.note:
        lines.append(f"note: {k.note}")
    if k.error:
        lines.append(f"<span style='color:{theme.current().bad}'>failed: {k.error}</span>")
    return "<br>".join(lines)


def pair_band_text(f: KPI | None, r: KPI | None) -> tuple[str, str]:
    """Band cell text and tooltip for a front/rear row.

    One band when both axles share it; ``F … / R …`` when they differ, with
    ``—`` for an axle without a band, so a rear value never sits beside the
    front band.
    """
    bf = f.band_text() if f is not None else ""
    br = r.band_text() if r is not None else ""
    if f is None or r is None or (bf == br and f.band_source == r.band_source):
        k = f if f is not None else r
        return (bf or br), (k.band_source if k is not None else "")
    text = bf if bf == br else f"F {bf or '—'} / R {br or '—'}"
    tip = "\n".join(f"{axle}: {band or 'no band'}" + (f" ({src})" if src else "")
                    for axle, band, src in (("front", bf, f.band_source),
                                            ("rear", br, r.band_source)))
    return text, tip


def stale_keys(ks: KpiSet) -> frozenset[str]:
    """Keys of ``ks`` whose values are left over from the previous geometry.

    The controller's KPI set carries them (``ProjectController.kpis``); any
    other set has none.
    """
    return frozenset(getattr(ks, "stale_keys", frozenset()))


class KpiTable(QtWidgets.QTableWidget):
    """KPIs of some groups. Axle-scoped rows pivot into Front | Rear columns."""

    def __init__(self, groups: Iterable[str] | None = None,
                 group_filter: Callable[[str], bool] | None = None) -> None:
        super().__init__()
        self.groups = list(groups) if groups else None
        self.group_filter = group_filter
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setWordWrap(False)
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(["KPI", "Front / value", "Rear", "Unit", "Band", "Source"])
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        for c in (1, 2, 3, 4):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        self.stale = False

    def set_kpis(self, ks: KpiSet, stale_groups: set[str] | None = None) -> None:
        """Fill the table. Values are greyed when their group is in ``stale_groups`` or
        when the set marks them stale (the controller's set while swept KPIs recompute)."""
        stale_k = stale_keys(ks)
        groups = self.groups or ks.groups()
        if self.group_filter is not None:
            groups = [g for g in groups if self.group_filter(g)]
        rows: list[tuple[str, KPI | None, KPI | None, KPI | None]] = []  # group header rows too
        for g in groups:
            items = ks.by_group(g)
            if not items:
                continue
            rows.append((g, None, None, None))
            seen: set[str] = set()
            for k in items:
                if k.scope in ("front", "rear"):
                    if k.label in seen:
                        continue
                    seen.add(k.label)
                    f = next((x for x in items if x.label == k.label and x.scope == "front"),
                             None)
                    r = next((x for x in items if x.label == k.label and x.scope == "rear"),
                             None)
                    rows.append((k.label, f, r, None))
                else:
                    rows.append((k.label, None, None, k))
        self.clearSpans()
        self.setRowCount(len(rows))
        muted = QtGui.QColor(theme.current().text_muted)

        def is_stale(k: KPI | None, group_stale: bool) -> bool:
            return k is not None and k.ok and (group_stale or k.key in stale_k)

        def value_item(k: KPI | None, group_stale: bool) -> QtWidgets.QTableWidgetItem:
            if k is None:
                return _item("")
            if not is_stale(k, group_stale):
                return _item(k.text(), k.status, True, kpi_tip(k))
            # a stale value is not coloured against its band: it is not this car's value
            it = _item(k.text(), None, True, kpi_tip(k, True))
            it.setForeground(muted)
            fnt = it.font()
            fnt.setItalic(True)
            it.setFont(fnt)
            return it

        for i, (label, f, r, single) in enumerate(rows):
            if f is None and r is None and single is None:
                it = _item(label, bold=True)
                it.setForeground(QtGui.QColor(theme.current().accent))
                self.setItem(i, 0, it)
                for c in range(1, 6):
                    self.setItem(i, c, _item(""))
                self.setSpan(i, 0, 1, 6)
                continue
            ref = single or f or r
            assert ref is not None
            gs = stale_groups is not None and ref.group in stale_groups
            self.setItem(i, 0, _item("   " + label, tip=kpi_tip(ref, is_stale(ref, gs))))
            if single is not None:
                self.setItem(i, 1, value_item(single, gs))
                self.setItem(i, 2, _item(""))
                self.setItem(i, 4, _item(single.band_text(), tip=single.band_source))
            else:
                self.setItem(i, 1, value_item(f, gs))
                self.setItem(i, 2, value_item(r, gs))
                band, band_tip = pair_band_text(f, r)
                self.setItem(i, 4, _item(band, tip=band_tip))
            self.setItem(i, 3, _item(ref.unit))
            src = ref.source + (" — " + ref.note if ref.note else "")
            if (single or f or r) and any(x is not None and x.error for x in (single, f, r)):
                src = "FAILED: " + next(x.error for x in (single, f, r)
                                        if x is not None and x.error)
            self.setItem(i, 5, _item(src, tip=src))
        self.resizeRowsToContents()


class Card(QtWidgets.QFrame):
    def __init__(self, label: str, unit: str = "") -> None:
        super().__init__()
        self.setObjectName("card")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(2)
        self.value = QtWidgets.QLabel("—")
        self.value.setObjectName("cardValue")
        # Word-wrapped so a narrow card moves the unit to the next line rather than
        # clipping it ("[deg/mm]" cut to "[deg/m" reads as a unit 1000x larger).
        self.lab = QtWidgets.QLabel(label + (f"  [{unit}]" if unit else ""))
        self.lab.setObjectName("cardLabel")
        self.lab.setWordWrap(True)
        self.lab.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.sub = QtWidgets.QLabel("")
        self.sub.setObjectName("cardLabel")
        self.sub.setWordWrap(True)
        lay.addWidget(self.lab)
        lay.addWidget(self.value)
        lay.addWidget(self.sub)
        lay.addStretch(1)
        self.setMinimumWidth(120)

    def set(self, text: str, status: str = "info", sub: str = "", tip: str = "") -> None:
        self.value.setText(text)
        col = theme.status_color(status) if status in ("in", "out", "fail", "ok", "warn") \
            else theme.current().text
        self.value.setStyleSheet(f"color: {col};")
        self.sub.setText(sub)
        self.setToolTip(tip)

    def set_kpi(self, k: KPI | None, sub: str = "", stale: bool = False) -> None:
        """``stale``: the value belongs to the previous geometry -- shown muted and
        italic, not coloured against its band."""
        if k is None:
            self.set("—", "info", sub)
            return
        if stale and k.ok:
            self.set(k.text(), "info", (sub or k.band_text()) + "  (stale)", kpi_tip(k, True))
            self.value.setStyleSheet(f"color: {theme.current().text_muted}; font-style: italic;")
            return
        self.set(k.text(), k.status, sub or k.band_text(), kpi_tip(k))


class SimpleTable(QtWidgets.QTableWidget):
    def __init__(self, headers: Sequence[str]) -> None:
        super().__init__()
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(list(headers))
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        # the last column takes what is left and elides: long source / meaning text
        # used to wrap and turn every row into a paragraph (full text in the tooltip)
        hh.setSectionResizeMode(len(headers) - 1, QtWidgets.QHeaderView.Stretch)
        self.setWordWrap(False)
        self.setTextElideMode(QtCore.Qt.ElideRight)

    def set_rows(self, rows: Sequence[Sequence[Any]], status_col: int | None = None,
                 right_cols: Iterable[int] = ()) -> None:
        right = set(right_cols)
        self.setRowCount(len(rows))
        for i, r in enumerate(rows):
            st = str(r[status_col]) if status_col is not None else None
            for j, v in enumerate(r):
                txt = v if isinstance(v, str) else (
                    "—" if isinstance(v, float) and not math.isfinite(v) else
                    (f"{v:.4g}" if isinstance(v, float) else str(v)))
                it = _item(txt, st if j == status_col else None, j in right)
                if len(txt) > 40:
                    it.setToolTip(txt)
                self.setItem(i, j, it)
        self.resizeRowsToContents()


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot(title: str = "", x: str = "", y: str = "", legend: bool = True) -> pg.PlotWidget:
    pw = pg.PlotWidget()
    pi = pw.getPlotItem()
    if title:
        pi.setTitle(title, size="10pt")
    pi.setLabel("bottom", x)
    pi.setLabel("left", y)
    pi.showGrid(x=True, y=True, alpha=0.25)
    for ax in ("left", "bottom"):
        pi.getAxis(ax).enableAutoSIPrefix(False)
    if legend:
        pi.addLegend(offset=(8, 8), labelTextSize="8pt")
    pw.setMinimumHeight(220)
    return pw


def restyle_plots(root: QtWidgets.QWidget) -> int:
    """Re-apply the current theme to every plot already built under ``root``.

    ``pg.setConfigOptions`` only reaches plots created afterwards, so a theme
    toggle left the existing ones on the old background. The series are
    redrawn (with the new palette) by the recompute that follows the toggle.
    Returns the number of plots restyled.
    """
    t = theme.current()
    fg = QtGui.QColor(t.plot_fg)
    n = 0
    for pw in root.findChildren(pg.PlotWidget):
        pw.setBackground(t.plot_bg)
        pi = pw.getPlotItem()
        for name in ("left", "bottom", "right", "top"):
            ax = pi.getAxis(name)
            ax.setPen(fg)
            ax.setTextPen(fg)
            if ax.label is not None:
                ax.label.setDefaultTextColor(fg)
        if pi.titleLabel is not None and pi.titleLabel.text:
            pi.setTitle(pi.titleLabel.text, size="10pt", color=t.plot_fg)
        if pi.legend is not None:
            pi.legend.setLabelTextColor(fg)
        n += 1
    try:
        import pyqtgraph.opengl as gl

        for gv in root.findChildren(gl.GLViewWidget):
            gv.setBackgroundColor(t.plot_bg)
            n += 1
    except ImportError:          # no OpenGL: the 3D view is a notice
        pass
    return n


def pen(i: int, width: float = 2.0, style: QtCore.Qt.PenStyle = QtCore.Qt.SolidLine) -> QtGui.QPen:
    p = pg.mkPen(theme.current().series[i % len(theme.current().series)], width=width)
    p.setStyle(style)
    p.setCosmetic(True)
    return p


def band_region(pw: pg.PlotWidget, lo: float, hi: float, orientation: str = "horizontal"
                ) -> None:
    """Shade a design-intent band (information, not a target to chase)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return
    c = QtGui.QColor(theme.current().ok)
    c.setAlpha(28)
    reg = pg.LinearRegionItem(values=(lo, hi), orientation=orientation, movable=False,
                              brush=pg.mkBrush(c), pen=pg.mkPen(None))
    reg.setZValue(-10)
    pw.addItem(reg)


def save_text(parent: QtWidgets.QWidget, text: str, default: str, filt: str) -> str | None:
    path, _ = QtWidgets.QFileDialog.getSaveFileName(parent, "Save", default, filt)
    if not path:
        return None
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return path
