"""Colours, palette and stylesheet. One place, two themes (dark default, light).

Status colours carry meaning everywhere in the app and are never decorative:
inside band, outside band, failed solve, information only.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    panel: str
    panel_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    accent_text: str
    ok: str
    warn: str
    bad: str
    info: str
    plot_bg: str
    plot_fg: str
    grid: str
    series: tuple[str, ...]


DARK = Theme(
    name="dark", bg="#15181d", panel="#1d2128", panel_alt="#242a33", border="#323a46",
    text="#e6e9ee", text_muted="#9aa4b2", accent="#3b82f6", accent_text="#ffffff",
    ok="#2f9e6e", warn="#d39b2a", bad="#d9534f", info="#5b6676",
    plot_bg="#15181d", plot_fg="#c8ced8", grid="#2c333d",
    series=("#4c9aff", "#ff8a4c", "#47c28b", "#e5c04a", "#b38cff", "#ff6b8a", "#4cd3e0",
            "#a3a3a3"),
)
LIGHT = Theme(
    name="light", bg="#f4f6f9", panel="#ffffff", panel_alt="#eef1f5", border="#d4dae3",
    text="#1b2330", text_muted="#5d6878", accent="#2563eb", accent_text="#ffffff",
    ok="#1f8a5b", warn="#b7791f", bad="#c53030", info="#7a8494",
    plot_bg="#ffffff", plot_fg="#2a3442", grid="#e3e7ee",
    series=("#2563eb", "#ea580c", "#16a34a", "#ca8a04", "#7c3aed", "#db2777", "#0891b2",
            "#6b7280"),
)

_current: Theme = DARK


def current() -> Theme:
    return _current


def status_color(status: str) -> str:
    t = _current
    return {"in": t.ok, "ok": t.ok, "out": t.warn, "warn": t.warn, "fail": t.bad,
            "info": t.info, "n/a": t.info}.get(status, t.info)


def status_text(status: str) -> str:
    return {"in": "inside band", "out": "outside band", "fail": "failed", "info": "—",
            "ok": "ok", "warn": "check", "n/a": "n/a"}.get(status, status)


def apply(app: QtWidgets.QApplication, theme: Theme) -> None:
    global _current
    _current = theme
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    c = QtGui.QColor
    pal.setColor(QtGui.QPalette.Window, c(theme.bg))
    pal.setColor(QtGui.QPalette.WindowText, c(theme.text))
    pal.setColor(QtGui.QPalette.Base, c(theme.panel))
    pal.setColor(QtGui.QPalette.AlternateBase, c(theme.panel_alt))
    pal.setColor(QtGui.QPalette.Text, c(theme.text))
    pal.setColor(QtGui.QPalette.Button, c(theme.panel_alt))
    pal.setColor(QtGui.QPalette.ButtonText, c(theme.text))
    pal.setColor(QtGui.QPalette.Highlight, c(theme.accent))
    pal.setColor(QtGui.QPalette.HighlightedText, c(theme.accent_text))
    pal.setColor(QtGui.QPalette.ToolTipBase, c(theme.panel_alt))
    pal.setColor(QtGui.QPalette.ToolTipText, c(theme.text))
    pal.setColor(QtGui.QPalette.PlaceholderText, c(theme.text_muted))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, c(theme.text_muted))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, c(theme.text_muted))
    app.setPalette(pal)
    app.setStyleSheet(stylesheet(theme))
    pg.setConfigOptions(antialias=True, background=theme.plot_bg, foreground=theme.plot_fg)


def stylesheet(t: Theme) -> str:
    return f"""
    QWidget {{ font-size: 10pt; }}
    QMainWindow, QDialog {{ background: {t.bg}; }}
    QListWidget#nav {{ background: {t.panel}; border: none; border-right: 1px solid {t.border};
        padding: 6px 0; outline: none; }}
    QListWidget#nav::item {{ padding: 7px 14px; border-radius: 6px; margin: 1px 6px; }}
    QListWidget#nav::item:selected {{ background: {t.accent}; color: {t.accent_text}; }}
    QListWidget#nav::item:hover:!selected {{ background: {t.panel_alt}; }}
    QGroupBox {{ border: 1px solid {t.border}; border-radius: 8px; margin-top: 14px;
        padding: 10px 8px 8px 8px; background: {t.panel}; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
        color: {t.text_muted}; font-weight: 600; }}
    QLabel#pageTitle {{ font-size: 17pt; font-weight: 600; }}
    QLabel#pageSubtitle {{ color: {t.text_muted}; }}
    QLabel#muted {{ color: {t.text_muted}; }}
    QLabel#cardValue {{ font-size: 16pt; font-weight: 600; }}
    QLabel#cardLabel {{ color: {t.text_muted}; font-size: 9pt; }}
    QFrame#card {{ background: {t.panel}; border: 1px solid {t.border}; border-radius: 10px; }}
    QFrame#banner {{ background: {t.panel_alt}; border: 1px solid {t.border};
        border-radius: 8px; }}
    QTableWidget, QTableView, QTreeWidget, QPlainTextEdit, QTextEdit, QTextBrowser {{
        background: {t.panel}; border: 1px solid {t.border}; border-radius: 6px;
        gridline-color: {t.border}; }}
    QHeaderView::section {{ background: {t.panel_alt}; color: {t.text_muted}; border: none;
        border-right: 1px solid {t.border}; border-bottom: 1px solid {t.border}; padding: 4px 6px;
        font-weight: 600; }}
    QTabWidget::pane {{ border: 1px solid {t.border}; border-radius: 6px; top: -1px;
        background: {t.panel}; }}
    QTabBar::tab {{ background: transparent; padding: 6px 14px; color: {t.text_muted};
        border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ color: {t.text}; border-bottom: 2px solid {t.accent}; }}
    QPushButton {{ background: {t.panel_alt}; border: 1px solid {t.border}; border-radius: 6px;
        padding: 5px 12px; }}
    QPushButton:hover {{ border-color: {t.accent}; }}
    QPushButton#primary {{ background: {t.accent}; color: {t.accent_text}; border: none; }}
    QPushButton:disabled {{ color: {t.text_muted}; }}
    QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {{ background: {t.panel_alt};
        border: 1px solid {t.border}; border-radius: 5px; padding: 2px 4px; min-height: 20px; }}
    QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QComboBox:focus {{
        border-color: {t.accent}; }}
    QToolBar {{ background: {t.panel}; border-bottom: 1px solid {t.border}; spacing: 4px;
        padding: 3px; }}
    QStatusBar {{ background: {t.panel}; border-top: 1px solid {t.border}; }}
    QScrollArea {{ border: none; background: transparent; }}
    QSplitter::handle {{ background: {t.border}; }}
    QProgressBar {{ border: 1px solid {t.border}; border-radius: 4px; text-align: center;
        background: {t.panel_alt}; max-height: 14px; }}
    QProgressBar::chunk {{ background: {t.accent}; border-radius: 3px; }}
    """
