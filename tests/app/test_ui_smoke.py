"""VD Studio UI smoke test: build the window, visit every page, edit, undo, save.

Runs headless (``QT_QPA_PLATFORM=offscreen``) without OpenGL. Skipped when the
desktop extra (PySide6 / pyqtgraph) is not installed.
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VDSTUDIO_NO_GL", "1")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyqtgraph")

from apps.vdstudio.ui import theme  # noqa: E402
from apps.vdstudio.ui.controller import ProjectController  # noqa: E402
from apps.vdstudio.ui.main_window import NAV, MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(a, theme.DARK)
    return a


def _pump(app, seconds: float, until=None) -> None:
    t = time.time()
    while time.time() - t < seconds:
        app.processEvents()
        time.sleep(0.01)
        if until is not None and until():
            return


@pytest.fixture(scope="module")
def window(app):
    ctrl = ProjectController()
    win = MainWindow(ctrl)
    win.show()
    _pump(app, 30, lambda: not ctrl.dynamic_stale)
    yield win
    for p in win.pages.values():
        if hasattr(p, "shutdown"):
            p.shutdown()
    ctrl.shutdown()
    win.ctrl._set_dirty(False)
    win.close()


def test_every_page_builds_and_shows(app, window) -> None:
    assert set(window.pages) == {key for _, key, *_ in NAV}
    for key in window.pages:
        window.goto(key)
        _pump(app, 0.2)
        assert window.stack.currentWidget() is window.pages[key]


def test_swept_kpis_arrive(window) -> None:
    ks = window.ctrl.kpis
    assert not window.ctrl.dynamic_stale
    assert ks.get("front.camber_gain") is not None and ks.get("front.camber_gain").ok
    assert ks.get("steering.ackermann").ok


def test_edit_is_real_time_and_undoable(app, window) -> None:
    ctrl = window.ctrl
    before = ctrl.static.value("front.rc_static")
    ctrl.set_value("front.rc_height_mm", 45.0)
    # 48.27, not 45.5, since the 2026-10-12 instant-axis RC (see test_engine_kpis)
    assert ctrl.static.value("front.rc_static") == pytest.approx(48.27, abs=0.1)
    assert ctrl.dirty
    ctrl.undo()
    assert ctrl.static.value("front.rc_static") == pytest.approx(before)
    ctrl.redo()
    assert ctrl.project.front.rc_height_mm == 45.0
    ctrl.undo()


def test_geometry_failure_is_shown_not_hidden(app, window) -> None:
    ctrl = window.ctrl
    # FVIC placed exactly above the lower ball joint: the wishbone line is vertical
    # and never reaches the pickup plane -- synthesis must refuse, not draw a car.
    ctrl.set_value("front.fvsa_length_mm", 38.0)
    _pump(app, 0.2)
    assert ctrl.geometry is None
    assert "non-finite" in ctrl.geometry_error
    assert len(ctrl.kpis.items) == 0
    ctrl.undo()
    assert ctrl.geometry is not None


def test_save_and_reopen(app, window, tmp_path) -> None:
    path = tmp_path / "smoke.vdproj.json"
    window.ctrl.save(path)
    assert not window.ctrl.dirty
    window.ctrl.open(path)
    assert window.ctrl.path == path
    assert window.ctrl.geometry is not None
