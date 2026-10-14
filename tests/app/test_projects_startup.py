"""Project file on start (argv > last > team file > built-in), recent list, save prompt."""

from __future__ import annotations

import os
import time

import pytest

from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import recent as R
from apps.vdstudio.engine.project import Project, load_project, project_to_dict, save_project

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VDSTUDIO_NO_GL", "1")


def test_push_recent_puts_the_file_first_without_duplicates(tmp_path) -> None:
    a, b, c = (str(tmp_path / n) for n in ("a.vdproj.json", "b.vdproj.json", "c.vdproj.json"))
    r = R.push_recent([], a)
    r = R.push_recent(r, b)
    r = R.push_recent(r, os.path.join(str(tmp_path), ".", "a.vdproj.json"))   # same file
    assert r == [a, b]
    r = R.push_recent([str(tmp_path / f"{i}.json") for i in range(10)], c, limit=8)
    assert len(r) == 8 and r[0] == c
    assert R.drop_recent([a, b, c], b) == [a, c]


def test_as_list_reads_back_any_qsettings_shape() -> None:
    assert R.as_list(None) == [] and R.as_list("") == []
    assert R.as_list("C:/x.vdproj.json") == ["C:/x.vdproj.json"]       # one entry -> str
    assert R.as_list(["a", "", "b"]) == ["a", "b"] and R.as_list(5) == []


def test_startup_order(tmp_path) -> None:
    given, last, team = (tmp_path / n for n in ("given.json", "last.json", "team.json"))
    for p in (given, last, team):
        p.write_text("{}")
    assert R.startup_path(str(given), str(last), team) == (given, "given on the command line")
    assert R.startup_path(None, str(last), team) == (last, "last project")
    assert R.startup_path(str(tmp_path / "gone.json"), str(last), team)[0] == last
    assert R.startup_path(None, str(tmp_path / "gone.json"), team) == (team, "team project")
    path, why = R.startup_path(None, None, tmp_path / "no_team.json")
    assert path is None and "sla_geometry.py" in why


def test_dialog_dir(tmp_path) -> None:
    assert R.dialog_dir(tmp_path / "car.vdproj.json") == tmp_path
    assert R.dialog_dir(None) == R.PROJECTS_DIR


def test_team_project_ships_and_resolves() -> None:
    assert R.TEAM_PROJECT.is_file()
    p = load_project(R.TEAM_PROJECT)
    assert G.resolve(p).wheelbase_mm > 0


def test_save_is_atomic_and_round_trips(tmp_path) -> None:
    path = tmp_path / "car.vdproj.json"
    path.write_text("old", encoding="utf-8")
    save_project(Project(), path)
    assert not (tmp_path / "car.vdproj.json.tmp").exists()
    assert project_to_dict(load_project(path)) == project_to_dict(Project())
    assert path.read_text(encoding="utf-8").endswith("}\n")


# -- the window ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def win(tmp_path_factory):
    pytest.importorskip("PySide6")
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from apps.vdstudio.ui.controller import ProjectController
    from apps.vdstudio.ui.main_window import MainWindow

    ini = tmp_path_factory.mktemp("settings") / "vdstudio.ini"
    settings = QtCore.QSettings(str(ini), QtCore.QSettings.IniFormat)
    ctrl = ProjectController()
    w = MainWindow(ctrl, settings=settings)
    yield w
    t = time.time()
    while ctrl.dynamic_busy and time.time() - t < 30:
        app.processEvents()
        time.sleep(0.01)
    for p in w.pages.values():
        if hasattr(p, "shutdown"):
            p.shutdown()
    ctrl.shutdown()
    ctrl._set_dirty(False)
    w.close()


@pytest.fixture()
def boxes(monkeypatch):
    """Record message boxes instead of showing them; ``answer`` sets the question reply."""
    from PySide6 import QtWidgets

    seen: dict[str, list[str]] = {"question": [], "warning": [], "critical": []}
    state = {"answer": QtWidgets.QMessageBox.Cancel}

    def make(kind):
        def f(parent, title, text, *a, **k):
            seen[kind].append(text)
            return state["answer"] if kind == "question" else QtWidgets.QMessageBox.Ok
        return staticmethod(f)

    for kind in seen:
        monkeypatch.setattr(QtWidgets.QMessageBox, kind, make(kind))
    return seen, state


def test_save_remembers_and_the_next_start_reopens_it(win, boxes, tmp_path, monkeypatch) -> None:
    from PySide6 import QtWidgets

    target = tmp_path / "my car"          # no suffix typed: it is added
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    win.ctrl.new()
    assert win.save_project(save_as=True)
    saved = tmp_path / "my car.vdproj.json"
    assert saved.is_file() and win.ctrl.path == saved
    assert win.settings.value("last_project") == str(saved.resolve())
    assert win._recent()[0] == str(saved.resolve())
    labels = [a.text() for a in win.recent_menu.actions()]
    assert labels[0] == "&1  my car.vdproj.json" and "Clear list" in labels

    win.ctrl.new()                                    # as on a fresh start
    assert win.open_startup_project(None) == saved.resolve()
    assert win.ctrl.path == saved.resolve()
    other = tmp_path / "other.vdproj.json"
    save_project(Project(), other)
    assert win.open_startup_project(str(other)) == other   # the given file wins


def test_unsaved_changes_ask_to_save(win, boxes, tmp_path) -> None:
    from PySide6 import QtWidgets

    seen, state = boxes
    path = tmp_path / "dirty.vdproj.json"
    save_project(Project(), path)
    assert win.open_project(path)
    win.ctrl.project.notes = "changed"
    win.ctrl._set_dirty(True)
    state["answer"] = QtWidgets.QMessageBox.Cancel
    assert not win._confirm_discard() and win.ctrl.dirty
    assert "Save the changes to dirty.vdproj.json?" in seen["question"][-1]
    state["answer"] = QtWidgets.QMessageBox.Save
    assert win._confirm_discard() and not win.ctrl.dirty
    assert load_project(path).notes == "changed"
    win.ctrl._set_dirty(True)
    state["answer"] = QtWidgets.QMessageBox.Discard
    assert win._confirm_discard()
    win.ctrl._set_dirty(False)


def test_missing_recent_file_is_dropped(win, boxes, tmp_path) -> None:
    seen, _ = boxes
    gone = tmp_path / "gone.vdproj.json"
    win._set_recent([str(gone)])
    assert not win.open_project(gone)
    assert "no longer exists" in seen["warning"][-1] and win._recent() == []
    assert not win.recent_menu.actions()[0].isEnabled()      # "No recent projects"


def test_unreadable_last_project_falls_back_and_is_forgotten(win, boxes, tmp_path,
                                                             monkeypatch) -> None:
    seen, _ = boxes
    bad = tmp_path / "bad.vdproj.json"
    bad.write_text("{ not json", encoding="utf-8")
    win.settings.setValue("last_project", str(bad))
    monkeypatch.setattr(R, "TEAM_PROJECT", tmp_path / "no_team.vdproj.json")
    win.ctrl.new()
    assert win.open_startup_project(None) is None
    assert "Could not open" in seen["critical"][-1]
    assert win.settings.value("last_project", None) is None
    assert win.ctrl.path is None                                  # the built-in car
    assert isinstance(win.ctrl.project, Project)
