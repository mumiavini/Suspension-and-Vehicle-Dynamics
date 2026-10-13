"""The single source of state for the UI, and the background-work plumbing.

Every page edits the project through :meth:`ProjectController.set_value` (or
:meth:`apply`), never by mutating it. The controller then:

1. re-resolves the geometry synchronously (synthesis is analytic, ~3 ms) and
   recomputes the static KPIs (~10 ms) -- this is what makes the editors
   real-time;
2. after a short debounce, recomputes the swept KPIs (~2 s) on a worker
   thread. A newer edit supersedes an older computation: stale results are
   dropped by generation number and the old job is asked to stop. Until the
   new values land, the previous ones stay in :attr:`kpis` and are named in
   ``kpis.stale_keys`` (the widgets grey them). If the computation fails, the
   previous values are replaced by failed entries -- never left on screen as
   if they described the current car.

Undo/redo keeps whole-project snapshots (they are small).
"""

from __future__ import annotations

import copy
import math
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PySide6 import QtCore

from ..engine import geometry as G
from ..engine import kpis as K
from ..engine.project import Project, load_project, save_project, with_path


class _Signals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(float, str)
    finished = QtCore.Signal()


# Tasks stay referenced until their "finished" signal has been handled on the
# GUI thread. Dropping them earlier could delete the signal object while its
# queued "done" event is still pending, and the result would be lost.
_LIVE: set[Task] = set()


class Task(QtCore.QRunnable):
    """Run ``fn(progress, cancelled)`` on the thread pool; report back on the GUI thread."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = _Signals()
        self._cancel = threading.Event()
        self.setAutoDelete(False)

    def cancel(self) -> None:
        self._cancel.set()

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _emit(self, sig: str, *args: Any) -> None:
        # At application exit the signal object can be gone before the worker
        # finishes; there is nobody left to tell, so stop quietly.
        try:
            getattr(self.signals, sig).emit(*args)
        except RuntimeError:
            self._cancel.set()

    def _progress(self, f: float, m: str = "") -> None:
        if self._cancel.is_set():
            raise InterruptedError
        self._emit("progress", float(f), str(m))

    def run(self) -> None:  # noqa: D401
        try:
            res = self.fn(self._progress, self.cancelled)
        except InterruptedError:
            pass
        except Exception as exc:  # surfaced to the user, never swallowed
            if not self.cancelled():
                self._emit("failed", f"{type(exc).__name__}: {exc}\n\n"
                           + traceback.format_exc(limit=6))
        else:
            if not self.cancelled():
                self._emit("done", res)
        finally:
            self._emit("finished")


def run_task(fn: Callable[..., Any], on_done: Callable[[Any], None],
             on_error: Callable[[str], None] | None = None,
             on_progress: Callable[[float, str], None] | None = None) -> Task:
    t = Task(fn)
    t.signals.done.connect(on_done)
    if on_error is not None:
        t.signals.failed.connect(on_error)
    if on_progress is not None:
        t.signals.progress.connect(on_progress)
    _LIVE.add(t)
    t.signals.finished.connect(lambda: _LIVE.discard(t))
    QtCore.QThreadPool.globalInstance().start(t)
    return t


@dataclass
class LiveKpiSet(K.KpiSet):
    """The controller's KPI set: static + latest swept KPIs.

    ``stale_keys`` names the swept values left over from the previous geometry
    while the new ones are computed (empty once they land).
    """

    stale_keys: frozenset[str] = frozenset()


class ProjectController(QtCore.QObject):
    projectChanged = QtCore.Signal()            # any input changed (incl. load/new)
    projectReplaced = QtCore.Signal()           # new/open: editors must rebuild
    geometryChanged = QtCore.Signal(object)     # CarGeometry
    geometryFailed = QtCore.Signal(str)
    staticReady = QtCore.Signal(object)         # KpiSet
    dynamicReady = QtCore.Signal(object)        # KpiSet (static + dynamic)
    dynamicBusy = QtCore.Signal(bool)
    dynamicProgress = QtCore.Signal(float, str)
    dynamicStale = QtCore.Signal()              # swept values are from the previous geometry
    dynamicFailed = QtCore.Signal(str)          # swept KPIs failed; kpis hold failed entries
    dirtyChanged = QtCore.Signal(bool)
    message = QtCore.Signal(str)

    def __init__(self, project: Project | None = None) -> None:
        super().__init__()
        self.project: Project = project or Project()
        self.path: Path | None = None
        self.dirty = False
        self.geometry: G.CarGeometry | None = None
        self.geometry_error = ""
        self.static: K.KpiSet = K.KpiSet()
        self.kpis: K.KpiSet = LiveKpiSet()     # static + latest dynamic
        self.dynamic_stale = True
        self.dynamic_busy = False
        self.dynamic_error = ""                # why the last swept computation failed
        self._history: list[Project] = []
        self._future: list[Project] = []
        self._gen = 0
        self._task: Task | None = None
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(450)
        self._debounce.timeout.connect(self._start_dynamic)

    # -- editing -----------------------------------------------------------------
    def set_value(self, path: str, value: Any, *, record: bool = True) -> None:
        try:
            new = with_path(self.project, path, value)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.message.emit(f"Could not set {path}: {exc}")
            return
        self._commit(new, record)

    def apply(self, fn: Callable[[Project], Project | None], *, record: bool = True,
              geometry: bool = True) -> None:
        """Edit with a function. It may mutate a copy in place or return a new project.

        ``geometry=False`` for edits that cannot change the car (log entries,
        telemetry settings, stage notes): no re-solve, no swept-KPI job.
        """
        p = self.project.copy()
        out = fn(p)
        self._commit(out if out is not None else p, record, geometry)

    def _commit(self, new: Project, record: bool, geometry: bool = True) -> None:
        if record:
            self._history.append(self.project)
            self._history = self._history[-200:]
            self._future.clear()
        self.project = new
        self._set_dirty(True)
        if geometry:
            self.recompute()
        self.projectChanged.emit()

    def undo(self) -> None:
        if not self._history:
            return
        self._future.append(self.project)
        self.project = self._history.pop()
        self._set_dirty(True)
        self.recompute()
        self.projectReplaced.emit()
        self.projectChanged.emit()

    def redo(self) -> None:
        if not self._future:
            return
        self._history.append(self.project)
        self.project = self._future.pop()
        self._set_dirty(True)
        self.recompute()
        self.projectReplaced.emit()
        self.projectChanged.emit()

    def can_undo(self) -> bool:
        return bool(self._history)

    def can_redo(self) -> bool:
        return bool(self._future)

    # -- files ---------------------------------------------------------------------
    def new(self, project: Project | None = None) -> None:
        self.project = project or Project()
        self.path = None
        self._history.clear()
        self._future.clear()
        self._set_dirty(False)
        self.recompute()
        self.projectReplaced.emit()
        self.projectChanged.emit()

    def open(self, path: str | Path) -> None:
        p = load_project(Path(path))
        self.new(p)
        self.path = Path(path)
        self._set_dirty(False)
        self.projectChanged.emit()

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no file name")
        save_project(self.project, target)
        self.path = target
        self._set_dirty(False)
        self.projectChanged.emit()
        return target

    def _set_dirty(self, d: bool) -> None:
        if d != self.dirty:
            self.dirty = d
            self.dirtyChanged.emit(d)

    # -- computation -------------------------------------------------------------------
    def recompute(self) -> None:
        try:
            geo = G.resolve(self.project)
            st = K.compute_static(geo)
        except Exception as exc:  # GeometryError, pydantic ValidationError, ...
            self.geometry = None
            self.geometry_error = str(exc)
            self.static = K.KpiSet()
            self.kpis = LiveKpiSet()
            self._debounce.stop()
            self._cancel_dynamic()
            self._gen += 1                      # a job still running reports to nobody
            self.dynamic_error = ""
            if self.dynamic_busy:
                self.dynamic_busy = False
                self.dynamicBusy.emit(False)
            self.geometryFailed.emit(self.geometry_error)
            return
        self.geometry = geo
        self.geometry_error = ""
        self.static = st
        # a job still running computed the previous geometry: drop its result (the
        # debounced restart below cancels it)
        self._gen += 1
        # keep the previous dynamic values visible, marked stale, until the new ones land
        stale = frozenset(k for k in self.kpis.items if k not in st.items)
        merged = LiveKpiSet(dict(self.kpis.items), stale)
        merged.items.update(st.items)
        self.kpis = merged
        self.dynamic_stale = True
        self.dynamic_error = ""
        self.geometryChanged.emit(geo)
        self.dynamicStale.emit()
        self.staticReady.emit(st)
        self._debounce.start()

    def _cancel_dynamic(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def _start_dynamic(self) -> None:
        geo, st = self.geometry, self.static
        if geo is None:
            return
        self._cancel_dynamic()
        self._gen += 1
        gen = self._gen

        def work(progress, cancelled):
            return K.compute_dynamic(geo, None, progress, cancelled, static=st)

        def done(dyn: K.KpiSet) -> None:
            if gen != self._gen:
                return
            self.kpis = LiveKpiSet(st.merge(dyn).items)
            self.dynamic_stale = False
            self.dynamic_error = ""
            self.dynamic_busy = False
            self._task = None
            self.dynamicBusy.emit(False)
            self.dynamicReady.emit(self.kpis)

        def failed(msg: str) -> None:
            if gen != self._gen:
                return
            first = (msg.splitlines() or ["unknown error"])[0]
            err = "swept KPIs failed: " + first
            # The previous geometry's swept values must not stay on screen as if they
            # described this car: each becomes a failed entry (NaN + the reason).
            items = dict(self.kpis.items)
            for key, k in self.kpis.items.items():
                if key not in st.items:
                    items[key] = replace(k, value=math.nan, error=err)
            self.kpis = LiveKpiSet(items)
            self.dynamic_stale = False
            self.dynamic_error = err
            self.dynamic_busy = False
            self._task = None
            self.dynamicBusy.emit(False)
            self.dynamicFailed.emit(err)
            self.dynamicReady.emit(self.kpis)
            self.message.emit("Swept KPIs failed: " + first)

        self.dynamic_busy = True
        self.dynamicBusy.emit(True)
        self._task = run_task(work, done, failed,
                              lambda f, m: self.dynamicProgress.emit(f, m)
                              if gen == self._gen else None)

    def shutdown(self, wait_ms: int = 3000) -> None:
        """Stop background work before the application exits."""
        self._debounce.stop()
        self._cancel_dynamic()
        for t in list(_LIVE):
            t.cancel()
        QtCore.QThreadPool.globalInstance().waitForDone(wait_ms)

    def snapshot(self) -> Project:
        return copy.deepcopy(self.project)
