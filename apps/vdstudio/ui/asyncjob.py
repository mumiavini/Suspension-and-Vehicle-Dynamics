"""Debounced, cancel-previous background jobs for pages that draw curves."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6 import QtCore

from .controller import Task, run_task


class Debounced(QtCore.QObject):
    """``request(fn, on_done)``: run ``fn(progress, cancelled)`` after a quiet period.

    A newer request cancels the older job; only the newest result is delivered.
    """

    busy = QtCore.Signal(bool)
    progress = QtCore.Signal(float, str)
    failed = QtCore.Signal(str)

    def __init__(self, delay_ms: int = 250, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self._fire)
        self._pending: tuple[Callable[..., Any], Callable[[Any], None]] | None = None
        self._task: Task | None = None
        self._gen = 0

    def request(self, fn: Callable[..., Any], on_done: Callable[[Any], None],
                immediate: bool = False) -> None:
        self._pending = (fn, on_done)
        if immediate:
            self._timer.stop()
            self._fire()
        else:
            self._timer.start()

    def cancel(self) -> None:
        self._timer.stop()
        self._pending = None
        if self._task is not None:
            self._task.cancel()
            self._task = None
            self.busy.emit(False)

    def _fire(self) -> None:
        if self._pending is None:
            return
        fn, on_done = self._pending
        self._pending = None
        if self._task is not None:
            self._task.cancel()
        self._gen += 1
        gen = self._gen

        def done(res: Any) -> None:
            if gen == self._gen:
                self._task = None
                self.busy.emit(False)
                on_done(res)

        def err(msg: str) -> None:
            if gen == self._gen:
                self._task = None
                self.busy.emit(False)
                self.failed.emit(msg)

        self.busy.emit(True)
        self._task = run_task(fn, done, err,
                              lambda f, m: self.progress.emit(f, m) if gen == self._gen else None)
