"""Which project file the app opens on start, and the recent-projects list.

Qt-free: the window stores these lists in QSettings, this module only decides.

On start the app opens, in order:

1. the file given on the command line (or dropped onto the shortcut);
2. the last project opened or saved, if the file still exists;
3. the team project shipped in the repository, ``projects/FSAE_2027.vdproj.json``;
4. otherwise the built-in 2027 car, read from ``sla_geometry.py`` /
   ``steering_geometry.py`` (the same as the "New" button).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from .._paths import REPO
from .project import PROJECT_SUFFIX

PROJECTS_DIR = REPO / "projects"
TEAM_PROJECT = PROJECTS_DIR / f"FSAE_2027{PROJECT_SUFFIX}"
MAX_RECENT = 8


def _key(p: str | os.PathLike[str]) -> str:
    """Comparison key: absolute, normalised, case-folded where the filesystem folds case."""
    return os.path.normcase(os.path.abspath(os.fspath(p)))


def as_list(value: object) -> list[str]:
    """A QSettings value read back as a list of strings (one entry comes back as a str)."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(v) for v in value if v]
    return []


def push_recent(recent: Iterable[str], path: str | os.PathLike[str],
                limit: int = MAX_RECENT) -> list[str]:
    """``path`` first, without duplicates, at most ``limit`` entries."""
    first = os.path.abspath(os.fspath(path))
    out = [first]
    seen = {_key(first)}
    for p in recent:
        k = _key(p)
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out[:limit]


def drop_recent(recent: Iterable[str], path: str | os.PathLike[str]) -> list[str]:
    k = _key(path)
    return [p for p in recent if _key(p) != k]


def startup_path(argv_path: str | None, last: str | None,
                 team: Path = TEAM_PROJECT) -> tuple[Path | None, str]:
    """The file to open on start and why, or ``(None, reason)`` for the built-in car."""
    if argv_path and os.path.isfile(argv_path):
        return Path(argv_path), "given on the command line"
    if last and os.path.isfile(last):
        return Path(last), "last project"
    if team.is_file():
        return team, "team project"
    return None, "built-in 2027 car from sla_geometry.py / steering_geometry.py"


def dialog_dir(current: Path | None) -> Path:
    """Where the Open / Save-as dialogs start: next to the current file, else projects/."""
    if current is not None and current.parent.is_dir():
        return current.parent
    return PROJECTS_DIR if PROJECTS_DIR.is_dir() else REPO
