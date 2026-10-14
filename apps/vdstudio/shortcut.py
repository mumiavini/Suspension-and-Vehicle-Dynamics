"""Create (or remove) VD Studio shortcuts on the Desktop and in the app menu.

    python -m apps.vdstudio.shortcut            # create
    python -m apps.vdstudio.shortcut --remove   # remove

Run it with the Python of the environment that has the desktop extra (the
Windows installer, ``Install VD Studio shortcut.bat``, does this for you). The
shortcuts start that same interpreter directly, with no console window and no
``uv`` on the way, in the repository folder:

* Windows: ``VD Studio.lnk`` on the Desktop and in the Start menu, target
  ``<venv>\\Scripts\\pythonw.exe -m apps.vdstudio``. Created through
  PowerShell's ``WScript.Shell`` with every path passed in environment
  variables, so folders with spaces or quotes need no escaping.
* Linux: ``vdstudio.desktop`` in ``~/.local/share/applications`` and on the
  XDG Desktop.
* macOS: ``VD Studio.command`` on the Desktop (it opens a Terminal window).

The shortcuts point at this folder: run it again after moving the repository.
Nothing here imports Qt.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from apps.vdstudio import APP_NAME
from apps.vdstudio._paths import REPO

RESOURCES = REPO / "apps" / "vdstudio" / "resources"
ICON_ICO = RESOURCES / "vdstudio.ico"
ICON_PNG = RESOURCES / "vdstudio.png"
DESCRIPTION = "VD Studio - PUCPR Racing suspension and steering workbench"
MODULE_ARGS = ("-m", "apps.vdstudio")

# Reads everything from the environment; prints one created/removed path per line.
_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$dirs = @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))
$ws = New-Object -ComObject WScript.Shell
foreach ($dir in $dirs) {
    if (-not $dir) { continue }
    $path = Join-Path $dir ($env:VDS_NAME + '.lnk')
    if ($env:VDS_REMOVE -eq '1') {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path; Write-Output $path }
        continue
    }
    $lnk = $ws.CreateShortcut($path)
    $lnk.TargetPath = $env:VDS_TARGET
    $lnk.Arguments = $env:VDS_ARGS
    $lnk.WorkingDirectory = $env:VDS_WORKDIR
    $lnk.IconLocation = $env:VDS_ICON + ',0'
    $lnk.Description = $env:VDS_DESCRIPTION
    $lnk.Save()
    Write-Output $path
}
"""


class ShortcutError(RuntimeError):
    """The shortcut could not be created; the message says what to do."""


def gui_python(executable: str | os.PathLike[str] | None = None) -> Path:
    """The windowless interpreter next to ``executable`` (``pythonw.exe`` on Windows)."""
    exe = Path(executable or sys.executable)
    if os.name == "nt":
        w = exe.with_name("pythonw.exe")
        if not w.is_file():
            raise ShortcutError(f"{w} does not exist. Re-create the environment: "
                                "uv sync --extra desktop --inexact")
        return w
    return exe


def windows_env(target: Path, *, remove: bool = False) -> dict[str, str]:
    """The environment variables the PowerShell script reads."""
    return {
        "VDS_NAME": APP_NAME,
        "VDS_TARGET": str(target),
        "VDS_ARGS": " ".join(MODULE_ARGS),
        "VDS_WORKDIR": str(REPO),
        "VDS_ICON": str(ICON_ICO),
        "VDS_DESCRIPTION": DESCRIPTION,
        "VDS_REMOVE": "1" if remove else "0",
    }


def _windows(remove: bool) -> list[Path]:
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        raise ShortcutError("PowerShell was not found; use 'VD Studio.bat' to start the app.")
    target = Path(sys.executable) if remove else gui_python()
    env = {**os.environ, **windows_env(target, remove=remove)}
    proc = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", _PS_SCRIPT],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False)
    if proc.returncode != 0:
        raise ShortcutError("PowerShell could not write the shortcut:\n"
                            + (proc.stderr or proc.stdout).strip())
    return [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]


def _desktop_exec_arg(arg: str) -> str:
    """Quote one argument for a .desktop ``Exec`` key (freedesktop Desktop Entry spec)."""
    if arg and not any(c in arg for c in " \t\n\"'\\><~|&;$*?#()`="):
        return arg.replace("%", "%%")
    inner = "".join("\\" + c if c in '"`$\\' else c for c in arg)
    return ('"' + inner + '"').replace("%", "%%")   # "%" starts a field code


def desktop_entry(python: Path) -> str:
    """The text of the Linux ``.desktop`` file."""
    exec_line = " ".join(_desktop_exec_arg(a) for a in (str(python), *MODULE_ARGS)) + " %f"
    # string values escape the backslash once more (spec: "Exec" is a string first)
    exec_line = exec_line.replace("\\", "\\\\")
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        f"Name={APP_NAME}",
        f"Comment={DESCRIPTION}",
        f"Exec={exec_line}",
        f"Path={REPO}",
        f"Icon={ICON_PNG}",
        "Terminal=false",
        "Categories=Science;Engineering;",
        "",
    ])


def _xdg_desktop_dir() -> Path | None:
    exe = shutil.which("xdg-user-dir")
    if exe:
        out = subprocess.run([exe, "DESKTOP"], capture_output=True, text=True, check=False)
        d = Path(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
        if d is not None and d != Path.home() and d.is_dir():
            return d
    d = Path.home() / "Desktop"
    return d if d.is_dir() else None


def _linux(remove: bool) -> list[Path]:
    data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    paths = [data / "applications" / "vdstudio.desktop"]
    desk = _xdg_desktop_dir()
    if desk is not None:
        paths.append(desk / "vdstudio.desktop")
    done = []
    for p in paths:
        if remove:
            if p.exists():
                p.unlink()
                done.append(p)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(desktop_entry(gui_python()), encoding="utf-8")
        p.chmod(0o755)
        if p.parent == desk and shutil.which("gio"):   # GNOME: allow launching
            subprocess.run(["gio", "set", str(p), "metadata::trusted", "true"],
                           capture_output=True, check=False)
        done.append(p)
    return done


def command_script(python: Path) -> str:
    """The text of the macOS ``.command`` file."""
    def q(s: object) -> str:
        return "'" + str(s).replace("'", "'\\''") + "'"
    return (f"#!/bin/sh\ncd {q(REPO)} || exit 1\n"
            f"exec {q(python)} {' '.join(MODULE_ARGS)} \"$@\"\n")


def _macos(remove: bool) -> list[Path]:
    p = Path.home() / "Desktop" / f"{APP_NAME}.command"
    if remove:
        if p.exists():
            p.unlink()
            return [p]
        return []
    p.write_text(command_script(gui_python()), encoding="utf-8")
    p.chmod(0o755)
    return [p]


def install(remove: bool = False) -> list[Path]:
    """Create (or remove) the shortcuts for this platform; returns the files touched."""
    if not remove and importlib.util.find_spec("PySide6") is None:
        raise ShortcutError(
            f"PySide6 is not installed for {sys.executable}. Install the desktop extra "
            "first: uv sync --extra desktop --inexact")
    if not remove and not ICON_ICO.is_file():
        raise ShortcutError(f"The icon {ICON_ICO} is missing (scripts/make_icon.py makes it).")
    if os.name == "nt":
        return _windows(remove)
    if sys.platform == "darwin":
        return _macos(remove)
    return _linux(remove)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m apps.vdstudio.shortcut",
                                 description=f"Create {APP_NAME} shortcuts.")
    ap.add_argument("--remove", action="store_true", help="remove the shortcuts instead")
    args = ap.parse_args(argv)
    try:
        paths = install(remove=args.remove)
    except ShortcutError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    verb = "Removed" if args.remove else "Created"
    if not paths:
        print("Nothing to remove." if args.remove else "No shortcut was written.")
    for p in paths:
        print(f"{verb}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
