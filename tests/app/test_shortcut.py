"""Desktop shortcut installer, icon, and the windowless (pythonw) start in app.py."""

from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from apps.vdstudio import app as A
from apps.vdstudio import shortcut as S

REPO = Path(__file__).resolve().parents[2]
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def test_icon_files_hold_every_size_as_png() -> None:
    data = S.ICON_ICO.read_bytes()
    reserved, kind, n = struct.unpack_from("<HHH", data)
    assert (reserved, kind, n) == (0, 1, len(ICO_SIZES))
    sizes = []
    for i in range(n):
        w, h, _, _, planes, bpp, size, off = struct.unpack_from("<BBBBHHII", data, 6 + 16 * i)
        assert w == h and planes == 1 and bpp == 32
        assert data[off:off + 8] == b"\x89PNG\r\n\x1a\n" and off + size <= len(data)
        sizes.append(w or 256)
    assert tuple(sizes) == ICO_SIZES
    assert S.ICON_PNG.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert A.ICON_PNG == S.ICON_PNG


def test_icon_loads_in_qt() -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtGui

    _app = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication([])
    img = QtGui.QImage(str(S.ICON_PNG))
    assert (img.width(), img.height()) == (256, 256)


def _unquote_exec(value: str) -> list[str]:
    """Inverse of the Desktop Entry rules: string unescape, then Exec argument quoting."""
    s = value.replace("\\\\", "\\")
    args, cur, quoted, i = [], "", False, 0
    while i < len(s):
        c = s[i]
        if quoted and c == "\\":
            cur, i = cur + s[i + 1], i + 2
            continue
        if c == '"':
            quoted = not quoted
        elif c == " " and not quoted:
            args.append(cur)
            cur = ""
        else:
            cur += c
        i += 1
    args.append(cur)
    return [a.replace("%%", "%") for a in args if a != "%f"]


@pytest.mark.parametrize("folder", ["plain", "with space", "100%_car",
                                    "it's \"odd\" $HOME `x` \\b"])
def test_desktop_entry_exec_survives_awkward_paths(tmp_path, folder) -> None:
    py = tmp_path / folder / ".venv" / "bin" / "python"
    entry = S.desktop_entry(py)
    exec_value = next(line[5:] for line in entry.splitlines() if line.startswith("Exec="))
    assert _unquote_exec(exec_value) == [str(py), "-m", "apps.vdstudio"]
    assert f"Path={S.REPO}" in entry and f"Icon={S.ICON_PNG}" in entry
    assert "Terminal=false" in entry


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell")
def test_command_script_runs_from_a_folder_with_quotes(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "it's a repo"
    repo.mkdir()
    fake = tmp_path / "py thon"
    fake.write_text('#!/bin/sh\necho "$(pwd)"; for a in "$@"; do echo "[$a]"; done\n')
    fake.chmod(0o755)
    monkeypatch.setattr(S, "REPO", repo)
    script = tmp_path / "VD Studio.command"
    script.write_text(S.command_script(fake))
    out = subprocess.run(["sh", str(script), "my file.json"], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert out == [str(repo), "[-m]", "[apps.vdstudio]", "[my file.json]"]


def test_linux_creates_and_removes_both_entries(tmp_path, monkeypatch) -> None:
    if os.name == "nt" or sys.platform == "darwin":
        pytest.skip("Linux layout")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(S.shutil, "which", lambda name: None)     # no xdg-user-dir, no gio
    (tmp_path / "Desktop").mkdir()
    made = S.install()
    assert made == [tmp_path / "data" / "applications" / "vdstudio.desktop",
                    tmp_path / "Desktop" / "vdstudio.desktop"]
    assert all(p.stat().st_mode & 0o111 for p in made)
    assert S.install(remove=True) == made and not any(p.exists() for p in made)


def test_windows_passes_every_path_through_the_environment(tmp_path, monkeypatch) -> None:
    target = tmp_path / "my repo" / ".venv" / "Scripts" / "pythonw.exe"
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout="C:\\Users\\a\\Desktop\\VD Studio.lnk\n"
                                           "C:\\Users\\a\\Start Menu\\Programs\\VD Studio.lnk\n",
                                           stderr="")

    monkeypatch.setattr(S.shutil, "which", lambda name: "C:\\ps\\powershell.exe")
    monkeypatch.setattr(S.subprocess, "run", run)
    monkeypatch.setattr(S, "gui_python", lambda executable=None: target)
    out = S._windows(remove=False)
    assert [str(p).endswith("\\VD Studio.lnk") for p in out] == [True, True]
    (cmd, kw), = calls
    assert cmd[:4] == ["C:\\ps\\powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    script, env = cmd[4], kw["env"]
    assert str(tmp_path) not in script                 # no path is spliced into the script
    assert env["VDS_TARGET"] == str(target) and env["VDS_ARGS"] == "-m apps.vdstudio"
    assert env["VDS_WORKDIR"] == str(S.REPO) and env["VDS_ICON"] == str(S.ICON_ICO)
    assert env["VDS_NAME"] == "VD Studio" and env["VDS_REMOVE"] == "0"
    for key in ("VDS_TARGET", "VDS_ARGS", "VDS_WORKDIR", "VDS_ICON", "VDS_REMOVE"):
        assert f"$env:{key}" in script
    assert "GetFolderPath('Desktop')" in script and "GetFolderPath('Programs')" in script

    monkeypatch.setattr(S.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, stdout="", stderr="Access denied"))
    with pytest.raises(S.ShortcutError, match="Access denied"):
        S._windows(remove=False)


def test_install_refuses_without_the_desktop_extra(monkeypatch, capsys) -> None:
    monkeypatch.setattr(S.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(S.ShortcutError, match="uv sync --extra desktop --inexact"):
        S.install()
    assert S.main([]) == 1 and "PySide6 is not installed" in capsys.readouterr().err


def test_installer_batch_never_runs_an_exact_sync() -> None:
    raw = (REPO / "Install VD Studio shortcut.bat").read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")   # CRLF only
    text = raw.decode("ascii")
    syncs = [line for line in text.splitlines() if "uv sync" in line and "echo" not in line]
    assert syncs and all("--inexact" in s for s in syncs)    # keeps the legacy app's extras
    assert "-m apps.vdstudio.shortcut" in text and 'cd /d "%~dp0"' in text


def test_windowless_start_logs_to_a_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert A.redirect_missing_streams(tmp_path / "x.log") is None   # a console exists
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    log = tmp_path / "logs" / "vdstudio.log"
    assert A.redirect_missing_streams(log) == log
    print("hello from pythonw")
    sys.stderr.write("an error\n")
    sys.stdout.flush()
    assert log.read_text(encoding="utf-8") == "hello from pythonw\nan error\n"
    sys.stdout.close()


def test_log_path_follows_localappdata(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert A.log_path() == tmp_path / "VD Studio" / "vdstudio.log"


def test_error_dialog_hook(monkeypatch, tmp_path) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda parent, title, text: shown.append(text)))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    A.install_error_dialog(tmp_path / "vdstudio.log")
    try:
        raise ValueError("boom")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert len(shown) == 1 and "ValueError: boom" in shown[0] and "vdstudio.log" in shown[0]
    assert "Traceback" in err.getvalue() and "boom" in err.getvalue()
