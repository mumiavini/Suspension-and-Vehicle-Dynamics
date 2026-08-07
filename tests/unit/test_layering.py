"""Tests that vdcore/ never imports UI or visualization packages."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

FORBIDDEN_MODULES = frozenset({
    "streamlit",
    "plotly",
    "PySide6",
    "pyqtgraph",
    "pyvista",
    "matplotlib",
})

VDCORE_DIR = Path(__file__).resolve().parents[2] / "vdcore"
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "purity_violation"
CHECK_PURITY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_purity.py"


def _find_forbidden_imports(directory: Path) -> list[tuple[Path, int, str]]:
    """Walk a directory and find all forbidden imports."""
    violations: list[tuple[Path, int, str]] = []
    if not directory.is_dir():
        return violations
    for py_file in sorted(directory.rglob("*.py")):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in FORBIDDEN_MODULES:
                        violations.append((py_file, node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    names = ", ".join(a.name for a in node.names)
                    violations.append(
                        (py_file, node.lineno, f"from {node.module} import {names}")
                    )
    return violations


def test_vdcore_imports_no_ui_packages() -> None:
    """vdcore/ must never import streamlit, plotly, or other UI packages."""
    violations = _find_forbidden_imports(VDCORE_DIR)
    if violations:
        msg_lines = ["Forbidden imports in vdcore/:"]
        for filepath, lineno, stmt in violations:
            msg_lines.append(f"  {filepath}:{lineno} — {stmt}")
        raise AssertionError("\n".join(msg_lines))


def test_check_purity_catches_violations() -> None:
    """scripts/check_purity.py must exit 1 when given a directory with a forbidden import."""
    assert FIXTURES_DIR.is_dir(), f"Fixture directory missing: {FIXTURES_DIR}"
    assert CHECK_PURITY_SCRIPT.is_file(), f"Script missing: {CHECK_PURITY_SCRIPT}"

    result = subprocess.run(
        [sys.executable, str(CHECK_PURITY_SCRIPT), str(FIXTURES_DIR)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1, (
        f"Expected exit code 1 for purity violation, got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "FAIL" in result.stdout
    assert "plotly" in result.stdout
