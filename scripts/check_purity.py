"""Check that vdcore/ imports no UI or visualization packages.

Usage: python scripts/check_purity.py [directory]
Default directory: vdcore/

Exit 0 if clean, exit 1 if violations found.
"""

from __future__ import annotations

import ast
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


def check_file(filepath: Path) -> list[tuple[int, str]]:
    """Return list of (line_number, import_statement) for forbidden imports."""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    violations.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    names = ", ".join(a.name for a in node.names)
                    violations.append((node.lineno, f"from {node.module} import {names}"))
    return violations


def main() -> int:
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("vdcore")

    if not target_dir.is_dir():
        print(f"Directory not found: {target_dir}")
        return 0

    all_violations: list[tuple[Path, int, str]] = []
    file_count = 0

    for py_file in sorted(target_dir.rglob("*.py")):
        file_count += 1
        for lineno, stmt in check_file(py_file):
            all_violations.append((py_file, lineno, stmt))

    if all_violations:
        print("PURITY CHECK: FAIL")
        print("Forbidden imports found:")
        for filepath, lineno, stmt in all_violations:
            print(f"  {filepath}:{lineno} — {stmt}")
        return 1
    else:
        print(f"PURITY CHECK: PASS")
        print(f"No forbidden imports found in {target_dir}/")
        print(f"{file_count} files scanned")
        return 0


if __name__ == "__main__":
    sys.exit(main())
