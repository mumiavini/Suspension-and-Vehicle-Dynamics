"""PostToolUse hook: runs ruff, mypy, and purity check on vdcore/ files.

Reads the tool input from stdin JSON. If the edited file is not under
vdcore/, exits immediately with no output. Otherwise runs formatting,
linting, type checking, and purity validation, and outputs additionalContext.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    try:
        event = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        return

    tool_input = event.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path:
        return

    path = Path(file_path)

    try:
        path_resolved = path.resolve()
    except (OSError, ValueError):
        return

    vdcore_dir = Path.cwd() / "vdcore"
    try:
        path_resolved.relative_to(vdcore_dir.resolve())
    except ValueError:
        return

    messages: list[str] = []

    if path.suffix == ".py" and path.exists():
        fmt_result = subprocess.run(
            ["uv", "run", "ruff", "format", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if fmt_result.returncode != 0:
            messages.append(f"ruff format: {fmt_result.stderr.strip()}")

        check_result = subprocess.run(
            ["uv", "run", "ruff", "check", "--fix", str(path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if check_result.returncode != 0:
            messages.append(f"ruff check: {check_result.stdout.strip()}")

        module = str(path).replace("\\", "/").replace("/", ".").removesuffix(".py")
        parts = module.split(".")
        if len(parts) >= 2:
            mypy_target = ".".join(parts[:2])
        else:
            mypy_target = module

        mypy_result = subprocess.run(
            ["uv", "run", "mypy", "--no-error-summary", str(path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if mypy_result.returncode != 0:
            mypy_output = mypy_result.stdout.strip()
            if mypy_output:
                messages.append(f"mypy: {mypy_output}")

    purity_result = subprocess.run(
        [sys.executable, "scripts/check_purity.py"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if purity_result.returncode != 0:
        messages.append(f"PURITY VIOLATION: {purity_result.stdout.strip()}")

    if messages:
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "\n".join(messages),
            }
        }
        print(json.dumps(output))


if __name__ == "__main__":
    main()
