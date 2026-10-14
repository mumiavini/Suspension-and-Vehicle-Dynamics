@echo off
rem VD Studio launcher (Windows). Double-click, or drag a .vdproj.json onto it.
cd /d "%~dp0"
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run --extra desktop python -m apps.vdstudio %*
) else (
    if exist ".venv\Scripts\python.exe" (
        ".venv\Scripts\python.exe" -m apps.vdstudio %*
    ) else (
        echo uv was not found and there is no .venv. Install uv: https://docs.astral.sh/uv/
        pause
    )
)
