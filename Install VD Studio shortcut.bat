@echo off
rem One-time setup for VD Studio (Windows). Double-click it.
rem Installs the desktop dependencies into .venv and puts a "VD Studio" shortcut
rem on the Desktop and in the Start menu. Run it again after moving this folder.
rem To remove the shortcuts, run from a terminal:  "Install VD Studio shortcut.bat" --remove
cd /d "%~dp0"
if /i "%~1"=="--remove" goto :shortcut
where uv >nul 2>nul
if %errorlevel%==0 (
    echo Installing the VD Studio desktop dependencies with uv...
    uv sync --extra desktop --inexact
    if errorlevel 1 goto :fail
) else (
    echo uv was not found: using the existing .venv as it is.
)
:shortcut
if not exist ".venv\Scripts\python.exe" (
    echo There is no .venv in this folder. Install uv from https://docs.astral.sh/uv/ and run this file again.
    goto :fail
)
".venv\Scripts\python.exe" -m apps.vdstudio.shortcut %*
if errorlevel 1 goto :fail
echo.
if /i "%~1"=="--remove" (
    echo Done.
) else (
    echo Done. Open VD Studio from the Desktop icon or from the Start menu.
)
pause
exit /b 0

:fail
echo.
echo Setup did not finish - see the messages above.
pause
exit /b 1
