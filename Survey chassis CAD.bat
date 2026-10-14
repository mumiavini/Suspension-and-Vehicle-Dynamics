@echo off
rem Reads a SolidWorks chassis model (geometry only, read-only, nothing saved) into
rem projects\cad\<name>_survey.json and an isometric picture (.png) next to it.
rem Drag the .SLDPRT onto this file, or double-click it and paste the path.
setlocal
cd /d "%~dp0"
set "MODEL=%~1"
if not defined MODEL set /p "MODEL=Path of the .SLDPRT file (drag it into this window, then Enter): "
if not defined MODEL goto :nofile
set "MODEL=%MODEL:"=%"
if not exist "%MODEL%" goto :nofile
if not exist ".venv\Scripts\python.exe" (
    echo There is no .venv here. Run "Install VD Studio shortcut.bat" first.
    goto :fail
)
".venv\Scripts\python.exe" -c "import win32com.client" >nul 2>nul
if errorlevel 1 (
    echo Installing pywin32, needed to talk to SolidWorks...
    uv sync --extra desktop --inexact
    if errorlevel 1 goto :fail
)
echo Opening the model in SolidWorks (read-only). This can take a minute...
".venv\Scripts\python.exe" scripts\survey_chassis.py --file "%MODEL%"
if errorlevel 1 goto :fail
echo.
echo Done. Send the .json (and the .png) from projects\cad\ back for the analysis.
explorer "projects\cad"
pause
exit /b 0

:nofile
echo File not found: "%MODEL%"
:fail
echo.
echo The survey did not finish - see the messages above.
pause
exit /b 1
