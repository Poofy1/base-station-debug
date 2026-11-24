@echo off
rem lhdebug launcher. Double-click to start; sets itself up on first launch.
rem   run.bat              attach to SteamVR and open the dashboard
rem   run.bat simulate     demo with a simulated rig (no headset needed)
rem   run.bat report       print the findings of the latest recorded session
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if exist "%PY%" goto deps

echo [lhdebug] First launch: creating a Python environment in .venv ...
set "BOOT="
py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && set "BOOT=py -3"
if not defined BOOT python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>&1 && set "BOOT=python"
if not defined BOOT goto nopython
%BOOT% -m venv .venv
if errorlevel 1 goto failed

:deps
fc /b pyproject.toml ".venv\pyproject.installed" >nul 2>&1
if not errorlevel 1 goto launch
echo [lhdebug] Installing dependencies (one-time, needs internet) ...
"%PY%" -m pip install --disable-pip-version-check -q -e .
if errorlevel 1 goto failed
copy /y pyproject.toml ".venv\pyproject.installed" >nul

:launch
if "%~1"=="" (
    "%PY%" -m lhdebug run --open
) else (
    "%PY%" -m lhdebug %*
)
if errorlevel 1 pause
exit /b %errorlevel%

:nopython
echo.
echo [lhdebug] Python 3.11 or newer was not found.
echo           Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
echo           then run this file again.
pause
exit /b 1

:failed
echo.
echo [lhdebug] Setup failed - see the messages above.
pause
exit /b 1
