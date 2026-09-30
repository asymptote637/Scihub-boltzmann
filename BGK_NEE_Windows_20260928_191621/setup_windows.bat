@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
py -3.11 --version
if errorlevel 1 goto fail
if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements-gui.txt
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -X utf8 transfer.py check
if errorlevel 1 goto fail
echo Setup and preflight complete. Read README_FIRST.md before running.
pause
exit /b 0
:fail
echo Setup failed. Keep this output for Codex; do not start the long run.
pause
exit /b 1
