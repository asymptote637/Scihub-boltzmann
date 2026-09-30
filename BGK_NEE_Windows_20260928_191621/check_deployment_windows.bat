@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -X utf8 validation_windows\verify_deployment.py
if errorlevel 1 goto fail
echo Local deployment integrity and preflight passed.
pause
exit /b 0
:fail
echo Local deployment verification failed. Preserve logs before starting a run.
pause
exit /b 1
