@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
 echo Run setup_windows.bat first.
 pause
 exit /b 1
)
".venv\Scripts\python.exe" -X utf8 transfer.py dashboard
set run_result=%errorlevel%
pause
exit /b %run_result%
