@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
"D:\LBM\envs\bgk-nee-gpu-20260929\Scripts\python.exe" -X utf8 check_gpu.py
set GPU_EXIT=%ERRORLEVEL%
pause
exit /b %GPU_EXIT%
