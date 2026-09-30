@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
if "%~1"=="" goto dashboard
"D:\LBM\envs\bgk-nee-gpu-20260929\Scripts\python.exe" -X utf8 -u run_gpu.py %*
set GPU_EXIT=%ERRORLEVEL%
pause
exit /b %GPU_EXIT%

:dashboard
call "%~dp0open_dashboard_windows.bat"
exit /b %ERRORLEVEL%
