@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
if not exist "D:\LBM\envs\bgk-nee-gpu-20260929\Scripts\pythonw.exe" (
  echo The GPU workspace Python environment is missing.
  pause
  exit /b 1
)
start "" "D:\LBM\envs\bgk-nee-gpu-20260929\Scripts\pythonw.exe" -X utf8 "%~dp0open_dashboard.py"
