@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE="

if not "%LBM_PYTHON%"=="" (
    set "PYTHON_EXE=%LBM_PYTHON%"
)

if "%PYTHON_EXE%"=="" (
    set "BUNDLED_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if exist "%BUNDLED_PYTHON%" (
        set "PYTHON_EXE=%BUNDLED_PYTHON%"
    )
)

if "%PYTHON_EXE%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo Python was not found.
    echo Install Python 3.11+ or set LBM_PYTHON to python.exe.
    pause
    exit /b 1
)

set "PYTHONPATH=."
"%PYTHON_EXE%" desktop_app.py

if errorlevel 1 (
    echo.
    echo The desktop simulator exited with an error.
    pause
)

