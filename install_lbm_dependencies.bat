@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "BASE_PYTHON="

if not "%LBM_PYTHON%"=="" (
    set "BASE_PYTHON=%LBM_PYTHON%"
)

if "%BASE_PYTHON%"=="" (
    set "BUNDLED_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if exist "!BUNDLED_PYTHON!" (
        set "BASE_PYTHON=!BUNDLED_PYTHON!"
    )
)

if "%BASE_PYTHON%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "BASE_PYTHON=python"
    )
)

if "%BASE_PYTHON%"=="" (
    echo Python was not found.
    echo Install Python 3.11+ first, or set LBM_PYTHON to python.exe.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment: .venv
    "%BASE_PYTHON%" -m venv .venv
    if errorlevel 1 (
        echo.
        echo Failed to create .venv.
        pause
        exit /b 1
    )
)

set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"

echo Installing LBM desktop simulator dependencies into .venv...
"%PYTHON_EXE%" -m pip install -U pip
"%PYTHON_EXE%" -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo Dependencies installed. You can now double-click start_lbm_desktop.bat.
pause
