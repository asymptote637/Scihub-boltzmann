@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHON_EXE="

if exist ".venv\Scripts\python.exe" (
    call :check_python ".venv\Scripts\python.exe"
)

if not "%LBM_PYTHON%"=="" (
    call :check_python "%LBM_PYTHON%"
)

if "%PYTHON_EXE%"=="" (
    set "BUNDLED_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if exist "!BUNDLED_PYTHON!" (
        call :check_python "!BUNDLED_PYTHON!"
    )
)

if "%PYTHON_EXE%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        call :check_python "python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo No usable Python environment was found.
    echo.
    echo The desktop simulator needs:
    echo   numpy
    echo   matplotlib
    echo   pandas
    echo   PySide6
    echo   Pillow
    echo.
    echo If Python is installed, run:
    echo   install_lbm_dependencies.bat
    echo.
    echo This creates a local .venv and installs dependencies there.
    echo.
    echo Or set LBM_PYTHON to a python.exe that already has these packages.
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

exit /b %errorlevel%

:check_python
if "%~1"=="" exit /b 0
"%~1" -c "import numpy, matplotlib, pandas, PySide6, PIL" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_EXE=%~1"
)
exit /b 0
