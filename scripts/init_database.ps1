param(
    [string]$Python = $env:LBM_PYTHON
)

$ErrorActionPreference = "Stop"

if (-not $Python) {
    $SystemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($SystemPython) {
        $Python = $SystemPython.Source
    }
}

if (-not $Python) {
    $BundledPython = Join-Path $HOME ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path $BundledPython) {
        $Python = $BundledPython
    }
}

if (-not $Python) {
    throw "Python was not found. Install Python 3.11+ or set LBM_PYTHON to python.exe."
}

$env:PYTHONPATH = "src"
& $Python scripts\init_database.py
