param(
    [string]$Python = $env:LBM_PYTHON
)

$ErrorActionPreference = "Stop"

function Test-LbmPython {
    param([string]$Candidate)
    if (-not $Candidate) {
        return $false
    }
    try {
        & $Candidate -c "import numpy, matplotlib, pandas, PySide6, PIL" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

$SelectedPython = $null

if ($Python -and (Test-LbmPython $Python)) {
    $SelectedPython = $Python
}

if (-not $SelectedPython) {
    $BundledPython = Join-Path $HOME ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if ((Test-Path $BundledPython) -and (Test-LbmPython $BundledPython)) {
        $SelectedPython = $BundledPython
    }
}

if (-not $SelectedPython) {
    $SystemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($SystemPython -and (Test-LbmPython $SystemPython.Source)) {
        $SelectedPython = $SystemPython.Source
    }
}

if (-not $SelectedPython) {
    throw "No usable Python was found. Install dependencies with: python -m pip install -r requirements.txt, or set LBM_PYTHON to a python.exe that has numpy, matplotlib, pandas, PySide6, and Pillow."
}

$env:PYTHONPATH = "."
& $SelectedPython desktop_app.py
