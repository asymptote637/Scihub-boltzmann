# Packaging Guide

This project can be used directly from source or installed as an editable Python package.

## Development Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[analysis,dev]
```

If using the bundled Codex Python, set `LBM_PYTHON` or call the interpreter path directly.

## Runtime Dependencies

Core dependencies:

- `numpy`
- `matplotlib`
- `pandas`
- `PySide6`
- `Pillow`

Development dependencies:

- `pytest`
- `ruff`

The optional Streamlit UI also needs `streamlit`.

## Command-Line Entrypoints

The root-level runner is:

```powershell
$env:PYTHONPATH="."
python main.py --case lid_driven_cavity
```

The older package-style runner is:

```powershell
$env:PYTHONPATH="src"
python -m lbm_lab.runner --config configs\lid_driven_cavity.toml
```

Desktop UI:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_desktop.ps1
```

## Build a Wheel

Install build tooling:

```powershell
python -m pip install build
```

Build:

```powershell
python -m build
```

The build artifacts will appear in `dist/`. Do not commit `dist/` unless a release workflow explicitly requires it.

## Packaging Notes

- Generated files under `results/`, `logs/`, and `database/` are not package data.
- Keep examples small and config-driven.
- Avoid embedding local machine paths in committed configs.
- If future releases include sample figures, place curated images under `docs/figures/`, not under `results/`.

