# Contributing

Contributions should improve numerical correctness, reproducibility, documentation, or usability without mixing generated research outputs into the repository.

## Before You Start

1. Open or identify an issue describing the change.
2. Keep the change scoped: one solver feature, one boundary condition, one UI improvement, or one documentation update.
3. Check `docs/publication_policy.md` before adding data or output files.

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .[analysis,dev]
```

Run tests:

```powershell
$env:PYTHONPATH=".;src"
python -m pytest -q
```

## Coding Guidelines

- Use `(NY, NX, 9)` for distribution arrays.
- Use `[y, x]` indexing for fields.
- Keep D2Q9 direction ordering consistent with `boundary_conditions.py`.
- Add boundary conditions as independent functions and register them explicitly.
- Validate numerical parameters before running long simulations.
- Prefer configuration files and small smoke tests over hard-coded research runs.
- Do not commit generated simulation outputs.

## Numerical Changes

For solver or boundary-condition changes, include:

- A short explanation of the method.
- A smoke test or regression test when practical.
- Stability notes for `tau`, `omega`, Mach number, and mass drift.
- Any known limitations.

## UI Changes

For desktop UI changes:

- Keep the UI responsive during simulation.
- Do not run long calculations on the Qt main thread.
- Preserve access to diagnostics: residual, `q`, `q_avg`, mass drift, `tau`, `omega`, and Mach number.

## Pull Request Checklist

- Tests pass.
- README or docs updated when behavior changes.
- No generated `results/`, SQLite databases, logs, caches, or local environment files are staged.
- New public files do not contain personal paths, tokens, credentials, or unpublished data.

