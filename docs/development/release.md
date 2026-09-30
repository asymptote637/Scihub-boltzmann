# Release Guide

This document describes how to prepare and publish a clean release.

## Versioning

Use semantic versioning:

- `MAJOR`: incompatible solver, file format, or public API changes.
- `MINOR`: new cases, boundary conditions, UI features, or compatible output additions.
- `PATCH`: bug fixes, documentation updates, stability checks, and small UI improvements.

The current project version is defined in `pyproject.toml`.

## Release Checklist

Before tagging a release:

1. Confirm the working tree is clean.
2. Run tests:

   ```powershell
   $env:PYTHONPATH=".;src"
   python -m pytest -q
   ```

3. Run a small command-line smoke case:

   ```powershell
   $env:PYTHONPATH="."
   python main.py --case lid_driven_cavity --nx 48 --ny 48 --re 100 --u-ref 0.05 --max-iter 300 --min-iter 100 --report-interval 50
   ```

4. Start the desktop UI and verify it opens:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\scripts\start_desktop.ps1
   ```

5. Check `.gitignore` still excludes:

   - `results/**`
   - `database/*.sqlite`
   - `logs/**`
   - `data/raw/**`
   - `data/processed/**`
   - caches and local state

6. Update `README.md` if startup commands or supported cases changed.
7. Update `docs/project/about.md` if project scope changed.
8. Update this file if the release process changed.

## Tagging

Create an annotated tag:

```powershell
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin main
git push origin v0.1.0
```

## GitHub Release Notes

Use this structure:

```markdown
## Highlights

- Added ...
- Improved ...

## Validation

- `python -m pytest -q`
- Small lid-driven cavity smoke run

## Notes

- Generated results, local SQLite databases, logs, and raw data are intentionally excluded from the repository.
```

## What Not To Release

Do not attach full local run folders unless they are intentionally curated examples. Large simulation data should go to a data archive such as Zenodo, OSF, Figshare, or Git LFS, with a link from the release notes.

