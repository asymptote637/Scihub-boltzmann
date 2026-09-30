# Contributing

Thanks for helping improve Scihub Boltzmann.

Please read the detailed contribution guide:

- `docs/development/contribute.md`

Before opening a pull request, make sure tests pass:

```powershell
$env:PYTHONPATH=".;src"
python -m pytest -q
```

Do not commit generated simulation outputs, local SQLite databases, logs, caches, raw unpublished data, or credentials. See `docs/development/publication_policy.md`.

