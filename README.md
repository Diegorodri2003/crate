# crate

Local sample librarian: analyze messy folders, search by character, preview a rename plan, then apply with undo.

```bash
python make_fixtures.py          # synthetic messy/ library
python -m unittest tests.test_crate
python api.py                    # http://127.0.0.1:8765
```

Modules stay aligned with `contracts.py`:

- `analyzer.analyze(path)` → `SampleRecord`
- `index.upsert` / `index.search`
- `organiser.scan` / `plan` / `apply` / `undo`
- `api.py` is the HTTP + HTML shell
