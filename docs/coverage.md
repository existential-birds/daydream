# Coverage reporting and ratchet

## What runs where

`make check` (and therefore the pre-push hook and CI's check job) runs the test suite with coverage flags from the pytest `addopts` in `pyproject.toml`. The terminal report shows missing lines. `coverage.xml` is written and uploaded as the `coverage-report` CI artifact.

## Local coverage

Run the same command that CI uses:

```bash
make install
uv run pytest -n auto
```

The terminal term-missing report is produced automatically via addopts. The `coverage.xml` file is written in the repository root and is gitignored — do not commit it.

## The floor

The current `fail_under` value is:

| Measured % | Raw value | Git SHA | Date | Machine context |
|------------|-----------|---------|------|-----------------|
| 87 | 87.65% | 480c76a | 2026-08-27 | blacksmith-4vcpu-ubuntu-2404 |

## Ratchet procedure

1. `git pull` latest main.
2. `make install`.
3. `uv run pytest -n auto` and read the `TOTAL` percentage from the terminal report (or `uv run coverage report` to re-read the last `.coverage`).
4. Round **down** to a whole percent.
5. If the new percentage is greater than the current `fail_under`, edit `[tool.coverage.report] fail_under` in `pyproject.toml` and update the baseline table above with the new value, date, SHA, and context.
6. Commit with the measured evidence in the commit message.

The floor only ever rises. Lowering it requires a documented regression in the PR description.

## Why atif is excluded

`daydream/atif/` is vendored from the Harbor framework (see `daydream/atif/NOTICE`). The mechanical-edit-only policy (D-03) forbids hand-fixes, so including it in coverage would set the floor from code no one may edit. This exclusion mirrors the mypy, ruff, and vulture exemptions.
