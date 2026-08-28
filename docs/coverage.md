# Coverage reporting and ratchet

## What runs where

`make test` (and therefore `make check`, the pre-push hook, and CI's check job) runs the full test suite with coverage flags carried by that invocation — not global pytest `addopts`. A targeted/dev run (`pytest tests/foo.py`) is a plain, ungated pytest run, because coverage flags in global addopts would turn any subset run into a hard `fail_under` failure even when all its tests pass (#336). The terminal report shows missing lines. `coverage.xml` is written and uploaded as the `coverage-report` CI artifact.

## Local coverage

Run the same command that CI uses:

```bash
make install
uv run pytest -n auto --cov --cov-branch --cov-report=term-missing --cov-report=xml
```

The terminal term-missing report is produced automatically. The `coverage.xml` file is written in the repository root and is gitignored — do not commit it. (Or just run `make test`, which is identical.)

## The floor

The current `fail_under` value is:

| Measured % | Raw value | Git SHA | Date | Machine context |
|------------|-----------|---------|------|-----------------|
| 86 | 86.94% | 411a50d | 2026-08-28 | blacksmith-4vcpu-ubuntu-2404 |

Regression note: the 2026-08-27 baseline (87.65% at 480c76a) was a whole-percent
round up to 87 that the full-suite run could not reproduce once the coverage flags
moved from the global `addopts` to the `make test`/CI invocation row (measured
86.94%). The floor was lowered to the reproduced measurement (rounded down), per
the ratchet's regression rule.

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
