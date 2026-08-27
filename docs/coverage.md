# Coverage baseline and ratchet procedure

## Baseline history

| Date       | Git SHA  | Raw coverage | `fail_under` | Machine context              |
|------------|----------|--------------|--------------|------------------------------|
| 2026-08-27 | 480c76a  | 87.65%       | 87           | blacksmith-4vcpu-ubuntu-2404 |

## Local coverage report

Run the full test suite to generate `coverage.xml`:

```bash
make test
```

The terminal term-missing report and XML artifact are produced automatically via pytest addopts. The `make coverage-report` target verifies the XML exists.

## Ratchet procedure

1. Run `make test` and note the `TOTAL` coverage percentage.
2. If it exceeds the current `fail_under`, round down to a whole percent.
3. Update `pyproject.toml` `[tool.coverage.report] fail_under` to the new floor.
4. Update the comment and this baseline table with the new value, date, SHA, and context.
5. Open a dedicated PR so the coverage-only diff is reviewable in isolation.
