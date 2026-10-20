# Coverage reporting and ratchet

## What runs where

`make test` (and therefore `make check` and the pre-push hook) runs the full test suite with coverage flags carried by that invocation — not global pytest `addopts`. A targeted/dev run (`pytest tests/foo.py`) is a plain, ungated pytest run, because coverage flags in global addopts would turn any subset run into a hard `fail_under` failure even when all its tests pass (#336). The terminal report shows missing lines. `coverage.xml` is written and uploaded as the `coverage-report` CI artifact.

## CI aggregation

Full CI defaults to sixteen `ubuntu-24.04` pytest jobs with four workers each.
Each job collects the complete root suite and uses the committed timing profile
in `scripts/ci_test_durations.json` for a deterministic duration-weighted partition.
The profile supplies scheduling hints; the actual collection owns test selection,
and newly collected tests receive a positive fallback weight. Each receipt records
the complete collection and selected/executed tests, and binds the partition
algorithm and profile checksum. Every receipt also binds its
coverage data to the checkout commit/tree, workflow run and attempt, shard index,
and coverage checksum. Per-test setup/call/teardown durations remain available
in the shard artifacts for profiling.

Within each shard, selected tests are ordered by decreasing estimated duration.
The command passes the public `--dist=loadgroup` option to pytest-xdist. In the
[pinned scheduler](https://github.com/pytest-dev/pytest-xdist/blob/v3.8.0/src/xdist/scheduler/loadgroup.py),
each unmarked test is its own scheduling unit, allowing workers to begin the
expensive head concurrently. This avoids ordinary load scheduling's initial
bulk batches concentrating many long tests on one worker. No private scheduler
override is installed. This scheduling choice does not establish the two-minute
whole-job target.

Partial shards collect branch coverage without enforcing the whole-project floor.
The aggregation job requires every expected nonempty shard, matching collection
and checkout identities, complete execution, successful pytest exits, and valid
branch data before combining. Missing, duplicate, unexpected, cancelled, or
corrupt evidence fails the gate. Relative source filenames allow coverage from
separate runner workspaces to combine. The final report and XML use the unchanged
configuration in `pyproject.toml`, including the aggregate floor and exclusions.
The published artifact remains `coverage-report`; raw shard evidence is available
as `shard-INDEX` for every index in the planned matrix.

Complete failed executions retain their coverage data for a diagnostic XML report.
Diagnostic combination still fails the gate after publishing that report; it
cannot authorize merge-group proof. Missing, corrupt, or mismatched shard inputs
cannot produce a complete diagnostic report.

Use **Re-run all jobs** when retrying a full gate. Receipts bind every shard to
the workflow run and immutable attempt; **Re-run failed jobs** cannot reuse
successful shards from an earlier attempt. Full reruns replace every shard
artifact and the coverage report before any new proof is published.

The required `check` runs even when dependencies fail or are cancelled. It
requires successful static checks and, in full mode, successful shards and
coverage aggregation. Basic/reuse modes require those full-only jobs to be
skipped. Only a successful complete merge-group gate can publish reusable proof.

Local coverage needs no CI artifacts and still runs the entire suite in one
invocation. Manual CI dispatch accepts one through twenty jobs for the
[controlled benchmark](ci-sharding-benchmark.md). One selects the original
sequential eight-worker gate; every other count uses four workers per job.
Sixteen is the initial duration-weighted experiment toward a maximum of two
minutes per whole shard job, including setup and artifact upload; that target
has not yet been demonstrated. The frozen four-job measurement remains separate
historical evidence.
The sequential benchmark also publishes `baseline-tests.xml` inside
`coverage-report`, allowing exact test and skip identity comparisons against
shard manifests. Its reporting flags use the optional `PYTEST_REPORT_ARGS`
Makefile variable, which is empty for normal local invocations and applies only
to the root pytest command. Nested pytest subprocesses do not inherit those
flags through `PYTEST_ADDOPTS`.

## Local coverage

Run the complete local coverage gate:

```bash
make install
uv run pytest -n auto --cov --cov-branch --cov-report=term-missing --cov-report=xml
```

The terminal term-missing report is produced automatically. The `coverage.xml` file is written in the repository root and is gitignored — do not commit it. (Or just run `make test`, which is identical.)

## The floor

The current `fail_under` value is:

| Measured % | Raw value | Git SHA | Date | Machine context |
|------------|-----------|---------|------|-----------------|
| 90 | 90.59% | 060e463f | 2026-10-12 | macOS arm64, Python 3.12.13, 16 workers |

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
