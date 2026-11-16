# Handoff: test isolation for #1800 and #1790 (and #1801 verification)

Prepared from `existential-birds/daydream` at `67bba2e05b0546972f3be776006451aefc0c991e`
(`origin/main`, fetched 2026-11-16). This file lives only on the preparation branch
`claude/sweet-planck-2gplm1`. It is context for the implementer and **must not be
included in the PR**.

Sources:
[#1800](https://github.com/existential-birds/daydream/issues/1800),
[#1790](https://github.com/existential-birds/daydream/issues/1790),
[#1801](https://github.com/existential-birds/daydream/issues/1801). None of the three
has comments or amendments. The user wants all of them in **one PR**.

## Summary of scope

| Issue | State at HEAD | Work |
|---|---|---|
| #1801 `test_finalization_boundary` ran the real `gh` | Closed by PR #1906 (`1756782`), which is an ancestor of HEAD | None. Re-verify, and cite the result in the PR (no closing keyword). |
| #1800 the post-findings test leaks a stubbed `get_current_recorder` | Open; still reproduces | Test-only fix in `tests/test_post_findings_integration.py` |
| #1790 the CLI signal test writes into the real `~/.daydream` | Open; still reproduces | One-line test fix in `tests/test_cli.py` |

Expected diff: 2 test files and `CHANGELOG.md`, no production code, and test count
unchanged. Use `Fixes #1800` and `Fixes #1790`. Mention #1801 as verified, with
no closing keyword, because it is already closed.

## #1801: verification only

At HEAD the review fixture in `tests/deep_orchestrator/test_review_completion.py`
requests `fake_gh`. Repro, using a logging `gh` shim that exits 1:

```bash
S=$(mktemp -d) && printf '#!/bin/sh\necho "$@" >> %s/calls.log\nexit 1\n' "$S" > "$S/gh" && chmod +x "$S/gh"
PATH="$S:$PATH" uv run pytest -q -n0 -p no:cacheprovider tests/deep_orchestrator/test_review_completion.py::test_finalization_boundary
cat "$S/calls.log"
```

Observed during preparation: `10 passed` and no `calls.log` (no `gh` calls). Rerun this
on the final HEAD and put the result in the Test Plan.

## #1800: stub leak from `test_artifact_post_never_acquires_live_trajectory_details`

### The test (`tests/test_post_findings_integration.py`, about lines 433-461)
- Inside `with monkeypatch.context() as patch:` it patches four names, each to a
  `forbidden` function that appends to `attempts` and raises `AssertionError("artifact posting attempted live trajectory acquisition")`:
  - `"daydream.trajectory.get_current_recorder"`
  - `"daydream.pr_run_info.render_live_run_info"`
  - `"daydream.pr_comment_renderer.render_run_info_block"`
  - `"tempfile.TemporaryDirectory"`
- It then runs `cli_main(_post_argv(artifact))` in-process (`tests/harness/scripts.py`).
- It asserts exit code 0, `attempts == []`, and the body contents.

### Root cause (verified)
- Post-findings lazily imports the built-in flow modules **while the patch is active**.
  The path is `commands/admin.py` → `pr_review.py:1200 resolve_review_renderers(get_registry())`
  → `extensions/loader.py:106-115 get_registry()` → `register_builtins` →
  `extensions/builtins.py:83-87 _register_builtin_flows`, which imports
  `daydream.deep.orchestrator` and `daydream.improve.orchestrator`.
- Modules that use `from daydream.trajectory import get_current_recorder` bind the stub. After
  the test, 7 stale references remain: `deep.review_terminal` (`get_current_recorder` and
  `render_live_run_info`, from the top-level import at line 23), `deep.merge_steps`,
  `deep.fix_steps`, `deep.diagram_steps`, `exploration_runner` and `improve.artifacts`.
- `test_review_completion.py` now imports `daydream.deep.review_terminal` at collection, so
  **the exact pair from the issue now passes**. The leak itself still reproduces:

```bash
uv run pytest -q -n0 -p no:cacheprovider \
  tests/test_post_findings_integration.py::test_artifact_post_never_acquires_live_trajectory_details \
  tests/deep_orchestrator/test_review_completion.py::test_comment_pins_pr_head_independently_and_excludes_local_edits \
  tests/deep_orchestrator/test_findings_artifacts.py
```

  At HEAD this gives `2 failed, 1 passed` (re-observed during preparation). After the fix
  it should give `3 passed`. The reverse order passes at HEAD. Other files that fail after the post-findings test
  include `deep_orchestrator/test_fix_guards_and_scope.py`, `test_pipeline_intent_and_resume.py` and
  `test_repair_job_acceptance.py`.
- `make check` does **not** show the leak. Each xdist worker collects the whole suite first,
  which imports every module. Only focused or ordered runs leak.

### The test's detection is weak today (important)
Patching the package attribute misses every module that bound the real function
earlier. A simulated regression, where `pr_review` imports `get_current_recorder` by
name and calls it while posting, **passes the current test**. Pre-importing alone
(option a) fixes the leak but keeps this blind spot. Patching every lookup site
(option b) means 12 or more sites, and it silently misses any new importer.

### Recommended fix (prototyped and validated on a scratch copy)
- `get_current_recorder` in `daydream/trajectory/context.py` reads the module global
  `_RECORDER_VAR` at call time. Every alias is that same function, so patching the
  ContextVar guards all of them. `_RECORDER_VAR` is already the documented test seam:
  see the comment at `context.py:12-15` and the autouse `_reset_trajectory_recorder` in
  `tests/conftest.py:436-447`.
- `trajectory/recorder.py` and `trajectory/scopes.py` import `_RECORDER_VAR` by name.
  The `daydream.trajectory` package imports both, and conftest imports `recorder`
  before each test, so they bind the real ContextVar before the patch.
- Call `get_registry()` before patching, so the built-in flow modules, `review_terminal`
  included, bind the real `render_live_run_info`. `get_registry()` is not cached
  (`loader.py:106-115`), so the call only imports modules.
- With only the ContextVar change and no pre-import, the `render_live_run_info` leak into
  `review_terminal` remains. Both changes are needed.

The shape of the change:

```python
from types import SimpleNamespace
from daydream.extensions import get_registry   # the import pr_review.py uses
...
    get_registry()  # import the built-in flow modules before patching, so none keeps a stub
    with monkeypatch.context() as patch:
        # Every get_current_recorder alias reads this ContextVar, including names bound before patching.
        patch.setattr("daydream.trajectory.context._RECORDER_VAR", SimpleNamespace(get=forbidden))
        patch.setattr("daydream.pr_run_info.render_live_run_info", forbidden)
        patch.setattr("daydream.pr_comment_renderer.render_run_info_block", forbidden)
        patch.setattr("tempfile.TemporaryDirectory", forbidden)
        assert cli_main(_post_argv(artifact)) == 0
```

The old `"daydream.trajectory.get_current_recorder"` patch is replaced, because the
ContextVar patch covers it. Results on the scratch copy:
- 0 stale stub references after the test.
- The failing set above passes (7 passed in a wider ordered set).
- The whole file passes (23 passed).
- The simulated by-name regression is now caught, both alone and after full collection.
- ruff and mypy are clean.

To prove detection power in the PR, re-run the simulated regression locally and then
revert it: add a by-name `from daydream.trajectory import get_current_recorder` call
to the post-findings path. Leave it out of the commit.

## #1790: `~/.daydream` leak from `test_signal_flushes_all_runner_recorders`

### The test (`tests/test_cli.py:241-317`, parametrized on `SIGTERM`)
- It builds `child_env = dict(os.environ)`, removes six credential variables, and updates
  `DAYDREAM_EXT_DIR`, `DAYDREAM_SIGNAL_GH_LOG`, `DAYDREAM_SIGNAL_READY` and `PATH` (with a
  fake `gh` first). It never sets `HOME`.
- It runs `sys.executable -m daydream --non-interactive --no-archive --no-eval --flow signal-fixture --pr-number 1 <git_repo>`,
  sends SIGTERM after READY, and expects exit code 130 plus three partial evidence documents
  under `git_repo/.daydream/runs`. The test takes about 1.3 s.

### Root cause
- `daydream/artifacts/ownership.py:31-41`: `_default_private_base()` returns
  `Path.home() / ".daydream"`, and `private_root_locations()` builds `runtime/` and
  `workspaces/` under it. No `DAYDREAM_*` variable overrides it, so `HOME` is the only
  seam a subprocess has.
- The suite isolates it only in-process, through the autouse `artifact_runtime_root`
  fixture (`tests/conftest.py:460-468`). That fixture monkeypatches `_default_private_base`,
  which a child process never sees.

### Fix
Add `"HOME": str(tmp_path / "home")` to the existing `child_env.update({...})` mapping
on its own line. Appending it to the existing line would make that line 132 columns
(ruff E501, 120 max).
- `git_repo` is `tmp_path/repo`, so the private base stays outside source ownership.
- Git identity comes from `tests/harness/git_helpers.py:init_repo`, which is repo-local,
  and the flow never commits. `gh` is a fake.
- The prototype passed, with no slowdown.

No new assertion. The acceptance check is that the run leaves `~/.daydream` unchanged,
shown by comparing it before and after.

### Empirical evidence from preparation
- Before the fix, running the single test created
  `~/.daydream/{runtime,workspaces}/<key>/` with `owner.json` pointing at this test's
  `repo`. After the fix nothing was created, and the run landed in
  `<basetemp>/test_signal_flushes_all_runner0/home/.daydream/`.
- A full suite at HEAD (`uv run pytest -n logical -p no:cacheprovider -q --no-cov`)
  gave **1109 passed** in about 74 s. Comparing `~` before and after showed this test
  as the only writer under the home directory.
- No other test that starts a subprocess writes to the real home (`test_backend_lifecycle`,
  `test_observability_dependencies`, `test_ci_shard`, `test_gate_single_source`,
  `test_makefile_hooks`, `test_workflow_templates`, `test_zip_review`). A suite-wide
  `HOME` guard is **not** warranted: it would change `Path.home()` for every in-process
  test and overlap the existing isolation.

## Out of scope (do not fix in this PR, and do not list it as deferred work)
Preparation found the same class of leak elsewhere: a by-name import binds a stub
that a test patched while the module was first imported. These are separate issues:
- `tests/test_runner.py:724,776-777` (`test_fix_cycle_failing_tests_bounded_fix_then_handoff`).
  A `runner._resolve_backend` lambda leaks into `committed_review`, and console spies
  leak into `deep.review_steps`. Running it before
  `test_comment_pins_pr_head_independently_and_excludes_local_edits` fails that test.
- `tests/test_phases.py:915-916` and `tests/test_workspace.py:80` leak console-print
  lambdas. These are low risk.

## Validation commands (from Makefile, CONTRIBUTING.md and AGENTS.md)
- The #1800 failing set above: `3 passed` expected after the fix.
- `uv run pytest -q -n0 -p no:cacheprovider tests/test_post_findings_integration.py`
- `uv run pytest -q -n0 -p no:cacheprovider "tests/test_cli.py::test_signal_flushes_all_runner_recorders"`,
  with `find ~/.daydream -maxdepth 4 2>/dev/null | sort` captured before and after;
  the two listings must be identical.
- The #1801 gh-shim repro above.
- `make lint`, `make typecheck`, `make deadcode`, then the full gate `make check`
  (lockcheck + install + static + test + coverage-report). Snapshot `~/.daydream`
  before and after `make check` as #1790's acceptance proof.
- Run the `test-audit` skill (`.agents/skills/test-audit/SKILL.md`) on both changed test files.
- Coverage ratchet: `docs/coverage.md`. Never lower `fail_under`.
- The simplification sweep is **not** required: it applies only to PRs over 5 files or
  500 lines (CONTRIBUTING.md).

## Publication notes
- Install the hooks first with `make hooks`. In the preparation container,
  `.git/hooks` had no hooks installed. The pre-push hook checks signatures and runs
  `make check`. Commit signing is configured (`gpg.format=ssh`, `commit.gpgsign=true`).
- Changelog: test-only fixes still get an entry under `[Unreleased]` → `### Fixed`.
  See the #1801 entry for the house style (it explains the impact and links the issue).
- SemVer impact: none (tests only). State this in the PR.
- PR body: `.github/PULL_REQUEST_TEMPLATE.md`, written with the `pr` skill (the template
  takes precedence). Fill every checklist item, with evidence in the Test Plan.
