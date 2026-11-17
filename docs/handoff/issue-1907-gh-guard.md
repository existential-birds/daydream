# Handoff: issue #1907 — suite-wide fail-closed `gh` guard

Context gathered on 2026-11-17 by a read-only preparation session. The handoff
branch is `claude/wonderful-cori-84bzyq`. **This file is never merged.** Read it with
`git show origin/claude/wonderful-cori-84bzyq:docs/handoff/issue-1907-gh-guard.md`,
and build the PR branch from the verified base, not from the handoff branch.

Source: https://github.com/existential-birds/daydream/issues/1907. It has no comments.
It follows up #1801, whose test_review_completion.py fix landed in #1906.

## Baseline (verified)

- Base `main` = `origin/main` = `4a38b108af426cc98d2bc912f325339095bc8579`. This was
  fetched fresh during preparation, and the tree was clean.
- Host `gh` is installed at `/usr/local/bin/gh`.
- `make install` was needed: without it, `tests/test_hub.py` fails to collect with
  `ModuleNotFoundError: httpx2`.
- Repro: the issue's logging shim, written to a scratch directory, plus the command
  `uv run pytest -q -n logical`.
  - Result: `1105 passed, 1 error (test_hub import)` in about 63 seconds.
  - The shim logged **74 calls from 53 test IDs (49 functions) in 17 files**, exactly
    as the issue reports: 68 `api /user`, 3 `pr list --head feature`, 1 `repo view`,
    1 `pr view 7` and 1 `pr diff 7`. The per-test list is in the appendix.
- Calls from tests that use `fake_gh` and that FakeGh answers with rc=1 because no
  response is canned: these are measured with a probe that wrapped `_handle_gh`.
  - 50 are `GET user` and 24 are `no pr-view response configured`. They come from
    about 25 tests, mainly test_review_completion, test_diagram_only_integration,
    test_integration and test_diagram_overlap.
  - Many of these failures are deliberate, for example the PR-lookup-failure tests.
  - **Scope decision:** `fake_gh` keeps its current answer-with-rc=1 semantics. The
    guard fails only calls that no explicit fake owns. State this in the PR. Making
    FakeGh strict would be a separate, much larger change.

## Production spawn routes (all in `daydream/git_ops/process.py`)

- **Sync:** `_run_gh` (about l.497) → `_retrying_subprocess(["gh", ...], env=env)` →
  `subprocess.run` (l.185). argv[0] is the literal `"gh"`, looked up on PATH.
- **Async:** `_run_gh_async` (about l.557) → `asyncio.create_subprocess_exec("gh", ...,
  env=env, start_new_session=True)` (l.582). Its only caller is `_gh_api_read`
  (`git_ops/github.py:288`): remote_ci, branch rules and bounded pages.
- **`env`:** comes from `auth.environment_for_request()`. `INHERIT` gives `None`, so the
  child gets the live `os.environ`. `StaticGitHubAuth` takes a full env snapshot at
  construction, so a PATH shim installed by the autouse fixture is already in it.
- **Identity:** `runner._run_with_context` (l.474-488) → `github_app.resolve_run_identity`
  → `resolve_user_identity` (`github_app.py:325`). It runs `gh_api("/user")` and
  catches `except Exception`, returning `"unknown"`. AssertionError is swallowed; only
  BaseException types (`pytest.fail` / `Failed`) escape. So: **record the call and
  fail at teardown**. Do not depend on an exception reaching the test.
- **`workspace._resolve_base`** (l.454) runs `gh pr list` only when `shutil.which("gh")`
  finds a `gh`. With a guard shim on PATH, it always finds one. That is intended,
  because it makes the outcome independent of the host.
- **Git credential helpers:** Daydream never configures `gh auth git-credential`. A
  user-level helper could reach `gh` during network Git operations; a PATH shim
  catches that.

## Test harness facts

- `tests/harness/fake_gh.py`:
  - `install_fake_gh` puts its shim first on PATH (the async/process route).
  - It patches `daydream.git_ops.process.subprocess.run`, which is the global
    `subprocess.run`, with a router that captures `real_run` at install time.
  - Its `bin_dir.mkdir()` has no `exist_ok`.
- `block_real_gh` (l.534) uses the raise-based pattern the issue rejects. Its only user
  is the autouse `_block_real_gh` at `tests/test_github_app.py:30`. **Supersede and
  delete both.**
- `tests/conftest.py`, the only conftest:
  - The autouse `_isolate_github_cli_env` (l.367) only strips tokens and sets
    `GH_CONFIG_DIR`; the real binary still runs.
  - `fake_gh` is at l.526 and uses `tmp_path/"fake-gh-state"`.
  - Autouse fixtures are set up before requested fixtures, and they share one
    `monkeypatch`. So `fake_gh`'s router captures the guard's router as `real_run`
    and wins, and its PATH shim is first. Takeover is automatic.
- Teardown-failure pattern to reuse: a yield fixture that asserts after `yield`, like
  `no_ci_remote` → `NoCIRemote.finish` (`harness/remote_ci.py:291`). It is reported
  as an ERROR at teardown and works under xdist.
- Keep guard state outside `tmp_path`, in a sibling such as
  `tmp_path.parent / f"{tmp_path.name}-gh-guard"`, the way `dataset_store` and
  `artifact_runtime_root` do. Some tests use `tmp_path` itself as a repo, and
  `test_fake_gh_contract.py:16` runs `install_fake_gh(tmp_path)`.

### Tests that own `gh` themselves (must keep passing, unguarded)

- `tests/test_fix_footprint.py::_install_scope_boundary_shims` (l.64-129) puts its own
  `git` and `gh` shims on PATH, and `gh issue create` reaches them through **sync**
  `subprocess.run`. **A sync interceptor that catches every `gh` breaks these two
  tests.**
  - Rule: resolve `gh` with `shutil.which("gh", path=<the call's env PATH or
    os.environ PATH>)`.
  - If it resolves to the guard's own shim, or to anything outside the pytest
    basetemp (the host binary), record the call and refuse it with rc=1.
  - If it resolves to a test-owned shim under basetemp, run it.
  - (Alternatively, move this test to `fake_gh`, but only if that keeps its proof,
    which is the file bytes at `issue create` time.)
- `tests/test_cli.py::_write_fake_gh` / `test_signal_flushes_all_runner_recorders`
  (l.166, l.242): a child `python -m daydream` with its own shim first on PATH. The
  guard shim sits later on PATH and never runs.
- Real-process lifecycle through FakeGh's PATH shim:
  `test_git_ops_async.py::test_timed_out_request_kills_reaps_its_process_group_and_closes_fds`
  (l.233) and `test_bounded_github_reads_fail_closed_at_shape_and_limit_boundaries`
  (l.170).
- In-test patches of `subprocess.run` or `create_subprocess_exec` override the guard and
  restore it afterwards. Examples: test_git_ops.py (12 sites), test_git_ops_async.py:59
  and l.115, test_github_app.py:42, test_cli_github_app.py:62 and l.200,
  test_runner_execution_factory.py:66, test_runner.py:374, test_improve_plans.py:642,
  test_analyzer.py:775 and l.800, test_artifact_visibility.py:1231 and l.1256, and
  test_review_completion.py:436.
- Any async interceptor must filter on argv[0] == "gh", because
  `create_subprocess_exec` also starts backends and test commands.
- Async env gap: an env such as `StaticGitHubAuth({"PATH": "/usr/bin"})` (test_runner.py:143) leaves the guard shim off PATH, so an async gh could resolve a host gh there. The async patch must resolve against the call's env PATH.
- Tests that replace PATH entirely (`test_git_ops.py:239` and l.546,
  `harness/git_helpers.py:184`, `test_backend_pi.py:904`) drop the shim. None of them
  run `gh`. The interceptor covers the env-PATH case anyway.

## Suggested design (smallest cohesive)

The guard is one autouse fixture in `tests/conftest.py`, with its helpers in
`tests/harness/fake_gh.py`, the existing owner of `gh` interception:

1. Write a guard shim (`#!/bin/sh`: append argv to a log, exit 1), prepended to PATH.
   This covers async, child processes and Git credential helpers.
2. Patch sync `daydream.git_ops.process.subprocess.run` and async
   `daydream.git_ops.process.asyncio.create_subprocess_exec`, both filtered to `gh`.
   Apply the resolution rule above; record the call and return rc=1 when blocked.
3. At teardown, if any call was recorded, `pytest.fail` with the test's argv list.
4. `fake_gh` and test-owned shims take over without coordination, as described above.
5. Prove the guard by extending `tests/test_fake_gh_contract.py`. It has one test,
   which keeps the count net-neutral; the removed `block_real_gh` user offsets any
   addition.
   - Call the guard's check/record helpers directly for sync, async and a call
     swallowed by production (for example, through `resolve_user_identity`).
   - Assert that the teardown check fails, and that a call reaching a test-owned shim
     does not.

## Routing the 53 offenders

See the per-file section below. The default is to request `fake_gh` at the smallest
shared entry point. Its rc=1 for unconfigured `GET user` reproduces today's
`"unknown"` identity. Use a canned response where the test needs a real answer.

Every offender already sees a failing gh call today: either the exit-1 shim or a `GitError`
that the caller swallows. No test in these files reads the identity or the GitHub login.
So routing a test through `fake_gh`, which answers unconfigured calls with rc=1, changes no
assertion. Canned responses would *change* behavior: for example, a `repo-view` reply would
set `pr_repo`. Prefer rc=1.

**Where to put it.** Use the module-level line
`pytestmark = pytest.mark.usefixtures("fake_gh")` unless a narrower point is listed below.
It is safe where a test also requests `fake_gh` directly, because pytest gives both the same
instance.

**Where not to put it.** Do not add `fake_gh` to the shared helpers `tests/harness/stub_backend.py`
(`silence`, `install_stub_backend`) or `cli_runner` (conftest l.505). Many other tests use them,
and a second `install_fake_gh` on the same state dir raises `FileExistsError`.

Repo fixtures all live in subdirectories of `tmp_path`, so `tmp_path/fake-gh-state` never
shows up as an untracked repo file.

| File | Offenders | Change | Notes |
|---|---|---|---|
| `tests/test_improve_flow.py` | about 22 | `pytestmark` | `repo view` is from `runner.py:500`, because the CLI sets `discover_github_provenance`. rc=1 → `pr_repo` stays None, as today. l.280 `GIT_CONFIG_GLOBAL` is fine. |
| `tests/test_runner.py` | 6 | `pytestmark` | `pr view 7` is from `deep/review_terminal.py:88` `_author_description`; failure → description None, which is not asserted. `pr list --head` is from `workspace._resolve_base` (l.454-467); rc=1 → falls back to main. Do NOT change to `base="main"`, which would change the path under test (`test_workspace.py:64` owns that contract). l.323/l.374 wrap `subprocess.run` after the fixture, which is fine. l.143 `StaticGitHubAuth({"PATH": "/usr/bin"})`: an *async* gh call under such an env bypasses a PATH shim, which is why the guard must also patch async exec. |
| `tests/test_cli_main_integration.py` | 6 | `pytestmark` | `_silence_cli_and_runner` (l.48) is a plain function, and it stubs `gh_repo_view`/`gh_pr_view` but not `/user`. The codex `bin_dir` holds only `codex`. |
| `tests/test_missing_test_command_acceptance.py` | 1 (2 tests in file) | add the `fake_gh` param to `test_runner_rejects_missing_canonical_before_fix_or_test`, or `pytestmark` | `_arrange` (l.40) has no `tmp_path`. `pr list` comes from `_resolve_base`. l.81 prepends `tmp_path` (only a `pytest` script) to PATH. |
| `tests/test_findings.py` | 1 | add the `fake_gh` param to `test_build_write_load_preserves_the_diagram_contract` (l.86) | `pr diff 7` comes from `pr_review.classify` → `pr_changed_files` (l.533-546), because base==head. FakeGh has no `pr diff` handler, so it answers "unsupported invocation" with rc=1 → `GitError` → `set()`. With `issues=[]` the result is never read. |
| `tests/test_dataset_capture.py` | 1 | add `fake_gh` to the `config` fixture (l.55), or `pytestmark` | |
| `tests/test_observability_failure_integration.py` | 4 | add `fake_gh` to the autouse `isolated_operator_environment` (l.63), or `pytestmark` | |
| `tests/test_observability_backend_protocols.py` | 1 | `pytestmark` | `install_fake_cli_process` patches async exec for osprey only. |
| `tests/test_observability_integration.py` | 1 | `pytestmark` | |
| `tests/test_artifact_visibility_integration.py` | 3 (all tests in file) | `pytestmark` | `_forbidden_path_hits` (`harness/protocol_cli.py:69`): no caller passes `forbidden_paths`, so the shim path on PATH is fine. |
| `tests/test_extension_seam_integration.py` | 3 | `pytestmark` | |
| `tests/test_extension_prompts_integration.py` | 1 | `pytestmark` | |
| `tests/deep_orchestrator/test_findings_artifacts.py`, `test_fix_gate_cleanup_and_precision.py`, `test_fix_guards_and_scope.py`, `test_pipeline_intent_and_resume.py`, `test_repair_job_acceptance.py` | 1 each | `pytestmark` per file | They patch only `git_ops.*` functions above the subprocess layer. A `tests/deep_orchestrator/conftest.py` autouse `fake_gh` is an alternative; prefer whichever is smaller. |

Production also has a needless lookup: `diagram_steps.py:745` passes `[]` issues, and `classify`
still runs `pr diff`. It is out of scope for this test-only issue.

## Validation commands (from Makefile / CONTRIBUTING.md)

- `make install`, then `make hooks`, once.
- Focused runs: `uv run pytest -q <files>`.
- Acceptance: the issue's shim repro with `uv run pytest -q -n logical`. The log must
  be empty and the suite must pass.
- Gate: `make check` (under 2 minutes; the baseline pytest takes about 63 seconds).
- Do not lower the coverage ratchet (`docs/coverage.md`).
- Test changes: apply `.agents/skills/test-audit/SKILL.md`. The PR touches more than
  5 files, so also run the `.agents/skills/daydream-deadcode-sweep/SKILL.md` PR mode.
- CHANGELOG: this is a test-only change, so either record "N/A: test-only" in the PR
  checklist or add a `Fixed` contributor note. Follow CONTRIBUTING's versioning
  section.

## Appendix: baseline shim log (count, test id, gh args)

```text
      4 tests/deep_orchestrator/test_findings_artifacts.py::test_embedded_fixing_is_selected_explicitly_and_persists_the_suite_verdict[operator-ignore-records-failure] :: api /user
      1 tests/deep_orchestrator/test_fix_gate_cleanup_and_precision.py::test_deep_run_reports_persistent_git_timeout_distinctly :: api /user
      1 tests/deep_orchestrator/test_fix_guards_and_scope.py::test_test_healing_guard_reverts_generated_migration_edit :: api /user
      4 tests/deep_orchestrator/test_pipeline_intent_and_resume.py::test_pr_body_reaches_the_reviewer_as_bounded_untrusted_context :: api /user
      1 tests/deep_orchestrator/test_repair_job_acceptance.py::test_operator_grant_resumes_the_same_job_without_erasing_history :: api /user
      1 tests/test_artifact_visibility_integration.py::test_runner_claude_sdk_cancellation_completes_disconnect_before_freeze :: api /user
      1 tests/test_artifact_visibility_integration.py::test_runner_external_adapter_cancellation_reaps_process_and_freezes_once :: api /user
      1 tests/test_artifact_visibility_integration.py::test_runner_improve_claude_strict_uses_audit_root_and_inline_sanctioned_input :: api /user
      1 tests/test_cli_main_integration.py::test_artifact_visibility_cli_codex_publication_collision_restores_and_exits_one :: api /user
      5 tests/test_cli_main_integration.py::test_cli_main_only_fix_selects_fixing :: api /user
      1 tests/test_cli_main_integration.py::test_cli_main_rejects_workspace_copy_traversal :: api /user
      1 tests/test_dataset_capture.py::test_terminal_failure_retains_owned_trace_and_partial_evidence :: api /user
      1 tests/test_extension_prompts_integration.py::test_legacy_markdown_plan_writer_override_blocks_with_sanitized_diagnostics[asyncio] :: api /user
      1 tests/test_extension_seam_integration.py::test_bound_custom_flow_rejects_public_artifact_before_backend_entry :: api /user
      2 tests/test_extension_seam_integration.py::test_extension_step_rewrites_items_before_the_review_is_sealed :: api /user
      1 tests/test_extension_seam_integration.py::test_pr_feedback_not_selectable_via_flow :: api /user
      1 tests/test_findings.py::test_build_write_load_preserves_the_diagram_contract :: pr diff 7
      1 tests/test_improve_flow.py::test_an_edited_file_left_unquoted_is_repaired_before_the_plan_lands[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_audit_dispatch_interval_preserves_isolation_when_all_failed[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_branch_cancellation_joins_provider_streams_before_snapshot_cleanup[asyncio-audit] :: api /user
      1 tests/test_improve_flow.py::test_branch_cancellation_joins_provider_streams_before_snapshot_cleanup[asyncio-vet] :: api /user
      1 tests/test_improve_flow.py::test_branch_fanout_preserves_serial_limits_and_quick_filtering[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_branch_focus_manifest_retains_deleted_paths_without_reading_missing_files[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_branch_focus_on_base_branch_reports_and_exits_cleanly[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_capable_improve_partial_plan_failure_is_safe_and_publishes_the_rest[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_custom_improve_survey_uses_compatible_two_pass_fallback[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_full_improve_denies_glob_through_inward_then_outward_symlink[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_full_improve_uses_independent_unborn_snapshot[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_group_scope_slices_search_but_report_names_the_unaudited_rest[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_improve_cli_discovers_nested_workspace_and_preserves_downstream_facts :: api /user
      1 tests/test_improve_flow.py::test_improve_cli_discovers_nested_workspace_and_preserves_downstream_facts :: repo view --json nameWithOwner -q .nameWithOwner
      1 tests/test_improve_flow.py::test_improve_git_environment_refusal_names_key_without_value[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_plan_subverb_repairs_invalid_plan_once_then_blocks_without_truncating[asyncio] :: api /user
      2 tests/test_improve_flow.py::test_publication_failures_are_not_reported_as_planning_failures[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_report_names_unaudited_partitions_and_failed_groups[asyncio] :: api /user
      2 tests/test_improve_flow.py::test_unborn_commit_anchored_improve_modes_fail_before_backend[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_unborn_improve_cancellation_cleans_snapshot[asyncio] :: api /user
      1 tests/test_improve_flow.py::test_unsupported_improve_backend_fails_atomic_preflight[asyncio] :: api /user
      2 tests/test_improve_flow.py::test_vet_dispatch_interval_batch_failure_fails_closed_per_batch[asyncio] :: api /user
      2 tests/test_missing_test_command_acceptance.py::test_runner_rejects_missing_canonical_before_fix_or_test :: api /user
      2 tests/test_missing_test_command_acceptance.py::test_runner_rejects_missing_canonical_before_fix_or_test :: pr list --head feature --state open --json number,headRefName,headRefOid,baseRefName,url,headRepository,headRepositoryOwner
      1 tests/test_observability_backend_protocols.py::test_osprey_strict_protocol_fixture_through_runner :: api /user
      1 tests/test_observability_failure_integration.py::test_runner_cleanup_outage_preserves_captured_root_and_review_output :: api /user
      1 tests/test_observability_failure_integration.py::test_runner_exact_allocation_bills_children_and_chain_matches_wire :: api /user
      1 tests/test_observability_failure_integration.py::test_runner_late_missing_metrics_bill_chain_children_stay_custom :: api /user
      1 tests/test_observability_failure_integration.py::test_runner_partial_outcome_preserves_the_budget_stop_record :: api /user
      1 tests/test_observability_integration.py::test_runner_redacts_encoded_and_decoded_url_credentials_despite_malformed_unrelated_url :: api /user
      2 tests/test_runner.py::test_artifact_session_runner_cancellation_finalizes_then_propagates :: api /user
      1 tests/test_runner.py::test_artifact_session_runner_cancellation_finalizes_then_propagates :: pr view 7 --json number,title,body,state,headRefName,baseRefName,headRefOid,url,headRepository,headRepositoryOwner
      1 tests/test_runner.py::test_artifact_session_runner_controlled_custom_flow_publishes_after_model[git-redirect-capture-worktree] :: api /user
      1 tests/test_runner.py::test_artifact_session_runner_controlled_custom_flow_publishes_after_model[none] :: api /user
      1 tests/test_runner.py::test_artifact_session_runner_controlled_custom_flow_publishes_after_model[source-missing-worktree] :: api /user
      1 tests/test_runner.py::test_artifact_session_runner_controlled_custom_flow_publishes_after_model[source-replaced] :: api /user
      1 tests/test_runner.py::test_fix_cycle_failing_tests_bounded_fix_then_handoff :: api /user
      1 tests/test_runner.py::test_fix_cycle_failing_tests_bounded_fix_then_handoff :: pr list --head feature --state open --json number,headRefName,headRefOid,baseRefName,url,headRepository,headRepositoryOwner
      1 tests/test_runner.py::test_run_rejects_head_mismatch_on_real_worktree :: api /user
      1 tests/test_runner.py::test_unborn_improve_approved_head_rejected_before_backend :: api /user
```
