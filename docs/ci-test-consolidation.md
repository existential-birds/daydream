# CI test consolidation

This focused batch follows the maintainer's request to explore redundant tests.
Its baseline is `98499566a4ebed1a550d45a1a304bf00d2ad5fea`. Three discovery lanes
covered backend/trajectory, deep flows, and archive/Improve/tooling; independent
reviewers compared every removed assertion with its keeper. Distinct transport,
deadline, cancellation, privacy, confinement, persistence, and Docker contracts
remain covered. There are no production changes in this batch.

## Transfers

Nodes below are within the named file unless a different file is shown. Every
flow keeper still enters the real runner with temporary Git/filesystem state.

| Removed or combined scenario | Remaining keeper | Preserved proof |
|---|---|---|
| `test_backend_pi.py::test_runner_real_path_pi_provider_axis` (two provider rows) | `test_runner_pi_stack_review_effort_reaches_native_submission` (five existing rows) and `test_runner_custom_flow_pi_reports_unbounded_allowance` (two rows) | Empty settings, fallback model and default/overridden provider reach native argv through the real factory/backend. Built-in and extension phases both remain covered. |
| `test_trajectory_phase_events.py::test_deep_run_accept_gate_wraps_fix_test_verify` and `test_parallel_fix_registers_subtrajectories` | `test_real_fix_fallback_records_multiple_invocations_in_one_fork` | ATIF validation; all five manifest phase timings; no alternatives phase; exactly two FIX summaries with descriptor, start/end, sibling reference and no legacy step IDs. Real host tests, push, archive and evaluation remain. |
| `deep_orchestrator/test_evidence_reuse_gates.py::test_the_retained_test_evidence_reaches_the_commit_gate` and `test_the_pre_push_reuse_decision_is_persisted_in_the_real_flow` | `test_real_flow_skips_the_pre_push_suite_run_but_still_runs_the_hook` | Finalized identity/tree binding, one host-suite execution, real hook and push, successful remote-CI gate, persisted reuse decision and post-commit verification. Its obsolete single-use helper is removed. |
| `deep_orchestrator/test_fix_failures_and_quality_gate.py::test_fix_quality_gate_persists_verbosity_regression` (two output-location rows) | Same node, one run | Live and archived quality records are equal, with retained edit, verbosity regression and warning flags asserted. |
| `deep_orchestrator/test_diagram_and_sharding.py::test_deep_large_diff_produces_review_and_record_shards` and `test_deep_sharding_off_keeps_single_agent_per_stack` | `test_resolved_sharding_controls_publish_actual_review_scopes` | Existing enabled/disabled/configuration rows assert complete actual scopes and exact published review/record filenames, including absence of shards when disabled. |
| Same file: `test_no_parse_phase_and_records_from_output_schema` | `deep_orchestrator/test_pipeline_intent_and_resume.py::test_pipeline_order` | Structured language/structural records, no parse prompt/model, nonempty first-pass requests, exact assigned hunk files and delivered full diff; ordering and merge resume remain. |
| `deep_orchestrator/test_review_reuse_path.py::test_store_directory_survives_a_fresh_run_and_is_readable_by_the_next` | `test_identical_rerun_restores_completed_units_and_current_run_evidence[all-units]` | Store publication, exploration provenance, unknown entry survival, warm exact-byte evidence and no paid review calls. |
| `test_archive_data_capture.py::test_deep_archive_excludes_preexisting_untracked_files_from_patch_and_push` | `test_default_deep_run_populates_eval_captures_patch_and_current_merge_phase_state` | Generated migration is retained in recommendation and pushed tree; pre-existing notes are absent from both and remain dirty in the source checkout. Dataset/dump/evaluation binding remains. |
| Same file: `test_dump_artifacts_copies_credentials_in_diff` (token/PEM rows) | Same node with both files in one diff | Both archived canaries, exact dump bytes, both scanner categories, no console leaks, session binding and successful reopening. Four runner calls become two. |
| `test_improve_flow.py::test_capable_improve_stub_retains_commands_and_avoids_provider_overload[asyncio]` | `test_capable_improve_stub_partial_failure_is_successful_and_safe[asyncio]` | Same 43-entry recon menu, verified commands, exact malformed-command rejection, interactive choice, successful and bounded failed plans, privacy. Separate successful planning flows remain. |

## Fixture and wait costs

`tests/harness/git_helpers.py` writes one fixture revision before staging its
explicit paths in one real `git add`. A 520-file before/after control, including
base support merges, omitted base files, a spaced filename and untracked notes,
produced identical Git trees, diffs and status. Local setup fell from 18.783 to
1.104 seconds. This is fixture evidence, not a full-suite timing claim.

`test_git_ops.py::test_resolve_diff_merge_base_redacts_external_git_timeout`
still executes the actual external shim and production three-attempt retry loop.
Only its intentionally hung probe deadline changes from five seconds to 0.5;
three persisted entry markers prove each child reaches its sleeping branch.
Message/path/SHA redaction remains asserted.

Both judge timeout tests retain actual read/post-EOF deadlines, cancellation,
three attempts and six kills. Their existing asyncio proxy accelerates only
retry backoff and now explicitly checks the one- and two-second requests.

## Preservation controls and validation

Independent isolated Python 3.14 checkouts deliberately broke production owners.
Keepers failed at the intended assertions for: wrong Pi provider/default model;
built-in-only model/provider overrides; legacy FIX step IDs; missing test timing;
missing post-commit evidence; missing archived quality record; wrong shard
filename; omitted generated migration; missing API-key/PEM detection; and omitted
malformed-command diagnostics. Every mutated file was restored byte-for-byte,
verified by SHA-256, and its keeper passed afterward.

Review found and repaired a built-in Pi coverage gap before landing. The evidence
keeper's old fixture pushed successfully but silently returned failure because
it lacked remote-CI identity. It now asserts successful completion through the
no-CI harness, preserving that harness's actual pre-push hook and SHA seeding.

Collection decreases from 7,116 to 7,103 cases (13 net). Test files remove 183
net lines; shared support adds nine net lines. No production code, exclusions or
coverage floor are changed. The initial 891-case owner/consumer batch had one
fixture failure; after repair, 165 affected checks passed with one skip.
Independent reviewer selections and all effective restored controls passed;
the [control records](evidence/ci-test-consolidation.json) retain their outcomes
and byte-restoration hashes.

The complete `make check` passed on Python 3.14.3: 7,089 tests, 14 skips,
90.63% branch coverage, and all lint/type/dead-code/workflow/naming gates.
Local timing under concurrent exploration is not a CI comparison. Matched
Blacksmith timing remains in progress; this batch does not yet establish the
50% spend target.
