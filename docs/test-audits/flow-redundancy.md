# Flow test redundancy audit

Pinned merged main: `6c48d368d529a89748c93e7bb61ae9ac6566936e`.
Branch: `test-audit/flow-redundancy`.

This batch consolidates repeated successful runner executions and matching
contract assertions. Additional exact evidence is in
[Shared-run test consolidations](shared-run-consolidations.md). It is a focused
flow audit, not a completed redundancy campaign over every Daydream test. No
quota determined the deletions. Unique assertion blocks moved before their old
functions were removed; distinct faults, transports and configurations remain.

## Baseline and environment

Collection: **9,004 cases / 5,431 distinct test nodes without parameter suffixes**.
The requested files contain 669 cases: Improve 148/121 declarations, deep
479/303 declarations, integration 42/31 declarations.

- Python 3.12.13, macOS 26.6.1 arm64, four pytest workers.
- `uv lock --check` passed before dependency synchronization; all extras installed.
- Lock SHA256: `e035f70c23015967fb6d69d5f09dac7d77081e8cf943e60e0c41bd58680da5e1`.
- Existing branch coverage, source, exclusions and 86% floor remain unchanged.
- Original full baseline: 8,990 passed, 14 skipped, 73 warnings;
  90.55% integrated coverage; pytest 1,041.49s / wall 1,042.56s.
- Baseline hosted CI run `37543672849`: check 594s, tests 511.41s;
  8,999 passed / 5 skipped / 90.53%; RL check 179s. The four-minute
  check-job target was not met on main.

Original baseline timing includes host-authenticated GitHub identity requests.
A matched focused comparison uses absent GH credentials and an empty config root
on both baseline and candidate, the same selections, four workers, Python,
locked dependencies and machine, and no coverage on either focused run. Its
result must be distinguished from the combined harness/consolidation change.

## Exact coverage transfers

Names below cover the initial flow portion: all 23 removed nodes are single
cases. This portion deletes no parameter rows. Keeper names remain stable. The
companion report covers the subsequent broader matching-input consolidations.

### Improve — `tests/test_improve_flow.py`

| Removed declaration | Named keeper | Regression preserved |
| --- | --- | --- |
| `test_improve_recon_writes_artifacts_and_never_mutates_source` | `test_improve_timing_completeness_preserves_p09_audit_isolation` | Service report, root flow/phase labels, run markers in report/index/plans, diagnostic provenance and unchanged source. |
| `test_trajectory_records_improve_flow_and_phases` | same timing keeper | Root/child labels, nested recon/survey event order and exact successful dispatch counts/children. |
| `test_improve_model_calls_run_in_audit_worktree_not_target` | same timing keeper | One outside-source snapshot cwd, no source ancestor/descendant overlap, snapshot cleanup and unchanged Git/config state. |
| `test_improve_run_leaves_no_stray_audit_worktree` | same timing keeper | Real `git worktree list` contains no registered audit worktree. |
| `test_every_agent_call_in_every_mode_is_read_only` | timing keeper and `test_plan_subverb_skips_audit_and_writes_single_plan` | Nonempty, read-only calls through both actual full and description flows. |
| `test_n_selected_findings_produce_n_plans_first_attempt` | `test_a_finding_audited_by_several_stack_groups_yields_one_plan` | Exactly three first-attempt plans, concrete shell command expansion, expected-success prose, TODO count and no blocked index rows. |
| `test_generalist_fallback_audits_and_plans_with_no_stack_skills` | same multigroup keeper | Generic/Python/React groups plus actual plan/command/index outcomes. |
| `test_recon_prompt_names_audited_subtrees_for_per_service_commands` | `test_audit_fans_out_per_partition_group_on_scaled_monorepo` | Actual recon request: subtree commands, scope, single ordered untrusted boundary and snapshot-anchored service paths. |
| `test_clean_full_coverage_reports_nothing_skipped` | `test_run_with_no_findings_writes_report_and_empty_plan_diagnostics` | All four partitions audited, no omissions, standard-tier explanation, no writers and empty diagnostics. |
| `test_vet_rejects_unconfirmed_finding_with_reason_and_persists` | `test_previously_rejected_finding_is_not_revetted_or_rereported` | Vetted exclusion and durable title/reason immediately after the first run, followed by actual second-run replay suppression. |
| `test_report_orders_by_leverage_without_non_actionable_direction_section` | `test_non_interactive_run_selects_top_findings_and_writes_plans` | Leverage order, cleanup pressure, omission reporting and absent obsolete direction section. |
| `test_plan_writer_is_told_to_leave_the_executor_no_decisions` | `test_rendered_plan_gives_a_literal_executor_no_room_to_guess` | Complete actual planner prompt/schema constraints alongside complete rendered executor artifact checks. |

Matching inputs were verified through the full tests, shared fixtures,
ImproveStubBackend and production callers. Three-plan runs all use the same
`n_findings=3` and Phantom veto. The old generalist test's empty registry versus
keeper's absent registry has no production reader at this baseline: builtin
Registry stack rules determine routing. Report caps eight and nine produce the
same payloads from exactly eight categories. Description text does not control
read-only flags. One-shot/session tests were retained because their payload
cardinality differs; repo-scan helper tests and all schema/fault variants remain.

Owners: runner audit workspace/registry dispatch; Improve recon, audit/vet,
planning/authoring, plan reservation/publication, reporting; trajectory dispatch
recording. Relevant history: #291 core plans, #560 audit worktree, #1153 independent
snapshot confinement, #1158 truthful phase timing. Real temporary Git, filesystem,
loop and production stores remain; only model/GitHub boundaries are scripted.

### Deep — `tests/deep_orchestrator/`

| Removed `file::declaration` | Keeper | Regression preserved |
| --- | --- | --- |
| `test_pipeline_intent_and_resume.py::test_deep_run_writes_hunk_index_after_diff` | pipeline `test_pipeline_order` | Index/patch existence, index mtime ordering, loaded nonempty index and changed-file membership. |
| `test_findings_artifacts.py::test_both_ttt_artifacts_written_on_the_happy_path` | same pipeline keeper | Nonblank intent and actual serialized empty alternatives. |
| `test_pipeline_intent_and_resume.py::test_resume_overwrites` | same file `test_resume_per_stack_reruns_all` | Stale markdown input before real per-stack resume and replacement afterward. |
| `test_record_identity_and_retained_tree.py::test_every_shipped_item_carries_a_unique_item_uid_on_a_multi_stack_run` | same file `test_every_merged_item_carries_source_uids_on_a_multi_stack_run` | Multiple-item negative control, nonempty durable handles, uniqueness and fresh minted ordinals. |
| `test_record_identity_and_retained_tree.py::test_shipped_item_carries_id_item_uid_and_provenance_independently` | same provenance keeper | Dense display ids, structural record identity, separate item namespace/provenance and integer display id. |
| `test_diagram_and_sharding.py::test_per_stack_prompt_points_at_hunk_index` | same file `test_no_parse_phase_and_records_from_output_schema` | Actual reviewer changed-line authority/transport and actual schema-produced language/structural records. |
| `test_review_merge_and_verifier.py::test_merge_prompt_lists_records_in_sorted_order` | pipeline `test_pipeline_order` | Actual merge pointer block exists, contains paths and is sorted. |
| `test_findings_artifacts.py::test_fresh_run_discards_stale_deep_artifacts_before_writing_its_diff_key` | same pipeline keeper | Real stale-file input removed before new diff-key certification. |
| `test_findings_artifacts.py::test_start_at_merge_proceeds_when_the_diff_is_unchanged` | same pipeline keeper | Real second merge resume from generated artifacts, after all first-run assertions, with a fresh backend and its own observed calls. |

Default multi-stack fixture, config and backend inputs match. Identity tests all
used the identical `_fresh_uid_run`; schema/prompt tests used the same model
capturing factory. Stale inputs moved intact. The generated-artifact positive
resume still executes; only its repeated fresh run disappears. Negative freshness,
malformed attribution, arbiter, cancellation, sharding/diagram, fix and publication
fault rows are untouched.

Owners: deep preamble/freshness and coverage restore, per-stack writes,
hunk-index consumers, merge prompt construction, findings normalization and host
record/item identities. Histories: #45 pipeline/resume/order, #299 artifact writers
and freshness, #745 persisted index/schema output, #1111/#1119 identity separation,
#1432 folded alternatives, #1440 typed snapshot-bound publication.

### Integration — `tests/test_integration.py`

| Removed declaration | Keeper | Regression preserved |
| --- | --- | --- |
| `test_run_comment_full_flow` | `test_run_comment_resolves_pr_through_real_cli_boundary[branch]` and `[explicit]` | Actual noninteractive POST plus nonempty canonical items/file, submitted path ownership and real diff artifact. |
| `test_run_comment_missing_pr_exits_nonzero` | `test_run_comment_pr_lookup_failure_and_absence_exit_nonzero[absence-branch]` | Same real two-commit input, production empty PR-list lookup, exit 1, correct warning and no POST. |

Real publication replaces the older fake-post argument assertion. Canonical
artifact checks moved before removal. The missing-PR keeper reaches the same
`NO_PR` stop branch through actual transport rather than a mocked lookup. Owners:
runner/deep posting, PR resolution/placement and GitHubReviewTransport. Histories:
#75 auto-post, #330 deliverable failure, #1149 compatible explicit/default context.
Agent-suite fixing, head versus merge CI, interrupts versus task cancellation,
redaction ordering and eight lookup failure/mode rows remain.

## Shared harness repair

Root fixtures isolated App credentials but ordinary runner identity resolution
could still inherit host gh tokens/config. A new autouse fixture hides four token
variables plus GH_HOST/GH_REPO and uses per-test GH_CONFIG_DIR. Test-provided auth
and exact static/refresh environments remain available; monkeypatch restores the
host at teardown. The installed executable, process policy and auth owners remain
real. Existing `live_gh` opt-ins are exempt, and the smoke file now declares its
already registered marker while retaining its original skip gate.

`test_git_ops.py::test_gh_api_raises_without_auth` now reaches `/user` and requires
`gh auth login`, replacing a relative-repository failure at the wrong guard.
Control: identical strengthened test with old fixtures failed `DID NOT RAISE
GitError`; candidate passed. No production mutation was needed. Explicit App,
overlapping sessions, inherited/static/refresh sync/async, PATH gh identity scripts,
local no-remote guards and opt-in preflight keepers remain independent.

The prior stable-empty-CI discovery acceleration and lightweight fake-gh imports
are preserved. Credential isolation does not supply a fake success or replace a
publication/store pipeline.

## Preservation and validation

Independent read-only reviewers compared complete deleted bodies, fixture chains,
owners/callers/history and final keeper bytes. All 23 removals approved without a
lost-contract gap; every other retained function AST/parameter table in those
files is unchanged. Reviewed candidate SHA256s matched the applied files. Separate
independent auth/live-boundary review also approved. No restored or uncertain
transferred contract required a production mutation.

Final source preservation is independently approved for all portions. The
ordinary pre-push make check supplies the final integrated gate, and the PR
records its actual results plus hosted CI. CI runner size, coverage settings
and hooks are unchanged.

## Direct CLI no-CI consumer

`tests/test_cli.py::test_explicit_review_argv_uses_target_remote_ci_verdict_drives_exit[no_ci-0]`
was the slowest baseline case at 35.19s. It used the seeder directly, so it had
missed the previous NoCIRemote acceleration. It now reuses that unchanged harness
clock after stable exact-head empty evidence. Both rows, real CLI/fix/push/hook,
external subprocess requests, target cwd, verdict/output/handoff checks and all
limits remain. New artifact assertions require matching pushed/evidence SHA,
required stable polls and elapsed discovery deadline. The failed row does not
install the wrapper.

Unaccelerated keepers in `tests/test_remote_ci.py` retain discovery deadline plus
stable-poll classification, explicit start/budget deadlines, later real request
timeout from a trusted snapshot, delayed policy/head changes and actual process
group cancellation. Integration cancellation/interrupt rows and harness seeder
fault/handshake tests remain. No archive assertion is claimed for this CLI node.

## Assignment batches

The installed xdist load scheduler initially assigned 562 consecutive cases to
each of four workers. Matching all baseline nodes to JUnit durations found initial
batch costs of 1,030.45s / 85.33s / 118.20s / 122.93s, with no unmatched cases.
The first batch alone nearly consumed the full 1,042.56s wall time. The original
scheduler does not steal already assigned cases.

The initial root pytest addopts retained strict markers and added `--maxschedchunk=10`, limiting
each assignment batch while retaining four workers, the entire collection, normal
fixture/teardown protocols and all coverage settings. This is not a hard total
queue-length limit: fast completions can accumulate several batches. Independent
review inspected the locked xdist 3.8.0 scheduler, option parser and worker
protocol. Module/session fixtures may be instantiated on more workers; their
cost and all retained contracts must be checked by measurement and the full gate.

Matched focused default-scheduler result: baseline 1,399 passed / 4 skipped,
342.71s wall; consolidated candidate 1,376 passed / 4 skipped, 335.90s wall.
This single comparison measured 6.81s (1.99%) savings, not a statistical estimate
of sustained CI improvement. The exact CLI no-CI row including fixtures went
from 34.927s to 4.884s. All 89 unaccelerated remote-CI controls passed in 3.30s.
The initial scheduled full gate passed: 8,967 passed/14 skipped/73 warnings,
90.55% integrated coverage, 642.62s pytest/648.57s wall for make check. The raw
pytest comparison is 1,041.49s ->642.62s (38.30%). This includes scheduler, auth
isolation and harness changes; it does not isolate consolidation savings. Final
broader results and hosted CI are required before claiming the target.

## Additional validation and size accounting

- Installed-gh live preflight explicitly passed with `DAYDREAM_LIVE_GH=1`, proving
  the marker exemption preserves real authenticated CLI/credential-helper use.
  The second live import needs operator-supplied PR numbers; it was not enabled.
- Ruff passed; mypy passed over 723 files; root and RL vulture scans passed;
  root and RL lockchecks and `git diff --check` passed.
- Initial flow collection was 8,981 cases / 5,408 functions: exactly 23 removed,
  zero added and no parameter rows lost. Scheduled collection is identical.
- Production/workflow changes: zero. Tests: +190/-401 lines (net -211).
  Shared support: +14. Pytest configuration: +3/-1 (net +2).
  This report is separate documentation. Baseline Python lines were production
  105,189, tests 117,060 and support 7,950; corresponding initial-flow counts were
  105,189 / 116,849 / 7,964, using the same tracked-file categories.
- Full baseline skips: four opt-in live cases, nine platform/filesystem cases,
  and one unavailable prime-rl workspace. Those prerequisites are independent
  of pruning. The same focused comparison skipped four cases on both sides.

## Broader requested count reduction

Final collection after the shared-run batch and auth isolation: **8,199 cases / 5,361 functions**.
This removes 70 functions (1.29%) and 805 net cases (8.94%). 721 case wrappers
belonged to one repository logging scan; all722 sources and every AST rule remain
covered by its same named keeper. The remaining 84 net cases are exact matching
run/assertion consolidations or private organization/self-identity checks.

The records/runtime/tooling inventories covered 66/33/52 associated files and
1,171/1,095/2,008 cases respectively (associated files overlap). Candidate blocks
received complete source/fixture/owner/caller/history and independent preservation
review. Other inventory entries were conservatively retained or remain unassessed;
this is not an exhaustive completed review of all 9,004 original rows. The user's
30% hypothesis did not produce a deletion quota. Distinct failure evidence,
transport/security/persistence/package contracts and real paths were retained.

See the companion ledger for every additional affected declaration/row, exact
matching keeper, assertion transfer, owner/history, retained false positives and
isolated behavioral mutation results. Final validation metrics and CI conclusions
are reported in the PR; do not interpret the initial full gate as the final gate
after these subsequent edits.

Final size accounting against the same baseline categories: production 0 lines
changed; tests +467/-960 (net -493); shared support +14; pytest configuration +3/-1
(net +2). Python totals: production 105,189, tests 116,567, support 7,964. Evidence
documentation is separate. The same functions and sources remain in production;
no test-only production export, injection flag or seam was introduced.

Matched expanded owner/shared-consumer comparison (same 46 selections, n4,
maxschedchunk10, no coverage on either, absent GH auth, same Python/lock/machine):
baseline 2,781 passed/4 skipped/20 warnings, 112.98s pytest/113.51s wall;
candidate 1,995 passed/4 skipped/20 warnings, 87.54s pytest/88.02s wall.
Observed wall saving 25.49s (22.46%) in this single pair. This selection includes
logging wrappers and the direct CLI clock change; it does not isolate the effect
of function deletions or establish a sustained hosted CI result. The original
focused comparison isolated the earlier flow batch before scheduling changes.

Final independent source reviews approve flow/harness, runtime, records and
tooling changes. One Ruff line-length finding was corrected with an AST-identical
line wrap; Ruff, mypy (723 files), root/RL vulture, root/RL locks and diffcheck pass.
The full integrated gate is the ordinary pre-push make check. The PR records its
actual passed/skipped/coverage/time result and subsequent hosted CI, including
whether the under-four-minute check-job objective is achieved.

## Hosted auth guidance repair

CI 37553131563 (same 4-vCPU class and actual root Python 3.14.8 as the hosted
baseline) completed root job 302s/tests 262.91s: 1 failed, 8,191 passed, 5 skipped;
RL passed 151s. The four-minute target was not met. Sole failure was
`test_git_ops.py::test_gh_api_raises_without_auth`: installed gh correctly rejects
/user in Actions with `set the GH_TOKEN environment variable`, while the new
matcher expected local `gh auth login` wording.

Exact red reproduction on the local installed CLI:
`GITHUB_ACTIONS=true uv run --no-sync python -m pytest tests/test_git_ops.py::test_gh_api_raises_without_auth -q`.
It failed with the identical Actions guidance. Before editing, an independent
review approved preserving the same real auth guard as explicit `[local]` and
`[actions]` rows: remove/set GITHUB_ACTIONS and require the corresponding exact
guidance. Both repaired rows passed in 0.61s, and final independent review confirms
only this declaration changed. No token/config/skip relaxation or dummy auth was
introduced. This justified additional mode case changes final collection to
8,198 cases/5,361 functions: 70 functions and 806 net cases removed. The original
local guard and new CI-context failure mode are both retained.

The ordinary pre-push full gate's 90.54% versus the earlier direct-shell
candidate's 90.55% differed only at git_ops/snapshot.py:118, the absent-GIT_EXEC_PATH
return. Hooks export that variable; direct shell does not. Snapshot security tests,
clean child environment and real pre-push default-helper proof remain unchanged.
This is 0.0031 percentage points of ambient hook/shell branch difference, not a
removed contract or changed coverage setting. New auth repair gets its own focused
Git/auth/shared-consumer and full normal hook validation before landing.

## Refined dispatch and hosted timing evidence

The follow-up experiment used `--maxschedchunk=1 --durations=40`, retaining
strict markers, workers, complete collection, dependencies and coverage settings.
Independent review inspected the locked xdist 3.8.0 scheduler, parser, worker
lookahead/completion/shutdown and restart protocol before editing. With the full
collection, initial dispatch still sends two cases. In ordinary completion-driven
scheduling each finished item is removed and at most one is appended, retaining
at most two queued cases. Survivor queues can grow on crash/restart refill passes;
this is not a hard global queue-length guarantee after restart. Running cases
are never moved; the original dispatch, fixture and teardown protocols remain.

Ten-case dispatch can append further batches after sub-0.1s completions. Whether
this caused the hosted 262.91s test runtime is an uninstrumented hypothesis. Local
full 408.62s was already close to its JUnit summed case 1572.08/4=393.02s lower bound;
no large local improvement is predicted. Additional messages/module setup can
cost time. The built-in 40-duration summary records existing reports, not queue
depths, for actual hosted owner timing evidence. No assertion/row/fixture is lost.

Matching Git/auth consumers in Actions mode: 10-case dispatch 447 passed/4 skipped,
22.46s; 1-case dispatch 447/4,22.26s. Same selection, n4, Python/lock/machine and no
coverage on either. Difference 0.20s is not a material performance claim. Full
normal hook gate and actual hosted CI decide preservation and target throughput.
At that experimental revision the root collector had 8,198 cases/5,361 functions. The earlier matched
expanded comparison intentionally used 10-case dispatch on both historical sides.


## Automation environment repair and scheduling rollback

Before editing, exact CI-context reproduction showed the `[local]` row still
inherited `CI=true` after removing `GITHUB_ACTIONS`. Real gh correctly rejected
`/user`, but emitted automation guidance. The existing `[local]` and `[actions]`
nodes now explicitly clear both flags; a new `[automation]` row sets only `CI`,
while Actions sets both flags to prove precedence. Each requires its specific
context guidance, and both CI modes require the full GH_TOKEN instruction.
The installed CLI, real temporary repo, isolated absent credentials/config and
production request remain unchanged. Independent pre-edit and final preservation
reviews approved this repair. With ambient CI and Actions both true, all three
rows passed (1.93s); Git/auth/shared consumers passed 584 cases with four skips
(53.15s). No skip or credential relaxation was introduced.

The one-item dispatch experiment was reverted to ten-item batches; the duration
summary remains. Matching local full hook runs were 408.62s at ten versus 430.85s
at one, so no speed benefit was established. Hosted CI 37557357315 took 371.07s
for tests / 410s for the check job and failed only the local auth matcher. Its
actual interpreter was Python 3.12.14, while the first candidate and baseline
used 3.14.8. Earlier setup-python supplied 3.12.12 (below the project's 3.12.13
floor) and uv downloaded 3.14.8; later setup supplied qualifying 3.12.14, which
uv used directly. These logs explain the interpreter drift; they do not establish
how much runtime it caused. Hosted times across that change are not a controlled
scheduler comparison. Runner size, Python workflow declaration, dependencies and
coverage settings remain unchanged. The under-four-minute objective remains
unmet; final green CI must be reported independently of that objective.


## HTTP judge retry waiting: before-edit preservation evidence

Three existing nodes in `tests/test_benchmark_verifier_judge.py` remain intact:
`test_retry_policy_retries_transport_and_5xx_then_fails_after_exhaustion`,
`test_retry_policy_retries_openrouter_error_envelope`, and
`test_both_providers_produce_identical_verdicts_and_errors`. They exercise actual
shipped HTTP retry/parser/provider logic over immediate fake network responses:
two transport failures then success, three 503 failures, HTTP200/OpenRouter502
recovery, and each provider's native envelope/exhaustion. No run, row or input is
removed. Full tests, fixture loader, shipped owner/callers and histories
c03ac839 (#777/#804), df414f47 (#917), 364134be (#965/#978) were inspected, then the
bounded plan received independent preservation review before editing.

Only a scoped module-binding proxy records HTTP backoff and yields through the
captured real `asyncio.sleep(0)`. It does not mutate the shared asyncio module.
Each separate outcome checks literal requested delays `[1, 2]`; successful first
requests require none. Existing attempts/verdict/error checks remain. The parity
context exits before its native CLI cases. Named independent timing keepers
`test_claude_cli_timeout_kills_child_every_attempt`,
`test_claude_cli_post_eof_wait_timeout_kills_child`,
`test_claude_cli_post_eof_wait_settles`,
`test_claude_cli_streaming_oversize_kills_child`, and
`test_judge_pairs_caps_concurrency_and_enforces_pair_cap` keep real deadlines,
backoff, child cleanup and overlapping requests unchanged. Packaged source,
assets/isolation and real image execution contracts remain unchanged.

Before timing: same three nodes, n4/ten-case dispatch/Python3.12.13/locked
dependencies/machine, no coverage: three passed, 6.64s pytest/7.06s wall;
calls6.01+6.01+3.00s. Concurrent waits mean 15 seconds of worker time is not a
15-second suite-wall saving. After timing and shared-consumer/full-gate evidence
are recorded in the PR. This adjustment alone cannot establish under-four-minute
CI.


Final HTTP timing used exactly the same selection/configuration: three passed,
0.68s pytest/1.24s wall, versus 6.64s/7.06s before (single pair). Independent final
AST review confirmed all original stimuli/assertions survive and every other test
and packaged byte is unchanged. A control changed only shipped HTTP backoff from
`2**attempt` to `2**attempt + 1` in an isolated checkout: all three keepers failed
at the new delay/order guards; exact owner restoration passed all three. Test and
owner bytes were restored and the control checkout diff was empty. Shared
verifier/judge/calibration/assets/isolation/package consumers passed 154 cases
with one warning in 7.15s, including actual Docker images. Both retained native
CLI timeout cases still paid real backoff/deadlines (3.15s and 3.16s). Final
collection is 8,199 cases/5,361 functions; this waiting change adds/removes none.
