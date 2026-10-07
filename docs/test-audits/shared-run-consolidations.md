# Shared-run test consolidations

Baseline: `6c48d368d529a89748c93e7bb61ae9ac6566936e`.
This supplements [the flow audit](flow-redundancy.md) with exact evidence recorded
before editing, then independently reviewed and applied. It covers a focused
batch of matching-input consolidations.
The inventories were broad; only candidates whose complete tests, fixtures,
owners, callers and relevant history were reviewed were edited. Unassessed and
uncertain cases remain. No percentage quota determined deletion.

Final collection is 8,197 cases / 5,361 functions, versus 9,004 / 5,431:
70 functions and 807 net cases removed. Of the case reduction, 721 wrappers
belong to the repository logging scan; its one retained test still inspects all
722 original sources with every original rule. The other net 86 cases combine
matching outcomes or retire private organization/self-identity assertions.
Count reduction alone is not a performance result.

All production and packaged workflow bytes are unchanged. Real CLI/runner,
temporary Git/filesystem/event-loop, transport, faults, security, cancellation,
persistence and artifact ownership tests remain. Local dump helper strengthening
keeps its None default; shared support changes are described in the flow report.

Independent preservation review required two clarifications, implemented before
validation: exact retry equality (rather than approximate equality), and checking
fresh init file mode before recovery-capable workspace readers. All transferred
assertions and unchanged retained ASTs were compared against final bytes. The
retry keeper rejected an isolated 5.0000001 result while the old approximate
keeper accepted it; restored production passed. Suppressing actual archive dump
copy made the real keeper fail on missing manifest.json; restored production
passed. Four isolated logging source mutations (Name, Attribute, lazy import,
prefix) failed at intended file:line guards; exact restoration passed. None of
these production mutations is in the branch.

Evidence corrections from independent review: F1 dump history is 725ae82a (#258);
E1 count-reader history is b74a9b5d (#741/#761), matrix history 7588c8a7 (#1106/#1116).
The native Pi generation fixture contains assistant/tool events, no user message;
its retired name overstated user proof. Runtime C5 now uses exact equality and
carries the negative row plus warning knob identity. Tooling B6 executes
init -> immediate gitignore bytes/mode -> validation -> status.

## Records, archive, trajectory and corpus

## Candidate evidence before edits

### P1 — process traces from the same frozen store, flag enabled

Affected:
- `tests/test_corpus_projection_process_traces.py::test_flag_on_records_carry_identity_lineage_and_distinct_ids` (1case)
- `tests/test_corpus_projection_process_traces.py::test_flag_on_split_files_contain_derived_records` (1case)

Keeper: `tests/test_corpus_projection_process_traces.py::test_flag_on_emits_process_trace_and_task_only_records`.

All three call `_build(tmp_path, emit_process_traces=True)` with exactly the same seeded dispositions accepted/ambiguous, immutable run sess-a, actual LocalRecordStore writes, snapshot selection, MIT license policy and default split rates/salt. Production owner `training/corpus_projection/projector.py::build_frozen_corpus` reads store.read_snapshot and sessions_from_snapshot, projects first segment, derives process-trace/task-only IDs and copies lineage, then writes corpus and all split files. `tests/harness/record_projection.py` supplies actual validated records through commit_run/append_observation, never prewrites output files. Consumer entrypoint is `commands/corpus.py` via corpus build; load_v2_projection/coordinator consume emitted package files. The flag and package schema are independent shipped contracts: retain every assertion.

Regressions: missing derived types/tier/no-outcome labels; duplicate host-derived record IDs; missing schema/run/license/split identity; non-decisive record wrongly counted excluded; adjudication report dropped; split files omit derived records. Transfer the entire identity/lineage block and complete split membership/count assertions into the keeper after its one build, reusing already-read corpus records. Do not replace split filenames with producer-derived expected names. Keep every default/explicit-disabled and posterior-leak/admission refusal path independently.

History: latest file owner9328e0d0 (#1493), unified record-based training; retain source-bound raw evidence improvements from6c48d368. No support/production deletion unlocked. Low risk after complete transfers. Focus `uv run pytest tests/test_corpus_projection_process_traces.py tests/test_corpus_projection.py tests/test_corpus_projection_reproducibility.py tests/test_training_coordinator_projection.py tests/test_record_dataset_journey.py` after lockcheck and after running checkout tests stop.

### P2 — default flag-disabled projection and artifact contract share first build

Affected `tests/test_corpus_projection_process_traces.py::test_flag_off_emits_only_outcome_finding_records` (1case). Keeper `::test_flag_off_is_the_default`.

The keeper's first `_build(tmp_path)` is identical to the retired test and its second `_build(tmp_path/'b', emit_process_traces=False)` proves default/explicit parity. Move records nonempty/all-outcome and summary.records_by_type exact set into keeper before the second build. Regressions: default option produces non-outcome records while identical explicit false happens also to produce those wrong records; parity alone cannot detect it, so transfer the assertions. Owner/fixtures/callers/history/validation P1; no additional run or new mocks.

### A1 — default absent-value archive manifest

Affected singleton nodes in tests/test_archive.py:
- `test_build_manifest_without_evaluation`
- `test_manifest_fix_quality_gate_none_when_absent`
- `test_manifest_recommended_patch_capture_defaults_pre_test`
- `test_build_manifest_omits_unresolved_profile_identity`

Keeper `tests/test_archive.py::test_manifest_to_dict_structure`. All invoke `_build(tmp_path)` with identical default immutable snapshot/root metrics, default identity(profileNone, phases.fixTrue), GitContext(), no optional keyword arguments. The unresolved-profile node directly calls `_build(...).to_dict()`; others assign m. Production owner archive/manifest.py::build_manifest_from_snapshot constructs missing metrics asNone, profile absent, gateNone, and pre_test capture only when fix-capable/non-feedback; Manifest.to_dict preserves serialized omission versus null distinctions. Public consumers are finalize_archive_run, index.upsert_run, dataset frozen archive capture and CLI archive dump. No input file system state differs.

Regression mapping and transfers: no-evaluation total_findings/cost_per_finding/erosion/verbosityNone and obsolete grounding/coverage metrics absent; no-gate property and serialized fieldNone; recommended_capture property and serialized stringpre_test; all four profile identity keys absent. Carry these into structure keeper using its m/d before deletion. Keep explicit-evaluation, profile-present, diagram, feedback, captureNone/explicitpost_test, null-in-hunk and source/Git metrics variants, and all real SQL rows. These are wire-format contracts, not low-value dataclass assertions to discard.

History:4c9a96a53 initial archive/trajectory tests, abd521e87 manifest enrichment,9328e0d0 current schema2.0. Shared `_build` helpers remain needed by many tests. Low risk. Validate tests/test_archive.py, tests/test_archive_integration.py, tests/test_archive_data_capture.py and test_record_dataset_journey.py.

### A2 — normal bundle file outputs from one assembler

Affected `tests/test_archive.py::test_bundle_file_path[review-output]`, `::test_bundle_file_path[diff-patch]` (2rows in1declaration), and `::test_bundle_findings_artifact_skipped_without_route` (1case). Keeper `::test_bundle_deep_directory`.

Every selected case calls exactly `_setup_bundle(tmp_path)` then `_assemble_bundle(target, run_dir, recorder)` with default NORMAL flow, no destinations, same frozen root and source artifacts. Parametrized relative_path/expected are used only in assertions; they do not change setup or production execution. Add exact checks for review-output.md='review findings', diff.patch='diff content', existing deep/intent.md='intent', and findings.json absence to keeper. All remain independent artifact bytes/ownership assertions.

Production `archive/bundle.py::_copy_snapshot_bundle` first projects validated frozen trajectory bytes then `_copy_run_artifacts` separately copies deep tree, review, diff, recommendation and explicit findings route. Caller finalize_archive_run constructs final bundle; external dump and data capture consume it. Source fixture writes review/deep/diff input, not destination artifacts. Strong real-run archive owner keepers remain tests/test_archive_integration.py; malformed sibling/session, missing files, diagram stale artifacts, explicit findings route and freeze/live divergence stay retained with their own inputs/faults. History archive helper tests4c9a96a53, latest snapshot rewrite0d24691e/9328e0d0. No support/production deletion. Low risk. Same archive command A1.

### T1 — clean root finalization assertions on identical event stream

Affected `tests/test_trajectory.py::test_recorder_does_not_mark_partial_on_clean_exit` (1case). Keeper `::test_recorder_writes_schema_valid_trajectory_on_clean_exit`.

Both use conftest.recorder(make_recorder(tmp_path)), enter one REVIEW invocation, observe user prompthello and observe_text_and_result(world), then leave without exception. Keeper already checks disk existence and ATIF validity. Transfer absent extra.partial assertion after reading its persisted payload. Detects accidental partial flag on normal finalization; schema validity alone permits that flag. Production recorder.__aexit__ marks _aborted only on actual exc_type, _prepare_document/_build_trajectory serializepartial; shared scopes invokes invocation.finish. Runner._open_recorder and run_agent own real lifecycle; their integration coverage remains untouched. History clean recorder4c9a96a53 versus partial-state6b43c5ea5. No support/API deletions. Low risk; validate tests/test_trajectory.py, tests/test_agent_recorder_integration.py, tests/test_trajectory_generation_lifecycle.py, tests/test_trajectory_phase_events.py.

### T2 — child/run identity persisted through same fork

Affected `tests/test_trajectory.py::test_sibling_inherits_session_id` (1case). Keeper `::test_fork_child_trajectory_id_distinct_from_root`.

Both default recorder(make_recorder), fork fix-0 with one FIX invocation and default output, then one root REVIEW invocation/default output. Fixture recorder versus manually constructed default has same parameters. Keeper already validates childATIF, child.session_id==recorder.session_id, exact qualified trajectory_id, root sibling refs and invocation trajectory identity/path. Transfer `parent_traj['session_id']==child_traj['session_id']` into keeper. Detects root/file identity mismatch not covered merely by child identity. Owner scopes.fork_scope copies parent.session_id; recorder._build_trajectory and lifecycle refs project it. Real runners/deep dispatch and shared dispatch-child proof retained. History fork scope initial4c9a96a53; ATIF per-document upgrade0fa3ca00/current scope refactor. No support deletion; low risk. T1 commands.

### T3 — no active recorder already a lifecycle control

Affected `tests/test_trajectory.py::test_no_recorder_no_op_get_current_returns_none` (1case). Keeper `::test_context_var_set_inside_and_cleared_after`.

Both start under autouse ContextVar reset and immediately assert get_current_recorder()isNone. Keeper also proves inside binding and outside reset using real persisted invocation. Exact initial-state assertion already exists; delete redundant singleton with no transfer. Owner trajectory/context.py ContextVar(defaultNone); recorder enter/set and exit/reset. Preserve fork/context cancellation/fault scopes. History 4c9a96a53. No support deletion; low risk. T1 commands.

### R1 — rubric output version and absent intrinsic term from same score

Affected `tests/test_training_rubric.py::test_breakdown_stamps_rubric_version` (1case). Keeper `::test_missing_correctness_is_none_not_zero`.

Both call identical score_review(model, findings=[_finding()], fp_count=0,total_findings=1,breakdown=True). model fixture deterministic external OutcomeModel protocol boundary(score_comment0.5), finding{'id':'nit','text':'nit'}. Move reward_version.startswith(REWARD_VERSION_RUBRIC) assertion into keeper. Detects omitted/wrong version alongside existing absent intrinsic/localization/tool-grounded terms. Production training/rubric.py::score_review uses shared intrinsic scorer, constructs weighted terms and canonical/custom stamp, consumed by coordinator/RFT reward evaluation; positive/zero/fp/malformed/custom fingerprint paths remain independent. History intrinsic-only reward migrationdaffb6e2/current9328e0d0. No support deletion; low risk. Validate tests/test_training_rubric.py tests/test_training_reward.py tests/test_training_coordinator_projection.py.

### R2 — duplicate intrinsic/posterior rejected example

Affected `tests/test_training_reward.py::test_composite_is_pure_intrinsic_posterior_is_sibling` (1case). Keeper `::test_outcome_applies_posterior_penalty_golden[rejected]`, with contestedrow retained.

The keeper rejected row and retired labeled call use identical ScoringInputs([consistent,uncertain],True,4000), pr_feedbackrejected, default weights/prior. Existing row asserts PosteriorBreakdown, penalty1.0, false_positiveaxis, composite0.7,cost0.5. Transfer base unlabeled call for same inputs, exact RewardBreakdown type and serialized absence/presenceposterior_cost, along with base.composite0.7. Can add those base assertions to both rows to avoid arbitrary row branching; contested retains its own cost/penalty. Keep intrinsic golden with issue_ids, formatinvalid, unknown label, weight sensitivity and calibrated-prior tests. Owner reward.py::score_trajectory branches only on format/correctness/labeled penalty and never subtracts posterior from intrinsic. Production harvest.build_annotation, rft replay, rubric consume scorer. Historyabd521e87 scorer unification, daffb6e2 intrinsic semantics. No production/support deletion. Low risk; R1 commands plus tests/test_training_harvest.py.

### R3 — internal import identity and self-comparison

Affected `tests/test_training_reward.py::test_same_function_scores_producer_and_eval_caller_paths` (1case). Recommendation D, no meaningful identity contract. It getattr()s harvest_mod.score_trajectory, asserts that object is score_trajectory, then calls that same asserted object twice with identical inputs to compare outputs. The credible failure is importing a behavior-preserving wrapper or reorganizing the owner, even when all stored scores remain correct. `harvest.score_trajectory` is not public extension API; harvest.build_annotation is the actual consumer.

Remaining owners: test_intrinsic_composite_is_golden proves numerical formula, test_outcome_applies_posterior_penalty_golden[rejected/contested] proves labels, test_cli_harvest_captures_record_judgments_rewards_and_license[applied-accepted] drives actual CLI/current store/output with externally stubbedGitHub, and `tests/test_training_harvest.py::test_annotation_preserves_intrinsic_and_posterior_separation` (all 3 prior sufficiency rows) verifies actual serialized composite and calibrated surprise. Historyabd521e87 (#931 strict typing baseline) introduced the current identity assertion; its test name and assertion express a scorer-unification check; today production scalar/record outcomes own the contract. Deleting node also removes unused `from daydream.training import harvest as harvest_mod` in this test file only; no production seam removed. Risk low; R2 commands and independent preservation review must confirm no public API contract requires import objectidentity.

### V1 — executable provenance from one unchanged capture

Affected `tests/test_provenance.py::test_version_is_package_version` and `::test_install_source_is_known_or_unknown` (2cases). Keeper `::test_to_dict_never_omits_unknown`.

All three invoke capture_executable_provenance() with identical unpatched ambient process package/environment and actual Git reads. Transfer p.version==daydream.__version__ and p.install_source in{editable,git,package,unknown} into keeper after its one capture. Existing exact serialized5keys stays. Owner archive/provenance.py captures independent fields(head/status/install/env); archive finalizer serializes result and SQLite indexes it. Distinct mockedGiterror, distribution failure, explicit digest and unset digest cases remain retained. History package provenanceabd521e87 and subsequent#1106 fixes. No support removal. Low risk. Validate tests/test_provenance.py tests/test_archive_integration.py tests/test_archive_data_capture.py.

### E1 — three scalar corruption tests join the already shared corruption matrix

Affected singleton nodes:
- `tests/test_analyzer.py::test_shipped_count_wrong_shape_merged_items_propagates`
- `::test_shipped_count_missing_items_key_propagates`
- `::test_shipped_count_corrupt_merged_items_propagates_json_decode_error`

Keeper `tests/test_analyzer.py::test_location_and_duplication_propagate_corrupt_merged_items[syntax-invalid/wrong-shape]`; add a missing-items row('{}',ValueError). Current wrong-shape payload exactly`{"items":{"a":1}}` and syntax payload exactly`{not json` match the scalar nodes (whitespace immaterial JSON parsing). Seed_stack_records(deep,'python',n=4) before corruption in keeper to retain fallback evidence pressure supplied by wrong-shape/syntax scalar controls. Add analyze_findings(dd) raises expected_error alongside existing analyze_location/analyze_shipped_duplication assertions. Missing-items has no fallback inputs originally, but new keeper proves the stronger present-fallback case; same shared loader guard executes before fallback.

Regressions retained independently for every reader: corrupt syntax, malformed items mapping, and missing items must raise instead of falling back to pre-merge records or emitting bogus clean values. Production analyzer._load_shipped_items is the sole guard called by _shipped_counts(analyze_findings), analyze_location and analyze_shipped_duplication; analyze_session calls all three in order. The consolidation retains each reader call explicitly, rather than relying on first-reader failure to imply the others behave correctly. `_worked_example_dirs` adds legitimate diff/hunk source but neither changes malformed shipped input or rejects it early; location calls loader before _hunk_ranges. Missing and valid-empty shipped populations remain different retained tests. Fixtures manually seed producer-input artifacts, not analyzer-output expectations. History count-reader controls b74a9b5d (#741/#761) and location/duplication matrix7588c8a7 (#1106/#1116); tests maintained through daffb6e2/9328e0d0. No support deletion. Expected3declarations removed,2cases removed (keeper2->3rows). Low risk. Validate test_analyzer.py plus archive_data_capture.py and archive.py after lockcheck. Complete candidate/readers/assertion blocks inspected; full source-quality family remains outside this transfer.

### F1 — normal dump bundle proof moves to the real successful deep run

Affected `tests/test_archive_data_capture.py::test_dump_artifacts_copies_full_bundle_to_target_dir` (1case). Keeper `::test_default_deep_run_populates_eval_captures_patch_and_current_merge_phase_state`.

Old dump test `_install_deep_capture_backend(...real_internal_phases=False)` replaces test/heal and commit-push with successful mocks, then runs actual runner with one api.py finding/fix and dump_artifacts requested. Keeper uses same committed multistack source and api.py finding/fix through real internal phases, actual bare remote Git commit/push, event loop, dataset capture and normal archive. Add optional dump_path:Path|None argument to LOCAL helper `_run_real_phases_deep` (only this file's consumers) and pass the matching `dump_artifacts` RunConfig field; keeper calls with isolated tmp_path/'uploaded-artifacts'. Preserve all keeper's existing real commit/push, original-source snapshot, verified item UID, captured scoring assertions. Transfer exact requested-dump manifest/trajectory/diff/evaluation file existence and manifest bytes equality into keeper. Can strengthen with all four artifact bytes equality, preserving old assertions plus real bundle identity.

Regression: requested diagnostic dump route creates no complete bundle, loses a required file or serializes a different manifest from installed archive. Actual production `runner.run` registers DUMP_DIRECTORY; run_artifacts passes frozen run/tree to archive.finalize; finalizer writes evaluation/manifest, copies assembly_dir to dump_path, validates frozen tree and atomically installs archive. Dump decision has no branch on agent response count or whether phases were mocked; keeper with real phases is stronger delivered-bundle proof. Keeping separate old mocked internal phase run adds no independent release/runtime contract. Failure/secret dump cases and optional collection failures remain retained because they have distinct bytes/fault/security outcomes. No shared fixture changes; local helper update only; no production changes.

History original capture/dump regression725ae82a (#258); real-phase capture architecture and current records updates0d24691e/9328e0d0/6c48d368. Baseline XML reports removed dump test4.384seconds elapsed testcase time (not isolated walltime or claimed savings). Focus archive_data_capture.py plus archive_integration.py and dataset_capture.py/record_dataset_journey.py. Low risk after independent review that requested route is actually executed, not prewritten. This flow consolidation changes keeper configuration, so root coordinator should explicitly accept the strengthening before implementation.


## Native runtime and default deep UI

## D1: internal transport AST ownership guard and its self-tests

Exact removed nodes in tests/test_backend_lifecycle.py:

- `test_the_guard_detects_each_forbidden_shape`: all7rows, whose source strings are (1)await transport.wait, (2)t=CliTransport then await t.wait, (3)typed t:CliTransport wait, (4)except TransportExitError, (5)tuple exception, (6)_transport.TransportExitError, (7)raise AdapterError(category='PROCESS_EXIT').
- `test_no_adapter_owns_the_reap_or_the_process_exit_raise` singleton.
- `test_the_owner_still_carries_the_shared_surface` singleton.

These detect source location/name organization and detector correctness, not observable transport behavior. Moving equivalent reap/exit handling to another module with identical cleanup/errors fails the guard; no public/extension/shipped contract requires `_transport.py` to retain exact helper declarations or adapters to avoid those source patterns. CLAUDE/CONTRIBUTING/docs search finds no such source-location obligation. Public Backend/create_backend/event/cancellation semantics remain; private imported helper names do not establish an extension API.

History a603b868 (#1246 closing#1221) introduced shared reap/exit refactor with structural liveness checks. Its history explains the internal seam, not a requirement to pin future function placement. Keep production seam/code unchanged in this batch. These three tests are precisely the internal organization pattern user requested auditing.

Named behavior keepers (all remain): same file `test_reap_suppresses_transport_exit_and_returns_the_code` all0/1/-15rows, `test_raise_for_exit_kwarg_shape_is_per_adapter`, `test_teardown_is_idempotent_and_drops_the_transport`, `test_process_exit_message_reports_the_count_it_prints`, exact Codex/Pi/Osprey process-exit message/count/category nodes and Pi OOM retryability, `test_codex_clean_exit_lifecycle`, `test_pi_clean_exit_lifecycle`, `test_osprey_clean_exit_lifecycle`. These drive original adapters/shared transport with only child spawning replaced and observe child reaped/stdin+FD closed/backend transport list empty and correct typed diagnostic on failure. Moving unchanged code is deliberately not a regression; omitting reap/closing/typed exit mapping fails these actual behavior keepers.

Stronger real-process counterparts: tests/test_transport.py::test_transport_streams_jsonl_lines_with_exit_code, ::test_transport_nonzero_exit_raises_exit_error, ::test_transport_teardown_is_idempotent_and_group_signalling, ::test_transport_cancel_all_is_shielded; tests/test_artifact_visibility_integration.py::test_runner_external_adapter_cancellation_reaps_process_and_freezes_once[codex/pi/osprey] retain real runner/Git/CLI/child lifecycle and one frozen snapshot; native protocol CLI tests in backend Codex/Pi/Osprey retain different stdin/@file/sandbox inputs. No claimed stronger keeper replaces distinct backend transport.

Production owner read complete daydream/backends/_transport.py; adapters import reap/raise_for_exit/teardown and retain their parser/error wording. Callers searches show private source scanner helpers referenced only inside this test module. Delete test-only scanner block and ast import: `_REPO_ROOT`, `_BACKENDS_DIR`, `_OWNER`, `_OWNER_SURFACE`, `_is_transport_exit_error`, `_names_cli_transport`, `_constructs_cli_transport`, `_transport_bound_names`, `_forbidden_sites` (none production). Keep Path import for real adapter invocations. Production deletion unlocked none. Risk low; validation backend lifecycle+transport+all backend native+real runner artifact/cancel consumers.

## C2: Pi generation/identity/provenance share exact native fixture replay

Keeper `tests/test_backend_pi.py::test_pi_generation_lifecycle_start_end_pair_around_tool`.

Removed singleton nodes:

- `test_pi_native_ms_start_converts_exactly_and_chronology_holds`
- `test_pi_user_and_tool_results_do_not_create_generations`
- `test_pi_turn_end_carries_native_identity`
- `test_pi_usage_events_carry_turn_end_and_terminal_provenance`

All FIVE tests have exact identical execution call `_collect_events(PiBackend(model='glm-5.2'),'go',fixture='generation_lifecycle.jsonl')`. Same real native PiBackend parser, external spawn-only mock, cwd/tmp parent/env/input/model/output schema/flags. Helper constructs recorded raw native JSONL via tests/harness/pi_replay -> process_replay -> real Backend.execute; no normalized-event fake replaces protocol differences.

Regression/assertion transfers: native start missing/unit-wrong/out-of-order end -> transfer exact1788690314289ms, exact1788690314289000000ns and host-end chronology assertions. Extra non-assistant generation -> preserve count2 positive controls and full tool-events truthiness/no-generation assertion block (keeper already counts2 starts/ends). Native TurnEnd omitted/wrong finish/model/provider/provenance -> transfer complete final TurnEnd nonempty/stop/glm4.6/nous/native source/empty messageid/no messageid source/host timestamp assertions. Metrics/cost provenance mislabeled -> transfer nonempty metrics/all turn_end source, nonempty costs/final terminal source/reported cost. Existing keeper exact generation UUID/response identity, ordered reasoning/text/toolCall choice, call001/read_file/path, generation-before-execution ordering, no duplicate choice and second independent generation assertions remain.

History94457cba (#1175) introduced all these assertions together for P18 observability/native generation contract. Same native fixture rows carry two assistant start/end generations, tool execution, final turn boundary and usage. No transport/role/clock/fault variation distinguishes separate calls. Complete Pi production owner and fixture read: message_start creates host-local ID, message_end seals exact native timestamp+choice, tool events link later, turn_end emits metrics+native identities, terminal_events emits reported usage. Keep sanitized fixture-label test and ALL distinct error/malformed/EOF/cancel/attachment/transport/provider/read-only/finalization cases.

Support deletion none; _collect_events stays widely used. Production none. Risk low with intact transfer groups; validation full backend_pi+observability/conformance+contract parity+artifact visibility/stream idle.

## D3: private Pi tools inventory already observed at actual argv/config seam

Removed singleton `tests/test_backends_events.py::test_pi_selected_tools_count_derives_from_read_only_tool_constant`.

Regression: `_PI_READ_ONLY_TOOLS` literal/list count changes. Strong exact matching mode keeper `tests/test_backend_pi.py::test_pi_request_event_config_matches_exact_argv` observes read_only=True native spawn '--tools'='read,find,ls,grep' and PiRequestConfig selected_tools_count4/presentTrue; `test_read_only_restricts_tools` separately observes read-only list and absence in default mode. Actual public RequestEvent/argv behavior is stronger than inspecting a private constant. All exact strings/count unchanged in keeper; no assertion transfer needed.

Production callers only Pi execute args construction and typed config count; rg finds no external/private API consumers. Native tools-disabled/finalization tests remain for distinct0tools mode and thinking/stdin/file transport. History94457cba introduced private count/source test alongside actual argv/config seam; no extra shipped bytes/extension contract from constant's identifier. Delete its now-unused import `_PI_READ_ONLY_TOOLS` from events test module only. Public event fields/defaults/unions/export sets/admission security remain. Risk low; full events+Pi+observability+native CLI validate. No production edits.

## C4: persisted user/agent text steps share exactly one agent call

Removed singleton `tests/test_agent_recorder_integration.py::test_text_event_creates_agent_step`.

Keeper same file `test_user_prompt_becomes_user_step`. Both use `_scripted([TextEvent(text='hello back')])`, same appended ResultEvent(None,None), `_run_with_recorder(backend,tmp_path,prompt='hi')`, phase REVIEW/defaultNORMAL and real recorder/filesystem. Transfer exact `_single_agent_step(traj)['message']=='hello back'` to keeper after user-step assertions. Helper proves schema-valid persisted document and exactly1agent step; keep full existing user source/message/absence of agent-only fields. Regression missing or wrong agent text/extra agent step remains caught; absent/wrong user metadata remains caught. No metrics/phase/cancel/error payload changes.

History 4c9a96a5 (#55) introduced ATIF recording/user/agent mapping. Owner agent.run_agent enters real recorder invocation, observes prompt user step then native events; invocation._observe_text appends text, observer closes step and persistence writes file. Real user+agent mapping proof remains supplementary to retained full runner/archive/contract native streams. No test-support/production removal. Risk low; validate agent_recorder+trajectory+agent+native contract+full runner consumers.

## C5: private Pi retry facade rows duplicate exact table inputs

Removed declaration `tests/test_backend_lifecycle.py::test_pi_facades_delegate_to_the_shared_parsers`, rows `[default]`, `[override]`, `[empty-warns]`, `[negative-warns]`.

Keeper `tests/test_backend_pi.py::test_pi_retry_env_knobs` attempts-default/attempts-env-override/attempts-empty-warns exact rows; negative input not currently in keeper so FIRST add row env DAYDREAM_PI_RETRY_ATTEMPTS, parse=_pi_retry_attempts, '-1', expected_PI_DEFAULT_RETRY_ATTEMPTS, negative warning fragment. Also carry warning knob identity for all invalid rows: after assert warn_fragment, assert f'{env_var}={env_value!r}' in caplog.text. Defaults/override old test made no silence assertion, so no lost positive-control claim. Existing shared int/float parser rows retain full generic type/nonfinite/sign/absent warning contracts; wrapper tests share exact os.environ -> facade -> common parser route.

Historya603b868 (#1221) introduced shared parse/facade guard; subsequent dedup commits retained facade rows already covered in Pi matrix. Callers `_pi_retry_attempts` used Pi.__init__, actual execution-input/nativeargv retry policy keeper independently retains override behavior. Private facade has no separate extension signature contract. Remove now-unused Pi facade/default constant imports from lifecycle module only (PiBackend/PiError remain). Four removed cases+one negative keeper case means3net cases/1function. Risk low, validate lifecycle+Pi+agent retry/budget.

## F6/F7: consolidate two duplicate executions inside existing Pi tests

No function/row removed; support interface unchanged.

`test_pi_request_event_config_matches_exact_argv` currently launches twice with identical model/schema/read_only=True/persist_session=False/prompt='structured please'/simple_text fixture. Use ONE `replay_process(backend,make_mock_process_from_fixture('simple_text.jsonl'),Path('/tmp'),'structured please',output_schema=schema,read_only=True,persist_session=False)` to obtain events and spawn recorder; derive flat_args=list(mock_exec.call_args.args). Preserve every original argv/config/system/schema assertion. This binds observed config and argv to SAME actual invocation and removes one duplicate native replay. Native output-schema mismatch semantics not weakened (no new parser/assertion helper); _run_and_capture_args/_collect_events stay for other cases.

`test_pi_error_turn_sets_explicit_incomplete_boundary` currently launches same error_turn.jsonl twice at modelglm5.2,cwd/tmp,promptgo: first only requires PiError, second collects partial events and swallows PiError, with redundant nested stale-process patch. Use ONE fresh mock child, collect events inside original `pytest.raises(PiError)` then read complete TurnEnd field assertions. Keep type/partial event outcome; no error field relaxation. Separate `test_error_turn_raises_pi_error` message-match test and actual billed-failure observability/native terminal schemas remain. Saves one identical native replay. History94457cba (#1175) added boundary test; F removes extra test-only control without changing native stream or outward API.

Risk low; full Pi and native observability/stream/cancel contract validation. Count reduction0, expected invocation saving2, not a measured performance result.

## Additional deep flow C8: default stage display calls share existing preflight run

Removed singleton `tests/deep_orchestrator/test_pipeline_intent_and_resume.py::test_stage_ui_surfacing`.

Keeper same file `test_preflight_notice[False]` (custom_builder=False parameter row). Both use default multi_stack_target/default stub/_run_deep; raw prompt `_accept_intent_decline_other`; noise-only UI patched. Original stage node captures review_steps+merge_steps print_stage_progress and suppresses preflight; keeper currently captures preflight metadata and suppresses review progress. Move complete progress_calls list/_capture_progress spy setup to keeper, replacing review progress no-op and also patching merge owner; leave preflight capture. After one run preserve numbers{1..5} and every total5 assertions. Applies harmless stronger assertion to custom_builder=True too; no row deleted. Every unique preflight stage list/stacks/agent count/exploration metadata contract remains. The default false row has identical production/config/backend inputs; extra observing preflight does not alter flow. Keep forced interactive fix prompt, error stages, phase event persistence and actual UI rendering tests separately.

History5bfe1163 (#45) introduced D30/D44 after same default run. Owners stage print bindings in review_steps/merge_steps and default phase ordering already read fully; custom builder controls folded alternatives count but both routes preserve5UI stage numbers. No helper/production edits. Risk low; validate whole pipeline module+UI/deep consumers. Removes1function/1case/real runner invocation.

## Retained false positives / broad judgment

- Backend conformance/ATIF parity: `_loaders` builds genuinely different SDK/Codex/Pi native messages and real parsers, not identical normalized fake. Cross-backend parity and native absolute checks independent. Metrics identity/cost permitted deltas intentionally differ; keep read-only/native transport rows.
- Public AgentEvent dataclass default/type/union/export and backward-compatible constructor tests protect extension protocol; supplied values are not merely disposable copier tests. Typed config wrong types/closed enums/range/nonfinite/security omissions protect trace admission; retain different field/backend cases.
- SDK version/process injection, Codex disposable clone/path binding/Darwin real-Git concurrent cache, Pi settings/API-key/read-only/instruction attachments/finalization/tool-free stdin, Osprey wire errors/stderr/native metadata/cancellation remain distinct transport risks.
- Agent retries/cancellation/stalls/recovery allowances/backoff and actual fix/heal/hostcommit hooks preserve independent failure modes; call counts sometimes prove retry/lifecycle and are not deleted merely as mock assertions.
- Public required keyword-only phase checked by reflection plus actual TypeErrors is an API contract, unlike D1 private module owner placement. Keep it.
- Prompt/schemas and package workflow/source-byte checks are shipped model instruction/config promises; simple builder assertions are not removed merely for string matching. Full runner/CLI boundary keepers remain primary behavior proof.


## Configuration and benchmark workflows

## B1: identical model/effort default lookup assertions

File: tests/test_config.py. Static here protects explicit default policy; do not discard policy merely because it is static. The proposed changes combine exact duplicated inputs with complete policy assertions, leaving default resolution/runner tests intact.

### C: test_per_stack_review_and_arbiter_split (one case)

Actual assertions: Claude per_stack_review sonnet; Claude arbiter opus; Codex per_stack_review terra; Codex arbiter sol. All read PHASE_DEFAULT_MODELS directly; no fixtures/config differences.

Named keepers: test_phase_default_models_claude_tier_assignments, test_phase_default_models_codex_tier_assignments. Codex already checks per_stack_review and arbiter literals in its mid/heavy loops. Claude already checks per_stack_review but omits arbiter: add arbiter to Claude heavy loop BEFORE deletion. This preserves regression for accidentally moving arbiter to mid tier. No new run/setup needed.

History: ce7e5719 (#175, Sonnet-first per-stack with scoped arbiter). Default cheap-review/heavy-arbiter policy is retained, not relaxed.

### C: test_suppression_uses_cheap_tier (one case)

Actual assertions: Claude suppression sonnet, Codex suppression terra. Same direct table inputs. Codex generic tier keeper already checks suppression. Add suppression to Claude mid loop, then remove the duplicate node. Keepers same two model-tier functions. History f7d539a0 (#248 precision-mode evidenced minor-finding suppression).

### D: test_plan_write_is_pinned_to_the_top_model_tier[claude] and [codex] (two cases)

Actual regression: plan_write/review/arbiter model values unequal. Both named model-tier keepers independently assert those three fields equal their explicit top-tier literal. After B1 arbiter transfer, they logically imply the equality check on the same dictionaries and backends. No assertion transfer needed. History 9bacaa42 (#291 Improve plan-writing core). No model value/default behavior changes.

### D: test_diagram_phase_is_mid_tier_on_both_model_backends (one case)

Actual assertions: Claude diagram equals intent; Codex diagram equals intent; Codex diagram effort medium. Named keepers: both model-tier assignment tests explicitly pin diagram and intent to the same mid literal; test_deep_phase_effort_tier_assignments explicitly checks diagram medium. Identical dict inputs and failure modes; no unique field removed. History 10e319a7 (#1120 grounded diagrams). Diagram byte/schema/registry/config/runtime/grounding tests remain unchanged.

### C: test_plan_write_is_pinned_to_max_reasoning_on_every_backend[claude/codex/pi] (three cases)

Keeper: test_improve_phase_effort_tier_assignments[claude/codex/pi]. Both sets have exactly the same backend inputs, no fixture state, and direct policy lookups. Add exact original `PHASE_DEFAULT_EFFORT[backend]['plan_write'] == 'max'` to existing per-backend keeper, retaining recon/audit/vet assertions on Improve's half. Do not assert only IMPROVE_PHASE_DEFAULT_EFFORT's plan_write: original checks merged production table, whose independent merge contract must remain. History 9bacaa42 (#291). Three removed cases/functions grouped into one removed declaration.

### D: test_no_pr_feedback_skill_constants (one case)

This asserts two internal config attribute names do not exist. Search finds no current production callers/imports or extension exports for PR_FEEDBACK_FETCH_SKILL / PR_FEEDBACK_RESPOND_SKILL; only this test names them. It detects adding unused dead internal constants, which by itself changes no supported contract. User-facing removed command rejection remains explicitly proved by tests/test_cli.py::test_feedback_subcommand_is_unknown (real cli.main parse, exit2, error and zero runner dispatch). Native shallow/deep no-skill routing remains in tests/test_extension_skills_integration.py, and extension API/version/Registry public methods tests remain retained. History b27c5fc3 (#907 feedback and skill execution removal). Remove now-unused `from daydream import config` import, not any production symbol.

Owners/callers for B1: daydream/config.py full tables; run_config._resolved_model/_resolved_reasoning_effort; runner._resolve_backend forwards effective values; phase/default models also archived through run_artifacts. Config tables are unchanged and public Backend/create_backend resolution remains. CLI/runner default model/effort and overrides and real model-capturing deep tests remain. No support/production seam deletion unlocked. Risk low with explicit Claude arbiter/suppression and merged plan effort transfer. Focused test_config, test_cli, test_runner, deep model/effort consumers + full gate.

B1 expected reduction: six declarations / nine cases; all meaningful default assertions retained.

## B2: absent config loader assertions share one actual filesystem read

File: tests/test_config_file.py.
Removed singleton nodes C:
- test_improve_config_absent_defaults_empty
- test_diagram_config_absent_defaults_to_unset
- test_an_absent_retry_recovery_allowance_stays_silent

Keeper: test_absent_config_is_empty.
Every node receives the same empty real tmp_path, no pyproject/dotfile, no backend/config overrides, and calls production load_file_config(tmp_path). Move Improve empty roots, all five diagram unset/empty fields, retry allowance None and no retry-warning assertion into keeper. Add caplog fixture to keeper, retain its model/backend/phases/reasoning assertions. Do not substitute DaydreamFileConfig constructor for actual file loader.

Regression: absent defaults drift, accidental diagram enable/threshold/root default, false invalid retry warning. Same missing-file inputs and public loader path retained. Production load_file_config reads optional TOML, merges {}, invokes coercers and constructs policy; retry coerce_declared_retry_allowance returns None silently for absent value. All callers are real CLI/config loaders/runner flows. Malformed config, per-key merging, invalid declared warning, explicit diagram/improve config and direct constructor defaults stay separate.

History: Improve absence 9bacaa42 (#291), diagram absence 10e319a7 (#1120), retry silence c9b61931 (#1234), initial config 469f2839 (#141). No production/support deletions. Risk low. Keep test_empty_config_helper: direct constructor default is a distinct configuration/extension contract because loader passes explicit values and could hide default-class drift.

Expected reduction three declarations/cases. Focused complete config_file/config, CLI, runner and config consumers.

## B3: four exact private coercer rows duplicated by public TOML loader rows

File: tests/test_config_file.py::test_quality_gate_threshold_coercion.
Proposed row deletions only: [negative], [nan], [inf], [bool]. Keep declaration and other seven rows unchanged.

Named keepers: test_quality_gate_thresholds_in_file_config_degrade_to_none[negative/nan/inf/bool] respectively. Same raw inputs: -0.1, IEEE NaN, positive infinity, True. Real TOML parser produces those exact primitive inputs and load_file_config routes all FOUR quality thresholds through the same _coerce_non_negative_float owner. Keeper asserts all four observable policy fields None. Original private rows assert only direct helper output None. Credible regressions accepting negative values, nonfinite comparisons, or bool-as-int fail the loader keepers as well.

Preserve negative-inf/string/list/absent/zero/valid/int-coerced rows since exact public-input keeper equivalence is not established here. Preserve finite config values and runner quality-gate/fault behavior. History 32383d0b (fix-round/session/threshold clamp). Production owner read in full: config_file's numeric coercers and complete load_file_config. Helper has four quality fields plus budget consumers; removal neither changes exports nor reduces actual file-field route coverage. No support/production changes. Risk low. Focused config_file plus quality-gate owners/full gate. Reduction four cases, zero declarations.

## B4: calibration fixture metadata checks share actual packaged document

File tests/test_benchmark_calibrate.py.
Removed C singleton nodes: test_fixture_is_24_pairs_12_12 and test_fixture_covers_all_eight_categories.
Keeper: test_fixture_provenance_declares_unverified_llm_origin.
All load exact same packaged _load_fixture(), no fixture mutation/options. Carry exact original pair count, actual label count match12/nonmatch12, REQUIRED_CATEGORIES inclusion into provenance keeper before/around its original origin/human_reviewed/unverified/class_balance/categories/len assertions. Do not rely only on derived provenance to imply actual label counts; transfer them explicitly. Shared frozen byte-derived fixture contract remains covered; no fixture inventory is dropped.

Credible failures: changed class balance/cardinality, omitted expected category, wrong provenance declaration. Same actual pairs and required category list retained. Production _load_fixture_document/_load_fixture/_load_provenance feed receipt invalidation and both receipt provenance fields. Full calibrate.py and complete tests read, packaged judge/template loading remains real. History fc21b1c1 (#872 configured semantic-match diagnostic). Source-free fixture and prompt-size/loader sibling identity tests remain separate. No support/production deletion. Reduction two declarations/cases. Focused complete calibrate plus verifier/package consumers.

## B5: calibration successful run and receipt privacy share one actual run

Removed C node: tests/test_benchmark_calibrate.py::TestCalibrateAcceptance::test_acceptance_zero_source_leakage.
Keeper: same class test_run_calibration_pass_writes_receipt.
Exact matching inputs: _scripted_responses(_load_fixture()), _scripted_http, ws_factory(tmp_path), yes=True, env=_env(), injected external HTTP. Both run full run_calibration and actual packaged judge for 72 scripted calls with real event loops/filesystem. Move the entire receipt read and credential/source-negative assertions into success keeper. Keep success exit, actual receipt existence and counter72. No fake supplies receipt bytes; production _build_receipt/_write_receipt/atomic storage remain real.

Credible regression: success receipt contains credential field/token or source excerpt despite successful scoring; exact actual output inspected in keeper after the same run. Negative classification/stability/confirmation, CLI OAuth/HTTP transport distinction and allowlists remain retained. No promotion of helper test over public run. History fc21b1c1 (#872). No production/support changes. Reduction one declaration/case and one redundant72-call success run. Focused complete calibrate, benchmark CLI/packaging and full gate.

## B6: fresh benchmark init/status/validation share one exact initialized workspace

Removed C singleton nodes in tests/test_benchmark_workspace.py:
- test_init_gitignore_ignores_everything_except_itself
- test_status_surfaces_unresolved_identity
- test_validate_fresh_workspace_returns_2
Keeper: test_status_fresh_workspace_is_empty_and_unresolved.
Every original initializes tmp_path/'ws' with repo='O/R', reviewer ['h1.example.com'], judge ['h2.example.com']; no differences/faults. Carry gitignore text '*' and '.gitignore' plus0600 assertion, code2/incomplete label from actual validate_workspace, and source hostname/repository-id/visibility assertions from actual workspace_status. Retain keeper empty-state/resolution/ledger assertions.

Preserve first validation freshness input by performing transferred validate_workspace before workspace_status if combining; initialization already commits transaction and creates lock, so both public readers' recovery of empty transaction directory changes no state. Neither fabricates fixtures/case/status outcomes. Do not replace actual public status/validation calls with schema constructors. Real subprocess tests/test_benchmark_workspace_cli.py::test_benchmark_init_status_validate_roundtrip remains primary CLI init/status/validate proof, including stdout privacy labels/status and exit2. Case/bundle fidelity/readiness/repair/mutation faults remain untouched.

Production owners read: init_workspace, workspace_status, validate_workspace, optional preflight reader, empty _derived_state; actual WorkspaceLock/Transaction/storage routes retain real temp FS. CLI uses these exact owners. History 7ce27fde (#790 private workspace/schema). Important distinct tests retained: init layout/modes (separate OWNER/REPO/api host input), host case normalization, rejection/nonempty directory, corruption/missing manifest and every snapshot/authoring checksum/rollback/control. No support/production changes. Reduction three declarations/cases. Risk low; full benchmark_workspace and CLI/shared storage consumers/full gate.

## B7: unused renderer backend parameter duplicates exact shipped output

File/node: tests/test_benchmark_harbor_env_policy.py::test_rendered_job_config_table[pi] and [claude].
C: remove parametrization and unused backend argument; keep one unparametrized test_rendered_job_config_table with complete existing assertions.
Both rows run identical render_job_config(oracle=False), parse same YAML, union same reviewer/verifier env names and assert same expected renderer set. Backend argument is never referenced; render_job_config accepts only oracle. No fixture/hook reads backend param. Source/caller search and full module read establish there is no selected backend transport/config being varied here. Host and container tests DO use backend, produce different kept/scrubbed sets and MUST retain both pi/claude rows.

Actual regression: shipped renderer drops/adds policy env name. One exact same render+assertion catches it. Package build calls render_job_config oracle=False/True separately; Oracle row, renderer key order, placeholder contents, packaged bytes and executable sandbox tests remain unchanged. History 04a6fd7a (#1316 central Harbor environment policy). No support/production changes. Reduction one redundant parameter case, zero declarations. Focused complete harbor_env_policy and harbor_package/build/agent/skill_free/execution-isolation consumers; full gate.

## Retained false positives / uncertain patterns

- Static package/runtime lock, wheel metadata, Dockerfile pins, compiled inventory/schema/byte hashes, bare-import/container assets, environment isolation and execution tests are independent shipped/privacy contracts. Reading source/bytes alone is not a deletion reason.
- test_services.py::test_no_service_containment_shape_lives_outside_services_py and source-policy literal guards were flagged as organizational checks. History f87732c5 (#1239) records deliberate single-owner acceptance; matching all *future* consumers cannot be established from present behavioral tests alone. Conservatively retained pending an independent decision on whether this architectural contract itself remains required. No source scan/helper/import deletion proposed in this ledger.
- Extension Registry public introspection/absence/export/frozen dataclass tests have real fork/API obligations. Do not remove them merely because they inspect names. Error/no backend dispatch tests preserve fail-before-requests.
- Static Improve issue posting predicate is not proven redundant with no-App publication runner tests; enabled posting influences App token acquisition before flow. Retain until matching configured-App real path covers both enabled/disabled modes.
- Parameterized invalid YAML, unsafe paths/commands, symlink traversal, readonly worktrees, Git authentication/processes and fault phases often reach different guards/type branches despite common exception type. No broad pruning justified by count or names.
- Direct DaydreamFileConfig constructor default proof remains distinct from public loader explicit keyword values.


## Repository logging scan

Affected `tests/test_cutover_ast.py::test_no_legacy_debug_logging_references`:
all 722 per-file rows become the same declaration with one collection-time
SOURCE_FILES snapshot. `_all_py_files` and every original AST-check subtree are
unchanged. This retains forbidden names/attributes/lazy imports and27 prefixes
under daydream/ and tests/, sorted inventory, UTF-8 bytes, ast.parse filename,
__pycache__ exclusion, self-exclusion and file:line diagnostics. No scoped fixture
or selector consumes py_file; no workflow selects a specific old row. Source
files are parsed rather than imported. It remains a source policy, not a claim
that runtime recording tests prove repository-wide absence.

History 4c9a96a5 (#55) introduced CUT-08; 18790567 fixed self-exclusion; d0269cb3,
dd4a2a6e and de8f3e24 changed narration/formatting only. Keeper is the same named
scan over the identical722 source paths. One failure now reports the first
source offender; the old multi-row run could report several files. There is no
documented all-offenders diagnostic contract. Scan CPU remains;721 fixture and
worker wrappers disappear. Source policy itself is preserved and mutation-proved.

## Validation scope

Focused owners and shared consumers include complete changed files; actual
corpus projection/reproducibility/coordinator/dataset journey, archive/capture,
trajectory lifecycle/phase consumers, harvest, benchmark workspace CLI/storage,
packaged Harbor build/agent, native adapters/transport/observability/conformance,
contract suites, retry/budget, real adapter-cancellation runner coverage, runner
and CLI. Matched commands use the same 46 paths, four workers, explicit ten-case
assignment batches, Python/dependencies/machine, absent GH auth and no coverage
on both sides. Results are reported with full gate and CI evidence in the PR.

The full gate keeps its original branch-coverage settings, exclusions and floor;
CI runner size, workers and workflows remain. Original baseline/full-gate and
focused-flow results are in the companion report. Final `make check` and hosted
CI outcomes belong to the actual validation record; this ledger does not infer
performance from fewer cases or predict the four-minute CI target.
