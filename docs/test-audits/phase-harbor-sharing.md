# Share matching phase and Harbor executions

Merged baseline: `6c48d368d529a89748c93e7bb61ae9ac6566936e`.
Immediate baseline: `6a7c41172ee11f6590f848bde5cfc5528b46a0ca`.
Current collection: **8,182 cases / 5,346 functions**, down **822 cases (9.13%) /
85 functions** against merged main. This follow-up removes ten singleton
functions/cases. Production/support/settings changes: zero.

## Exact affected nodes and matching keepers

All nodes below are in the named module. The complete bodies, fixtures, owner
functions/modules, callers, public/extension obligations and relevant history
were read before editing; independent plan reviewers approved the mappings.

### tests/test_benchmark_harbor_build.py

| Removed singleton | Named keeper | Preserved regression |
| --- | --- | --- |
| `test_compile_lock_records_requested_base_sha` | `test_compile_findings_case_full_tree_and_gold_oracle_agree` | Requested-base/original-base provenance, real manifest digest equality and copied requested-base digest sensitivity. |
| `test_compiled_tree_contains_no_raw_authoring_files` | same full-tree keeper | Exact raw authoring-path exclusion scan and root allow-set. |
| `test_compile_writes_task_md_and_inventories_its_digest` | same full-tree keeper | Actual Task.md bytes match approved rendering/digest and both inventories; hidden surfaces excluded. |
| `test_compiled_agent_and_verifier_surfaces_exclude_task_md` | same full-tree keeper | Exact file-type checks, both recursive subdirectory scans and wheel-less environment's three-file inventory. |
| `test_compiled_policy_comes_from_workspace_allowlists` | same full-tree keeper | Literal reviewer h1 / judge h2 policies and separation. |
| `test_task_md_prose_describes_reported_axes_contract` | `test_render_task_spec_is_deterministic_and_sectioned` | Axes reported wording and absence of the false grading claim. |

The first five share exact `_seed_ready_workspace(tmp_path, fake_gh)` defaults,
then one `compile_workspace(ws)` with wheel=None. This is a fresh real Git origin,
import/mirror/bundle, accepted finding and approved ready workspace each time;
there is no cached repository or fabricated compiled tree. Fresh source path
labels do not select content/policy or enter authoring digest. Full compile
keeper retains all 30 original assertions and receives all 22 source assertions
and observation statements. Original case_doc is read **before** compile as in
its source. The keeper still compiles once. Appended observations do not change
files, HEAD, ready state or approval; the digest mutation uses only copied memory.

Renderer keeper retains all nine original assertions, both deterministic calls
and Other-title copy sensitivity. It receives the source's actual original-raw
render and two literal predicates. Original raw remains unchanged.

Compiler/package owners and seed/import/curation/snapshot/CLI callers were read
completely. History: `8a002ee7` compiler/authoring isolation, `aa3205b5`
requested-base fidelity, `45c753c6` approved Task.md, `3ee91d60` policy separation,
`46f900927` reported-axis correction. Explicit wheel/CLI/Harbor models/Docker
build and execution, metric subprocess, invalid/stale/clean/recompile/leak/fault
inputs retain their separate tests and real shipped-artifact proof.

### tests/test_phases.py

| Removed singleton/block | Exact keeper/input | Preserved regression |
| --- | --- | --- |
| `test_phase_understand_intent_correction_prompt_keeps_no_pr_no_skill_directives` | `test_phase_understand_intent_correction_then_confirm` | Same two-turn correction helper; all ten literal/count/grounding/no-PR/no-skill assertions added after the original three. |
| `test_option4_fallback_puts_unknown_cause_in_hypotheses` | `test_phase_test_and_heal_option4_summarizer_fallback_writes_minimal[turn0-expected_substrings0]` | Same RuntimeError class/message and helper defaults; five headings/output/UNKNOWN placement predicates before any unrelated action. |
| `test_build_fix_prompt_carries_generated_file_rule` | `test_build_fix_prompt_concise_mode`, existing default call | Four generated/migration/manifest/lockfile literals; both old calls and four original assertions stay. |
| `test_exploration_pointer_marks_results_untrusted` | `test_exploration_pointer_names_only_bounded_files_and_scopes_read_clause` | Same absent-file pointer; all four boundary/order/None assertions plus all five original checks. |
| Second no-evidence block in `test_hook_aware_push_reuses_evidence_but_still_runs_the_hook` | True/1 loop iteration inside `test_hook_aware_push_runs_suite_exactly_once` | Identical fresh repo, passing hook, fixed bytes, host response, retained-tree/config and actual commit/push call; keeper additionally checks result/argv/cwd/remote SHA. |

No parameter row is deleted. The thrown fallback additions are RuntimeError-only;
the malformed result row, table and IDs stay. Both hook/no-hook iterations stay.
The entire reuse-hit block and its real touching hook/remote verification stay;
only its repeated no-evidence baseline is removed. No hook is bypassed.
ContextVar bindings reset; per-repo audit state and fresh host fake closures
supply no hidden cross-repository input. All helpers/autouses/registry/API seams
and unrelated bodies/rows remain exact. Total source assertions transferred:23.

Owners reviewed: actual intent correction, testing, handoff/fallback/write,
prompt/pointer, publish, host command, run context and Git hook/commit/remote
paths, with public exports and real deep callers. Histories `77b13411`,
`7c6f03d0`, `241ae3c0`, `f19c841b`, `10fcde4b`, `d50078ed`, `ae866954`,
`3f7f1cc6`, `023fc5b7`, `87819298` establish the exact independent contracts.
Real runner/CLI keepers, hook faults, permission/index/rollback/confinement,
transport/cancellation and artifact/session variants remain separate.

## Independent final review and actual-owner controls

Independent whole-module AST projections verified all authorized changes:
Harbor 30+22 compiler and 9+2 renderer assertions; phases 23 exact transfers and
only the second hook baseline block removed. All other declarations, decorators,
rows and helper bodies remain unchanged. Reviewers were independent of the
respective file's author.

Seven isolated production mutations failed at intended transferred predicates:

1. Ignore requested_base only in authoring digest: final copied-base inequality.
2. Render agent hosts from judge: literal h1 policy assertion.
3. Omit reported wording: renderer's literal reported assertion.
4. Omit second correction's slash-command directive: second prompt assertion.
5. Put UNKNOWN under Verified facts: facts-prefix exclusion assertion.
6. Omit actual generated-file prompt suffix: default generated literal.
7. Move untrusted boundary after referenced files: boundary-before-summary order.

Existing preceding checks passed before those failures. Each owner was restored
byte-for-byte and each named keeper passed again. The control checkout was
restored clean and removed. No control or production mutation is shipped.

Evidence under `/tmp/daydream-test-audit-6c48d368/`: Harbor/phases ledgers,
independent plan reviews, cutover AST accounting, `harbor-final-preservation-review.md`,
`phases-final-preservation-review.md`, `phase-harbor-mutation-plan.md`, exact
snippets and `phase-harbor-control-results.json` with failing/restored logs.

## Matching measurement and validation

Same Python3.12.13, locked dependencies, n4, ten-case dispatch, macOS machine,
no coverage, command settings and fresh explicit basetemps:

| Selection | Cases | Pytest | Wall |
| --- | ---: | ---: | ---: |
| Sources and keepers | 19 | 6.44s | 7.14s |
| Strengthened keepers | 9 | 3.47s | 4.19s |

One small-selection pair saves2.95s wall, not a whole-suite or CI result. Complete
owners/shared consumers passed **791 / three existing skips /15 warnings**,
**79.06s pytest /79.77s wall**: phases, Harbor build/package/agent, benchmark
workspace/CLI/storage/schema/snapshot, real intent/resume/fix gates/isolation,
test execution, footprint and Git harness. All three skips are macOS rejection
of non-UTF8 filenames; Linux CI runs those cases. Actual Docker proof passed.
Lockcheck, final collection and diff check passed. Final ordinary make check,
coverage, skips and hosted CI are recorded in PR #1497 after this commit.

The prior full hosted check took6m47s. These small exact sharing changes do not
establish a route under four minutes, nor a30% quota. Production, fixtures,
worker/coverage settings, workflow and shipped package bytes remain unchanged.
