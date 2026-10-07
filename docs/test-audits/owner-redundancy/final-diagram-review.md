# Independent final diagram/parser preservation review

Baseline: f7494ba50b512b4449dd13563031057590c9d030.
Candidate: /private/tmp/daydream-owner-redundancy working-tree diff against that exact baseline.
Reviewer: training_audit (independent of diagram author and pre-edit backend/parser reviewer).

**Source preservation approved. No blocking findings.** This review made no source edits and ran no tests. Focused validation and isolated behavioral mutation controls are owned by root and remain required before landing; this is not a runtime or gate-result claim.

## Complete changed scope

Only tests/test_diagram_render.py and tests/test_tree_sitter_index.py change in this boundary. Six distinct declarations / seventeen collected cases are retired: four rendering declarations/five cases consolidate, and two parser implementation inventory declarations/twelve language rows delete. Production parser/classifier/renderer/report code, external/public reexports, fixture goldens, schema/grounding/trigger files and real integration journeys have zero diff against the pinned baseline.

Read the full changed test surfaces (including adversarial/omission/threshold/grammar edge coverage), every original deleted test body and affected parameter table from git show at the pinned baseline, final keepers and their literal specs/helpers, independent .mmd fixture bytes, complete diagram_render/deep.render and tree_sitter_index.statements/runtime/__init__ owners. Checked diagram_steps report/application and reviews/diagrams re-render callers, root fixture chains/real Git helpers, real runner integration and diagram-only publication keepers and their external Backend/FakeGh fixture seams. Reviewed docs/test-audits/owner-redundancy/diagram-map.md.

## Deletion-to-keeper checks

1. `test_diagram_block_contains_only_the_rendered_diagram[sequence-spec0]` -> `test_sequence_mermaid_matches_golden_fixture_byte_for_byte`. Original SEQUENCE_SPEC/_rendered payload identical. Literal HTML heading, fence, blank lines, Mermaid body and closing details all remain exact equality. Body expectation now reads independent committed diagram_sequence.mmd, with one final LF removed after the direct renderer assertion pins its original newline convention. This is stronger than generating expected Mermaid with the same renderer. No fixture edits or producer-computed expected wrapper.

2. `test_diagram_block_contains_only_the_rendered_diagram[flowchart-spec1]` -> `test_flowchart_mermaid_matches_golden_fixture_byte_for_byte`. Identical FLOWCHART_SPEC/_rendered payload and wrapper contract. Independent diagram_flowchart.mmd supplies exact body; heading remains literal Flowchart. Wrong dispatch/render body/wrapper bytes still fail.

3. `test_blocks_render_sequence_first_then_flowchart` -> `test_blocks_always_rerender_and_never_echo_a_stored_mermaid_string`. Same unpoisoned _both_rendered() input is rendered before poisoning. Sequence position before flowchart, exactly two folds and exact one-blank-line joining all copied verbatim. Original poisoned Mermaid scenario remains separately explicit after this plain execution; no ordering claim relies on a poisoned fixture instead of the original input.

4. `test_rendering_is_deterministic_and_does_not_mutate_the_spec` -> same keeper. Exact original deepcopies of _both_rendered input preserved, first and second render receive the same copied object, results==before checked, direct sequence/flowchart renderer versus deepcopied spec equality preserved. Additional later poison invocation cannot substitute for the retained plain determinism/immutability path. Mutating copied input or nondeterministic result remains observable.

5. `test_diagram_section_survives_a_round_trip_through_the_report_and_back` -> same keeper. Exact original plain blocks, render_report(_items()), insert_diagrams_section and literal `## Diagrams` section membership remain. Both actual Mermaid strings remain checked in report. Report surgery is executed on real renderer output before poison run. Existing tests/test_deep_render.py separately retain placement, byte-idempotent replacement, empty removal, coverage/neighbor preservation and trailing-newline behavior.

6. `test_branch_table_entries_all_occur_in_probe_sources[go,javascript,python,rust,tsx,typescript]` deletes dependency syntax/table subset checks, _BRANCH_PROBE_SOURCES and its sole private _node_types helper. Original only asserts configured BRANCH_NODE_TYPES is a subset of fixture tree types. Removing a required member already passed; adding an inert never-produced name fails while Daydream behavior stays unchanged because owner uses set membership rather than constructing syntax queries. Literal branch_statement_lines outcomes remain for Python flat elif/match, TS nested chains/switch, Go all switch kinds, Rust match/let/try, TSX JSX and JavaScript; same-source TSX/JS compatibility remains. Branch predicate container lines, unknown parser fallback, malformed source and parser safety gates stay. No production table/export/query/grammar changes.

7. `test_terminal_table_entries_all_occur_in_probe_sources[go,javascript,python,rust,tsx,typescript]` deletes the analogous inventory of TERMINAL_NODE_TYPES/TERMINAL_CALL_NAMES versus parsed fixture names. It never invoked is_terminal_line; removal of an actual supported node/callee already passed. Exact literal terminal outcomes remain for Python return/raise/sys.exit/os._exit/exit/quit, TypeScript return/throw/process.exit, Go return/os.Exit/panic/log.Fatal/log.Fatalf/log.Panic/t.Fatal/t.Fatalf, Rust return/process exits/aborts/panic/unreachable/todo and rejection of println, and native TSX/JavaScript fixtures. TSX/JS native fixtures differ from old TypeScript inventory source; map correctly does not claim exact full callee equivalence. What retires is a fixture-table consistency assertion, not tested end-to-end terminal behavior. Broader executable-statement grammar compatibility inventory stays. Public BRANCH_NODE_TYPES, TERMINAL_NODE_TYPES and TERMINAL_CALL_NAMES reexports unchanged.

## Retained boundary proof

- tests/test_deep_diagram_integration.py::test_sequence_auto_trigger_renders_grounded_diagram uses real runner.run, temporary committed source Git data, actual grounding/schema/render/report/artifacts and captured external GitHub submission; literal SEQUENCE_GOLDEN and exact heading/report placement asserted.
- ::test_flowchart_auto_trigger_renders_grounded_diagram retains actual branch-heavy trigger, pinned root range/branch_points, literal FLOWCHART_GOLDEN and posted body.
- ::test_both_signals_render_sequence_first retains real runner-generated both-kind publication order.
- tests/test_diagram_only_integration.py::test_diagram_only_posts_a_marked_issue_comment[sequence,flowchart] retains real runner/Git/filesystem issue-comment ownership and kind marker, with no review/merge artifact.
- ::test_review_findings_artifact_carries_diagrams_and_phase_b_renders_them retains actual findings-out -> post-findings CLI -> immutable re-rendered PR review publication. Other immutable evidence/schema/confinement, omission, failure, caps and previous-artifact ownership cases are untouched.
- CandidateRoot constructor/reexport check in tests/test_diagram_trigger.py remains, as do DiagramThresholds config behavior and native library public tables.
- All existing sanitizer/injection tests remain: Mermaid statement prevention, HTML wrapper escaping, empty-label fallback, truncation before escaping, edge/node shape poisoning and malformed/degraded records.

## Required root proof still pending at review time

Execute and record actual-owner mutations for wrong HTML wrapper, reversed kind iteration, input spec mutation and dropped report insertion, plus Python branch/terminal table removal. Each named keeper must fail at its behavioral assertion, not import/collection; restore exact prior bytes and rerun. Complete focused shared consumers and final make check with unchanged worker/coverage/runner settings and ordinary signed Git hooks. No count reduction establishes a performance result.

The stale parser comment was refreshed after the focused process ended to: "Literal line outcomes exercise branch and terminal classification through real grammars." EOF blank lines were stripped in both changed source files. These edits affect comments/whitespace only; the approved test bodies and assertions remain the same. Source preservation approval continues to apply.

## Frozen source byte pins (final refresh)

Pinned baseline remains f7494ba50b512b4449dd13563031057590c9d030. Root reported no AST/test-body change after the preservation review: only EOF blank-line removal and the parser comment correction above. Refreshed candidate byte pins below identify the frozen reviewed source; no test execution or in-checkout edits were performed for this refresh. Isolated root mutation controls are still in progress.

| File | Baseline SHA256 | Frozen candidate SHA256 |
| --- | --- | --- |
| `tests/test_diagram_render.py` | `c90d27baa15119988eea12dc8686ac61c145828c62deb2e7d119f4a1ed2bb795` | `f878458749f793c043953ec4595bfe2a2f6b8f043bf057126b2c920952560321` |
| `tests/test_tree_sitter_index.py` | `1148cc5a945869271fe5673d5ed2e75a3510243606091c276338482229a38a8d` | `8c5ecc21a54acaec31f371024f08720b2f6a3e53aa1c942e0c7554261851fdbe` |
