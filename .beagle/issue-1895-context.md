# Issue #1895 implementation context

Gathered read-only at base `main` = `29cddcbe97314e932620a338a45a04f55c4a390d` (2026-11-17).
Source: https://github.com/existential-birds/daydream/issues/1895 (no comments).
Line numbers are as observed at that SHA; re-verify before editing.

This file is handoff context only. It lives on branch `claude/admiring-goodall-eek1ac`
under the git-ignored `.beagle/`; read it with
`git show origin/claude/admiring-goodall-eek1ac:.beagle/issue-1895-context.md`.
Do NOT bring it into the implementation PR.

## Decisions (user + reconciliation)

- **User direction: maximum removal of code and tests. Do NOT write regression tests.**
  This overrides the issue's "prove each removal by extending the owner test". Edit existing
  tests only as far as needed for them to pass against the new formats (version strings, dropped
  keys/kwargs); delete tests that only prove the removed constant behavior. No new assertions such
  as "`model` not in keys". Net test count must go down.
- **Issue drift:** the issue says `daydream.run.v2 → v3`, but unreleased commits a280b3f (v3) and
  29cddcb/#1898 (v4) already bumped it. Bump **`daydream.run.v4` → `daydream.run.v5`** once, covering
  both the claims/derivation removal and the manifest identity change (dataset records embed
  `provenance.effective_configuration = asdict(ManifestRunIdentity)`, `dataset_capture.py:162`).
  Follow 29cddcb's pattern: swap the literal everywhere, accept only v5, no conversion. Refusals keep
  existing messages (`invalid RunRecord: schema_version` → `StoreError("invalid_or_unknown_record_schema")`;
  Hub `_manifest` → `invalid_or_unknown_manifest`). `daydream.hub.v3` stays unchanged (precedent).
- **Run manifest:** `MANIFEST_SCHEMA_VERSION` `"3.0"` → `"4.0"`. **Archive index** `SCHEMA_VERSION`
  10 → 11 (columns removed; AGENTS.md: indexes must be empty or current complete schema; refusal message
  in `index._check_schema` unchanged).
- **Verdicts artifact** `recommendation-verdicts.json` has no version constant; precedent #1896 dropped
  selection sub-keys with only a CHANGELOG Breaking entry. Drop the `selection` block, no bump.
- **Analyzer `evaluation.json`** has no version and no reader of lens keys; drop keys, no bump.
- **Keep** the `MERGED_ITEMS_SCHEMA` lens enum (`phases/schemas.py:19`, only `per-stack`): it backs the
  "retired lens fails the findings check" refusal (`review_terminal.py:194`,
  `tests/deep_orchestrator/test_review_completion.py` case `retired-lens`).
- CHANGELOG `[Unreleased]`: one `**Breaking (0.y.z):**` entry (Removed) linking #1895: run record v5
  (no `findings.claims`/`derivation`, no `effective_configuration.model/deep`), manifest 4.0 / index 11
  (no `run.model`/`run.deep`), verdicts without `selection`, evaluation without `whole_file_anchors`,
  `items[].lens`, `pairs[].a_lens/b_lens`. Earlier records/indexes refused with existing messages; nothing
  converted. Do not edit released entries.

## Lane A — verifier selection (owner: verify/fix + UI)

- Delete `daydream/deep/verify_selection.py` entirely (`SelectionDecision`, `select_items`).
- `daydream/phases/verify.py` `phase_verify_recommendations` (L78-127): drop import (L13-16),
  `decisions`/`selection_block`/`pending_items` (L95-101), the `if not pending_items` early-return branch
  (L103-107; unreachable from flows — fix gate stops on empty items at `fix_steps.py:187-189`), the
  `payload["selection"]` line, docstring selection prose; pass `items=items`. Change `_asked_verdicts`
  (L130-133) to take `items` and keep filtering stray `issue_id`s by integer item `id` (real contract).
- `daydream/deep/fix_steps.py`: remove `item_uid` import (L24, only used L291), `_step_verify` comment
  L264 and `skipped=` arg L269, the skipped list / "omits canonical item" raise / `selected is False`
  branch in `_verdict_buckets` (L276-302), delete `_selection_decisions` (L305-318).
- `daydream/ui/summary.py`: `format_verdict_join` drop `skipped` param/docstring/`("Skipped", ...)` row;
  `print_verification_summary` drop L192-198 + docstring; delete `_is_count` (L205-207).
  `daydream/ui/__init__.py` re-exports `format_verdict_join` (L33, L58) — update if signature use changes.
- `daydream/dataset_capture.py:84-85`: drop keys `verify_decision` and `verifier_reason` (nothing writes
  them; keep `verifier_verdict`, `unverified_assumptions`).
- Unrelated "Skipped" hits — leave: `ui/colorize.py:33`, `pr_review.py:1042`, `dataset_hub.SkippedRecord`,
  `improve/*`, `phases/fix_fanout.items_skipped`, `trajectory/types.py:70`, harvest, search, bot_setup.
- Tests: delete `tests/test_verify_selection.py` (whole file). Its stray-`issue_id` proof
  (`test_fresh_verdicts_bind_only_to_the_findings_asked_about`) is the only unique assertion: retarget it
  to `_asked_verdicts(verdicts, items)` and move it into the existing verify test module (e.g.
  `tests/test_phases.py`) — a move, not a new test. Delete
  `tests/deep_orchestrator/test_fix_gate_cleanup_and_precision.py::test_verdict_buckets_reject_unchecked_or_missing_selection`
  (L91-97) and its now-unused `fix_steps` import (L11). Edit `tests/test_runner.py:685-688` `_stub_verify`
  → `{"verdicts": []}`; delete selection asserts `tests/test_archive_data_capture.py:212-214`; delete
  `tests/test_ui.py:64` (`"selected" not in out`).

## Lane B — dataset record v5 (owner: dataset*, training, trace-analysis skill, shared v5 literal)

- `daydream/dataset.py`: delete `_CLAIM` (L72), `claims=` (L89-90) and `derivation=_MAP` (L91) in the
  findings schema (keep `items`, `terminal_coverage`); in `_validate_run_evidence` (L365-368) delete the
  claims comprehension and the `len(set(claims)) != len(claims) or` clause (keep duplicate-identity msg).
  v4→v5 at L99, 142, 165, 172, 182, 220. Optional: a single module constant if it removes literals.
- `daydream/dataset_capture.py:183` v5; L196-197 drop comment + `"claims": []`, `"derivation": {}`.
- `daydream/dataset_hub.py:32` `_SCHEMAS` v5.
- `daydream/training/record_evidence.py:160-170` (claim join `source = next(...)`) and L177 (`**source,`)
  in `sessions_from_snapshot` — delete; `stack` already comes from provenance
  (`corpus_projection/projector.py:91-98`).
- Not targets: `training/harvest.py:168`, `archive/pipeline.py:742` (prose), committed_review claims.
  Keep record `items[].source_uids` (real data).
- Tests: `tests/harness/dataset.py:9-13` (v5, drop claims/derivation), `tests/harness/record_projection.py:45-46`,
  `tests/test_dataset_store.py:79-81`, `tests/test_dataset_capture.py:126-128` (v5, comment),
  `tests/test_dataset_hub.py:51,204,238`. In `tests/test_corpus_projection.py:204`
  `test_cli_posterior_annotation_preserves_captured_native_profile_with_claim_stack` delete L212, 214-215,
  233 and drop `_with_claim_stack` from the name.
- External readers (`make test-scripts` runs them): `.agents/skills/daydream-trace-analysis/`
  `references/format.md:6` (v5) and `:22` (claims/derivation → items/terminal_coverage);
  `scripts/cohort.py:95-96`, `scripts/trace.py:4,94`, `scripts/fixture_evidence.py:64-71`
  (`v4_document`/`v4_record` → v5 names, as last bump did), `scripts/test_cohort.py`,
  `scripts/test_reports.py`, `scripts/test_trace.py`. `docs/observability.md:71` v5.

## Lane C — manifest/index + analyzer (owner: archive/, run_artifacts, run_snapshot, eval/)

- `daydream/archive/manifest.py`: `MANIFEST_SCHEMA_VERSION` "4.0" (L29); drop `Manifest.model` (L123),
  `Manifest.deep` (L130), `to_dict` keys (L193, L202), builder lines (L280, L292).
- `daydream/run_snapshot.py:49,56` drop `ManifestRunIdentity.model/.deep`;
  `daydream/run_artifacts.py:324,331` drop `model=None`, `deep=True`.
- `daydream/archive/_schema.py`: `SCHEMA_VERSION = 11` (L8); drop `RunColumn("model")` (L30),
  `RunColumn("deep")` (L37). `daydream/archive/index.py:126` drop `"deep": int(manifest.deep)`.
- `daydream/eval/analyzer.py`: delete `whole_file_anchors` (L321, branch L341-345, key L378),
  `location.items[].lens` (L363), `pairs[].a_lens/b_lens` (L439, 443); fix docstrings L310, 392, 432.
  Optional stale prose: `pr_review.py:87,178`, `improve/prioritize.py:324` ("structural" lens).
- Tests/scripts: `tests/test_archive_data_capture.py:157-171` "3.0"→"4.0" only;
  `tests/test_archive_runs_column_declaration.py:55-63` "earlier release" case → version 10 vs 11;
  `tests/test_runner.py:840` drop `review.deep`; drop `model=`/`deep=` kwargs in `tests/test_archive.py:168-170`,
  `tests/test_archive_integration.py:38,42`, `scripts/measure_finalization_memory.py:160-162`;
  `tests/harness/trajectory.py:212` drop `"model": "opus"` default if it feeds the manifest identity.
  `tests/test_analyzer.py`: remove `lens` from `_item` helper (L869,874), delete L704 (wonder lens lines),
  in L903-987 drop `whole_file_anchors`, `lens=` args, the `lenses` assertions; item 5 (line 0) is now
  scored like any line — adjust counts or the item.

## Validation (from Makefile/CONTRIBUTING)

`make hooks` once; focused: `uv run pytest <files>`, `make lint typecheck deadcode test-scripts check-naming`;
gate: `make check` (pre-push runs it; never bypass; coverage ratchet per docs/coverage.md — must not drop).
PR >5 files → run `.agents/skills/daydream-deadcode-sweep` in PR mode before publication; test changes →
`.agents/skills/test-audit`. PR body from `.github/PULL_REQUEST_TEMPLATE.md` via `.agents/skills/pr`.
