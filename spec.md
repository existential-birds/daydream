# Daydream #1093 — Remove Legacy Loaders and Versioned Project Names

**Created:** 2026-09-11
**Status:** Ready for planning
**Source:** existential-birds/daydream issue #1093 (binding scope correction of 2026-09-02) · explore.md @ main `a527c97`
**Requirer:** 697971929952550984 · Thread: 1518562888841691136:1547919994891800629

## Core Value

The training and rollout surfaces carry exactly one canonical, neutrally-named frozen-corpus implementation — no legacy loaders, no internal `v1`/`v2` project-owned names, and no stale claims of production runs — so #1097 can freeze the first production corpus on unambiguous ground.

## Problem Statement

#1081/#1092 made the frozen corpus-v2 projection the canonical training input, but the repo still ships two generations of loaders, the legacy `daydream train --corpus` branch, versioned module/CLI/task-identity names, and docs claiming production runs that never happened. Because this is greenfield (no production training run has ever completed), none of it deserves compatibility treatment — yet it still has to be pruned and renamed in one atomic cleanup before #1097 freezes the first production corpus, after which renames would break real data lineage.

## Requirements

### Must Have

- `daydream train` accepts only the neutral canonical frozen-corpus input; the `--corpus` flag, the `PipelineConfig.corpus` field, and the `stacks.load_dataset` branch are gone.
- No supported production code imports or calls the removed legacy corpus loaders (`stacks.load_dataset`, the harvested-`index.json` loader `rl/.../corpus.py`, and their taskset/image-builder caller branches).
- Legacy-workflow-only fixtures, tests, configs, and launch instructions are removed (legacy records fixtures, `tests/fixtures/corpus-mini|corpus-reference` harvested-index fixtures, `test_training_stacks.py` v1-loader tests, legacy `corpus build` verb docs/tests, `plan-notes.md` if it exists only for the legacy workflow).
- No project-owned tracked filename, path, package/module name, CLI/API name, task identity, config identifier, or user-facing title contains internal `v1`/`v2` naming. Audit list (from explore §B): `rl/daydream_review_v1/` (dir + package + taskset/harness ids `daydream-review-v1`), `daydream/training/corpus_v2/`, `stacks_v2.py`, `rubric_v2.py`, `daydream/training/schema/{v1,v2}.json`, `curation-manifest-v1.json` + fixture, `build_corpus_v2_50.py` fixture, `daydream corpus build-v2`, `run_build_corpus_v2()`, `derive_curation_id_v2()`, versioned test filenames (`test_corpus_v2*.py` ×7, `test_stacks_v2_{gate,load}.py`, `test_training_contract_v1_v2.py`, `test_training_coordinator_v2.py`, `test_training_rft_v2_sha.py`, `test_training_rubric_v2.py`).
- Renamed versioned-identifier-in-data strings (`member-v1`, `calibration-artifact-v1`, `claude-pretooluse-v1`, `codex-cli-*-v1`, `auto-v1`, `AUDIT_ROOT_ISOLATION_V1`) are each classified: renamed as project-owned, or documented as an external contract. Data fixtures pinning them are updated in the same change (greenfield ⇒ rename-safe).
- The canonical frozen-corpus projection/loading path is preserved intact: split + lineage validation, C5/C8 fail-closed enforcement, full-SHA task identity validation, bundles/license/provenance/segments/tiers.
- `rl/daydream_review_v1/` Stage-3 environment (Taskset/Harness/Runtime, taskset/image builders, 13 test files) is preserved and renamed — not deleted.
- The renamed RL environment is decoupled from the legacy harvested-index loader: no `CorpusSource`/`harvested_corpus()` construction or `config.corpus_dir` harvest plumbing remains in taskset.py / build_images.py / `configs/eval-{stub,docker}.toml`, leaving a clean seam for #1098 to bridge canonical reconstructable task records.
- If any harness preflight remains under a neutral name (it does), the `repo_path` writability check at harness.py:113 interpolates the path as one shell argument via `shlex.quote`, with regression tests covering whitespace and shell-significant paths (fold-in of closed #705).
- The `verifiers` pin/documentation skew is either resolved (pyproject pin aligned with the documented resolution) or explicitly left to #1098 with a note in the touched package metadata; whichever way, the state is stated, not ambiguous.
- CI `training-dry.yml` drives the canonical corpus input: a committed projection-dir fixture that passes `_SUCCESS`/lineage/split-digest verification replaces the `PipelineConfig(corpus=fixture)` records-JSONL drive.
- Documentation accurately describes the repository as having no completed production training run: stale claims removed from `CHANGELOG.md:366` and `docs/training-launch.md`; `README.md` points at the canonical pipeline only.
- Every rename updates all consumers atomically in one change set — no compatibility aliases, no dual paths, no deprecation shims (permitted because greenfield).
- `make check` and `make rl-check` pass after the change.

### Should Have

- A CI grep-gate (or documented check command) asserting zero project-owned `corpus_v2|stacks_v2|rubric_v2|daydream_review_v1|build-v2|run_build_corpus_v2` occurrences outside external contracts, so the naming decision can't regress.
- Remaining versioned occurrences that ARE external contracts (`verifiers.v1`, ATIF v1.7, vendor `/v1` paths, honeyhive `v2`, otel `v1` proto) carry a one-line "external contract" comment/notation at their site, per issue criterion 6.

### Out of Scope

- Bridging the RL environment to canonical reconstructable task records — owned by #1098; this issue only leaves the decoupling seam.
- Freezing the first production corpus — owned by #1097; this issue lands the naming decision before it.
- Any behavioral change to the canonical projection pipeline (split/lineage/C5/C8/SHA logic, reward rubric behavior) — rename-only, behavior preserved.
- Deprecation aliases or migration tooling — greenfield; no run history to migrate.
- CodeRabbit or review-thread tooling — governed by the EB merge workflow, not this issue.

## Constraints

- **Task identity:** taskset/harness `id` strings are task identity; the rename must be atomic across `rl/train/rl.toml` (×3), taskset.py defaults, Makefile, ci.yml, READMEs, and lockfiles — a half-renamed task identity silently forks the GRPO environment.
- **Timeline:** land the naming decision before #1097 freezes the first production corpus; after freeze, versioned-name renames would break corpus lineage.
- **Compatibility:** no production training run exists ⇒ rename-safe everywhere in project-owned data; conversely, once fixtures/content-addressed records are rewritten, nothing old may be kept.
- **Dependencies:** serialized after #721 (already merged as `a527c97`; rebase condition satisfied — verified).
- **Upstream decoupling:** the RL env's bridge to canonical task records belongs to #1098; leave the seam, don't build the bridge.

## Key Decisions

### Naming scheme for renamed surfaces
- **Decision:** Neutral names: `rl/daydream_review_v1/` → `rl/daydream_review/` (package `daydream_review`, task ids `daydream-review`); `daydream/training/corpus_v2/` → `daydream/training/corpus_projection/`; `stacks_v2.py` folded into a neutral `stacks.py` that keeps only the canonical loader (legacy `load_dataset` + `legacy_policy` deleted); `rubric_v2.py` → `rubric.py`; CLI `corpus build-v2` → `corpus build` (legacy `corpus build` verb dies with the v1 branch first); `run_build_corpus_v2()` → `build_frozen_corpus()`; `v2.json` → `record-schema.json` (v1.json deleted with the legacy branch); `build_corpus_v2_50.py` → neutral fixture-script name.
- **Alternatives considered:** `corpus/` as the package name (rejected — `daydream/training/corpus.py` already exists, a confusing same-name pair); folding `corpus_v2/` into `corpus.py` (rejected — 9 modules merged into a 1,236-line file worsens structure with zero benefit); `projection/` (viable, `corpus_projection/` preferred for self-description).
- **Rationale:** Descriptive neutral names with no generation markers, no collision with surviving modules, and a single canonical surface per concept.

### Composition with the surviving `corpus.py` module
- **Decision:** `daydream/training/corpus.py` (harvest/projection pipeline) is NOT the frozen-projection loader and stays; the new `corpus_projection/` package sits alongside it. `needs-spike-before-planning`: before plan-lock, verify which parts of `corpus.py` are legacy-only vs canonical after the v1-verb removal (its v1 `corpus build` projection path and `tests/test_training_corpus.py` are expected to die with the legacy branch — confirm against main so the two modules' surviving responsibilities don't overlap).
- **Alternatives considered:** Treating corpus.py as legacy and deleting wholesale (rejected — it contains non-legacy harvest logic and 15+ consumers; deletion is a separate decision).
- **Rationale:** The sweep shows both modules survive but their split is unverified; overlap there would recreate the two-generations problem the issue exists to remove. **Spike required:** before plan-lock, verify the legacy/canonical split of `corpus.py` (legacy `corpus build` verb, `run_build_corpus` vs v2 projector) against main and revise this decision if the result diverges.

### Legacy deletion surface (explore §A)
- **Decision:** Remove exactly: `stacks.load_dataset` + `legacy_policy` stamping; `--corpus` CLI arg + `PipelineConfig.corpus` + coordinator else-branch; `rl/.../corpus.py` (`EvaluablePR`/`CorpusSource`/`harvested_corpus()`) + the taskset config.corpus_dir branch + build_images DEFAULT_CORPUS plumbing; legacy-only fixtures (`corpus-mini`, `corpus-reference`, records-50 as a loader input), `tests/test_training_stacks.py`, v1-slices of coordinator/contract tests, legacy launch docs, `plan-notes.md` if legacy-only. Keep the Taskset/Harness/Runtime and image builders.
- **Alternatives considered:** Deleting all of `rl/daydream_review_v1/` (rejected by the binding scope correction — only real Stage-3 verifiers env; deletion conflicts with #91 and strands GRPO).
- **Rationale:** Follows the September 2 scope correction exactly: loader dies, environment survives renamed and decoupled.

### verifiers pin/doc skew
- **Decision:** Resolve while touching package metadata: align `rl/daydream_review/pyproject.toml` pin to the documented resolution (`verifiers==0.2.1`, vendored by prime-rl v0.7.0, per the README/0.2.1 rationale and `test_vendored_verifiers_suite.py` shadowing) and reconcile the comment block. `needs-spike-before-planning`.
- **Alternatives considered:** Deferring to #1098 (allowed by the issue, rejected because the metadata is already being touched and the README documents the intended resolution — leaving a known-wrong pin through a rename is noise).
- **Rationale:** The repo's own docs and test shadowing say 0.2.1 is the resolution in use; the 0.3.1 pin is skew, not a decision. **Spike required:** before plan-lock, verify the vendored-suite shadowing actually resolves imports to 0.2.1 in the renamed env (uv run + PYTHONPATH) and revise this decision if a 0.3.1 dependency emerges.

### CI training-dry fixture
- **Decision:** Commit a small projection-dir fixture (generated by a committed script in the `build_corpus_v2_50.py` lineage) that passes `_SUCCESS`/lineage/split-digest verification, and repoint `training-dry.yml:54` to `PipelineConfig` with the canonical input.
- **Alternatives considered:** Repointing to a live build inside CI (slower, brittle); keeping a records fixture via a compat shim (forbidden — no aliases).
- **Rationale:** The dry-run must exercise the same canonical path production will use; a committed verified projection fixture is the only way to do that without the legacy loader.

### Versioned identifier classification
- **Decision:** Rename all versioned-in-data strings found by the audit (`member-v1`, `derive_curation_id_v2`, `calibration-artifact-v1`, event kinds, `AUDIT_ROOT_ISOLATION_V1`) as project-owned — safe because only fixtures have written them. External contracts (`verifiers.v1`, ATIF v1.7, vendor `/v1`, honeyhive `v2`, otel `v1` proto) stay and are annotated per Should-Have.
- **Alternatives considered:** Preserving data-identifiers as contracts (rejected — nothing outside fixtures consumes them; preserving would be exactly the dead-compatibility surface the issue removes).
- **Rationale:** Greenfield: no data written by real runs exists, so identifier renames have zero external blast radius.

## Reference Points

- Issue #1093 body + binding September 2, 2026 scope correction (authoritative scope).
- Issue #705 (closed) — shell-safe `repo_path` preflight requirement and its regression tests; carried forward at harness.py:113.
- #1098 (RL env ↔ canonical task-record bridge) and #1097 (first production corpus freeze) — the downstream consumers that fix this issue's ordering.
- #164 — the Stage-3 verifiers environment being preserved and renamed.
- explore.md at `/opt/data/workspaces/daydream/issue-1093/explore.md` — line-level audit (§A removal surfaces, §B rename table, §D #705 finding, §E pin skew).

## Open Questions

- None blocking — the scope correction resolved the only genuine ambiguity (preserve-and-rename the RL env). The three `needs-spike-before-planning` markers above are verification work for the planner, not unanswered spec questions.

## Future Considerations

- #1098: bridge the renamed RL environment to canonical reconstructable task records (replaces the removed `CorpusSource` seam).
- #1097: freeze the first production corpus (unblocked once this naming decision lands).
- Revisit `member-v1`-style content-addressed identifiers if external consumers of published plan records ever appear — then they become real contracts and gain documentation.
- Fold remaining docs cleanup (`plan-notes.md`, launch-doc numbers) into a broader docs-accuracy pass if new drift appears after #1097.