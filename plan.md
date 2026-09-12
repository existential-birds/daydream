# Daydream #1093 — Remove Legacy Loaders and Versioned Project Names: Implementation Plan

> **Source spec:** `/opt/data/workspaces/daydream/issue-1093/spec.md`
> **For downstream agents:** Execute task-by-task. Each task uses `- [ ]` checkboxes for tracking. Do not skip the test-first steps — they catch wiring bugs that pure-logic tests catch nowhere else.

**Goal:** Leave the training and rollout surfaces with exactly one canonical, neutrally-named frozen-corpus implementation — legacy loaders gone, every project-owned `v1`/`v2` name renamed, no stale production-run claims — so #1097 can freeze the first production corpus on unambiguous ground.

**Architecture:** Four atomic slices, ordered so each leaves the tree green: (1) delete the legacy v1 corpus surfaces (`--corpus` flag, `stacks.load_dataset`, the vendored harvested-index loader and its taskset/image-builder callers, legacy-only fixtures/tests/docs); (2) rename the surviving canonical surfaces to neutral names (`corpus_v2/` → `corpus_projection/`, `stacks_v2.py` folded into neutral `stacks.py`, `rubric_v2.py` → `rubric.py`, `corpus build-v2` → `corpus build`, the RL env directory/package/task-identity rename, schema/fixture renames); (3) classify and rename remaining versioned data identifiers (`member-v1`, `calibration-artifact-v1`, `AUDIT_ROOT_ISOLATION_V1`); (4) re-wire CI (`training-dry.yml`) to a committed projection fixture, carry the #705 `shlex.quote` fold-in forward, resolve the verifiers pin skew, and land the zero-versioned-names CI grep gate. Every rename updates all consumers in the same task — no compatibility aliases (greenfield: no production training run has ever completed).

**Tech Stack:** Python 3.12 (uv-managed), pytest, argparse CLI, TOML configs, GitHub Actions CI, prime-rl v0.7.0 + verifiers (vendored 0.2.1) for the RL env, uv lockfiles.

---

## Spike Findings (Task 0 executed during planning against main @ `a10e0af`)

All three spec `needs-spike-before-planning` markers were verified against a fresh clone of `main` (`a10e0af`, past the #721 merge `a527c97`). The spec's Key Decisions survive intact, with these concrete refinements:

1. **corpus.py legacy/canonical split (Spike 1) — spec decision CONFIRMED with one refinement.** `daydream/training/corpus.py` is a *pipeline module* (harvest/projection of curated annotations into a records JSONL), not just the legacy verb handler. Its helpers are canonical, load-bearing dependencies of the frozen projection: `corpus_v2/projector.py` imports `_is_posterior_leak, _trajectory_set_hash`; `corpus_v2/segments.py` imports `_build_spans`; `corpus_v2/provenance.py` imports `_stack_for_skill`; `reward_model.py` imports `_is_admitted_outcome_gold`. **However** the legacy surface *inside* it is only the surface `run_build_corpus` + `corpus build` verb serve: producing a v1-schema records JSONL. After this issue's deletion of the only loader of that JSONL (`stacks.load_dataset`, removed here), `run_build_corpus` has **zero** production consumers (verified: only the legacy coordinator branch and tests call it). Decision refined: keep `corpus.py`'s helpers + `BuildCorpusConfig`/`CorpusFilters` (shared infra), **delete `run_build_corpus`, `BuildCorpusConfig.out_path`/`dry_run`/`emit_schema_only` legacy emission path, and the `corpus build` verb** in this same change set — it is the other half of the legacy loader branch, its deletion is implied by the issue's "no supported production code imports or calls the removed legacy corpus loaders," and keeping it would preserve a producer of v1-schema records that nothing can load.
2. **verifiers pin (Spike 2) — spec decision CONFIRMED.** `rl/daydream_review_v1/pyproject.toml` pins `verifiers==0.3.1`; README (lines 22, 221-228) and `tests/test_vendored_verifiers_suite.py` both document 0.2.1 as the resolution (vendored by prime-rl v0.7.0, shadowed via PYTHONPATH). The uv.lock resolves 0.3.1. Align the pin to `==0.2.1` and re-lock; the vendored-shadowing test is the verification harness.
3. **training-dry fixture (Spike 3) — spec decision CONFIRMED.** `training-dry.yml` drives `PipelineConfig(corpus=fixture)` over `tests/fixtures/training/records-50/records.jsonl` (the legacy path). No committed projection-dir fixture exists. `tests/fixtures/training/build_corpus_v2_50.py` exists and generates the projection deterministically — extend/rename it to emit a committed projection dir that passes `_SUCCESS`/lineage/split-digest verification.

## Assumptions

- **RL-env image builder keeps a fixture-driven PR loop.** The decoupling removes the *harvested-index* corpus source (`CorpusSource`/`harvested_corpus()` over `index.json`), but `images/build_images.py` still needs PR inputs (clone_url/base_sha/head_sha) to build PR-snapshot images. Assumption: those inputs are re-sourced from a neutral in-repo task-source module (the fixture + manifest entries already carry `clone_url`/SHAs via `fixture://` and the corpus-reference fixture), with the manifest as the source of truth for repo entries. If the executor finds `build_images.py` structurally cannot build non-fixture images without the harvested corpus, stop and re-scope the decoupling seam with the orchestrator — #1098 expects the image builders preserved.
- **`member-v1` rename changes the content-addressed prefix.** `member-v1:<sha>` aliases (improve/prioritize.py) are renamed project-owned; the fixture-only blast radius is `tests/test_improve_publish.py`. Greenfield permits this; if a hidden consumer surfaces mid-task, stop and classify it as an external contract instead.
- **`calibration-artifact-v1` and `AUDIT_ROOT_ISOLATION_V1` are project-owned and renamed** (`calibration-artifact`, `AUDIT_ROOT_ISOLATION`). `claude-pretooluse-v1` / `codex-cli-0.153.4-json-code-mode-v1` / `auto-v1` event strings stay — they name external tool capabilities/protocol versions, not project-owned generations, and gain the Should-Have external-contract annotation.
- **#1098 owns any verifiers follow-up.** After the pin resolves to 0.2.1 here, nothing about verifiers remains ambiguous; the spec's "or explicitly leave it to #1098" branch is not taken.
- Re-reading the files the spec named surfaced two corrections vs the explore snapshot: `_corpus_subverbs` in cli.py is spelled `_CORPUS_SUBVERBS` (lowercase variant in explore), and `run_build_corpus_v2` lives at `corpus_v2/projector.py:826` (explore said 826 — confirmed). Both are cosmetic; noted here so the executor trusts the explore map.

## Patterns

### Pattern: Atomic rename (move + all-consumer update in one commit)
**Applies when:** a task renames a module/package/file/CLI name (Tasks 4-7).
**Reference example:** Task 4's `corpus_v2/` → `corpus_projection/` rename.
**Transformation:** `git mv` the file/dir → update every import/attribute/config/lockfile/doc reference found by `grep -rn '<old>'` (excluding external contracts) → run the subject's test file + the module suite → commit. Never leave a re-export shim, alias, or stale docstring.

### Pattern: Legacy-surface deletion (callers first, then surface, then orphans)
**Applies when:** a task deletes a loader/branch (Tasks 2-3).
**Reference example:** Task 2's `--corpus` removal.
**Transformation:** rewrite/delete each caller → delete the surface → delete orphaned fixtures/tests/docs that exist only for the removed path → run the module suite → commit.

---

## File Structure

### Files to Delete
- `daydream/training/stacks.py` legacy half: `load_dataset` + `legacy_policy` stamping (module survives folded into neutral `stacks.py`, Task 5)
- `daydream/training/stacks_v2.py` (content moved into `stacks.py`, Task 5)
- `daydream/training/corpus.py` legacy emission: `run_build_corpus` + records-JSONL writing (helpers stay, Task 3)
- `tests/test_training_stacks.py` (v1-loader tests, Task 2)
- `tests/fixtures/training/records-50/` (legacy loader input, Task 2)
- `tests/fixtures/corpus-mini/`, `tests/fixtures/corpus-reference/` (harvested-index fixtures, Task 3)
- `rl/daydream_review_v1/daydream_review_v1/corpus.py` (vendored harvested-index loader, Task 3 — dir renamed first in Task 6 ordering; see Task ordering note)
- `tests/test_training_corpus.py` v1 slices / legacy-only tests (Task 3)
- `plan-notes.md` (repo root, legacy-workflow-only, Task 8)
- `daydream/training/schema/v1.json` (v1 record schema, dies with the v1 branch, Task 7)

### Files to Create
- `tests/fixtures/training/projection-50/` — committed frozen-projection fixture dir (`_SUCCESS`, `SHA256SUMS`, `curation-manifest.json`, `corpus.jsonl`, split manifests, `lineage.json`), generated by the renamed `build_projection_50.py` (Task 9)
- `tests/fixtures/training/build_projection_50.py` — renamed from `build_corpus_v2_50.py`, extended to write the committed fixture dir (Task 9)
- `scripts/check-naming.sh` — CI grep gate: zero project-owned versioned names outside external contracts (Task 10)

### Files to Modify
- `daydream/cli.py` — remove `--corpus` arg, rename `corpus build-v2` → `corpus build`, delete legacy `corpus build` verb (Tasks 2, 4)
- `daydream/training/coordinator.py` — remove `PipelineConfig.corpus` + else-branch; rename `corpus_v2` field → `projection` (Tasks 2, 5)
- `daydream/training/{stacks.py,stacks_v2.py}` — fold into neutral `stacks.py` (Task 5)
- `daydream/training/corpus_v2/` → `daydream/training/corpus_projection/` (Task 4)
- `daydream/training/rubric_v2.py` → `daydream/training/rubric.py` (Task 4)
- `daydream/training/schema/{v2.json → record-schema.json, v1.json deleted}` (Task 7)
- `daydream/improve/prioritize.py`, `daydream/training/calibration.py`, `daydream/backends/__init__.py` (+ consumers) — versioned data-identifier renames (Task 8)
- `rl/daydream_review_v1/` → `rl/daydream_review/` (dir, inner package, uv.lock, task ids, Makefile, ci.yml, READMEs, pyproject) — Task 6
- `rl/daydream_review/{taskset.py, images/build_images.py, configs/eval-{stub,docker}.toml}` — decouple from legacy loader (Task 6)
- `rl/daydream_review/daydream_review/harness.py:113` — `shlex.quote(repo_path)` (#705 fold-in, Task 8)
- `rl/daydream_review/pyproject.toml` + `uv.lock` — verifiers pin 0.3.1 → 0.2.1 (Task 8)
- `.github/workflows/training-dry.yml` — repoint to committed projection fixture (Task 9)
- `.github/workflows/ci.yml`, `Makefile` — add `check-naming` gate (Task 10)
- `README.md`, `docs/training-launch.md`, `CHANGELOG.md:366` — stale-claim purge (Task 8)

> **Task-ordering note:** the RL directory rename (Task 6) precedes the RL decoupling content edits — decoupling is written directly at the renamed paths so no intermediate path churn occurs. The vendored `corpus.py` deletion listed under Task 3's *content* is executed inside Task 6 at the renamed location.

---

## Task 1: Baseline — clone, environment, and green suite

**Files:** none (verification only)

- [ ] **Step 1: Clone and sync**

```bash
git clone https://github.com/existential-birds/daydream /tmp/daydream-1093-impl && cd /tmp/daydream-1093-impl
git checkout -b eb/daydream/issue-1093 main
uv sync --all-extras
cd rl/daydream_review_v1 && uv sync --all-extras && cd ../..
```

- [ ] **Step 2: Baseline green**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -n auto -q tests/ -x --timeout=300`
Expected: PASS (0 failures) — record the count. If main is red, STOP and report; do not plan on a red base.

Run: `cd /tmp/daydream-1093-impl/rl/daydream_review_v1 && uv run pytest -n auto -q`
Expected: PASS (0 failures) — record the count.

- [ ] **Step 3: Commit** (empty commit marking baseline, optional) — skip; no tree change.

---

## Task 2: Remove the legacy `daydream train --corpus` branch

**Files:**
- Modify: `daydream/cli.py` (`_TrainParser` corpus_group, ~line 2282-2351)
- Modify: `daydream/training/coordinator.py` (`PipelineConfig.corpus`, `__post_init__` ~64-132, else-branch ~710-714)
- Delete: `tests/test_training_stacks.py`, `tests/fixtures/training/records-50/`
- Test: `tests/test_training_coordinator.py` (existing v1-slice tests are updated/removed here)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_training_coordinator.py` (module already has PipelineConfig construction tests — reuse its existing import block and temp-path helpers):

```python
def test_pipeline_config_rejects_legacy_corpus_kwarg():
    """#1093: the v1 `corpus` input is gone; the canonical input is `projection`."""
    from daydream.training.coordinator import PipelineConfig
    with pytest.raises(TypeError):
        PipelineConfig(out_dir=Path("/tmp/x"), corpus=Path("/tmp/corpus.jsonl"))  # type: ignore[call-arg]

def test_pipeline_config_projection_only():
    from daydream.training.coordinator import PipelineConfig
    cfg = PipelineConfig(out_dir=Path("/tmp/x"), projection=Path("/tmp/proj"))
    assert cfg.projection == Path("/tmp/proj")
```

Note: the second test passes only after the rename lands in Task 5 (`corpus_v2` → `projection`); at this task's Step 2 it fails with `TypeError: ... unexpected keyword 'projection'` — write both now, expect exactly the first to go green in this task and the second to stay red until Task 5. Mark the second with `@pytest.mark.xfail(reason="projection rename lands in Task 4 of #1093", strict=True)` so the suite stays green at every task boundary; Task 5 removes the marker.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_training_coordinator.py::test_pipeline_config_rejects_legacy_corpus_kwarg`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'corpus'` is the *desired* failure only after the fix; before the fix the test fails because `PipelineConfig(corpus=...)` constructs successfully and no `TypeError` is raised: `DID NOT RAISE TypeError`.

- [ ] **Step 3: Implement against the test**

**Files touched:** `daydream/cli.py`, `daydream/training/coordinator.py`

**Behavior contract:**
- `PipelineConfig` drops the `corpus: Path | None = None` field; `__post_init__` enforces exactly `projection is not None` (rename the existing `corpus_v2` field to `projection` **in this task** — the field, the exactly-one-of constraint collapse, and the docstring in one edit, so no intermediate state has two input fields).
- `run_pipeline` loses the `else:` legacy branch (`stacks.load_dataset` call): the projection path becomes unconditional (`if config.projection is None: raise ValueError("no projection input: PipelineConfig requires projection=<frozen projection dir>")` is unreachable but keeps the type-narrowing explicit per mypy strict).
- `cli.py` `_TrainParser` loses the `--corpus` argument and the mutually-exclusive `corpus_group` reduces to `--corpus-v2` alone, renamed `--projection` (dest `projection`) in the same edit; the coordinator construction at cli.py:2351 passes `projection=args.projection`.
- All error messages name the offending field (`projection`), never a legacy name.

**Reference:** `daydream/training/coordinator.py:690-713` — the projection branch already in place is the surviving shape; the else-branch dies.

- [ ] **Step 4: Run the new test AND the relevant suite, verify both green**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_training_coordinator.py`
Expected: PASS with zero regressions. Any test constructing `PipelineConfig(corpus=...)` (the legacy-slice tests the explore flagged) is deleted in this task — they are legacy-only.

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_corpus_v2.py tests/test_stacks_v2_load.py`
Expected: PASS (projection loader path untouched).

- [ ] **Step 5: Sweep modified files for leftovers**

In `coordinator.py` and `cli.py`: remove the `stacks`/`stacks_v2` imports that only the legacy branch used, doc-comment references to "v1 input"/"exactly one of corpus and corpus_v2", and the `corpus_group` help text mentioning two inputs. Delete `tests/test_training_stacks.py` and `tests/fixtures/training/records-50/` (orphaned with the loader). Grep `rg 'load_dataset\b' daydream tests` — expect zero outside `stacks.py` itself.

- [ ] **Step 6: Commit**

```bash
git add daydream/cli.py daydream/training/coordinator.py tests/test_training_coordinator.py
git rm -r tests/test_training_stacks.py tests/fixtures/training/records-50
git commit -m "refactor(training): remove legacy --corpus input, rename canonical input to projection"
```

---

## Task 3: Delete the legacy records builder + vendored harvested-index loader (content)

**Files:**
- Modify: `daydream/training/corpus.py` (delete `run_build_corpus`, records emission)
- Modify: `daydream/cli.py` (delete `corpus build` verb handler `_handle_build_corpus_command`, `_build_build_corpus_parser`, `_CORPUS_SUBVERBS["build"]`)
- Delete: `tests/test_training_corpus.py` legacy-only tests, `tests/fixtures/corpus-mini/`, `tests/fixtures/corpus-reference/`
- Defer: `rl/daydream_review_v1/daydream_review_v1/corpus.py` deletion happens in Task 6 (dir renamed there first); this task only verifies the main-tree surface.

- [ ] **Step 1: Write the failing test**

In `tests/test_corpus_v2.py` (which explore flagged at lines 740-776 as importing `load_dataset`), replace the v1-loader assertions with a deletion pin:

```python
def test_legacy_records_builder_gone():
    """#1093: `run_build_corpus` (v1 records JSONL emission) is removed."""
    import daydream.training.corpus as corpus_mod
    assert not hasattr(corpus_mod, "run_build_corpus")

def test_corpus_build_verb_gone():
    """#1093: the legacy `daydream corpus build` verb no longer dispatches."""
    from daydream.cli import _CORPUS_SUBVERBS
    assert "build" not in _CORPUS_SUBVERBS  # build-v2 renamed in Task 4 lands as "build"
```

Note: after Task 4 renames `build-v2` → `build`, the second assertion flips meaning. Write it now pinning `"build" not in` AND `"build-v2" not in` (both verbs absent at this task's end); Task 4 updates it to pin exactly `{"build"}` as the sole subverb.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_corpus_v2.py::test_legacy_records_builder_gone tests/test_corpus_v2.py::test_corpus_build_verb_gone`
Expected: FAIL — `assert not hasattr` fails since `run_build_corpus` exists; second fails on `"build" in _CORPUS_SUBVERBS`.

- [ ] **Step 3: Implement against the test**

**Files touched:** `daydream/training/corpus.py`, `daydream/cli.py`

**Behavior contract:**
- `daydream/training/corpus.py` keeps `_build_spans`, `_build_record`, `_annotation_*`, `_resolve_projection_path`, `CorpusFilters`, `_stack_for_skill`, `_build_query`, `_query_index`, `_stratify`, `_is_posterior_leak`, `_rubric_decisive_only`, `_is_admitted_outcome_gold`, `_is_admitted`, `_trajectory_set_hash`, `_collapse_versions`, `_summary` (all have canonical consumers — Spike 1) and DELETES `run_build_corpus`, `BuildCorpusConfig`, and the module-level records-JSONL write path (`run_build_corpus`'s body + its docstring reference at corpus.py:47,998).
- `cli.py` deletes the `corpus build` verb: `_handle_build_corpus_command`, `_build_build_corpus_parser`, and the `"build":` entry in `_CORPUS_SUBVERBS`; `_CORPUS_USAGE` drops the build line.
- Errors propagate via existing ValueError/argparse paths — no `.unwrap_or`-style fallbacks introduced; deleted code paths raise nothing because they no longer exist (callers were deleted in Task 2 or are tests deleted here).

**Reference:** `daydream/training/corpus.py:984-1236` (`run_build_corpus` + emission) is the deletion surface; helper consumers listed at Spike 1 stay.

- [ ] **Step 4: Run the new test AND the relevant suite, verify both green**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_corpus_v2.py::test_legacy_records_builder_gone tests/test_corpus_v2.py::test_corpus_build_verb_gon`
Expected: PASS.

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_corpus_reproducibility.py tests/test_training_query.py tests/test_corpus_leakage.py tests/test_training_record.py tests/test_corpus_lineage.py tests/test_corpus_v2_reproducibility.py tests/test_training_stratify.py tests/test_archive_sanitize.py`
Expected: PASS — these import the surviving helpers; any failure names a helper accidentally deleted in Step 3 (restore it, it has a canonical consumer).

- [ ] **Step 5: Sweep modified files for leftovers**

In `corpus.py`: remove the module docstring's references to `run_build_corpus` as the entry point, the `dry_run`/`emit_schema_only` mentions, and any import that only `run_build_corpus` used (e.g. `sys`, `time` if now unused). In `cli.py`: remove the `--as-of`/`--max-stack-share` parser fragments that only the deleted parser built, plus orphaned imports. Delete `tests/test_training_corpus.py` tests that import `run_build_corpus`/`BuildCorpusConfig` (grep first: keep any test importing only `_build_record`/`_build_spans` helpers — they test canonical code; move those to a surviving file if the whole file would otherwise die). Delete `tests/fixtures/corpus-mini/` and `tests/fixtures/corpus-reference/`.

- [ ] **Step 6: Commit**

```bash
git add daydream/training/corpus.py daydream/cli.py tests/test_corpus_v2.py
git rm -r tests/test_training_corpus.py tests/fixtures/corpus-mini tests/fixtures/corpus-reference
git commit -m "refactor(training): delete legacy records builder and corpus build verb"
```

---

## Task 4: Rename `corpus_v2/` → `corpus_projection/`, `rubric_v2.py` → `rubric.py`

**Files:**
- Move: `daydream/training/corpus_v2/` → `daydream/training/corpus_projection/` (9 modules)
- Move: `daydream/training/rubric_v2.py` → `daydream/training/rubric.py`
- Modify: every importer (cli.py, coordinator.py, adjudication/* ×7, archive/* ×3, reward_model.py, rft.py, stacks.py/stacks_v2.py, 15+ test files)

- [ ] **Step 1: Write the failing test**

New file `tests/test_neutral_names.py` (this file is extended by Tasks 5-8; it is the rename gate):

```python
"""#1093 naming gate: no project-owned v1/v2 names in importable surfaces."""
import importlib

def test_canonical_projection_package_imports_under_neutral_name():
    m = importlib.import_module("daydream.training.corpus_projection")
    assert hasattr(m, "run_build_corpus_v2") is False  # renamed in this task
    assert hasattr(m, "build_frozen_corpus")
    assert hasattr(m, "BuildFrozenCorpusConfig")

def test_old_package_name_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("daydream.training.corpus_v2")

def test_rubric_module_neutral():
    importlib.import_module("daydream.training.rubric")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("daydream.training.rubric_v2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_neutral_names.py`
Expected: FAIL — `ModuleNotFoundError: daydream.training.corpus_projection` on the first import.

- [ ] **Step 3: Implement against the test**

**Files touched:** `daydream/training/corpus_v2/*` (moved), all importers

**Behavior contract:**
- `git mv daydream/training/corpus_v2 daydream/training/corpus_projection`; update all imports `daydream.training.corpus_v2(.*)` → `daydream.training.corpus_projection\1` across the tree (grep list from explore §B: cli.py, coordinator.py, adjudication/harvest.py, adjudication/publish.py, archive/hydrate_rules.py, archive/hydrate.py, reward_model.py, rft.py, plus tests).
- Inside the moved package: `run_build_corpus_v2` → `build_frozen_corpus`, `BuildCorpusV2Config` → `BuildFrozenCorpusConfig` (`__init__.py` exports updated; all call sites: cli.py:701,786,788, projector.py internals).
- `git mv daydream/training/rubric_v2.py daydream/training/rubric.py`; update its importers (rft.py:206 docstring, test import sites).
- `derive_curation_id_v2` → `derive_curation_id` (archive/hydrate_rules.py:74, hydrate.py, adjudication/publish.py).
- No behavior change: pure rename; error propagation unchanged (existing exceptions pass through).

**Reference:** Pattern *Atomic rename*; the import-update mechanics mirror any prior module move in `git log --oneline -- daydream/training/` — use `git grep` not memory for the site list.

- [ ] **Step 4: Run the new test AND the relevant suite, verify both green**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_neutral_names.py`
Expected: PASS.

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_corpus_v2.py tests/test_corpus_v2_license.py tests/test_corpus_v2_reproducibility.py tests/test_corpus_v2_schema.py tests/test_corpus_v2_share_caps.py tests/test_training_coordinator.py tests/test_training_contract_v1_v2.py`
Expected: PASS with zero regressions (imports updated).

- [ ] **Step 5: Sweep modified files for leftovers**

Grep `rg 'corpus_v2|rubric_v2' daydream tests` — expect zero outside `tests/test_neutral_names.py` pins and external-contract comments (none here). Update stale docstrings mentioning "corpus_v2" (coordinator.py:30,151,297,361; reward_model.py:159) to say "projection records" — wording, not behavior.

- [ ] **Step 6: Commit**

```bash
git add -A daydream tests
git commit -m "refactor(training): rename corpus_v2 package to corpus_projection, rubric_v2 to rubric"
```

---

## Task 5: Fold `stacks_v2.py` into neutral `stacks.py`

**Files:**
- Move+merge: `daydream/training/stacks_v2.py` content → `daydream/training/stacks.py`
- Delete: legacy `load_dataset` + `legacy_policy` from `stacks.py`
- Modify: `coordinator.py` import (`stacks_v2` → `stacks`), `tests/test_stacks_v2_{gate,load}.py` renames, `tests/test_corpus_v2.py:740-776`

- [ ] **Step 1: Write the failing test**

Extend `tests/test_neutral_names.py`:

```python
def test_stacks_neutral_surface():
    import daydream.training.stacks as stacks
    assert hasattr(stacks, "load_v2_projection")
    assert not hasattr(stacks, "load_dataset")          # legacy v1 loader gone
    assert not hasattr(stacks, "legacy_policy")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("daydream.training.stacks_v2")

def test_coordinator_uses_neutral_stacks():
    import daydream.training.coordinator as coord
    assert "stacks_v2" not in inspect.getsource(coord)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_neutral_names.py::test_stacks_neutral_surface`
Expected: FAIL — `stacks_v2` module still exists / `load_dataset` still present.

- [ ] **Step 3: Implement against the test**

**Files touched:** `daydream/training/stacks.py`, `coordinator.py`

**Behavior contract:**
- `stacks.py` retains only: `load_v2_projection`, `recompute_split_from_record_id`, `V2Projection`, and the C5/C8/split-identity enforcement helpers (`_enforce_v2_identity_and_gates`, `_v2_lineages`, `_enforce_license_gates` — the latter shared by the projection loader). `load_dataset`, `legacy_policy`, and the v1 docstring framing are deleted.
- `stacks_v2.py` is deleted; its wrapper content (`V2Projection`, `load_v2_projection`, `recompute_split_from_record_id`) moves into `stacks.py` unchanged in behavior.
- `coordinator.py` imports `from daydream.training.stacks import V2Projection, load_v2_projection, recompute_split_from_record_id`.
- `load_v2_projection` keeps its C5/C8 fail-closed ValueError propagation exactly (spec Must-Have: canonical path preserved).

**Reference:** `daydream/training/stacks.py:121-220` (the v2 loader) + `stacks_v2.py` (thin wrapper) — fold the wrapper into the loader module, delete the v1 half (lines 43-120, 40 `__all__` updated).

- [ ] **Step 4: Run the new test AND the relevant suite, verify both green**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_neutral_names.py`
Expected: PASS (the Task 2 `xfail` marker on `test_pipeline_config_projection_only` is now removed and the test asserts green).

Run: `cd /tmp/daydream-1093-impl && git mv tests/test_stacks_v2_load.py tests/test_stacks_load.py && git mv tests/test_stacks_v2_gate.py tests/test_stacks_gate.py && sed -i 's/daydream.training.stacks_v2/daydream.training.stacks/g' tests/test_stacks_load.py tests/test_stacks_gate.py && uv run pytest -q tests/test_stacks_load.py tests/test_stacks_gate.py`
Expected: PASS with zero regressions.

- [ ] **Step 5: Sweep modified files for leftovers**

Grep `rg 'stacks_v2|legacy_policy|load_dataset\b' daydream tests` — expect zero (the loader's `legacy_policy` stamping and v1 docstring at stacks.py:12-20 die with the fold). Remove the merged-away module's header docstring duplication.

- [ ] **Step 6: Commit**

```bash
git add daydream/training/stacks.py daydream/training/coordinator.py tests/
git rm daydream/training/stacks_v2.py
git commit -m "refactor(training): fold stacks_v2 into neutral stacks module, drop legacy loader"
```

---

## Task 6: Rename + decouple the RL environment (`rl/daydream_review_v1/` → `rl/daydream_review/`)

**Files:**
- Move: `rl/daydream_review_v1/` → `rl/daydream_review/`; inner `daydream_review_v1/` package → `daydream_review/`
- Delete (at new location): `rl/daydream_review/daydream_review/corpus.py` (vendored harvested-index loader)
- Modify: `rl/train/rl.toml` (ids ×3 at lines 81,90,109), `taskset.py`, `Makefile`, `.github/workflows/ci.yml`, `README.md:313`, `rl/train/README.md`, `pyproject.toml` (both root `daydream/pyproject.toml:137` comment and the env's own), uv.lock
- Modify (decoupling): `taskset.py` (drop `CorpusSource`/`harvested_corpus()` branch), `images/build_images.py` (drop `DEFAULT_CORPUS` + `--corpus`), `configs/eval-{stub,docker}.toml` (drop `corpus_dir`)

- [ ] **Step 1: Write the failing test**

New file `rl/daydream_review/tests/test_env_identity.py` (the env has its own test suite; this joins it):

```python
"""#1093: the Stage-3 environment is neutrally named and decoupled from the legacy loader."""
import importlib

def test_package_neutral_name():
    import daydream_review
    assert "daydream_review_v1" not in daydream_review.__file__

def test_task_identity_neutral():
    from daydream_review.taskset import DEFAULT_TASKSET_ID  # or wherever the id default lives
    assert DEFAULT_TASKSET_ID == "daydream-review"
    assert "v1" not in DEFAULT_TASKSET_ID

def test_legacy_loader_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("daydream_review.corpus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/daydream-1093-impl/rl/daydream_review_v1 && uv run pytest -q tests/test_env_identity.py`
Expected: FAIL — package dir still `daydream_review_v1`; import fails before assertions.

- [ ] **Step 3: Implement against the test**

**Files touched:** the moved tree, rl.toml, Makefile, ci.yml, READMEs, uv.lock

**Behavior contract:**
- `git mv rl/daydream_review_v1 rl/daydream_review` and inner `git mv rl/daydream_review/daydream_review_v1 rl/daydream_review/daydream_review`; all intra-package imports update mechanically.
- Task identity rename is atomic: `rl/train/rl.toml` taskset/harness `id = "daydream-review"` (×3 sites, lines 74-110), `taskset.py` id defaults, `Makefile` `rl-check` paths (line 20, 67), `.github/workflows/ci.yml` rl-check job, `README.md:313`, `rl/train/README.md:5,38,114`, `daydream/pyproject.toml:137`, env `pyproject.toml` name → `daydream-review`. Regenerate `rl/daydream_review/uv.lock` via `uv lock`.
- Decoupling: delete `daydream_review/corpus.py`; `taskset.py` loses the `harvested_corpus(config.corpus_dir)` source branch, the `CorpusSource` import (line 37), the `index.json` indexed-count warning block (lines ~997-1008), and `config.corpus_dir` plumbing, keeping the Stage-0 gate, C5 repo-exclusion, base_sha pinning, and manifest membership checks — re-homed to iterate the **image manifest's repo entries** (each `_ManifestEntry` already carries `clone_url`) plus golden comments from a neutral per-repo golden path; `configs/eval-{stub,docker}.toml` drop `corpus_dir`.
- `images/build_images.py` drops `DEFAULT_CORPUS` (line 66), the `--corpus` arg (line 455), the `harvested_corpus` import (line 46) and PR loop — image builds iterate `manifest.toml` entries directly (fixture slug via the `fixture://` sentinel materializes `daydream_review.fixture`; reference entry keeps its SHAs from the manifest). `_validate_red_flags`'s `prs` parameter takes the manifest-derived set.
- Comment header in manifest.toml and pyproject description updated to neutral names.

**Reference:** `rl/daydream_review_v1/daydream_review_v1/taskset.py:983-1060` — the surviving gate sequence is the contract; only the *source of `prs`* changes (manifest instead of corpus loader). Spike note: the executor must confirm the fixture-image build path still resolves PR SHAs from the manifest; if `build_images.py` cannot iterate manifest-only without corpus PRs, STOP and re-scope per the Assumption.

- [ ] **Step 4: Run the new test AND the relevant suite, verify both green**

Run: `cd /tmp/daydream-1093-impl/rl/daydream_review && uv run pytest -q tests/test_env_identity.py`
Expected: PASS.

Run: `cd /tmp/daydream-1093-impl/rl/daydream_review && uv run pytest -n auto -q`
Expected: PASS — all 13+ env test files green, zero regressions (test files themselves keep their names; they are not versioned).

Run: `cd /tmp/daydream-1093-impl && make rl-check`
Expected: PASS.

- [ ] **Step 5: Sweep modified files for leftovers**

Grep `rg 'daydream_review_v1|daydream-review-v1' --hidden -g '!.git'` from repo root — expect zero (uv.lock regenerated, ci.yml, Makefile, READMEs, pyprojects all updated). Grep `rg 'harvested_corpus|CorpusSource|corpus_dir' rl/daydream_review` — expect zero. Update manifest.toml's comment block that describes pairing with `tests/fixtures/corpus-reference` (that fixture died in Task 3).

- [ ] **Step 6: Commit**

```bash
git add -A rl Makefile .github/workflows/ci.yml README.md daydream/pyproject.toml
git commit -m "refactor(rl): rename daydream_review_v1 env to daydream_review, decouple from legacy corpus loader"
```

---

## Task 7: Rename schema/fixture filenames; purge stale production-run claims

**Files:**
- Move: `daydream/training/schema/v2.json` → `daydream/training/schema/record-schema.json`; Delete: `v1.json`, `curation-manifest-v1.json` (rename to `curation-manifest.json`)
- Move: `tests/fixtures/training/curation-manifest-v1-fixture.json` → `curation-manifest-fixture.json`; `tests/fixtures/training/build_corpus_v2_50.py` → `build_projection_50.py`
- Modify: schema.py, corpus.py (helpers), coordinator.py, cli.py, corpus_projection/* users of the schema filenames; all versioned test-file names
- Modify: `README.md:315`, `docs/training-launch.md`, `CHANGELOG.md:366`

- [ ] **Step 1: Write the failing test**

Extend `tests/test_neutral_names.py`:

```python
def test_schema_files_neutral():
    from pathlib import Path
    schema_dir = Path(daydream.training.__file__).parent / "schema"
    assert (schema_dir / "record-schema.json").is_file()
    assert not (schema_dir / "v1.json").exists()
    assert not (schema_dir / "v2.json").exists()
    assert (schema_dir / "curation-manifest.json").is_file()
    assert not (schema_dir / "curation-manifest-v1.json").exists()

def test_versioned_test_files_gone():
    from pathlib import Path
    tests_dir = Path(__file__).parent
    for stale in ("test_corpus_v2.py", "test_stacks_v2_load.py", "test_training_contract_v1_v2.py",
                  "test_training_coordinator_v2.py", "test_training_rft_v2_sha.py", "test_training_rubric_v2.py"):
        assert not (tests_dir / stale).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_neutral_names.py::test_schema_files_neutral tests/test_neutral_names.py::test_versioned_test_files_gone`
Expected: FAIL — `v2.json` still present / versioned test files exist.

- [ ] **Step 3: Implement against the test**

**Files touched:** `daydream/training/schema/`, the schema users, test filenames, docs

**Behavior contract:**
- `git mv daydream/training/schema/v2.json daydream/training/schema/record-schema.json`; update path constants/users in schema.py, corpus.py, coordinator.py, cli.py, corpus_projection/bundle.py (grep `rg "v2\.json|v1\.json|curation-manifest-v1" daydream tests`). `v1.json` is deleted (legacy-only per Spike 1 — the legacy records writer that consumed it died in Task 3).
- `git mv tests/fixtures/training/curation-manifest-v1-fixture.json tests/fixtures/training/curation-manifest-fixture.json` and `build_corpus_v2_50.py` → `build_projection_50.py` (its internal `run_build_corpus_v2` call already renamed in Task 4 to `build_frozen_corpus`; docstring updated to "mirrors the records-50" gone).
- Test-file renames via `git mv` with content updated: `test_corpus_v2*.py` ×7 → `test_corpus_projection*.py`, `test_stacks_{load,gate}.py` (done Task 5), `test_training_contract_v1_v2.py` → `test_training_contract.py`, `test_training_coordinator_v2.py` → `test_training_coordinator_projection.py`, `test_training_rft_v2_sha.py` → `test_training_rft_sha.py`, `test_training_rubric_v2.py` → `test_training_rubric.py`.
- Docs: `CHANGELOG.md:366` claim "feeding the SFT/RL corpus pipeline from real production runs" rewritten to state no production training run has completed; `docs/training-launch.md` legacy-loader sections removed (lines ~30-60 per explore §A4) and its "validation run numbers" reframed as fixture-scale validation; `README.md` pipeline pointers reference `corpus build` (canonical) only.

**Reference:** explore §B rename table rows for schema/fixture/test filenames; explore §F for doc claims.

- [ ] **Step 4: Run the new test AND the relevant suite, verify both green**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_neutral_names.py`
Expected: PASS.

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_corpus_projection.py tests/test_corpus_projection_schema.py tests/test_training_contract.py tests/test_training_coordinator_projection.py tests/test_training_rft_sha.py tests/test_training_rubric.py`
Expected: PASS with zero regressions (pure renames).

- [ ] **Step 5: Sweep modified files for leftovers**

Grep `rg 'v2\.json|v1\.json|curation-manifest-v1|build_corpus_v2_50' daydream tests docs README.md` — expect zero. Grep `rg 'production run|real production' docs CHANGELOG.md README.md` — expect only accurate statements.

- [ ] **Step 6: Commit**

```bash
git add -A daydream tests docs README.md CHANGELOG.md
git commit -m "refactor(training): neutral schema/fixture/test names, purge stale production-run claims"
```

---

## Task 8: Versioned data-identifier renames, #705 fold-in, verifiers pin

**Files:**
- Modify: `daydream/improve/prioritize.py:260,269` (`member-v1` → `member`), `tests/test_improve_publish.py`
- Modify: `daydream/training/calibration.py:45` (`calibration-artifact-v1` → `calibration-artifact`) + its fixture consumers
- Modify: `daydream/backends/__init__.py:189` (`AUDIT_ROOT_ISOLATION_V1` → `AUDIT_ROOT_ISOLATION`, value `"claude-pretooluse-v1"` → `"claude-pretooluse"`) + `runner.py:57,1033`, `prompt_budget.py:15,221`, `claude.py:42,949`, `tests/test_backend_codex.py:1838` contract string
- Modify: `rl/daydream_review/daydream_review/harness.py:113` (`shlex.quote`)
- Modify: `rl/daydream_review/pyproject.toml` + `uv.lock` (verifiers pin)
- Modify: `daydream/backends/codex.py:63`, `claude.py` event strings — external-contract annotation only

- [ ] **Step 1: Write the failing tests**

Three files:

`tests/test_neutral_names.py`:
```python
def test_audit_root_isolation_constant_neutral():
    from daydream.backends import AUDIT_ROOT_ISOLATION
    assert AUDIT_ROOT_ISOLATION == "claude-pretooluse"
    with pytest.raises(ImportError):
        from daydream.backends import AUDIT_ROOT_ISOLATION_V1  # noqa: F401

def test_calibration_schema_version_neutral():
    from daydream.training.calibration import ARTIFACT_SCHEMA_VERSION
    assert ARTIFACT_SCHEMA_VERSION == "calibration-artifact"

def test_member_marker_prefix_neutral():
    from daydream.improve.prioritize import member_marker
    assert member_marker("x").startswith("member:")
```

`rl/daydream_review/tests/test_harness.py` (extend the existing preflight test class — see existing assertions at tests/test_harness.py:351-384):
```python
def test_preflight_quotes_repo_path():
    """#705 fold-in: repo_path is one shell argument, not an interpolated glob."""
    harness = _make_harness(repo_path=Path("/data/repo with spaces & $dollar"))
    # assert the command list contains exactly: f"{checks} && test -d {shlex.quote(path)}"
    assert "test -d '/data/repo with spaces & $dollar'" in result_command_string
```

`tests/test_vendored_verifiers_suite.py` already validates 0.2.1 shadowing — the pin change is verified by it plus `uv lock --check`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_neutral_names.py::test_audit_root_isolation_constant_neutral tests/test_neutral_names.py::test_calibration_schema_version_neutral tests/test_neutral_names.py::test_member_marker_prefix_neutral`
Expected: FAIL — constants still carry `-v1`.

Run: `cd /tmp/daydream-1093-impl/rl/daydream_review && uv run pytest -q tests/test_harness.py::test_preflight_quotes_repo_path`
Expected: FAIL — command string contains the unquoted `test -d /data/repo with spaces & $dollar`.

- [ ] **Step 3: Implement against the test**

**Files touched:** listed above

**Behavior contract:**
- `AUDIT_ROOT_ISOLATION_V1 = "claude-pretooluse-v1"` → `AUDIT_ROOT_ISOLATION = "claude-pretooluse"`; all four importer sites + the `__all__` entry (backends/__init__.py:1404) update; `tests/test_backend_codex.py:1838` contract string updates to `"codex-cli-0.153.4-json-code-mode"`. Error semantics unchanged (capability comparison is equality; runner.py:1033 mismatch raises exactly as before).
- `ARTIFACT_SCHEMA_VERSION = "calibration-artifact"`; emitted calibration artifacts and their fixtures update in the same change (greenfield, fixture-only consumers).
- `member_marker` prefix `member-v1:` → `member:` (improve/prioritize.py:269); `tests/test_improve_publish.py` fixture strings update.
- `harness.py:113`: `f"{checks} && test -d {shlex.quote(self.config.repo_path)}"` — `import shlex` added; the writability preflight at lines 187/208 already quotes (match its style).
- External-contract annotations (Should-Have): one-line comment at `verifiers.v1` env class, ATIF v1.7 site, `/inference/v1/generate` (rl.toml:84), honeyhive `v2`, otel v1 proto — e.g. `# external contract: vendor API path, not a project-owned name`.
- verifiers pin: `rl/daydream_review/pyproject.toml` `dependencies` `verifiers==0.2.1` with the comment block reconciled to match README's resolution; run `uv lock` in the env dir to regenerate uv.lock; `tests/test_vendored_verifiers_suite.py` green confirms 0.2.1 imports resolve.

**Reference:** `rl/daydream_review/daydream_review/harness.py:187-208` — the writability preflight's existing `shlex.quote` usage is the exact style to mirror at line 113.

- [ ] **Step 4: Run the new test AND the relevant suite, verify both green**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_neutral_names.py tests/test_improve_publish.py tests/test_backend_codex.py`
Expected: PASS.

Run: `cd /tmp/daydream-1093-impl/rl/daydream_review && uv run pytest -q tests/test_harness.py tests/test_vendored_verifiers_suite.py`
Expected: PASS — the preflight quoting test green, vendored-suite shadowing confirms 0.2.1.

- [ ] **Step 5: Sweep modified files for leftovers**

Grep `rg 'member-v1|calibration-artifact-v1|AUDIT_ROOT_ISOLATION_V1|claude-pretooluse-v1' daydream tests rl` — expect zero. Grep `rg 'verifiers==0\.3\.1' rl` — expect zero.

- [ ] **Step 6: Commit**

```bash
git add daydream tests rl
git commit -m "refactor: neutralize versioned data identifiers, quote harness preflight path (#705), resolve verifiers pin to 0.2.1"
```

---

## Task 9: Committed projection fixture + CI `training-dry.yml` repoint

**Files:**
- Modify: `tests/fixtures/training/build_projection_50.py` (emit a committed projection dir)
- Create: `tests/fixtures/training/projection-50/` (committed: `_SUCCESS`, `SHA256SUMS`, `curation-manifest.json`, `corpus.jsonl`, `lineage.json`, split manifests)
- Modify: `.github/workflows/training-dry.yml:54`
- Test: `tests/test_training_dry_fixture.py` (new)

- [ ] **Step 1: Write the failing test**

New file `tests/test_training_dry_fixture.py`:
```python
def test_committed_projection_fixture_passes_loader_gates(tmp_path):
    """The committed fixture must load via the canonical projection loader."""
    from daydream.training.stacks import load_v2_projection
    projection = load_v2_projection(Path("tests/fixtures/training/projection-50"), allow_copyleft=frozenset())
    assert len(projection.records) > 0
    assert projection.digest  # directory-level digest present

def test_pipeline_dry_run_over_committed_fixture(tmp_path):
    from daydream.training.coordinator import PipelineConfig, run_pipeline
    manifest = run_pipeline(
        PipelineConfig(projection=Path("tests/fixtures/training/projection-50"), out_dir=tmp_path),
        dry_run=True,
    )
    assert manifest["stages"]["stage0"]["status"] == "complete"
    assert manifest["stages"]["stage0"]["gate"]["passed"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_training_dry_fixture.py`
Expected: FAIL — `tests/fixtures/training/projection-50` does not exist (`FileNotFoundError` from the loader).

- [ ] **Step 3: Implement against the test**

**Files touched:** `build_projection_50.py`, the new fixture dir, `training-dry.yml`

**Behavior contract:**
- `build_projection_50.py` extended: after building the 50-record projection into a target dir, it writes the full projection-dir shape (`_SUCCESS`, `SHA256SUMS`, `curation-manifest.json`, `corpus.jsonl`, split manifests `train/validation/holdout.jsonl`, `lineage.json`) so `load_v2_projection` passes `_SUCCESS`/lineage/split-digest verification fail-closed. Commit its output: `uv run python tests/fixtures/training/build_projection_50.py --out tests/fixtures/training/projection-50`.
- `training-dry.yml` "Training coordinator dry run" step: `PipelineConfig(corpus=fixture, ...)` → `PipelineConfig(projection=Path("tests/fixtures/training/projection-50"), ...)`; the fixture path variable and the records-JSONL drive lines are removed; the manifest assertions stay identical.
- Error propagation: the loader's existing fail-closed behavior (missing `_SUCCESS`/digest mismatch → its own exception) is the contract — no new error handling added.

**Reference:** `tests/fixtures/training/build_corpus_v2_50.py` (Task 7 renamed) — extend the same generation flow with the directory-materialization step the projector already performs (`corpus_v2/projector.py:1189` `_atomic_write` calls are the shape to reproduce).

- [ ] **Step 4: Run the new test AND the relevant suite, verify both green**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_training_dry_fixture.py`
Expected: PASS.

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_training_coordinator.py tests/test_stacks_load.py`
Expected: PASS — the dry-run path over the committed fixture is exactly the CI path.

- [ ] **Step 5: Sweep modified files for leftovers**

Grep `rg 'records-50|PipelineConfig\(corpus' .github tests daydream` — expect zero. The `training-dry.yml` header comment ("over the committed 50-record fixture") updated to name the projection fixture.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/training .github/workflows/training-dry.yml tests/test_training_dry_fixture.py
git commit -m "ci(training): drive training-dry over committed projection fixture"
```

---

## Task 10: Naming grep-gate + `make check` integration + full verification

**Files:**
- Create: `scripts/check-naming.sh`
- Modify: `Makefile`, `.github/workflows/ci.yml`

- [ ] **Step 1: Write the failing test**

The gate itself is the test — add a repo-root pytest wrapper so a red gate fails the suite:

```python
# tests/test_naming_gate.py
def test_no_project_owned_versioned_names():
    """#1093 Should-Have: the naming decision cannot regress."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/check-naming.sh"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"versioned project names remain:\n{result.stdout}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_naming_gate.py`
Expected: FAIL if any earlier task left a versioned name (the gate script exists but returns nonzero listing occurrences) — this is the audit's teeth; with Tasks 2-9 done it should pass, and a failure here names the exact leftover site.

- [ ] **Step 3: Implement against the test**

**Files touched:** `scripts/check-naming.sh` (new), `Makefile`, `.github/workflows/ci.yml`

**Behavior contract:**
- `scripts/check-naming.sh`: greps the tree (`git ls-files`, excluding `.beagle/`, external-contract allowlist paths) for `corpus_v2|stacks_v2|rubric_v2|daydream_review_v1|build-v2|run_build_corpus_v2|curation-manifest-v1|member-v1|calibration-artifact-v1|AUDIT_ROOT_ISOLATION_V1|claude-pretooluse-v1` and fails nonzero with the offending `file:line` list; the allowlist is an inline array of external-contract patterns (`verifiers\.v1|ATIF|/inference/v1/|honeyhive\.ai/v2|opentelemetry.*v1`) with a one-line comment each naming the external contract.
- `Makefile`: `check: check-naming` (or a new target wired into `check`) so `make check` enforces it locally; `.github/workflows/ci.yml` gains the same step.

**Reference:** Makefile target style at `Makefile:66-67` (`rl-check` recipe shape).

- [ ] **Step 4: Run the new test AND the relevant suite, verify both green**

Run: `cd /tmp/daydream-1093-impl && uv run pytest -q tests/test_naming_gate.py`
Expected: PASS.

Run: `cd /tmp/daydream-1093-impl && make check && make rl-check`
Expected: PASS — spec Must-Have terminal condition. (If `make check` includes the root suite, expect the full suite green with zero regressions.)

Run: `cd /tmp/daydream-1093-impl && bash scripts/check-naming.sh && echo "grep -rn 'load_dataset\\([^_]' daydream tests | wc -l → 0" && uv run pytest -n auto -q`
Expected: full green.

- [ ] **Step 5: Sweep modified files for leftovers**

Grep the allowlist itself: every entry must carry its external-contract comment; no allowlisted pattern may accidentally cover a project-owned name (narrow patterns, not `.*v1.*`).

- [ ] **Step 6: Commit**

```bash
git add scripts/check-naming.sh Makefile .github/workflows/ci.yml tests/test_naming_gate.py
git commit -m "ci(naming): gate zero project-owned versioned names in make check and CI"
```

---

## Self-Review Outcome

- **Spec coverage:** every spec Must-Have maps to a task — legacy `--corpus` branch (Task 2), `stacks.load_dataset` (Tasks 2,5), vendored harvested-index loader + callers (Task 6), legacy fixtures/tests/docs (Tasks 2,3,7), full rename audit incl. `rl/daydream_review_v1`/`corpus_v2`/`stacks_v2`/`rubric_v2`/schema/CLI/task-identity (Tasks 4-7), versioned-identifier classification (Task 8), canonical-path preservation (Tasks 2,4,5,9 explicitly assert the loader/gates stay green), RL env preserved+renamed+decoupled (Task 6), #705 fold-in (Task 8), verifiers pin (Task 8), training-dry repoint (Task 9), stale-claim purge (Task 7), atomic renames/no aliases (Pattern + per-task sweeps), `make check`+`make rl-check` (Task 10). Should-Haves: grep-gate (Task 10), external-contract annotations (Task 8). Out-of-scope respected: no #1098 bridge, no behavior change to projection logic, no deprecation shims.
- **Placeholders:** none — every test step has exact assertions; every command is copy-pasteable; every impl step is a behavior contract with a reference pointer.
- **Type consistency:** verified — `build_frozen_corpus`/`BuildFrozenCorpusConfig` (Task 4) match Task 1's test; `projection` field (Task 2) matches Task 9's `PipelineConfig(projection=...)`; `load_v2_projection` import path `daydream.training.stacks` (Task 5) matches Task 9's test; `AUDIT_ROOT_ISOLATION` (Task 8) matches its test.
- **Consumer check:** every renamed surface's production consumers named in-task (coordinator, cli, adjudication/*, archive/*, reward_model, rft, rl.toml, Makefile, ci.yml); no new public surface introduced beyond `scripts/check-naming.sh` (consumers: `make check` + ci.yml + the gate test — production CI consumers, not test-only).
- **Discriminating assertion:** the naming gate pins *absence* (grep-zero), not dispatch; Task 6's decoupling test pins `ModuleNotFoundError` on the deleted vendored loader; Task 2's TypeError test would fail on a no-op impl that kept `corpus` (constructs fine → no raise); Task 9's fixture test pins the loader gates (fail-closed), not just file existence.
- **Spike candidates:** all three spec spikes executed against main @ `a10e0af` during planning (findings in the Spike Findings section); Task 6 carries one residual STOP condition (manifest-only image builds) surfaced as an explicit Assumption.
- **Failure-propagation:** every deleted branch removes fallible paths rather than adding them; remaining fallible ops (projection loader, gates) keep their existing exception types — contract stated per task; no `unwrap_or` fallback patterns introduced anywhere.
- **Project conventions:** commit messages follow the repo's `type(scope): summary` convention (verified against `git log --oneline`); tests live in `tests/` (root) and `rl/daydream_review/tests/` (env-local); `uv`-managed commands throughout; `make check`/`make rl-check` as terminal gates per repo convention.
- **Parallel-implementation gate:** not applicable — this plan removes a parallel legacy implementation rather than adding one; the naming gate (Task 10) is the anti-regression equivalent.