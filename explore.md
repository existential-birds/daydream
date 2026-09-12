# Explore — issue #1093: training: remove unused loaders and versioned project names

Recon against **main @ `a527c97`** (fresh shallow clone, includes #1192 / issue #721 merge —
the parent-card rebase condition is already satisfied; no #721 rename overlap found in the
audited surfaces). Issue body confirmed: the **September 2, 2026 scope correction is present
and binding** (no daydream-improve plan embedded; issue authored by anderskev, label
`area:training`, 0 comments).

## A. Legacy corpus-loading surfaces to REMOVE (verified in code)

1. **`daydream train --corpus` branch**
   - `daydream/cli.py:2282-2289` — `--corpus` arg in the mutually-exclusive `corpus_group`
     of `_TrainParser` (sibling `--corpus-v2` is the canonical input).
   - `daydream/training/coordinator.py:690-713` — the `else:` branch loading v1 records via
     `stacks.load_dataset(corpus_path, ...)`; `PipelineConfig.corpus` field + `__post_init__`
     exactly-one-of constraint (~line 64-90). Canonical keep: `load_v2_projection` path.
   - CI impact: `.github/workflows/training-dry.yml:54` drives `PipelineConfig(corpus=fixture,...)`
     with the records-50 JSONL — must be repointed to the canonical `corpus_v2` input
     (a committed projection fixture must be generated or `records-50` replaced by a
     projection dir; no committed projection fixture exists today under `tests/fixtures/training/`).
2. **`daydream/training/stacks.load_dataset`** (v1 loader, `stacks.py:43-120`, exported in
   `__all__` line 40). The whole module is a docstring-framed v1/v2 sibling pair; `load_dataset_v2`
   (line 121) is the C5/C8-enforcing loader that `stacks_v2.py` wraps. Plan: fold the v2 loader
   into the canonical surface, delete the legacy `load_dataset` + `legacy_policy` stamping.
   Callers: `coordinator.py:712` (legacy branch), `tests/test_training_stacks.py` (whole file
   is v1-loader tests), `tests/test_training_coordinator.py:84-109`, `tests/test_corpus_v2.py:740-776`
   (imports both; v2 uses survive).
3. **Vendored harvested-index.json loader** (RL env): `rl/daydream_review_v1/daydream_review_v1/corpus.py`
   — self-contained `EvaluablePR` / `CorpusSource` / `harvested_corpus()` over a harvest dir's
   `index.json`. Sole callers: `taskset.py:994-997` (the `config.corpus_dir` source branch) and
   `images/build_images.py` (`DEFAULT_CORPUS = tests/fixtures/corpus-mini`).
   Related legacy-only fixtures: `tests/fixtures/corpus-mini/index.json`,
   `tests/fixtures/corpus-reference/index.json` (+ their `results/benchmark_data.json`).
   **Scope note:** issue says remove "loader + its taskset/image-builder **callers**" — i.e. the
   legacy corpus-source branch inside taskset/build_images, not the Taskset/Harness/Runtime
   themselves (see Preserve below). `taskset.py:997` already notes the harvest shape is
   superseded; #1098 will bridge the env to canonical reconstructable task records — the
   decoupling seam is exactly the `CorpusSource` construction in taskset + the corpus-dir
   plumbing in `configs/eval-{stub,docker}.toml`.
4. **Docs/launch instructions legacy-only**: `docs/training-launch.md` lines ~30-60 describe the
   `stacks.load_dataset` C5/C8 path and `run_build_corpus` exports over the records fixture;
   `README.md` ~line 315 still points at the fixture-based pipeline; `CHANGELOG.md:366` claims
   trajectories feed the corpus pipeline "from real production runs" (stale claim, criterion 7).

## B. v1/v2 RENAME audit (project-owned names; confirmed present)

| Name | Where | Notes |
|---|---|---|
| `rl/daydream_review_v1/` (dir + inner package `daydream_review_v1/`) | rl env (#164) | **PRESERVE + RENAME.** Package name, taskset/harness ids `daydream-review-v1` (`rl/train/rl.toml:81,90,109`, taskset.py ids), `Makefile` rl-check paths, `.github/workflows/ci.yml` rl-check job, `README.md:313`, `rl/train/README.md:5,38,114`, `daydream/pyproject.toml:137` comment. Taskset/harness `id` strings are task identity — rename atomically with the dir. |
| `daydream/training/corpus_v2/` (9 modules) | canonical projection | Rename to e.g. `daydream/training/corpus/`… but note `daydream/training/corpus.py` ALREADY EXISTS (harvest/projection pipeline, 1236 lines) — name collision the implementer must resolve (package `corpus/` vs module `corpus.py` is legal Python but confusing; consider `projection/` or fold). Consumers: cli.py, coordinator.py, adjudication/* (7 files), archive/* (3), reward_model.py, rft.py, 15+ test files. |
| `daydream/training/stacks_v2.py` | split loader wrapper | Rename + fold v2 loader from `stacks.py` into it or into one neutral module. |
| `daydream/training/rubric_v2.py` | reward rubric | 1 import site each in tests; `rft.py:206` docstring ref. |
| Schema/fixture filenames | `daydream/training/schema/v1.json`, `v2.json`, `curation-manifest-v1.json`; `tests/fixtures/training/curation-manifest-v1-fixture.json`; `tests/fixtures/training/build_corpus_v2_50.py` | v1.json/v2.json are two generations of the record schema — v1 is legacy-only (delete with v1 branch); users in schema.py, corpus.py, coordinator.py, cli.py, corpus_v2/*. |
| CLI/API names | `daydream corpus build-v2` (`cli.py:572,578,689,2384,2393,2398`), `run_build_corpus_v2()` (corpus_v2/projector.py:826, exported) | user-facing verb — rename to neutral (e.g. `corpus build` vs legacy `corpus build`… note a LEGACY `corpus build` already exists at README:231; decide: the legacy projection verb is also v1-era — resolve both). |
| Task identity | taskset/harness `id = "daydream-review-v1"` in rl.toml ×3 + README refs | part of the rl rename above. |
| Tests | `tests/test_corpus_v2*.py` ×7, `test_stacks_v2_{gate,load}.py`, `test_training_contract_v1_v2.py`, `test_training_coordinator_v2.py`, `test_training_rft_v2_sha.py`, `test_training_rubric_v2.py` | rename with their subjects. |
| Other versioned project-owned identifiers found | `member-v1` (improve/prioritize.py:260,269 — published plan-record `kind`; internal but content-addressed), `derive_curation_id_v2` (archive/hydrate_rules.py:74 + hydrate.py, adjudication/publish.py), `calibration-artifact-v1` (calibration.py:45 — emitted artifact schema id), `claude-pretooluse-v1`, `codex-cli-...-v1`, `auto-v1` (event/artifact kind strings in observability/backends) | Implementer should classify each: versioned-identifier-in-data may be external-contract-like once written; greenfield means data written only by fixtures, so renames are safe. |

### NOT versioned-name violations (leave alone)
`verifiers.v1` env class, ATIF v1.7, `/inference/v1/generate` (vendor API path, rl.toml:84),
`docs.honeyhive.ai/v2`, opentelemetry `v1` proto, `AUDIT_ROOT_ISOLATION_V1` (internal constant,
candidate for neutral rename but is a protocol string in pretooluse guards — implementer to
classify), `vulture`, `qwen3`, `prime-rl v0.7.0`.

## C. PRESERVE (do not delete)

- `corpus_v2/` projection logic: split + lineage validation, C5/C8 fail-closed, full-SHA task
  identity (`identity.py`), bundles/license/provenance/segments/tiers — all intact at a527c97.
- `rl/daydream_review_v1/` Stage-3 environment: taskset.py, harness.py, verifier.py, backends.py,
  rundir.py, gate_refusal.py, stub_upstream.py, fixture.py, images/{base,repo}.Dockerfile +
  build_images.py + manifest.toml, 13 test files. Only decouple from the legacy loader.

## D. #705 fold-in — concrete finding

`rl/daydream_review_v1/daydream_review_v1/harness.py:113`:
```python
["sh", "-c", f"{checks} && test -d {self.config.repo_path}"]
```
`repo_path` is interpolated **UNQUOTED** into `sh -c` (the writability preflight at lines 187/208
DOES use `shlex.quote`). `tests/test_harness.py:351-384` asserts the unquoted string form, so the
regression coverage for whitespace/shell-significant paths is currently absent at this seam.
The standalone harness preflight REMAINS under a neutral name → per issue criterion 5/#705,
carry forward: `shlex.quote(self.config.repo_path)` at line 113 + whitespace and
shell-significant path regression tests.

## E. verifiers pin/doc skew

`rl/daydream_review_v1/pyproject.toml:9` pins **`verifiers==0.3.1`** while
`rl/daydream_review_v1/README.md:22,221-228` says the resolution is to keep **`verifiers==0.2.1`**
pinned (prime-rl v0.7.0 vendors 0.2.1; `test_vendored_verifiers_suite.py` shadows per PYTHONPATH).
The pyproject comment block (lines 6,23-65) references the 0.2.1 rationale — the pin and the
docs/comment disagree. Either resolve to one version while touching package metadata, or
explicitly hand to #1098 and document.

## F. Stale production-run claims (criterion 7)

- `docs/training-launch.md` — "Every number below comes from an artifact produced by a validation
  run… nothing is a placeholder"; measured wall-times/costs over the 50-record fixture are
  validation-scale but the framing reads launch-record. Line ~30-60 also describes the legacy
  loader path (dies with section A1).
- `README.md:315` — correctly says "not yet harvested production data" (accurate; keep/update).
- `CHANGELOG.md:366` — "feeding the SFT/RL corpus pipeline from real production runs" — stale.
- `plan-notes.md` (repo root, spike workspace notes) — legacy-workflow-only doc; candidate for
  removal under criterion 1 ("documentation that exists only for that workflow").

## G. Risks / decisions for the planner

1. **Name collision**: `daydream/training/corpus.py` exists; renaming `corpus_v2/` → `corpus/`
   needs a deliberate target (`corpus_projection/`, fold into `corpus.py`, or rename legacy away
   too). Decide before implement.
2. **CI training-dry**: must switch to a committed projection-dir fixture (generate one via
   `build_corpus_v2_50.py`-style script) or repoint the workflow; the fixture must pass
   `_SUCCESS`/lineage/split-digest verification.
3. **rl rename blast radius**: Makefile, ci.yml, rl/train/README.md, rl.toml ids ×3, package dir +
   inner package + uv.lock, README:313, pyproject comment, plus `uv run` lockfile regen. Serialized
   after #721 (already merged as a527c97 — done).
4. **`member-v1` / `calibration-artifact-v1` / event-kind versioned strings**: classify
   external-contract vs project-owned; greenfield ⇒ rename-safe, but data fixtures pin them.
5. **Taskset id strings are task identity** — renaming changes task identity; greenfield permits
   it (no run history), but it must be atomic across rl.toml + taskset.py defaults + docs.
6. Legacy `daydream corpus build` (v1 projection verb, README:229-233) likely dies with the v1
   branch — its tests (`test_training_corpus.py`, coordinator v1 tests) are legacy-only.

## Verification commands for implementer

```
grep -rn --include='*' -E 'corpus_v2|stacks_v2|rubric_v2|daydream_review_v1|build-v2|run_build_corpus_v2' .   # expect 0 (excl. .git, external contracts)
grep -rn 'load_dataset\([^_]' daydream tests  # legacy loader gone
make check && make rl-check
uv run python -c "from daydream.training.coordinator import PipelineConfig"  # no --corpus arg
```
