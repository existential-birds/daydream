# Phase 1: Vendor ATIF Foundation - Context

**Gathered:** 2026-04-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Vendor Harbor's `harbor.models.trajectories.*` Pydantic models, `harbor.utils.trajectory_validator`, and a representative pair of golden trajectory fixtures into a self-contained `daydream/atif/` package — Apache-2.0 attributed, importing only from stdlib + `pydantic` — so Phase 2's recorder has a typed, validated substrate without `harbor` becoming a runtime dependency. Phase 1 is purely additive: no production module imports from `daydream/atif/` yet, and the existing 343 tests are unaffected.

</domain>

<decisions>
## Implementation Decisions

### Source Provenance & Drift Policy
- **D-01:** Vendor from the **latest tagged Harbor release** at the time the vendoring commit lands. Tagged releases are stable, semver-meaningful, and easier to point consumers at than a main-branch SHA.
- **D-02:** Provenance recorded **inline in `daydream/atif/NOTICE`** — single source of truth alongside the Apache-2.0 attribution. Required line shape: `Vendored from harbor-framework/harbor@<tag>, commit <sha>, on <YYYY-MM-DD>`. No separate `VENDOR.md`; no per-file comment headers (original Harbor copyright headers stay intact, but no daydream-specific provenance is added per file).
- **D-03:** **Mechanical-only edit policy** for vendored Harbor source. The ONLY allowed transformation is import-path renames:
  - `harbor.models.trajectories` → `daydream.atif.models`
  - `harbor.utils.trajectory_validator` → `daydream.atif.validator`
  - Internal `from harbor.models.trajectories import …` references inside vendored files get the same rename treatment.
  - **NOT allowed in Phase 1:** reformatting to daydream's ruff/mypy style, dropping unused fields/methods, simplifying docstrings, removing comments. Original copyright headers stay. This keeps re-vendor diffs near-clean against upstream.
- **D-04:** **Re-vendor wholesale on demand** for future Harbor updates. When ATIF v1.7 lands or a Harbor bug fix needs picking up: blow away `daydream/atif/models/` + `daydream/atif/validator.py`, re-copy from new tag, redo the import-rename, update NOTICE. No local patches carried; no VENDOR.md drift log to maintain. Whoever needs the upgrade does the re-vendor.

### Models Layout & Public API Surface
- **D-05:** Inside `daydream/atif/models/`, **mirror Harbor's file split exactly**. Whatever files Harbor ships under `harbor/models/trajectories/` (e.g., separate files for `trajectory.py`, `step.py`, etc.) are copied 1:1. Keeps the diff against upstream small; supports the wholesale re-vendor strategy from D-04.
- **D-06:** **`daydream/atif/__init__.py` re-exports models + a `validate()` function**. Concretely: `from daydream.atif import Trajectory, Agent, Step, ToolCall, Observation, ObservationResult, Metrics, FinalMetrics, validate`. Declared explicitly via `__all__`. One-stop shop for Phase 2's recorder construction code; programmatic validator surfaced as a top-level `validate` callable, not as a class users have to instantiate.
- **D-07:** **Strip Harbor's `python -m harbor.utils.trajectory_validator` CLI entry point.** Drop the `if __name__ == "__main__":` block and any argparse/CLI scaffolding. Daydream uses the validator programmatically only (per REQUIREMENTS TEST-04 round-trip pattern). Rationale: smaller surface, less code to vendor, matches the milestone's hard-cutover discipline. Note: this is a deletion, not an edit — consistent with mechanical-only policy because the `__main__` block is a self-contained bottom-of-file segment, not a structural change to the validator class.
- **D-08:** **`validate()` returns Harbor's existing surface** — wraps a freshly-constructed `TrajectoryValidator`, returns its `.validate(input) -> bool` result. `get_errors() -> list[str]` accessible via `from daydream.atif.validator import TrajectoryValidator` for tests that need detailed error inspection. No `ValidationResult` dataclass; no `raise on invalid` exception path. Pure passthrough to keep the mechanical-only policy intact.

### Schema-Version Acceptance Range
- **D-09:** **Validator accepts ATIF-v1.0 through ATIF-v1.6** — Harbor's default `Literal["ATIF-v1.0", "ATIF-v1.1", "ATIF-v1.2", "ATIF-v1.3", "ATIF-v1.4", "ATIF-v1.5", "ATIF-v1.6"]` is left as-is. **Validation breadth and emission breadth are independent**: daydream still EMITS v1.6 only (per PROJECT.md), but the validator must accept the OpenHands v1.5 golden fixture for VEND-05/TEST-04 to pass. Narrowing the Literal would also be a non-mechanical edit, which D-03 forbids.
- **D-10:** **Future-version trajectories (v1.7+) reject as unknown** — the natural Pydantic `Literal` mismatch produces a clear validation failure. No best-effort accept, no warn-and-continue. The documented forward-compat path is "re-vendor when Harbor updates" (D-04), not "tolerate unknown versions in place."

### Golden Fixture Selection & Breadth
- **D-11:** **Vendor one Terminus-2 (v1.6) + one OpenHands (v1.5) fixture** as the representative pair for round-trip testing. Smallest disk + test-runtime footprint that still proves the validator works end-to-end across both schema versions in our accepted range and both reference agents. Matches PROJECT.md / REQUIREMENTS wording exactly.
- **D-12:** **Fixture path layout: `tests/fixtures/atif_golden/<source>/<file>.json`** with source-namespaced subdirs:
  - `tests/fixtures/atif_golden/terminus2/<filename>.json`
  - `tests/fixtures/atif_golden/openhands/<filename>.json`
  - Self-documenting; future `TEST-04` parametrized test discovers via `Glob("tests/fixtures/atif_golden/**/*.json")` while excluding `_invalid/`.
- **D-13:** **Vendor (or hand-author) one deliberately-broken fixture under `tests/fixtures/atif_golden/_invalid/`** so Phase 5's negative-path test (deliberate-break catch from PITFALLS.md "Looks Done But Isn't" checklist) has a concrete file to point at. Concrete suggestion: `_invalid/non-sequential-step-id.json` — a trajectory where `steps[1].step_id == 3` instead of `2`. Tiny effort; closes the negative-test gap up front.

### Claude's Discretion
- Exact filename for the broken negative-path fixture (D-13) — `non-sequential-step-id.json`, `bad-step-id.json`, etc. Pick whichever reads cleanest in the parametrized test output.
- Whether to drop a one-line `daydream/atif/__init__.py` docstring summarizing the package, or leave the file with `__all__` only. Either is acceptable.
- Order of imports inside `daydream/atif/__init__.py` `__all__` declaration. Group by category if helpful (models first, validate last).
- Whether to add a `daydream/atif/__init__.py` `__version__` constant pinned to the upstream Harbor tag (e.g., `__version__ = "atif-v1.6 / harbor-0.5.0"`). Not required by VEND-* but cheap and useful for diagnostics.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### ATIF Specification
- `docs/reference/atif_format.md` — Authoritative ATIF spec describing the format, Pydantic model usage, validator behavior, and the OpenHands accumulated-to-delta conversion example. Schema version table (lines 396–402) lists v1.0–v1.4; the actual current Harbor release ships through v1.6 and our validator must accept that.
- ATIF RFC: `https://github.com/laude-institute/harbor/blob/main/docs/rfcs/0001-trajectory-format.md` — Source RFC referenced by the spec.

### Harbor Source Pin Targets
- `https://github.com/harbor-framework/harbor` — repo to clone and tag-pin from. The vendoring commit's NOTICE entry must record the exact tag + commit SHA + date.
- `https://github.com/harbor-framework/harbor/tree/main/src/harbor/models/trajectories` — source dir for the Pydantic models (mirror file split per D-05).
- `https://github.com/harbor-framework/harbor/blob/main/src/harbor/utils/trajectory_validator.py` — source for the validator (strip `__main__` per D-07).
- `https://github.com/harbor-framework/harbor/tree/main/tests/golden` — source dir for golden fixtures (pick one Terminus-2 + one OpenHands per D-11).

### Project Planning
- `.planning/PROJECT.md` — Active requirements, Out of Scope decisions, and the Key Decisions table (especially: vendoring rationale, Pydantic >=2.11.7 floor, ATIF v1.6 emission pin, Harbor-not-runtime-dep due to litellm supply-chain quarantine).
- `.planning/REQUIREMENTS.md` — VEND-01..05 are this phase's requirements verbatim. Traceability section confirms VEND-* are scoped to Phase 1 only.
- `.planning/ROADMAP.md` — Phase 1 success criteria; reminder that the existing 343-test suite must still pass post-vendoring (vendoring is purely additive code).
- `.planning/research/PITFALLS.md` — Pitfall 14 (`phases.py` / `ui.py` bloat avoidance) and Pitfall 11 (test brittleness — schema-validity + behavior-predicate pattern, not full-tree snapshot equality) are referenced by later phases but worth Phase 1 awareness so test patterns set here propagate cleanly.
- `.planning/codebase/STRUCTURE.md` — `Where to Put New Code` rules; new top-level package under `daydream/` is precedented (`daydream/prompts/`, `daydream/deep/`).
- `.planning/codebase/CONVENTIONS.md` — naming + import organization. Vendored Harbor code is exempt from these (mechanical-only edit policy from D-03), but our `daydream/atif/__init__.py` shim and any test code we write IS subject to them.

### Test Patterns
- `tests/conftest.py` — for the `_reset_trajectory_recorder` fixture pattern (Phase 2 concern, but Phase 1 must not introduce conftest changes that conflict).
- `tests/fixtures/codex_jsonl/` and `tests/fixtures/diffs/` — existing precedent for vendored-fixture path layout under `tests/fixtures/<category>/`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`daydream/prompts/` and `daydream/deep/` precedent**: top-level packages under `daydream/` with their own `__init__.py` and dedicated submodules. `daydream/atif/` follows the same shape — no architectural novelty needed.
- **`tests/fixtures/codex_jsonl/`**: existing fixture-namespacing pattern under `tests/fixtures/<category>/`. New `tests/fixtures/atif_golden/<source>/` slots in cleanly.
- **`pyproject.toml [project.dependencies]`**: existing explicit-deps list (`anyio`, `claude-agent-sdk==0.1.52`, `rich`, `pyfiglet`, tree-sitter family). `pydantic>=2.11.7` lands here per VEND-03; `claude-agent-sdk==0.1.52` already pins pydantic transitively, so resolution will be a no-op floor add.

### Established Patterns
- **No `__all__` except `daydream/backends/__init__.py`** (per CONVENTIONS.md). Phase 1 introduces a second exception in `daydream/atif/__init__.py` because we want a stable, discoverable public surface for downstream phases. This is a deliberate convention-extension, not a violation — call it out in the planning phase.
- **Mechanical-only vendoring is unprecedented in this repo** — no other module under `daydream/` carries third-party source. The `daydream/atif/NOTICE` + `daydream/atif/LICENSE` files are new pattern; PR description should document why these exist.
- **`from __future__ import annotations`** is used elsewhere; vendored Harbor code may or may not use it. Mechanical-only policy means we don't add it where it's missing.

### Integration Points
- **No production daydream module imports from `daydream/atif/` in Phase 1.** Recorder integration is Phase 2's concern (CORE-01). VEND-04 explicitly verifies "no `from harbor import …` references anywhere in `daydream/` or `tests/`" — but does NOT require any production code to import from `daydream.atif` yet.
- **`tests/test_atif_models.py` is Phase 5's deliverable** (TEST-04). Phase 1's testing burden is limited to: (a) confirm the vendored validator runs against vendored goldens via a smoke check (can be a one-off script or a tiny `tests/test_atif_vendor_smoke.py` that's later expanded by TEST-04), (b) confirm `pyproject.toml` resolves cleanly with the explicit `pydantic>=2.11.7`, (c) confirm the existing 343-test suite still passes.
- **`Makefile` targets** (`make lint`, `make typecheck`, `make test`) need to be clean against the new vendored code. mypy's `ignore_missing_imports = true` already handles any pydantic-related stub gaps; ruff `target-version = "py312"` may flag stylistic differences in vendored code — solution per D-03 is to add a `[tool.ruff.lint.per-file-ignores]` exemption for `daydream/atif/**` (don't lint vendored code) rather than reformatting to fit ruff.
- **`pre-push` hook** runs lint + typecheck + full test suite — must be green after the vendoring commit lands.

</code_context>

<specifics>
## Specific Ideas

- **PROJECT.md line 100** explicitly cites the Harbor `litellm>=1.80.8` supply-chain quarantine (March 2026) as the reason vendoring is required instead of taking Harbor as a runtime dep. The NOTICE file should mention this implicitly — "vendored to avoid Harbor's transitive dependency surface" — without naming the specific package, since the rationale will outlast the specific incident.
- **REQUIREMENTS VEND-04** says "no remaining `from harbor import …` references anywhere in the daydream source tree" — the verification is a `grep -r 'from harbor' daydream/ tests/ scripts/` returning zero matches. Worth wiring into a Phase 1 smoke-check or a one-shot script run in the planning phase.
- **PITFALLS.md lines 504–505** warn against assuming `python -m harbor.utils.trajectory_validator` is on PATH — since we strip the CLI entry point per D-07, this concern goes away by construction. Note in CHANGELOG that the daydream-vendored validator is programmatic-only.
- **Apache-2.0 conformance**: NOTICE must include upstream Harbor copyright + the NOTICE text Harbor itself ships (if any). LICENSE is a verbatim copy of Apache-2.0. Both files live at `daydream/atif/{LICENSE,NOTICE}` per DOCS-05.

</specifics>

<deferred>
## Deferred Ideas

- **`daydream/atif/__init__.py` `__version__` constant** (e.g., `__version__ = "atif-v1.6 / harbor-0.5.0"`) for diagnostics — not required by VEND-* but cheap. Picked up if Phase 1 planning has spare scope; otherwise deferred to a future maintenance pass.
- **Multimodal `ContentPart` / `ImageSource` support** — out of scope per PROJECT.md; v1.6 models are vendored as-is so the multimodal classes ride along, but daydream emits text-only and never constructs them.
- **`--no-redact` escape hatch on the Phase 4 redactor** — mentioned in PITFALLS.md Pitfall 8 mitigation; deferred to post-milestone per PROJECT.md (privacy-default-on is the explicit decision).
- **`daydream replay <trajectory.json>` and `daydream stats <trajectory.json>` subcommands** — listed under v2 requirements in REQUIREMENTS.md; never a Phase 1 concern.
- **Dual-write phase keeping `_log_debug` alongside trajectories** — explicitly Out of Scope per PROJECT.md; do not reintroduce in any phase.

</deferred>

---

*Phase: 01-vendor-atif-foundation*
*Context gathered: 2026-04-26*
