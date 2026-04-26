# Phase 1: Vendor ATIF Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-26
**Phase:** 01-vendor-atif-foundation
**Areas discussed:** Source provenance & drift policy, Models layout + public API surface, Schema-version acceptance range, Golden fixture selection & breadth

---

## Source provenance & drift policy

### Q1: Which Harbor source point should we vendor from?

| Option | Description | Selected |
|--------|-------------|----------|
| Latest tagged Harbor release | Pin to whatever the most recent tagged version of harbor-framework/harbor is when we vendor. Stable, semver-meaningful, easy to tell consumers what they're looking at. Record tag in NOTICE. | ✓ |
| Specific main-branch commit SHA | Pin to a current main-branch commit. More flexibility if we want a fix that hasn't been released yet. Slightly less obvious provenance. | |
| Latest release that publishes ATIF-v1.6 + golden fixtures | Pin to the oldest Harbor release that ships everything we need (v1.6 Pydantic models + Terminus-2 v1.6 + OpenHands v1.5 goldens). Smaller diff against what's tested upstream. | |

**User's choice:** Latest tagged Harbor release.

### Q2: Where do we record the vendored source pin?

| Option | Description | Selected |
|--------|-------------|----------|
| Inline in `daydream/atif/NOTICE` | One file. NOTICE already has to exist for Apache-2.0; add a 'Vendored from harbor-framework/harbor@<tag>, commit <sha>, on <date>' line. Single source of truth, no duplicate. | ✓ |
| Separate `daydream/atif/VENDOR.md` | NOTICE stays minimal Apache-2.0 attribution; VENDOR.md documents source pin, drift policy, update procedure, and any local patches. More room to write update steps. | |
| Top of every vendored .py file as a comment header | Explicit at every read site; no central drift if files diverge. | |

**User's choice:** Inline in `daydream/atif/NOTICE`.

### Q3: How much may we edit vendored Harbor source files?

| Option | Description | Selected |
|--------|-------------|----------|
| Mechanical-only: rename imports, nothing else | Only allowed change is `harbor.models.trajectories.*` → `daydream.atif.models.*` and `harbor.utils.trajectory_validator` → `daydream.atif.validator`. Original copyright headers, docstrings, dead-code comments, all unchanged. Future upstream updates apply as a near-clean diff. | ✓ |
| Mechanical + drop unused parts | Rename imports AND delete code paths daydream provably won't use. Smaller surface area; more code to re-prune on every update. | |
| Full edit — treat as our code from day one | Reformat to match daydream's ruff/mypy style, simplify docstrings, drop unused fields. Easiest day-to-day; merge from upstream becomes a manual rewrite. | |

**User's choice:** Mechanical-only: rename imports, nothing else.

### Q4: How do we handle future Harbor updates?

| Option | Description | Selected |
|--------|-------------|----------|
| Re-vendor wholesale on demand | When ATIF v1.7 lands or a Harbor bug needs picking up, blow away `daydream/atif/models/` + `validator.py`, re-copy from new tag, redo the import-rename, update NOTICE. Clean, predictable, low cognitive load — we never carry local patches. | ✓ |
| Patch in place + record diffs in VENDOR.md | Apply targeted fixes locally; document each patch in a VENDOR.md so the next re-vendor knows what to re-apply. More flexible but creates ongoing drift management work. | |
| Defer the question — cross that bridge later | Don't decide now. Phase 1 just vendors. Whoever hits the upgrade need first chooses the strategy then. | |

**User's choice:** Re-vendor wholesale on demand.

---

## Models layout + public API surface

### Q1: How should files be organized inside `daydream/atif/models/`?

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror Harbor's file split exactly | Whatever files Harbor ships under harbor/models/trajectories/ — we copy 1:1. Keeps the diff against upstream small. Re-vendoring is a clean copy. | ✓ |
| Flatten to a single `models/__init__.py` | All Pydantic classes consolidated into one file. Simpler imports for daydream code; breaks the 1:1 mirror of upstream. | |
| Single top-level `daydream/atif/models.py` | Drop the directory entirely — contradicts VEND-01 which specifies the directory form. | |

**User's choice:** Mirror Harbor's file split exactly.

### Q2: What does `from daydream.atif import …` expose at the package root?

| Option | Description | Selected |
|--------|-------------|----------|
| Models + validate() function | `from daydream.atif import Trajectory, Agent, Step, ToolCall, Observation, ObservationResult, Metrics, FinalMetrics, validate`. One-stop shop for Phase 2's recorder; programmatic validator surfaced as a function, not a class. `__all__` declared explicitly. | ✓ |
| Models only — validator stays at `daydream.atif.validator` | Top-level only re-exports Pydantic models. Validator must be imported separately. Cleaner namespace separation; one extra import line in tests. | |
| Nothing — callers always reach into submodules | `daydream/atif/__init__.py` stays empty. Callers always write `from daydream.atif.models import Trajectory`. Most explicit; verbose at every call site. | |

**User's choice:** Models + validate() function.

### Q3: Keep Harbor's `python -m harbor.utils.trajectory_validator` CLI entry point?

| Option | Description | Selected |
|--------|-------------|----------|
| Strip the CLI entry point | Drop the `if __name__ == "__main__"` block and any argparse code. Daydream uses the validator programmatically only. Smaller surface, less to vendor, less to test. | ✓ |
| Keep it — may be useful for ad-hoc debugging | Preserve `python -m daydream.atif.validator <file>`. Slightly more code to vendor; one more public surface. | |

**User's choice:** Strip the CLI entry point.

### Q4: How should the top-level `validate()` function report errors?

| Option | Description | Selected |
|--------|-------------|----------|
| Return Harbor's existing TrajectoryValidator surface | Keep what Harbor ships: a `TrajectoryValidator` class with `.validate(input) -> bool` and `.get_errors() -> list[str]`. Top-level `validate()` is a thin wrapper. Mechanical-only edit policy stays clean. | ✓ |
| Raise on invalid — Pythonic exception path | `validate(traj)` returns None on success, raises `TrajectoryValidationError` on failure. More idiomatic for `pytest.raises`. Requires a wrapper layer on top of Harbor's class. | |
| Return a result dataclass | `validate(traj) -> ValidationResult(ok: bool, errors: list[str])`. Most ergonomic; some boilerplate. Wrapper layer over Harbor. | |

**User's choice:** Return Harbor's existing TrajectoryValidator surface.

---

## Schema-version acceptance range

### Q1: Which schema versions should the vendored validator accept?

| Option | Description | Selected |
|--------|-------------|----------|
| v1.0–v1.6 — keep Harbor's default | Validator's `schema_version: Literal["ATIF-v1.0", ..., "ATIF-v1.6"]` left alone. Required to accept the OpenHands v1.5 golden fixture. Daydream still EMITS v1.6 only. | ✓ |
| Narrow to v1.6 only | Edit the Literal to `["ATIF-v1.6"]`. Symmetric with our emission. Rejects the OpenHands v1.5 golden fixture; non-mechanical edit. | |
| v1.5–v1.6 — accept the goldens we ship, reject older | Accept exactly what our golden fixtures use. Rejects v1.0–v1.4. Also a non-mechanical edit. | |

**User's choice:** v1.0–v1.6 — keep Harbor's default.

### Q2: If a future ATIF v1.7 trajectory file is fed to our vendored validator (before we re-vendor), what should happen?

| Option | Description | Selected |
|--------|-------------|----------|
| Reject as unknown version | Pydantic Literal naturally rejects unknown values; consumers see a clear validation error. Re-vendor on Harbor update is the documented path. | ✓ |
| Warn + best-effort accept | Catch the Literal failure, log a warning, attempt to validate with v1.6 rules. Could mask real schema breaks. | |

**User's choice:** Reject as unknown version.

---

## Golden fixture selection & breadth

### Q1: How many of Harbor's golden trajectories do we vendor?

| Option | Description | Selected |
|--------|-------------|----------|
| One Terminus-2 (v1.6) + one OpenHands (v1.5) | Representative pair — covers both schema versions in our accepted range and both reference agents. Smallest disk + test-runtime footprint that still proves the validator works end-to-end. | ✓ |
| All of Harbor's `tests/golden/` fixtures | Maximum coverage. Adds disk size and a slower parametrized test loop; may include fixtures we don't strictly need. | |
| Smallest single fixture only | Minimal proof of life. Insufficient for cross-version coverage; doesn't actually exercise OpenHands shape. | |

**User's choice:** One Terminus-2 + one OpenHands.

### Q2: Where do golden fixtures live in our tree?

| Option | Description | Selected |
|--------|-------------|----------|
| `tests/fixtures/atif_golden/<source>/<file>.json` | Source-namespaced subdirs. Self-documenting, parametrized test discovers via Glob. | ✓ |
| `tests/fixtures/atif_golden/*.json` flat | All fixtures dumped flat with descriptive filenames. Simpler glob; loses natural grouping. | |

**User's choice:** Source-namespaced subdirs.

### Q3: Should we also vendor a deliberately-broken trajectory fixture for negative-path testing in Phase 5?

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — include one minimally-broken fixture | Vendor (or hand-author) e.g., `atif_golden/_invalid/non-sequential-step-id.json` so TEST-04's negative test has something concrete to point at. Tiny effort; closes a 'looks-done-but-isn't' checklist item. | ✓ |
| No — hand-author it inline in the test in Phase 5 | Keep Phase 1 strictly to vendoring valid Harbor outputs. Phase 5's negative test mutates a valid fixture in-memory or hand-rolls a bad dict in the test itself. | |
| Defer — Phase 5 owns this decision | Don't decide now. | |

**User's choice:** Yes — include one minimally-broken fixture.

---

## Claude's Discretion

- Exact filename for the broken negative-path fixture — `non-sequential-step-id.json`, `bad-step-id.json`, etc. Pick whichever reads cleanest in parametrized test output.
- Whether to drop a one-line `daydream/atif/__init__.py` docstring summarizing the package, or leave the file with `__all__` only.
- Order of imports inside the `__all__` declaration.
- Whether to add a `daydream/atif/__init__.py` `__version__` constant pinned to the upstream Harbor tag — not required by VEND-* but cheap and useful for diagnostics. Listed as deferred-but-cheap; pick up if Phase 1 planning has spare scope.

## Deferred Ideas

- `daydream replay <trajectory.json>` / `daydream stats <trajectory.json>` subcommands (v2 requirements in REQUIREMENTS.md).
- Multimodal `ContentPart` / `ImageSource` support (Out of Scope per PROJECT.md; classes ride along but daydream never constructs them).
- `--no-redact` escape hatch on the Phase 4 redactor (deferred to post-milestone per PROJECT.md).
- Dual-write phase keeping `_log_debug` alongside trajectories (explicitly Out of Scope; do not reintroduce).
