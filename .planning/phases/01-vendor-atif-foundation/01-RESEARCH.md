# Phase 1: Vendor ATIF Foundation - Research

**Researched:** 2026-04-26
**Domain:** Apache-2.0 source vendoring of Pydantic v2 models + validator from a tagged GitHub release into a brownfield Python 3.12 / uv project
**Confidence:** HIGH

## Summary

Vendor `harbor.models.trajectories.*` (11 files, ~16 KB total) and `harbor.utils.trajectory_validator` (1 file, 289 lines, 10.6 KB) from Harbor **v0.5.0** (commit `5795e7638fbe0ee5d7923b6311df2c9f3747dcf0`, published 2026-04-23) into `daydream/atif/`. Apply two-substitution import rename. Strip the validator's `def main()` + `if __name__ == "__main__":` block (clean truncation from L226 to EOF — no dangling imports because argparse/sys are imported inside `main()` itself). Add `pydantic>=2.11.7` to `[project.dependencies]` (no-op for resolution: `uv.lock` already pins pydantic 2.12.5 transitively via `mcp`). Vendor one Terminus-2 v1.6 fixture (`hello-world-invalid-json.trajectory.json`, 7.4 KB) and one OpenHands v1.5 fixture (`hello-world.trajectory.json`, 27.7 KB). Hand-author one negative fixture under `_invalid/`. Add a ruff `per-file-ignores` exemption for `daydream/atif/**`. Add a smoke test under `tests/` that loads each golden JSON and runs it through the vendored validator.

**Primary recommendation:** Execute the vendor as a single mechanical operation — `git clone --depth=1 --branch v0.5.0`, `cp -r`, two `sed` substitutions, truncate validator at L225, write LICENSE+NOTICE, append explicit pydantic dep + ruff per-file-ignores stanza to `pyproject.toml`, run `uv sync` (expect no resolution change), drop one smoke test, run `make check`. Do NOT touch any production daydream module — this phase is purely additive code under `daydream/atif/` plus three lines in `pyproject.toml`.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| ATIF Pydantic models (Trajectory, Step, ToolCall, etc.) | `daydream/atif/models/` (vendored library code) | — | Authoritative third-party source; never imported by production code in this phase |
| ATIF validator (cross-reference + image-path checks) | `daydream/atif/validator.py` (vendored library code) | — | Programmatic-only after stripping CLI; consumers call `validate(trajectory)` |
| Public API re-export (`Trajectory`, `Step`, `validate`, …) | `daydream/atif/__init__.py` (daydream-authored shim) | — | Stable surface for Phase 2's recorder to import |
| Apache-2.0 attribution | `daydream/atif/{LICENSE,NOTICE}` (daydream-authored docs) | — | Required by Apache-2.0 §4(d); single source of provenance per D-02 |
| Golden trajectory fixtures (round-trip test corpus) | `tests/fixtures/atif_golden/<source>/` (data) | `tests/fixtures/atif_golden/_invalid/` (negative) | Mirrors existing `tests/fixtures/codex_jsonl/` precedent |
| Smoke test for vendoring correctness | `tests/test_atif_vendor_smoke.py` (new) | — | One-shot proof that goldens validate; later expanded to `test_atif_models.py` in Phase 5 (TEST-04) |
| Dependency declaration | `pyproject.toml [project.dependencies]` | `[tool.ruff.lint.per-file-ignores]` | Explicit pydantic floor + lint-exempt vendored tree |

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| VEND-01 | `harbor.models.trajectories.*` source vendored into `daydream/atif/models/` (Apache-2.0 LICENSE + NOTICE included) | §"Harbor Source Map" lists all 11 files with sizes; §"Apache-2.0 NOTICE Composition" provides minimum-viable LICENSE+NOTICE content |
| VEND-02 | `harbor.utils.trajectory_validator` source vendored into `daydream/atif/validator.py` with no external Harbor imports | §"Vendoring Command Sequence" provides the exact `sed` rename one-liner; §"Validator Truncation Boundary" pinpoints L226–end as the strip range |
| VEND-03 | `pydantic>=2.11.7` promoted from transitive to explicit `[project.dependencies]` entry | §"Pydantic Floor Compatibility" confirms current `uv.lock` pins 2.12.5; floor add is a no-op for resolution |
| VEND-04 | Vendored modules import only from stdlib + `pydantic`; no remaining `from harbor import …` references anywhere in the daydream source tree | §"Verification Commands" provides the grep audit; current tree already has zero matches (verified) |
| VEND-05 | Harbor's golden trajectory fixtures (Terminus-2 + OpenHands) vendored into `tests/fixtures/atif_golden/` | §"Golden Fixture Selection" identifies the smallest v1.6 + v1.5 candidates with line/byte counts |
</phase_requirements>

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Vendor from the **latest tagged Harbor release** at the time the vendoring commit lands. Tagged releases are stable, semver-meaningful, and easier to point consumers at than a main-branch SHA.
- **D-02:** Provenance recorded **inline in `daydream/atif/NOTICE`** — single source of truth alongside Apache-2.0 attribution. Required line shape: `Vendored from harbor-framework/harbor@<tag>, commit <sha>, on <YYYY-MM-DD>`. No separate `VENDOR.md`; no per-file comment headers (original Harbor headers stay intact, but no daydream-specific provenance is added per file).
- **D-03:** **Mechanical-only edit policy.** The ONLY allowed transformation is import-path renames:
  - `harbor.models.trajectories` → `daydream.atif.models`
  - `harbor.utils.trajectory_validator` → `daydream.atif.validator`
  - Internal `from harbor.models.trajectories import …` references inside vendored files get the same rename treatment.
  - **NOT allowed in Phase 1:** reformatting to daydream's ruff/mypy style, dropping unused fields/methods, simplifying docstrings, removing comments. Original copyright headers stay.
- **D-04:** **Re-vendor wholesale on demand** for future Harbor updates. No local patches; no VENDOR.md drift log.
- **D-05:** Inside `daydream/atif/models/`, **mirror Harbor's file split exactly**.
- **D-06:** **`daydream/atif/__init__.py` re-exports models + a `validate()` function** declared explicitly via `__all__`. One-stop shop for Phase 2's recorder; programmatic validator surfaced as a top-level `validate` callable.
- **D-07:** **Strip Harbor's `python -m harbor.utils.trajectory_validator` CLI entry point.** Drop the `if __name__ == "__main__":` block and any argparse/CLI scaffolding. Programmatic-only.
- **D-08:** **`validate()` is pure passthrough** to a freshly-constructed `TrajectoryValidator`, returning `.validate(input) -> bool`. `get_errors() -> list[str]` accessible via `from daydream.atif.validator import TrajectoryValidator`. No `ValidationResult` dataclass; no exception path.
- **D-09:** **Validator accepts ATIF-v1.0 through ATIF-v1.6** — Harbor's existing `Literal` is left as-is.
- **D-10:** **Future-version trajectories (v1.7+) reject naturally** via Pydantic `Literal` mismatch.
- **D-11:** **Vendor one Terminus-2 (v1.6) + one OpenHands (v1.5) fixture** as the representative pair for round-trip testing.
- **D-12:** **Fixture path layout:** `tests/fixtures/atif_golden/terminus2/<filename>.json` and `tests/fixtures/atif_golden/openhands/<filename>.json`.
- **D-13:** **Hand-author one deliberately-broken fixture under `tests/fixtures/atif_golden/_invalid/`** (suggested: `non-sequential-step-id.json`).

### Claude's Discretion

- Exact filename for the broken negative-path fixture (D-13).
- Whether to add a `daydream/atif/__init__.py` docstring summarizing the package, or leave the file with `__all__` only.
- Order of imports inside `daydream/atif/__init__.py` `__all__` declaration.
- Whether to add a `daydream/atif/__init__.py` `__version__` constant pinned to the upstream Harbor tag.

### Deferred Ideas (OUT OF SCOPE)

- `__version__` constant for diagnostics (deferred unless spare scope).
- Multimodal `ContentPart` / `ImageSource` daydream-side support (out of scope per PROJECT.md; classes ride along but daydream emits text-only).
- `--no-redact` escape hatch (deferred to post-milestone).
- `daydream replay` / `daydream stats` subcommands (v2 requirements).
- Dual-write phase keeping `_log_debug` alongside trajectories (explicitly Out of Scope).

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | `>=2.11.7` (lockfile resolves to `2.12.5` today) | Vendored ATIF model declarations; runtime validation | Already a transitive dep via `mcp` (claude-agent-sdk's transitive); promoting to explicit floor is the Harbor team's canonical declared minimum [VERIFIED: pydantic 2.12.5 in `uv.lock` line `version = "2.12.5"`; latest pydantic on PyPI = 2.13.3, verified via `https://pypi.org/pypi/pydantic/json` 2026-04-26] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| (stdlib) `pathlib`, `json`, `typing`, `datetime` | n/a | Used inside vendored validator + Step model | Already imported by Harbor source; no new dep |
| pytest | `>=9.0.3` (current dev dep) | Smoke test for goldens | Existing harness |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Vendor Harbor source | Add `harbor` as runtime dep | 21+ transitive deps including `litellm` (March 2026 PyPI quarantine for malicious `.pth`); explicitly rejected in PROJECT.md `## Out of Scope` |
| Vendor models + validator | Hand-roll Pydantic models from JSON Schema | Schema drift risk; reinvents Harbor's validator's image-path checks and tool_call_id intra-step scoping; rejected in PITFALLS.md technical-debt table |
| Vendor at v0.5.0 (latest tag) | Pin to a main-branch SHA | Tagged releases are semver-meaningful and easier to point consumers at; D-01 mandates tagged release |

**Installation:**

```bash
# Append to [project.dependencies] in pyproject.toml:
#     "pydantic>=2.11.7",
uv sync   # expect: "Resolved N packages" with no version changes (lockfile already at 2.12.5)
```

**Version verification:** [VERIFIED: 2026-04-26]
- Harbor latest tag: `v0.5.0` published 2026-04-23 [VERIFIED via `https://api.github.com/repos/harbor-framework/harbor/releases/latest`]
- Harbor v0.5.0 commit SHA: `5795e7638fbe0ee5d7923b6311df2c9f3747dcf0` [VERIFIED via `https://api.github.com/repos/harbor-framework/harbor/tags?per_page=10`]
- pydantic in current `uv.lock`: 2.12.5 [VERIFIED via grep on uv.lock]
- Latest pydantic on PyPI: 2.13.3 [VERIFIED via `https://pypi.org/pypi/pydantic/json` 2026-04-26]

## Architecture Patterns

### System Architecture Diagram

```
                        Phase 1 = additive only
                       (no production import yet)

┌─────────────────────────────┐
│ Harbor v0.5.0 (upstream)    │
│ src/harbor/models/          │
│   trajectories/{11 files}   │   git clone --depth=1 --branch v0.5.0
│ src/harbor/utils/           │ ────────────────────────────────────┐
│   trajectory_validator.py   │                                     │
│ tests/golden/terminus_2/*   │                                     ▼
│ tests/golden/openhands/*    │             ┌────────────────────────────────────┐
└─────────────────────────────┘             │  cp -r + sed import-rename         │
                                            │  + truncate validator at L225      │
                                            └─────────────────────────────────────┘
                                                            │
                                                            ▼
            ┌──────────────────────────────────────────────────────────────────┐
            │  daydream/atif/                                                  │
            │   ├── __init__.py        (daydream-authored: re-exports + __all__)│
            │   ├── LICENSE            (Apache-2.0 verbatim)                   │
            │   ├── NOTICE             (provenance per D-02)                   │
            │   ├── models/            (mirror of Harbor's 11-file split)      │
            │   │     ├── __init__.py                                          │
            │   │     ├── agent.py / content.py / final_metrics.py             │
            │   │     ├── metrics.py / observation.py / observation_result.py  │
            │   │     ├── step.py / subagent_trajectory_ref.py                 │
            │   │     ├── tool_call.py / trajectory.py                         │
            │   └── validator.py       (Harbor validator, L1–L225 only)        │
            └──────────────────────────────────────────────────────────────────┘
                                                            │
                                                            ▼
            ┌──────────────────────────────────────────────────────────────────┐
            │  tests/fixtures/atif_golden/                                     │
            │   ├── terminus2/hello-world-invalid-json.trajectory.json (v1.6) │
            │   ├── openhands/hello-world.trajectory.json (v1.5)              │
            │   └── _invalid/non-sequential-step-id.json (negative)           │
            │                                                                 │
            │  tests/test_atif_vendor_smoke.py                                 │
            │   loads each golden, asserts validate() succeeds (or fails)     │
            └──────────────────────────────────────────────────────────────────┘

            pyproject.toml additions:
            • [project.dependencies]   "pydantic>=2.11.7"
            • [tool.ruff.lint.per-file-ignores]   "daydream/atif/**" = ["ALL"]

            Production daydream/* code: UNTOUCHED in Phase 1.
```

### Recommended Project Structure

```
daydream/
└── atif/                                  # NEW (vendored library)
    ├── __init__.py                        # daydream-authored re-export shim
    ├── LICENSE                            # verbatim Apache-2.0
    ├── NOTICE                             # provenance + Harbor attribution
    ├── models/                            # mirror harbor/models/trajectories/
    │   ├── __init__.py                    # vendored, import-renamed
    │   ├── agent.py                       # vendored, import-renamed
    │   ├── content.py                     # vendored (no internal imports)
    │   ├── final_metrics.py               # vendored
    │   ├── metrics.py                     # vendored
    │   ├── observation.py                 # vendored, import-renamed
    │   ├── observation_result.py          # vendored, import-renamed
    │   ├── step.py                        # vendored, import-renamed (4 internal imports)
    │   ├── subagent_trajectory_ref.py     # vendored
    │   ├── tool_call.py                   # vendored
    │   └── trajectory.py                  # vendored, import-renamed (3 internal imports)
    └── validator.py                       # vendored L1–L225 only, import-renamed (1 import)

tests/
└── fixtures/
    └── atif_golden/                       # NEW
        ├── terminus2/
        │   └── hello-world-invalid-json.trajectory.json    # 7,405 B  v1.6  5 steps
        ├── openhands/
        │   └── hello-world.trajectory.json                 # 27,697 B v1.5  6 steps
        └── _invalid/
            └── non-sequential-step-id.json                 # ~400 B   v1.6  hand-authored
└── test_atif_vendor_smoke.py              # NEW — one-shot validator round-trip
```

### Pattern 1: Programmatic-only validator wrapper (the `validate()` re-export)

**What:** `daydream/atif/__init__.py` exposes `validate()` as a top-level callable that constructs a fresh `TrajectoryValidator` and returns the bool result.

**When to use:** Any caller (Phase 2 recorder finalize, Phase 5 round-trip test) that wants a one-liner schema check without managing validator state.

**Example:**

```python
# daydream/atif/__init__.py  (Claude-authored shim — NOT vendored, NOT subject to D-03)
"""ATIF v1.6 trajectory models and validator (vendored from Harbor v0.5.0)."""

from daydream.atif.models import (
    Agent,
    ContentPart,
    FinalMetrics,
    ImageSource,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from daydream.atif.validator import TrajectoryValidator


def validate(trajectory: dict | str | object, *, validate_images: bool = True) -> bool:
    """Validate an ATIF trajectory (dict, JSON string, or path).

    Pure passthrough to a freshly-constructed TrajectoryValidator (D-08).
    Returns True iff the trajectory matches ATIF v1.0–v1.6 schema.
    """
    return TrajectoryValidator().validate(trajectory, validate_images=validate_images)


__all__ = [
    "Agent",
    "ContentPart",
    "FinalMetrics",
    "ImageSource",
    "Metrics",
    "Observation",
    "ObservationResult",
    "Step",
    "SubagentTrajectoryRef",
    "ToolCall",
    "Trajectory",
    "TrajectoryValidator",
    "validate",
]
```

**Note:** `Observation` (the wrapper) and `ObservationResult` (the inner) are BOTH exported — this matches Harbor's own `__init__.py` (verified). CONTEXT.md D-06 listed `ObservationResult` in the example but Harbor's full surface includes both. Phase 2's recorder will use `Observation` to attach result lists to steps.

### Pattern 2: Mechanical sed-rename across the vendored tree (D-03)

**What:** Two regex substitutions applied to every `.py` file under `daydream/atif/`.

**When to use:** Once, immediately after copying the source tree from a Harbor checkout. Re-applied identically on every wholesale re-vendor (D-04).

**Example (BSD/GNU portability):**

```bash
# macOS BSD sed (note the empty '' after -i)
find daydream/atif -name '*.py' -type f -exec sed -i '' \
  -e 's|from harbor\.models\.trajectories|from daydream.atif.models|g' \
  -e 's|from harbor\.utils\.trajectory_validator|from daydream.atif.validator|g' \
  {} +

# Linux GNU sed (no '' after -i)
find daydream/atif -name '*.py' -type f -exec sed -i \
  -e 's|from harbor\.models\.trajectories|from daydream.atif.models|g' \
  -e 's|from harbor\.utils\.trajectory_validator|from daydream.atif.validator|g' \
  {} +
```

Two anchors are sufficient because:
- All vendored model files use `from harbor.models.trajectories.<X> import <Y>` (verified by inspecting `step.py`, `trajectory.py`, `observation.py`, `observation_result.py`).
- The validator's only Harbor import is `from harbor.models.trajectories import Trajectory` (verified at L13).
- No vendored file uses bare `import harbor` or `import harbor.X` style.

**Cross-platform alternative** (recommended for the plan, since `daydream` developers are on macOS but CI runs on Linux):

```bash
# Pure Python fallback — runs identically on macOS BSD sed and Linux GNU sed
python3 - <<'EOF'
import pathlib, re
for p in pathlib.Path("daydream/atif").rglob("*.py"):
    text = p.read_text()
    text = re.sub(r"from harbor\.models\.trajectories", "from daydream.atif.models", text)
    text = re.sub(r"from harbor\.utils\.trajectory_validator", "from daydream.atif.validator", text)
    p.write_text(text)
EOF
```

### Pattern 3: Validator truncation (D-07)

**What:** Delete lines 226–289 of the vendored `validator.py` (the `def main():` function and its `if __name__ == "__main__":` block).

**Why this is safe (and consistent with mechanical-only D-03):**

The `argparse` and `sys` imports are inside `def main():` (lines 228–229), NOT at module top — verified at L7–L13:

```python
import json                              # L7
from pathlib import Path                 # L8
from typing import Any, Dict, List, Union  # L9
                                         # L10 (blank)
from pydantic import ValidationError     # L11
                                         # L12 (blank)
from harbor.models.trajectories import Trajectory  # L13
```

So truncating at L225 (the line ending `def validate_trajectory(...)`'s body — last statement is `return validator.validate(trajectory)` at L224, blank L225) leaves NO dangling imports at module top.

**Concrete commands** (after the cp + sed steps above):

```bash
# Cut at L225 inclusive — the last "real" line is `return validator.validate(trajectory)` at L224.
# Pure Python is portable; BSD vs GNU sed differs on `i\` semantics.
python3 - <<'EOF'
import pathlib
p = pathlib.Path("daydream/atif/validator.py")
lines = p.read_text().splitlines(keepends=True)
# Truncate from `def main():` line (1-indexed L226) to EOF.
trimmed = []
for line in lines:
    if line.startswith("def main():"):
        break
    trimmed.append(line)
# Strip trailing blank lines accumulated before def main().
while trimmed and trimmed[-1].strip() == "":
    trimmed.pop()
trimmed.append("\n")  # restore single trailing newline (POSIX file convention)
p.write_text("".join(trimmed))
EOF
```

### Anti-Patterns to Avoid

- **Reformatting vendored code to match daydream's ruff style.** Violates D-03; makes re-vendor diffs noisy. Solution: ruff `per-file-ignores` (see Don't Hand-Roll table).
- **Adding `from __future__ import annotations` to vendored files.** Harbor's models don't use it (verified for all 11 files). They use Python 3.10+ `str | None` directly, which works fine under `python_version = "3.12"`. Adding the future import is a non-mechanical edit.
- **Per-file daydream copyright headers.** D-02 forbids it. Original Harbor copyright headers stay; daydream provenance lives in NOTICE only.
- **Constructing ATIF models inside `daydream/atif/__init__.py`.** The `__init__.py` is a re-export shim. ALL `Step()` / `ToolCall()` / `Trajectory()` construction lives in `daydream/trajectory.py` (Phase 2 deliverable). PITFALLS.md Pitfall 14.
- **Importing from `daydream.atif.*` anywhere in `daydream/runner.py`, `daydream/agent.py`, or `daydream/phases.py` in this phase.** Phase 1 is purely additive. Production integration is Phase 2 (CORE-01).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ATIF schema validation | Hand-rolled JSON Schema → Python validator | Harbor's vendored `TrajectoryValidator` | Harbor's validator does cross-reference checks (intra-step `tool_call_id` scoping, image-path resolution, `step_id` sequentiality) that a JSON-Schema-only check would miss; PITFALLS.md technical-debt table flags this as "never acceptable" |
| Lint compliance for vendored Pydantic models | Reformat code to match `ruff check daydream` | `[tool.ruff.lint.per-file-ignores]` exemption | D-03 mechanical-only edit policy; reformat would create per-file diff noise on every re-vendor |
| Apache-2.0 LICENSE text | Type or summarize the license | `curl https://www.apache.org/licenses/LICENSE-2.0.txt > daydream/atif/LICENSE` (or copy from Harbor v0.5.0's root LICENSE — verified to be plain Apache-2.0 verbatim, 11,357 B) | License text must be verbatim per Apache-2.0 §4(a); summarizing voids attribution |
| Negative-path fixture authoring | Generate via Harbor pipeline | Hand-author by copying a valid 2-step trajectory and changing `step_id: 2` to `step_id: 3` | A 2-step deliberate-break is the smallest valid-then-broken case; size <500 B; reads cleanly in pytest output |
| Cross-platform sed | Maintain BSD-vs-GNU `-i` branches in the plan | Use the Python `re.sub()` fallback shown in Pattern 2 | macOS dev box vs Linux CI portability; one less footgun |
| Per-file vendor provenance comments | Add `# Vendored from harbor@v0.5.0` to every file head | Single `daydream/atif/NOTICE` per D-02 | Re-vendor diff noise; D-02 explicit rule |

**Key insight:** Mechanical-only vendoring is itself a "don't hand-roll" stance — every shortcut that "looks cleaner" (reformatting, dropping unused multimodal classes, simplifying validator error messages) increases re-vendor cost (D-04). The cost of NOT hand-rolling is one ruff per-file-ignores stanza.

## Common Pitfalls

### Pitfall 1: BSD vs GNU `sed -i` portability

**What goes wrong:** `sed -i 's/.../.../g' file.py` works on Linux/CI but fails on macOS with `sed: 1: "file.py": invalid command code f`. Conversely, `sed -i '' 's/.../.../g'` works on macOS but creates a literal `''` backup file on Linux.

**Why it happens:** BSD sed (macOS default) requires an explicit empty-string argument after `-i` for in-place. GNU sed (Linux) treats anything after `-i` as a backup-suffix.

**How to avoid:** Use the Python `re.sub()` fallback shown in Pattern 2 above. Eliminates the divergence; identical command on every platform; no shell-escaping surprises with the regex anchors.

**Warning signs:** Plan instruction reads "run `sed -i ...`" without specifying which sed; CI log shows `''` files being created.

---

### Pitfall 2: Ruff's `I` rule (isort) flags vendored import order

**What goes wrong:** `make lint` runs `ruff check daydream`, which includes the `I` rule (isort). Harbor's vendored files may use a different stdlib/third-party ordering than daydream's convention; ruff complains; fixing the order is a non-mechanical edit (violates D-03).

**Why it happens:** Default `select = ["E", "F", "I", "W"]` in `pyproject.toml` runs against the entire `daydream/` tree.

**How to avoid:** Add a `[tool.ruff.lint.per-file-ignores]` stanza to `pyproject.toml` exempting `daydream/atif/**` from all ruff rules. See "Code Examples" below for exact syntax.

**Warning signs:** First `make lint` run after vendoring shows multiple `I001` (Import block is un-sorted) errors under `daydream/atif/`.

---

### Pitfall 3: Mypy ignoring vendored models silently

**What goes wrong:** `[tool.mypy]` has `ignore_missing_imports = true` — this means if a vendored file has a typo'd import (e.g., a missed sed replacement), mypy stays silent. The error only surfaces at runtime when something tries to `import daydream.atif.models.step`.

**Why it happens:** `ignore_missing_imports` is global; it can't distinguish "imports a non-stub-having third-party package" (legitimate) from "imports a typo'd internal module" (bug).

**How to avoid:**
- Add the smoke test (`tests/test_atif_vendor_smoke.py`) — at minimum it does `from daydream.atif import Trajectory, validate` at import time. Any sed miss surfaces as `ImportError` immediately on `pytest -v`.
- After running the sed step, run `python -c "import daydream.atif"` once before committing. Forces actual import resolution.

**Warning signs:** `make typecheck` passes but `make test` fails with `ModuleNotFoundError: No module named 'harbor.models.trajectories'` — sed missed a file.

---

### Pitfall 4: Pydantic forward-reference resolution at import time

**What goes wrong:** Vendored `step.py` imports `Metrics`, `Observation`, `ToolCall`, `ContentPart` from sibling modules. Pydantic v2 builds models at class-definition time and resolves forward refs. If any sibling module fails to import, `Step` raises `PydanticUserError` at the import-line, not at first use.

**Why it happens:** Pydantic v2 uses `model_rebuild()` automatically on first model usage but the import-time evaluation of `list[Step]` annotations in `Trajectory` may trigger ref resolution earlier than expected.

**How to avoid:** Run the smoke test in CI; it's a 5-line `from daydream.atif import Trajectory; Trajectory.model_validate({...})` that proves all 11 model files cohere. Verified offline that all 11 files use plain `str | None`-style annotations (no `from __future__ import annotations`), which means refs resolve eagerly — failures fail loud.

**Warning signs:** `pytest --collect-only` raises `pydantic.errors.PydanticUserError: ... is not fully defined`; usually points at a missed sibling import.

---

### Pitfall 5: Vendored validator's `image_path` checks against absent fixture image dir

**What goes wrong:** `TrajectoryValidator.validate(path)` defaults `validate_images=True`, which walks `trajectory_data` looking for `ImageSource` entries with `path` fields and asserts they exist on disk. Harbor's golden fixtures are pure-text trajectories (no images) so this is a no-op for them. But if a future Phase 2 test passes a trajectory with a relative image path, the validator's `_trajectory_dir` resolution (relative to the JSON file) may not work as expected from the daydream test layout.

**Why it happens:** Harbor's validator infers `_trajectory_dir = path.parent` when given a path; uses that as the image-resolution root.

**How to avoid:** For the smoke test in this phase, pass `validate_images=True` (default) — the goldens have no images, so it's vacuously true. Document in the smoke test's docstring that Phase 2 should pass `validate_images=False` when validating in-memory dicts that lack a filesystem anchor.

**Warning signs:** Smoke test fails with `Image file not found: ...` against a fixture that has no images — likely cause is a wrong `path` field in the negative fixture; doesn't apply to v1.5/v1.6 happy-path goldens.

## Runtime State Inventory

> Phase 1 is greenfield-additive (no rename/refactor of existing daydream code). Section omitted intentionally per the research template's guidance.

## Code Examples

### Exact `pyproject.toml` deltas

Insert into `[project.dependencies]` (alphabetical insert position is between `pyfiglet` and `rich`, but the existing list is not sorted — append before `tree-sitter` block is fine):

```toml
[project]
# ... existing lines ...
dependencies = [
    "claude-agent-sdk==0.1.52",
    "anyio>=4.0",
    "rich>=13.0",
    "pyfiglet>=1.0",
    "pydantic>=2.11.7",        # NEW (VEND-03): explicit floor for vendored ATIF models
    "tree-sitter==0.25.2",
    # ... existing tree-sitter lines ...
]
```

Append a new stanza (the file currently has `[tool.ruff]` and `[tool.ruff.lint]` only; `per-file-ignores` is a sub-table of `[tool.ruff.lint]`):

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.ruff.lint.per-file-ignores]
# Vendored from Harbor v0.5.0; see daydream/atif/NOTICE.
# Mechanical-only edit policy (D-03): no reformatting allowed.
"daydream/atif/**" = ["E", "F", "I", "W"]
```

[CITED: ruff docs `https://docs.astral.sh/ruff/settings/#lint_per-file-ignores`] The `per-file-ignores` table accepts glob patterns (matched against file paths); listing every active rule code disables linting for that glob. Using `["ALL"]` would also work but the explicit list mirrors `[tool.ruff.lint] select` and is robust if `select` expands later.

### Exact `daydream/atif/NOTICE` content (D-02 conformant)

```
daydream/atif/
==============

This directory contains source code vendored from the Harbor framework
(https://github.com/harbor-framework/harbor), Apache License 2.0.

Vendored from harbor-framework/harbor@v0.5.0, commit 5795e7638fbe0ee5d7923b6311df2c9f3747dcf0, on 2026-04-26

Vendored to avoid Harbor's transitive dependency surface. The trajectory
models and validator are pure-Pydantic + stdlib (~700 LOC) and benefit
from being decoupled from Harbor's runtime install graph.

Re-vendor wholesale on Harbor updates; do not carry local patches.
The only allowed transformation is import-path renames:

    harbor.models.trajectories  -> daydream.atif.models
    harbor.utils.trajectory_validator -> daydream.atif.validator

Plus removal of the validator's `python -m harbor.utils.trajectory_validator`
CLI entry point (programmatic use only).

----

Original Harbor LICENSE (Apache-2.0) is preserved at daydream/atif/LICENSE.

Harbor copyright (from upstream LICENSE):
    Copyright 2024-2026 The Laude Institute / Harbor Framework contributors

Daydream attribution:
    Daydream incorporates this vendored code under the terms of the
    Apache License, Version 2.0. See daydream/atif/LICENSE for the
    full license text.
```

[CITED: Apache-2.0 §4(d)] requires recipients to retain a NOTICE if the original work has one, and §4(c) requires preserving copyright/attribution notices. Harbor v0.5.0 has a top-level LICENSE but **no top-level NOTICE file** [VERIFIED: HTTP 404 on `https://raw.githubusercontent.com/harbor-framework/harbor/v0.5.0/NOTICE` 2026-04-26], so we are free to author daydream's own NOTICE. The Harbor copyright line above is a reasonable approximation; the planner may want to verify it against Harbor's `LICENSE` file's `[yyyy] [name of copyright owner]` line during execution.

### Exact `daydream/atif/LICENSE` content

Verbatim Apache-2.0 — the canonical text, 11,357 bytes [VERIFIED: matches Harbor v0.5.0's `LICENSE` file size from GitHub API].

```bash
# Fetch verbatim during execution (do NOT type or summarize):
curl -fsSL https://raw.githubusercontent.com/harbor-framework/harbor/v0.5.0/LICENSE > daydream/atif/LICENSE
# OR copy from the cloned tarball:
cp /tmp/harbor-v0.5.0/LICENSE daydream/atif/LICENSE
```

### Smoke test (`tests/test_atif_vendor_smoke.py`)

Mirrors the existing `FIXTURES_DIR = Path(__file__).parent / "fixtures" / ...` precedent from `tests/test_backend_codex.py`:

```python
"""Smoke test for the vendored ATIF foundation (Phase 1).

Verifies VEND-01..05 success criteria:
- Models import cleanly from daydream.atif
- Validator accepts every Terminus-2 + OpenHands golden fixture
- Validator rejects the deliberately-broken negative fixture

Phase 5 (TEST-04) replaces this with a parametrized test_atif_models.py.
"""

import json
from pathlib import Path

import pytest

from daydream.atif import Trajectory, TrajectoryValidator, validate

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "atif_golden"


def test_models_import_cleanly() -> None:
    """VEND-01: top-level public surface is importable."""
    from daydream.atif.models import (
        Agent,
        FinalMetrics,
        Metrics,
        ObservationResult,
        Step,
        ToolCall,
        Trajectory,
    )
    # Exercise __all__: simple no-op references silence flake8 F401.
    assert all(cls.__module__.startswith("daydream.atif.models") for cls in (
        Agent, FinalMetrics, Metrics, ObservationResult, Step, ToolCall, Trajectory,
    ))


def _golden_paths() -> list[Path]:
    return sorted(p for p in GOLDEN_DIR.rglob("*.json") if "_invalid" not in p.parts)


@pytest.mark.parametrize("golden_path", _golden_paths(), ids=lambda p: p.name)
def test_golden_fixtures_validate(golden_path: Path) -> None:
    """VEND-05 + D-09: every Terminus-2 (v1.6) and OpenHands (v1.5) golden validates."""
    assert validate(golden_path) is True


def test_invalid_fixture_rejected() -> None:
    """D-13: deliberately-broken fixture fails validation."""
    invalid_path = GOLDEN_DIR / "_invalid" / "non-sequential-step-id.json"
    validator = TrajectoryValidator()
    assert validator.validate(invalid_path) is False
    # Surface the specific error category for diagnostic clarity.
    assert any("step_id" in err.lower() for err in validator.errors), validator.errors


def test_validate_via_dict_roundtrip() -> None:
    """D-08: programmatic validate() accepts a dict, not just a path."""
    sample_path = GOLDEN_DIR / "terminus2" / "hello-world-invalid-json.trajectory.json"
    data = json.loads(sample_path.read_text())
    # validate_images=False: in-memory dict has no filesystem anchor.
    assert validate(data, validate_images=False) is True
    # Round-trip through Trajectory model proves model_validate works.
    Trajectory.model_validate(data)
```

### Negative-path fixture (`tests/fixtures/atif_golden/_invalid/non-sequential-step-id.json`)

Hand-authored, ~440 B, 2 steps with `step_id: [1, 3]` — Harbor's validator's "step_id sequentiality" check fires on the second step (Pitfall 1 from PITFALLS.md):

```json
{
  "schema_version": "ATIF-v1.6",
  "session_id": "test-non-sequential-step-id",
  "agent": {
    "name": "test-agent",
    "version": "0.0.0",
    "model_name": "test-model"
  },
  "steps": [
    {
      "step_id": 1,
      "source": "user",
      "message": "First user message."
    },
    {
      "step_id": 3,
      "source": "agent",
      "message": "Skipped step_id=2 deliberately to exercise the validator's sequentiality check."
    }
  ]
}
```

This fixture is intentionally minimal:
- v1.6 schema_version (matches the validator's accepted Literal range).
- One user step + one agent step (the smallest non-trivial agent run).
- Only the `step_id` field is broken; everything else is shape-valid.
- ~440 B keeps test corpus small.

### Vendoring command sequence (full execution recipe)

```bash
# 1. Clone Harbor v0.5.0 to a workspace temp dir (NOT under daydream/).
git clone --depth=1 --branch v0.5.0 \
  https://github.com/harbor-framework/harbor.git /tmp/harbor-v0.5.0

# 2. Verify commit SHA matches the NOTICE we'll write.
cd /tmp/harbor-v0.5.0 && git rev-parse HEAD
#   expected: 5795e7638fbe0ee5d7923b6311df2c9f3747dcf0
cd -

# 3. Create target dirs.
mkdir -p daydream/atif/models tests/fixtures/atif_golden/{terminus2,openhands,_invalid}

# 4. Copy models (11 files including __init__.py).
cp /tmp/harbor-v0.5.0/src/harbor/models/trajectories/*.py daydream/atif/models/

# 5. Copy validator.
cp /tmp/harbor-v0.5.0/src/harbor/utils/trajectory_validator.py daydream/atif/validator.py

# 6. Apply the two import-rename substitutions (Python fallback for portability).
python3 - <<'EOF'
import pathlib, re
for p in pathlib.Path("daydream/atif").rglob("*.py"):
    text = p.read_text()
    text = re.sub(r"from harbor\.models\.trajectories", "from daydream.atif.models", text)
    text = re.sub(r"from harbor\.utils\.trajectory_validator", "from daydream.atif.validator", text)
    p.write_text(text)
EOF

# 7. Truncate validator.py at L225 (strip def main + __main__ block per D-07).
python3 - <<'EOF'
import pathlib
p = pathlib.Path("daydream/atif/validator.py")
lines = p.read_text().splitlines(keepends=True)
trimmed = []
for line in lines:
    if line.startswith("def main():"):
        break
    trimmed.append(line)
while trimmed and trimmed[-1].strip() == "":
    trimmed.pop()
trimmed.append("\n")
p.write_text("".join(trimmed))
EOF

# 8. Write LICENSE + NOTICE.
cp /tmp/harbor-v0.5.0/LICENSE daydream/atif/LICENSE
# (write daydream/atif/NOTICE per the content above)

# 9. Write daydream-authored daydream/atif/__init__.py (re-export shim per D-06).
# (see Pattern 1 above for content)

# 10. Copy golden fixtures.
cp /tmp/harbor-v0.5.0/tests/golden/terminus_2/hello-world-invalid-json.trajectory.json \
   tests/fixtures/atif_golden/terminus2/
cp /tmp/harbor-v0.5.0/tests/golden/openhands/hello-world.trajectory.json \
   tests/fixtures/atif_golden/openhands/

# 11. Hand-author negative fixture (see content above).

# 12. Append pyproject.toml deltas (pydantic dep + per-file-ignores).

# 13. Resolve and verify.
uv sync                                            # expect: no version changes
uv run python -c "import daydream.atif; print('OK')"  # smoke import
make check                                         # lint + typecheck + test
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Python 3.7-style typing (`Optional[str]`, `List[X]`) | Python 3.10+ `str \| None`, `list[X]` syntax | Harbor moved to 3.12 native syntax in earlier release | Vendored models import cleanly under daydream's `python_version = "3.12"` mypy config; no `from __future__ import annotations` needed |
| Harbor as a runtime dependency | Vendored ~700 LOC under `daydream/atif/` | This phase (April 2026) | Avoids 21+ transitive deps + March-2026 `litellm` PyPI quarantine |
| ATIF v1.4 (mentioned in older daydream docs) | ATIF v1.6 (Harbor's current default; validator accepts v1.0–v1.6) | Harbor v0.5.0 raised the default | Goldens are at v1.5 (OpenHands) / v1.6 (Terminus-2); validator's `Literal` covers both |
| `python -m harbor.utils.trajectory_validator <file>` | Programmatic `from daydream.atif import validate; validate(path)` | This phase | Smaller surface, no PATH dependency, matches D-07 |

**Deprecated/outdated:**
- `pydantic<2.11.7`: model_validate() semantics differ; Harbor's models depend on the v2.11+ behavior. (Not a real risk — `uv.lock` resolves to 2.12.5.)
- ATIF v1.0–v1.4: still accepted by the vendored validator (Literal range), but daydream emits v1.6 only per PROJECT.md.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Harbor's v0.5.0 LICENSE file is verbatim Apache-2.0 (not a customized variant) | NOTICE/LICENSE Composition | If Harbor added custom clauses, copying the file is correct, but the daydream NOTICE's "Apache License 2.0" reference may be inaccurate. Mitigation: planner verifies via `head -10 /tmp/harbor-v0.5.0/LICENSE` during execution. (Verified offline that the first 5 lines match standard Apache-2.0 boilerplate — `[ASSUMED]` only because we did not read the entire 11k-byte file.) |
| A2 | The Harbor copyright holder line (`Copyright 2024-2026 The Laude Institute / Harbor Framework contributors`) used in the NOTICE template is accurate | NOTICE Composition | Wrong copyright holder would weaken the attribution. Mitigation: planner reads the actual copyright line from Harbor's LICENSE file (typically near the top, format `Copyright [yyyy] [name]`) and substitutes it. |
| A3 | `[tool.ruff.lint.per-file-ignores]` with `"daydream/atif/**" = ["E", "F", "I", "W"]` correctly suppresses the same rules listed in `select` | pyproject.toml deltas | If the syntax is slightly off (e.g., needs glob `daydream/atif/*` not `daydream/atif/**`), `make lint` fails on the first vendored import-order issue. Mitigation: cross-check during execution with `uv run ruff check daydream/atif` after the stanza is added. |
| A4 | Harbor's `validate_trajectory(path)` module-level helper at L213–L223 should remain in the vendored validator (we only strip from `def main():` at L226) | Validator Truncation Boundary | If `validate_trajectory()` is also considered "CLI scaffolding," D-07's intent might cover it. However it's a programmatic helper, not a CLI entry point — leaving it in is consistent with D-08's "programmatic-only" framing. |
| A5 | Hand-authored negative fixture's `extra: {}` omission is valid | Negative-path fixture | The vendored validator should accept `extra` as optional/null. Verified by the fact that `hello-world-invalid-json.trajectory.json` v1.6 fixture has no top-level `extra` field. |

## Open Questions

1. **Harbor's actual LICENSE copyright line text**
   - What we know: Harbor v0.5.0 has a top-level LICENSE file (11,357 B) that is verbatim Apache-2.0.
   - What's unclear: The exact `Copyright [yyyy] [name of copyright owner]` line at the bottom of the standard Apache template — we have not read past the first 5 lines.
   - Recommendation: Plan should include a `head -200 /tmp/harbor-v0.5.0/LICENSE | grep -i copyright` step before writing daydream's NOTICE. The Phase 2 PR can update the NOTICE if the placeholder is wrong.

2. **Whether to vendor a second OpenHands fixture (the larger `traces.json` files)**
   - What we know: D-11 says "one Terminus-2 + one OpenHands." `tests/golden/openhands/` has 2 trajectory files; we picked the 27.7 KB one. The `*.traces.json` files (57–83 KB) are intermediate not-quite-trajectories (different schema).
   - What's unclear: D-11 specifies "trajectory" not "traces" — confirms we should skip `*.traces.json`.
   - Recommendation: Do NOT vendor `*.traces.json` files. They're agent-internal event logs, not ATIF trajectories. (Resolved by D-11 wording — surfacing it for plan reviewer awareness.)

3. **Should `Observation` (the wrapper) be in `__all__` even though CONTEXT.md D-06 only listed `ObservationResult`?**
   - What we know: Harbor's own `__init__.py` exports both `Observation` and `ObservationResult`. The recorder (Phase 2) will need `Observation` to attach result lists to steps (a `Step` has `observation: Observation | None`, and `Observation.results: list[ObservationResult]`).
   - What's unclear: D-06's example list (`Trajectory, Agent, Step, ToolCall, Observation, ObservationResult, Metrics, FinalMetrics, validate`) DOES include `Observation` — re-reading CONTEXT.md, I see `Observation` is in the example. Phase 1 should mirror Harbor's full export set (11 models + `validate` + `TrajectoryValidator` per D-08 access pattern).
   - Recommendation: Mirror Harbor's `__init__.py` re-export set 1:1 plus add `validate` and `TrajectoryValidator`. Resolved by reading D-06 carefully.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `git` | Cloning Harbor v0.5.0 | ✓ (assumed; daydream is itself a git repo) | — | — |
| `python3` | sed-replacement script + LICENSE truncation | ✓ (Python 3.12 required by daydream) | ≥3.12 | — |
| `curl` | Verbatim LICENSE fetch (alt to git clone path) | ✓ (macOS/Linux default) | — | Use `git clone` path instead |
| `uv` | `uv sync` resolution check | ✓ (per CLAUDE.md `make install: uv sync`) | — | — |
| GitHub network access | Cloning Harbor repo | Sandbox blocks `github.com` | — | `dangerouslyDisableSandbox: true` for the clone step ONLY (already verified during research that this is the path); plan instruction must call this out |
| pydantic 2.11.7+ | Vendored model imports | ✓ (lockfile at 2.12.5) | 2.12.5 | — |

**Missing dependencies with no fallback:** None. All hard requirements are present on the dev box.

**Missing dependencies with fallback:** Sandbox network access for the `git clone`. Plan should note that the `git clone --depth=1 --branch v0.5.0 ...` step requires `dangerouslyDisableSandbox: true` (an existing established pattern; the user has whitelisted it for git operations per `~/.claude/CLAUDE.md`).

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 + pytest-asyncio 1.3.0 (existing) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (`asyncio_mode = "auto"`) |
| Quick run command | `uv run pytest -v tests/test_atif_vendor_smoke.py` |
| Full suite command | `uv run pytest -v` (343 existing + smoke test) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| VEND-01 | Models import cleanly via `from daydream.atif.models import …` | unit | `uv run pytest -v tests/test_atif_vendor_smoke.py::test_models_import_cleanly` | ❌ Wave 0 |
| VEND-02 | Validator code is in `daydream/atif/validator.py`, no Harbor imports remain | unit | `uv run python -c "import daydream.atif.validator; assert 'harbor' not in daydream.atif.validator.__file__"` + `! grep -r 'from harbor' daydream/atif/` | ❌ Wave 0 |
| VEND-03 | `pydantic>=2.11.7` declared in `[project.dependencies]`; `uv sync` resolves | integration | `grep -E '"pydantic>=2\.11\.7"' pyproject.toml && uv sync --check` | (config-only, no test file) |
| VEND-04 | No `from harbor` imports in `daydream/` or `tests/` | static | `! grep -rn 'from harbor\|^import harbor' daydream/ tests/` | (verifiable via shell only) |
| VEND-05 | Harbor goldens validate via vendored validator | integration | `uv run pytest -v tests/test_atif_vendor_smoke.py::test_golden_fixtures_validate` | ❌ Wave 0 |
| (Phase guard) | Pre-existing 343 tests still pass | regression | `uv run pytest -v` reports ≥343 passing | ✓ existing |
| (Phase guard) | Lint clean against vendored tree exemption | static | `uv run ruff check daydream` | ✓ existing |
| (Phase guard) | Mypy clean | static | `uv run mypy daydream` | ✓ existing |

### Sampling Rate

- **Per task commit:** `uv run pytest -v tests/test_atif_vendor_smoke.py && uv run ruff check daydream/atif && uv run mypy daydream/atif`
- **Per wave merge:** `make check` (full lint + typecheck + 343+ tests)
- **Phase gate:** `make check` green + zero matches for `grep -rn 'from harbor' daydream/ tests/`

### Wave 0 Gaps

- [ ] `tests/test_atif_vendor_smoke.py` — covers VEND-01, VEND-02, VEND-05 (and D-08, D-13 negative path)
- [ ] `tests/fixtures/atif_golden/terminus2/hello-world-invalid-json.trajectory.json` — Terminus-2 v1.6 corpus
- [ ] `tests/fixtures/atif_golden/openhands/hello-world.trajectory.json` — OpenHands v1.5 corpus
- [ ] `tests/fixtures/atif_golden/_invalid/non-sequential-step-id.json` — negative fixture
- [ ] `tests/conftest.py` — **no changes needed** in this phase; the smoke test loads fixtures via a module-level `GOLDEN_DIR` constant mirroring `test_backend_codex.py`'s `FIXTURES_DIR` pattern. (CORE-10's `_reset_trajectory_recorder` autouse fixture is Phase 2's deliverable; Phase 1 must not pre-introduce conflicting fixtures.)

## Project Constraints (from CLAUDE.md)

| Directive | Source | Compliance Notes |
|-----------|--------|------------------|
| Use `uv` for all commands (`uv run`, `uv sync`) | CLAUDE.md "Runtime" + Makefile | Plan uses `uv sync` for resolution check and `uv run pytest` for test execution. |
| Python 3.12+ required (`requires-python = ">=3.12"`) | `pyproject.toml` | Vendored Harbor models use `str \| None` syntax which is Python 3.10+ — compatible. |
| Conventional commits | `~/.claude/CLAUDE.md` (user instructions) | Plan's commit messages must follow `<type>(<scope>): <subject>` form (e.g., `feat(atif): vendor harbor v0.5.0 trajectory models and validator`). |
| Branch naming `anderskev/<service>/<issue>-<desc>` | `~/.claude/CLAUDE.md` (user instructions) | Current branch is already `anderskev/daydream/53-replace-debug-logging-with-atif-trajectory` — plan operates on this existing branch; no new branch needed. |
| Run git commands with sandbox disabled | `~/.claude/CLAUDE.md` (user instructions) | Plan must annotate every git/curl-to-github step with `dangerouslyDisableSandbox: true`. |
| No `print()` in library code | CONVENTIONS.md | The smoke test uses pytest assertions, not prints. The vendored validator has print statements inside `def main()` which we strip — no leftover prints. |
| GSD workflow enforcement | Project CLAUDE.md | This phase is being executed via `/gsd-plan-phase 1` after `/gsd-discuss-phase 1`; compliant. |
| Module-bloat ban: no `Step()`/`ToolCall()`/`Trajectory()` construction in `phases.py` or `ui.py` | Project CLAUDE.md "Constraints" | Phase 1 adds NO model construction anywhere in production code. The smoke test constructs `Trajectory.model_validate(data)` once — this is in a test file, not phases.py/ui.py. Compliant. |

## Sources

### Primary (HIGH confidence)

- GitHub API `repos/harbor-framework/harbor/releases/latest` → tag `v0.5.0`, published 2026-04-23 [VERIFIED 2026-04-26]
- GitHub API `repos/harbor-framework/harbor/tags?per_page=10` → SHA `5795e7638fbe0ee5d7923b6311df2c9f3747dcf0` [VERIFIED]
- GitHub Contents API `contents/src/harbor/models/trajectories?ref=v0.5.0` → 11 files with sizes [VERIFIED]
- GitHub Contents API `contents/src/harbor/utils?ref=v0.5.0` → trajectory_validator.py (10,635 B) [VERIFIED]
- GitHub Contents API `contents/tests/golden/{terminus_2,openhands}?ref=v0.5.0` → fixture file list [VERIFIED]
- raw.githubusercontent.com `harbor/v0.5.0/src/harbor/utils/trajectory_validator.py` → confirmed L226 = `def main():`, L228–229 = `import argparse, sys` inside main, L13 = single Harbor model import [VERIFIED]
- raw.githubusercontent.com `harbor/v0.5.0/src/harbor/models/trajectories/__init__.py` → 11-name `__all__` [VERIFIED]
- raw.githubusercontent.com `harbor/v0.5.0/src/harbor/models/trajectories/{trajectory,step,observation,observation_result}.py` → import patterns confirmed; no `from __future__` in any file [VERIFIED for all 11 files]
- raw.githubusercontent.com `harbor/v0.5.0/tests/golden/terminus_2/hello-world-invalid-json.trajectory.json` → 5 sequential steps, schema_version `ATIF-v1.6` [VERIFIED via JSON parse]
- raw.githubusercontent.com `harbor/v0.5.0/tests/golden/openhands/hello-world.trajectory.json` → 6 steps, schema_version `ATIF-v1.5` [VERIFIED via JSON parse]
- HTTP 404 on `harbor/v0.5.0/NOTICE` → Harbor has no top-level NOTICE file [VERIFIED]
- `pypi.org/pypi/pydantic/json` → latest 2.13.3 (April 2026) [VERIFIED]
- `daydream/uv.lock` → pydantic resolved to 2.12.5 transitively via mcp [VERIFIED via grep]
- `daydream/pyproject.toml`, `Makefile`, `scripts/hooks/pre-push`, `tests/conftest.py`, `tests/test_backend_codex.py` → fixture loading pattern, lint/typecheck/test commands [VERIFIED via Read]

### Secondary (MEDIUM confidence)

- Apache-2.0 §4 attribution requirements (general knowledge of Apache-2.0 NOTICE conventions) [CITED: `https://www.apache.org/licenses/LICENSE-2.0`]
- ruff `[tool.ruff.lint.per-file-ignores]` syntax [CITED: `https://docs.astral.sh/ruff/settings/#lint_per-file-ignores`]

### Tertiary (LOW confidence)

- Harbor's exact copyright-holder line text (we did not fetch and read the full LICENSE body — only confirmed first 5 lines match Apache template). Plan should grep the LICENSE during execution (Open Question 1).

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — pydantic version, Harbor tag, file inventory all verified live against GitHub today.
- Architecture: HIGH — file paths, sed substitutions, validator boundary all derived from direct inspection of the v0.5.0 source tree.
- Pitfalls: HIGH — derived from reading the actual vendored files; BSD-vs-GNU sed concern is well-known platform reality; ruff per-file-ignores stanza is documented behavior.
- NOTICE/LICENSE composition: MEDIUM — we know Harbor v0.5.0 has no upstream NOTICE (verified) and standard Apache-2.0 LICENSE; the exact copyright-holder line is the one open item (A1, A2).

**Research date:** 2026-04-26
**Valid until:** 2026-05-26 (30 days; Harbor moves slowly between minor releases — v0.4.0 → v0.5.0 was a multi-month gap; if a v0.6.0 ships before this phase merges, planner should re-verify the latest tag)

## RESEARCH COMPLETE
