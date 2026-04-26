# Phase 1: Vendor ATIF Foundation - Pattern Map

**Mapped:** 2026-04-26
**Files analyzed:** 13 (6 daydream-authored, 5+ vendored, 2 config/docs)
**Analogs found:** 6 / 6 daydream-authored files (vendored files are exempt per D-03)

## Convention Exemption Reminder (D-03)

**This phase introduces an unprecedented module category in the daydream codebase: mechanical-only vendored Apache-2.0 source.** Vendored files under `daydream/atif/models/` and `daydream/atif/validator.py` are EXEMPT from daydream's CONVENTIONS.md per CONTEXT.md D-03. The "analog" for these files is *Harbor's own file at the same relative path* — the pattern is **1:1 copy + sed import-rename**, not a daydream-style refactor. The planner must NOT request convention-conformance edits (no `from __future__ import annotations` adds, no docstring rewrites, no import reordering, no `__all__` adjustments to vendored `models/__init__.py`).

Files that ARE subject to daydream conventions in this phase:
- `daydream/atif/__init__.py` (daydream-authored re-export shim)
- `tests/test_atif_vendor_smoke.py` (daydream-authored smoke test)
- `pyproject.toml` (config-only edits)
- `daydream/atif/NOTICE` and `daydream/atif/LICENSE` (text files; verbatim/templated content)
- `tests/fixtures/atif_golden/_invalid/non-sequential-step-id.json` (hand-authored JSON; minimal style)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `daydream/atif/__init__.py` | package re-export shim | n/a (import-time) | `daydream/deep/__init__.py` | exact (pure re-export + `__all__`) |
| `daydream/atif/models/__init__.py` | vendored package init | n/a | Harbor `src/harbor/models/trajectories/__init__.py` | mechanical-only (D-03) |
| `daydream/atif/models/*.py` | vendored Pydantic models | n/a | Harbor `src/harbor/models/trajectories/*.py` | mechanical-only (D-03) |
| `daydream/atif/validator.py` | vendored programmatic validator | request-response (callable) | Harbor `src/harbor/utils/trajectory_validator.py` (L1–L225) | mechanical-only + truncate (D-03, D-07) |
| `daydream/atif/NOTICE` | provenance + Apache attribution | n/a | none in repo (new pattern; D-02) | new — see RESEARCH.md template |
| `daydream/atif/LICENSE` | Apache-2.0 verbatim | n/a | none in repo (new pattern) | new — verbatim copy from Harbor |
| `pyproject.toml` (MOD) | config | n/a | existing `[project.dependencies]`, `[tool.ruff.lint]` blocks in same file | self-analog |
| `tests/test_atif_vendor_smoke.py` | integration test (fixture loader) | file-I/O (read JSON) → validator | `tests/test_backend_codex.py` (FIXTURES_DIR pattern) | exact (file-I/O fixture loader) |
| `tests/fixtures/atif_golden/terminus2/hello-world-invalid-json.trajectory.json` | vendored JSON corpus | n/a | `tests/fixtures/codex_jsonl/*.jsonl` | role-match (vendored fixture under namespaced subdir) |
| `tests/fixtures/atif_golden/openhands/hello-world.trajectory.json` | vendored JSON corpus | n/a | `tests/fixtures/codex_jsonl/*.jsonl` | role-match |
| `tests/fixtures/atif_golden/_invalid/non-sequential-step-id.json` | hand-authored negative fixture | n/a | none in repo (new pattern) | new — see RESEARCH.md L585–605 template |

## Pattern Assignments

### `daydream/atif/__init__.py` (package re-export shim)

**Analog:** `daydream/deep/__init__.py` (the closest analog — pure re-export + `__all__` with no inline class/function definitions, unlike `daydream/backends/__init__.py` which mixes Protocol/dataclass declarations into the same file).

**Module docstring pattern** (`daydream/deep/__init__.py:1-4`):
```python
"""Deep-review mode package.

Exports run_deep once the orchestrator exists (plan 05-09).
"""
```

Apply to atif: a one-line docstring summarizing the package origin (e.g., `"""ATIF v1.6 trajectory models and validator (vendored from Harbor v0.5.0)."""`).

**Re-export block pattern** (`daydream/deep/__init__.py:6-23`):
```python
from daydream.deep.artifacts import (
    alternatives_path,
    check_deep_artifacts,
    dedup_candidates_path,
    deep_dir,
    intent_path,
    per_stack_records_path,
    per_stack_review_path,
)
from daydream.deep.dedup import CandidatePair, build_dedup_candidates
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.deep.orchestrator import run_deep, total_agent_count
```

Apply to atif: imports grouped by submodule (`daydream.atif.models`, `daydream.atif.validator`). Multiple names from one module collected into a single `from X import (...)` block per CONVENTIONS.md "Import Organization."

**`__all__` declaration pattern** (`daydream/deep/__init__.py:25-43`):
```python
__all__ = [
    "DOC_REVIEW_NOTICE",
    "CandidatePair",
    "StackAssignment",
    "alternatives_path",
    "build_dedup_candidates",
    ...
    "run_deep",
    "total_agent_count",
]
```

Note: `daydream/deep/__init__.py` sorts `__all__` entries alphabetically (with `UPPER_SNAKE` constants first, then `PascalCase`, then `lower_case` callables). `daydream/backends/__init__.py:128-140` uses the same alphabetical-within-case-group ordering. Apply this same convention to atif's `__all__`.

**Inline `validate()` function pattern** — there is no perfect analog because `daydream/deep/__init__.py` is pure re-export. The closest pattern for a small wrapper function is in `daydream/backends/__init__.py:102-122`:

```python
def create_backend(name: str, model: str | None = None) -> Backend:
    """Create a backend by name.

    Args:
        name: Backend name ("claude" or "codex").
        model: Optional model override. Each backend has its own default.

    Returns:
        A Backend instance.

    Raises:
        ValueError: If the backend name is unknown.

    """
    if name == "claude":
        from daydream.backends.claude import ClaudeBackend
        return ClaudeBackend(model=model or "opus")
    ...
```

Apply to atif's `validate()`:
- Google-style docstring with `Args:` / `Returns:` sections (per CONVENTIONS.md "Comments")
- Type-annotated parameters and return type (`-> bool`)
- One-liner body delegating to `TrajectoryValidator().validate(...)` per D-08

The exact body shape from RESEARCH.md L226–232 is:
```python
def validate(trajectory: dict | str | object, *, validate_images: bool = True) -> bool:
    """Validate an ATIF trajectory (dict, JSON string, or path).

    Pure passthrough to a freshly-constructed TrajectoryValidator (D-08).
    Returns True iff the trajectory matches ATIF v1.0–v1.6 schema.
    """
    return TrajectoryValidator().validate(trajectory, validate_images=validate_images)
```

**Convention extension call-out** (per CONTEXT.md "Established Patterns"): CONVENTIONS.md says `__all__` is used "only in `daydream/backends/__init__.py`." Phase 1 introduces `daydream/atif/__init__.py` as a deliberate second exception. The planner should note this in the PR description as a convention-extension, not a violation. The pattern is *already precedented* by the `__all__` lists in `daydream/deep/__init__.py:25-43` and `daydream/prompts/__init__.py:12-17` (both of which CONVENTIONS.md missed) — so the "second exception" framing is conservative.

---

### `daydream/atif/models/**.py` and `daydream/atif/validator.py` (vendored)

**Analog:** Harbor's own files at the same relative path — `src/harbor/models/trajectories/*.py` and `src/harbor/utils/trajectory_validator.py` from the v0.5.0 tag (commit `5795e7638fbe0ee5d7923b6311df2c9f3747dcf0`).

**Pattern:** **Mechanical-only 1:1 copy with two sed substitutions** per D-03:
- `from harbor.models.trajectories` → `from daydream.atif.models`
- `from harbor.utils.trajectory_validator` → `from daydream.atif.validator`

Plus, ONLY for `validator.py`: truncate at the line `def main():` (per D-07). RESEARCH.md L296–336 documents the safe truncation boundary and confirms `argparse`/`sys` imports are inside `def main()` itself (not at module top), so truncation leaves no dangling imports.

**Use the cross-platform Python `re.sub()` recipe from RESEARCH.md L284–293**, NOT raw `sed -i` (RESEARCH.md Pitfall 1: BSD vs GNU `sed -i` portability). Concrete script body:

```python
import pathlib, re
for p in pathlib.Path("daydream/atif").rglob("*.py"):
    text = p.read_text()
    text = re.sub(r"from harbor\.models\.trajectories", "from daydream.atif.models", text)
    text = re.sub(r"from harbor\.utils\.trajectory_validator", "from daydream.atif.validator", text)
    p.write_text(text)
```

**Truncation snippet** (RESEARCH.md L320–335):
```python
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
```

**No daydream-style refactor allowed:** no docstring edits, no `from __future__ import annotations` adds, no import reordering, no removing comments or `# Copyright …` headers. Original Harbor copyright headers stay verbatim per D-02. The ruff `[tool.ruff.lint.per-file-ignores]` stanza on `daydream/atif/**` is what protects vendored code from `make lint`.

---

### `daydream/atif/NOTICE` (provenance + Apache-2.0 attribution)

**Analog:** None in repo (new pattern). Use the verbatim template from RESEARCH.md L461–494 with the literal provenance line:

```
Vendored from harbor-framework/harbor@v0.5.0, commit 5795e7638fbe0ee5d7923b6311df2c9f3747dcf0, on 2026-04-26
```

The template is reproduced inline in RESEARCH.md (lines 461–494) and includes:
- Header line + Apache-2.0 acknowledgement
- Provenance line in the exact `harbor-framework/harbor@<tag>, commit <sha>, on <YYYY-MM-DD>` shape required by D-02
- Rationale paragraph: "Vendored to avoid Harbor's transitive dependency surface" (per CONTEXT.md "Specific Ideas" — does not name `litellm` directly)
- Allowed-transformations list: the two sed renames + the `__main__` strip
- Pointer to `daydream/atif/LICENSE` for the full license text
- Harbor copyright line: `Copyright 2024-2026 The Laude Institute / Harbor Framework contributors` (Open Question 1 in RESEARCH.md — planner should verify against the actual `LICENSE` file's copyright line during execution and substitute if different)
- Daydream attribution paragraph (Apache §4(c) compliance)

---

### `daydream/atif/LICENSE` (Apache-2.0 verbatim)

**Analog:** None in repo (new pattern). **Verbatim copy** from Harbor v0.5.0's top-level `LICENSE` file (11,357 B; verified standard Apache-2.0 boilerplate per RESEARCH.md A1).

**Pattern:** Do NOT type or summarize. Copy directly from the cloned tarball:

```bash
cp /tmp/harbor-v0.5.0/LICENSE daydream/atif/LICENSE
```

Or fetch from raw GitHub:
```bash
curl -fsSL https://raw.githubusercontent.com/harbor-framework/harbor/v0.5.0/LICENSE > daydream/atif/LICENSE
```

Apache-2.0 §4(a) requires verbatim license text retention; summarizing voids attribution.

---

### `pyproject.toml` (MOD — config-only)

**Analog:** Itself — the existing `[project.dependencies]` block at lines 6–16 and `[tool.ruff.lint]` block at lines 41–42.

**Existing `[project.dependencies]` block** (`pyproject.toml:6-16`):
```toml
dependencies = [
    "claude-agent-sdk==0.1.52",
    "anyio>=4.0",
    "rich>=13.0",
    "pyfiglet>=1.0",
    "tree-sitter==0.25.2",
    "tree-sitter-python==0.25.0",
    "tree-sitter-typescript==0.23.2",
    "tree-sitter-go==0.25.0",
    "tree-sitter-rust==0.24.2",
]
```

**Insertion pattern (VEND-03):** Add `"pydantic>=2.11.7",` to this list. The list is not alphabetized today (`claude-agent-sdk` first, then `anyio`, then `rich`...), so a sensible position is between `pyfiglet` and the `tree-sitter` block — logical grouping by "core deps" then "tree-sitter family." The trailing comma must be preserved.

**Existing `[tool.ruff.lint]` block** (`pyproject.toml:41-42`):
```toml
[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

**Append pattern:** Add a new sub-table immediately after the existing `[tool.ruff.lint]` block:
```toml
[tool.ruff.lint.per-file-ignores]
# Vendored from Harbor v0.5.0; see daydream/atif/NOTICE.
# Mechanical-only edit policy (D-03): no reformatting allowed.
"daydream/atif/**" = ["E", "F", "I", "W"]
```

The exact rule codes (`["E", "F", "I", "W"]`) mirror the `select` list from `[tool.ruff.lint]` line 42 — robust if `select` expands later. RESEARCH.md L447–457 confirms this exact stanza shape per the ruff docs.

---

### `tests/test_atif_vendor_smoke.py` (smoke test for VEND-01/02/05 + D-13)

**Analog:** `tests/test_backend_codex.py` — the only existing test file that uses the `FIXTURES_DIR = Path(__file__).parent / "fixtures" / "..."` pattern (verified via `grep -l "FIXTURES_DIR" tests/test_*.py`).

**Module docstring + imports pattern** (`tests/test_backend_codex.py:1-19`):
```python
# tests/test_backend_codex.py
"""Tests for CodexBackend with canned JSONL fixtures."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream.backends import (
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend, CodexError, _unwrap_shell_command

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "codex_jsonl"
```

Apply to atif smoke test:
- File-leading comment line `# tests/test_atif_vendor_smoke.py` (matches existing convention in `test_backend_codex.py:1`)
- Triple-quoted module docstring (multi-line is fine; RESEARCH.md L514–521 has a serviceable template)
- Standard import order: stdlib → third-party → local `daydream.*` (per CONVENTIONS.md "Import Organization")
- Multiple names from one module in a single `from X import (...)` block
- Module-level `GOLDEN_DIR = Path(__file__).parent / "fixtures" / "atif_golden"` constant — direct mirror of the `FIXTURES_DIR` pattern at `test_backend_codex.py:19`
- `UPPER_SNAKE_CASE` for the constant (per CONVENTIONS.md "Naming Patterns")

**Test function naming pattern** (`tests/test_backend_codex.py:51, 77`):
```python
@pytest.mark.asyncio
async def test_simple_text_events():
    ...

@pytest.mark.asyncio
async def test_tool_use_events():
    ...
```

Apply to atif smoke test: `def test_<behavior>():` form. Note the smoke test is **synchronous** (file I/O + Pydantic validate; no async); omit `@pytest.mark.asyncio` and use plain `def test_*` — pytest's `asyncio_mode = "auto"` won't interfere with sync tests.

**Parametrized fixture-discovery pattern** — there is no exact analog in the repo for a `glob`-based parametrize, but the construction mirrors standard pytest. RESEARCH.md L555 has the recommended shape:

```python
def _golden_paths() -> list[Path]:
    return sorted(p for p in GOLDEN_DIR.rglob("*.json") if "_invalid" not in p.parts)


@pytest.mark.parametrize("golden_path", _golden_paths(), ids=lambda p: p.name)
def test_golden_fixtures_validate(golden_path: Path) -> None:
    """VEND-05 + D-09: every Terminus-2 (v1.6) and OpenHands (v1.5) golden validates."""
    assert validate(golden_path) is True
```

The `_golden_paths()` private helper follows the underscore-prefix convention for private helpers (per CONVENTIONS.md "Naming Patterns").

**Full smoke test body** is reproduced in RESEARCH.md L513–578 — copy that template, adjusting only for any planner discretion items (e.g., negative-fixture filename per CONTEXT.md D-13's discretion clause).

---

### `tests/fixtures/atif_golden/terminus2/hello-world-invalid-json.trajectory.json` and `tests/fixtures/atif_golden/openhands/hello-world.trajectory.json` (vendored JSON corpus)

**Analog:** `tests/fixtures/codex_jsonl/*.jsonl` — the existing pattern of vendored upstream test data under `tests/fixtures/<category>/`.

**Existing fixture inventory** (verified via `ls tests/fixtures/codex_jsonl/`):
```
output_text_blocks.jsonl       411 B
simple_text.jsonl              331 B
streamed_structured_output.jsonl 999 B
structured_output.jsonl        406 B
tool_use.jsonl                 919 B
toplevel_text.jsonl            734 B
turn_completed_result.jsonl    510 B
turn_failed.jsonl              138 B
```

**Pattern observations:**
- Vendored fixtures live flat under `tests/fixtures/<category>/` (no nested subdirs in `codex_jsonl/`)
- File names are descriptive of the recorded scenario (`simple_text`, `tool_use`, `turn_failed`)
- Sub-KB files preferred (largest is 999 B); keeps test corpus small

**Apply to atif:** D-12 mandates a deeper layout — `tests/fixtures/atif_golden/<source>/<filename>.json` — because we have two upstream sources to namespace (Terminus-2 and OpenHands). This is a precedented extension of the `<category>/` layout, not a new pattern. Files are larger (7.4 KB and 27.7 KB per RESEARCH.md L9) but still well within "tiny test corpus" territory.

**Vendoring command** — RESEARCH.md L668–672:
```bash
cp /tmp/harbor-v0.5.0/tests/golden/terminus_2/hello-world-invalid-json.trajectory.json \
   tests/fixtures/atif_golden/terminus2/
cp /tmp/harbor-v0.5.0/tests/golden/openhands/hello-world.trajectory.json \
   tests/fixtures/atif_golden/openhands/
```

Note the upstream directory rename (`terminus_2/` → `terminus2/` per D-12) — the only intentional deviation from upstream layout. This is acceptable because the fixture content is still verbatim (only the path changes); the `tests/fixtures/atif_golden/` namespace is daydream-owned.

---

### `tests/fixtures/atif_golden/_invalid/non-sequential-step-id.json` (hand-authored negative fixture)

**Analog:** None in repo (new pattern). Use the verbatim 2-step template from RESEARCH.md L584–605:

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

**Design rationale (RESEARCH.md L608–612):**
- v1.6 `schema_version` (matches the validator's accepted Literal range)
- One user step + one agent step (smallest non-trivial agent run)
- Only the `step_id` field is broken — everything else is shape-valid; isolates the failure mode
- ~440 B keeps test corpus small

**Filename discretion** (per CONTEXT.md "Claude's Discretion"): D-13 suggests `non-sequential-step-id.json`. Other reasonable names: `bad-step-id.json`, `step-id-skipped.json`. Whichever reads cleanest in the parametrized test output — `non-sequential-step-id.json` is the recommended choice because it directly names the invariant being violated (Harbor's validator's "step_id sequentiality" check, RESEARCH.md L582).

## Shared Patterns

### Module Docstring (CONVENTIONS.md "Comments")

**Source:** Every daydream source module
**Apply to:** `daydream/atif/__init__.py`, `tests/test_atif_vendor_smoke.py`

Pattern: triple-quoted docstring as the first statement after any leading file-path comment. Vendored files (per D-03) keep their existing Harbor docstrings unchanged.

Example (`daydream/deep/__init__.py:1-4`):
```python
"""Deep-review mode package.

Exports run_deep once the orchestrator exists (plan 05-09).
"""
```

### Import Organization (CONVENTIONS.md "Import Organization")

**Source:** Every daydream source module — enforced by ruff's `I` rule
**Apply to:** `daydream/atif/__init__.py`, `tests/test_atif_vendor_smoke.py` (NOT the vendored files — they're exempt via `per-file-ignores`)

Pattern (ordered):
1. Standard library imports (`from pathlib import Path`, `import json`)
2. Third-party imports (`import pytest`)
3. Local `daydream.*` imports (`from daydream.atif import ...`)

Multiple names from one module collected into a single `from X import (...)` block. No star imports. No relative imports — full `daydream.<module>` paths only.

Example (`tests/test_backend_codex.py:3-17`):
```python
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream.backends import (
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend, CodexError, _unwrap_shell_command
```

### `__all__` Declaration (Phase 1 deliberate convention extension)

**Source:** `daydream/backends/__init__.py:128-140`, `daydream/deep/__init__.py:25-43`, `daydream/prompts/__init__.py:12-17`
**Apply to:** `daydream/atif/__init__.py` only

Pattern:
- One symbol per line
- Trailing comma on every entry
- Sorted alphabetically within case-groups (constants → classes → callables) — this is the convention used by both `daydream/backends/__init__.py:128-140` and `daydream/deep/__init__.py:25-43`
- Listed as the LAST top-level statement in the file (after all imports + the `validate()` definition)

CONTEXT.md "Established Patterns" notes this is a deliberate convention extension; CONVENTIONS.md L62 says `__all__` is used "only in `daydream/backends/__init__.py`" but in practice three modules already use it. Phase 1's addition is precedented.

### File-I/O Test Fixture Loading

**Source:** `tests/test_backend_codex.py:19`
**Apply to:** `tests/test_atif_vendor_smoke.py`

Pattern:
```python
GOLDEN_DIR = Path(__file__).parent / "fixtures" / "atif_golden"
```

Module-level `UPPER_SNAKE_CASE` constant; computed at import time; tests reference it directly (no fixture function needed for static paths).

### Google-Style Docstring (CONVENTIONS.md "Comments")

**Source:** `daydream/backends/__init__.py:102-115` (`create_backend()`)
**Apply to:** `daydream/atif/__init__.py`'s `validate()` function

Pattern:
```python
def function_name(arg: Type) -> ReturnType:
    """One-line summary.

    Args:
        arg: Description of arg.

    Returns:
        Description of return value.

    Raises:
        ExceptionType: When this happens.

    """
```

The trailing blank line before the closing `"""` is the daydream convention (verified at `daydream/backends/__init__.py:114-115`).

## No Analog Found

| File | Role | Reason | Recommended Source |
|------|------|--------|---------------------|
| `daydream/atif/NOTICE` | Apache-2.0 attribution + provenance | First vendored module in the repo; no precedent for `NOTICE` files | RESEARCH.md L461–494 verbatim template |
| `daydream/atif/LICENSE` | Apache-2.0 verbatim | First vendored module in the repo; no precedent | Verbatim copy from Harbor v0.5.0's `LICENSE` (11,357 B) |
| `tests/fixtures/atif_golden/_invalid/non-sequential-step-id.json` | Hand-authored negative-path JSON fixture | First negative-fixture pattern in the repo | RESEARCH.md L584–605 verbatim template |
| `daydream/atif/models/**.py` | Vendored Pydantic models | Vendored from Harbor; D-03 mechanical-only edit policy means there IS no daydream analog by design | Harbor v0.5.0 `src/harbor/models/trajectories/*.py` |
| `daydream/atif/validator.py` | Vendored programmatic validator | Same as above | Harbor v0.5.0 `src/harbor/utils/trajectory_validator.py` (L1–L225) |

**Planner note:** The "no analog" entries above are NOT gaps to fill with daydream-style code. The vendored files are intentionally outside daydream conventions (D-03); the LICENSE/NOTICE/negative-fixture entries have explicit verbatim templates in RESEARCH.md that the planner should reference rather than re-invent.

## Metadata

**Analog search scope:** `daydream/`, `tests/`, `pyproject.toml`
**Files scanned:** 6 daydream source files + 1 test file + 1 config file (early-stopped at strong matches per agent guidance)
**Analogs read in full:**
- `daydream/backends/__init__.py` (141 lines)
- `daydream/deep/__init__.py` (43 lines)
- `daydream/prompts/__init__.py` (17 lines)
- `tests/test_backend_codex.py` (lines 1–100; full structure captured)
- `pyproject.toml` (47 lines)
- `tests/conftest.py` (first 50 lines)
- `tests/fixtures/codex_jsonl/` (directory listing only — fixture content irrelevant to pattern)

**Pattern extraction date:** 2026-04-26

## PATTERN MAPPING COMPLETE
