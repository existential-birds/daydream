---
phase: 1
slug: vendor-atif-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-26
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.3 + pytest-asyncio 1.3.0 (existing) |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` (`asyncio_mode = "auto"`) |
| **Quick run command** | `uv run pytest -v tests/test_atif_vendor_smoke.py` |
| **Full suite command** | `uv run pytest -v` (343 existing + smoke test) |
| **Estimated runtime** | ~30 seconds (full suite); <1s (smoke test) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -v tests/test_atif_vendor_smoke.py && uv run ruff check daydream/atif && uv run mypy daydream/atif`
- **After every plan wave:** Run `make check` (lint + typecheck + full 343+ tests)
- **Before `/gsd-verify-work`:** `make check` green AND `grep -rn 'from harbor\|^import harbor' daydream/ tests/` returns zero matches
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

> Task IDs (`{N}-{plan}-{task}`) and Threat Refs are filled in by the planner; this table seeds requirement → verification command pairs.

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| TBD | TBD | 0 | (Wave 0) | — | N/A | scaffolding | `test -d daydream/atif/models && test -f daydream/atif/validator.py && test -f daydream/atif/__init__.py && test -f daydream/atif/NOTICE && test -f daydream/atif/LICENSE` | ❌ W0 | ⬜ pending |
| TBD | TBD | 0 | (Wave 0) | — | N/A | scaffolding | `test -f tests/fixtures/atif_golden/terminus2/hello-world-invalid-json.trajectory.json && test -f tests/fixtures/atif_golden/openhands/hello-world.trajectory.json && test -f tests/fixtures/atif_golden/_invalid/non-sequential-step-id.json` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | VEND-01 | — | N/A | unit | `uv run pytest -v tests/test_atif_vendor_smoke.py::test_models_import_cleanly` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | VEND-02 | — | N/A | static | `uv run python -c "import daydream.atif.validator; assert 'harbor' not in daydream.atif.validator.__file__"` AND `! grep -rn 'from harbor\|^import harbor' daydream/atif/` | ❌ W0 | ⬜ pending |
| TBD | TBD | 1 | VEND-03 | — | N/A | integration | `grep -E '\"pydantic>=2\\.11\\.7\"' pyproject.toml && uv sync --check` | ✓ existing | ⬜ pending |
| TBD | TBD | 1 | VEND-04 | — | N/A | static | `! grep -rn 'from harbor\|^import harbor' daydream/ tests/` | ✓ existing | ⬜ pending |
| TBD | TBD | 1 | VEND-05 | — | N/A | integration | `uv run pytest -v tests/test_atif_vendor_smoke.py::test_golden_fixtures_validate` | ❌ W0 | ⬜ pending |
| TBD | TBD | 2 | (Phase guard) | — | N/A | regression | `uv run pytest -v` reports ≥343 passing | ✓ existing | ⬜ pending |
| TBD | TBD | 2 | (Phase guard) | — | N/A | static | `uv run ruff check daydream` | ✓ existing | ⬜ pending |
| TBD | TBD | 2 | (Phase guard) | — | N/A | static | `uv run mypy daydream` | ✓ existing | ⬜ pending |
| TBD | TBD | 2 | (D-13 negative) | — | N/A | unit | `uv run pytest -v tests/test_atif_vendor_smoke.py::test_invalid_fixture_rejected` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_atif_vendor_smoke.py` — covers VEND-01, VEND-02, VEND-05, and D-13 negative path. Uses module-level `GOLDEN_DIR = Path(__file__).parent / "fixtures" / "atif_golden"` mirroring `tests/test_backend_codex.py`'s `FIXTURES_DIR` pattern.
- [ ] `tests/fixtures/atif_golden/terminus2/hello-world-invalid-json.trajectory.json` — Terminus-2 v1.6 corpus (7,405 B from Harbor v0.5.0).
- [ ] `tests/fixtures/atif_golden/openhands/hello-world.trajectory.json` — OpenHands v1.5 corpus (27,697 B from Harbor v0.5.0).
- [ ] `tests/fixtures/atif_golden/_invalid/non-sequential-step-id.json` — hand-authored 2-step trajectory where `steps[1].step_id == 3` instead of `2` (D-13).
- [ ] **No `tests/conftest.py` changes** — Phase 1 must NOT pre-introduce conflicting fixtures. Phase 2's CORE-10 owns the `_reset_trajectory_recorder` autouse fixture.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Apache-2.0 NOTICE attribution wording is human-readable and accurate | VEND-04 (DOCS-05 spirit) | License-conformance review is a human judgment call; automated check only confirms file presence | Reviewer reads `daydream/atif/NOTICE` and confirms: (1) provenance line matches `Vendored from harbor-framework/harbor@<tag>, commit <sha>, on <YYYY-MM-DD>`, (2) Apache-2.0 attribution is present, (3) rationale ("vendored to avoid transitive dependency surface") appears. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (smoke test + 3 fixtures)
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
