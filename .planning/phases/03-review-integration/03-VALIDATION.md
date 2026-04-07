---
phase: 3
slug: review-integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-07
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Source: `03-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0+ with pytest-asyncio 0.24+ |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`, `asyncio_mode = "auto"`) |
| **Quick run command** | `uv run pytest tests/test_phases.py -x` |
| **Full suite command** | `uv run pytest -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** `uv run pytest tests/test_phases.py -x`
- **After every plan wave:** `uv run pytest -v`
- **Before `/gsd:verify-work`:** Full suite green + `make check`
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Req ID | Behavior | Test Type | Automated Command | File Exists |
|--------|----------|-----------|-------------------|-------------|
| QUAL-02 | `FEEDBACK_SCHEMA` requires `confidence` enum + `rationale` | unit | `uv run pytest tests/test_phases.py::test_feedback_schema_requires_confidence_and_rationale -x` | ❌ W0 |
| QUAL-02 | `ALTERNATIVE_REVIEW_SCHEMA` requires `confidence` + `rationale` | unit | `uv run pytest tests/test_phases.py::test_alternative_review_schema_requires_confidence_and_rationale -x` | ❌ W0 |
| QUAL-02 | Parser rejects responses missing `confidence`/`rationale` | unit | `uv run pytest tests/test_phases.py::test_parse_feedback_rejects_unlabeled -x` | ❌ W0 |
| QUAL-01 | `phase_review` prompt mentions Dependency Impact | unit | `uv run pytest tests/test_phases.py::test_review_prompt_includes_dependency_impact -x` | ❌ W0 |
| QUAL-03 | Review prompt distinguishes filter-fix vs flag-reviewed convention cases | unit | `uv run pytest tests/test_phases.py::test_review_prompt_distinguishes_convention_cases -x` | ❌ W0 |
| OUTP-01 | `PLAN_SCHEMA` requires `references[]` of `{file, symbol}` | unit | `uv run pytest tests/test_phases.py::test_plan_schema_requires_references -x` | ❌ W0 |
| OUTP-01 | Plan prompt forbids fabricated references | unit | `uv run pytest tests/test_phases.py::test_plan_prompt_forbids_fabrication -x` | ❌ W0 |
| OUTP-02 | All four `_build_*_prompt` helpers prepend `to_prompt_section()` | unit | `uv run pytest tests/test_phases.py::test_all_phase_builders_inject_exploration -x` | ⚠️ extend |
| OUTP-02 | All four `_build_*_prompt` helpers share confidence/convention instructions | unit | `uv run pytest tests/test_phases.py::test_all_phase_builders_use_shared_instructions -x` | ❌ W0 |
| OUTP-01 | TTT plan renderer dims ungrounded steps | unit | `uv run pytest tests/test_ui.py::test_plan_renderer_dims_ungrounded_steps -x` | ❌ W0 |
| OUTP-02 | E2E: labeled issues for both `--ttt` and normal flows via MockBackend | integration | `uv run pytest tests/test_integration.py::test_exploration_enriched_output_both_flows -x` | ❌ W0 |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_phases.py` — schema + prompt unit tests (extend existing file)
- [ ] `tests/test_ui.py` — plan renderer ungrounded-step test (extend existing file)
- [ ] `tests/test_integration.py` — exploration-enriched both-flows test (extend existing file)
- [ ] Shared `ExplorationContext` fixture in `tests/conftest.py`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real-LLM confidence labels are calibrated (not all HIGH) | QUAL-02 | LLM-behavior dependent; no deterministic check | Run `daydream <repo>` against a sample PR; visually inspect that confidence labels vary with grounding |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
