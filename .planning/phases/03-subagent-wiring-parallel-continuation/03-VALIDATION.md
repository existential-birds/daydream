---
phase: 3
slug: subagent-wiring-parallel-continuation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-27
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 9.0.3 + pytest-asyncio 1.3.0 |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest tests/test_trajectory.py -x -v` |
| **Full suite command** | `uv run pytest -v` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/test_trajectory.py -x -v`
- **After every plan wave:** Run `uv run pytest -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | SUBA-01 | — | N/A | unit | `uv run pytest tests/test_trajectory.py::test_sequential_phases_single_file -x` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | SUBA-05 | — | N/A | unit | `uv run pytest tests/test_trajectory.py::test_continuation_appends_no_sibling -x` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 1 | SUBA-06 | — | N/A | unit | `uv run pytest tests/test_trajectory.py::test_sibling_inherits_session_id -x` | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 1 | SUBA-07 | — | N/A | unit | `uv run pytest tests/test_trajectory.py::test_fork_contextvar_isolation -x` | ❌ W0 | ⬜ pending |
| 03-01-05 | 01 | 1 | SUBA-08 | — | N/A | unit | `uv run pytest tests/test_trajectory.py::test_step_id_isolation_across_siblings -x` | ❌ W0 | ⬜ pending |
| 03-01-06 | 01 | 1 | SUBA-09 | — | N/A | unit | `uv run pytest tests/test_trajectory.py::test_parent_metrics_exclude_children -x` | ❌ W0 | ⬜ pending |
| 03-02-01 | 02 | 1 | SUBA-02 | — | N/A | unit | `uv run pytest tests/test_trajectory.py::test_fix_parallel_siblings -x` | ❌ W0 | ⬜ pending |
| 03-02-02 | 02 | 1 | SUBA-03 | — | N/A | unit | `uv run pytest tests/test_trajectory.py::test_deep_per_stack_siblings -x` | ❌ W0 | ⬜ pending |
| 03-02-03 | 02 | 1 | SUBA-04 | — | N/A | unit | `uv run pytest tests/test_trajectory.py::test_exploration_siblings -x` | ❌ W0 | ⬜ pending |
| 03-02-04 | 02 | 1 | SUBA-02 | T-3-01 | _safe_descriptor strips path separators | unit | `uv run pytest tests/test_trajectory.py::test_safe_descriptor -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Fork-related tests in `tests/test_trajectory.py` — stubs for SUBA-02, SUBA-03, SUBA-04, SUBA-06, SUBA-07, SUBA-08, SUBA-09
- [ ] Continuation/sequential verification tests — stubs for SUBA-01, SUBA-05
- [ ] `_safe_descriptor` security test — path traversal prevention for T-3-01

*Existing test infrastructure covers framework/conftest requirements.*

---

## Manual-Only Verifications

*All phase behaviors have automated verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
