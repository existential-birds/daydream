---
phase: 1
slug: exploration-infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-05
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.0+ with pytest-asyncio 0.24+ |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest -x -q` |
| **Full suite command** | `uv run pytest -v` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -x -q`
- **After every plan wave:** Run `uv run pytest -v && uv run ruff check daydream && uv run mypy daydream`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | INFR-01 | unit | `uv run pytest tests/test_backends_init.py -x` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | INFR-02 | unit | `uv run pytest tests/test_exploration.py -x` | ❌ W0 | ⬜ pending |
| 01-01-03 | 01 | 1 | INFR-03 | unit | `uv run pytest tests/test_exploration.py::test_degradation -x` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | AGNT-03 | unit + regression | `uv run pytest tests/test_backends_init.py tests/test_backend_claude.py tests/test_backend_codex.py -x` | ✅ (extend) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_exploration.py` — stubs for INFR-02, INFR-03 (ExplorationContext, degradation)
- [ ] Add `agents` kwarg tests to `tests/test_backends_init.py` — covers AGNT-03
- [ ] Add `agents` kwarg tests to `tests/test_backend_claude.py` — covers AGNT-03 (ClaudeBackend)
- [ ] Add `agents` kwarg tests to `tests/test_backend_codex.py` — covers AGNT-03 (CodexBackend ignores)
- [ ] Add SDK import verification test — covers INFR-01

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| SDK version pinned in pyproject.toml | INFR-01 | Build config, not runtime | Verify `pyproject.toml` contains `claude-agent-sdk==0.1.52` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
