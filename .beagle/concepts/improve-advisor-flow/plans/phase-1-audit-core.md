# Improve Advisor Flow — Phase 1 (Audit → Plans Core) Implementation Plan

> **Source spec:** `.beagle/concepts/improve-advisor-flow/spec.md` (Phase 1 requirements, lines 18–30)
> **Source behavior:** `/Users/ka/github/improve` — `skills/improve/SKILL.md`, `references/audit-playbook.md`, `references/plan-template.md`. Where the spec is silent, the source repo is the reference.
> **For downstream agents:** Execute task-by-task. Each task uses `- [ ]` checkboxes for tracking. Do not skip the test-first steps — they catch wiring bugs that pure-logic tests miss. The verification gate for **every** task is `make check` (Step 4 of each task names the narrower commands to run first).

**Goal:** `daydream improve <path>` audits a repository as a read-only senior advisor — recon, category audits, vetting, prioritization — and writes self-contained implementation plans to `daydream_plans/`, with every agent interaction recorded as an ATIF trajectory.

**Architecture:** A new `improve` flow family registered as a peer of deep/shallow/review/pr-feedback through the same `Registry`/`run_flow` machinery (`daydream/flows/engine.py:106-124`, `daydream/extensions/builtins.py:56-75`). Step bodies live in a new `daydream/improve/` package mirroring `daydream/deep/`. All agent work goes through `run_agent()` with `read_only=True` (`daydream/agent.py:348`); every write to `daydream_plans/` is performed host-side from structured agent output, so the audited tree is never touched by an agent tool. Audit fan-out reuses the CapacityLimiter + task-group + `maybe_fork` pattern from `daydream/phases.py:2714-2857`. Findings extend the existing `ArtifactFinding` entry (`daydream/findings.py:90-114`) with advisory axes rather than forking a second findings model.

**Tech Stack:** Python 3.12, anyio, existing `Backend`/`AgentEvent` seam (claude/codex/pi), pydantic-validated ATIF v1.7 recorder, jsonschema, pytest.

---

## Assumptions (confirm or correct before/while executing)

1. **Leverage formula.** The playbook says "leverage = impact ÷ effort, discounted by confidence and fix-risk" (`audit-playbook.md:125`) but gives no numbers. This plan pins: impact HIGH=3 / MED=2 / LOW=1; effort S=1 / M=2 / L=3; confidence multiplier HIGH=1.0 / MED=0.7 / LOW=0.4; risk multiplier LOW=1.0 / MED=0.8 / HIGH=0.6; `leverage = (impact / effort) * confidence_mult * risk_mult`. Tiebreakers per `audit-playbook.md:126-130`.
2. **Non-interactive default** = the top `min(5, len(vetted defect findings))` by leverage (the spec's "top 3–5"); direction findings are never auto-planned. The default is recorded in `daydream_plans/README.md`.
3. **Cross-run persistence lives in `daydream_plans/`** (`rejected.json` + `README.md` index): `.daydream/` run artifacts are per-run/archived, and the spec's only durable write surface is `daydream_plans/` (spec line 20).
4. **Agents never write; the host writes.** All improve-phase `run_agent` calls pass `read_only=True`; plan files, index, and rejections are rendered and written by daydream Python code from structured output. This is the strongest honoring of the trust constraint (spec line 87) and builds on the existing per-backend read-only enforcement (`daydream/backends/claude.py:196-214`, `codex.py:115`, `pi.py:478`).
5. **Sub-verbs nest under `improve`:** `daydream improve plan "<description>"` and `daydream improve review-plan <file>`. `review-plan` only accepts files under `daydream_plans/` (the only tree daydream may rewrite).
6. **Enum additions are additive:** `DaydreamRunFlow.IMPROVE` and `DaydreamPhase.RECON/AUDIT/VET/PLAN_WRITE` extend the ATIF `Step.extra` literal sets (`daydream/trajectory.py:162-192`); no schema break, no `EXTENSION_API_VERSION` bump (additive flows/slots/prompts don't bump — `docs/extensions.md:118-119`).
7. **Effort tier numbers** are ported from the source skill's table (`SKILL.md:52-61`): quick = {correctness, security, tests}, concurrency 1, HIGH-confidence only, top ~6; standard = all nine categories, concurrency 4; deep = all nine, concurrency 8, LOW-confidence "investigate" items included.
8. **Audit slot naming:** `audit:<category>:<stack>` (stack-specific) and `audit:<category>` (stack-agnostic fallback), resolved through the registry like `stack:<key>` slots today (`daydream/extensions/builtins.py:23-27`).

## Patterns

**P1 — Improve real-path test harness.** Real-path tests enter through `runner.run` with `RunConfig(flow_name="improve", ...)` against a real temp git repo, patching only `daydream.runner.create_backend` to return an `_ImproveStubBackend` that dispatches on prompt markers and **records the `read_only` kwarg** of every `execute()` call. Model it on `_StubBackend` / `_install_stub_backend` in `tests/test_deep_orchestrator.py:29-654`; assert observable outcomes (exit code, files under `daydream_plans/` and `.daydream/improve/`, trajectory content), never "function was called". The stub and its installer live in `tests/test_improve_flow.py` and are shared by later tasks. New fixture `improve_monorepo_target` (Task 3, `tests/conftest.py`) is the standard repo: `apps/billing/` + `apps/catalog/` (python), `web/` (tsx), root `README.md`, committed on `main`.

**P2 — Registered-surface doc drift.** Any task that registers a new flow, step, skill slot, or prompt MUST update the matching inventory table in `docs/extensions.md` (`docs/extensions.md:204-331`) in the same task — `tests/test_extension_contract_doc.py::test_contract_doc_names_every_registered_surface` pins the doc to the registry and will fail otherwise.

## File Structure

**Create**

| Path | Responsibility |
|------|----------------|
| `daydream/improve/__init__.py` | Package marker |
| `daydream/improve/artifacts.py` | `.daydream/improve/` path helpers (mirror `daydream/deep/artifacts.py`) |
| `daydream/improve/services.py` | Service/package enumeration: config-declared roots + Task-0 heuristics |
| `daydream/improve/prompts.py` | Audit/vet/plan-writer prompt builders + output JSON schemas (playbook-derived) |
| `daydream/improve/prioritize.py` | Leverage scoring, ordering, defect/direction partition, cross-service aggregation |
| `daydream/improve/plans.py` | Plan-file + index + `rejected.json` rendering, numbering, reconcile |
| `daydream/improve/orchestrator.py` | `STEPS` tuple + step bodies (recon, audit, vet, prioritize, select, write-plans, report) |
| `tests/test_improve_services.py` | Unit: enumeration + scoping |
| `tests/test_improve_prioritize.py` | Unit: leverage, ordering, partition, aggregation |
| `tests/test_improve_plans.py` | Unit: plan/index/rejected rendering + reconcile |
| `tests/test_improve_flow.py` | Real-path: stub backend + full-pipeline tests (Pattern P1) |

**Modify**

| Path | Change |
|------|--------|
| `daydream/findings.py` | Shared `FINDING_ENTRY_SCHEMA` + optional advisory fields on `ArtifactFinding` |
| `daydream/config.py` | `AUDIT_CATEGORIES`, `AUDIT_SKILL_MAP`, `EFFORT_TIERS`, phase-model/effort table rows |
| `daydream/config_file.py` | `[tool.daydream.improve]` sub-table (service roots/groups) |
| `daydream/trajectory.py` | `DaydreamRunFlow.IMPROVE`, `DaydreamPhase.RECON/AUDIT/VET/PLAN_WRITE` |
| `daydream/exploration_runner.py` | Repo-scoped `repo_scan()` entry beside diff-scoped `pre_scan()` |
| `daydream/extensions/builtins.py` | Seed audit skill slots, improve prompts, `improve` flow |
| `daydream/runner.py` | `RunConfig` improve fields, `_run_improve` preamble, dispatch branch, `skip_tests` |
| `daydream/cli.py` | `improve` verb: `KNOWN_VERBS`, parser, sub-verbs `plan` / `review-plan` |
| `docs/extensions.md` | Flow/step, slot, prompt inventory rows (Pattern P2) |
| `tests/conftest.py` | `improve_monorepo_target` fixture |
| `tests/test_findings.py` | Schema-extension round-trip tests |
| `tests/test_cli_verbs.py` | `improve` verb dispatch tests |
| `CLAUDE.md`, `README.md` | Command surface docs (final task) |

---

### Task 0: Spike — service-enumeration heuristics beyond declared config

The spec's open question (spec lines 150-152) as an explicit investigation, not an assumption. Config-declared roots are the guaranteed path; this spike bounds which layout heuristics Task 3 implements.

**Files:**
- Create: `.beagle/concepts/improve-advisor-flow/plans/notes/service-heuristics.md` (findings memo — planning artifact, not shipped code)

- [ ] **Step 1: Enumerate candidate heuristics and probe real repos**

Candidate signals to evaluate, each a cheap filesystem/manifest read: `package.json` `workspaces`, `pnpm-workspace.yaml` `packages`, Cargo workspace `members`, `go.work` `use`, `uv`/`poetry` workspace tables in root `pyproject.toml`, `docker-compose.yml` `services` with `build:` contexts, and the generic "directories one level below a conventional root (`apps/`, `services/`, `packages/`, `crates/`, `cmd/`) that contain their own manifest (`pyproject.toml`/`package.json`/`go.mod`/`Cargo.toml`/`mix.exs`)".

Run each candidate against at least: this repo (single-package, must yield 0 services / "not a monorepo"), `/Users/ka/github/improve` (skill repo, must yield 0), and two constructed fixtures — an `apps/<service>` FastAPI-style layout (the customer validation case, spec line 148) and a mixed `packages/` + root-app layout. Capture what each signal returns.

**Verify:** memo lists, per signal, the probe command/snippet and its actual output per repo.

- [ ] **Step 2: Decide the Phase-1 heuristic set**

Keep only signals that (a) require no third-party parser beyond stdlib/tomllib, and (b) produced zero false positives on the single-package repos. Expected outcome: manifest-workspace signals (package.json workspaces, pnpm-workspace globs, cargo members, go.work) + the manifest-under-conventional-root scan. Record rejected signals with one-line reasons.

- [ ] **Step 3: Route the outcome**

If the surviving set is empty or unreliable, Phase 1 ships config-declared roots only (still spec-compliant — config is the guaranteed path) and this is reported back to the user as a spec-level note before Task 3 executes. Otherwise the memo's surviving list IS the heuristic inventory Task 3 implements.

No commit (planning artifact only; commit it with Task 3 if the executor prefers versioned notes).

---

### Task 1: Extend the finding entry with advisory axes

**Files:**
- Modify: `daydream/findings.py`
- Test: `tests/test_findings.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_finding_entry_accepts_advisory_fields_and_round_trips(tmp_path: Path) -> None:
    entry = _finding(impact="HIGH", effort="S", risk="LOW", leverage=3.0,
                     category="correctness", services=["billing"], provenance=None)
    artifact = {**_valid_artifact_envelope(), "findings": [entry]}
    jsonschema.validate(artifact, FINDINGS_SCHEMA)          # extended schema accepts them
    loaded = ArtifactFinding(**entry)
    assert (loaded.impact, loaded.effort, loaded.risk) == ("HIGH", "S", "LOW")

def test_pr_artifact_without_advisory_fields_still_validates() -> None:
    # Regression pin: the existing PR two-phase artifact (no advisory keys)
    # must remain schema-valid and load into ArtifactFinding unchanged.
    jsonschema.validate(_valid_artifact_envelope(), FINDINGS_SCHEMA)
```

Reuse the existing artifact-builder helpers already in `tests/test_findings.py`; add `_finding(**overrides)` only if no equivalent helper exists there (grep first).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_findings.py -x -k advisory`
Expected: FAIL — `jsonschema.ValidationError` (`additionalProperties: False` rejects `impact`).

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/findings.py`

**Behavior contract:**
- Extract the finding-entry object schema (`daydream/findings.py:56-79`) into a module-level `FINDING_ENTRY_SCHEMA` referenced by `FINDINGS_SCHEMA["properties"]["findings"]["items"]`, and add **optional** properties: `impact` (`["string","null"]` enum HIGH/MED/LOW or null), `effort` (S/M/L or null), `risk` (LOW/MED/HIGH or null), `leverage` (`["number","null"]`), `category` (string/null), `services` (array of strings, default absent), `provenance` (`introduced`/`inherited`/null). None are added to `required` — existing PR artifacts stay byte-valid.
- `ArtifactFinding` (`daydream/findings.py:90-114`) gains matching fields, all defaulted (`None` / `field(default_factory=list)` for `services`), so `ArtifactFinding(**f)` in `load_findings_artifact` (`findings.py:252`) keeps working for old artifacts.
- `_finding_dict` (`findings.py:136-148`) is unchanged in behavior for the PR path — it must not start emitting the new keys (Phase A artifact output stays byte-identical).
- `FINDINGS_SCHEMA_VERSION` stays `1`: additive optional properties, and the strict-schema loader's equality checks (`findings.py:236-245`) are untouched.

**Reference:** `daydream/findings.py:43-114` — the schema/dataclass pair to extend; keep the docstring inventory in the module header current.

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_findings.py -x` → PASS.
Then: `uv run pytest tests/test_findings.py tests/test_post_findings_integration.py` → PASS (the privileged-poster path proves old artifacts still load). Then `make check` → green.

- [ ] **Step 5: Sweep**

Remove any now-duplicated inline schema fragment and update the module docstring's `Exports:` list to name `FINDING_ENTRY_SCHEMA`.

- [ ] **Step 6: Commit**

```bash
git add daydream/findings.py tests/test_findings.py
git commit -m "feat(findings): add optional advisory axes to the finding entry schema"
```

---

### Task 2: Improve constants and config-file surface

**Files:**
- Modify: `daydream/config.py`, `daydream/config_file.py`
- Test: `tests/test_config_file.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_improve_config_table_parses_service_roots(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream.improve]\nservice_roots = ["apps/*"]\n'
        '[tool.daydream.improve.service_groups]\ncore = ["apps/billing", "apps/catalog"]\n'
    )
    cfg = load_file_config(tmp_path)
    assert cfg.improve_service_roots == ["apps/*"]
    assert cfg.improve_service_groups == {"core": ["apps/billing", "apps/catalog"]}

def test_improve_config_absent_defaults_empty(tmp_path: Path) -> None:
    assert load_file_config(tmp_path).improve_service_roots == []
```

In `tests/test_config.py`: assert `set(AUDIT_CATEGORIES) == {"correctness", "security", "performance", "tests", "tech-debt", "dependencies", "dx", "docs", "direction"}`, `EFFORT_TIERS["quick"].categories == ("correctness", "security", "tests")`, and that every `AUDIT_SKILL_MAP` value is a `plugin:skill`-shaped string.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config_file.py -x -k improve`
Expected: FAIL — `AttributeError: ... 'DaydreamFileConfig' object has no attribute 'improve_service_roots'`.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/config.py`, `daydream/config_file.py`

**Behavior contract:**
- `config.py` gains: `AUDIT_CATEGORIES: tuple[str, ...]` (the nine playbook categories, `audit-playbook.md:9-104`); a frozen `EffortTier` dataclass + `EFFORT_TIERS: dict[str, EffortTier]` with fields `categories` (tuple or `None` = all), `max_concurrency` (1/4/8), `high_confidence_only` (True/False/False), `max_findings` (6/None/None), `include_investigate` (False/False/True) — values per Assumption 7; and `AUDIT_SKILL_MAP: dict[str, dict[str, str]]` keyed `category → stack → skill`, seeded with what beagle ships today: `correctness` → the six `REVIEW_SKILLS` values (`config.py:178-185`), `security`/`performance` → `{"elixir": "beagle-elixir:elixir-security-review" / "beagle-elixir:elixir-performance-review"}`, `tests` → `{"python": "beagle-python:pytest-code-review", "go": "beagle-go:go-testing-code-review", "rust": "beagle-rust:rust-testing-code-review", "elixir": "beagle-elixir:exunit-code-review"}`, `tech-debt` → `{"*": "beagle-core:review-structure"}` (stack-agnostic, `"*"` key). Gap categories (`dependencies`, `dx`, `docs`, `direction`) have empty dicts — playbook prompts fill them (spec line 128-130).
- `PHASE_DEFAULT_MODELS` (`config.py:98-131`) gains rows for both backends: `recon`/`audit` on the mid tier (`claude-sonnet-5` / `gpt-5.6-terra`), `vet`/`plan_write` on the heavy tier (`claude-opus-4-8` / `gpt-5.6-sol`) — advisor economics: the expensive model judges and writes plans (`SKILL.md:14`). `PHASE_DEFAULT_EFFORT["codex"]` gains `recon: "low"`, `audit: "high"`, `vet: "xhigh"`, `plan_write: "high"` (mirroring the arbiter rationale at `config.py:140-146`).
- `DaydreamFileConfig` gains `improve_service_roots: list[str]` and `improve_service_groups: dict[str, list[str]]` (both default-empty), parsed from `merged["improve"]` in `load_file_config` (`config_file.py:222-269`) with the module's junk-degrades-to-default convention (`_coerce_string_list` reuse; a non-dict `improve` table yields defaults). `_merge_section` (`config_file.py:133-170`) merges the `improve` sub-table per-key like `phases`.

**Reference:** `daydream/config_file.py:28-91` (dataclass + accessor shape), `daydream/config.py:98-163` (table conventions and tiering comments — extend the comment blocks, don't fork them).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_config_file.py tests/test_config.py -x` → PASS.
Then: `make check` → green (proves `mypy` accepts the new frozen dataclass and no existing table consumer broke).

- [ ] **Step 5: Sweep**

Update the `config.py` module docstring `Exports:` inventory and the `DaydreamFileConfig` docstring attribute list for the new keys.

- [ ] **Step 6: Commit**

```bash
git add daydream/config.py daydream/config_file.py tests/test_config.py tests/test_config_file.py
git commit -m "feat(config): audit categories, effort tiers, audit skill map, [tool.daydream.improve]"
```

---

### Task 3: Service enumeration

**Files:**
- Create: `daydream/improve/__init__.py`, `daydream/improve/services.py`
- Modify: `tests/conftest.py` (add `improve_monorepo_target` fixture)
- Test: `tests/test_improve_services.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_config_declared_roots_win(tmp_path_repo: Path) -> None:
    cfg = DaydreamFileConfig(improve_service_roots=["apps/*"])
    services = enumerate_services(tmp_path_repo, cfg)
    assert [s.name for s in services] == ["billing", "catalog"]
    assert services[0].root == Path("apps/billing")

def test_heuristics_detect_manifest_under_conventional_root(tmp_path_repo: Path) -> None:
    # no config: apps/billing + apps/catalog each hold their own pyproject.toml
    services = enumerate_services(tmp_path_repo, DaydreamFileConfig())
    assert {s.name for s in services} == {"billing", "catalog"}

def test_single_package_repo_yields_no_services(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='solo'\n")
    assert enumerate_services(tmp_path, DaydreamFileConfig()) == []

def test_scope_filters_search_not_read(tmp_path_repo: Path) -> None:
    services = enumerate_services(tmp_path_repo, DaydreamFileConfig())
    scoped = filter_scope(services, "apps/billing")
    assert [s.name for s in scoped] == ["billing"]
```

`tmp_path_repo` is a local helper building the `apps/<service>` layout (the customer validation case). Also add the shared `improve_monorepo_target` git fixture to `tests/conftest.py`, modeled on `multi_stack_target` (`tests/conftest.py:148-213`): `apps/billing/{pyproject.toml,api.py}`, `apps/catalog/{pyproject.toml,api.py}`, `web/App.tsx`, root `README.md` + `pyproject.toml`, one commit on `main` — improve audits the checked-out state, so no feature branch is needed.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_services.py -x`
Expected: FAIL — `ModuleNotFoundError: No module named 'daydream.improve'`.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/improve/services.py`, `daydream/improve/__init__.py`

**Behavior contract:**
- Frozen `Service` dataclass: `name` (last path segment, deduped by suffixing the parent dir on collision), `root` (repo-relative `Path`), `source` (`"config"` | `"heuristic:<signal>"`).
- `enumerate_services(repo_root: Path, file_config: DaydreamFileConfig) -> list[Service]`: config-declared `improve_service_roots` globs are resolved first and win wholesale (heuristics never add to an explicit declaration — the operator's config is authoritative, spec line 30); with no declaration, apply exactly the Task-0 surviving heuristic set. Deterministic ordering (sorted by root). Zero services is a normal result meaning "single-package repo".
- `filter_scope(services, scope: str) -> list[Service]`: `scope` is a service name, a root path, or a glob matched against `Service.root`; unknown scope raises `ValueError` naming the known services (the caller turns it into an actionable CLI error).
- Filesystem reads only — no `git` calls, no subprocess; malformed manifests degrade to "signal absent" (mirror `load_toml_or_empty`, `config_file.py:94-114`).

**Reference:** the Task-0 memo (heuristic inventory) and `daydream/config_file.py:94-114` (tolerant-parse convention).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_services.py -x` → PASS.
Then: `make check` → green.

- [ ] **Step 5: Sweep** — new files only; nothing to sweep.

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/__init__.py daydream/improve/services.py tests/test_improve_services.py tests/conftest.py
git commit -m "feat(improve): service enumeration from config roots plus layout heuristics"
```

---

### Task 4: Prioritization — leverage, ordering, partition

**Files:**
- Create: `daydream/improve/prioritize.py`
- Test: `tests/test_improve_prioritize.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_leverage_orders_impact_over_effort_discounted() -> None:
    hi = _f(impact="HIGH", effort="S", confidence="HIGH", risk="LOW")   # 3.0
    lo = _f(impact="HIGH", effort="L", confidence="LOW", risk="HIGH")   # 0.24
    assert leverage_score(hi) == pytest.approx(3.0)
    assert [f is hi for f in order_by_leverage([lo, hi])][0]

def test_high_confidence_security_floats_above_equal_leverage() -> None:
    sec = _f(category="security", confidence="HIGH", impact="MED", effort="M")
    bug = _f(category="correctness", confidence="HIGH", impact="MED", effort="M")
    assert order_by_leverage([bug, sec])[0] is sec

def test_direction_findings_partition_separately() -> None:
    defects, direction = partition_direction([_f(category="direction"), _f(category="correctness")])
    assert [f["category"] for f in direction] == ["direction"]
```

`_f(**overrides)` builds a minimal finding dict (the Task-1 entry shape). One more test pins ordering stability: equal-leverage same-category findings keep input order.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_prioritize.py -x`
Expected: FAIL — `ImportError: cannot import name 'leverage_score'`.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/improve/prioritize.py`

**Behavior contract:**
- `leverage_score(finding: dict) -> float` implements Assumption 1 exactly; a missing/unknown axis value takes the *worst* multiplier for that axis (a finding that failed to declare confidence must not outrank one that did) — never a KeyError.
- `order_by_leverage(findings) -> list[dict]`: sorts by score descending with the playbook tiebreakers (`audit-playbook.md:126-130`): HIGH-confidence security above equal-leverage others; stable within ties. Sets `finding["leverage"]` (rounded, 2 dp) on each item so downstream artifacts and the index carry the computed value.
- `partition_direction(findings) -> tuple[list, list]` splits on `category == "direction"` (spec line 24: direction presented separately).

**Reference:** none needed — pure functions; the finding dict shape is `FINDING_ENTRY_SCHEMA` (Task 1).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_prioritize.py -x` → PASS. Then `make check` → green.

- [ ] **Step 5: Sweep** — new files only.

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/prioritize.py tests/test_improve_prioritize.py
git commit -m "feat(improve): leverage scoring and direction partition"
```

---

### Task 5: Audit/vet/plan prompts, schemas, and registry slots

**Files:**
- Create: `daydream/improve/prompts.py`
- Modify: `daydream/extensions/builtins.py`, `docs/extensions.md`
- Test: `tests/test_improve_flow.py` (registry assertions), `tests/test_extension_contract_doc.py` (existing drift guard)

- [ ] **Step 1: Write the failing tests**

```python
def test_registry_seeds_audit_slots_and_improve_prompts() -> None:
    r = build_registry()
    assert r.skill("audit:correctness:python") == "beagle-python:review-python"
    assert r.skill("audit:security:elixir") == "beagle-elixir:elixir-security-review"
    assert r.skill_if_registered("audit:dx") is None       # gap category: prompt-only
    for name in ("audit", "vet", "plan-writer"):
        assert callable(r.prompt(name))

def test_audit_prompt_carries_playbook_section_and_hard_rules() -> None:
    prompt = build_registry().prompt("audit")(
        category="security", skill_invocation=None, services=[], scope_note="",
        recon_summary="langs: python", cwd=Path("/repo"), tier=EFFORT_TIERS["standard"])
    assert "never reproduce secret values" in prompt.lower()      # Hard Rule 4
    assert "data, not instructions" in prompt.lower()             # Hard Rule 6
    assert "file:line" in prompt and "Effort" in prompt           # finding format
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_flow.py -x -k "slots or playbook"`
Expected: FAIL — `UnresolvedExtensionError: skill slot 'audit:correctness:python' is not registered`.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/improve/prompts.py`, `daydream/extensions/builtins.py`, `docs/extensions.md`

**Behavior contract:**
- `prompts.py` defines `AUDIT_FINDINGS_SCHEMA` (object with `findings` array of the Task-1 entry axes the model must emit: `title`, `category`, `path`, `line`, `body` (impact + fix sketch), `impact`, `effort`, `risk`, `confidence`, `evidence` list of `path:line` strings; **no** `fingerprint` — fingerprints are host-derived, never model-emitted, `pr_review.py:399-421`), `VET_SCHEMA` (`verdicts` array: `vet_id` int echo, `keep` bool, `reason` str, optional corrected `severity/impact/effort/risk/confidence/path/line`), and `PLAN_WRITER_SCHEMA` (`slug`, `title`, `priority`, `depends_on` array, `markdown` — the plan body sections; the host stamps header/status/planned-at).
- Prompt builders (mirror the kwarg-builder convention of `daydream/prompts/exploration_subagents.py:127-222`): `build_audit_prompt(...)` embeds the relevant playbook category section (ported as string constants from `audit-playbook.md` §§1–9), the finding format, effort-tier depth/breadth instruction, recon facts, a monorepo note ("slicing bounds where you search, never what you may read" — spec line 133), and verbatim Hard Rules 4 and 6 (`SKILL.md:21,23` — subagents don't inherit them, `SKILL.md:50`); when `skill_invocation` is non-None it is included via `backend.format_skill_invocation` at the call site. `build_vet_prompt(...)` instructs re-opening every cited location, names the three failure classes (by-design, mis-attributed evidence, duplicates — `SKILL.md:68`), defaults to reject-with-reason when unconfirmable, and includes the `beagle-core:review-verification-protocol` invocation string (Should-Have, spec line 75). `build_plan_writer_prompt(...)` embeds the plan template section contract (from `plan-template.md:17-156`) and the zero-context-executor bar (`plan-template.md:3-9`, `SKILL.md:96-104`).
- `builtins.py`: a new `_register_improve_builtins(registry)` (called from `register_builtins`, `builtins.py:19-30`) seeds `audit:<category>:<stack>` for every `AUDIT_SKILL_MAP` entry (a `"*"` stack key seeds the stack-agnostic `audit:<category>` slot) and registers prompts `audit`, `vet`, `plan-writer` via `override_prompt`.
- `docs/extensions.md`: add the new slots to the Skill slots table (`docs/extensions.md:277-295`) and the three prompts + kwargs to the Prompts table (`docs/extensions.md:296-316`) — Pattern P2.

**Reference:** `daydream/extensions/builtins.py:19-54` (seeding shape), `daydream/prompts/exploration_subagents.py:29-125` (schema-constant + `_schema_block` convention).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_flow.py tests/test_extension_contract_doc.py -x` → PASS.
Then: `uv run pytest tests/test_extensions_registry.py tests/test_ext_validate_cli.py` → PASS (`daydream ext validate` resolve-checks the new slots for free). Then `make check` → green.

- [ ] **Step 5: Sweep** — update the `builtins.py` module docstring ("all four flow definitions" → five, in Task 7 if not here).

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/prompts.py daydream/extensions/builtins.py docs/extensions.md tests/test_improve_flow.py
git commit -m "feat(improve): playbook prompts, output schemas, and audit skill slots"
```

---

### Task 6: Repo-scoped exploration entry

**Files:**
- Modify: `daydream/exploration_runner.py`
- Test: `tests/test_improve_flow.py` (unit-level, stub backend direct)

- [ ] **Step 1: Write the failing test**

```python
async def test_repo_scan_seeds_specialists_from_tracked_files(tmp_git_repo: Path) -> None:
    stub = _ImproveStubBackend(tmp_git_repo)   # answers pattern-scanner prompt
    ctx = await repo_scan(stub, tmp_git_repo, max_files=500)
    assert any(c.name == "OpenAPI First" for c in ctx.conventions)
    prompt = stub.calls[0]["prompt"]
    assert "api.py" in prompt                  # seeded from ls-files, not a diff
    assert stub.calls[0]["read_only"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_flow.py -x -k repo_scan`
Expected: FAIL — `ImportError: cannot import name 'repo_scan'`.

- [ ] **Step 3: Implement against the test**

**Files touched:** `daydream/exploration_runner.py`

**Behavior contract:**
- `async def repo_scan(backend, repo_root, *, max_files=500) -> ExplorationContext`: seeds `FileInfo` entries from `git ls-files` (via `daydream.git_ops`, capped at `max_files`, roles `"modified"`-equivalent `"tracked"`), then runs the **pattern-scanner** specialist only (conventions/guidelines are the repo-scoped signal recon needs; dependency-tracer and test-mapper stay diff-scoped) with the existing timeout/turn caps (`exploration_runner.py:49-53`), `read_only=True` on its `run_agent` call, under `maybe_fork(recorder, "explore-pattern_scanner")`.
- Degrades exactly like `pre_scan`: specialist failure or empty results returns the static context; never raises (D-08 convention, `exploration_runner.py:244-246`).
- `pre_scan` (`exploration_runner.py:180-285`) behavior is untouched — shared pieces (`_parse_envelope`, coercers) are reused, not copied.

**Reference:** `daydream/exploration_runner.py:180-285` — mirror `_run_specialist` and the merge; the delta is the file seed (ls-files vs diff) and the single-specialist tier.

- [ ] **Step 4: Run the new test AND the suite**

Run: `uv run pytest tests/test_improve_flow.py -x -k repo_scan` → PASS.
Then: `uv run pytest tests/test_exploration_runner.py` (existing diff-scoped coverage; exact filename may differ — run the module's tests) → PASS. Then `make check` → green.

- [ ] **Step 5: Sweep** — update the module docstring (now two entries: diff-scoped `pre_scan`, repo-scoped `repo_scan`) and `__all__` (`exploration_runner.py:288-294`).

- [ ] **Step 6: Commit**

```bash
git add daydream/exploration_runner.py tests/test_improve_flow.py
git commit -m "feat(exploration): repo-scoped repo_scan entry for advisory recon"
```

---

### Task 7: Flow skeleton — recon step, runner preamble, CLI verb, trajectory enums

The first end-to-end slice: `daydream improve <path>` runs recon and a report, through the real registry/flow/recorder machinery. Later tasks insert steps between recon and report.

**Files:**
- Create: `daydream/improve/artifacts.py`, `daydream/improve/orchestrator.py`
- Modify: `daydream/trajectory.py`, `daydream/extensions/builtins.py`, `daydream/runner.py`, `daydream/cli.py`, `docs/extensions.md`
- Test: `tests/test_improve_flow.py` (Pattern P1), `tests/test_cli_verbs.py`

- [ ] **Step 1: Write the failing tests**

```python
async def test_improve_recon_writes_artifacts_and_never_mutates_source(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    before = _git_status_porcelain(improve_monorepo_target)
    code = await run(RunConfig(target=str(improve_monorepo_target),
                               flow_name="improve", non_interactive=True))
    assert code == 0
    dd = improve_monorepo_target / ".daydream" / "improve"
    services = json.loads((dd / "services.json").read_text())
    assert {s["name"] for s in services["services"]} == {"billing", "catalog"}
    assert (dd / "report.md").is_file()
    assert all(c["read_only"] for c in stub.calls)         # every agent read-only
    assert _git_status_porcelain(improve_monorepo_target) == before  # tracked tree untouched
```

CLI test (in `tests/test_cli_verbs.py`, following its existing verb-dispatch style):

```python
def test_improve_verb_builds_improve_config() -> None:
    config = _parse_improve_args(["improve", "/tmp/x", "--effort", "deep", "--focus", "security"])
    assert (config.flow_name, config.improve_effort, config.improve_focus) == ("improve", "deep", "security")

def test_improve_rejects_unknown_effort() -> None:
    with pytest.raises(SystemExit):
        _parse_improve_args(["improve", "/tmp/x", "--effort", "extreme"])
```

The `_ImproveStubBackend` (Pattern P1) records `read_only` per call and answers, at minimum, the pattern-scanner and recon prompts; the trajectory-path assertion below rides the recorder the preamble opens.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_flow.py -x -k never_mutates`
Expected: FAIL — `UnresolvedExtensionError: flow 'improve' is not registered` (surfaced as the Extension Error exit path, code 1).

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/trajectory.py`, `daydream/improve/artifacts.py`, `daydream/improve/orchestrator.py`, `daydream/extensions/builtins.py`, `daydream/runner.py`, `daydream/cli.py`, `docs/extensions.md`

**Behavior contract:**
- `trajectory.py`: add `DaydreamRunFlow.IMPROVE = "improve"` (`trajectory.py:182-192`) and `DaydreamPhase.RECON = "recon"`, `AUDIT = "audit"`, `VET = "vet"`, `PLAN_WRITE = "plan_write"` (`trajectory.py:162-179`).
- `artifacts.py`: `improve_dir(target) -> Path` (`.daydream/improve/`), plus path helpers `services_path`, `recon_path`, `audit_findings_path(dd, category, stack)`, `vetted_findings_path`, `report_path` — mirror `daydream/deep/artifacts.py` naming.
- `orchestrator.py`: `STEPS: tuple[FlowStep, ...] = (FlowStep(name="recon", run=_step_recon), FlowStep(name="improve-report", run=_step_report, config_phase="recon"))` for now. `_step_recon`: enumerate services (Task 3) honoring `config.improve_scope` via `filter_scope` (a `ValueError` → `print_error` + `Stop(1)`), write `services.json`; run `repo_scan` (Task 6) via `ctx.backend_for("recon")` under `phase_scope(DaydreamPhase.RECON)`; run one read-only recon agent (`run_agent(..., phase=DaydreamPhase.RECON, read_only=True)`) that returns structured recon facts (languages, build/test/lint commands, conventions, intent docs — the `SKILL.md:27-37` recon inventory) written to `recon.json`; detect stacks for later fan-out with `detect_stacks(git ls-files list, skill_availability)` (`daydream/deep/detection.py:145`, availability via `get_installed_skills()` fallback pattern, `deep/orchestrator.py:1537-1541`) into `ctx.data["stacks"]`. `_step_report`: render `report.md` (services, stacks, what ran; grows in later tasks) and print a summary.
- `builtins.py._register_builtin_flows` (`builtins.py:56-75`): register `improve` STEPS + `registry.set_flow("improve", [step.name for step in improve.STEPS])`.
- `runner.py`: `RunConfig` gains `improve_effort: str = "standard"`, `improve_focus: str | None = None`, `improve_scope: str | None = None`, `improve_plan_description: str | None = None`, `improve_review_plan: str | None = None`. `_dispatch_selected_flow` (`runner.py:699-733`) gains `if name == "improve": return await _run_improve(work, config)` — **before** the generic custom-flow fallthrough, and without `_require_reviewable_branch` (repo-scoped audit on `main` is the normal case). `_run_improve` preamble: open the recorder via `_open_recorder(..., flow_kind=DaydreamRunFlow.IMPROVE)` (mandatory factory, `runner.py:313-329`), build `FlowContext` with `ctx.data` seeds (`improve_dir`, effort tier resolved from `EFFORT_TIERS[config.improve_effort]`), print the info block (target, effort, focus, model via `ctx.backend_for("recon").model`, escaped identity — mirror `_run_custom_flow`, `runner.py:1044-1061`), and `return await run_flow(ctx.registry, "improve", ctx)`. Extend the `skip_tests` condition (`runner.py:632`) with `or config.flow_name == "improve"`.
- `cli.py`: add `"improve"` to `KNOWN_VERBS` (`cli.py:61`); dispatch in `main()` before `_parse_args` (mirror the `feedback` shape, `cli.py:848-853` + `1616-1618`): `_build_improve_parser()` with `TARGET` positional, `--effort {quick,standard,deep}` (default `standard`), `--focus {security,performance,tests,branch,next}`, `--scope <service|glob>`, plus the shared flags improve honors (`--backend`, `--model`, `--non-interactive`, `--yes`, `--no-archive` — reuse `_add_shared_arguments` where its flags apply); builds `RunConfig(flow_name="improve", ...)` and runs `anyio.run(run, config)`. Sub-verbs `plan`/`review-plan` are *parsed and rejected* with "not implemented until Task 14"? — **No dead surface and no lies:** do NOT parse sub-verbs in this task at all; Task 14 adds them.
- `docs/extensions.md`: add the `improve` flow table (steps + config keys, currently `recon`/`improve-report`) — Pattern P2; extend it in Tasks 8–11 as steps land.

**Reference:** `daydream/runner.py:1021-1063` (`_run_custom_flow` — the closest preamble analog; the delta: no diff seed, no `post_to_pr`, improve-specific `ctx.data`), `daydream/deep/orchestrator.py:1426-1447` (STEPS registration shape).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_flow.py tests/test_cli_verbs.py -x` → PASS.
Then: `uv run pytest tests/test_extension_contract_doc.py tests/test_cli_main_integration.py` → PASS. Then `make check` → green.

- [ ] **Step 5: Sweep**

`builtins.py` docstring "all four flow definitions" → five; `runner.py` module docstring dispatch map (`runner.py:1-29`) gains the improve line; `cli.py` module docstring verb list (`cli.py:11-26`).

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/artifacts.py daydream/improve/orchestrator.py daydream/trajectory.py \
        daydream/extensions/builtins.py daydream/runner.py daydream/cli.py docs/extensions.md \
        tests/test_improve_flow.py tests/test_cli_verbs.py
git commit -m "feat(improve): improve flow skeleton, CLI verb, and read-only recon step"
```

---### Task 8: Audit fan-out step

**Files:**
- Modify: `daydream/improve/orchestrator.py`, `docs/extensions.md`
- Test: `tests/test_improve_flow.py`

- [ ] **Step 1: Write the failing tests**

```python
async def test_standard_effort_fans_out_all_nine_categories_read_only(
    improve_monorepo_target: Path, monkeypatch) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    await run(RunConfig(target=str(improve_monorepo_target), flow_name="improve",
                        non_interactive=True))
    audited = json.loads(_dd(improve_monorepo_target, "audit-findings.json").read_text())
    assert set(audited["categories_run"]) == set(AUDIT_CATEGORIES)
    audit_calls = [c for c in stub.calls if c["marker"] == "audit"]
    assert audit_calls and all(c["read_only"] for c in audit_calls)
    # skill-mapped slot used where registered: correctness/python carried the beagle invocation
    assert any("beagle-python:review-python" in c["prompt"] for c in audit_calls)

async def test_quick_effort_restricts_categories(...) -> None:
    await run(RunConfig(..., improve_effort="quick", non_interactive=True))
    audited = json.loads(_dd(...).read_text())
    assert set(audited["categories_run"]) == {"correctness", "security", "tests"}

async def test_failed_category_is_reported_not_silently_dropped(...) -> None:
    stub.fail_categories = {"performance"}
    code = await run(RunConfig(..., non_interactive=True))
    assert code == 0
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert "performance" in report.lower() and "not audited" in report.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_flow.py -x -k "fans_out or quick_effort"`
Expected: FAIL — `FileNotFoundError` for `audit-findings.json` (no audit step yet).

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/improve/orchestrator.py`, `docs/extensions.md`

**Behavior contract:**
- New `FlowStep(name="audit", run=_step_audit)` inserted after `recon` in `STEPS`. Assignment computation: for each category in the resolved tier's categories (filtered by focus in Task 12): for each detected stack (from `ctx.data["stacks"]`) with a registered `audit:<category>:<stack>` slot → one (category, stack, skill) agent; remaining coverage for that category → one (category, None) agent using the playbook prompt (or the stack-agnostic `audit:<category>` slot when registered). Skill invocations route through `backend.format_skill_invocation` (never raw slot strings — `phases.py:2777,2817`).
- Fan-out mechanics copy `phase_per_stack_reviews` (`daydream/phases.py:2762-2857`): `anyio.CapacityLimiter(tier.max_concurrency)`, task group, default-arg closure capture, `maybe_fork(recorder, f"audit-{category}[-{stack}]")` per agent, failures collected as `{assignment: "Type: msg"}` — surfaced in `report.md`'s not-audited section, never silently dropped. Each agent call: `run_agent(backend, work.repo, prompt, phase=DaydreamPhase.AUDIT, output_schema=AUDIT_FINDINGS_SCHEMA, read_only=True)` via `ctx.backend_for("audit")`.
- Host-side post-processing per finding: stamp `category`, attribute `services` by matching evidence paths against `ctx.data["services"]` roots, compute `fingerprint` via `compute_fingerprint(path, title, body)` (`daydream/pr_review.py:399-421`), enforce tier caps (`high_confidence_only`, `max_findings` — the cut is leverage-ordered so the dropped tail is lowest-leverage, and the count dropped is recorded for the report). Write `audit-findings.json` (`{"categories_run": [...], "failed": {...}, "findings": [...]}`) plus per-assignment artifacts under `.daydream/improve/`.
- A finding without at least one `path:line` evidence entry is discarded host-side (playbook: "A finding is only a finding with evidence", `audit-playbook.md:5`), counted in the report.

**Reference:** `daydream/phases.py:2714-2857` — mirror the limiter/task-group/failure-map shape; the delta is assignment = category × stack, tier-driven concurrency, `read_only=True`, and structured output instead of review files.

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_flow.py -x` → PASS.
Then: `uv run pytest tests/test_extension_contract_doc.py` → PASS (doc row added). Then `make check` → green.

- [ ] **Step 5: Sweep** — none beyond keeping `STEPS` docstring current.

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/orchestrator.py docs/extensions.md tests/test_improve_flow.py
git commit -m "feat(improve): tier-driven category-by-stack audit fan-out"
```

---

### Task 9: Vet step — re-verification, rejection persistence, re-report suppression

**Files:**
- Create: `daydream/improve/plans.py` (rejected.json read/write only, in this task)
- Modify: `daydream/improve/orchestrator.py`, `docs/extensions.md`
- Test: `tests/test_improve_flow.py`, `tests/test_improve_plans.py`

- [ ] **Step 1: Write the failing tests**

```python
async def test_vet_rejects_unconfirmed_finding_with_reason_and_persists(
    improve_monorepo_target: Path, monkeypatch) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.vet_reject_titles = {"Phantom N+1"}          # stub emits this finding, then rejects it
    await run(RunConfig(target=str(improve_monorepo_target), flow_name="improve",
                        non_interactive=True))
    vetted = json.loads(_dd(improve_monorepo_target, "vetted-findings.json").read_text())
    assert all(f["title"] != "Phantom N+1" for f in vetted["findings"])
    rejected = json.loads((improve_monorepo_target / "daydream_plans" / "rejected.json").read_text())
    assert rejected["rejected"][0]["title"] == "Phantom N+1"
    assert rejected["rejected"][0]["reason"]

async def test_previously_rejected_finding_is_not_revetted_or_rereported(...) -> None:
    await run(config)                                  # run 1: rejects Phantom N+1
    stub.calls.clear()
    await run(config)                                  # run 2: same audit output
    vet_calls = [c for c in stub.calls if c["marker"] == "vet"]
    assert all("Phantom N+1" not in c["prompt"] for c in vet_calls)   # pre-filtered by fingerprint
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert "previously rejected" in report.lower()
```

Unit (`tests/test_improve_plans.py`): `load_rejections` / `record_rejections` round-trip; malformed/absent file yields empty.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_flow.py -x -k "vet_rejects or not_revetted"`
Expected: FAIL — `FileNotFoundError: ... vetted-findings.json`.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/improve/orchestrator.py`, `daydream/improve/plans.py`, `docs/extensions.md`

**Behavior contract:**
- `plans.py` (this task's slice): `load_rejections(plans_dir) -> dict[str, dict]` (fingerprint-keyed; absent/malformed file → `{}`, tolerant like `load_toml_or_empty`) and `record_rejections(plans_dir, entries)` appending `{fingerprint, title, path, reason, rejected_at_sha}` — versioned envelope `{"schema_version": 1, "rejected": [...]}`, written pretty-printed (mirror `write_findings_artifact`, `findings.py:191-194`). `daydream_plans/` is created on first write.
- New `FlowStep(name="vet", run=_step_vet)` after `audit`. Pre-filter: findings whose fingerprint is in `load_rejections` are removed before any agent call and counted as `previously_rejected` for the report (spec line 25: rejected findings not re-reported). Remaining findings go to vet agents in category batches (positional 1-based `vet_id`, echo-checked like `arb_id` — reuse the id-echo guard semantics of `_apply_adjudication_verdicts`, `deep/orchestrator.py:527-561`): `run_agent(..., phase=DaydreamPhase.VET, output_schema=VET_SCHEMA, read_only=True)` via `ctx.backend_for("vet")`.
- Fail polarity is **closed**: a missing verdict, a `vet_id` mismatch, or `keep: false` rejects the finding — spec line 25 requires every *presented* finding to have been re-verified. `keep: false` verdicts carry the model's reason; missing/mismatched verdicts record reason `"vet returned no verdict"` . All rejections (model-rejected only — *not* the id-mismatch/no-verdict mechanical drops, which get re-audited next run rather than permanently suppressed) are persisted via `record_rejections`.
- Kept findings absorb the verdict's corrected fields (severity/impact/effort/risk/confidence/path/line — vet corrects mis-attributed evidence, `SKILL.md:68`) and are written to `vetted-findings.json`.

**Reference:** `daydream/deep/orchestrator.py:464-570` (`_apply_adjudication_verdicts` — mirror the id-echo guard and fail-polarity discipline; the delta: fail-closed always, plus persistence of reasons).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_flow.py tests/test_improve_plans.py -x` → PASS.
Then: `make check` → green.

- [ ] **Step 5: Sweep** — none (additive).

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/orchestrator.py daydream/improve/plans.py docs/extensions.md \
        tests/test_improve_flow.py tests/test_improve_plans.py
git commit -m "feat(improve): fail-closed vet pass with persistent rejection memory"
```

---

### Task 10: Prioritize, selection gate, and report

**Files:**
- Modify: `daydream/improve/orchestrator.py`, `docs/extensions.md`
- Test: `tests/test_improve_flow.py`

- [ ] **Step 1: Write the failing tests**

```python
async def test_non_interactive_selects_top_findings_never_touching_stdin(
    improve_monorepo_target: Path, monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", _forbidden_input)   # test_deep_orchestrator.py:3241 pattern
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target, n_findings=8)
    code = await run(RunConfig(target=str(improve_monorepo_target), flow_name="improve",
                               non_interactive=True))
    assert code == 0
    selected = json.loads(_dd(improve_monorepo_target, "selected.json").read_text())
    assert len(selected["selected"]) == 5                      # top min(5, n) by leverage
    assert selected["mode"] == "non-interactive-default"

async def test_interactive_selection_honors_user_choice(...) -> None:
    _force_interactive(monkeypatch)                            # test_deep_orchestrator.py:608
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "2")
    await run(RunConfig(..., non_interactive=False))
    selected = json.loads(_dd(...).read_text())
    assert len(selected["selected"]) == 1

async def test_report_orders_by_leverage_and_separates_direction(...) -> None:
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert report.index("high-leverage-title") < report.index("low-leverage-title")
    assert "## Direction" in report and "not audited" in report.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_flow.py -x -k selection`
Expected: FAIL — `FileNotFoundError: ... selected.json`.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/improve/orchestrator.py`, `docs/extensions.md`

**Behavior contract:**
- New `FlowStep(name="prioritize", run=_step_prioritize)` (host-only, no agent): loads vetted findings, `partition_direction`, `order_by_leverage` (Task 4), writes the ordered lists back to `vetted-findings.json` and renders the findings table + direction section into `ctx.data` for the report.
- New `FlowStep(name="select-plans", run=_step_select)`: interactive path prompts with the numbered leverage-ordered defect table and reads a comma/range selection via `daydream.agent.prompt_user` (default = top `min(5, n)`); non-interactive (`get_non_interactive()`, `agent.py` state set by `run`) takes the default silently — mirror the gate discipline of `_step_fix_gate` (`deep/orchestrator.py:1169-1197`) including EOF safety. Writes `selected.json` (`{"mode": "interactive"|"non-interactive-default", "selected": [fingerprints]}`). Zero vetted defect findings → print success, still write the report, `Stop(0)`.
- `_step_report` (Task 7) now renders the full advisory report: leverage-ordered defect table (`# | Finding | Category | Impact | Effort | Risk | Confidence | Evidence`, `SKILL.md:70-72`), direction findings in their own section (2–4 max at standard effort), counts of vet-rejected / previously-rejected / evidence-discarded findings, and the **what-was-not-audited statement** (tier coverage bounds + failed assignments + scope slicing) — spec lines 24, 29 and `SKILL.md:62`.
- Direction findings are excluded from the selection default (Assumption 2); selecting them interactively is allowed (they become design/spike plans in Task 11).

**Reference:** `daydream/deep/orchestrator.py:1169-1197` (gate + `resolve_or_prompt` discipline; the delta: multi-select numbers instead of y/N — a plain `prompt_user` read, parsed leniently, invalid input reprompts once then falls back to the default).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_flow.py -x` → PASS. Then `make check` → green.

- [ ] **Step 5: Sweep** — none.

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/orchestrator.py docs/extensions.md tests/test_improve_flow.py
git commit -m "feat(improve): leverage-ordered report and finding selection gate"
```

---

### Task 11: Plan writing and index

**Files:**
- Modify: `daydream/improve/plans.py`, `daydream/improve/orchestrator.py`, `docs/extensions.md`
- Test: `tests/test_improve_plans.py`, `tests/test_improve_flow.py`

- [ ] **Step 1: Write the failing tests**

Unit — the plan-file contract (`tests/test_improve_plans.py`):

```python
def test_rendered_plan_carries_every_template_section() -> None:
    text = render_plan(_finding(), body_markdown="## Steps\n...", planned_at="abc1234",
                       number=1, slug="fix-n-plus-one", commands={"Tests": "uv run pytest"})
    for section in ("## Status", "## Why this matters", "## Current state",
                    "## Commands you will need", "## Scope", "## Steps",
                    "## Test plan", "## Done criteria", "## STOP conditions"):
        assert section in text
    assert "abc1234" in text and "Drift check" in text      # planned-at SHA + drift instruction

def test_index_reconcile_keeps_numbering_monotonic(tmp_path: Path) -> None:
    write_plans(tmp_path / "daydream_plans", [_sel(slug="a")], planned_at="abc1234")
    write_plans(tmp_path / "daydream_plans", [_sel(slug="b")], planned_at="def5678")
    names = sorted(p.name for p in (tmp_path / "daydream_plans").glob("[0-9]*.md"))
    assert names == ["001-a.md", "002-b.md"]                # no renumbering, no duplicate 001
```

Real-path (`tests/test_improve_flow.py`):

```python
async def test_non_interactive_run_writes_plans_and_index(...) -> None:
    code = await run(RunConfig(..., non_interactive=True))
    plans_dir = improve_monorepo_target / "daydream_plans"
    plan_files = sorted(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert 1 <= len(plan_files) <= 5
    index = (plans_dir / "README.md").read_text()
    assert "non-interactive default" in index.lower()        # spec line 27: default recorded
    assert head_sha(improve_monorepo_target)[:7] in plan_files[0].read_text()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_plans.py -x -k template`
Expected: FAIL — `ImportError: cannot import name 'render_plan'`.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/improve/plans.py`, `daydream/improve/orchestrator.py`

**Behavior contract:**
- `render_plan(...)` produces the ported handoff template (`plan-template.md:17-156`): executor instructions + drift-check preamble (`git diff --stat <planned-at>..HEAD -- <in-scope paths>`), Status block (priority/effort/risk/depends-on/category/planned-at — no Issue field in Phase 1), and the agent-authored body sections. The host stamps `planned_at` (from `git_ops.head_sha(work.repo)`, `runner.py:528-537` shows the guarded call shape) and the recon-verified commands table — never trusting the model for either. Direction findings render with a design/spike framing note (`audit-playbook.md:104`).
- New `FlowStep(name="write-plans", run=_step_write_plans, config_phase="plan_write")` before `improve-report`: for each selected finding, one `run_agent(..., phase=DaydreamPhase.PLAN_WRITE, output_schema=PLAN_WRITER_SCHEMA, read_only=True)` via `ctx.backend_for("plan_write")` (parallel under the tier limiter, `maybe_fork(recorder, f"plan-{slug}")`); the host validates the returned `slug` (`[a-z0-9-]{1,60}`, else derived from the title) and writes `daydream_plans/NNN-<slug>.md`.
- `write_plans` / index behavior — **reconcile, don't duplicate** (`SKILL.md:93`): read the existing `README.md` table if present; numbering continues monotonically from the highest existing `NNN`; a finding whose fingerprint already has a plan row or a `rejected.json` entry is skipped (skips reported). `README.md` carries the execution-order/status table (`plan-template.md:164-187`: Plan | Title | Priority | Effort | Depends on | Status), a machine-readable fingerprint marker per row (HTML comment), a dependency-notes section, the "Findings considered and rejected" section rendered from `rejected.json`, and — on non-interactive runs — the sentence recording that the top-N default was applied.
- Failure propagation: a plan-writer agent failure skips that plan (recorded in report + index as `BLOCKED (plan-writing failed)`) and never aborts the other writes; exit code stays 0 when ≥1 plan landed, 1 when all selected plans failed.

**Reference:** `plan-template.md:17-196` (the template + quality bar — port headings verbatim), `daydream/phases.py:2762-2857` (parallel fan-out shape).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_plans.py tests/test_improve_flow.py -x` → PASS. Then `make check` → green.

- [ ] **Step 5: Sweep** — none.

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/plans.py daydream/improve/orchestrator.py docs/extensions.md \
        tests/test_improve_plans.py tests/test_improve_flow.py
git commit -m "feat(improve): self-contained plan writing with reconciling index"
```

---

### Task 12: Focus modes — security / performance / tests / next

**Files:**
- Modify: `daydream/improve/orchestrator.py`
- Test: `tests/test_improve_flow.py`

- [ ] **Step 1: Write the failing tests**

```python
async def test_focus_security_audits_single_category(...) -> None:
    await run(RunConfig(..., improve_focus="security", non_interactive=True))
    audited = json.loads(_dd(...,"audit-findings.json").read_text())
    assert audited["categories_run"] == ["security"]

async def test_focus_next_is_direction_only_and_plans_are_spikes(...) -> None:
    await run(RunConfig(..., improve_focus="next", non_interactive=True))
    audited = json.loads(_dd(...,"audit-findings.json").read_text())
    assert audited["categories_run"] == ["direction"]
    plan = next((improve_monorepo_target / "daydream_plans").glob("0*.md")).read_text()
    assert "spike" in plan.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_flow.py -x -k focus`
Expected: FAIL — all nine categories run despite the focus.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/improve/orchestrator.py`

**Behavior contract:**
- Category resolution becomes one pure helper `resolve_categories(tier, focus) -> tuple[str, ...]`: focus `security`/`performance`/`tests` → that single category (recon still runs, `SKILL.md:111`); `next` → `("direction",)` with direction-depth instructions in the audit prompt (4–6 grounded suggestions, `SKILL.md:113`); no focus → tier categories. `branch` is Task 13, not a category filter.
- With focus `next`, the selection default includes direction findings (they are the only findings) and Task 11's spike framing applies to every plan.
- Focus composes with effort (`quick security` narrows both — `SKILL.md:110`); an effort tier that excludes the focused category does not silently drop it — the focused category always runs.

**Reference:** `SKILL.md:107-118` (invocation-variant semantics).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_flow.py -x` → PASS. Then `make check` → green.

- [ ] **Step 5: Sweep** — none.

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/orchestrator.py tests/test_improve_flow.py
git commit -m "feat(improve): security/performance/tests/next focus modes"
```

---

### Task 13: Branch focus mode

**Files:**
- Modify: `daydream/improve/orchestrator.py`
- Test: `tests/test_improve_flow.py`

- [ ] **Step 1: Write the failing tests**

```python
async def test_branch_focus_scopes_audit_to_merge_base_diff_and_tags_provenance(
    improve_branch_target: Path, monkeypatch) -> None:
    # fixture: main + feature branch touching apps/billing/api.py only
    stub = _install_improve_stub(monkeypatch, improve_branch_target)
    await run(RunConfig(target=str(improve_branch_target), flow_name="improve",
                        improve_focus="branch", non_interactive=True))
    audit_calls = [c for c in stub.calls if c["marker"] == "audit"]
    assert all("apps/billing/api.py" in c["prompt"] for c in audit_calls)
    vetted = json.loads(_dd(improve_branch_target, "vetted-findings.json").read_text())
    assert {f["provenance"] for f in vetted["findings"]} <= {"introduced", "inherited"}

async def test_branch_focus_on_base_branch_reports_and_exits_cleanly(...) -> None:
    # fixture checked out on main: nothing to diff against itself
    code = await run(RunConfig(..., improve_focus="branch", non_interactive=True))
    assert code == 1   # actionable error, mirrors WrongBranch semantics
```

Add `improve_branch_target` beside `improve_monorepo_target` in `tests/conftest.py` (feature-branch variant, mirroring `multi_stack_target`'s branch dance at `tests/conftest.py:192-212`).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_flow.py -x -k branch_focus`
Expected: FAIL — audit prompts carry the whole-repo scope.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/improve/orchestrator.py`, `tests/conftest.py`

**Behavior contract:**
- Focus `branch`: scope = files changed since the merge-base (`git_ops.diff(work.repo, work.base_branch)` file list — reuse `_diff_changed_files`, `deep/orchestrator.py:368-390`; import it rather than copying) plus the diff text passed to audit prompts; all tier categories run at light depth (`SKILL.md:112`: "Light recon, all categories, usually no subagents" → concurrency clamps to 1).
- Every finding is tagged `provenance: "introduced"` or `"inherited"` — the audit prompt instructs the tagging against the supplied diff, and the vet prompt confirms it; a finding missing the tag after vet defaults to `"inherited"` (never blames the branch without evidence). The report table separates the two groups (`SKILL.md:112`).
- On the base branch / zero commits ahead (`work.head_branch == work.base_branch`, `WorkContext` fields at `workspace.py:46-69`): actionable `print_error` offering a full audit, `Stop(1)` — improve does NOT reuse `_require_reviewable_branch` (`runner.py:675-696`) because only this focus mode needs a branch; the plain repo-scoped run must keep working on `main`.

**Reference:** `daydream/deep/orchestrator.py:368-390`, `daydream/runner.py:675-696` (guard message tone).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_flow.py -x` → PASS. Then `make check` → green.

- [ ] **Step 5: Sweep** — none.

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/orchestrator.py tests/conftest.py tests/test_improve_flow.py
git commit -m "feat(improve): branch focus with introduced/inherited provenance tagging"
```

---

### Task 14: `improve plan <description>` and `improve review-plan <file>`

**Files:**
- Modify: `daydream/cli.py`, `daydream/improve/orchestrator.py`, `daydream/improve/plans.py`
- Test: `tests/test_cli_verbs.py`, `tests/test_improve_flow.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_improve_plan_subverb_parses_description() -> None:
    config = _parse_improve_args(["improve", "plan", "add rate limiting", "/tmp/x"])
    assert config.improve_plan_description == "add rate limiting"

async def test_plan_subverb_skips_audit_and_writes_single_plan(...) -> None:
    await run(RunConfig(..., improve_plan_description="add rate limiting",
                        non_interactive=True))
    assert not _dd(improve_monorepo_target, "audit-findings.json").exists()
    plans = list((improve_monorepo_target / "daydream_plans").glob("0*-*.md"))
    assert len(plans) == 1 and "rate limiting" in plans[0].read_text().lower()

async def test_review_plan_rejects_files_outside_daydream_plans(...) -> None:
    code = await run(RunConfig(..., improve_review_plan="README.md", non_interactive=True))
    assert code == 1                                   # only daydream_plans/ is writable

async def test_review_plan_tightens_in_place(...) -> None:
    plan = improve_monorepo_target / "daydream_plans" / "001-fix.md"
    ...  # seed a thin plan file
    await run(RunConfig(..., improve_review_plan=str(plan), non_interactive=True))
    assert "## STOP conditions" in plan.read_text()    # tightened output rewrote it
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli_verbs.py -x -k improve_plan`
Expected: FAIL — argparse treats `plan` as the TARGET positional.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/cli.py`, `daydream/improve/orchestrator.py`, `daydream/improve/plans.py`

**Behavior contract:**
- CLI: inside the improve dispatch, a manual sub-verb check on the first post-`improve` token (the `feedback` manual-dispatch precedent, `cli.py:848-853`): `plan` consumes a required description argument; `review-plan` consumes a required file path; both then accept the shared improve flags + TARGET.
- Flow gating via `enabled` predicates on the existing steps (`FlowStep.enabled`, `extensions/api.py:91`): `plan <description>` runs recon → a single investigation+plan-write pass (the plan-writer prompt embeds the description and the recon facts; ambiguities the agent cannot resolve from the codebase are listed in the plan's STOP-conditions/open-questions section rather than prompted one-at-a-time in Phase 1 — the interactive Q&A loop of `SKILL.md:114` needs conversational turns daydream doesn't have); audit/vet/prioritize/select are `enabled=False`. Index reconcile applies (the single plan gets the next `NNN`).
- `review-plan <file>`: refuses paths not under `<target>/daydream_plans/` with an actionable error (`Stop(1)`); otherwise one read-only agent critiques against the template quality bar (`plan-template.md:189-197`) and returns the tightened full markdown; the host overwrites the file only when the returned content still contains every required section (Task 11's section list — a degenerate rewrite is rejected, original left intact, exit 1 with the critique printed).
- Both sub-verbs skip service enumeration output requirements but still write `report.md` naming what ran.

**Reference:** `SKILL.md:114-115` (variant semantics), `daydream/cli.py:832-860` (`_parse_args` manual dispatch shape).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_cli_verbs.py tests/test_improve_flow.py -x` → PASS. Then `make check` → green.

- [ ] **Step 5: Sweep** — `cli.py` module docstring verb list gains the sub-verbs.

- [ ] **Step 6: Commit**

```bash
git add daydream/cli.py daydream/improve/orchestrator.py daydream/improve/plans.py \
        tests/test_cli_verbs.py tests/test_improve_flow.py
git commit -m "feat(improve): plan-from-description and review-plan sub-verbs"
```

---

### Task 15: Monorepo scoping, cross-service aggregation, coverage statement

**Files:**
- Modify: `daydream/improve/prioritize.py`, `daydream/improve/orchestrator.py`
- Test: `tests/test_improve_prioritize.py`, `tests/test_improve_flow.py`

- [ ] **Step 1: Write the failing tests**

Unit (aggregation):

```python
def test_same_pattern_across_services_aggregates_to_one_finding() -> None:
    a = _f(title="Unbounded query in list endpoint", path="apps/billing/api.py", services=["billing"])
    b = _f(title="Unbounded query in the list endpoint", path="apps/catalog/api.py", services=["catalog"])
    merged = aggregate_cross_service([a, b])
    assert len(merged) == 1
    assert set(merged[0]["services"]) == {"billing", "catalog"}
    assert len(merged[0]["evidence"]) >= 2            # per-service evidence retained

def test_distinct_findings_do_not_aggregate() -> None:
    assert len(aggregate_cross_service([_f(title="SQL injection"), _f(title="Slow CI cache")])) == 2
```

Real-path:

```python
async def test_scope_slices_search_but_report_names_the_unaudited_rest(...) -> None:
    await run(RunConfig(..., improve_scope="apps/billing", non_interactive=True))
    audit_calls = [c for c in stub.calls if c["marker"] == "audit"]
    assert all("apps/billing" in c["prompt"] for c in audit_calls)
    assert any("bounds where the audit searches" in c["prompt"].lower() for c in audit_calls)
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert "catalog" in report and "not audited" in report.lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_improve_prioritize.py -x -k aggregate`
Expected: FAIL — `ImportError: cannot import name 'aggregate_cross_service'`.

- [ ] **Step 3: Implement against the tests**

**Files touched:** `daydream/improve/prioritize.py`, `daydream/improve/orchestrator.py`

**Behavior contract:**
- `aggregate_cross_service(findings) -> list[dict]` groups findings by `category` + normalized-title bigram Jaccard ≥ 0.5 — import `_normalize_title`/`_bigrams`/`_jaccard` from `daydream/deep/dedup.py:55-83` (promote them to public names there if lint objects; do not copy) — where the group spans ≥2 distinct `services`. The aggregate keeps the highest-leverage member's axes, unions `services` and `evidence`, and lists per-service locations in `body`. Same-service near-duplicates are left alone (vet already deduped within a batch). Runs between vet and prioritize.
- Scope plumbing (from Task 7's `--scope`): the audit prompt states the slice explicitly and instructs "slicing bounds where you *search*, never what you may *read*; cross-service boundary findings (traffic and data flow between services) remain in scope" (spec lines 29, 133).
- `report.md` gains the top-offender summary (services ranked by summed leverage of their findings — Should-Have, spec line 74) and, when sliced, the explicit list of services/dirs not audited.

**Reference:** `daydream/deep/dedup.py:55-134` (similarity primitives + threshold rationale).

- [ ] **Step 4: Run the new tests AND the suite**

Run: `uv run pytest tests/test_improve_prioritize.py tests/test_improve_flow.py -x` → PASS. Then `make check` → green.

- [ ] **Step 5: Sweep** — if dedup helpers were made public, update `deep/dedup.py`'s docstring and any `_`-prefixed imports elsewhere (grep `_normalize_title`, `_bigrams`, `_jaccard`).

- [ ] **Step 6: Commit**

```bash
git add daydream/improve/prioritize.py daydream/improve/orchestrator.py daydream/deep/dedup.py \
        tests/test_improve_prioritize.py tests/test_improve_flow.py
git commit -m "feat(improve): monorepo scope slicing and cross-service finding aggregation"
```

---

### Task 16: Trust-boundary proofs and doc sweep

**Files:**
- Modify: `tests/test_improve_flow.py`, `CLAUDE.md`, `README.md`

- [ ] **Step 1: Write the failing/gap-closing tests**

```python
async def test_full_run_leaves_tracked_tree_and_untracked_set_untouched(
    improve_monorepo_target: Path, monkeypatch) -> None:
    # end-to-end (recon → plans) with a stub that ALSO tries a Write tool turn;
    # the ONLY new paths after the run are daydream_plans/ and .daydream/.
    before_status = _git_status_porcelain(improve_monorepo_target)
    await run(RunConfig(target=str(improve_monorepo_target), flow_name="improve",
                        non_interactive=True))
    assert _git_status_porcelain(improve_monorepo_target) == before_status
    new_untracked = _untracked(improve_monorepo_target)
    assert all(p.startswith(("daydream_plans/", ".daydream/")) for p in new_untracked)

async def test_every_agent_call_in_every_mode_is_read_only(...) -> None:
    for cfg in (_base(), _base(improve_focus="next"), _base(improve_plan_description="x")):
        stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
        await run(cfg)
        assert stub.calls and all(c["read_only"] for c in stub.calls)

async def test_trajectory_records_improve_flow_and_phases(...) -> None:
    phases = _scan_trajectory_extra(run_root, traj, "daydream_phase")   # test_deep_orchestrator helper
    assert {"recon", "audit", "vet", "plan_write"} <= set(phases)
```

- [ ] **Step 2: Run — identify any failures**

Run: `uv run pytest tests/test_improve_flow.py -x`
Expected: PASS if Tasks 7–15 held the contract; any failure here is a Task-7-to-15 bug to fix at its root before proceeding (never by weakening these assertions).

- [ ] **Step 3: Documentation**

**Files touched:** `CLAUDE.md`, `README.md`

**Behavior contract:**
- `CLAUDE.md`: add `daydream improve` lines to the Commands golden-path block, the `improve` flow to the execution-flow list and module-responsibility table (`daydream/improve/`), and the new env-free config keys to the config section.
- `README.md`: user-facing `improve` section (verb, effort tiers, focus modes, sub-verbs, `daydream_plans/` output contract, read-only guarantee).

- [ ] **Step 4: Full gate**

Run: `make check` → green. This is the phase-completion gate: `uv lock --check` + ruff + mypy + full pytest.

- [ ] **Step 5: Sweep**

Grep for leftovers across the phase: `grep -rn "TODO\|XXX" daydream/improve/` → empty; `grep -rn "improve" docs/extensions.md` rows match the final registered STEPS/slots/prompts.

- [ ] **Step 6: Commit**

```bash
git add tests/test_improve_flow.py CLAUDE.md README.md
git commit -m "test(improve): trust-boundary real-path proofs; docs for the improve verb"
```

---

## Self-Review Outcome

- **Spec coverage (Phase 1 lines 19–30):** verb + read-only + `daydream_plans/`-only writes → Tasks 7, 11, 16; effort tiers → Tasks 2, 8; focus modes + sub-verbs → Tasks 12–14; nine categories → Tasks 2, 8; finding axes + leverage ordering + direction separation → Tasks 1, 4, 10; vet + rejection memory → Task 9; self-contained plans + index → Task 11; interactive/non-interactive selection → Task 10; beagle skills per slot → Tasks 2, 5, 8; monorepo enumeration/scoping/aggregation/coverage statement → Tasks 3, 15; generic service enumeration + open question → Tasks 0, 3.
- **Out-of-scope check:** no tracker/issue code, no cron, no execute/reconcile verbs, no corpus changes anywhere in the plan (Phases 2–5). The `Issue` field of the source plan template is intentionally omitted in Task 11.
- **Spike candidates:** the one load-bearing unverified behavior — heuristic service detection quality — is Task 0. Everything else builds on seams read this session (flow registry, `run_agent(read_only=True)`, fingerprints, dedup, CapacityLimiter fan-out) that existing tests already exercise.
- **Trust boundaries:** Task 16 pins (a) tracked tree + untracked set unchanged, (b) `read_only=True` on every agent call in every mode, and the per-backend enforcement is pre-existing tested behavior; Task 14 pins the `review-plan` path-confinement rejection; Task 9 pins fail-closed vet with the mechanical-drop/model-reject persistence distinction.
- **Consumer check:** every new public surface has a production consumer in this plan (schema fields ← audit/vet/report; `repo_scan` ← recon step; slots/prompts ← audit step; `enumerate_services` ← recon; `aggregate_cross_service` ← vet→prioritize seam; CLI flags ← `_run_improve`).
- **Test discipline:** every behavior task is failing-test-first, asserts observable state (files, exit codes, prompt content, git status), and Step 4 names both the narrow command and the broader suite + `make check`. Real-path tests enter through `runner.run` with only `create_backend` patched, per the repo's mandatory testing standard.
- **Known open items (explicitly not deferred silently):** none — the two source-skill behaviors intentionally narrowed for Phase 1 are stated inline with rationale: `plan <description>` resolves ambiguities into the plan's open-questions section instead of interactive Q&A (Task 14), and mechanical vet drops are re-audited rather than permanently suppressed (Task 9).
