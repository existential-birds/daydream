# Stack Research

**Domain:** ATIF v1.x trajectory emission from a Python 3.12 CLI agent built on `claude-agent-sdk` (brownfield daydream)
**Researched:** 2026-04-26
**Overall Confidence:** HIGH on token-extraction surfaces and ATIF model shape; MEDIUM on the Harbor dependency choice (Harbor has a heavy core install but the trajectory submodule itself is light); HIGH on the recommended fallback path

---

## TL;DR

1. **Do not take `harbor` as a runtime dependency.** Its core PyPI distribution pins ~21 heavy transitive deps (`litellm`, `fastapi`, `uvicorn`, `datasets`, `supabase`, `ruff`, `jinja2`, `tenacity`, `dirhash`, `typer`, `shortuuid`, `pyyaml`, `toml`, `requests`, `httpx`, `python-dotenv`, `packaging`, `pathspec`, `claude-agent-sdk>=0.1.17`, `rich`, `pydantic`). There is **no slim sub-distribution** that exposes only `harbor.models.trajectories`. Worse: `litellm` had an active PyPI supply-chain compromise (versions 1.82.7 / 1.82.8 quarantined March 2026) — taking Harbor as a dep transitively pulls a moving target whose security posture is not under our control.
2. **Vendor the Pydantic models.** `harbor.models.trajectories` and `harbor.utils.trajectory_validator` only require `pydantic>=2.11.7` and stdlib (`json`, `pathlib`, `typing`, `datetime`). Pydantic v2 is already a transitive dep via `claude-agent-sdk` 0.1.52 / `mcp` 1.26.0. Copy the 11 model files (~700 LOC total), pin `pydantic>=2.11.7` explicitly, and own the file. Use Harbor's golden trajectory fixtures (`tests/golden/`) as your test corpus.
3. **Reconsider ATIF v1.4.** Harbor's current `Trajectory` model defaults to **ATIF-v1.6** (added `ContentPart` / `ImageSource`, multi-part content). v1.4 is accepted by the validator but no longer the recommended target. PROJECT.md's "pinned to v1.4" decision was made before this research; flag it in the roadmap and let the orchestrator decide whether to update or proceed.
4. **Token extraction is straightforward** — the daydream Claude backend is leaving data on the floor. `AssistantMessage.usage` (per-step) and `ResultMessage.usage` + `ResultMessage.model_usage` (cumulative) are already available in `claude-agent-sdk==0.1.52`. The current `CostEvent` emission path passes `input_tokens=None, output_tokens=None` despite the data being present in `ResultMessage` since SDK 0.1.x — this is a one-line fix.
5. **Codex backend is at parity for input/output tokens, missing for cost and cache.** `turn.completed.usage` already provides `input_tokens` and `output_tokens`. There is no native cost or cache-token field; `cost_usd`, `cached_tokens`, `cache_creation_input_tokens` will be `None` for Codex runs. ATIF spec allows Metrics fields to be optional, so this is acceptable.
6. **Stdlib `uuid.uuid4()` + `datetime.now(timezone.utc).isoformat()` are sufficient.** No need for `ulid-py`. ATIF golden trajectories use UUIDs for `session_id` (e.g. `025B810F-B3A2-4C67-93C0-FE7A142A947A`); `tool_call_id` is opaque (Claude provides `block.id`, Codex generates `uuid.uuid4()` already).

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `pydantic` | `>=2.11.7,<3` | Type-safe ATIF model construction + validation | Already transitive via `claude-agent-sdk`/`mcp`; Harbor's models import only pydantic + stdlib + datetime; v2.11.7 is what Harbor pins so vendored models import cleanly. **HIGH confidence.** |
| Vendored ATIF models | (copy from `harbor` upstream) | The 11 Pydantic classes that comprise an ATIF trajectory: `Trajectory`, `Agent`, `Step`, `ToolCall`, `Observation`, `ObservationResult`, `Metrics`, `FinalMetrics`, `ContentPart`, `ImageSource`, `SubagentTrajectoryRef` | Avoids pulling Harbor's 21-package transitive footprint (incl. `litellm`, `fastapi`, `uvicorn`, `datasets`, `supabase`). The submodule itself is ~700 LOC of pure Pydantic — small enough to vendor, copyright-noted, and re-synced on minor schema bumps. **HIGH confidence in approach; MEDIUM confidence in mechanics until LICENSE/attribution is verified.** |
| Vendored validator | (copy `harbor.utils.trajectory_validator`) | Programmatic + CLI validation of produced trajectories | Single file, depends only on `pydantic.ValidationError` + the vendored models. Gives you `validate_trajectory(d) -> bool` and `python -m daydream.atif.validator path.json` parity with Harbor. **HIGH confidence.** |
| `uuid` (stdlib) | n/a | `session_id` and tool-call ID generation | Harbor's golden trajectories use UUIDv4-shaped strings for `session_id`. Codex backend already uses `uuid.uuid4()` for synthesized tool-call IDs. No third-party dep needed. **HIGH confidence.** |
| `datetime` (stdlib) | n/a | ISO 8601 step timestamps | ATIF requires ISO 8601; `datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")` produces the canonical `YYYY-MM-DDTHH:MM:SSZ` form found in Harbor's goldens. **HIGH confidence.** |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pytest` | `>=9.0.3` (already present) | Trajectory unit tests | Required regardless. Use `tmp_path` for trajectory-write tests. |
| `pytest-asyncio` | `>=1.3.0` (already present) | Async test for trajectory emission during `run_agent()` | Required for tests that exercise the live backend stream. |
| Harbor golden fixtures | git submodule OR `git pull`'d snapshot | Schema regression tests | Copy `terminus_2/*.trajectory.json` (13 files) and `openhands/*.trajectory.json` (4 files) from `https://github.com/laude-institute/harbor/tree/main/tests/golden` into `tests/fixtures/atif_golden/`. Use them in a parametrized "validator round-trips Harbor's own goldens" test to catch schema drift. **HIGH confidence.** |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `mypy` (already present) | Type-check vendored Pydantic models | Set `disallow_any_generics = false` for the vendored module if Harbor's source uses `dict[str, Any]` permissively. |
| `ruff` (already present) | Lint vendored module | Add `# noqa` exemption header in vendored files only if Harbor's style triggers `daydream`'s rule set; prefer to keep the vendor clean. |

---

## Installation

```bash
# Add to pyproject.toml [project.dependencies]:
#   "pydantic>=2.11.7"   <-- promote from transitive to explicit
# (No new dependencies otherwise.)

uv sync

# Vendor Harbor's trajectory models + validator:
mkdir -p daydream/atif/models
git clone --depth 1 --branch <pinned-tag> https://github.com/laude-institute/harbor.git /tmp/harbor
cp /tmp/harbor/src/harbor/models/trajectories/*.py daydream/atif/models/
cp /tmp/harbor/src/harbor/utils/trajectory_validator.py daydream/atif/validator.py
# Then: rewrite imports from `harbor.models.trajectories.*` to `daydream.atif.models.*`
# and add an attribution comment block at the top of each vendored file.

# Vendor golden test fixtures:
mkdir -p tests/fixtures/atif_golden/{terminus_2,openhands}
cp /tmp/harbor/tests/golden/terminus_2/*.trajectory.json tests/fixtures/atif_golden/terminus_2/
cp /tmp/harbor/tests/golden/openhands/*.trajectory.json tests/fixtures/atif_golden/openhands/
```

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Vendor `harbor.models.trajectories` + validator (~700 LOC) | `pip install harbor` and import from it | If the daydream team is willing to take on a ~50MB transitive footprint *and* accept that `litellm` (recently compromised), `supabase`, `fastapi`, `uvicorn`, `datasets` will be installed alongside daydream every time. Justifiable only if there's a separate dependency on Harbor's eval/sandbox features (we don't). **NOT RECOMMENDED for daydream.** |
| Vendor models | Hand-roll Pydantic v2 models from the RFC | If Harbor's LICENSE forbids vendoring (verify before committing — Harbor is Apache-2.0 per the GitHub repo header, but confirm at vendor time). Hand-rolling means you own the schema interpretation; Harbor's models are battle-tested by Terminus-2 / OpenHands / Mini-SWE-Agent so vendoring is strictly safer. |
| Vendor models | Vendor JSON Schema and validate with `jsonschema` (transitive via SDK) | If you need cross-language schema sharing (we don't — Python-only). Pydantic gives type-safe construction *and* validation in one library; jsonschema only validates. **NOT RECOMMENDED.** |
| `uuid.uuid4()` for `session_id` | `ulid-py` (`python-ulid` 3.1+) | If you need lexicographically-sortable IDs *and* timestamp-prefix without a separate `created_at`. Harbor goldens use UUIDv4; no ATIF consumer expects ULID. Adding a dep for cosmetics is not justified. **NOT RECOMMENDED.** |
| ISO 8601 timestamps via stdlib | `pendulum` / `arrow` | If you need timezone math beyond UTC. ATIF only requires UTC ISO 8601 — stdlib is sufficient. **NOT RECOMMENDED.** |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `pip install harbor` as a runtime dep | Pulls 21 transitive packages (~50MB+). Includes `litellm` (recent supply-chain compromise quarantined on PyPI March 2026 — `litellm` 1.82.7 and 1.82.8 had a malicious `.pth` file), `fastapi`, `uvicorn`, `supabase`, `datasets`, `tinker`. None of these are needed by daydream. Harbor pins `>=` not `==`, so each `uv sync` could resolve to new transitive versions outside daydream's review. | Vendor `harbor.models.trajectories` + `harbor.utils.trajectory_validator` (~700 LOC, pure Pydantic + stdlib). |
| Custom JSON construction (dict literals) | Skips schema validation; silent drift between code and ATIF spec. The whole point of ATIF migration is interop with Harbor's validator. | Pydantic models with `BaseModel.model_validate()` and `BaseModel.model_dump_json(exclude_none=True)`. |
| `pydantic` v1 | Harbor's models use `model_validator`, `field_validator`, `Literal` — pydantic v2 idioms. The codebase's existing transitive `pydantic` is v2. v1 has been EOL since June 2024. | `pydantic>=2.11.7` (explicit pin in `pyproject.toml`). |
| `attrs` / `dataclasses` for trajectory models | No validation; manual JSON serialization; can't reuse Harbor's models or validator. | Pydantic v2. |
| Mid-run streaming JSON writes (e.g. `jsonlines` per step) | ATIF expects a single coherent JSON document with `final_metrics` populated. Streaming requires post-hoc rewrite. PROJECT.md already lists this as out-of-scope. | Build the trajectory in memory in a `TrajectoryRecorder` instance, write once at run completion. |
| `ulid-py`, `nanoid`, `cuid` | None of Harbor's reference agents use these. Adds an opinion the spec doesn't require. | `uuid.uuid4()` from stdlib. |

---

## Detailed Findings (one section per question)

### 1. Harbor PyPI footprint — heavy

**Verified against:** `https://github.com/laude-institute/harbor/blob/main/pyproject.toml` (and the mirror at `harbor-framework/harbor`, which appears to be the canonical org per the `harborframework.com` docs). Cross-checked with `pypi.org/project/harbor/0.5.0/json`.

- Latest PyPI release: **`harbor` 0.5.0** (2026-04-23). Wheel size ~1.05 MB, source tarball ~929 KB.
- **Required deps (21):** `pydantic>=2.11.7`, `shortuuid>=1.0.13`, `typer>=0.16.0`, `requests>=2.32.4`, `pyyaml>=6.0.2`, `rich>=14.1.0`, `toml>=0.10.2`, `tenacity>=9.1.2`, `python-dotenv>=1.1.1`, `litellm>=1.80.8`, `jinja2>=3.1.6`, `datasets>=4.4.1`, `dirhash>=0.5.0`, `claude-agent-sdk>=0.1.17`, `packaging>=25.0`, `fastapi>=0.128.0`, `uvicorn>=0.38.0`, `ruff>=0.13.0`, `pathspec>=1.0.3`, `supabase>=2.28.2`, `httpx>=0.27.0`.
- **No slim subpackage exists.** Optional extras (`e2b`, `daytona`, `islo`, `modal`, `runloop`, `tensorlake`, `gke`, `cloud`, `tinker`, `all`) are *additions* on top of the heavy core, not subsets of it. There is no `harbor[trajectories-only]` extra.
- **Transitive surface area:** `litellm` alone pulls `openai`, `anthropic`, `tiktoken`, `aiohttp`, `boto3` (variable). `datasets` pulls `pyarrow`, `pandas`, `huggingface-hub`. `fastapi` pulls `starlette`, `pydantic-core` (already wanted). `supabase` pulls `gotrue`, `postgrest`, `realtime`, `storage3`. Total transitive count is realistically 80–120 packages and ~150–250 MB on a fresh `uv sync`.
- **Security flag:** `litellm` 1.82.7 and 1.82.8 were quarantined by PyPI on 2026-03-24 due to a malicious `.pth` site-packages file that exfiltrated SSH keys, cloud tokens, and crypto wallets. Harbor's `>=1.80.8` constraint *would have* allowed these versions through. `harbor-framework/harbor#1265` documents the incident; the bound has not been tightened in the public pyproject.toml as of this research. **Adopting Harbor today re-exposes daydream to whichever litellm version is current at install time, with no contractual minimum-safety floor.**
- **Conclusion (HIGH confidence):** Do not depend on the `harbor` PyPI package. Vendor the trajectory submodule + validator. Both are pure-Pydantic, ~700 LOC, Apache-2.0 (verify license at vendor time and add an attribution NOTICE).

### 2. claude-agent-sdk 0.1.52 token extraction — fully available

**Verified against:** `https://code.claude.com/docs/en/agent-sdk/cost-tracking`, `https://code.claude.com/docs/en/agent-sdk/python`, and `https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/types.py`. Cross-referenced with `claude-agent-sdk-python#673`.

**`ResultMessage` (cumulative, end of `query()` call):**
```python
@dataclass
class ResultMessage:
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    stop_reason: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None         # ← cumulative session usage
    result: str | None = None
    structured_output: Any = None
    model_usage: dict[str, Any] | None = None   # ← per-model breakdown (camelCase keys)
    permission_denials: list[Any] | None = None
    errors: list[str] | None = None
    uuid: str | None = None
```

**`ResultMessage.usage` keys (snake_case):** `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`. Plus `service_tier` and a `cache_creation` sub-object (ephemeral token breakdown). `total_cost_usd` is a sibling float on `ResultMessage` itself.

**`ResultMessage.model_usage` keys (camelCase, per model):** `inputTokens`, `outputTokens`, `cacheReadInputTokens`, `cacheCreationInputTokens`, `webSearchRequests`, `costUSD`, `contextWindow`, `maxOutputTokens`. (Camel because passed-through unmodified from the underlying Claude Code CLI.)

**`AssistantMessage` (per-step, during streaming):**
```python
@dataclass
class AssistantMessage:
    content: list[ContentBlock]
    model: str
    parent_tool_use_id: str | None = None
    error: AssistantMessageError | None = None
    usage: dict[str, Any] | None = None       # ← per-step usage (same keys as ResultMessage.usage)
    message_id: str | None = None
    stop_reason: str | None = None
    session_id: str | None = None
    uuid: str | None = None
```

**`UserMessage`:** No `usage` field. (Tool results are nested via `ToolResultBlock.content`; users don't consume tokens.)

**Critical implications for daydream:**

- `daydream/backends/claude.py` lines 120–128 currently emit `CostEvent(cost_usd=msg.total_cost_usd, input_tokens=None, output_tokens=None)`. The data is right there in `msg.usage` — the `None`s are a bug. ATIF migration should fix this in the same PR that adds trajectory recording.
- For ATIF `Metrics(prompt_tokens, completion_tokens, cached_tokens, cost_usd)`:
  - `prompt_tokens` ← `msg.usage["input_tokens"]`
  - `completion_tokens` ← `msg.usage["output_tokens"]`
  - `cached_tokens` ← `msg.usage.get("cache_read_input_tokens", 0)` (ATIF's `cached_tokens` semantically means cache-read)
  - `cost_usd` ← `msg.total_cost_usd`
- For per-step `Metrics` (richer than current `CostEvent`): wire `AssistantMessage.usage` into a new `AgentEvent` (or extend `CostEvent` with `step_id` / `message_id` correlation). Note: parallel tool calls share one `message_id` and **must be deduplicated** to avoid double-counting (per Anthropic's cost-tracking guide).
- For ATIF `FinalMetrics`: read `ResultMessage.usage` cumulative dict + `total_cost_usd`. `model_usage` (camelCase) is ATIF-`extra` material if you want per-model accounting in the trajectory.

**Confidence: HIGH** — verified against official SDK source, cost-tracking guide, and one related issue thread.

### 3. Codex CLI token extraction — partial parity

**Verified against:** `daydream/backends/codex.py` lines 308–314 + `tests/fixtures/codex_jsonl/turn_completed_result.jsonl`, `structured_output.jsonl`, `tool_use.jsonl`.

**Codex `turn.completed` event shape (from fixtures):**
```jsonl
{"type":"turn.completed","usage":{"input_tokens":200,"output_tokens":90}}
```

The codex CLI emits **only `input_tokens` and `output_tokens`**. There is no `cost_usd` field, no `cache_*` field, no per-model breakdown. This is consistent with how the existing `CodexBackend` already emits `CostEvent(cost_usd=None, input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"))`.

**ATIF `Metrics` field mapping for Codex:**

| ATIF field | Codex source | Status |
|------------|--------------|--------|
| `prompt_tokens` | `usage.input_tokens` | Available |
| `completion_tokens` | `usage.output_tokens` | Available |
| `cached_tokens` | n/a | `None` — Codex CLI doesn't expose cache hits |
| `cost_usd` | n/a | `None` — Codex CLI doesn't compute client-side cost |

**Recommendation:** Accept Codex parity at "input/output tokens only." All Pydantic Metrics fields are optional in ATIF v1.4+; emit them as `None` (or omit via `exclude_none=True`). Add a `extra={"backend": "codex"}` annotation at the trajectory `Agent` level so consumers can interpret missing cost fields correctly.

**Confidence: HIGH** — verified against actual Codex JSONL fixtures in this repo.

### 4. UUID + timestamp libs — stdlib is sufficient

**Verified against:** Harbor golden trajectories (e.g. `tests/golden/terminus_2/hello-world-context-summarization.trajectory.json` uses `"session_id": "..."` plus shorthand `01H...`-shaped IDs in some examples; the ATIF spec example in `docs/reference/atif_format.md` uses `"025B810F-B3A2-4C67-93C0-FE7A142A947A"` — UUIDv4 form).

- `uuid.uuid4()` produces 36-char hex strings — interop-safe with every Harbor consumer.
- `datetime.now(timezone.utc).isoformat()` produces `2026-04-26T14:03:22.123456+00:00`. ATIF goldens prefer the trailing-`Z` form: `.isoformat().replace("+00:00", "Z")` or `strftime("%Y-%m-%dT%H:%M:%S.%fZ")` — pick one and apply consistently.
- `ulid-py` / `python-ulid` not justified — ATIF doesn't require lexicographic ordering; UUIDs are universally accepted.

**Confidence: HIGH.**

### 5. Alternative ATIF libraries — none mature

**Searched:** PyPI for `atif`, `atif-py`, `atif-format`. Searched general web for "atif python pydantic". Searched Phoenix (Arize) since they recently added ATIF ingestion.

- **No `atif`-named PyPI package exists** as of 2026-04-26. PyPI returns 404 for `pypi.org/simple/atif/`.
- **Arize Phoenix consumes ATIF** but does not publish a separate emitter library — they ingest via `arize-phoenix-client>=2.3.0` using `upload_atif_trajectories_as_spans()`, which expects you to build the trajectory yourself.
- **Harbor is the only first-party emitter.** Every other Harbor-integrated agent (Terminus-2, OpenHands, Mini-SWE-Agent, Gemini CLI, Claude Code, Codex) uses `harbor.models.trajectories` directly, either as a dependency or by vendoring.
- **Conclusion:** vendoring Harbor's submodule is the recommended fallback (it's also what OpenHands does — see the `populate_context_post_run` example in `docs/reference/atif_format.md`).

**Confidence: HIGH** for "no alternative library exists"; **MEDIUM** for "vendoring is what most projects do" (OpenHands is the only verified example).

### 6. Pydantic version compatibility — clean

- Daydream's transitive Pydantic comes from `claude-agent-sdk==0.1.52` and `mcp==1.26.0`, both v2. No v1 anywhere.
- Harbor's models import `BaseModel`, `Field`, `model_validator`, `field_validator`, `Literal` — all pydantic v2 idioms.
- Harbor pins `pydantic>=2.11.7`. Verify `uv lock`'s current resolution, but daydream's transitive should already be ≥ 2.11.7 given how recent the SDK is.
- **Recommendation:** Promote `pydantic>=2.11.7` from transitive to explicit in `pyproject.toml` so it's controlled at daydream's level. (Implicit-via-transitive risks silent downgrade if `claude-agent-sdk` ever loosens its bound.)

**Confidence: HIGH.**

### 7. Test fixtures — Harbor's `tests/golden/` is the corpus

**Verified against:** `https://github.com/laude-institute/harbor/tree/main/tests/golden`.

**Available golden trajectories (verbatim listings):**

`tests/golden/terminus_2/` (13 JSON files):
- `hello-world-context-summarization-linear-history.trajectory.cont-1.json`
- `hello-world-context-summarization-linear-history.trajectory.json`
- `hello-world-context-summarization.trajectory.json`
- `hello-world-context-summarization.trajectory.summarization-1-{answers,questions,summary}.json`
- `hello-world-invalid-json.trajectory.json` (negative test)
- `hello-world-timeout.trajectory.json`
- (plus several `*.traces.json` files which are intermediate, not ATIF)

`tests/golden/openhands/` (4 JSON files):
- `hello-world.trajectory.json`
- `hello-world.trajectory.no_function_calling.json`
- (plus 2 `*.traces.json`)

**Schema versions in goldens:** Mixed. The OpenHands `hello-world.trajectory.json` declares `"schema_version": "ATIF-v1.5"`. The Terminus-2 `hello-world-context-summarization.trajectory.json` declares `"schema_version": "ATIF-v1.6"`. None are on v1.4. **This is the strongest signal that the v1.4 pin in PROJECT.md is stale.**

**Recommended use:** Copy the `*.trajectory.json` files into `tests/fixtures/atif_golden/` and add a parametrized test:

```python
@pytest.mark.parametrize("path", sorted(GOLDEN_DIR.rglob("*.trajectory*.json")))
def test_validator_accepts_harbor_goldens(path: Path) -> None:
    from daydream.atif.validator import validate_trajectory
    assert validate_trajectory(json.loads(path.read_text())) is True
```

This catches drift if the vendored models ever fall behind upstream.

**Confidence: HIGH.**

---

## Architectural Recommendations (for the roadmap)

Out of scope for STACK.md but the research turned these up; the roadmap should consider them:

1. **Fix the existing `CostEvent` data loss bug as part of the migration.** `daydream/backends/claude.py` lines 124–128 throw away `msg.usage`. Change in the same PR that adds trajectory recording — there's no separate phase for it, and not fixing it leaves the trajectory's `Metrics` blank.
2. **Add a `MetricsEvent` (or extend `CostEvent`) carrying per-step usage** indexed by `message_id`, so the trajectory recorder can attach `Metrics` to the correct `Step`. The current single end-of-call `CostEvent` is too coarse for ATIF's per-step model.
3. **Schema version: revisit the v1.4 pin.** The downstream consumer of this research may want to recommend ATIF-v1.6 (Harbor's current default). v1.5 added system-step observation extensions and extended `extra` semantics; v1.6 added `ContentPart`/`ImageSource` for multi-part content. Daydream doesn't emit images today, but the `Step.message` field's interpretation differs slightly across versions. Concrete delta to investigate: `Step` field `tool_definitions` (v1.5+, OpenHands uses it) is nice-to-have for replayability and is a one-liner to populate from `ClaudeAgentOptions.allowed_tools`.
4. **Vendor Harbor's `LICENSE` and `NOTICE` files** alongside the vendored module. Apache-2.0 requires attribution.
5. **Pin a vendor source tag** (e.g., `harbor==0.5.0` git tag) in a `daydream/atif/_VENDOR_SOURCE.md` file. Re-syncs become controlled, reviewable PRs rather than ambient drift.

---

## Stack Patterns by Variant

**If we *must* use Harbor as a real PyPI dep (e.g., the team standardizes on it elsewhere):**
- Constrain `litellm<1.82.7` explicitly in daydream's pyproject as defense-in-depth, even though Harbor doesn't.
- Use `uv lock --upgrade-package litellm` cadence carefully; treat Harbor like a security-relevant dep.
- Accept ~50MB+ install bloat.

**If the team revisits the v1.4 pin and adopts v1.6:**
- Vendor `harbor.models.trajectories.content` (`ContentPart`, `ImageSource`) — 11 files, not 9.
- Step `message` becomes `str | list[ContentPart]`. Daydream emits only text today, so `str` form remains the default emission.

**If `pydantic` ever drops v2.11 → v3 in a breaking way:**
- Vendored models would need a re-sync from upstream Harbor (which would have done the v3 migration first).
- Pin `pydantic<3` until verified.

---

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `pydantic>=2.11.7` | `claude-agent-sdk==0.1.52`, `mcp==1.26.0` | All v2; SDK uses `dataclass` not pydantic for its own types, so no version coupling beyond transitive. |
| Vendored Harbor models @ tag `0.5.0` | `pydantic>=2.11.7,<3` | Confirm at vendor time; re-sync if upstream bumps min pydantic. |
| ATIF schema version | Pydantic model `Literal` | Harbor's current model accepts `ATIF-v1.0` through `ATIF-v1.6`. Daydream picks one default; setting a default of `"ATIF-v1.4"` is valid but trails the spec. |
| `claude-agent-sdk==0.1.52` | `ResultMessage.usage` keys | All four cache keys (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`) verified present at this version. |

---

## Sources

- [Harbor pyproject.toml on `laude-institute/harbor`](https://raw.githubusercontent.com/laude-institute/harbor/main/pyproject.toml) — verified 21 core deps, no slim extra; HIGH confidence.
- [Harbor pyproject.toml on `harbor-framework/harbor`](https://raw.githubusercontent.com/harbor-framework/harbor/main/pyproject.toml) — identical to laude-institute mirror; HIGH confidence.
- [Harbor 0.5.0 on PyPI](https://pypi.org/project/harbor/) — version 0.5.0 (2026-04-23), 1.05 MB wheel; HIGH confidence.
- [Harbor `models/trajectories/__init__.py`](https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/models/trajectories/__init__.py) — 11 model classes exported; HIGH confidence.
- [Harbor `models/trajectories/trajectory.py`](https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/models/trajectories/trajectory.py) — schema_version `Literal` accepts `ATIF-v1.0` through `ATIF-v1.6`, defaults to `ATIF-v1.6`; HIGH confidence.
- [Harbor `models/trajectories/step.py`](https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/models/trajectories/step.py) — imports only pydantic + datetime + sibling submodules; HIGH confidence.
- [Harbor `utils/trajectory_validator.py`](https://raw.githubusercontent.com/laude-institute/harbor/main/src/harbor/utils/trajectory_validator.py) — only depends on `pydantic.ValidationError` + Harbor models; HIGH confidence.
- [Harbor `tests/golden/terminus_2/`](https://github.com/laude-institute/harbor/tree/main/tests/golden/terminus_2) — 13 trajectory JSONs, ATIF-v1.6; HIGH confidence.
- [Harbor `tests/golden/openhands/`](https://github.com/laude-institute/harbor/tree/main/tests/golden/openhands) — 4 trajectory JSONs, ATIF-v1.5; HIGH confidence.
- [Harbor releases page](https://github.com/harbor-framework/harbor/releases) — v0.5.0 latest, v0.4.0 moved sandbox deps to optional; MEDIUM confidence on release-note interpretation.
- [Harbor issue #1265 (litellm supply chain)](https://github.com/harbor-framework/harbor/issues/1265) — `litellm` 1.82.7/1.82.8 quarantined March 2026, malicious `.pth`; HIGH confidence on incident, MEDIUM on whether Harbor's lower bound has been tightened (not visible in current pyproject).
- [Claude Agent SDK Python — Cost Tracking](https://code.claude.com/docs/en/agent-sdk/cost-tracking) — official `usage` dict key list, per-step vs cumulative semantics, parallel-tool dedup warning; HIGH confidence.
- [Claude Agent SDK Python — Reference](https://code.claude.com/docs/en/agent-sdk/python) — `ResultMessage` and `AssistantMessage` field listings; HIGH confidence.
- [`claude-agent-sdk-python` types.py at HEAD](https://github.com/anthropics/claude-agent-sdk-python/blob/main/src/claude_agent_sdk/types.py) — full `ResultMessage` dataclass shape; HIGH confidence.
- [`claude-agent-sdk-python` issue #673](https://github.com/anthropics/claude-agent-sdk-python/issues/673) — corroborates `cache_creation_input_tokens` / `cache_read_input_tokens` keys present in `ResultMessage.usage`; HIGH confidence.
- Local repo: `daydream/backends/claude.py` lines 120–128 — confirmed `CostEvent` is emitted with `input_tokens=None, output_tokens=None`; HIGH confidence (read directly).
- Local repo: `daydream/backends/codex.py` lines 308–314 — confirmed Codex emits only `input_tokens` / `output_tokens` from `turn.completed.usage`; HIGH confidence.
- Local repo: `tests/fixtures/codex_jsonl/turn_completed_result.jsonl`, `structured_output.jsonl`, `tool_use.jsonl` — confirmed Codex JSONL `usage` shape; HIGH confidence.

---
*Stack research for: ATIF v1.x trajectory emission from daydream (Python 3.12 CLI on `claude-agent-sdk==0.1.52`)*
*Researched: 2026-04-26*
