# Observability Field Contract Matrix

**Versioned field contract for Daydream's emitted OpenTelemetry trace fields.**

- Schema/contract version `contract_version` = `94f432d` (resource attribute
  `daydream.observability.contract.version`).
- Semantic convention producer pin: open-telemetry/semantic-conventions-genai
  commit `94f432d7126f5884d30a2cdde6f4e89908ebb6fd` (registry + 4 content
  schemas pinned byte-for-byte in `tests/fixtures/observability_contract/manifest.json`).
- Dependency pins: OTel API/SDK/HTTP/gRPC/proto 1.44.0, traceloop 0.62.3,
  semconv-ai 0.5.1, HTTPX 0.28.1, AnyIO 4.14.2, Pi 0.85.1, claude-agent-sdk
  0.2.147.
- Verification commands: representative-real (`representative_real_run` receipts)
  and sanitized-replay (`sanitized_protocol_replay`) public readback use separate
  commands below. Many vendor/limitation rows are per-backend; Claude, Codex,
  and Osprey attempts stay structural INTERNAL `invoke_agent` scopes — only Pi
  creates generation model spans, and no backend writes private HoneyHive
  underscore fields. Readback output contains no private HoneyHive fields.
- Document version: this file is the authoritative expanded matrix; the
  machine-readable subset for readback is
  `tests/fixtures/observability_contract/readback-matrix.json`, which may only
  shrink relative to this document.

Every row below states: field name, source backend/event, source
authority/provenance, owning span, type, unit, cardinality, derivation,
completeness, capture/redaction, generic OTLP disposition, HoneyHive canonical
destination, LangSmith native destination, offline test node, live evidence
status, and applicability and omission reason. Rows are grouped by family;
absent values stay absent, known zero values stay zero, and no value is
invented (binding decisions 1–8, P18 plan §239–262).

## Resource attributes (one per TraceSession)

Owner: `TraceSession._build_resource` (`daydream/observability/runtime.py`)
from declared sources only; operator `OTEL_RESOURCE_ATTRIBUTES` is parsed
strictly and sanitized (`parse_operator_resource_attributes` /
`sanitize_operator_resource_attributes` in `daydream/observability/privacy.py`);
`Resource.create` is never used so ambient detectors cannot re-enter.
Resource precedence: reserved application/SDK identity overlays every operator
collision except a valid operator `service.instance.id`, which stays
authoritative.

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `service.name` | All backends, session resource | `config.service_name` (`OTEL_SERVICE_NAME`, default `daydream`); reserved overlay | Resource | string | – | 1 | Session config | Always | Secret scrub | Resource | Preserved as resource/metadata | Preserved as resource/metadata | `test_observability_runtime.py` (owned session + reserved overlay) | pending (native readback T7) | Applicable; operator collisions overwritten |
| `service.version` | All, resource | `version("daydream")` distribution | Resource | string | – | 1 | Installed package metadata | Always | Secret scrub | Resource | Preserved | Preserved | same | pending | Applicable |
| `service.instance.id` | All, resource | Operator value wins; else `TraceSession.run_id` UUID once per session | Resource | string | – | 1 | Env precedence + UUID generation | Always | Secret scrub | Resource | Preserved | Preserved | `test_observability_runtime.py` operator-wins + UUID-once | pending | Applicable |
| `telemetry.sdk.name` | All, resource | reserved `"opentelemetry"` | Resource | string | – | 1 | Constant | Always | – | Resource | Preserved | Preserved | same | pending | Applicable |
| `telemetry.sdk.language` | All, resource | reserved `"python"` | Resource | string | – | 1 | Constant | Always | – | Resource | Preserved | Preserved | same | pending | Applicable |
| `telemetry.sdk.version` | All, resource | pinned OTel SDK version | Resource | string | – | 1 | Installed metadata | Always | – | Resource | Preserved | Preserved | same | pending | Applicable |
| `daydream.observability.contract.version` | All, resource | constant `94f432d` | Resource | string | – | 1 | Constant | Always | – | Resource | Preserved | Preserved | `test_observability_semconv_contract.py` + runtime suite | pending | Applicable |
| Operator resource keys (sanitized) | All, resource | `OTEL_RESOURCE_ATTRIBUTES` strict env parse | Resource | string | – | 0..N | Env | Only valid pairs/values | Secret-like keys → `[REDACTED_CREDENTIAL]`; secret literals scrubbed | Resource | Preserved | Preserved | `test_observability_runtime.py` strict parser (9 cases) | pending | Conditional; whole variable rejected on any malformed/duplicate pair |
| `daydream.acceptance.kind=sanitized_protocol_replay` | sanitized replay only | replay tool resource marker | Resource | string | – | 0..1 | Operator env marker | Replay only | – | Resource | Preserved | Preserved | `test_observability_readback.py` hermetic replay | verified (hermetic wire) | Replay-only admission marker |

## Instrumentation scope

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scope name `daydream` + installed package version | All | `get_tracer("daydream", version("daydream"))` | All spans | string | – | 1 | Constant | Always | – | OTLP scope | Raw/native | Raw/native | `test_observability_otlp_wire.py` | pending | Applicable |
| Schema URL | All | omitted | Scope | – | – | 0 | – | – | – | absent | absent | absent | `test_observability_semconv_contract.py` (registry TODO pin) | pending | Inapplicable: upstream schema URL is TODO at the pinned commit; no fabricated URL |

## Identity and association (every span)

Emitted by `SpanScope`/`TraceSession` common attributes
(`daydream/observability/spans.py` `_scope_attributes`, runtime association).

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `daydream.run.id` | All, session | `TraceSession.run_id` (UUID per run) | All spans | string | – | 1 | Session identity | Always | – | Span attribute | session filter / metadata | Run metadata | `test_observability_integration.py` root identity | pending | Applicable; never substituted for trajectory/response ids |
| `daydream.session.id` | All, trajectory recorder | `associate_run_trajectory(recorder.session_id)` | All spans | string | – | 1 | Trajectory session identity | After trajectory association | – | Span attribute | `honeyhive.session_id` metadata | Trace/thread metadata | same | pending | Applicable |
| `daydream.trajectory.id` | All, trajectory recorder | recorder session id (same trajectory identity) | All spans | string | – | 1 | Trajectory identity | After association | – | Span attribute | metadata | metadata | same | pending | Applicable |
| `daydream.trajectory.descriptor` | All, P07/P10 recorder | recorder descriptor snapshot | All spans | string | – | 0..1 | P07 recorder | When recorder active | – | Span attribute | metadata | metadata | `test_observability_runtime.py` + P07 suites | pending | Conditional |
| `traceloop.association.properties.daydream_run_id` | All | run id alias | All spans | string | – | 1 | Alias | Always | – | Span attribute | metadata | metadata | runtime suite | pending | Compatibility alias |
| `traceloop.association.properties.session_id` | All | session id alias | All spans | string | – | 1 | Alias | After association | – | Span attribute | `metadata.session_id` | metadata | runtime suite + T5 identity asserts | pending | Compatibility alias; distinct from native conversation id |
| `daydream.flow` / `.step` / `.phase` / `.iteration` / `.stack` | All, run/step scopes | actual runner flow/step/phase/iteration/stack | run + step + agent + attempt + tool (+children) via `_scope_attributes` | string / int | – | 1 | Actual runner scopes | Always on scopes | – | Span attributes | metadata/event classification | metadata | `test_runner_exports_complete_portable_trace` | pending | Applicable; low cardinality |
| `daydream.span.kind` | All | constant per scope: run/step/agent/attempt/generation/tool | Every span | string | – | 1 | Scope kind | Always | – | Span attribute | `honeyhive_event_type` mapping | `langsmith.span.kind` mapping | `test_vendor_chain_model_tool_classification` | verified on wire (T4) | Applicable; classification authority |
| `daydream.attempt` | All, attempt scope | attempt number (1-based) | attempt | int | – | 1 | Actual attempt counter | Always | – | Span attribute | metadata | metadata | T5 lifecycle suite | verified | Applicable |
| `daydream.invocation.aggregate` | All, attempt scope | constant True on invocation aggregates | attempt | bool | – | 1 | Scope kind | Always | – | Span attribute | metadata | `langsmith.metadata.invocation_aggregate` | `test_langsmith_mapping_preserves_original_spans_and_only_attempts_own_usage` | verified | Applicable |

## Agent/operation identity

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gen_ai.operation.name=invoke_agent` | All, agent/attempt scopes | actual `run_agent` boundary | agent + attempt | string (registry enum) | – | 1 | Constant per scope | Always | – | Span attribute | metadata | Run type mapping | semconv contract test + T5 | verified | Applicable; structural attempts stay INTERNAL `invoke_agent` |
| `gen_ai.agent.name` | All, agent/attempt scopes | actual `run_agent` phase name | agent + attempt | string | – | 1 | Phase name | Always | – | Span attribute | `metadata.agent_name` | Run name/metadata | `test_honeyhive_agent_name_uses_standard_attribute_no_underscore_fields` | verified | Applicable; standard public identity, no underscore-private fields |
| `daydream.agent.name` | All, agent/attempt scopes | phase name (own spelling) | agent + attempt | string | – | 1 | Phase name | Always | – | Span attribute | metadata | metadata | T5 | verified | Applicable |
| `daydream.agent.role` | All, nested logical agents | enclosing logical-agent depth: root/subagent | agent scopes | string | – | 0..1 | Actual enclosing-scope relationship | On logical agent scopes | – | Span attribute | metadata | `langsmith.metadata.ls_agent_type` | `test_actual_nested_agent_scope_is_subagent_siblings_are_not` | verified | Conditional; only actual root/subagent scopes |
| `daydream.backend` | All, agent scope | actual backend kind | agent | string | – | 1 | Backend identity | Always | – | Span attribute | config/metadata | metadata | T5 | verified | Applicable |
| `daydream.configured.model` | All, agent scope | configured model (not observed) | agent | string | – | 1 | Config | Always | – | Span attribute | metadata | `invocation_params` | `test_configured_model_preserved` style | verified | Applicable; configured ≠ observed. Aggregate rule: `model_mode=single` emits the standard request model; `multi_or_dynamic` (e.g. Claude with nonempty agents) keeps only the Daydream configured-model provenance and never claims one standard model |

## Effective request configuration (Task 1 contract)

Emitted from closed typed `EffectiveRequestConfig` facts only; every admitted
parameter has exact bounds (`daydream/backends/*.py`, request events).

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gen_ai.request.model` | All, RequestEvent | exact request model (single mode only) | attempt | string | – | 1 | RequestEvent.model_name | single-model mode | – | Span attribute | `config.model` | `invocation_params` | `test_observability_backend_protocols.py` single-model cases | verified | Conditional: `model_mode=single`; multi/dynamic keeps only `daydream.configured.model` |
| `gen_ai.request.reasoning_effort` | All, RequestEvent | exact request reasoning effort | attempt | string (enum) | – | 0..1 | RequestEvent.reasoning_effort | When supplied | – | Span attribute | `config.reasoning_effort` | `invocation_params` | T1/T5 | verified | Compatibility alias for `gen_ai.request.reasoning.level`; unknown values omitted |
| `daydream.request.timestamp` | All, RequestEvent | exact request timestamp | attempt | string | – | 0..1 | RequestEvent.timestamp | When supplied | – | Span attribute | metadata | metadata | T5 | verified | Conditional |
| `daydream.output_mode` | All | output mode constant | attempt | string | – | 0..1 | Request event | When known | – | Span attribute | metadata | metadata | T1 | verified | Conditional |
| `daydream.backend.config.*` (bounded) | Per-backend effective config | exact argv/option facts per backend admission contract | attempt | bool/int/string/enum | – | 0..N | Effective config table (Task 1) | per admission table | Secret/path-free | Span attributes | metadata | `invocation_params` | `test_observability_config.py` + backend protocol suites | verified | Conditional; unsupported identities omitted with fixed diagnostics, never truncated aliases |

## Generation lifecycle (pi native interval; only real generation model spans)

`AttemptObserver._seal_generation` + `_end_generations` (`spans.py:566-714`),
clocked by the manifest-pinned or live host receipt, sealed at `message_end`
before tools, SDK-end exactly once after billing-owner resolution.

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `daydream.span.kind=generation` + `traceloop.span.kind=llm` + `gen_ai.operation.name=chat` | Pi GenerationStart/End, sealed child | sealed provider generation boundary | generation child (CLIENT) | string | – | 1 | Sealed lifecycle | Per sealed generation | – | Span kind/attrs | `honeyhive_event_type=model` | `run_type=llm` | `test_honeyhive_generation_child_is_model_and_attempt_stays_chain` | verified | Pi-only; opaque backends create none |
| `daydream.generation.id` | Pi GenerationEndEvent | invocation-local generation identity (not provider response id) | generation | string | – | 1 | Sealed event | Always | – | Span attribute | metadata | metadata | `test_pi_replay_exact_native_timing_choice_and_billing_through_runner` | verified | Applicable; never substituted for `gen_ai.response.id` |
| `daydream.generation.boundary_complete` | GenerationEndEvent | sealed boundary flag | generation | bool | – | 1 | Sealed event | Always | – | Span attribute | metadata | metadata | T5 | verified | Applicable |
| `daydream.generation.end_source` | GenerationEndEvent | host/observed provenance | generation | string | – | 1 | Sealed event | Always | – | Span attribute | metadata | metadata | T5 | verified | Applicable |
| `daydream.generation.native_started_at_unix_ms` | Pi message_end.timestamp | native provider start, strict int ms | generation | int (non-bool) | ms | 1 | Exact conversion `ms*1_000_000` → ns | Always when sealed | – | Span attribute | metadata | metadata | `test_native_timestamp_validator_bounds` + T5 | verified | Applicable; bool/out-of-range rejected, no clamping |
| `daydream.generation.native_started_at_unix_ns` | derived | `ms * 1_000_000` | generation | int | ns | 1 | Multiplication-only | Always | – | Span attribute | metadata | metadata | native-ms contract test | verified | Applicable; malformed/missing → explicit incomplete disposition |
| `daydream.generation.sealed_end_unix_ns` | host `message_end` receipt | host-observed completion (not provider latency) | generation | int | ns | 1 | Pinned/live receipt | Always | – | Span attribute | metadata | metadata | 395.332s replay asserts | verified | Applicable; end before tools, never TTFT |
| `daydream.generation.duration_ns` | derived | `sealed_end − native_start` | generation | int | ns | 1 | Exact subtraction | Both endpoints known | – | Span attribute | metadata | metadata | 395332000000 assert | verified | Applicable; reversed/missing → fail-closed fallback, no invented duration |
| `daydream.generation.billed` | billing owner resolution | ledger decision 5 per-draft flag | generation | bool | – | 1 | Frozen owner | Always | – | Span attribute | native usage gate | usage gate | `test_generation_billed_owner_carries_native_usage_only_on_resolved_owner` | verified | Applicable |
| `gen_ai.response.id` | Pi GenerationEndEvent.responseId | exact provider response identity (late arrival allowed) | generation | string | – | 0..1 | Sealed event | When observed and child billed | – | Span attribute | metadata | metadata | T5 late-response cases | verified | Conditional; never derived/guessed |
| `gen_ai.response.model` / `gen_ai.provider.name` / `gen_ai.response.finish_reasons` | sealed generation / terminal result | exact observed identity | generation or attempt (billed owner) | string / string / array(enum) | – | 1 | Observed | When billed child or last-event on attempt | – | Span attributes | metadata/config | Run metadata/invocation | T5 exact values assert | verified | Conditional; opaque backends keep structural-only |
| `daydream.generation.choice_parts` | GenerationEndEvent choice (typed ordered provider parts) | sealed immutable provider choice: text/reasoning/tool-call id/name/arguments in provider order | generation | JSON string | – | 1 | Sealed choice | Full mode only | FULL MODE ONLY — no model-child or transcript content in metadata mode (metadata omits); content-gated like output | Span attribute | metadata (content) | metadata/tool input (content) | `test_pi_metadata_mode_omits_generation_choice_content` | verified | Conditional; full mode only; typed ordered provider-choice parts exist BEFORE tool execution and are never duplicated by later tool spans |
| `gen_ai.output.messages` | sealed generation choice | typed ordered assistant parts | generation | JSON string | – | 1 | Sealed choice | Full mode | FULL MODE ONLY | Span attribute | Native outputs | Native inputs/outputs | same + T5 | verified | Conditional; full mode only |
| `gen_ai.input.messages` (generation children) | – | must stay ABSENT on generation children | generation | – | – | 0 | – | – | – | – | – | – | `assert "gen_ai.input.messages" not in gen_attrs` (T5) + replay gate | verified | Inapplicable: no exact per-request provider history exposed; structural invocation input (prompt/system/schema) lives on the attempt only and is never fabricated as per-request generation history |

## Attempt aggregate, usage, billing (decision 5)

`AttemptObserver` configured reconciliation with the trajectory
pending-generation ledger (`daydream/trajectory.py` T2); exactly one
(single billing) owner per attempt — NO double billing and no parent/child
usage duplicate — usage on the owner only. Accounting is complete/partial:
standard totals are emitted only when complete; known partial custom
measurements are preserved under `daydream.usage.partial.*`; no invented
usage ever.

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `daydream.billing.owner` | closed ledger resolution | `structural_attempt` / `generation_children` / `none` / `unresolved` etc. | attempt | string | – | 1 | Decision-5 matrix | Always | – | Span attribute | native-usage gate | native-usage gate | `TestBillingOwnerResolution` (8 tests) | verified | Applicable; frozen once, never switched; no double billing, no parent/child duplicate |
| `gen_ai.usage.input_tokens` / `.output_tokens` / `.cache_read.input_tokens` / `.cache_creation.input_tokens` / `.reasoning.output_tokens` / `.cost` / `.total_tokens` | Pi/Claude/Osprey/Codex metrics+terminal totals | exact reported measurements, correlated with scope/source/authority/completeness | billed owner | int / int / int / int / int / double / int (derived input+output) | tokens / USD | 1 per key | Observed exact values; totals only when complete | Standard totals only when complete; partial stays custom | – | Span attributes | `honeyhive_metadata.{prompt,completion,cache_read,cache_write,reasoning}_tokens`, `cost` | `langsmith.usage_metadata` | `test_owner_none_and_unbilled_generation_get_no_native_usage` + T4 wire dedup | verified | Conditional; known zeros retained; no invented usage; duplicate totals idempotent; no parent/child duplicate |
| `daydream.usage.partial.*` | partial-only evidence | known partial custom measurements | attempt | int/double | tokens/USD | 0..N | Observed partials | Partial only | – | Span attribute | custom metadata (non-billed) | custom metadata | T2 partial-owner tests | verified | Conditional; no native billing aliases on non-owners |
| `daydream.message_usage` / `daydream.model_usage` | per-message/per-model records | exact per-message/model usage maps | attempt | JSON | tokens/usd/ms | 0..1 | Observed records | When any exist | – | Span attribute | custom metadata | custom metadata | T2/T4 | verified | Conditional |
| `daydream.models` / `daydream.providers` | ordered observed identity list | distinct ordered observed model/provider values | attempt | array(string) | – | 0..2 | Observed lists | When observed | – | Span attributes | metadata | metadata | T1/T5 | verified | Conditional; max 16, overflow omits whole list |
| `daydream.reasoning` | exposed thinking content | joined reasoning parts | attempt | string | – | 0..1 | Observed | Full mode | FULL MODE ONLY | Span attribute | custom metadata (content) | custom metadata (content) | T5 content gating | verified | Conditional; no hidden CoT inference |
| `daydream.invocation.transcript` | trajectory TurnEnd + terminal result | separate trajectory transcript (never provider-choice fields) | attempt | JSON (content) | – | 0..1 | Terminal result | Full mode | FULL MODE ONLY | Span attribute | native outputs | native outputs | T2/T5 | verified | Conditional; provider-choice fields never carry the transcript |

## Tool lifecycle

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gen_ai.operation.name=execute_tool`, `gen_ai.tool.name`, `gen_ai.tool.call.id` | ToolStartEvent | exact tool start identity | tool | string | – | 1 | Observed | Always | – | Span attributes | tool classification | `run_type=tool` | `test_parallel_tools_reuse_ids` style | verified | Applicable; name ASCII≤128 profile |
| `daydream.tool.started_at` / `.completed_at` / `.duration_ms` | ToolStart/Result | exact observed lifecycle timing | tool | string/string/int | ms | 1 | Observed | Always | – | Span attributes | metadata | metadata | `daydream.tool.*` asserts | verified | Applicable; no provider timing invention |
| `daydream.tool.status` / `.exit_code` / `.error` / `.cancelled` / `.truncated` | ToolResultEvent | exact typed tool outcome | tool | string/int/bool | – | 1 | Observed | Always (interrupted tools close explicitly on attempt abort) | – | Span attributes | metadata | metadata | tool-outcome matrix (T2) | verified | Applicable; failed ≠ successful |
| `traceloop.entity.input` / `.output` (tool) | ToolStart/Result | args/result content | tool | JSON | – | 0..1 | Observed | Full mode | FULL MODE ONLY (args/results); status metadata always | Span attribute | tool input/output (content) | tool input/output (content) | T5 content gating | verified | Conditional; full mode only |

## Output/result/status

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `daydream.outcome` / `daydream.exit_code` | scope finish/abort | actual terminal outcome + exit code | agent/tool | string/int | – | 1 | Terminal state | Always | – | Span attribute | `status` | Run `status` | `test_runner_exports_complete_portable_trace` | verified | Applicable; no fabricated stacks |
| `error.type` / `exception.type` / `exception.message` | typed failure events | safe failure type/message | failed span | string | – | 1 | Typed failure | failures | message full-mode only / sanitized | Span attribute/event | native error | Run error/exception | `test_backends_events.py` failure cases | verified | Conditional; exception message full mode only |
| `daydream.error.message` | failure scope | sanitized failure text | failed scope | string | – | 0..1 | Observed | Full mode | FULL MODE ONLY | Span attribute | native error (content) | native error (content) | T5 | verified | Conditional |
| `daydream.backend_diagnostic.codes` / `.counts` | DiagnosticEvent reducer | bounded fixed diagnostic codes + counts | attempt | array(string)/array(int) | – | 0..1 | Observed | When diagnostics observed | Fixed codes only, no raw text | Span attributes | metadata | metadata | T2/T5 diagnostic suites | verified | Conditional; arbitrary backend text never becomes error.type |
| `daydream.duration_ms` / `daydream.duration_api_ms` | ResultEvent | exact terminal durations (whole invocation) | attempt | int | ms | 0..2 | Observed | Terminal only | – | Span attributes | metadata | metadata | T5 | verified | Conditional; per-message durations stay in message_usage |
| `daydream.started_at` | invocation-scope MetricsEvent | exact request start timestamp | attempt | string | – | 0..1 | Observed | When supplied | – | Span attribute | metadata | metadata | T5 | verified | Conditional |

## Owned OpenLLMetry compatibility (Task 3)

Only Daydream-owned context; ambient Traceloop callbacks/metadata never leak;
normal and notebook (IPython) environments share the same owned
BatchSpanProcessor path with zero global provider/instrumentor signals.

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `traceloop.workflow.name` / `traceloop.entity.name` / `traceloop.entity.path` / `traceloop.entity.version` | owned scope attributes | derived solely from current Daydream scope/session | run/agent/tool | string | – | 1 | Owned aliases | Always | privacy-validated | Span attributes | supported vendor metadata | supported vendor metadata | `test_owned_session_installs_no_global_signals_or_instrumentors` + `test_notebook_mode_uses_same_owned_batch_path_and_metadata` | verified | Applicable; ambient callback state ignored |
| `traceloop.entity.input` / `.output` (attempt, content) | Request/Result | invocation prompt/system/schema + structured/final output | attempt | JSON | – | 0..1 | Observed | Full mode | FULL MODE ONLY | Span attribute | native inputs/outputs | native inputs/outputs | `test_runner_metadata_policy_preserves_structure_and_omits_content` | verified | Conditional; metadata mode absent |
| `gen_ai.system_instructions` | supplied system parts | separate supplied invocation system parts | attempt | JSON (content) | – | 0..1 | Observed | Full mode | FULL MODE ONLY | Span attribute | native system (content) | native system (content) | T5 | verified | Conditional; never copied to generation children |

## Destination-only compatibility (Task 4)

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `honeyhive_event_type` | exporter clone | `daydream.span.kind` → chain/model/tool | per-span clone | string | – | 1 | Public mapping | Always | – | vendor-only clone | event classification | – | `test_vendor_chain_model_tool_classification` | verified | HoneyHive-only |
| `honeyhive.session_id` / `.session_auto_create` / `.session_name` | exporter clone | Daydream run/trajectory identity only | per-span clone | string/bool/string | – | 1 | Daydream identity, never native conversation id | Always | – | vendor-only clone | native session fields | – | `test_honeyhive_session_derives_only_from_daydream_identity` | verified | HoneyHive-only |
| `honeyhive_metadata.*` (prompt/completion/cache/reasoning/cost) | exporter clone | billed owner only (decision 5) | billed owner | int/double | tokens/USD | per key | Public mapping | Closed owner only | – | vendor-only clone | native metadata | – | T4 wired-owner tests | verified | HoneyHive-only; non-owners get none |
| `langsmith.span.kind` | exporter clone | chain/llm/tool mapping, no fake agent type | per-span clone | string | – | 1 | Public mapping | Always | – | vendor-only clone | – | native run type | `test_vendor_chain_model_tool_classification` | verified | LangSmith-only |
| `langsmith.metadata.ls_agent_type` | exporter clone | actual root/subagent logical scopes only | agent clone | string | – | 0..1 | Enclosing scope role | Actual agent scopes | – | vendor-only clone | – | Messages view control | `test_langsmith_ls_agent_type_only_on_actual_agent_scopes` | verified | Conditional; never retry aggregates/structural spans |
| `langsmith.metadata.invocation_aggregate` | exporter clone | structural billed attempt only | attempt clone | bool | – | 1 | Owner gate | Owner = structural_attempt | – | vendor-only clone | – | aggregation metadata | T4 owner tests | verified | Conditional |
| `langsmith.usage_metadata` | exporter clone | `_langsmith_usage` from billed owner attributes | billed owner | JSON | tokens/cost | 1 | Public mapping | Closed owner | – | vendor-only clone | – | native usage_metadata | T4 | verified | Conditional |

## Delivery/diagnostics/resource state

| field name | source backend/event | source authority/provenance | owning span | type | unit | cardinality | derivation | completeness | capture/redaction | generic OTLP disposition | HoneyHive canonical destination | LangSmith native destination | offline test node | live evidence status | applicability and omission reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| queued/accepted/rejected/unverified counts + fixed dispositions | DeliveryLedger / exporter lifecycle | decoded bounded acknowledgments; `force_flush=True` is NOT acceptance | session-local | int + fixed codes | – | per destination | Transport lifecycle | Real lifecycle | no headers/body/endpoints/credentials | local receipt only | local receipt only | local receipt only | `test_observability_otlp_delivery.py` + T4A suite | verified (local) | Applicable; partial success never retried; UI/API proof separate |
| OTLP HTTP canonical full-success ack | otlp_compat transport | HTTP 200 + `application/x-protobuf` content type + complete empty body `b""` | – | status | – | 1 | Binding decision 8 | Full success only | – | transport verdict | – | – | T4A zero-byte ack tests | verified (local) | Applicable; wrong/missing content type, nonempty decode failure, oversized/truncated are terminal/unverified |
| gRPC bridge ownership | otlp_compat `GrpcBridge` | the pinned 1.44 channel/stub bridge touches exactly `_client, _channel, _headers, _timeout, _shutdown, _initialize_channel_and_stub`; one owned `Export(...)` per retry | – | – | – | 1 | Task 0 frozen surface | – | – | transport | – | – | T4A gRPC guards | verified (local) | Applicable; no delegated exporter retry loops. gRPC RESOURCE_EXHAUSTED retries only with valid RetryInfo; a valid zero-byte protobuf success (`HTTP 200` + protobuf content type + complete empty body) is distinct from a gRPC bridge ack; both are documented success paths |
| private credential-provider rejection | otlp_compat | OTel HTTP private requests.Session credential-provider path fails closed BEFORE client/send with one fixed sanitized diagnostic | – | – | – | 0..1 | Reviewed fixed behavior | – | fixed diagnostic only | transport | – | – | T4A pre-send rejection | verified (local) | Applicable; never an unsafe fallback |

## Explicit unavailable/inapplicable rows per backend

The four backends (Claude, Codex, Pi, Osprey) have distinct limitation rows
below — no value is invented and no field below is emitted on the named
backend. Every absence has this source and disposition. In particular,
claude/codex/osprey attempts stay structural INTERNAL `invoke_agent` scopes;
only Pi creates generation model spans.

| Field | Claude | Codex | Pi | Osprey | Disposition / reason |
| --- | --- | --- | --- | --- | --- |
| agent id/description/version (`gen_ai.agent.id/.description/.version`) | absent | absent | absent | absent | No stable hosted agent identity exists; no transient/synthetic ID (decision 6). Only `gen_ai.agent.name` (phase) is emitted. |
| Provider response ID (`gen_ai.response.id`) | absent | absent | present (sealed) | absent | Only Pi exposes `responseId`; others have no observable provider response identity. Never guessed. |
| Request token/top-p/top-k/penalty/seed/stop/choice controls | absent | absent | absent | absent | Not exposed by the public invocation surface; only bounded effective config facts are emitted. |
| Provider endpoint / base URL | absent | absent | absent | absent | Private path/base-URL values are never emitted (`Effective Configuration Admission Contract`). |
| Hidden system prompts / tool definitions | absent | absent | absent | absent | Proprietary CLI internals; not exposed by the stream. Structural invocation input only. |
| Unobserved timestamps | absent | absent | absent | absent | No synthetic start/end; missing/invalid evidence gets explicit incomplete disposition, never clamped (decision 4). |
| TTFT / queue / prefill / decoding / reasoning duration | absent | absent | absent | absent | Not observable; duration is sealed end − native start only (Pi), never provider latency. |
| Modalities | absent | absent | absent | absent | Not exposed by the invocation surface. |
| Links and model spans on opaque backends | absent | absent | n/a | absent | Only Pi creates generation model spans; Claude/Codex/Osprey attempts stay structural `invoke_agent` (INTERNAL). |
| Per-generation invocation history (`gen_ai.input.messages` on children) | absent | absent | absent | absent | No exact per-request provider history exposed; invocation prompt is structural on the attempt, never copied to children (decision 3). |
| Osprey persona/toolset labels (arbitrary) | n/a | n/a | n/a | absent | Only presence booleans/counts are eligible; arbitrary labels/paths/variables/env omitted. |
| Claude settings sources / config paths | absent | n/a | n/a | n/a | Enum/bool/int + presence/count only; names/paths absent. |
| Codex cwd/schema path/stdin/environment | n/a | absent | n/a | n/a | Never emitted; bounded facts only. |
| Pi system/prompt content in metadata mode | n/a | n/a | absent | n/a | Content is full-mode-only; metadata mode keeps identity/counts/timing. |

## Compatibility aliases and removal policy

**Alias table** (kept for compatibility; canonical spellings are the pinned
registry forms where they exist):

| Compatibility alias | Canonical target | Status |
| --- | --- | --- |
| `gen_ai.request.reasoning_effort` | `gen_ai.request.reasoning.level` (registry) | Documented alias; emitted today, unknown values omitted |
| `gen_ai.system` | `gen_ai.provider.name` legacy alias | Documented alias (registry-absent); emitted with provider name |
| `gen_ai.usage.cache_creation.input_tokens` | `gen_ai.usage.cache_write.input_tokens` (registry) | Documented alias; emitted with cache creation tokens |
| `gen_ai.usage.total_tokens` | custom derived input+output | Daydream-owned derived field (registry-absent) |
| `gen_ai.usage.cost` | custom reported cost | Daydream-owned field (registry-absent); currency via `daydream.usage.*` provenance |
| `traceloop.*` owned compatibility surface | daydream-owned equivalents | Owned aliases, never ambient callback values |
| `honeyhive_metadata.*` / `langsmith.usage_metadata` | native vendor aliases | Vendor-compat clones, billed owner only |

**Removal policy:** a compatibility alias may be removed only when every
consumer and both native destinations can read the canonical target; removal
requires a task-0-style manifest pin update, the AC ledger row marked, and a
docs/alias-table update in the same change. Underscore-prefixed private
HoneyHive fields (`_detectedAgent*`, `_agentsDetected`,
`_pathToDisplayNameMap`) are never written, asserted as stable, or migrated
to; if the API ever returns them they are read-only observations only.

## Vendor-reality notes (native readback evidence, 2026-09-09)

- HoneyHive elides empty `config`/`metrics`/`feedback` containers on returned
  events and synthesizes one aggregate `session` event per session; those
  containers are conditional (type-checked when present), never required.
- HoneyHive routes events into a session via the `session_id` filter only when
  the span carries the `traceloop.association.properties.*` identity — the
  identity rows above apply to **All spans**, generations included (verified:
  without identity, generations were orphaned into a synthetic run-id session).
- LangSmith stores the OpenLLMetry association property `daydream_run_id` as
  run metadata at `extra.metadata.daydream_run_id`, and rejects dotted-key
  metadata filters; top-level `metadata`/`invocation_params`/`usage_metadata`
  are not projected by `/runs/query`.
- **LangSmith OTLP ingest window:** the LangSmith API rejects OTLP batches
  whose span start times are older than ~24 hours (live 422:
  `start_time for post must be within ≈24 hours of current time`). The
  sanitized protocol replay deliberately pins generation timestamps to its
  manifest's historical interval (exact historical-equivalent reconstruction),
  so a replay whose pinned interval is older than the window is structurally
  absent from LangSmith. The readback verifier reports this honestly as
  READBACK_NOT_FOUND (zero discovered runs) — never as root ambiguity.

## Verification commands

Representative-real and sanitized-replay public readback use separate commands:

```bash
# Representative real run receipt
python scripts/verify_observability_readback.py \
  --receipt /path/to/representative-receipt.json \
  --result /path/to/result.json --deadline 30

# Sanitized protocol replay (operator boundary; only the pi subprocess is fake)
python scripts/replay_observability_acceptance.py \
  --fixture tests/fixtures/pi_jsonl/long_generation_replay.jsonl \
  --repo /path/to/clean/public/repo \
  --fake-pi /path/to/fake-pi --receipt /path/to/replay-receipt.json

# Then verify the replayed native trees
python scripts/verify_observability_readback.py \
  --receipt /path/to/replay-receipt.json --result /path/to/replay-result.json --deadline 30
```

Both tools share one immutable monotonic deadline (`--deadline`), an owned
`httpx.AsyncClient(trust_env=False, follow_redirects=False)` with no redirects,
bounded bodies, fixed redacted timeout disposition and atomic result receipts,
with deterministic response/client cleanup. API storage evidence is never UI
evidence; `stored contract passed` is distinct from `UI not inspected`, and the
UI limitation is documented in the acceptance ledger.

## AC-01..AC-31 traceability appendix

Validated against the Task 0 ledger (`P18-ac-ledger.md`); each row names its
exact test node or the live readback disposition. "Pending" live evidence is
owned by Task 7 native readback; every offline row is green in the current
tree.

| AC | Matrix/ledger binding | Evidence disposition |
| --- | --- | --- |
| AC-01 | This matrix (all rows above) plus registry-derived inventory | `test_observability_semconv_contract.py` (21 green) + this doc; live readback pending T7 |
| AC-02 | manifest.json byte/hash pins + alias table above | `test_manifest_pins_expected_commit_and_files` green; no lock change |
| AC-03 | agent identity rows; CLIENT/INTERNAL gate | `test_client_invoke_agent_requires_provider_name`; UI observation pending (T7, no connected browser) |
| AC-04 | identity/association rows | `test_observability_backend_protocols.py` identity asserts (T5) |
| AC-05 | redaction/ambient rows + privacy contact | `test_observability_runtime.py` ambient suite green; metadata wire leg in T5 |
| AC-06 | owned compatibility rows; notebook parity | `test_notebook_mode_uses_same_owned_batch_path_and_metadata` + zero-global tests |
| AC-07 | model/provider/response identity rows | protocol suite (`test_pi_replay...`, multi-model) green |
| AC-08 | effective request config rows | `test_observability_config.py` + backend protocol suites green |
| AC-09 | generation lifecycle rows (ordered choice parts) | trajectory lifecycle + T5 protocol tests green |
| AC-10 | replay rows (exact 395.332s values) | `test_pi_replay_exact_native_timing_choice_and_billing_through_runner` green; native API survival pending T7 |
| AC-11 | finish-reason rows | T5/trajectory stop-reason tests green |
| AC-12 | structural input rows | schema validation tests green; native readback pending |
| AC-13 | tool lifecycle rows | parallel/reused-ID tool tests green |
| AC-14 | billing owner rows | `TestBillingOwnerResolution` (8) + T4 native-owner tests green |
| AC-15 | usage/provenance/dedup rows | T2 ledger + T4 wire dedup tests green; native pending |
| AC-16 | timestamp rows | native-ms validator + T2 `TestNativeTimingValidation` + T5 exact replay green |
| AC-17 | resource rows | `test_observability_runtime.py` strict parser + wire checks green |
| AC-18 | identity/parentage/kind rows | T4 wire fidelity tests green |
| AC-19 | dropped-count rows | `test_destination_copies_preserve_dropped_attribute_counts` green; remaining bounds T4A |
| AC-20 | delivery rows | T4A ack matrix green (local) |
| AC-21 | deadline rows | T4A HTTPX/AnyIO deadline tests + this file's loopback peer tests green |
| AC-22 | resolver/precedence rows | T4A factory tests green |
| AC-23 | isolation/shutdown rows | T3 processor + T4A exactly-once tests green |
| AC-24 | status/error rows | T2/T4/T5 failure matrix green |
| AC-25 | agent-name rows | T4 standard-attribute tests green; UI pending |
| AC-26 | HH matrix rows + readback tool | matrix/tooling green (this task); exact HH readback pending T7 |
| AC-27 | project routing rows | T4 loopback project tests green; exact LS discovery pending |
| AC-28 | LS matrix rows + readback tool | matrix/tooling green; exact LS native tree pending |
| AC-29 | focused RED/GREEN rows | T2 29 tests + Task 6 readback/replay hermetic tests (this file) green |
| AC-30 | replay/readback receipts | `scripts/replay_observability_acceptance.py` + `verify_observability_readback.py` hermetic green; live export pending T7 |
| AC-31 | docs/matrix/ledger + gates | This doc + ledger rows; full `make check` + independent review + CI pending T7 |

## Limitations

- Native API storage proof is not UI rendering proof; derived display labels
  require authenticated UI observation (HoneyHive UI explicitly mandatory
  before #1156 closes).
- LangSmith exact identity falls back to the vendor-returned canonical IDs
  only; no formatted-UUID guessing or broad historical queries.
- This matrix documents the frozen contract at contract version `94f432d`;
  any emission change requires a matrix/ledger update in the same change.