# Observability

Read before changing tracing, timing, billing, or backend lifecycle evidence.
Tracing is operator-controlled and off by default. Repository config cannot
activate it or choose credentials/endpoints. CLI `--trace-to` replaces
`DAYDREAM_TRACE_TO`; `--no-tracing` overrides both. Full content is the default;
`--trace-content metadata` omits conversation/tool payloads and exception text.

```bash
export LANGSMITH_PROJECT=daydream
# Set LANGSMITH_API_KEY in the launching environment.
daydream --review /path/to/project --trace-to langsmith --trace-content metadata
```

HoneyHive needs `HH_API_KEY` and the deployment's `HH_API_URL`. Generic `otlp`
uses standard `OTEL_EXPORTER_OTLP_*` settings; Python exporter timeouts are
seconds. Preset endpoints do not inherit generic OTLP credentials or redirects.
Read [config](../daydream/observability/config.py) and
[exporters](../daydream/observability/exporters.py) for transport mappings.

## Ownership and evidence

Use a per-run provider and task-local context, never the global provider or
`Traceloop.init()`. Flow steps contain logical invocations, attempts, generations,
and tools. Parallel invocations are siblings. Join observers and cleanup before
freezing evidence. Export errors emit sanitized diagnostics without changing review
outcomes. Flush success and HTTP acknowledgement do not establish vendor ingestion;
verify native trees using [readback tooling](../scripts/verify_observability_readback.py).
The [replay manifest](../tests/fixtures/observability_contract/replay-manifest.json)
pins sanitized replay inputs; replay is distinct from real-provider acceptance.

Only the resolved billing owner emits usage/cost; parent aggregation must not
charge it again. Cache tokens are within input; reasoning tokens within output.
Missing values stay absent; observed zero stays zero. Pi generation intervals need
native boundaries; unfinished generations create no invented completion or billing.
Use interval unions for overlapping wall time. Host-rendered request sizes measure
exact attempt inputs, not provider tokenization or provider-hidden context.

[Span construction](../daydream/observability/spans.py),
[trajectory metrics](../daydream/trajectory/), and
[wire contract tests](../tests/test_observability_otlp_wire.py) own exact fields.
[Semantic convention fixtures](../tests/fixtures/observability_contract/) pin
external compatibility. Vendor attributes are added to copies of shared spans.
Transport activity, model progress, tool execution, and semantic coverage remain
distinct. Never infer provider root causes or missing tools from absent telemetry.

## Pi host-tool lifecycle evidence

Host requests/rejections retain native IDs, roles and generation history;
admission creates no synthetic tool events or billing. Lifecycle observations
separately record authority closure, native settlement and aggregate cleanup of
callbacks, native process and endpoint, after cleanup on every exit path. Domain
acceptance retains canonical findings independently of later native failure;
rejected selector payloads remain raw evidence without entering semantic handoffs.
A terminal host-admission diagnostic retains the typed failure code, mutation
certainty, safe condition/input context and available native call/generation
correlation. The binding's failure and underlying cause remain host evidence;
native details alone cannot establish acceptance, safe rejection or retryability.
Observed usage remains billable even when the host failure stops the invocation.
A successful callback cannot establish native settlement; suppressing a late
result cannot prove cleanup. See
[the lifetime contract](runtime.md#pi-host-tool-lifetime-and-limits).

## Committed-review evidence

The committed-review program records investigation and each fresh challenge as
separate native invocations through the run's recorder. Original captured source,
claim UID and semantic revision identify challenge work; a same-conversation
second look cannot discharge it. Retained challenge verdicts include substantive
grounds and canonical source references. Missing verdicts and native failures
remain unfinished work rather than rejection evidence.

Native configuration inputs are frozen before dispatch, including the merged
original project/global per-model thinking policy and project-trust input. The
packaged extension resolves inherited settings with the original native trust
policy. A selection-only SDK session uses in-memory settings/session storage,
loads no resources or tools, performs no model generation, and is disposed before
the ordinary invocation continues. Native setters apply its selected model and
clamped thinking without writing settings. Fully absent model configuration retains
Daydream's DEFAULT_PI_MODEL/nous fallback; an explicit blank project model retains
native saved-model reset and registry fallback. Requested model/provider/thinking
metadata is null where native selection remains authoritative. Explicit phase/environment
overrides and model thinking suffixes retain their native precedence over saved
providers and enabled-model scopes. Native scope resolution preserves the saved
model when admitted by the scope and otherwise selects its first resolved entry.
The PI_API_KEY compatibility alias retains ordinary Daydream routing, including
NOUS_API_KEY for its implicit default provider; credentials remain in the child
environment and are separate from requested native selection metadata.
Actual generation model/provider observations remain
separate evidence from requested inputs; frozen configuration alone proves neither.

The sealed result records admission, native settlement and cleanup independently
for each assignment. A completed process or accepted submission cannot establish
review completion. Close authority and join or isolate callback/evidence writers
before freezing the run. Required JSON output, report, trajectory, capture and
external delivery have separate outcomes; an output failure cannot erase admitted
findings or make a prior run appear to be the current successful result.

## Codex lifecycle evidence

Success requires the latest public turn's completion, stdout EOF, and an
independently observed zero child exit. Repeated terminal events cannot renew
absolute drain deadlines or double-charge usage. After terminal expiry, prohibit
retry of possibly completed mutations. Preserve bounded drain, reap, and cleanup
in [the adapter](../daydream/backends/codex.py) and
[lifecycle tests](../tests/test_codex_stream_lifecycle.py).

Lifecycle milestones are host-observed monotonic elapsed times. Pull-driven read
absence does not prove empty OS buffers. Child exit is independent of pipe EOF;
an event-loop sampler cannot diagnose a blocked loop until it resumes. Retain the
first expiry stage and secondary cleanup expiry. Supported native shutdown can
contain sequential fallback waits; the host drain allowance is policy, not proof
that all native awaits terminate. The historical stall's upstream cause remains
unresolved; public milestones support diagnosis without inventing private events.
