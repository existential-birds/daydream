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

Pi records HTTP response status and allowlisted request/retry headers through
the native response hook, with host receipt time and correlation to the following
assistant generation.
Responses without an observed generation retain a null generation ID. Missing
responses remain unknown: the hook need not run for rejected requests or a
connection failure before headers arrive. These observations do not establish a
provider outage or change retry decisions.

On ordinary invocation errors, retain a redacted diagnostic tail (20 lines, up to
1,024 characters each) from the merged native stream, including bounded cleanup output.
Pi routes extension events through stderr, so that stream stays merged to preserve
protocol ordering. Cleanup stops at 1,024 additional lines, a 64 KiB threshold,
or its time bound, and accepts no late model/tool results. Process exit, EOF and
teardown times are host observations; the failure diagnostic states whether the
child was still running when host cleanup began.
An exit or EOF after that boundary cannot establish an independent native crash.
Underlying exception causes discarded inside Pi remain unavailable.

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

[Committed reviews](runtime.md#committed-review-requestresult) record discovery and
fresh challenges as separate native invocations. Claim UID/revision and captured
source identify challenge evidence. `settings.native_contexts` holds opaque native
configuration identities and phase associations; actual generation settings remain
separate trajectory observations. Credentials stay private and refreshable.

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

Partial invocation summaries sample the active host-owned lifecycle collector
without finalizing it or completing generations. A run snapshot samples each
active invocation once for its common cutoff; later activity cannot mutate prior
snapshot bytes. Completed attempt snapshots retain their ordinary terminal
semantics. Active fork summaries preserve dispatch/scope identities and original
sibling references; missing/corrupt observations never establish provider work.
