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

Bound host-tool calls retain native request/result IDs and genuine native roles
through the existing tool and generation projections. A host authorization error
is a tool result, so a corrected request remains in the same native history.
Host admission does not emit synthetic tool events or provider billing.

Invocation lifecycle observations distinguish authority closure, native
settlement, and aggregate cleanup across callbacks, read processes, and the
endpoint. Cleanup runs before those observations freeze, including cancellation
and setup failure.
Missing native settlement remains incomplete even when the host read succeeded;
missing cleanup proof cannot be inferred from a suppressed late result.

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
