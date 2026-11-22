# Observability

Read before changing tracing, timing, billing, or backend lifecycle evidence.
Tracing is operator-controlled and off by default. Repository config cannot
activate it or choose credentials/endpoints. CLI `--trace-to` replaces
`DAYDREAM_TRACE_TO`; `--no-tracing` overrides both. Full content is the default;
`--trace-content metadata` omits conversation/tool payloads and exception text.
Selecting the built-in `local` destination records a [private file](#ownership-and-evidence)
with no network or credentials, alone or with remote destinations
(`--trace-to local --trace-to otlp` or `DAYDREAM_TRACE_TO=local,otlp`).

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

## Local evidence

Every recorded run keeps complete local OpenTelemetry evidence, with no tracing
destination, network access or credentials, independently of `--trace-to`. Each
document (the run's root and one per fork) is an append-only file of OTLP/JSON log
lines: `runs/<session_id>/evidence.otlp.jsonl` and
`runs/<session_id>/forks/<descriptor>.otlp.jsonl`. Each line is a complete
`ExportLogsServiceRequest` holding one `LogRecord` of the `daydream.evidence` scope
with a `daydream.evidence.seq` attribute and, while tracing is active, the current span's
trace and span ids. [The journal](../daydream/trajectory/journal.py) owns encoding;
[the document view](../daydream/trajectory/view.py) owns record names and reduction.

- **Records.** `daydream.evidence.open` (run, document and backend identity),
  `daydream.transcript.step` (one finished, redacted transcript step: messages,
  reasoning, tool calls with their results, usage, diagnostics, `extra.turn_boundary`),
  `daydream.agent.model`, `daydream.phase_event`, `daydream.subtrajectory`
  (invocation and fork summaries, including generation billing and backend lifecycle),
  `daydream.profile`, `daydream.evidence.snapshot` (a signal flush's in-flight work) and
  `daydream.evidence.seal`. GenAI attributes (`gen_ai.response.model`,
  `gen_ai.usage.*`) name usage where a convention covers it; the record body keeps
  everything else.
- **Acceptance and completeness.** A record is accepted once its whole line is written
  with no in-memory queue; the seal fsyncs. Only a seal whose record count and SHA-256
  match the bytes before it makes a document complete. A write, encoding or fsync
  failure warns once, stops that document and leaves it unsealed; a failed fork also
  marks the root seal `capture_complete: false`, and the run snapshot is `partial`.
  Readers report a truncated line, a broken sequence or a mismatched seal as
  incomplete evidence, never as complete.
- **Bounded memory.** Steps are appended as each invocation finishes; no whole-run
  transcript stays in memory. References name a document's accepted prefix (size and
  SHA-256), so a signal snapshot stays valid while the document grows.
- **Privacy.** Local evidence uses the same redaction as before (credentials, keys and
  user paths), not the remote export scrubbing, which stays for remote destinations and
  the opt-in `local` trace file below.
- **Consumers.** Archive, evaluation, `daydream summarize`, PR run info and dataset
  capture read documents through one checked read that reduces them to a view.
  Dataset run records (`daydream.run.v5`) embed each document's exact lines with their
  source SHA-256. [`daydream.benchmarks.harbor`](../daydream/benchmarks/harbor.py)
  exports Harbor-compatible ATIF for benchmarks; ordinary runs never import it.

`local` writes `otel-trace.jsonl` (mode 0600) in the run's private directory,
`~/.daydream/runtime/<workspace>/runs/<session_id>/`, once the artifact session opens.
Each line is one OTLP/JSON `ExportTraceServiceRequest` written as its span ends; the run
span ends last, so a complete file ends with the root span. The first open, encoding or
write failure logs one warning naming only the exception type and stops local recording,
leaving the file without its root span while the review and remote export continue. The
file is never frozen, published, archived or captured.

Only the resolved billing owner emits usage/cost; parent aggregation must not
charge it again. Cache tokens are within input; reasoning tokens within output.
Missing values stay absent; observed zero stays zero. Usage repeated under one message
id counts once, and an invocation total can raise but never lower summed turn usage, so
evidence steps and attempt spans report the same totals (except two Codex turns in one
invocation, [#1796](https://github.com/existential-birds/daydream/issues/1796)). Pi
generation intervals need native boundaries; unfinished generations create no invented
completion or billing.
Use interval unions for overlapping wall time. Host-rendered request sizes measure
exact attempt inputs, not provider tokenization or provider-hidden context. Each attempt
span records them as attributes before backend startup, so a startup failure keeps them,
and records each backend diagnostic as an event in observed order: the code always, the
message and metadata only with full content. The run span carries the run's GitHub
repository and, when known, its pull request number as `daydream.pr_repo` and
`daydream.pr_number`, and spans inside a forked recorder carry that fork's evidence document
id as `daydream.trajectory.id`. Each phase scope adds `daydream.phase.start` and
`daydream.phase.end` events to the active span with `daydream.phase` and the evidence
`scope_id` as `daydream.phase_scope.id`; the end event adds `daydream.phase_scope.status`
and, when set, `daydream.phase_scope.reason_code`. Each fan-out adds
`daydream.dispatch.start` and `daydream.dispatch.end` events with the evidence `dispatch_id` as
`daydream.dispatch.id`. The evidence recorder assigns both ids, so a run without one
records neither event.

[Span construction](../daydream/observability/spans.py),
[evidence and billing](../daydream/trajectory/), and
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

Native `compaction_start`/`compaction_end` events become `pi_compaction` diagnostics
on the active step: reason, abort/retry flags, token estimates, summary size and
SHA-256 (never the summary text) and the summary call's usage when Pi reports it.
That usage is billed in the invocation's terminal totals rather than as a reviewer
generation. Readiness also records the model's declared `contextWindow`/`maxTokens`
and compaction settings as a `pi_native_capacity` diagnostic.

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

## Claude host-tool lifecycle evidence

Claude host-tool attempts record the same lifecycle facts with `backend: claude`:
authority closure, native settlement, cleanup (the in-process MCP session ended after
the client disconnected) and host call/rejection counts, on every exit path. Host
calls keep their native `mcp__daydream__<tool>` names in the trajectory, and the
assignment instructions are recorded as supplied system instructions. A
`claude_native_capacity` diagnostic records the context window and output limit from
terminal model usage with `preflight: unavailable`; absent values stay absent. See
[Claude host tools](runtime.md#claude-host-tools).

## Codex host-tool lifecycle evidence

Codex host-tool attempts add the same host facts to the Codex lifecycle record:
authority closure, native settlement, cleanup (the loopback server stopped after the
process was reaped) and host call/rejection counts, on every exit path. Host calls keep
their bare tool names in the trajectory, as Codex reports them; a rejected call is a
failed tool result, and the request event carries the assignment instructions as its
system prompt. Every Codex MCP tool result, from the loopback server or any other, is
recorded as its text blocks joined with newlines, or as JSON for any other content,
including a result that carries only structured content (earlier releases recorded a
Python repr). No native capacity diagnostic exists. See
[Codex host tools](runtime.md#codex-host-tools).

## Committed-review evidence

`run_committed_review` owns one tracing run with `daydream.flow` `committed-review`, from
`config.observability` or, when that is unset, the operator's environment. Destinations
resolve through the bound [registry](extensions.md#trace-exporters), else the built-in
exporters. Its run span fails, as the `deep` flow's exit does, when analysis fails or the output
is not installed. A call made inside another run records into its own session, never the
enclosing one; the `deep` flow's review step stays inside that flow's run.

[Committed reviews](runtime.md#committed-review-requestresult) record each
[work item](runtime.md#assignment-policy-and-work) as a separate native invocation;
concurrent local work uses forked sibling trajectories. Challenge work records under the
`challenge` phase (earlier releases recorded it under `verify`) and all other work under `review`;
`verify` belongs to the fix flow's verification. Claim UID/revision, verdict
`work_id` and captured source identify challenge evidence. Execution entries carry
`work_id`, `allocation`, `observed_starts` (every observed start) and `overshoot`
(observed starts that were neither charged nor refunded); superseded attempts remain
evidence. `settings.native_contexts` holds opaque native
configuration identities and phase associations; actual generation settings remain
separate trajectory observations. Credentials stay private and refreshable.

Each document's usage ledger also keeps a share per evidence phase, charged at the same
sites as its total and folded with a fork under the same rule, so the shares sum to the
total. A retried attempt's invocation summary records `retry_failure_class`, the bounded
`FailureClass` that admitted the retry. The [review summary](runtime.md#committed-review-summary)
reads them.

## Budget stops

Every [budget stop](runtime.md#budget-ledger) appears on the trace once, as
`daydream.budget.owner`, `daydream.budget.dimension` (`time` or `tool_calls`),
`daydream.budget.limit`, `daydream.budget.spent`, `daydream.budget.remaining_at_dispatch`
and `daydream.budget.reserved`. The stop that ends an agent invocation is recorded as
attributes on its agent span, including one that ends in an exception (a provider retry
hint longer than the remaining deadline). Every other stop is a `daydream.budget.stop`
event with the same attributes on the active span: a review-finalization turn's stop on
its agent span, and a dispatch the host refuses before any agent runs on the enclosing
step or run span. Values are counts or durations in seconds, never monotonic timestamps.
A retry-ladder stop is a `daydream.retry.stop` event on its agent span carrying the evidence
`agent_budget_stop` fields as `daydream.retry.*`, with or without a recorder.
A committed-review dispatch the host schedule refuses is also retained in local evidence as a
`schedule_budget_stop` phase event with the same facts and its public `budget_reason`:
`host_wall_budget_exhaustion` for `time` and `host_tool_budget_exhaustion` for `tool_calls`.

## Codex lifecycle evidence

Success requires the latest public turn's completion, stdout EOF, and an
independently observed zero child exit. Repeated terminal events cannot renew
absolute drain deadlines or double-charge usage. After terminal expiry, prohibit
retry of possibly completed mutations. Preserve bounded drain, reap, and cleanup
in [the adapter](../daydream/backends/codex.py) and
[lifecycle tests](../tests/test_codex_stream_lifecycle.py).
Codex's stderr is not part of the public stream. It is drained separately, so it adds
nothing to parser coverage or raw-line counts. A redacted excerpt of the last 20 stderr
lines, without backtrace frames, is kept only for the exception raised on a non-zero exit.
The host tears the process down before it formats that exception, so the excerpt includes
stderr that a descendant kept open.

Lifecycle milestones are host-observed monotonic elapsed times. Pull-driven read
absence does not prove empty OS buffers. Child exit is independent of pipe EOF;
an event-loop sampler cannot diagnose a blocked loop until it resumes. Retain the
first expiry stage and secondary cleanup expiry. Supported native shutdown can
contain sequential fallback waits; the host drain allowance is policy, not proof
that all native awaits terminate. The historical stall's upstream cause remains
unresolved; public milestones support diagnosis without inventing private events.
