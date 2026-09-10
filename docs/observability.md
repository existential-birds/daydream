# Observability

Daydream exports OpenTelemetry traces using OpenLLMetry 0.62.3. Built-in destinations
cover LangSmith, HoneyHive, and generic OTLP collectors over HTTP/protobuf or gRPC.
An extension can register another OpenTelemetry exporter without changing the review
pipeline or backend implementations.

## Enable tracing

Tracing is off by default. Select a destination for any review, improve, or custom flow:

```bash
daydream /path/to/project --trace-to langsmith
daydream improve /path/to/project --trace-to langsmith
daydream /path/to/project --trace-to langsmith --trace-to otlp
```

For repeated use, select destinations in the launching shell or CI job:

```bash
export DAYDREAM_TRACE_TO="langsmith,otlp"
daydream /path/to/project
```

CLI destinations replace the environment list. `--no-tracing` disables tracing even
when the environment enables it. Duplicate destination names are rejected. Backend
credentials alone, or an OTLP endpoint alone, do not enable tracing.

Settings are operator-owned: `.daydream.toml` and `[tool.daydream]` in the repository
being reviewed cannot activate tracing, choose an endpoint, or select credentials.

## LangSmith

```bash
export LANGSMITH_API_KEY="your-key"
export LANGSMITH_PROJECT="daydream"
daydream /path/to/project --trace-to langsmith
```

The project defaults to `daydream`. Daydream sends HTTP/protobuf spans to
`https://api.smith.langchain.com/otel/v1/traces` with the API key and project headers.
No LangSmith Python SDK is required.

Set `LANGSMITH_ENDPOINT` to a regional or self-hosted API base. For example:

```bash
export LANGSMITH_ENDPOINT="https://eu.api.smith.langchain.com"
```

For a self-hosted base that requires `/api/v1`, include that prefix in the base URL.
Daydream appends `/otel/v1/traces`. Set `LANGSMITH_WORKSPACE_ID` when the API key
requires an explicit workspace; it becomes the `x-tenant-id` header.

Attempts appear as LLM runs, structural scopes as chains, and tool executions as
tools. The destination adds LangSmith's native usage metadata to preserve known
cache/reasoning token details and reported cost. Parent spans rely on LangSmith's
aggregation; they do not repeat attempt usage.

Endpoint and field mappings follow the
[LangSmith OpenTelemetry integration](https://docs.langchain.com/langsmith/trace-with-opentelemetry).
Usage details use the `langsmith.usage_metadata` format emitted by
[LangSmith's own OTLP exporter](https://github.com/langchain-ai/langsmith-sdk/blob/4792a53f807e9129fd4f5c243056d1976e2cff80/js/src/experimental/otel/exporter.ts#L219).

## Missing tracing dependencies

If tracing reports `No module named 'opentelemetry.exporter.otlp.proto.grpc'`,
`No module named 'traceloop'`, or missing/incompatible tracing dependencies, refresh
the installation. This can happen after pulling changes into an editable install:
the command reads the new source, but its Python environment still has the old
dependencies. It affects LangSmith and HoneyHive even though both send HTTP,
because OpenLLMetry imports both OTLP transports.

From the Daydream clone, reinstall the command and its dependencies:

```bash
uv tool install --reinstall --editable .
command -v daydream
```

The command should resolve to the uv tool executable directory (`uv tool dir --bin`),
usually `~/.local/bin`. If it resolves to a pyenv shim or another installation,
remove that stale Daydream installation or put the uv tool directory first on
`PATH`. The install requires Python 3.12.13 or newer.

To run directly from the clone with its synchronized dependencies:

```bash
uv run daydream /path/to/project --trace-to langsmith
uv run daydream /path/to/project --trace-to honeyhive
```

`uv sync` updates the clone's `.venv`; it does not update a separately installed
`daydream` command. See [uv's tool management documentation](https://docs.astral.sh/uv/concepts/tools/).

## HoneyHive

Use the deployment API base for your HoneyHive organization and region:

```bash
export HH_API_KEY="your-key"
export HH_API_URL="https://api.dp1.us.prod.honeyhive.ai"
daydream /path/to/project --trace-to honeyhive
```

The URL above is a HoneyHive documentation example; use your deployment's actual
base. Both variables are required. Daydream appends `/opentelemetry/v1/traces` and
sends HTTP/protobuf with bearer authentication. The API key selects the project.
Use a write-enabled project API key, not a workspace control-plane key. You do not
need a workspace ID header; the project key determines its workspace as well.

HoneyHive creates a session around each run through its OTLP session-creation
attributes. Run spans, steps, and logical agents appear as chains, backend attempts
as model events, and tool executions as tools. The adapter maps known cost,
cache-read/write counts, and reasoning tokens into native metadata fields on
attempts only. It preserves the portable attributes and does not change spans sent
to other destinations.

HoneyHive documents a $0.10 initial session cost in its
[normalizer](https://docs.honeyhive.ai/v2/sdk-reference/semconv-reference). Its displayed
session cost can therefore exceed the sum of billed attempts. Use attempt-level
cost for reconciliation; Daydream does not subtract that offset from actual usage.

The adapter follows HoneyHive v2's documented
[OTLP endpoint and authentication](https://docs.honeyhive.ai/v2/integrations/truefoundry)
and [semantic convention mappings](https://docs.honeyhive.ai/v2/sdk-reference/semconv-alignment).
No HoneyHive SDK is required.

## Generic OpenTelemetry platforms

Point the `otlp` destination at your collector or platform's ingestion endpoint.
For HTTP/protobuf:

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
daydream /path/to/project --trace-to otlp
```

`OTEL_EXPORTER_OTLP_ENDPOINT` is a base URL; the HTTP exporter appends `/v1/traces`.
Use `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` for an exact traces endpoint:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://collector.example.com/v1/traces"
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer%20your-token"
```

For a local gRPC collector:

```bash
export OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
daydream /path/to/project --trace-to otlp
```

The generic adapter honors the standard trace-specific variables and their general
OTLP fallbacks for endpoint, headers, protocol, timeout, compression, and TLS
configuration. HTTP/protobuf is the default transport. Daydream's Python OTLP
exporter uses seconds for timeout values: set `OTEL_EXPORTER_OTLP_TRACES_TIMEOUT=10`
for a 10-second timeout, not `10000`. TLS certificates and client credentials follow
the selected OpenTelemetry exporter's configuration contract.

LangSmith and HoneyHive use their own endpoint and credential variables. Generic
OTLP endpoint/header settings do not redirect these presets. Their owned HTTP
sessions do not inherit `.netrc` authentication and do not follow redirects.

## Content and trace structure

Enabled tracing captures full Daydream-visible content by default. Choose metadata
mode to keep execution structure and numeric usage while omitting content:

```bash
daydream /path/to/project --trace-to langsmith --trace-content metadata
export DAYDREAM_TRACE_CONTENT="metadata"
```

`--trace-content full` overrides the environment policy. Full mode includes the
effective prompt, Daydream-supplied system instructions, responses, exposed reasoning
summaries, output schemas, structured results, and tool arguments/results. Shared
redaction removes recognized credentials and known secret values before export.
Metadata mode omits these payloads and exception messages/stack traces.

The hierarchy follows real Daydream execution:

```text
run
└── executed flow step
    └── logical agent invocation
        ├── failed attempt
        │   └── tool
        └── successful attempt
            └── tool
```

Parallel agent invocations are siblings. Retries keep their own partial content,
usage, and failure status. Interrupted tools close explicitly. Skipped flow steps
do not create executed-step spans. Run and session identifiers connect spans to
the corresponding trajectory artifacts. During execution, those artifacts live
in private source-owned storage. After finalization, the same identifiers
connect the published source output and archived bundle. An ephemeral worktree
does not become a second run identity.

An attempt represents one call to a Daydream backend, which may contain multiple
internal model turns and tools. Backends expose different turn boundaries, so
Daydream reports known usage for the whole attempt. Available terminal totals
reconcile with intermediate metrics without adding both copies. Cache reads and
cache creation are part of input tokens; reasoning tokens are part of output tokens.
Unavailable values remain absent, and known zero values remain zero.

All four backends share the same trace path. Data hidden by a backend's public
stream—such as a proprietary CLI's internal system prompt or individual provider
request timings—cannot be captured. Daydream records the effective request it sends
and the response/tool data the backend exposes. Native tool timestamps and durations
are stored as metadata; span intervals follow Daydream's observed execution. Attempt
duration metadata comes only from terminal results that report the whole invocation.
Per-message durations stay with their message usage metadata. Synthesized completion
events do not establish an unobserved start time.

Model request/response attributes belong to attempts. Logical agents retain their
configured model as `daydream.configured.model`; child tools use the `execute_tool`
operation and do not inherit the model-call attributes of their parent attempt.

For example, current Codex CLI JSON streams omit provider identity. Daydream leaves
that field absent instead of guessing from the model name or a partial configuration
file. A native provider value is preserved when supplied. Codex cost uses Daydream's
configured model pricing because its stream reports tokens rather than monetary cost.

Set `OTEL_SERVICE_NAME` to change the service label from `daydream`.

## Trajectory timing

Trajectory timing is recorded even when trace export is off. Identified phase
events pair a start and terminal event by session, scope and phase. Terminal
states distinguish success, partial results, failure, cancellation, timeout and
intentional skips. MERGE and DIAGRAM have their own scopes; a tiny, host-only
merge does not fabricate an agent invocation.

A parallel dispatch records its start before its children begin and its completion
after they join. Its planned, attempted and completed counts describe that dispatch,
and its references identify the child documents actually written. Each backend
invocation has a document-qualified identity; a fork wrapper is not another call.

The archive manifest's `metrics.timing_coverage` reports attributed and unattributed
wall time, coverage ratio, agent completeness and bounded diagnostics. Repeated and
overlapping intervals are unioned, so adding individual phase durations is not a
valid way to reconstruct run duration. Unpaired or malformed evidence is diagnosed,
not silently counted as complete. Evaluation and manifest totals use the same
frozen trajectory input.

A signal flush freezes the root and active children at one `snapshot_at` cutoff.
Open invocations remain incomplete, without invented end times. If children have
started before the root has a step, the partial root contains a system snapshot
event, not an agent call. A later final write has a new cutoff and does not alter
the earlier snapshot's bytes. Current-session MERGE events also govern archived
merge state; a stale successful report cannot override a current failed merge.

## Lifecycle and failures

Each run owns its tracer provider and selected exporters. Daydream does not replace
the process-wide OTel provider or initialize OpenLLMetry's global instrumentors.
This supports repeated invocations and keeps separate runs isolated.

Unknown destinations, missing required credentials, and invalid configuration fail
before agent work begins. Once execution starts, export failures produce sanitized
diagnostics and preserve the review outcome. At exit, Daydream closes spans and
waits for exporter flush/shutdown off the event loop, including after cancellation.
One 10-second cleanup budget covers all destinations. A deadline warning means
remaining delivery is not confirmed; successful review completion does not itself
prove ingestion by the destination.

Extension exporters must use finite network timeouts and cooperative shutdown. A
noncooperative exporter may continue cleanup in a worker after the caller's wait
expires. The runtime does not start another shutdown against that exporter.

For a custom destination, see the
[trace exporter extension contract](extensions.md#trace-exporters).

## Field contract and versioned matrix

Every emitted field is frozen in [observability-fields.md](observability-fields.md):
name, source backend/event, source authority/provenance, owning span, type,
unit, cardinality, derivation, completeness, capture/redaction, generic OTLP
disposition, HoneyHive canonical destination, LangSmith native destination,
offline test node, live evidence status, and applicability/omission reason. The
matrix also lists per-backend unavailable/inapplicable rows, the compatibility
alias table and removal policy, and an AC-01..AC-31 traceability appendix
validated against the Task 0 acceptance ledger. The machine-readable subset
used by the readback verifier is
`tests/fixtures/observability_contract/readback-matrix.json`; it may only
shrink relative to the markdown matrix.

## Native readback verification

After a run shuts down, `scripts/verify_observability_readback.py` proves the
stored HoneyHive/LangSmith native trees against an immutable receipt:

```bash
HH_API_URL=... HH_API_KEY=... LANGSMITH_API_KEY=... \
python scripts/verify_observability_readback.py \
  --receipt /path/to/receipt.json --result /path/to/result.json --deadline 30
```

The verifier keeps ONE immutable monotonic deadline (`--deadline`, default 30s)
across every request, capped streamed JSON-body read, stability poll and
backoff. It owns exactly one `httpx.AsyncClient(trust_env=False,
follow_redirects=False)` — no redirects are ever followed — closes it on every
exit path, and each request runs
as `async with client.stream(...)` under `anyio.fail_after(remaining)`. A
deadline expiry emits only the fixed redacted `READBACK_TIMEOUT` disposition
and starts no further request or poll; timeout output never contains a URL,
header, response body, exception text or credential. Real loopback peers that
delay response headers or trickle a JSON body cannot extend a 0.12-second
verifier budget past 0.35 seconds; the peer observes connection closure within
one second and the request log proves no later page or stability poll began.
Responses are capped at
4 MiB + 1 byte and malformed/incomplete/trailing JSON fails closed. The input
receipt is validated BEFORE any client is constructed; the standalone result
receipt is written atomically and contains only destination, IDs, counts,
names, types, booleans, stable hashes and pass/fail matrix rows.

HoneyHive is read through the documented `POST /v1/events/search` endpoint
with an exact-session `session_id` filter, pages 1..1000, strict
`{events, count}` validation, duplicate/wrong-session/malformed rejection,
and two stable complete post-shutdown snapshots. LangSmith is read through the
documented `POST /runs/query` discovery with exact `daydream.run.id` equality
in one explicit project and a bounded start time; the verifier requires one
trace/root, freezes only the vendor-returned IDs, then uses the documented
exact `id`/`trace_id` read semantics and reaches two equal complete tree
snapshots before the deadline. It never derives IDs from HoneyHive or from the
Daydream UUID. Keys come only from the environment; base URLs are validated
with the same production policy as the exporters.

The sanitized protocol replay is a separate operator boundary:
`scripts/replay_observability_acceptance.py` accepts only the checked-in
manifest-pinned sanitized fixture (byte/hash pinned in
`tests/fixtures/observability_contract/replay-manifest.json`) and a clean
public disposable repository whose origin is on the manifest allowlist; it
requires a caller-provided external fake `pi` executable at the subprocess
boundary, validates authorization and the exact destination set
(`otlp,honeyhive,langsmith`) BEFORE constructing exporters, sets
`daydream.acceptance.kind=sanitized_protocol_replay`, drives the REAL
PiBackend/run_agent/trace_run path once with its own bounded loopback OTLP
wire oracle, requires local wire success, and writes an immutable receipt
labeled `sanitized_protocol_replay` with model-call count 0, operational cost
$0 and hashes/IDs only. The historical-equivalent reported cost ($0.00402781)
rides as labeled synthetic telemetry, never actual provider billing.

API storage evidence is never UI evidence: `stored contract passed` is
distinct from `UI not inspected`. Authenticated UI observation of the derived
agent label remains a separate, mandatory gate (HoneyHive UI explicitly)
before #1156 closes.

## Vendor acknowledgment realities (binding decision 8)

LangSmith's canonical OTLP success acknowledgment is HTTP 200 with a
zero-byte body and **no Content-Type header** (confirmed with real exports
and minimal probes, 2026-09-09). The owned transport classifies any
`200` response whose content type is missing or not
`application/x-protobuf` as `OTLP_MALFORMED_ACK` and records it
`unverified` in the per-destination ledger — per binding decision 8,
absence of the protobuf content type is terminal/unverified, and
acceptance counts are never invented from non-conforming acks. LangSmith
does store the payload (proven exclusively by the native readback gate,
never by the ack), so with that destination the ledger will show
`delivered=0 / unverified` even when storage succeeded. Delivery to
LangSmith is therefore judged by `scripts/verify_observability_readback.py`
results, not by the exporter ack ledger. HoneyHive returns the canonical
protobuf-content-type empty ack and is classified `empty_ok` (full
success).
