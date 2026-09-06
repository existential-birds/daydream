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
the corresponding trajectory artifacts.

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
