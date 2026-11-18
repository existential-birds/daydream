# Capture and evidence map

## Containers and identity

The local store usually has `~/.daydream/dataset/runs/<digest>.jsonl`. Each
nonblank line is a `daydream.run.v5` record, frequently hundreds of KB or more.
Use the record's `run_id`, not the basename. Physical line numbers locate the
record; JSON pointers locate evidence inside it. Earlier record versions are not
read; inspect them with an earlier revision of this skill. For a cohort row, pass both its
`--run-id` and physical `--line` to the trace helper so repeated IDs select the
frozen record unambiguously. Indexes in pointers are zero-based; `step_id` is
the recorded ID, not its array position.

| Location | Meaning |
| --- | --- |
| `schema_version`, `run_id`, `captured_at`, `outcome`, `trace_id` | Run identity and terminal outcome; trace ID can be null |
| `original_task`, `trajectories`, `findings`, `verification`, `recommended_patch`, `final_state` | `{status, reason, value}` evidence wrappers |
| `completeness` | Separate acquisition, trajectory and terminal review availability |
| `provenance` | Producer identity, repository/revision, configuration/profile, resolved phases, collection diagnostics |
| `trajectories.value.root_trajectory_id` | Root document identity; do not assume documents are root-first |
| `trajectories.value.documents[]` | Embedded evidence documents: `trajectory_id`, `session_id`, `sha256`, `otlp_jsonl` (the exact OTLP/JSON lines) |
| `findings.value.items`, `.terminal_coverage` | Canonical findings with their identities, scope and phase outcomes |

Wrapper states include `available`, `unavailable`, `unproduced`, `withheld`,
`failed`. Preserve state and reason. An unavailable section is not an empty
result. An `available` wrapper requires a non-null `value`; fields inside that
value may still be null. Success means the run completed, not that every tool
succeeded or every finding is correct.

## Evidence documents and pointers

Each document is Daydream's local OpenTelemetry evidence: one OTLP/JSON logs
request per line, each holding one `daydream.evidence` record with a
`daydream.evidence.seq` attribute (`daydream.evidence.open`,
`daydream.transcript.step`, `daydream.agent.model`, `daydream.phase_event`,
`daydream.subtrajectory`, `daydream.profile`, `daydream.evidence.snapshot`,
`daydream.evidence.seal`). The helpers verify each embedded document against its
`sha256` (a mismatch is an error, never silently read), check the sequence and
seal, and reduce the records to the same document view Daydream's own reducer
builds (`scripts/evidence.py`). An unsealed, truncated or seal-mismatched document
reads as `extra.partial` with its accepted records; `evidence_problem` names why.

Pointers address the record with each document's raw `otlp_jsonl` replaced by its
`view`: `/trajectories/value/documents/<i>/view/steps/<n>/...`. The working copy keeps
`trajectory_id`, `session_id`, `sha256` and `evidence_problem` beside `view`, so
`search` reports each value once. Report pointers in this form; `reports.py`
resolves evidence the same way. For a standalone document the pointer addresses
the view directly (`/steps/<n>/...`).

Views have `trajectory_id`, shared `session_id`, `parent_trajectory_id`, `run_flow`,
`agent`, `steps`, `extra`, `final_metrics`. Steps have `step_id`, `timestamp`, `source`, `message`, optional
`reasoning_content`, `model_name`, `llm_call_count`, `metrics`, `tool_calls`,
`observation`, `extra.daydream_phase` / `daydream_run_flow`, and `extra.turn_boundary`
(`turn_end`, `result` or `invocation_end`: the event that closed the step).

## Calls, results and missing actions

`tool_calls[]` holds `tool_call_id`, `function_name`, `arguments`.
`observation.results[]` holds `source_call_id`, `content`, optional `extra` with
`is_error`, `exit_code`, `status`, cancellation/truncation and native metadata.
Result order can differ from call order. IDs can repeat in another invocation,
even within the same root document; qualify by document + step, and invocation
when recorded. The helper does not guess associations across steps. Investigate
unmatched or interrupted outputs and late results separately.

Dispatch observations can contain `subagent_trajectory_ref[]` without a
`source_call_id`; these are graph links, not ordinary tool responses. A fork's
reference can be a logical archived path, and redacted absolute paths are not
usable local paths. Never replay captured commands to inspect them.

Codex often coalesces many shell calls into one agent step. Pi commonly emits
fine-grained `read` / `grep` / `find` / `bash` / `structured_output` calls plus
reasoning. Shell batches can themselves contain several actions. Step count and
`llm_call_count` are not comparable native turn/generation counts. Inspect
`extra.backend_diagnostics`: unknown native items such as `web_search` may not
appear in normalized tool calls. Non-JSON diagnostics occur in successful runs,
too; absence from the tool list is not proof that an action did not happen.

## Time, costs and retries

- `extra.phase_events[]`: pair start/end by `(session_id, scope_id, phase)`.
  Metadata includes stage/assignment, `failure_class`, `stop_reason`,
  `schema_rejection`, retry feedback, observed tools and hard/advisory budgets.
  `deep` can include arbiter work; `alternatives` can include Wonder. Preserve
  scope and shard IDs.
- `extra.subtrajectories[]`: direct invocation rows have `invocation_id`,
  `step_ids`, start/end, often `backend_lifecycle`. Fork wrappers instead have
  `descriptor`, `dispatch_id`, `fork_id`, `trajectory_id`, `invocations` and phase
  events. These repeat sibling metadata. Do not count wrappers as invocations or
  count nested rows again after reading sibling documents.
- Invocation/phase spans are useful intervals. Overlap means summation
  exaggerates wall time; use interval unions for totals. The helper deliberately
  reports intervals without an additive total. A step timestamp is not the
  start/end of each contained tool; native durations remain separate metadata.
- Lifecycle `*_s` values are host-observed monotonic seconds relative to an
  invocation, not UTC timestamps. They describe startup, native activity,
  terminal drain, EOF, exit/reap and cleanup; they cannot establish provider
  receipt or an upstream root cause. Missing fields remain unknown.
- Root `final_metrics.extra.daydream_metric_scope=whole_run_including_forks`
  means root billing includes forks. `total_steps_scope=local_trajectory` still
  makes its step count local. Cached tokens are within prompt tokens; reasoning
  is within completion. Zero observed cost is not evidence of free execution.
- Token/cost metrics may occur only on the last step of an invocation. Do not
  assign them to its final tool or combine step, root and fork totals.
- `agent_budget_stop` can describe retry-only cost, excluding initial useful
  work. Read `retry_stop_reason`, `circuit_state` and earlier failures. Retry
  overhead is backoff plus backend retry time; entire invocation duration is not
  retry waste. A budget stop's owner, dimension, limit, spend and remainder are
  on the trace (`daydream.budget.*` agent-span attributes or a
  `daydream.budget.stop` span event), not in this event.

## Beyond the embedded capture

The canonical evidence layout is `<artifact-root>/runs/<session_id>/evidence.otlp.jsonl`
with `forks/*.otlp.jsonl` siblings; `trace.py` reads one such document directly.
Interrupted documents stay unsealed and read as partial; a truncated last line is
reported (`evidence_problem`), never silently dropped. `daydream evidence verify RUN_DIR`
checks every document's seal, count and digest. Locate the artifact root
from the actual run configuration or archive, not redacted captured paths.

Failure archives can contain richer `deep/` evidence: `review-coverage.json`,
`fix-failures.json`, `stabilization-failed.json`, `repair-job.json`,
`repair-checkpoint.json`, `remote-ci-handoff.json`; also input diff, review output,
recommended patch and findings. Inspect only the phase relevant to the failure.
`verification.value` may retain fix/test/quality-gate/push/remote-CI verdicts even
when a generic final message is unhelpful.

For finding provenance, follow [finding identity](../../../../docs/extensions.md#finding-identity).

## Source owners in the Daydream checkout

Use these when the capture is ambiguous or a newer producer changes its schema;
the helper is a navigation tool, not a replacement for runtime contracts.

| Owner | Read for |
| --- | --- |
| `daydream/dataset.py`, `dataset_capture.py` | Envelope schema, status and captured artifacts |
| `daydream/trajectory/layout.py`, `types.py` | Physical layout, identity, event vocabularies |
| `daydream/trajectory/journal.py`, `view.py` | Evidence encoding, seal verification, document view |
| `daydream/trajectory/invocation.py`, `scopes.py`, `recorder.py` | Tool correlation, interrupted outcomes, forks, metric ownership |
| `daydream/trajectory/timing.py` | Authoritative phase interval/latency reduction |
| `daydream/eval/analyzer.py` | Existing cost/tool/timing signals |
| `daydream/deep/artifacts.py`, `daydream/archive/bundle.py` | Failure artifacts and archived evidence |
| `docs/observability.md`, `daydream/observability/spans.py` | Lifecycle, billing and timing interpretation |

The existing analyzer consumes a `RunWriteSnapshot`/artifact workspace; it is not
a direct embedded JSONL parser. Prefer the bundled CLI for navigation, then use
the owning reducer when a precise latency or billing conclusion needs it.

## Helper verification

```bash
python3 -m unittest discover -s "$SKILL/scripts" -p 'test_*.py'
```

Tests use synthetic captures in a temporary directory; private run contents
are neither copied into the repository nor required for routine verification.
