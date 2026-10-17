# Capture and evidence map

## Containers and identity

The local store usually has `~/.daydream/dataset/runs/<digest>.jsonl`. Each
nonblank line is a `daydream.run.v1` record, frequently hundreds of KB or more.
Use the record's `run_id`, not the basename. Physical line numbers locate the
record; JSON pointers locate evidence inside it. For a cohort row, pass both its
`--run-id` and physical `--line` to the trace helper so repeated IDs select the
frozen record unambiguously. Indexes in pointers are zero-based; `step_id` is
the recorded ID, not its array position.

| Location | Meaning |
| --- | --- |
| `schema_version`, `run_id`, `captured_at`, `outcome`, `trace_id` | Run identity and terminal outcome; trace ID can be null |
| `original_task`, `trajectories`, `findings`, `verification`, `scoring`, `recommended_patch`, `final_state` | `{status, reason, value}` evidence wrappers |
| `completeness` | Separate acquisition, trajectory and terminal review availability |
| `provenance` | Producer identity, repository/revision, configuration/profile, resolved phases, latency routing, collection diagnostics |
| `trajectories.value.root_trajectory_id` | Root document identity; do not assume documents are root-first |
| `trajectories.value.documents[]` | Embedded ATIF documents, normally `ATIF-v1.7` |
| `findings.value.claims`, `.derivation`, `.terminal_coverage` | Stack claims, adjudication/dedup artifacts, scope and phase outcomes |

Wrapper states include `available`, `unavailable`, `unproduced`, `withheld`,
`failed`. Preserve state and reason. An unavailable section is not an empty
result. An `available` wrapper requires a non-null `value`; fields inside that
value may still be null. Success means the run completed, not that every tool
succeeded or every finding is correct. Missing correctness scoring leaves
finding correctness unestablished.

Documents have `trajectory_id`, shared `session_id`, `agent`, `steps`, `extra`,
`final_metrics`. Steps have `step_id`, `timestamp`, `source`, `message`, optional
`reasoning_content`, `model_name`, `llm_call_count`, `metrics`, `tool_calls`,
`observation`, and `extra.daydream_phase` / `daydream_run_flow`.

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
  work. Read `retry_stop_reason`, `circuit_state`, `retry_recovery_spent_s` and
  earlier failures. Recovery allowance includes backoff and backend retry time;
  entire invocation duration is not retry waste.

## Beyond the embedded capture

The canonical trajectory layout is `<artifact-root>/runs/<session_id>/trajectory.json`
and `trajectories/*.json` siblings; interrupted documents can end `.json.partial`.
An intact partial-suffix JSON is readable; truncated JSON is an explicit parser
error, not permission to silently discard its tail. Locate the artifact root
from the actual run configuration or archive, not redacted captured paths.

Failure archives can contain richer `deep/` evidence: `review-coverage.json`,
`fix-failures.json`, `stabilization-failed.json`, `repair-job.json`,
`repair-checkpoint.json`, `remote-ci-handoff.json`; also input diff, review output,
recommended patch and findings. Inspect only the phase relevant to the failure.
`verification.value` may retain fix/test/quality-gate/push/remote-CI verdicts even
when a generic final message is unhelpful.

For finding provenance, `uid` identifies a pre-merge record, `item_uid` a merged
item, `source_uids` the contributing records; integer `id` is renumberable.
Content fingerprints measure similarity, not identity. Preserve these when
following a repeated finding through review → adjudication → merge.

## Source owners in the Daydream checkout

Use these when the capture is ambiguous or a newer producer changes its schema;
the helper is a navigation tool, not a replacement for runtime contracts.

| Owner | Read for |
| --- | --- |
| `daydream/dataset.py`, `dataset_capture.py` | Envelope schema, status and captured artifacts |
| `daydream/trajectory/layout.py`, `types.py` | Physical layout, identity, event vocabularies |
| `daydream/trajectory/invocation.py`, `scopes.py`, `recorder.py` | Tool correlation, interrupted outcomes, forks, metric ownership |
| `daydream/trajectory/timing.py` | Authoritative phase interval/latency reduction |
| `daydream/eval/analyzer.py`, `latency_report.py` | Existing cost/tool/timing signals and corpus comparisons |
| `daydream/deep/artifacts.py`, `daydream/archive/bundle.py` | Failure artifacts and archived evidence |
| `docs/observability.md`, `docs/observability-fields.md` | Lifecycle, billing and timing interpretation |
| `scripts/measure_review_runtime.py` | Isolated matched-fixture before/after review harness |

The existing analyzer consumes a `RunWriteSnapshot`/artifact workspace; it is not
a direct embedded JSONL parser. Prefer the bundled CLI for navigation, then use
the owning reducer when a precise latency or billing conclusion needs it.

## Helper verification

```bash
python3 -m unittest discover -s "$SKILL/scripts" -p 'test_*.py'
```

Tests use synthetic captures in a temporary directory; private run contents
are neither copied into the repository nor required for routine verification.
