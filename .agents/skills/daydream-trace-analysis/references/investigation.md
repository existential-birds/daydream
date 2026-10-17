# Investigate one run

For a cohort, use the manifest row's `source` as `RUN` and pass its `run_id`
and physical `line` as `--run-id "$RUN_ID" --line "$RUN_LINE"` on every command
below. This selects the exact frozen record even when the source repeats an ID.

## Follow a lead

1. Inspect `summary` and `timeline` for outcome, evidence availability, scopes,
   assignments, finding yield, hard budgets, and advisory targets. Raw step
   counts differ across backends.
2. Use `mine` to find identical calls, explicit errors, and large strings. Exact
   tool-name/argument matching misses overlapping reads, differently batched
   shell commands, and semantically repeated investigations. Candidates are leads.
3. Retrieve a bounded window around an emitted step, call, or result pointer:

   ```bash
   python3 "$SKILL/scripts/trace.py" window "$RUN" \
     --pointer '/trajectories/value/documents/1/steps/3/tool_calls/0' \
     --before 2 --after 3 --chars 350
   python3 "$SKILL/scripts/trace.py" search "$RUN" \
     --query 'arguments are required: roots' --limit 5
   python3 "$SKILL/scripts/trace.py" show "$RUN" \
     --pointer '/trajectories/value/documents/1/steps/3' --chars 3000
   ```

   Use pointers actually returned for this run. `window` shows recorded neighbors
   in the same document, including explicit invocation/phase boundaries. Within
   each displayed step, it prioritizes the anchored call or result within the
   evidence limit, keeps retained rows in recorded order, and counts omitted
   rows. It does not infer causality or guarantee the terminal outcome is inside
   the window.
4. Reconstruct question → action → observation → next decision → recovery or
   terminal result. Look for repeated reads without new evidence, rediscovery
   after a handoff, failed syntax, wrong roots, contradictory scope assumptions,
   ignored results, and retries that leave the approach unchanged.
5. Check independent verification, fresh-stage context, changing working state,
   sharding overlap, and deliberate arbitration before classifying repetition.
   Same-path reads alone do not prove waste. Advisory targets are not hard
   limits; search exit 1 can mean no matches; error words do not establish failure.

Reports expose bounded arrays with `total` / `remaining`; page with `--offset`
and `--limit`. `--chars` bounds excerpts; `--evidence-limit` bounds examples and
results per call, and tools per window step. `show --offset` slices characters.
`search` matches literal string values, including prompts and quoted output;
inspect the owning step to distinguish instructions from actions. For queries
starting with `--`, use `--query=--expand`. Tools join results by ID within the
originating step. Unmatched/interrupted results are separate; missing results
leave the outcome unknown. Multi-record JSONL requires `--run-id` or `--line`;
cohort inspections use both to bind the exact manifest row.

## Triage a failure

Start with host evidence, then narrow the causal window:

- `outcome`, `completeness`, wrapper status/reason, and
  `provenance.collection_diagnostics` distinguish acquisition from execution.
  `unproduced` verification is not failed verification.
- `findings.value.terminal_coverage` retains admitted findings, incomplete
  scopes, phase outcomes, and diagnostics. `timeline` points to full metadata and
  `backend_lifecycle`.
- Step `extra.backend_diagnostics`, budget-stop/retry metadata, tool results,
  terminal output, and `verification.value` explain the relevant phase. Retrieve
  enclosing objects for keys such as `retry_stop_reason`, `schema_rejection`,
  or `expiry_stage`; `search` matches values rather than field names.

Distinguish a reviewed-code defect, recovered tool error, rejected finding,
transport/lifecycle failure, and acquisition failure. Report the earliest
supported cause and downstream symptoms; host evidence cannot prove unobserved
provider behavior. Read [format.md](format.md) for archives and source owners.
