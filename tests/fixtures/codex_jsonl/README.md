# Codex JSONL fixtures

Recorded and synthesized `codex` CLI experimental-JSON streams replayed through
the **real** `CodexBackend` parser in tests. The mock subprocess
(`tests/harness/codex_replay.py:make_mock_process`) yields these lines through
`stdout.readline()` exactly as the live CLI would; only the subprocess boundary
is stubbed, so the genuine JSONL parser in `daydream/backends/codex.py` runs end
to end.

## Recorded-real vs. synthesized

- **Synthesized** fixtures are generated from canonical scripts by
  `tests/contract/_loaders.py:_build_codex_jsonl` (and the harness wrappers in
  `tests/harness/scripts.py`). They are deterministic and easy to author, but by
  construction they cannot surprise the parser with real-CLI shapes.
- **Recorded-real** fixtures (`realpath_*.jsonl`) are captured from genuine
  `codex` CLI runs, then redacted. They guard against the
  "Codex Backend Gotchas" the synthesizer does not reproduce:
  - agent/reasoning text streamed via `item.updated` deltas while
    `item.completed` carries empty `content`;
  - content blocks typed `output_text` (not just `text`);
  - structured payloads carried on `turn.completed.result` (and/or `output`).

`realpath_parse.jsonl` exercises all three in one PARSE-phase stream and still
drives the real parser to a valid `FEEDBACK_SCHEMA` `structured_output` (one
issue), used by `tests/test_codex_realpath.py::test_codex_realpath_from_recorded_fixture`.

## Capturing a fresh fixture

The backend launches the CLI as (see `daydream/backends/codex.py`):

```
codex exec --experimental-json --skip-git-repo-check \
  [--output-schema <schema.json>] [-m <model>] -
```

To capture a phase stream, run the same invocation against a throwaway repo and
tee its stdout (the prompt is fed on stdin, closed immediately):

```bash
printf '%s' "$PROMPT" | \
  codex exec --experimental-json --skip-git-repo-check \
    --output-schema /tmp/feedback_schema.json - \
  > tests/fixtures/codex_jsonl/realpath_parse.jsonl
```

- `--output-schema` is only used for the PARSE phase (it constrains the agent
  to `FEEDBACK_SCHEMA`); omit it for REVIEW/FIX/TEST captures.
- One JSONL event per line; do not pretty-print.

## Redaction (mandatory before committing)

1. Replace real `thread_id` values with `th_REDACTED`.
2. Strip any absolute paths, usernames, repo names, hostnames, and API
   identifiers from `command`, `aggregated_output`, `text`, and `arguments`.
3. Remove any `command_execution` output that echoes secrets or environment.
4. Keep token counts (`usage`) — they are not sensitive and exercise the
   metrics path.
5. Re-run the consuming test to confirm redaction did not break the JSON.

## Shared with the #154 drift guard

This corpus is the **single** recorded-real fixture set. The #154 CLI-drift
guard replays these same files to detect when the live `codex` JSON shape moves
out from under the parser. Do not fork a second corpus for drift detection — add
or refresh fixtures here and both #151's real-path test and #154's drift guard
consume them.
