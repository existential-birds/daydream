# Codex JSONL fixtures

`codex` CLI experimental-JSON streams replayed through the **real**
`CodexBackend` parser in tests. The mock subprocess
(`tests/harness/codex_replay.py:make_mock_process`) yields these lines through
`stdout.readline()` exactly as the live CLI would; only the subprocess boundary
is stubbed, so the genuine JSONL parser in `daydream/backends/codex.py` runs end
to end.

## Recorded-real vs. synthesized — current state

- **Synthesized** fixtures are generated from canonical scripts by
  `tests/contract/_loaders.py:_build_codex_jsonl` (and the harness wrappers in
  `tests/harness/scripts.py`). They are deterministic and easy to author, but by
  construction they **cannot surprise the parser with real-CLI shapes** — a test
  that replays synthesized bytes proves the parser handles bytes *we invented*,
  not bytes `codex` actually emits.
- **Recorded-real** fixtures would be captured from genuine `codex` CLI runs,
  then redacted, to guard against the "Codex Backend Gotchas" the synthesizer
  does not reproduce (agent/reasoning text via `item.updated` deltas with empty
  `item.completed` content; `output_text` content blocks; structured payloads on
  `turn.completed.result`/`output`).

> **There are currently NO recorded-real fixtures in this directory.** A prior
> `realpath_parse.jsonl` was committed and labelled "recorded-real" but was in
> fact hand-authored (fake `th_REDACTED` thread id, mirrored the test's invented
> repo content, used a `content:[{output_text}]` shape real `codex` does not
> emit). It was removed. Real-path Codex coverage (#151) must be rebuilt on
> genuinely captured streams — see the capture procedure below. Do not
> reintroduce a synthesized fixture under a recorded-real name.

## Capturing a fresh fixture (genuine only)

The backend launches the CLI as (see `daydream/backends/codex.py`):

```
codex exec --experimental-json --skip-git-repo-check \
  [--output-schema <schema.json>] -
```

Capture against a throwaway repo, prompt fed on stdin (closed immediately),
JSONL on stdout:

```bash
printf '%s' "$PROMPT" | \
  codex exec --experimental-json --skip-git-repo-check \
    --output-schema /tmp/feedback_schema.json - \
  > /tmp/realpath_parse.jsonl   # capture to /tmp first, redact, then move into place
```

- **Use the default model — omit `-m`.** On a ChatGPT-account login, `-m
  gpt-5-codex` and `-m gpt-5` are rejected (`model is not supported when using
  Codex with a ChatGPT account`); the default model returns a full turn.
- `--output-schema` is only for the PARSE phase (it constrains the agent to
  `FEEDBACK_SCHEMA`, defined in `daydream/phases.py`); omit it for
  REVIEW/FIX/TEST captures.
- One JSONL event per line; do not pretty-print.
- A genuine capture has a real UUID `thread_id` (e.g.
  `019ea266-37c6-7090-8981-60748e3929d1`) and a real `turn.completed.usage`
  block. If those are absent, it was not captured from a real run.

## Redaction (mandatory before committing)

1. Replace real `thread_id` values with a fresh UUID-shaped placeholder (NOT a
   literal `th_REDACTED` — keep the real shape so the fixture stays faithful).
2. Strip any absolute paths, usernames, repo names, hostnames, and API
   identifiers from `command`, `aggregated_output`, `text`, and `arguments`.
3. Remove any `command_execution` output that echoes secrets or environment.
4. Keep token counts (`usage`) — they are not sensitive and exercise the
   metrics path.
5. Re-run the consuming test to confirm redaction did not break the JSON.

## Shared with the #154 drift guard

When recorded-real fixtures exist, this corpus is the **single** recorded-real
fixture set. The #154 CLI-drift guard replays these same files to detect when the
live `codex` JSON shape moves out from under the parser. Do not fork a second
corpus for drift detection — add or refresh fixtures here and both #151's
real-path test and #154's drift guard consume them.
