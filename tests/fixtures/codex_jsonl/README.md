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
- **Recorded-real** fixtures are genuine `codex` CLI captures, redacted before
  commit, guarding against real-CLI shapes the synthesizer's canonical scripts
  may not cover (agent/reasoning text via `item.updated` deltas with empty
  `item.completed` content; `output_text` content blocks; structured payloads on
  `turn.completed.result`/`output`).

> **There is one recorded-real fixture: `real/golden.jsonl`** — a genuine
> capture of `codex exec --experimental-json` (CLI 0.139.0) against the tiny
> sample repo, committed via the capture script (`scripts/capture-codex-golden.sh`)
> and consumed by `tests/test_codex_real_cli_contract.py`. See
> [`real/README.md`](real/README.md) for the capture details. A prior
> `realpath_parse.jsonl` was committed and labelled "recorded-real" but was in
> fact hand-authored (fake `th_REDACTED` thread id, mirrored the test's invented
> repo content, used a `content:[{output_text}]` shape real `codex` does not
> emit); it was removed. Do not reintroduce a synthesized fixture under a
> recorded-real name — refresh `real/golden.jsonl` with the capture script
> instead.

## Capturing a fresh fixture (genuine only)

The backend launches the CLI as (see `daydream/backends/codex.py`):

```bash
codex exec --experimental-json --model <model> \
  --sandbox <read-only|danger-full-access> --cd <cwd> \
  [-c <effort>] [--output-schema <schema.json>]
```

The prompt is fed on stdin (closed immediately), JSONL on stdout. The capture
script (`scripts/capture-codex-golden.sh`) runs the simplest faithful form —
`codex exec --experimental-json --sandbox read-only` with the prompt on stdin:

```bash
printf '%s' "$PROMPT" | \
  codex exec --experimental-json --sandbox read-only \
    --cd "$SAMPLE_REPO" \
  > /tmp/realpath_parse.jsonl   # capture to /tmp first, redact, then move into place
```

- **The backend always pins `--model`** (`codex.py:131-132`); daydream's
  default is `gpt-5.6-sol` (`daydream/config.py`). A manual capture without
  `--model` uses the account default, which is what `scripts/capture-codex-golden.sh`
  relies on. Verified June 2026 against `codex`
  0.137.0 with a ChatGPT-account login: the configured default is `gpt-5.5`
  (`~/.codex/config.toml`), which returns a full turn. Explicitly supported IDs
  on a ChatGPT login are `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini` (and
  `gpt-5.3-codex-spark` for ChatGPT Pro). The legacy `-m gpt-5-codex` / `-m
  gpt-5` are rejected (`model is not supported when using Codex with a ChatGPT
  account`), as are `gpt-5.2` / `gpt-5.3-codex` (API-key auth only).
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
