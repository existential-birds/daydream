# Real codex CLI golden fixture

`golden.jsonl` is a **real** capture of `codex exec --experimental-json` output
(codex CLI 0.139.0), committed so the parser can be checked against genuine CLI
output rather than synthesized fixtures. It is NOT hand-written; do not edit it
by hand.

## How it was captured

- **CLI:** `codex` 0.139.0 (`codex --version`).
- **Command:** `echo "<prompt>" | codex exec --experimental-json --sandbox read-only`
  (prompt read from stdin, read-only sandbox so nothing mutates the sample repo).
- **Model:** the account default at capture time (`gpt-5.5`).
- **Sample repo:** `tests/fixtures/real_cli_sample_repo/` (a 2-file git repo:
  `README.md` + `hello.py`).
- **Prompt:** "Read README.md, then read hello.py, then describe both files in
  one sentence each."
- **Result:** 8 JSONL lines — `thread.started`, `turn.started`, two
  `command_execution` start/completed pairs (ids `item_0`, `item_1`), an
  `agent_message` (`item_2`), and a `turn.completed` carrying `usage` with
  `input_tokens` / `cached_input_tokens` / `output_tokens` /
  `reasoning_output_tokens`.

To re-capture, run `scripts/capture-codex-golden.sh` and commit the refreshed
file (mention the codex CLI version in the commit message).

## When to refresh

- After a `codex` CLI version bump.
- After any change to the JSONL parser in `daydream/backends/codex.py`.
- When intentionally updating the structural golden to a new CLI/model shape.

## What the contract test asserts

`tests/test_codex_real_cli_contract.py::test_real_golden_parses_to_expected_events`
drives `CodexBackend.execute` through this golden and asserts **structural**
coverage (not byte-exact text, so a model swap that rewords the agent message
still passes): a text span, two paired tool calls with **zero orphans** (the
#153 deterministic-correlation contract re-asserted on real data), a
`MetricsEvent` with prompt/completion tokens and `cached_tokens` surfaced from
`cached_input_tokens`, and a `ResultEvent`. If the parser or the CLI shape
drifts, this test fails.
