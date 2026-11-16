# Context: #1816 + #1818 in one PR (Codex adapter)

Prepared 2026-11-16 without changing the product code. Line numbers are from base
`c41e07c`; confirm them again before editing.

## Sources

- [#1816](https://github.com/existential-birds/daydream/issues/1816): Codex ignores
  `persist_session=False` and still writes session history.
- [#1818](https://github.com/existential-birds/daydream/issues/1818): (1) the
  process-exit excerpt drops Codex's own cause line; (2) MCP tool results are recorded
  as `str(content)`, a Python repr.
- Both issues are follow-ups from [#1646](https://github.com/existential-birds/daydream/issues/1646),
  delivered in PR #1819. Neither issue has comments, so the acceptance criteria in the
  issue bodies are current. Both require extending existing tests with **no new test
  functions**.
- Since then, #1834 made committed review the only reviewer, so `persist_session=False`
  now covers every committed-review Codex assignment.

## Handoff location

This folder is committed only on the handoff branch `claude/peaceful-gates-ri0icp`,
force-added past the `plans/` ignore rule. It is **not** on `main` and must never enter
the implementation PR.
- Restore it into an implementation checkout as ignored, unstaged files:
  `git fetch origin claude/peaceful-gates-ri0icp && git restore --source=FETCH_HEAD --worktree -- plans/1816-1818-codex`
- Never `git add -f` it in the implementation branch.

## Baseline (fetched 2026-11-16)

- Repo: github.com/existential-birds/daydream.
- Implementation base: `origin/main` at `c41e07c1903bfecdf33b64febe615740370f35df`
  (#1913). Fetch again; if `main` moved, base on the newer SHA and record it.
- The preparation container had no `.venv`, so `make install` creates one. There was no
  `codex` on PATH.
- Commit signing there was SSH via `/usr/local/bin/ssh-sign-dispatch`. The git hooks
  were not installed by default, so run `make hooks` before committing.
- `/opt/node22/bin/pi` existed. Whether Pi provider auth works for the Daydream review
  is unverified; if it does not, that is a blocker to report, not a pass.

## Decisions

**D1 (#1816).** When `persist_session` is False, add `--ephemeral` to the `codex exec`
argv. The argv is built at codex.py:944-964; put the flag before any
`resume`/`-c`/host overrides. Persisted argv stays byte-for-byte unchanged.

**D2.** When `persist_session` is False, do not resume and return no
`ContinuationToken`. This mirrors Pi (pi.py:968, 1150) and Claude (claude.py:1404/1435).
- Today Codex builds the token at codex.py:1363-1370 whenever there is a thread_id and
  no disposable checkout. Its thread would not exist after an ephemeral run: resuming
  fails with `thread/resume failed: no rollout found`.
- `--ephemeral` combined with `resume` still appends to the existing rollout. So
  persist=False must ignore a Codex continuation, as Pi does, and never send both.
- Compute the resume decision once and reuse it for argv, `continuation_mode`,
  `session_id`/`session_source` and the read-only resume refusal (~codex.py:920).
- No current caller combines persist=False with a continuation:
  - persist=False callers are committed_review.py:1653, improve/* (Claude-only in
    practice) and the agent.py:423/665 passthrough. All of them discard the token.
  - Continuation users run on the default True (deep/diagram_steps.py,
    phases/test_evidence.py, phases/testing.py).

**D3.** Pass `persist_session=persist_session` into `CodexRequestConfig(...)` at
codex.py:1032. The shared field is `EffectiveRequestConfig.persist_session`
(backends/__init__.py:476) and is surfaced through `request_config_controls`. Delete the
comment at codex.py:1015-1018 and fix the docstring at codex.py:767-768 ("accepted but
ignored").

**D4 (#1818.1).** Change only the Codex accumulation at codex.py:1072-1082 and leave the
shared `_transport.process_exit_message` alone; Pi uses it too.
- Keep counting every non-JSON line.
- Skip blank lines.
- Once a line `.strip().casefold()` equals `stack backtrace:` (anyhow's report header),
  stop retaining later lines.
- Real output for all 4 versions is in `samples/`: the cause is on both
  `ERROR codex_core::session: ... required MCP servers failed to initialize: <name>` and
  `Error: thread/start: ...`, followed by a blank line, `Stack backtrace:` and 8-12
  `N: <unknown>` frames. Today the message shows only frames.
- Simulated over the real outputs, the new rule keeps the cause on all 4 versions in a
  847-1299 char message.
- Keep these invariants:
  - redact the full line, then cap it at 256 chars (`_bounded_process_excerpt`);
  - retain at most 20 lines and print at most 10;
  - diagnostics and logs carry only the count, never the payload.
- AUTH_CONFIG classification (codex.py:440) reads this message, so it also benefits.

**D5 (#1818.2).** Add a small Codex helper used at codex.py:1248. Rules, in order:
1. A list of only `{"type":"text","text":str}` blocks → `"\n".join(texts)`. An empty
   list gives `""`.
2. A str → unchanged.
3. None or missing → `""`.
4. Anything else, including non-text or mixed blocks →
   `json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.

Ignore `structured_content` when text exists, because FastMCP sends both.
- Use `\n`, not Pi's `""`: host results are 2 text blocks (metadata JSON + content,
  host_tools.py:698), and gluing them would be unreadable.
- Do not change Pi's `_render_tool_result`. Its native shape (`details`) and separator
  differ.
- Real Codex item shape, the same on all 4 versions:
  `"result":{"content":[{"type":"text","text":"refused"},{"type":"text","text":"second"}],"structured_content":null},"error":null,"status":"failed"`
- Real items never carry `isError`. Leave that check alone; it is out of scope.
- Empty content now records `""` instead of `"[]"`. That is truthful, and it trips
  `blank_result`/`empty_tool_result` (trajectory/summary.py:94, eval/analyzer.py:524),
  which is correct for an empty result.
- No consumer parses the repr; host admission is tracked on the binding.

**D6 docs.**
- docs/runtime.md "Codex host tools":
  - ~1000-1003: "...and Codex's own session files". Ephemeral runs write no session
    files; the shell snapshot still exists while the run is going and is removed at
    exit (true on all 4 versions).
  - ~1010-1011, Readiness: the raised error keeps Codex's cause line.
  - ~1055-1057, "Not offered": remove the session-history clause. Instead, state that
    non-persisted invocations pass `--ephemeral`, so Codex writes no rollout or thread
    rows, while its logs/state databases, `installation_id` and shell snapshots remain
    its own.
- docs/observability.md "Codex host-tool lifecycle evidence" (~line 167): one sentence
  saying MCP results record their text blocks newline-joined, else canonical JSON.
- AGENTS.md stays accurate ("never the Codex home"; the shell-snapshot note).

**D7.** CHANGELOG `[Unreleased]` → `### Fixed`: one bullet per behavior.
- Mirror the Codex entry style at CHANGELOG.md ~1144 and link the issues.
- Say that the recorded tool-result evidence changes for every Codex invocation.
- SemVer: patch-level fixes, no API break. State that in the PR.

## Real-build evidence (preparation, offline)

The probe used npm builds 0.139.0, 0.153.4, 0.163.0 and 0.169.0 with a fake local
Responses provider, no login and no model calls, each case in its own `CODEX_HOME`.
The probe scripts are in `probe/` (`probe2.sh` = persist/ephemeral/resume matrix). Their
`P=` path points at the preparation scratchpad, so to re-run them:
1. Install each build: `npm install --prefix <dir>/v0_169_0 @openai/codex@0.169.0`
   (about 400 MB per version).
2. Start `fake_responses.py` on `127.0.0.1:18765`.
3. Set `P=<dir>` in the script.

- `--ephemeral` ("Run without persisting session files to disk") is accepted by `exec`
  and `exec resume` on all 4 versions, before or after `resume <id>`.
- Persisted and ephemeral runs produce identical event sequences, identical
  `turn.completed` usage and exit 0.
- An MCP call with `required=true` and approve produces the same item shape under
  `--ephemeral`.
- Ephemeral removes `sessions/**/rollout-*.jsonl`, the `threads` row in
  `state_N.sqlite`, and (0.153.4+) `thread_history_1.sqlite` and `thread-writer-locks/`.
  Neither the thread id nor the prompt appears in the home.
- Plain resume is unchanged: it appends to the same rollout, with cumulative usage.

Record this evidence in the PR Test Plan as the real-build confirmation that #1816
asks for, labelled as gathered during preparation. Re-running the probe is optional
corroboration, not a gate.

## Tests: extend only, no new functions (AGENTS.md testing standard, test-audit skill)

1. `test_continuation_token_resumes` / `_continuation_token_resumes_case`
   (tests/test_backend_codex.py:275-319), the argv test:
   - add a persist=False case: `--ephemeral` in argv, `config.persist_session is False`,
     no `resume` even when a Codex token is passed, and `result.continuation is None`;
   - in the existing resume cases, assert `--ephemeral` is absent and
     `config.persist_session is True`.
   Argv is captured by patching `daydream.backends._transport.asyncio.create_subprocess_exec`.
2. `test_parser_diagnostic_precedes_nonzero_process_exit` (:720-741): feed real-shaped
   lines (cause, blank, `Stack backtrace:`, more than 20 frames) along with the secret
   lines. Assert the cause is in the message, the count still covers every line, there
   is no secret or path, and the message is ≤3000 chars.
3. `test_native_codex_conversation_records_delivery_shapes_and_correlated_tools`
   (:880-1053), "refused" item at :932-937:
   - use 2 text blocks and assert `output == "refused\nsecond"`;
   - add one non-text-block case asserting canonical JSON, either on an existing item or
     an added item.
   - The trajectory is recorded in this same test (:1035-1041), so assert the
     observation content there too, not in a new test.
- No fixtures or goldens contain `mcp_tool_call`, so none need regenerating.
- `tests/harness/protocol_cli.py:113-146` allowlists Codex argv. Add `--ephemeral` only
  if a protocol_cli Codex path passes persist=False; none does today.
  `codex_lifecycle_cli` ignores unknown argv.

## Lanes

- A (one owner, both issues): `daydream/backends/codex.py` + `tests/test_backend_codex.py`.
- B: `docs/runtime.md`, `docs/observability.md`, `CHANGELOG.md`, after A fixes behavior.
- C (read-only): the `code-review` skill or an independent reviewer on the integrated
  diff, plus the test-audit skill on the changed tests.

## Commands (from Makefile/pyproject)

- `make install`, then `uv run pytest tests/test_backend_codex.py tests/test_codex_stream_lifecycle.py tests/test_committed_review.py -q`
- `make check` = lockcheck, install, static, test, coverage-report. It has a budget of
  under 2 minutes and the coverage floor is `fail_under=90`.
- `git diff --check`.

## Publication notes for this cloud environment

- Use the GitHub MCP tools for PR and feedback. `gh` is for search only here; if a
  skill needs `gh`, use the MCP equivalents.
- Use `.github/PULL_REQUEST_TEMPLATE.md` with every checklist item, and the
  `.agents/skills/pr` skill.
- Use `Fixes #1816` and `Fixes #1818`.
- Run the simplification sweep only if the PR exceeds 5 files or 500 lines; expect 5
  files.
