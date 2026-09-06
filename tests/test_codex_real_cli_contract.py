"""Contract checks against REAL codex CLI output (parser-drift guard).

The historical parser-drift contract remains paired with a newer, separately
provenanced generic-tool omission capture:

1. ``test_real_golden_parses_to_expected_events`` (always-on): drives
   ``CodexBackend.execute`` through a committed golden fixture derived from
   REAL ``codex exec --experimental-json`` output (codex 0.139.0). Asserts
   the parser produces a structurally-correct ``AgentEvent`` stream with zero
   orphaned tool results — the #153 contract re-asserted on real data. This
   catches parser drift the next time someone re-captures the golden and the
   CLI shape has changed. Structural assertions are version-robust (not
   byte-exact) so a model/CLI swap that rewords the agent text still passes.

2. ``test_codex_live_smoke`` (Layer 2, opt-in): marked ``live_codex`` and
   skipped by default. Set ``DAYDREAM_CODEX_LIVE=1`` to opt in (mirrors the
   pi smoke test's ``DAYDREAM_PI_LIVE=1`` gate — binary presence alone is
   not sufficient because an unauthenticated codex exits non-zero). When
   trivial review against the in-repo sample repo and asserts a non-empty
   ``AgentEvent`` stream + clean trajectory, logging any unrecognized JSONL
   event types.

The golden is committed at ``tests/fixtures/codex_jsonl/real/golden.jsonl``;
use ``scripts/capture-codex-golden.sh`` to re-capture it.
The sanitized codex 0.153.4 public omission capture and its raw/sanitized
digests live beside it; its dedicated maintenance script creates only an
unpublished public candidate for later exact-run corroboration and review.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.backends import (
    DiagnosticEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends.codex import CodexBackend
from tests.harness.codex_replay import FIXTURES_DIR, make_mock_process_from_fixture

REAL_GOLDEN = "real/golden.jsonl"
REAL_FILE_CHANGE = "real/file-change.jsonl"
REAL_GENERIC_TOOL_FAILURE = "real/generic-tool-failure.jsonl"
REAL_GENERIC_TOOL_FAILURE_META = "real/generic-tool-failure.meta.json"

_logger = logging.getLogger(__name__)

# Opt-in gate for the live codex smoke test. Mirrors the pi smoke test's
# DAYDREAM_PI_LIVE=1 pattern: codex being on $PATH is NOT sufficient —
# with the binary installed but not authenticated, `codex exec` exits
# non-zero, producing a false red. Require explicit opt-in so the test
# is skipped by default and only runs when a human explicitly enables it.
_CODEX_LIVE_OPT_IN = os.environ.get("DAYDREAM_CODEX_LIVE") == "1"
_CODEX_AVAILABLE = shutil.which("codex") is not None


def test_generic_tool_transport_capture_has_truthful_sanitized_provenance() -> None:
    """The genuine public capture is immutable apart from its declared path redaction."""
    capture_script = Path(__file__).parents[1] / "scripts" / "capture-codex-generic-tool-failure.sh"
    fixture_path = FIXTURES_DIR / REAL_GENERIC_TOOL_FAILURE
    metadata_path = FIXTURES_DIR / REAL_GENERIC_TOOL_FAILURE_META
    fixture_bytes = fixture_path.read_bytes()
    metadata = json.loads(metadata_path.read_text())
    records = [json.loads(line) for line in fixture_bytes.splitlines()]

    assert capture_script.is_file() and os.access(capture_script, os.X_OK)
    assert hashlib.sha256(fixture_bytes).hexdigest() == metadata["fixture_sha256_sanitized"]
    assert metadata["capture_sha256_raw"] == (
        "896389a5da72401e74ee50324bedd745c513cb7468fa5dcb0fbf682e4b125683"
    )
    assert metadata["capture_sha256_raw"] != metadata["fixture_sha256_sanitized"]
    assert metadata["fixture_sanitized"] is True
    assert metadata["sanitizations"] == [
        {
            "field": "item.message",
            "occurrences": 1,
            "replacement": "/Users/[REDACTED_USER]/.codex/config.toml",
            "reason": "personal_home_path",
        }
    ]
    assert [
        record["type"]
        if record["type"] != "item.completed"
        else f"item.completed/{record['item']['type']}"
        for record in records
    ] == [
        "thread.started",
        "item.completed/error",
        "turn.started",
        "item.completed/error",
        "item.completed/agent_message",
        "item.completed/agent_message",
        "turn.completed",
    ]
    assert metadata["ordered_public_shape"] == [
        "thread.started",
        "item.completed/error",
        "turn.started",
        "item.completed/error",
        "item.completed/agent_message",
        "item.completed/agent_message",
        "turn.completed",
    ]
    assert metadata["cli_version"] == "0.153.4"
    assert metadata["captured_at_utc"] == "2026-09-06T02:31:50Z"
    assert metadata["capture_working_directory"] == "disposable_git_repo"
    assert metadata["public_flags"] == ["--json", "--sandbox", "read-only", "--enable", "code_mode"]
    assert metadata["exit_code"] == 0
    assert metadata["public_generic_function_items_exposed"] is False
    assert metadata["private_rollout_correlated"] is True
    assert metadata["private_custom_tool_calls"] == [
        {"name": "exec", "count": 1, "matching_outputs": 1, "error_class": "TypeError"}
    ]
    assert metadata["probe_prompt"] == (
        "This is a transport-contract probe. You MUST invoke the functions.exec tool exactly once "
        "with this exact JavaScript body: const r = await tools.wait({cell_id:\"definitely-missing-cell\","
        "yield_time_ms:250}); text(r); Do not call any other tool. Do not claim success unless you "
        "receive the tool output. Afterward report whether the nested wait call was visible to you "
        "and include its error category without inventing it."
    )

    public_items = [record["item"] for record in records if "item" in record]
    assert [item["type"] for item in public_items] == [
        "error",
        "error",
        "agent_message",
        "agent_message",
    ]
    for item in public_items:
        assert not ({"name", "call_id", "arguments", "output"} & item.keys())

    fixture_text = fixture_bytes.decode()
    metadata_text = metadata_path.read_text()
    assert "/Users/ka" not in fixture_text
    assert "/Users/ka" not in metadata_text
    assert "definitely-missing-cell" not in fixture_text
    for forbidden_key in {
        "private_rollout_bytes",
        "private_session_path",
        "home_path",
        "environment_context",
        "credentials",
        "call_id",
        "arguments",
        "output_text",
    }:
        assert forbidden_key not in metadata


def test_capture_script_publishes_only_public_candidate_without_replacing_fixture(
    tmp_path: Path,
) -> None:
    """A fake external CLI proves candidate capture is non-publishing and public-only."""
    capture_script = Path(__file__).parents[1] / "scripts" / "capture-codex-generic-tool-failure.sh"
    committed_fixture = FIXTURES_DIR / REAL_GENERIC_TOOL_FAILURE
    committed_metadata = FIXTURES_DIR / REAL_GENERIC_TOOL_FAILURE_META
    before_fixture = committed_fixture.read_bytes()
    before_metadata = committed_metadata.read_bytes()

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_codex = fake_bin / "codex"
    fake_codex.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "${1:-}" == "--version" ]]; then\n'
        "  printf '%s\\n' 'codex-cli 9.9.9'\n"
        "  exit 0\n"
        "fi\n"
        # Consume the real CLI's stdin contract before returning output. Exiting
        # without reading races the producer and can make pipefail report SIGPIPE.
        "IFS= read -r probe_prompt\n"
        'printf \'%s\\n\' "$probe_prompt" > "$FAKE_CODEX_PROBE_PATH"\n'
        'if [[ "${FAKE_CODEX_EXPOSE_PAIR:-}" == "1" ]]; then\n'
        "  printf '%s\\n' \\\n"
        "    '{\"type\":\"item.completed\",\"item\":{\"type\":\"error\"}}' \\\n"
        "    '{\"type\":\"item.completed\",\"item\":{"
        "\"type\":\"custom_tool_call\"}}' \\\n"
        "    '{\"type\":\"turn.completed\"}'\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \\\n"
        "  '{\"type\":\"thread.started\",\"thread_id\":\"candidate-thread\"}' \\\n"
        "  '{\"type\":\"item.completed\",\"item\":{\"id\":\"warning\","
        "\"type\":\"error\",\"message\":\"see /Users/fake-person/.codex/config.toml\"}}' \\\n"
        "  '{\"type\":\"turn.started\"}' \\\n"
        "  '{\"type\":\"turn.completed\",\"usage\":{\"input_tokens\":1,"
        "\"output_tokens\":1}}'\n"
    )
    fake_codex.chmod(0o755)
    candidate_root = tmp_path / "candidates"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["DAYDREAM_CODEX_CAPTURE_ROOT"] = str(candidate_root)
    probe_path = tmp_path / "received-probe.txt"
    env["FAKE_CODEX_PROBE_PATH"] = str(probe_path)
    env.pop("DAYDREAM_CODEX_PRIVATE_CORROBORATED", None)
    env.pop("DAYDREAM_CODEX_PRIVATE_ERROR_CLASS", None)

    captured = subprocess.run(
        [str(capture_script)],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert captured.returncode == 0, captured.stderr
    candidates = [path for path in candidate_root.iterdir() if not path.name.startswith(".")]
    assert len(candidates) == 1
    candidate_fixture = candidates[0] / "public.jsonl"
    candidate_metadata = candidates[0] / "public.meta.json"
    assert candidate_fixture.is_file() and candidate_metadata.is_file()
    metadata = json.loads(candidate_metadata.read_text())
    assert metadata["cli_version"] == "9.9.9"
    assert metadata["publication_status"] == "candidate_unpublished"
    assert probe_path.read_text() == metadata["probe_prompt"] + "\n"
    assert metadata["review_requirements"] == [
        "privately_correlate_this_exact_candidate_run_before_fixture_update"
    ]
    assert not [key for key in metadata if key.startswith("private_")]
    assert hashlib.sha256(candidate_fixture.read_bytes()).hexdigest() == metadata["fixture_sha256_sanitized"]
    assert "/Users/fake-person" not in candidate_fixture.read_text()
    assert committed_fixture.read_bytes() == before_fixture
    assert committed_metadata.read_bytes() == before_metadata

    rejected_root = tmp_path / "rejected-candidates"
    env["DAYDREAM_CODEX_CAPTURE_ROOT"] = str(rejected_root)
    env["FAKE_CODEX_EXPOSE_PAIR"] = "1"
    rejected = subprocess.run(
        [str(capture_script)],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "update the parser contract" in rejected.stderr
    assert not [path for path in rejected_root.iterdir() if not path.name.startswith(".")]
    assert committed_fixture.read_bytes() == before_fixture
    assert committed_metadata.read_bytes() == before_metadata


@pytest.mark.asyncio
async def test_generic_tool_transport_capture_maps_only_public_error_sentinel() -> None:
    backend = CodexBackend(model="gpt-5.5")
    mock_proc = make_mock_process_from_fixture(REAL_GENERIC_TOOL_FAILURE)

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = [event async for event in backend.execute(Path("/tmp"), "transport contract")]

    diagnostics = [event for event in events if isinstance(event, DiagnosticEvent)]
    assert [event.code for event in diagnostics] == [
        "codex_transport_coverage",
        "codex_transport_coverage",
    ]
    assert diagnostics[0].metadata["occurrences"] == 1
    assert diagnostics[-1].metadata == {
        "coverage": "incomplete",
        "reason": "uncorrelated_public_error_item",
        "occurrences": 2,
        "contract": "codex-cli-0.153.4-json-code-mode-v1",
    }
    assert not [event for event in events if isinstance(event, (ToolStartEvent, ToolResultEvent))]


@pytest.mark.asyncio
async def test_real_golden_has_no_parser_or_transport_diagnostic() -> None:
    backend = CodexBackend(model="gpt-5.5")
    mock_proc = make_mock_process_from_fixture(REAL_GOLDEN)

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = [event async for event in backend.execute(Path("/tmp"), "golden contract")]

    assert not [event for event in events if isinstance(event, DiagnosticEvent)]


@pytest.mark.asyncio
async def test_real_golden_parses_to_expected_events() -> None:
    """The committed REAL codex golden parses to a structurally-correct stream.

    The golden is genuine ``codex exec --experimental-json`` output (codex
    0.139.0). This asserts the parser still agrees
    with the live CLI on the observed event coverage: a text span, paired
    tool calls (zero orphans — the #153 contract), per-turn metrics with
    prompt/completion tokens, and a result event. Assertions are structural,
    not byte-exact, so re-capturing on a new model that rewords the agent
    message still passes.
    """
    assert (FIXTURES_DIR / REAL_GOLDEN).exists(), (
        f"real golden missing at {FIXTURES_DIR / REAL_GOLDEN}"
    )

    backend = CodexBackend(model="gpt-5.5")
    mock_proc = make_mock_process_from_fixture(REAL_GOLDEN)

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events: list[Any] = []
        async for event in backend.execute(Path("/tmp"), "Read README.md then hello.py"):
            events.append(event)

    # Text span: agent_message produced at least one TextEvent.
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert text_events, "real golden produced no TextEvent (agent_message lost)"

    # Tool spans: the golden has TWO command_execution pairs (README + hello.py).
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_starts) == 2, f"expected 2 tool starts, got {len(tool_starts)}"
    assert len(tool_results) == 2, f"expected 2 tool results, got {len(tool_results)}"

    # Zero orphans: every ToolResultEvent.id pairs with a ToolStartEvent.id.
    # This re-asserts the #153 deterministic-correlation contract on REAL data.
    start_ids = {e.id for e in tool_starts}
    result_ids = {e.id for e in tool_results}
    assert result_ids == start_ids, (
        f"orphaned tool result on real data: starts={start_ids} results={result_ids}"
    )

    # Per-turn metrics: turn.completed yields a MetricsEvent with prompt AND
    # completion tokens, and cached_tokens surfaced from cached_input_tokens.
    metrics_events = [e for e in events if isinstance(e, MetricsEvent)]
    assert metrics_events, "real golden produced no MetricsEvent (usage lost)"
    mev = metrics_events[0]
    assert mev.prompt_tokens is not None and mev.prompt_tokens > 0, (
        f"real usage input_tokens not surfaced: {mev.prompt_tokens}"
    )
    assert mev.completion_tokens is not None and mev.completion_tokens > 0, (
        f"real usage output_tokens not surfaced: {mev.completion_tokens}"
    )
    assert mev.cached_tokens is not None and mev.cached_tokens > 0, (
        f"real cached_input_tokens not surfaced: {mev.cached_tokens}"
    )
    # #192: reasoning_output_tokens must surface (real golden carries 79 —
    # 35% of output is reasoning, invisible before this change). Subset of
    # completion_tokens, NOT additive.
    assert mev.reasoning_tokens is not None and mev.reasoning_tokens > 0, (
        f"real reasoning_output_tokens not surfaced: {mev.reasoning_tokens}"
    )
    # #194: gpt-5.5 is in MODEL_PRICES → cost is synthesized at the backend
    # layer (D-16 reversed). The golden's tokens are non-trivial, so the
    # synthesized cost must be non-None and strictly positive.
    assert mev.cost_usd is not None and mev.cost_usd > 0, (
        f"real golden cost not synthesized for gpt-5.5: {mev.cost_usd}"
    )

    # Result event present (turn.completed → ResultEvent with continuation).
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert result_events, "real golden produced no ResultEvent"


@pytest.mark.asyncio
async def test_real_file_change_parses_to_expected_events() -> None:
    """The committed REAL file_change capture parses to the new-shape event stream.

    Mirrors ``test_real_golden_parses_to_expected_events`` but for a ``patch``
    (file_change) item: the parser must emit a ToolStart("patch") whose
    ``input["changes"]`` paths match the fixture's ``changes`` keys
    (repo-relative) and a ToolResult with ``status == "completed"``.

    **Blocked on a real capture** (issue #1125 Task 0 spike): the sandbox
    environment has no OpenAI credentials, so ``codex exec`` could not
    authenticate and no real ``file_change`` JSONL was captured. Per the plan
    contract, a "real" fixture must never be synthesized — so the fixture is
    simply absent and this test skips until a genuine capture lands at
    ``tests/fixtures/codex_jsonl/real/file-change.jsonl`` (re-capture via
    ``scripts/capture-codex-golden.sh``-style flow against an authenticated
    CLI). The assertions below run unmodified once the fixture exists.
    """
    fixture_path = FIXTURES_DIR / REAL_FILE_CHANGE
    if not fixture_path.exists():
        pytest.skip(
            "real file_change capture not yet available "
            f"({fixture_path}); Task 0 spike could not capture (no OpenAI auth)"
        )

    # The fixture is genuine codex output: extract the changes keys the CLI
    # reported so the assertion compares parser output against ground truth.
    changes_keys: list[str] = []
    for line in fixture_path.read_text().splitlines():
        if not line.strip():
            continue
        evt = json.loads(line)
        item = evt.get("item") or {}
        if item.get("type") == "file_change" or (
            evt.get("type") == "item.completed" and item.get("item_type") == "file_change"
        ):
            changes = item.get("changes") or {}
            changes_keys.extend(changes.keys())
    assert changes_keys, "fixture has a file_change item with no changes keys"

    # The parser normalizes absolute paths under execution_cwd to repo-relative
    # (codex.py file_change branch: commonpath/relpath), so the fixture's raw
    # changes keys must undergo the same transform before comparison. Mirror the
    # parser's guards: keys outside the cwd stay absolute, disjoint-drive
    # ValueError keeps them absolute too.
    execution_cwd = "/tmp"
    normalized_keys: list[str] = []
    for key in changes_keys:
        try:
            if os.path.isabs(key) and os.path.commonpath([key, execution_cwd]) == execution_cwd:
                normalized_keys.append(os.path.relpath(key, execution_cwd))
            else:
                normalized_keys.append(key)
        except ValueError:
            normalized_keys.append(key)

    backend = CodexBackend(model="gpt-5.5")
    mock_proc = make_mock_process_from_fixture(REAL_FILE_CHANGE)

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events: list[Any] = []
        async for event in backend.execute(Path("/tmp"), "apply the patch"):
            events.append(event)

    patch_starts = [e for e in events if isinstance(e, ToolStartEvent) and e.name == "patch"]
    assert patch_starts, "real file_change fixture produced no patch ToolStart"
    parsed_paths = [
        c["path"] for start in patch_starts for c in start.input.get("changes", [])
    ]
    # Compare against the cwd-normalized keys (parser output is relativized).
    assert sorted(parsed_paths) == sorted(normalized_keys), (
        f"parsed patch paths {parsed_paths} != fixture changes keys "
        f"(normalized) {normalized_keys}"
    )

    patch_ids = {e.id for e in patch_starts}
    patch_results = [e for e in events if isinstance(e, ToolResultEvent) and e.id in patch_ids]
    assert patch_results, "real file_change fixture produced no patch ToolResult"
    assert any(r.status == "completed" for r in patch_results), (
        f"no patch ToolResult with status 'completed': {[r.status for r in patch_results]}"
    )


@pytest.mark.live_codex
@pytest.mark.asyncio
@pytest.mark.skipif(
    not (_CODEX_AVAILABLE and _CODEX_LIVE_OPT_IN),
    reason="live codex smoke test; set DAYDREAM_CODEX_LIVE=1 (and ensure `codex` is on $PATH and logged in) to run",
)
async def test_codex_live_smoke() -> None:
    """Live smoke against the real codex binary — opt-in via DAYDREAM_CODEX_LIVE=1.

    Proves the subprocess seam, arg construction, and parser still agree with
    the live CLI. Skipped by default; set ``DAYDREAM_CODEX_LIVE=1`` to opt in
    (mirrors the pi smoke test's ``DAYDREAM_PI_LIVE=1`` gate — binary presence
    alone is not sufficient because an unauthenticated codex exits non-zero).
    Any unrecognized JSONL event type is logged at WARNING for triage.
    """
    sample_repo = Path(__file__).parent / "fixtures" / "real_cli_sample_repo"
    prompt = "Read README.md and summarize it in one sentence."

    proc = await asyncio.create_subprocess_exec(
        "codex",
        "exec",
        "--experimental-json",
        "--sandbox",
        "read-only",
        "--cd",
        str(sample_repo),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if proc.stdout is None or proc.stdin is None:
        pytest.skip("could not open codex stdio")
    proc.stdin.write(prompt.encode())
    proc.stdin.close()

    # Collect the raw JSONL lines, logging any line that is not valid JSON or
    # carries an unrecognized event type (the drift signal this test exists to
    # surface, per the issue's "log any unrecognized JSONL event types").
    known_types = {
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "item.started",
        "item.updated",
        "item.completed",
        "error",
    }
    # Failure event types that indicate a broken live run; their presence must
    # fail the test rather than pass false-green on a non-empty-but-failed stream.
    # Exception: quota/rate-limit failures are ENVIRONMENTAL, not code defects —
    # the test skips on those so a rate-limited dev machine doesn't go red.
    failure_types = {"turn.failed", "error"}
    quota_markers = ("usage limit", "rate limit", "quota", "credit", "upgrade to pro", "try again at")
    lines: list[str] = []
    failures: list[str] = []
    failure_messages: list[str] = []
    assert proc.stdout is not None
    while True:
        raw = await proc.stdout.readline()
        if not raw:
            break
        text = raw.decode().strip()
        if not text:
            continue
        try:
            evt = json.loads(text)
        except json.JSONDecodeError:
            _logger.warning("codex live smoke: non-JSON line: %r", text[:120])
            continue
        etype = evt.get("type", "")
        if etype not in known_types:
            _logger.warning("codex live smoke: unrecognized event type %r", etype)
        if etype in failure_types:
            failures.append(etype)
            # Capture the human-readable message for skip-vs-fail discrimination.
            msg = evt.get("message") or evt.get("error", {})
            if isinstance(msg, dict):
                msg = msg.get("message", "")
            if isinstance(msg, str) and msg:
                failure_messages.append(msg)
        lines.append(text)
    rc = await proc.wait()

    assert lines, "codex live smoke produced no JSONL output"

    # Environmental failures (quota / rate-limit / "try again at HH:MM") are
    # skip conditions, not regressions — the live binary ran, the seam works,
    # the account is just throttled. A genuine failure (parser drift, subprocess
    # seam break) has no quota marker and correctly fails the test.
    joined = " ".join(failure_messages).lower()
    if rc != 0 or failures:
        if any(marker in joined for marker in quota_markers):
            pytest.skip(f"codex live smoke hit an environmental limit (rc={rc}): {failure_messages[:1]}")
        assert rc == 0, f"codex live smoke exited non-zero: rc={rc}"
        assert not failures, f"codex live smoke reported failure events: {failures}"

    # Feed the live lines through the parser and assert a non-empty stream.
    backend = CodexBackend(model="live-smoke-model")
    from tests.harness.codex_replay import make_mock_process

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=make_mock_process(lines)):
        events: list[Any] = []
        async for event in backend.execute(sample_repo, prompt):
            events.append(event)

    assert events, "codex live smoke parsed to an empty AgentEvent stream"
