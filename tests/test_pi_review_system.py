"""Bounded review policy reaches Pi's effective system prompt per invocation."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.agent import run_agent
from daydream.backends.pi import PiBackend
from daydream.prompts.grounding import REVIEW_STOPPING_GUIDANCE
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.trajectory import DaydreamPhase
from tests.harness.pi_replay import make_mock_process, make_mock_process_from_fixture


async def test_pi_review_system_is_scoped_and_preserves_retry_and_stage_spend(tmp_path: Path) -> None:
    commands: list[tuple[str, ...]] = []
    system_prompts: list[str] = []

    async def spawn(*args: str, **kwargs: Any) -> Any:
        commands.append(args)
        value = args[args.index("--append-system-prompt") + 1]
        system_prompts.append(Path(value).read_text() if value.startswith("/") else value)
        if len(commands) == 3:
            return make_mock_process([json.dumps(event) for event in [
                {"type": "session", "sessionId": "failed-review"},
                *[{"type": "tool_execution_start", "toolCallId": path, "toolName": "read", "args": {"path": path}}
                  for path in ("api.py", "App.tsx")],
                {"type": "turn_end", "message": {"role": "assistant", "content": [],
                 "stopReason": "error", "errorMessage": "503 Service Unavailable"}},
            ]])
        return make_mock_process_from_fixture("simple_text.jsonl")

    backend = PiBackend(model="fixture-model", reasoning_effort="high")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = backend.retry_max_delay_s = 0
    budget = ReviewInvestigationBudget.from_limits(ReviewLimits(17, 3, 4))
    invocations: list[tuple[DaydreamPhase, dict[str, Any]]] = [
        (DaydreamPhase.DEEP, {"review_limits": ReviewLimits(17, 3, 3)}),
        (DaydreamPhase.FIX, {}),
        *[(DaydreamPhase.DEEP, {"investigation_budget": budget, "tool_call_budget": 4}) for _ in range(2)],
    ]
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=spawn):
        for phase, options in invocations:
            assert (await run_agent(backend, tmp_path, "Assigned work", phase=phase,
                                    progress_callback=lambda _: None, **options))[2] is None
    review_system, fix_system, initial, retry, integration = system_prompts
    paths = [args[args.index("--append-system-prompt") + 1] for args in commands]
    assert all(not Path(path).exists() for path in paths if path.startswith("/"))
    assert all(args[args.index("--thinking") + 1] == "high" for args in commands)
    assert "at most 17 seconds and 3 tool calls" in review_system
    assert REVIEW_STOPPING_GUIDANCE in review_system
    assert "Closed candidate decisions stay closed" not in review_system
    assert "repository-scoped" in review_system
    assert "Search before you read" not in review_system
    assert "typically 50" not in review_system
    assert REVIEW_STOPPING_GUIDANCE not in fix_system
    assert "Investigation allowance" not in fix_system
    assert budget.observed_tool_starts == 2 and budget.remaining_tool_calls == 2
    assert "4 remaining cumulative tool starts" in initial
    assert "Hard remaining cumulative reviewer allowance: 2 tool calls" in retry
    assert "2 remain for this reviewer after 2 observed starts" in retry
    assert "2 remaining cumulative tool starts" in integration
    for instructions in (initial, retry, integration):
        assert REVIEW_STOPPING_GUIDANCE not in instructions
        assert "Closed decisions stay closed" in instructions
        assert "by closed_candidate_ids" in instructions
        assert "Return exactly assigned target or triage candidate IDs" in instructions
        assert "host publishes terminal findings" in instructions


@pytest.mark.parametrize('native_fault', [None, 'read-truncated', 'first-line-truncated',
                                        'structured-truncated', 'native-exit-code'])
async def test_actual_pi_process_preserves_strict_root_and_native_capture_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_fault: str | None,
) -> None:
    import os
    import sys

    from daydream.agent import StructuredOutputFailure
    from daydream.phases.schemas import REVIEW_STAGE_SCHEMA
    from daydream.review_evidence import ReviewEvidence

    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    executable = bin_dir / 'pi'
    executable.write_text(
        f'#!{sys.executable}\n'
        'import json\nfrom pathlib import Path\n'
        f'NATIVE_FAULT = {native_fault!r}\n'
        'def emit(value): print(json.dumps(value), flush=True)\n'
        'def assistant(text):\n'
        '    message = {"role": "assistant", "content": [{"type": "text", "text": text}], '
        '"model": "fixture-model", "provider": "fixture", "stopReason": "stop"}\n'
        '    emit({"type": "message_end", "message": message})\n'
        '    emit({"type": "turn_end", "message": message})\n'
        'emit({"type": "session", "sessionId": "strict-root-fixture"})\n'
        'assistant("I will inspect the assigned source before judging.")\n'
        'emit({"type": "tool_execution_start", "toolCallId": "actual-source", '
        '"toolName": "read", "args": {"path": "api.py"}})\n'
        'source = Path(\"api.py\").read_text()\n'
        'result = {\"content\": [{\"type\": \"text\", \"text\": source}]}\n'
        'if NATIVE_FAULT == \"read-truncated\": result[\"details\"] = '
        '{\"truncation\": {\"truncated\": True}}\n'
        'if NATIVE_FAULT == \"first-line-truncated\": result[\"details\"] = '
        '{\"truncation\": {\"truncated\": False, \"firstLineExceedsLimit\": True}}\n'
        'if NATIVE_FAULT == \"structured-truncated\": result[\"structuredContent\"] = {\"truncated\": True}\n'
        'if NATIVE_FAULT == \"native-exit-code\": result[\"structuredContent\"] = {\"exit_code\": 7}\n'
        'emit({\"type\": \"tool_execution_end\", \"toolCallId\": \"actual-source\", '
        '\"isError\": False, \"result\": result})\n'
        'valid = {"targets": [{"target_id": "api.py", "status": "reviewed", "reason": ""}], '
        '"notes": "Reviewed assigned source", "candidates": [], "contradictions": []}\n'
        'assistant(json.dumps({\"invalid_outer\": valid} if NATIVE_FAULT is None else valid))\n'
        'emit({"type": "agent_end", "messages": []})\n',
    )
    executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(bin_dir) + os.pathsep + os.environ['PATH'])
    (tmp_path / 'api.py').write_text('VALUE = 1\n')
    budget = ReviewInvestigationBudget.from_limits(ReviewLimits(investigation_s=30, finalization_s=0, tool_calls=8))
    evidence = ReviewEvidence(REVIEW_STAGE_SCHEMA)
    evidence.configure_capture(tmp_path)
    output, _, reason = await run_agent(
        PiBackend(model='fixture-model'), tmp_path, 'Review the changed value in api.py.',
        phase=DaydreamPhase.DEEP, output_schema=REVIEW_STAGE_SCHEMA, require_full_schema=True,
        investigation_budget=budget, review_evidence=evidence, progress_callback=lambda _: None,
    )
    assert reason is None
    if native_fault is None:
        assert isinstance(output, StructuredOutputFailure)
        assert output.reason == 'malformed_output' and output.rejection is not None
        assert evidence.capture_failure(['api.py']) is False
    else:
        assert isinstance(output, dict) and output['targets'][0]['status'] == 'reviewed'
        assert evidence.capture_failure(['api.py']) is True
        receipt, = evidence.receipts
        if native_fault == 'native-exit-code':
            assert receipt.result.exit_code == 7
        else:
            assert receipt.result.truncated is True and evidence.native_truncated_results == 1
    assert budget.observed_tool_starts == 1 and budget.remaining_tool_calls == 7
