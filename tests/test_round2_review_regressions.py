"""Retain issue #1445 opt-out behavior through the Pi transport and host boundary."""

import pathlib

import pytest

import daydream.agent as agent_module


@pytest.mark.parametrize('validate', [True, False], ids=['schema-aware', 'opt-out'])
async def test_pi_backend_honors_structured_output_opt_out(tmp_path: pathlib.Path, validate: bool) -> None:
    """The host selects the empty result or largest span over Pi's text transport."""
    from unittest.mock import patch

    from daydream.backends.pi import PiBackend
    from daydream.phases.schemas import PER_STACK_RECORD_SCHEMA
    from daydream.trajectory import DaydreamPhase
    from tests.harness.pi_replay import FIXTURES_DIR
    from tests.harness.process_replay import make_mock_process_from_fixture

    process = make_mock_process_from_fixture(FIXTURES_DIR, 'issue1445_empty_results.jsonl', writable_stdin=validate)
    with patch('daydream.backends._transport.asyncio.create_subprocess_exec', return_value=process):
        result, _, reason = await agent_module.run_agent(
            PiBackend(model='glm-5.2'), tmp_path, 'Parse', phase=DaydreamPhase.REVIEW,
            output_schema=PER_STACK_RECORD_SCHEMA, validate_structured_output=validate, tools_disabled=validate,
        )
    assert reason is None
    assert result == ({'issues': []} if validate else ['module', 'moduleVersion', 'surface'])
