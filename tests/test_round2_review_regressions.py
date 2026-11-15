"""Retain issue #1445 opt-out behavior through the Pi transport and host boundary."""

import pathlib

import daydream.agent as agent_module


async def test_pi_backend_honors_structured_output_opt_out(tmp_path: pathlib.Path) -> None:
    """The host selects the largest span over Pi's text transport."""
    from unittest.mock import patch

    from daydream.backends.pi import PiBackend
    from daydream.output_schema import result_array_schema
    from daydream.trajectory import DaydreamPhase
    from tests.harness.pi_replay import make_mock_process_from_fixture

    process = make_mock_process_from_fixture('issue1445_empty_results.jsonl')
    with patch('daydream.backends._transport.asyncio.create_subprocess_exec', return_value=process):
        result, _, reason = await agent_module.run_agent(
            PiBackend(model='glm-5.2'), tmp_path, 'Parse', phase=DaydreamPhase.REVIEW,
            output_schema=result_array_schema('issues', {'id': {'type': 'integer'}}),
            validate_structured_output=False,
        )
    assert reason is None
    assert result == ['module', 'moduleVersion', 'surface']
