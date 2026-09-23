"""Native tool-free review calls preserve Pi reasoning and large prompt transport."""

import hashlib
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.agent import run_agent
from daydream.backends import PiRequestConfig, RequestEvent
from daydream.backends.pi import PiBackend, _schema_instruction
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.protocol_cli import install_protocol_cli


async def test_tools_disabled_large_prompt_uses_stdin_and_preserves_high_reasoning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "repo"
    target.mkdir()
    fixture = install_protocol_cli(tmp_path / "external", "pi")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PI_PROVIDER", "nous")
    monkeypatch.delenv("PI_API_KEY", raising=False)
    backend = PiBackend(model="fixture-model", reasoning_effort="high")
    prompt = "Repository evidence: 界\n" * 10_000
    schema: dict[str, Any] = {"type": "object", "properties": {"issues": {"type": "array"}}}
    full_prompt = prompt + _schema_instruction(schema)

    events = [event async for event in backend.execute(
        target, prompt, output_schema=schema, read_only=True, tools_disabled=True,
    )]

    observation = fixture.read_observations()[0]
    assert observation["stdin_bytes"] == len(full_prompt.encode("utf-8")) > 131_072
    assert observation["stdin_sha256"] == hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()
    assert observation["prompt_sha256"] == observation["stdin_sha256"]
    argv = observation["argv"]
    assert "--no-tools" in argv and "--tools" not in argv
    assert argv[argv.index("--thinking") + 1] == "high"
    assert "--append-system-prompt" in argv and "--system-prompt" not in argv
    request = next(event for event in events if isinstance(event, RequestEvent))
    assert request.prompt == full_prompt
    assert request.reasoning_effort == "high"
    assert isinstance(request.config, PiRequestConfig)
    assert request.config.finalization is False
    assert request.config.no_tools is True
    assert request.config.selected_tools_count == 0
    assert request.config.selected_tools_present is False


async def test_tools_disabled_bridge_does_not_leak_to_fix_or_change_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = install_protocol_cli(tmp_path / "external", "pi")
    target = tmp_path / "repo"
    target.mkdir()
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PI_PROVIDER", "nous")
    monkeypatch.delenv("PI_API_KEY", raising=False)
    backend = PiBackend(model="fixture-model", reasoning_effort="high")

    policy = "Resolve the single supplied evidence batch and stop."
    with patch.object(backend, "execute", wraps=backend.execute) as execute:
        await run_agent(backend, target, "Bounded evidence", phase=DaydreamPhase.DEEP,
                        tools_disabled=True, review_system_instructions=policy,
                        progress_callback=lambda _: None)
        await run_agent(backend, target, "Implement fix", phase=DaydreamPhase.FIX,
                        progress_callback=lambda _: None)
    assert execute.call_args_list[0].kwargs["review_instructions"] == policy
    assert "review_instructions" not in execute.call_args_list[1].kwargs
    final_events = [event async for event in backend.execute(target, "Serialize result", finalization=True)]

    observations = {
        row["prompt_sha256"]: row for row in fixture.read_observations()
    }
    review, fix, final = [observations[hashlib.sha256(text.encode()).hexdigest()]
                          for text in ("Bounded evidence", "Implement fix", "Serialize result")]
    assert "--no-tools" in review["argv"] and review["stdin_bytes"] > 0
    assert "--no-tools" not in fix["argv"] and fix["stdin_bytes"] == 0
    assert fix["argv"][fix["argv"].index("--thinking") + 1] == "high"
    assert "--no-tools" in final["argv"] and final["stdin_bytes"] == 0
    assert "--system-prompt" in final["argv"]
    final_request = next(event for event in final_events if isinstance(event, RequestEvent))
    assert final_request.reasoning_effort == "low"
    assert final_request.config is not None and final_request.config.finalization is True
    assert backend.reasoning_effort == "high"


async def test_tools_disabled_request_fails_before_dispatch_on_unsupported_backend(tmp_path: Path) -> None:
    backend = ScriptedBackend()
    with pytest.raises(NotImplementedError, match="tools_disabled"):
        await run_agent(backend, tmp_path, "Review", phase=DaydreamPhase.DEEP, tools_disabled=True)
    assert backend.calls == []


async def test_review_system_instructions_require_native_tool_free_support(tmp_path: Path) -> None:
    backend = ScriptedBackend(supports_tools_disabled=True)
    with pytest.raises(ValueError, match="tools_disabled"):
        await run_agent(backend, tmp_path, "Review", phase=DaydreamPhase.DEEP,
                        review_system_instructions="Stop after the evidence batch.")
    with pytest.raises(NotImplementedError, match="review_instructions"):
        await run_agent(backend, tmp_path, "Review", phase=DaydreamPhase.DEEP,
                        tools_disabled=True, review_system_instructions="Stop after the evidence batch.")
    assert backend.calls == []
