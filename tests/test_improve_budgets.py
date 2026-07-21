import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import AgentEvent, ResultEvent, ToolStartEvent
from daydream.runner import RunConfig, run


class _RunawayReconBackend:
    model = "mock-model"

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        async def _events() -> AsyncIterator[AgentEvent]:
            if "IMPROVE_RECON" in prompt:
                index = 0
                while True:
                    await anyio.sleep(0.01)
                    yield ToolStartEvent(
                        id=f"recon-{index}",
                        name="Glob",
                        input={"pattern": "**/*.py"},
                    )
                    index += 1

            properties = (output_schema or {}).get("properties", {})
            if "findings" in properties:
                yield ResultEvent(
                    structured_output={"findings": []},
                    continuation=None,
                )
            else:
                yield ResultEvent(
                    structured_output={"conventions": [], "guidelines": []},
                    continuation=None,
                )

        return _events()

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


@pytest.mark.anyio
async def test_improve_recon_budget_exhaustion_is_diagnostic_and_nonfatal(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _RunawayReconBackend()
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
    )
    monkeypatch.setattr(
        "daydream.runner.DEFAULT_TOOL_CALL_BUDGET",
        2,
        raising=False,
    )

    with anyio.fail_after(1):
        code = await run(
            RunConfig(
                target=str(improve_monorepo_target),
                flow_name="improve",
                non_interactive=True,
                quiet=True,
                archive=False,
            )
        )

    assert code == 0
    assert (
        improve_monorepo_target / ".daydream/improve/report.md"
    ).is_file()
    assert (
        improve_monorepo_target
        / ".daydream/improve/command-validation-diagnostics.json"
    ).is_file()
    trajectories = list(
        (improve_monorepo_target / ".daydream" / "runs").glob(
            "*/trajectory.json"
        )
    )
    assert len(trajectories) == 1
    trajectory = json.loads(trajectories[0].read_text())
    assert trajectory["extra"]["partial"] is True
    assert any(
        step.get("extra", {}).get("stop_reason")
        == "tool_call_budget_exceeded"
        for step in trajectory["steps"]
    )
