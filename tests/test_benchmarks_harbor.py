"""The benchmark-only ATIF export of a finished run's local evidence."""

from __future__ import annotations

import json
from pathlib import Path

from daydream.backends import MetricsEvent
from daydream.benchmarks.harbor import main
from daydream.trajectory import DaydreamPhase, dispatch_scope, run_document_path
from tests.harness.trajectory import make_recorder, observe_text_and_result


async def test_export_embeds_forks_once_and_writes_flat_subagent_files(tmp_path: Path) -> None:
    """Forks nest under their parents for usage, exist once as flat files for traces, and validate as ATIF."""
    run_dir = tmp_path / "runs" / "run"
    recorder = make_recorder(tmp_path, path=run_document_path(run_dir), session_id="run", artifact_run_dir=run_dir)
    async with recorder:
        async with dispatch_scope(recorder, phase=DaydreamPhase.REVIEW, descriptors=["a"]) as dispatch:
            async with recorder.fork("a", dispatch=dispatch) as child:
                async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                    invocation.observe(MetricsEvent(prompt_tokens=10, completion_tokens=2, cached_tokens=None,
                                                    cost_usd=None, message_id="m", model_name="model-a"))
                    observe_text_and_result(invocation, "child review")
                async with child.fork("nested"):
                    pass  # A fork that records no step keeps no document.
                async with child.fork("challenge") as nested:
                    async with nested.invocation(phase=DaydreamPhase.CHALLENGE) as invocation:
                        observe_text_and_result(invocation, "nested challenge")
    out = tmp_path / "export"
    assert main([str(run_dir), str(out)]) == 0
    root = json.loads((out / "trajectory.json").read_text())
    subagents = {path.name: json.loads(path.read_text()) for path in out.glob("trajectory.subagent-*.json")}
    assert root["schema_version"] == "ATIF-v1.7" and [step["llm_call_count"] for step in root["steps"]] == [0]
    (fork,) = root["subagent_trajectories"]
    assert [nested["trajectory_id"] for nested in fork["subagent_trajectories"]] == ["run:challenge"]
    assert sorted(document["trajectory_id"] for document in subagents.values()) == [
        "run:challenge", "run:dispatch:1:fork:1"]
    # The dispatch step's reference resolves to the flat file Harbor's trace reader discovers.
    (ref,) = root["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"]
    assert subagents[ref["trajectory_path"]]["trajectory_id"] == ref["trajectory_id"]
    # Usage lives on the fork's own steps exactly once; the root's folded total is not a step.
    metered = [step["metrics"] for document in (root, fork, *fork["subagent_trajectories"])
               for step in document["steps"] if step.get("metrics")]
    assert metered == [{"prompt_tokens": 10, "completion_tokens": 2}]
    assert root["final_metrics"]["total_prompt_tokens"] == 10
