"""Isolated Pi review comparison. Never posts, commits, or contacts GitHub.

Run with the same interpreter against an exported baseline and the candidate:
  uv run python scripts/measure_review_runtime.py --source /tmp/baseline --output /tmp/before
  uv run python scripts/measure_review_runtime.py --source . --output /tmp/after
The selected source tree supplies both the prompt and agent implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from daydream.agent import run_agent
    from daydream.backends import GenerationEndEvent, GenerationStartEvent, ToolResultEvent, ToolStartEvent
    from daydream.backends.pi import PiBackend
    from daydream.deep.prompts import build_per_stack_prompt
    from daydream.phases import PER_STACK_RECORD_SCHEMA
    from daydream.review_profile import build_default_profile
    from daydream.run_context import InteractionPolicy, RunContext
    from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = root / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    before = (
        '"""Batch statistics and tenant authorization."""\n\n'
        "def mean(values):\n    if not values:\n        return 0\n    return sum(values) / len(values)\n\n"
        "def can_access(role, tenant_id, document_tenant):\n"
        '    return role == "admin" and tenant_id == document_tenant\n'
    )
    after = before.replace("    if not values:\n        return 0\n", "").replace('"admin" and', '"admin" or')
    (repo / "batch.py").write_text(after)
    (repo / "README.md").write_text(
        "Empty batches must return 0. Only administrators in the document's tenant may access it.\n"
    )
    shared = repo / ".daydream"
    shared.mkdir()
    diff = "diff --git a/batch.py b/batch.py\n" + "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile="a/batch.py", tofile="b/batch.py",
    ))
    (shared / "diff.patch").write_text(diff)
    (shared / "intent.md").write_text("Simplify batch statistics and authorization while preserving behavior.")
    (shared / "summary.md").write_text("README.md specifies empty-batch and tenant-isolation contracts.")
    (shared / "affected_files.md").write_text("batch.py: modified; README.md: contracts. No dependencies.")
    (shared / "hunk-index.json").write_text(json.dumps({"batch.py": [[1, 8]]}))
    prompt = build_per_stack_prompt(
        strategy=build_default_profile().strategies["discovery.per_stack"].content,
        stack_name="python", files=["batch.py"], diff_path=shared / "diff.patch",
        intent_path=shared / "intent.md", alternatives_path=shared / "alternatives.json",
        output_path=shared / "review.md", cwd=repo, exploration_dir=shared,
        inline_diff=diff, include_alternatives=False,
    )
    (root / "prompt.txt").write_text(prompt)
    events = []
    started = time.monotonic()

    class MeasuredPi(PiBackend):
        async def execute(self, *a, **kw):
            stream = super().execute(*a, **kw)
            try:
                async for event in stream:
                    if isinstance(event, (ToolStartEvent, ToolResultEvent, GenerationStartEvent, GenerationEndEvent)):
                        events.append({
                            "at_s": time.monotonic() - started, "event": type(event).__name__,
                            "id": getattr(event, "id", None), "name": getattr(event, "name", None),
                        })
                    yield event
            finally:
                await stream.aclose()

    os.environ["PI_PROVIDER"] = "openrouter"
    backend = MeasuredPi(args.model, cwd=repo)
    kwargs = {}
    if "review_limits" in inspect.signature(run_agent).parameters:
        from daydream.review_budget import ReviewLimits
        kwargs["review_limits"] = ReviewLimits()
    recorder = TrajectoryRecorder(
        path=root / "trajectory.json", target_dir=repo, run_flow=DaydreamRunFlow.DEEP,
        agent_model_name=args.model, session_id="runtime-comparison",
    )
    async with recorder:
        result, _, reason = await run_agent(
            backend, repo, prompt, phase=DaydreamPhase.DEEP, output_schema=PER_STACK_RECORD_SCHEMA,
            wall_budget_s=600, run_context=RunContext(InteractionPolicy(quiet=True, interactive=False)), **kwargs,
        )
    report = {"elapsed_s": time.monotonic() - started, "model": args.model, "reason": reason,
              "tool_calls": sum(e["event"] == "ToolStartEvent" for e in events),
              "events": events, "result": result}
    (root / "measurement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {"events", "result"}}))


if __name__ == "__main__":
    asyncio.run(main())
