"""Isolated Pi review comparison. Never posts, commits, or contacts GitHub.

Use matching model, thinking and limits with each fixture on both revisions:
  uv run python scripts/measure_review_runtime.py --source /tmp/baseline --output /tmp/before --fixture clean
  uv run python scripts/measure_review_runtime.py --source . --output /tmp/after --fixture clean
Repeat with --fixture two-defect. Output directories must be new. The selected
source supplies the prompt and runtime, while this script fixes fixture/config.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


def fixture(name: str) -> tuple[str, str, str, str, str]:
    """Return filename, before, after, contract, and intent (no gold in prompt)."""
    if name == "two-defect":
        before = (
            '"""Batch statistics and tenant authorization."""\n\n'
            "def mean(values):\n    if not values:\n        return 0\n    return sum(values) / len(values)\n\n"
            "def can_access(role, tenant_id, document_tenant):\n"
            '    return role == "admin" and tenant_id == document_tenant\n'
        )
        after = before.replace("    if not values:\n        return 0\n", "").replace('"admin" and', '"admin" or')
        return (
            "batch.py", before, after,
            "Empty batches must return 0. Only administrators in the document's tenant may access it.\n",
            "Simplify batch statistics and authorization while preserving behavior.",
        )
    before = '''"""Tenant-scoped notification pages; callers pass an in-memory snapshot."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Notification:
    id: int
    tenant: str
    message: str
    archived: bool = False


@dataclass(frozen=True)
class Page:
    items: tuple[Notification, ...]
    next_cursor: int | None


def page_notifications(
    notifications: tuple[Notification, ...],
    tenant: str,
    *,
    after: int = 0,
    limit: int = 20,
) -> Page:
    """Return active notifications in increasing id order for one tenant."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    if after < 0:
        raise ValueError("after must be nonnegative")
    matching = []
    for notification in notifications:
        if notification.tenant != tenant:
            continue
        if notification.archived:
            continue
        if notification.id <= after:
            continue
        matching.append(notification)
    matching.sort(key=lambda notification: notification.id)
    items = tuple(matching[:limit])
    has_more = len(matching) > limit
    next_cursor = items[-1].id if has_more else None
    return Page(items=items, next_cursor=next_cursor)
'''
    old = '''    matching = []
    for notification in notifications:
        if notification.tenant != tenant:
            continue
        if notification.archived:
            continue
        if notification.id <= after:
            continue
        matching.append(notification)
    matching.sort(key=lambda notification: notification.id)
'''
    new = '''    matching = sorted(
        (
            notification
            for notification in notifications
            if notification.tenant == tenant
            and not notification.archived
            and notification.id > after
        ),
        key=lambda notification: notification.id,
    )
'''
    return (
        "notifications.py", before, before.replace(old, new),
        "Notification ids are unique positive integers. Pages exclude archived and other-tenant records. "
        "The exclusive cursor and increasing id order are stable for an immutable snapshot. "
        "next_cursor is the last returned id only if another page exists. Limits 1..100 are supported.\n",
        "Express notification filtering and ordering together while preserving pagination behavior.",
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--thinking", default="high")
    parser.add_argument("--fixture", choices=("two-defect", "clean"), default="two-defect")
    parser.add_argument("--investigation-s", type=float, default=480)
    parser.add_argument("--finalization-s", type=float, default=120)
    parser.add_argument("--tool-calls", type=int, default=48)
    args = parser.parse_args()
    if min(args.investigation_s, args.finalization_s, args.tool_calls) < 0:
        parser.error("limits must be nonnegative")
    source_digest = hashlib.sha256()
    for path in sorted((args.source.resolve() / "daydream").rglob("*.py")):
        source_digest.update(str(path.relative_to(args.source.resolve())).encode())
        source_digest.update(path.read_bytes())
    sys.path.insert(0, str(args.source.resolve()))
    from jsonschema import Draft202012Validator

    from daydream.agent import run_agent
    from daydream.backends import (
        GenerationEndEvent,
        GenerationStartEvent,
        MetricsEvent,
        RequestEvent,
        ResultEvent,
        ToolResultEvent,
        ToolStartEvent,
    )
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
    filename, before, after, contract, intent = fixture(args.fixture)
    (repo / filename).write_text(after)
    (repo / "README.md").write_text(contract)
    if args.fixture == "clean":
        (repo / "test_notifications.py").write_text('''from notifications import Notification, page_notifications


def test_pagination_preserves_tenant_archive_and_cursor_filters():
    snapshot = (
        Notification(4, "a", "four"), Notification(2, "b", "other"),
        Notification(3, "a", "hidden", archived=True), Notification(1, "a", "one"),
        Notification(5, "a", "five"),
    )
    first = page_notifications(snapshot, "a", limit=2)
    assert [item.id for item in first.items] == [1, 4]
    assert first.next_cursor == 4
    last = page_notifications(snapshot, "a", after=first.next_cursor, limit=2)
    assert [item.id for item in last.items] == [5]
    assert last.next_cursor is None
    assert page_notifications(snapshot, "missing").items == ()
''')
    shared = repo / ".daydream"
    shared.mkdir()
    diff = f"diff --git a/{filename} b/{filename}\n" + "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile=f"a/{filename}", tofile=f"b/{filename}",
    ))
    (shared / "diff.patch").write_text(diff)
    (shared / "intent.md").write_text(intent)
    (shared / "summary.md").write_text("README.md specifies the preserved behavior. " + contract)
    (shared / "affected_files.md").write_text(f"{filename}: modified; README.md: contracts. No dependencies.")
    (shared / "hunk-index.json").write_text(json.dumps({filename: [[1, len(after.splitlines())]]}))
    prompt = build_per_stack_prompt(
        strategy=build_default_profile().strategies["discovery.per_stack"].content,
        stack_name="python", files=[filename], diff_path=shared / "diff.patch",
        intent_path=shared / "intent.md", alternatives_path=shared / "alternatives.json",
        output_path=shared / "review.md", cwd=repo, exploration_dir=shared,
        inline_diff=diff, include_alternatives=False,
    )
    (root / "prompt.txt").write_text(prompt)
    events = []
    calls = []
    started = time.monotonic()

    class MeasuredPi(PiBackend):
        async def execute(self, *a, **kw):
            # The baseline does not expose an invocation flag; its explicit
            # host marker is used only for measurement, never runtime parsing.
            is_finalizer = kw.get("finalization", False) or "INVESTIGATION HAS ENDED" in a[1]
            call = {"finalizer": is_finalizer, "result_event": False, "valid_result": False,
                    "tool_starts": 0, "requests": []}
            calls.append(call)
            stream = super().execute(*a, **kw)
            try:
                async for event in stream:
                    if isinstance(event, RequestEvent):
                        call["requests"].append({"reasoning_effort": event.reasoning_effort,
                                                 "effective_config": asdict(event.config)})
                    if isinstance(event, ResultEvent):
                        call["result_event"] = True
                        call["valid_result"] = Draft202012Validator(PER_STACK_RECORD_SCHEMA).is_valid(
                            event.structured_output
                        )
                    if isinstance(event, ToolStartEvent):
                        call["tool_starts"] += 1
                    if isinstance(event, (ToolStartEvent, ToolResultEvent, GenerationStartEvent,
                                          GenerationEndEvent, MetricsEvent)):
                        record = {"at_s": time.monotonic() - started, "event": type(event).__name__,
                                  "finalizer": is_finalizer, "id": getattr(event, "id", None),
                                  "name": getattr(event, "name", None)}
                        if isinstance(event, MetricsEvent):
                            record.update(prompt_tokens=event.prompt_tokens, completion_tokens=event.completion_tokens)
                        events.append(record)
                    yield event
            finally:
                call["outcome"] = (
                    "valid_result" if call["valid_result"] else
                    "invalid_result" if call["result_event"] else "no_terminal_result"
                )
                await stream.aclose()

    os.environ["PI_PROVIDER"] = "openrouter"
    backend = MeasuredPi(args.model, cwd=repo, reasoning_effort=args.thinking)
    kwargs = {}
    if "review_limits" in inspect.signature(run_agent).parameters:
        from daydream.review_budget import ReviewLimits
        kwargs["review_limits"] = ReviewLimits(args.investigation_s, args.finalization_s, args.tool_calls)
    if "finalization_context" in inspect.signature(run_agent).parameters:
        from daydream.review_evidence import FinalizationContext
        kwargs["finalization_context"] = FinalizationContext(
            task="Finalize assigned Python review", assigned_files=(filename,),
            output_semantics="Return issues and truthful file verdicts. Empty issues is valid.",
            supplied_context=(("diff", diff), ("confirmed intent", intent), ("repository contract", contract)),
        )
    recorder = TrajectoryRecorder(
        path=root / "trajectory.json", target_dir=repo, run_flow=DaydreamRunFlow.DEEP,
        agent_model_name=args.model, session_id="runtime-comparison",
    )
    failure = None
    result = None
    reason = None
    try:
        async with recorder:
            result, _, reason = await run_agent(
                backend, repo, prompt, phase=DaydreamPhase.DEEP, output_schema=PER_STACK_RECORD_SCHEMA,
                wall_budget_s=args.investigation_s + args.finalization_s,
                run_context=RunContext(InteractionPolicy(quiet=True, interactive=False)), **kwargs,
            )
    except Exception as exc:
        # Retain metrics without dumping potentially credential-bearing exception text.
        failure = type(exc).__name__
    valid = Draft202012Validator(PER_STACK_RECORD_SCHEMA).is_valid(result)
    issues = result["issues"] if valid else []
    report = {
        "elapsed_s": time.monotonic() - started, "model": args.model, "thinking": args.thinking,
        "source_sha256": source_digest.hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "fixture": args.fixture, "reason": reason, "failure_type": failure,
        "limits": {"investigation_s": args.investigation_s, "finalization_s": args.finalization_s,
                   "tool_calls": args.tool_calls},
        "tool_calls": sum(e["event"] == "ToolStartEvent" for e in events),
        "token_metrics_available": any(e["event"] == "MetricsEvent" for e in events),
        "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in events),
        "completion_tokens": sum(e.get("completion_tokens", 0) for e in events),
        "budget_stops": int(reason is not None), "schema_valid": valid,
        "valid_empty_result": valid and not issues,
        "complete_clean_result": valid and not issues and reason is None and any(
            v["path"] == filename and v["verdict"] == "clean" for v in result["verdicts"]
        ),
        # Human adjudication of these evidence/location-bearing issues is required;
        # schema validity or a keyword match does not prove defect retention.
        "defect_retention": "requires examination of result evidence against fixture" if issues else "no findings",
        "finalizer_calls": [c for c in calls if c["finalizer"]],
        "calls": calls, "events": events, "result": result,
    }
    (root / "measurement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {"calls", "events", "result"}}))


if __name__ == "__main__":
    asyncio.run(main())
