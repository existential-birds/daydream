"""Scripted model answers shared by native Harbor and packaged-container proofs."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch


@contextmanager
def forbid_github() -> Iterator[list[str]]:
    """Count forbidden lookups even when the reviewer catches their exceptions."""
    from daydream import github_app
    from daydream.git_ops import process

    attempts: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        attempts.append("GitHub or credential resolution")
        raise AssertionError("immutable Harbor review attempted GitHub access")

    original_run = subprocess.run

    def no_gh(command: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(command, list) and Path(command[0]).name == "gh":
            return forbidden()
        return original_run(command, *args, **kwargs)

    with ExitStack() as patches:
        for owner, names in ((process, ("_run_gh", "_run_gh_async")),
                             (github_app, ("resolve_credentials", "resolve_user_identity", "resolve_run_identity"))):
            for name in names:
                patches.enter_context(patch.object(owner, name, forbidden))
        patches.enter_context(patch.object(subprocess, "run", no_gh))
        yield attempts


def _text(message: dict[str, Any]) -> str:
    content = message["content"]
    return content if isinstance(content, str) else "\n".join(block.get("text", "") for block in content)


def assignment_from_request(request: dict[str, Any]) -> dict[str, Any]:
    """Committed assignments live in system instructions, which survive native compaction."""
    text = "\n".join(_text(message) for message in request["messages"] if message["role"] in {"system", "developer"})
    assignment: dict[str, Any] = json.loads(text.split("\nCommitted review assignment:\n", 1)[1].split(
        "\nEnd committed review assignment.\n", 1)[0])
    return assignment


def looked_up_targets(request: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Targets returned by review_source; its first text line is compact JSON metadata."""
    found: dict[str, dict[str, Any]] = {}
    for message in request["messages"]:
        if message["role"] != "tool":
            continue
        try:
            metadata = json.loads(_text(message).split("\n", 1)[0])
        except ValueError:
            continue
        if isinstance(metadata, dict) and isinstance(metadata.get("target"), dict):
            found[metadata["target"]["target_id"]] = metadata["target"]
    return found


def answer_review(
    request: dict[str, Any], *, empty: bool = False, partial: bool = False,
) -> list[tuple[str, str, dict[str, Any]]]:
    assignment = assignment_from_request(request)
    kind = assignment["kind"]
    entries = assignment["catalog"]["entries"]
    if not any(message["role"] == "tool" for message in request["messages"]):
        return [(f"{kind}-read", "read", {"path": "feature.py"}), *[
            (f"{kind}-lookup-{index}", "review_source", {"target_id": entry["target_id"], "path": None, "cursor": None})
            for index, entry in enumerate(entries)]]
    targets = looked_up_targets(request)
    refs_by_target = {target_id: [{
        "source_id": next(option["source_id"] for option in part["source_options"] if option["side"] == "new"),
        "source_range": None, "basis": "supplied_diff",
    }] for target_id, part in targets.items()}
    feature_part = next(entry for entry in entries if entry["file"] == "feature.py")
    refs = refs_by_target[feature_part["target_id"]]
    if kind == "discovery":
        payload = {"targets": [{"target_id": target_id, "status": "reviewed", "reason": "Captured source checked.",
                                "evidence_refs": target_refs} for target_id, target_refs in refs_by_target.items()],
                   "claims": [] if empty else [{"uid": "", "revision": 0, "source_uids": [],
                       "title": "Captured feature has a reproducible defect", "payload": {
                           "claim": "The changed feature loses the required second value.",
                           "trigger": "Use the changed feature with the second input.",
                           "consequence": "The caller receives the wrong value.",
                           "grounds": "The captured second line demonstrates the changed value.",
                           "file": "feature.py", "line": 2, "evidence_refs": refs}}]}
    else:
        claim, = assignment["claims"]
        payload = {"verdicts": [{"uid": claim["uid"], "revision": claim["revision"],
            "verdict": "unresolved" if partial else "confirmed", "grounds": "Fresh captured source checked.",
            "evidence_refs": refs}], "claims": []}
    return [(f"{kind}-submit", "structured_output", payload)]
