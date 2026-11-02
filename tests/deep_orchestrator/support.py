"""Shared helpers for focused deep-orchestrator test modules."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from daydream import git_ops
from daydream.deep.fix_state import FixCycleState, capture_retained_tree
from daydream.extensions import Registry
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.flows.engine import FlowContext
from daydream.run_config import RunConfig
from tests.harness.git_helpers import (
    commit as _commit,
    git as _git,
    init_repo as _init_repo,
    work_context,
)
from tests.test_deep_orchestrator import (
    _merge_item,
    _prime_merge_resume,
    _record,
    _silence,
    _StubBackend,
)


def _silence_gate_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence noise-only UI in the deep path WITHOUT mocking prompt_user.

    Unlike ``_silence``, this deliberately leaves the real ``prompt_user`` in
    both the orchestrator and phases so the apply-fixes gate runs the genuine
    production code path under test.
    """
    _silence(monkeypatch, prompts=False)


def _forbidden_input(*_a: Any, **_kw: Any) -> str:
    raise AssertionError("input() was called in non-interactive mode -- stdin must not be touched")


def _capture_warnings(monkeypatch: pytest.MonkeyPatch, module_attr: str) -> list[str]:
    """Route ``module_attr``'s ``print_warning`` into a list and return it."""
    warnings: list[str] = []
    monkeypatch.setattr(module_attr, lambda console, msg, *a, **k: warnings.append(msg))
    return warnings


def _only_archived_run(archive_dir: Path) -> Path:
    """Return the single archived run directory, asserting there is exactly one."""
    run_dirs = list((archive_dir / "runs").iterdir())
    assert len(run_dirs) == 1, f"expected exactly one archived run, got {run_dirs}"
    return run_dirs[0]


def _install_post_recorder(monkeypatch: pytest.MonkeyPatch, received: list[dict[str, Any]]) -> None:
    """Stub the PR-posting boundary, recording each call's keyword arguments."""
    async def _record_post(*_args: Any, **kwargs: Any) -> None:
        received.append(kwargs)
    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _record_post)


def _prime_merge_resume_records(target: Path, *, python_severity: str | None) -> Path:
    """Write the per-stack records a `--start-at merge` resume needs on disk.

    Every detected stack (python, react, generic, structure) must have a records
    file or be a recorded failure, else the resume guard returns 1. The python
    record optionally carries ``python_severity`` to drive arbiter selection.
    """
    py_record = _record(description="py issue", evidence="api.py:1")
    if python_severity is not None:
        py_record |= {"severity": python_severity, "confidence": "HIGH", "rationale": "stub"}
    return _prime_merge_resume(
        target, python=[py_record], react=[_record(description="tsx issue", file="App.tsx", evidence="App.tsx:1")],
        generic=[_record(description="docs issue", file="README.md", evidence="README.md:1")],
        structure=[_record(description="structural issue", evidence="api.py:1")],
    )


def _iter_run_payloads(run_root: Path, traj: Path) -> Iterator[dict[str, Any]]:
    """Yield every parsed JSON object under *run_root* plus the *traj* file.

    Non-dict or unparseable files are skipped.
    """
    for tf in list(run_root.rglob("*.json")) + ([traj] if traj.exists() else []):
        try:
            payload = json.loads(tf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            yield payload


def _scan_trajectory_extra(run_root: Path, traj: Path, key: str, *, phase: str | None = None) -> list[str]:
    """Collect truthy extra values from main and fork trajectories, optionally filtering phase."""
    values: list[str] = []
    for payload in _iter_run_payloads(run_root, traj):
        for step in payload.get("steps", []):
            extra = step.get("extra") or {}
            if phase is not None and extra.get("daydream_phase") != phase:
                continue
            value = extra.get(key)
            if value:
                values.append(value)
    return values


def _scan_phase_events(run_root: Path, traj: Path, event: str) -> list[dict[str, Any]]:
    """Collect ``extra['phase_events']`` entries of a given ``event`` across all run JSONs.

    The group-budget marker lives in ``Trajectory.extra['phase_events']`` (not
    step ``extra``), and the fix work runs in a forked sibling trajectory, so scan
    every ``*.json`` beneath ``run_root`` plus the top-level ``traj``.
    """
    found: list[dict[str, Any]] = []
    for payload in _iter_run_payloads(run_root, traj):
        for ev in (payload.get("extra") or {}).get("phase_events", []):
            if isinstance(ev, dict) and ev.get("event") == event:
                found.append(ev)
    return found


def _single_fix_calls_for(stub: "_StubBackend", file_basename: str) -> list[dict[str, Any]]:
    """Match a fix prompt's primary File line, ignoring sibling paths in Allowed files."""
    out: list[dict[str, Any]] = []
    for c in stub.calls:
        prompt = c["prompt"]
        if not prompt.lower().startswith("fix this issue"):
            continue
        m = re.search(r"^File: (.+)$", prompt, re.M)
        if m is not None and Path(m.group(1).strip()).name == file_basename:
            out.append(c)
    return out


def _count_review_prompts(calls: list[dict[str, Any]]) -> int:
    """Count per-stack + structural review prompts in a captured call list.

    Discriminators mirror the production prompt builders:
      - per-stack / generic-fallback: ``"you are reviewing the <X> stack"``
      - structural: ``"you are the structural reviewer"``
    """
    n = 0
    for c in calls:
        pl = c["prompt"].lower()
        if "you are reviewing the" in pl and "stack" in pl:
            n += 1
        elif "you are the structural reviewer" in pl:
            n += 1
    return n


def _merged_items(deep: Path) -> list[dict[str, Any]]:
    """Return the canonical merged item list written under *deep*."""
    return cast("list[dict[str, Any]]", json.loads((deep / "merged-items.json").read_text())["items"])


def _source_uids_by_description(deep: Path) -> dict[str, Any]:
    """Map each merged item's description to its ``source_uids`` value.

    Keyed on ``description`` because ``normalize_items`` reassigns every ``id``
    at the merge write, so the reviewer-side numbering a test set up with is gone
    by the time the artifact lands.
    """
    return {str(item.get("description")): item.get("source_uids") for item in _merged_items(deep)}


def _prime_source_uid_merge_resume(target: Path, *, structure: list[dict[str, Any]] | None = None,) -> Path:
    """Prime generic:1, python:1, react:1, structure:1 as the known on-disk UID pool.

    Keep locations distinct so contested arbitration cannot rewrite descriptions.
    UIDs must exist in record files: merge validation rereads those files and rejects
    claims absent from their pool.
    """
    return _prime_merge_resume(target, python=[_record(description="py issue", evidence="api.py:1", uid="python:1")],
        react=[_record(description="tsx issue", file="App.tsx", evidence="App.tsx:1", uid="react:1")],
        generic=[_record(description="docs issue", file="README.md", evidence="README.md:1", uid="generic:1",)],
        structure=(structure
            if structure is not None
            else [_record(description="structural issue", line=5, evidence="api.py:5",
                          severity="high", uid="structure:1")]
        ),
    )


def _provenance_item(item_id: int, description: str, *, file: str = "api.py", line: int = 1, source_uids: Any = None,
    evidence: str | None = None, severity: str = "medium", rationale: str = "rationale", omit_source_uids: bool = False,
) -> dict[str, Any]:
    """Build one merge-agent item with an explicitly chosen provenance claim.

    ``_merge_item`` cannot serve here: these tests are about the ``source_uids``
    value itself, including the shapes a real model gets wrong (a null, a bare
    string, an omitted key), so every one of them has to be settable.
    """
    item: dict[str, Any] = {"id": item_id, "lens": "per-stack", "file": file, "line": line, "severity": severity,
        "description": description, "confidence": "MEDIUM", "rationale": rationale,
        "evidence": f"{file}:{line}" if evidence is None else evidence,
    }
    if not omit_source_uids:
        item["source_uids"] = source_uids
    return item


def _direct_fix_context(repo: Path, items: list[dict[str, Any]], *, changed_files: set[str], start_at: str = "review",
) -> Any:
    """Build the smallest real-Git FlowContext for fix-boundary unit tests."""

    dd = repo / ".daydream" / "deep"
    dd.mkdir(parents=True, exist_ok=True)
    items_file = dd / "merged-items.json"
    items_file.write_text(json.dumps({"items": items}))
    return FlowContext(config=RunConfig(
        target=str(repo), test_command=f"{sys.executable} -c pass", assume="yes", cleanup=False, start_at=start_at,
    ),
        work=work_context(repo, run_id="session-current"), registry=Registry(), allow_standalone_artifacts=True,
        data={"dd": dd, "items_file": items_file, "merged_report": dd / "merged-review.md",
            "changed_files": changed_files,
        },
    )


def _direct_fix_state(ctx: Any, items: list[dict[str, Any]], reviewed: set[str]) -> Any:
    footprint = AuthorizedFixFootprint.build(ctx.work.repo, reviewed, items)
    state = FixCycleState(session_id="session-current", stable_ref=git_ops.head_sha(ctx.work.repo),
        stable_head=git_ops.head_sha(ctx.work.repo), initial_index=git_ops.snapshot_index(ctx.work.repo),
        preexisting_untracked=git_ops.snapshot_untracked_paths(ctx.work.repo, include_runtime_artifacts=False),
        preexisting_gitlinks=git_ops.snapshot_worktree_gitlinks(ctx.work.repo), footprint=footprint,
    )
    ctx.data["fix_cycle_state"] = state
    ctx.data["items"] = items
    return state


def _base_repo(tmp_path: Path, name: str) -> Path:
    """Init ``tmp_path/name`` with a single committed ``a.py`` base file."""
    repo = tmp_path / name
    _init_repo(repo)
    (repo / "a.py").write_text("A = 1\n")
    _git(repo, "add", "a.py")
    _commit(repo, "base")
    return repo


def _remote_identity_context(
    tmp_path: Path, fake_gh: Any, *, head_repository: str | None, configured_repository: str = "base-user/project",
    base_ref: str = "main", configured_pr: int = 7,
) -> tuple[Any, str]:

    repo = _base_repo(tmp_path, "remote-identity")
    _git(repo, "checkout", "-b", "feature")
    (repo / "a.py").write_text("A = 2\n")
    _git(repo, "add", "a.py")
    _commit(repo, "feature")
    sha = git_ops.head_sha(repo)
    head_row = None
    head_owner = None
    if head_repository is not None:
        owner, name = head_repository.split("/", 1)
        head_row = {"name": name, "nameWithOwner": head_repository}
        head_owner = {"login": owner}
    fake_gh.set_response("repo-view", value="base-user/project")
    fake_gh.serve_pr_view({
            "number": 7, "title": "Fix", "body": "", "state": "OPEN", "headRefName": "feature", "baseRefName": base_ref,
            "headRefOid": sha, "url": "https://github.com/base-user/project/pull/7", "headRepository": head_row,
            "headRepositoryOwner": head_owner,
        }
    )
    ctx = _direct_fix_context(repo, [], changed_files=set())
    ctx.config.pr_number = configured_pr
    ctx.config.pr_repo = configured_repository
    return ctx, sha


def _finalization_fixture(tmp_path: Path) -> tuple[Any, Any, Any]:
    repo = _base_repo(tmp_path, "finalization")
    items = [{**_merge_item(1, "a.py", "high"), "item_uid": "item:a", "related_files": []}]
    ctx = _direct_fix_context(repo, items, changed_files={"a.py"})
    state = _direct_fix_state(ctx, items, {"a.py"})
    (repo / "a.py").write_text("A = 2\n")
    snapshot = capture_retained_tree(ctx.work, state)
    return ctx, state, snapshot


def _arbiter_stacks(severities: dict[str, str]) -> dict[str, dict[str, object]]:
    """Per-stack findings at three distinct ``(file, line)`` locations.

    Three file components mean three arbiter groups whenever the selection
    spans more than one co-located target. The descriptions differ so a dedup
    pass cannot fold the records together.
    """
    locations = {"python": ("api.py", "python finding"), "react": ("App.tsx", "react finding"),
        "generic": ("README.md", "generic finding"),
    }
    return {name: {
            "severity": severities[name], "confidence": "HIGH", "file": file, "line": 1, "description": description,
        }
        for name, (file, description) in locations.items()
    }
