"""Shared helpers for focused deep-orchestrator test modules."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from daydream import git_ops
from daydream.backends import AgentEvent, ResultEvent, TextEvent
from daydream.config_file import load_file_config
from daydream.deep.artifacts import deep_dir
from daydream.deep.fix_state import FixCycleState, capture_retained_tree
from daydream.extensions import Registry
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.flows.engine import FlowContext
from daydream.run_config import RunConfig
from daydream.runner import run as _run
from tests.harness.git_helpers import (
    commit as _commit,
    git as _git,
    init_repo as _init_repo,
    work_context,
)
from tests.test_deep_orchestrator import (
    Mute,
    _force_interactive,
    _install_stub_backend,
    _merge_item,
    _prime_merge_resume,
    _record,
    _record_issues,
    _run_deep,
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


def _make_record_issue(issues: list[tuple[Any, ...]]) -> Callable[..., str]:
    def _record_issue(repo: Any, *, title: str, body: str, **kwargs: Any) -> str:
        issues.append((repo, title, body))
        return "https://github.com/owner/repo/issues/1"
    return _record_issue


def _merged_item_files(target: Path) -> list[str]:
    """Return the ``file`` of every item in the canonical merged-items.json."""
    return [cast("str", it.get("file")) for it in _merged_items(target / ".daydream" / "deep")]


def _merged_item_descriptions(target: Path) -> list[str]:
    """Return the ``description`` of every item in the canonical merged-items.json."""
    return [it.get("description", "") for it in _merged_items(target / ".daydream" / "deep")]


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


def _root_phase_events(target: Path, phase: str) -> list[dict[str, Any]]:
    trajectories = list((target / ".daydream" / "runs").glob("*/trajectory.json"))
    assert len(trajectories) == 1
    payload = json.loads(trajectories[0].read_text(encoding="utf-8"))
    return [event for event in payload["extra"]["phase_events"] if event["phase"] == phase]


def _batched_group_size(stub: "_StubBackend", file_basename: str) -> int:
    """Return N from the failed batched ``Fix these N issues in <file>`` fix turn.

    The deep pipeline may inject an extra structural finding into a file group, so
    the group size is derived from the batched prompt rather than hard-coded.
    """
    for c in stub.calls:
        m = re.search(r"^Fix these (\d+) issues in (.+):$", c["prompt"], re.M)
        if m is not None and Path(m.group(2)).name == file_basename:
            return int(m.group(1))
    raise AssertionError(f"no batched fix turn found for {file_basename}")


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


def _install_accept_gate_pipeline(monkeypatch: pytest.MonkeyPatch, target: Path, mute: Mute) -> _StubBackend:
    """Accept the interactive fix gate, silence UI, and stub post/test/commit side effects.

    phase_fix stays real; return the installed backend.
    """
    _force_interactive(monkeypatch)
    _silence(monkeypatch, prompts=False)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")
    mute()
    return _install_stub_backend(monkeypatch, target)


async def _run_loop(target: Path, make_config: Any) -> int:
    """Run the deep pipeline in loop output mode with *target*'s file config."""
    return await _run(make_config(target, assume="yes", output_mode="loop", file_config=load_file_config(target),))


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


def _count_merge_prompts(calls: list[dict[str, Any]]) -> int:
    """Count cross-stack merge-agent prompts (discriminator: 'cross-stack merge agent')."""
    return sum(1 for c in calls if "cross-stack merge agent" in c["prompt"].lower())


def _eroded_main_repo(tmp_path: Path) -> Path:
    """Build a repo whose feature branch adds an eroded ``main()``: repeated
    ``--flag value`` / ``--flag=value`` branch pairs with no helper extracted
    (the SlopCodeBench B.2 canonical shape the anti-slop rubric targets)."""
    project = tmp_path / "eroded_main"
    project.mkdir()
    init = (
        "import sys\n"
        "\n"
        "\n"
        "def main(argv):\n"
        "    return 0\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main(sys.argv))\n"
    )
    (project / "main.py").write_text(init)
    _init_repo(project)
    _git(project, "add", ".")
    _commit(project, "init")
    _git(project, "checkout", "-b", "feature")
    eroded = (
        "import sys\n"
        "\n"
        "\n"
        "def main(argv):\n"
        "    if '--verbose value' in argv:\n"
        "        log('verbose on')\n"
        "    if '--verbose=value' in argv:\n"
        "        log('verbose on')\n"
        "    if '--debug value' in argv:\n"
        "        log('debug on')\n"
        "    if '--debug=value' in argv:\n"
        "        log('debug on')\n"
        "    if '--color value' in argv:\n"
        "        log('color on')\n"
        "    if '--color=value' in argv:\n"
        "        log('color on')\n"
        "    return 0\n"
        "\n"
        "\n"
        "def log(msg):\n"
        "    print(msg)\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main(sys.argv))\n"
    )
    (project / "main.py").write_text(eroded)
    _git(project, "add", ".")
    _commit(project, "change: add flag handling inline")
    return project


def _uid_records(deep: Path, stack: str) -> list[dict[str, Any]]:
    """Return the issues list on disk for *stack*'s per-stack records file."""
    return _record_issues(json.loads((deep / f"stack-{stack}-records.json").read_text()))


def _uid_list(deep: Path, stack: str) -> list[Any]:
    """Return the ``uid`` of every record on disk for *stack*, in file order."""
    return [record.get("uid") for record in _uid_records(deep, stack)]


def _high_record(**overrides: Any) -> dict[str, Any]:
    """Build a primed per-stack record the arbiter's severity branch selects.

    ``high`` severity is what makes ``select_arbiter_targets`` pick a record up,
    and arbitration is the only thing that rewrites the per-stack records files
    -- so it is also what makes an in-memory ``uid`` observable on disk.
    """
    record = {"severity": "high", "confidence": "HIGH", "rationale": "stub", "evidence": "api.py:1"}
    record.update(overrides)
    return _record(**record)


def _prime_uid_merge_resume(target: Path, python: list[dict[str, Any]]) -> Path:
    """Prime a resume with only Python eligible for arbitration.

    The structural record occupies api.py:5 alone so contested-location selection
    cannot pull it into the pass.
    """
    return _prime_merge_resume(
        target, python=python, react=[_record(description="tsx issue", file="App.tsx", evidence="App.tsx:1")],
        generic=[_record(description="docs issue", file="README.md", evidence="README.md:1")],
        structure=[_record(description="structural issue", line=5, evidence="api.py:5")],
    )


class _RejectingArbiterBackend(_StubBackend):
    """Reject one host UID from arbiter-input.json, even when reviewer IDs repeat."""

    def __init__(self, target: Path, reject_uid: str) -> None:
        super().__init__(target)
        self._reject_uid = reject_uid

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,) -> AsyncIterator[AgentEvent]:
        if "you are the arbiter" in prompt.lower():
            self.calls.append({"prompt": prompt, "model": self.model})
            match = re.search(r"listed in (\S*arbiter[-\w]*input\.json)", prompt)
            assert match is not None, "arbiter prompt did not point at its input artifact"
            entries = json.loads(Path(match.group(1)).read_text())
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"findings": [{
                            "arb_id": entry["arb_id"], "keep": entry["uid"] != self._reject_uid,
                            "severity": entry.get("severity") or "high",
                            "confidence": entry.get("confidence") or "HIGH",
                            "description": f"ARBITRATED: {entry.get('description')}",
                            "rationale": "arbiter second opinion",
                            "evidence": entry.get("evidence") or "api.py:1",
                        }
                        for entry in entries
                    ]
                }, continuation=None,
            )
            return
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


def _merged_items(deep: Path) -> list[dict[str, Any]]:
    """Return the canonical merged item list written under *deep*."""
    return cast("list[dict[str, Any]]", json.loads((deep / "merged-items.json").read_text())["items"])


def _run_uid_pool(deep: Path) -> set[str]:
    """Read the actual on-disk UID pool used to validate shipped source attribution."""
    return {uid
        for path in sorted(deep.glob("stack-*-records.json"))
        for record in _record_issues(json.loads(path.read_text()))
        if (uid := record.get("uid"))
    }


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


async def _fresh_uid_run(target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute,) -> Path:
    """Run the standard multi-stack fixture and return its deep artifact directory."""
    _silence(monkeypatch)
    mute_side_effects()
    _install_stub_backend(monkeypatch, target)
    assert await _run_deep(target) == 0
    return deep_dir(target, allow_standalone=True)


def _item_uids(deep: Path) -> list[str]:
    """Return the ``item_uid`` of every shipped item, in canonical file order."""
    return [str(item.get("item_uid")) for item in _merged_items(deep)]


def _direct_fix_context(repo: Path, items: list[dict[str, Any]], *, changed_files: set[str], start_at: str = "review",
) -> Any:
    """Build the smallest real-Git FlowContext for fix-boundary unit tests."""

    dd = repo / ".daydream" / "deep"
    dd.mkdir(parents=True, exist_ok=True)
    items_file = dd / "merged-items.json"
    items_file.write_text(json.dumps({"items": items}))
    return FlowContext(config=RunConfig(target=str(repo), assume="yes", cleanup=False, start_at=start_at),
        work=work_context(repo, run_id="session-current"), registry=Registry(), allow_standalone_artifacts=True,
        data={"dd": dd, "items_file": items_file, "merged_report": dd / "merged-review.md",
            "changed_files": changed_files,
        },
    )


def _direct_fix_state(ctx: Any, items: list[dict[str, Any]], reviewed: set[str]) -> Any:
    footprint = AuthorizedFixFootprint.build(ctx.work.repo, reviewed, items)
    state = FixCycleState(
        work=ctx.work, config=ctx.config, recipe=ctx.deep_data().get("test_recipe"),
        session_id="session-current", stable_ref=git_ops.head_sha(ctx.work.repo),
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
    snapshot = capture_retained_tree(state)
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
