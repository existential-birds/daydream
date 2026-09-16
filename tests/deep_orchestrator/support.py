"""Shared helpers for focused deep-orchestrator test modules."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from daydream.backends import AgentEvent, ResultEvent, TextEvent
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo
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

if TYPE_CHECKING:
    from daydream.pr_review import ReviewRenderers


def _silence_gate_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence noise-only UI in the deep path WITHOUT mocking prompt_user.

    Unlike ``_silence``, this deliberately leaves the real ``prompt_user`` in
    both the orchestrator and phases so the apply-fixes gate runs the genuine
    production code path under test.
    """
    _silence(monkeypatch, prompts=False)


def _merged_item_files(target: Path) -> list[str]:
    """Return the ``file`` of every item in the canonical merged-items.json."""
    items_file = target / ".daydream" / "deep" / "merged-items.json"
    items = json.loads(items_file.read_text())["items"]
    return [it.get("file") for it in items]


def _merged_item_descriptions(target: Path) -> list[str]:
    """Return the ``description`` of every item in the canonical merged-items.json."""
    items_file = target / ".daydream" / "deep" / "merged-items.json"
    items = json.loads(items_file.read_text())["items"]
    return [it.get("description", "") for it in items]


def _install_post_recorder(monkeypatch: pytest.MonkeyPatch, received: list[bool]) -> None:
    """Stub the PR-posting boundary, recording the ``approve_on_clean`` kwarg."""

    async def _record_post(
        target_dir: Any,
        merged_items_path: Any,
        *,
        console: Any,
        run_info: str,
        renderers: ReviewRenderers,
        post: Any,
        approve_on_clean: Any = False,
        diagram_blocks: Any = None,
        run_context: Any = None,
        auth: Any,
    ) -> None:
        received.append(approve_on_clean)

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
        target,
        python=[py_record],
        react=[_record(description="tsx issue", file="App.tsx", evidence="App.tsx:1")],
        generic=[_record(description="docs issue", file="README.md", evidence="README.md:1")],
        structure=[_record(description="structural issue", evidence="api.py:1")],
    )


def _scan_trajectory_extra(run_root: Path, traj: Path, key: str) -> list[str]:
    """Collect ``step["extra"][key]`` across every trajectory JSON written for a run.

    An aborted/forked turn writes sibling trajectory files under the per-run dir, so
    scan all ``*.json`` beneath ``run_root`` plus the top-level ``traj`` path. Non-dict
    or unparseable files are skipped. Returns only truthy values, in discovery order.
    """
    values: list[str] = []
    for tf in list(run_root.rglob("*.json")) + ([traj] if traj.exists() else []):
        try:
            payload = json.loads(tf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        for step in payload.get("steps", []):
            value = (step.get("extra") or {}).get(key)
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
    for tf in list(run_root.rglob("*.json")) + ([traj] if traj.exists() else []):
        try:
            payload = json.loads(tf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
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
    import re as _re

    for c in stub.calls:
        m = _re.search(r"^Fix these (\d+) issues in (.+):$", c["prompt"], _re.M)
        if m is not None and Path(m.group(2)).name == file_basename:
            return int(m.group(1))
    raise AssertionError(f"no batched fix turn found for {file_basename}")


def _single_fix_calls_for(stub: "_StubBackend", file_basename: str) -> list[dict[str, Any]]:
    """Single-finding ``phase_fix`` calls whose ``File:`` line names *file_basename*.

    Robust to issue #336's "Allowed files" clause, which legitimately lists every
    reviewed-diff file in every prompt: a substring check like
    ``"api.py" in prompt`` would over-count by also matching the App.tsx prompt
    (whose allowed-files clause names api.py). Filtering on the ``File:`` line
    captures only the calls actually fixing *file_basename*.
    """
    import re as _re

    out: list[dict[str, Any]] = []
    for c in stub.calls:
        prompt = c["prompt"]
        if not prompt.lower().startswith("fix this issue"):
            continue
        m = _re.search(r"^File: (.+)$", prompt, _re.M)
        if m is not None and Path(m.group(1).strip()).name == file_basename:
            out.append(c)
    return out


def _install_accept_gate_pipeline(monkeypatch: pytest.MonkeyPatch, target: Path, mute: Mute) -> _StubBackend:
    """Patch the deep pipeline for a fix-gate-ACCEPT run.

    Bundles the setup every accept-the-gate test shares: pin the interactive
    stdin/CI axis so a forced accept is honoured, silence the deep UI noise
    (including the recommendation verification summary), force every
    ``prompt_user`` seam to ``"y"`` (belt-and-suspenders alongside
    ``assume="yes"``, which short-circuits the gate before any prompt runs),
    and stub the non-idempotent PR-post / test / commit steps. ``phase_fix``
    stays REAL. Returns the stub backend.
    """
    _force_interactive(monkeypatch)
    _silence(monkeypatch, prompts=False)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *a, **kw: "y")
    mute()

    return _install_stub_backend(monkeypatch, target)


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


class _CommittingStubBackend(_StubBackend):
    """Stub backend for real-commit flow tests.

    ``_do_commit`` commits host-side (issue #726) — no agent turn, no commit
    prompt to answer — so this stub only serves the run's fix/heal turns.
    """

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ) -> Any:
        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
        ):
            yield event


class _PushingCommittingStubBackend(_StubBackend):
    """Run stub for real commit/push flow tests (commit is host-side, issue #726)."""

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ) -> Any:
        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
        ):
            yield event


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


def _uncovered_sweep_target(tmp_path: Path) -> Path:
    """Git repo whose diff has one file NO per-stack reviewer reads.

    ``notes.txt`` is an ambiguous-extension file routed to the generic stack;
    the stub leaves it unread (``per_stack_unread``) and its hunk is large
    enough (6 added lines) to clear the sweep's ``uncovered_sweep_min_hunk_lines``
    budget, so it is the single file the sweep covers. The other files' hunks
    are trivially small (<5 changed lines), so the sweep has exactly one target.
    """
    project = tmp_path / "sweep_target"
    project.mkdir()
    (project / "api.py").write_text("def hello():\n    return 'world'\n")
    (project / "App.tsx").write_text("export const App = () => <div>hello</div>;\n")
    (project / "README.md").write_text("# Project\n")
    _init_repo(project)
    _git(project, "add", ".")
    _commit(project, "init")
    _git(project, "checkout", "-b", "feature")
    (project / "api.py").write_text("def hello():\n    return 'universe'\n")
    (project / "App.tsx").write_text("export const App = () => <div>universe</div>;\n")
    (project / "README.md").write_text("# Project\n\nUpdated.\n")
    (project / "notes.txt").write_text("".join(f"line{i}\n" for i in range(1, 7)))
    _git(project, "add", ".")
    _commit(project, "change")
    return project


def _install_uncovered_sweep_stub(monkeypatch: pytest.MonkeyPatch, target: Path) -> _StubBackend:
    """Leave notes.txt unread by per-stack agents so the sweep can exercise it."""
    stub = _install_stub_backend(monkeypatch, target)
    stub.per_stack_emit_reads = True
    stub.per_stack_unread = frozenset({"notes.txt"})
    return stub


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
    """Prime a merge resume whose only arbiter-eligible records are *python*'s.

    Unlike ``_prime_merge_resume_records`` the structural record sits at
    ``api.py:5``, alone at that location: the contested-location branch would
    otherwise pull both it and the python record into arbitration whenever their
    severities diverge, which muddies "exactly this record was adjudicated".
    """
    return _prime_merge_resume(
        target,
        python=python,
        react=[_record(description="tsx issue", file="App.tsx", evidence="App.tsx:1")],
        generic=[_record(description="docs issue", file="README.md", evidence="README.md:1")],
        structure=[_record(description="structural issue", line=5, evidence="api.py:5")],
    )


class _RejectingArbiterBackend(_StubBackend):
    """Arbiter stub that rejects exactly the record carrying *reject_uid* (#1111).

    The shipped stub echoes ``keep=true`` for every ``arb_id``, so no test could
    observe a *drop* crossing the disk boundary. This one reads the uid the
    orchestrator wrote into ``arbiter-input.json`` and names its target by that
    uid, which is how a real reviewer would name a specific record among several
    that share the reviewer's ``id``.
    """

    def __init__(self, target: Path, reject_uid: str) -> None:
        super().__init__(target)
        self._reject_uid = reject_uid

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        if "you are the arbiter" in prompt.lower():
            self.calls.append({"prompt": prompt, "model": self.model})
            match = re.search(r"listed in (\S+arbiter-input\.json)", prompt)
            assert match is not None, "arbiter prompt did not point at its input artifact"
            entries = json.loads(Path(match.group(1)).read_text())
            yield TextEvent(text="")
            yield ResultEvent(
                structured_output={
                    "findings": [
                        {
                            "arb_id": entry["arb_id"],
                            "keep": entry["uid"] != self._reject_uid,
                            "severity": entry.get("severity") or "high",
                            "confidence": entry.get("confidence") or "HIGH",
                            "description": f"ARBITRATED: {entry.get('description')}",
                            "rationale": "arbiter second opinion",
                        }
                        for entry in entries
                    ]
                },
                continuation=None,
            )
            return
        async for event in super().execute(cwd, prompt, output_schema, continuation, agents, max_turns, read_only):
            yield event


def _merged_items(deep: Path) -> list[dict[str, Any]]:
    """Return the canonical merged item list written under *deep*."""
    return cast("list[dict[str, Any]]", json.loads((deep / "merged-items.json").read_text())["items"])


def _run_uid_pool(deep: Path) -> set[str]:
    """Every record uid this run actually minted, read back off its artifacts.

    The pool is read from disk rather than hardcoded because it is exactly what
    ``_validate_agent_source_uids`` builds to check the agent's claims against:
    asserting a shipped attribution is a subset of it is asserting the shipped
    item names a record that exists.
    """
    return {
        uid
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


def _panel_text(capsys: pytest.CaptureFixture[str]) -> str:
    """Return captured console output with rich's panel framing normalized away.

    ``print_warning`` renders inside a bordered panel, so a message long enough
    to wrap arrives with ``│`` gutters and newlines spliced into the middle of
    it. Dropping the border glyphs and collapsing whitespace lets a test assert
    the sentence the operator reads rather than the width it happened to wrap at.
    """
    text = capsys.readouterr().out.replace("│", " ").replace("║", " ")
    return " ".join(text.split())


def _prime_source_uid_merge_resume(
    target: Path,
    *,
    structure: list[dict[str, Any]] | None = None,
) -> Path:
    """Prime a four-stack merge resume whose uid pool is known before the run.

    Every stack ``multi_stack_target`` detects gets exactly one grounded record,
    so the pool is exactly ``{generic:1, python:1, react:1, structure:1}`` and a
    test can name a real uid (or a plausible non-existent one) in the merge
    agent's output up front -- which a fresh run cannot do, since the records do
    not exist until it has already merged them.

    Each record sits on its own file and the structural record sits alone at
    ``api.py:5``, so arbiter selection's contested-location branch never fires
    and no verdict rewrites a description these tests match on.

    Every primed record carries its ``uid`` ON DISK, exactly as a fresh run
    writes it at record birth. That is load-bearing rather than cosmetic:
    ``_validate_agent_source_uids`` builds the run's uid pool by re-reading the
    records FILES -- the same bytes the merge agent is pointed at -- so a resume
    primed with uid-less records (a pre-#1111 artifact) legitimately has an empty
    pool, and every uid a test made the agent cite would be dropped as invented.
    """
    return _prime_merge_resume(
        target,
        python=[_record(description="py issue", evidence="api.py:1", uid="python:1")],
        react=[_record(description="tsx issue", file="App.tsx", evidence="App.tsx:1", uid="react:1")],
        generic=[
            _record(
                description="docs issue",
                file="README.md",
                evidence="README.md:1",
                uid="generic:1",
            )
        ],
        structure=(
            structure
            if structure is not None
            else [
                _record(
                    description="structural issue",
                    line=5,
                    evidence="api.py:5",
                    uid="structure:1",
                )
            ]
        ),
    )


def _provenance_item(
    item_id: int,
    description: str,
    *,
    file: str = "api.py",
    line: int = 1,
    source_uids: Any = None,
    evidence: str | None = None,
    severity: str = "medium",
    rationale: str = "rationale",
    omit_source_uids: bool = False,
) -> dict[str, Any]:
    """Build one merge-agent item with an explicitly chosen provenance claim.

    ``_merge_item`` cannot serve here: these tests are about the ``source_uids``
    value itself, including the shapes a real model gets wrong (a null, a bare
    string, an omitted key), so every one of them has to be settable.
    """
    item: dict[str, Any] = {
        "id": item_id,
        "lens": "per-stack",
        "file": file,
        "line": line,
        "severity": severity,
        "description": description,
        "confidence": "MEDIUM",
        "rationale": rationale,
        "evidence": f"{file}:{line}" if evidence is None else evidence,
    }
    if not omit_source_uids:
        item["source_uids"] = source_uids
    return item


async def _fresh_uid_run(
    target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Mute,
) -> Path:
    """Run the standard multi-stack fixture and return its deep artifact directory."""
    from daydream.deep.artifacts import deep_dir

    _silence(monkeypatch)
    mute_side_effects()
    _install_stub_backend(monkeypatch, target)
    assert await _run_deep(target) == 0
    return deep_dir(target, allow_standalone=True)


def _item_uids(deep: Path) -> list[str]:
    """Return the ``item_uid`` of every shipped item, in canonical file order."""
    return [str(item.get("item_uid")) for item in _merged_items(deep)]


def _direct_fix_context(
    repo: Path,
    items: list[dict[str, Any]],
    *,
    changed_files: set[str],
    start_at: str = "review",
) -> Any:
    """Build the smallest real-Git FlowContext for fix-boundary unit tests."""
    from daydream import git_ops
    from daydream.extensions import Registry
    from daydream.flows.engine import FlowContext
    from daydream.runner import RunConfig
    from daydream.workspace import WorkContext

    dd = repo / ".daydream" / "deep"
    dd.mkdir(parents=True, exist_ok=True)
    items_file = dd / "merged-items.json"
    items_file.write_text(json.dumps({"items": items}))
    head = git_ops.head_sha(repo)
    return FlowContext(
        config=RunConfig(target=str(repo), assume="yes", cleanup=False, start_at=start_at),
        work=WorkContext(
            repo=repo,
            source=repo,
            base_branch="main",
            base_sha=head,
            head_branch="main",
            head_sha=head,
            is_ephemeral=False,
            run_id="session-current",
        ),
        registry=Registry(),
        allow_standalone_artifacts=True,
        data={
            "dd": dd,
            "items_file": items_file,
            "merged_report": dd / "merged-review.md",
            "changed_files": changed_files,
        },
    )


def _direct_fix_state(ctx: Any, items: list[dict[str, Any]], reviewed: set[str]) -> Any:
    from daydream import git_ops
    from daydream.deep.fix_steps import FixCycleState
    from daydream.fix_footprint import AuthorizedFixFootprint

    footprint = AuthorizedFixFootprint.build(ctx.work.repo, reviewed, items)
    state = FixCycleState(
        session_id="session-current",
        stable_ref=git_ops.head_sha(ctx.work.repo),
        stable_head=git_ops.head_sha(ctx.work.repo),
        initial_index=git_ops.snapshot_index(ctx.work.repo),
        preexisting_untracked=git_ops.snapshot_untracked_paths(ctx.work.repo, include_runtime_artifacts=False),
        preexisting_gitlinks=git_ops.snapshot_worktree_gitlinks(ctx.work.repo),
        footprint=footprint,
    )
    ctx.data["fix_cycle_state"] = state
    ctx.data["items"] = items
    return state


def _remote_identity_context(
    tmp_path: Path,
    fake_gh: Any,
    *,
    head_repository: str | None,
    configured_repository: str = "base-user/project",
    base_ref: str = "main",
    configured_pr: int = 7,
) -> tuple[Any, str]:
    from daydream import git_ops

    repo = tmp_path / "remote-identity"
    _init_repo(repo)
    (repo / "a.py").write_text("A = 1\n")
    _git(repo, "add", "a.py")
    _commit(repo, "base")
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
    fake_gh.serve_pr_view(
        {
            "number": 7,
            "title": "Fix",
            "body": "",
            "state": "OPEN",
            "headRefName": "feature",
            "baseRefName": base_ref,
            "headRefOid": sha,
            "url": "https://github.com/base-user/project/pull/7",
            "headRepository": head_row,
            "headRepositoryOwner": head_owner,
        }
    )
    ctx = _direct_fix_context(repo, [], changed_files=set())
    ctx.config.pr_number = configured_pr
    ctx.config.pr_repo = configured_repository
    return ctx, sha


def _finalization_fixture(tmp_path: Path) -> tuple[Any, Any, Any]:
    from daydream.deep.fix_steps import capture_retained_tree

    repo = tmp_path / "finalization"
    _init_repo(repo)
    (repo / "a.py").write_text("A = 1\n")
    _git(repo, "add", "a.py")
    _commit(repo, "base")
    items = [{**_merge_item(1, "a.py", "high"), "item_uid": "item:a", "related_files": []}]
    ctx = _direct_fix_context(repo, items, changed_files={"a.py"})
    state = _direct_fix_state(ctx, items, {"a.py"})
    (repo / "a.py").write_text("A = 2\n")
    snapshot = capture_retained_tree(ctx.work, state)
    return ctx, state, snapshot
