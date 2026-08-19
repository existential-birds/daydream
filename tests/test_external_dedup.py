"""Tests for external-bot finding dedup.

Covers the four layers of the feature:
  1. ``reconcile.fetch_external_findings`` — competitor-comment inventory + author filter.
  2. ``deep.dedup.build_external_dedup_candidates`` — location pre-filter.
  3. ``phases.phase_dedup_external`` — the adjudicated suppression (real path: real
     temp files + merged-items.json, mocking only the GitHub fetch and the agent seam).
  4. ``pr_review.parsed_issues_from_items`` — the disposition is honored so suppressed
     items never reach the PR or the findings artifact.
  5. ``deep.orchestrator._spine_dedup_external`` — enable gate (opt-in + resume guard).
  6. ``deep.orchestrator._step_dedup_external`` — PR resolution branches and GitError
     handling.
  7. The full ``dedup-external`` FlowStep, driven end-to-end through
     ``runner.run`` with only the ``gh`` subprocess boundary faked (``fake_gh``),
     exercising the real step→phase wiring and the real GraphQL fetch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, runner
from daydream.backends import ResultEvent
from daydream.config_file import DaydreamFileConfig
from daydream.deep.dedup import build_external_dedup_candidates
from daydream.pr_review import parsed_issues_from_items
from daydream.reconcile import ExternalComment, fetch_external_findings
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_gh import FakeGh

# --- fetch_external_findings ----------------------------------------------


def _ext_thread(path: str, line: int | None, *, original: int | None = None,
                author: str, body: str, url: str) -> dict[str, Any]:
    return {
        "path": path,
        "line": line,
        "originalLine": original,
        "comments": {"nodes": [{"body": body, "url": url, "author": {"login": author}}]},
    }


def _ext_page(nodes: list[dict[str, Any]], *, next_cursor: str | None = None) -> dict[str, Any]:
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "reviewThreads": {
                        "pageInfo": {"hasNextPage": next_cursor is not None, "endCursor": next_cursor},
                        "nodes": nodes,
                    }
                }
            }
        }
    }


def test_fetch_external_findings_filters_by_author(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    page = _ext_page([
        _ext_thread("a.py", 10, author="greptile-apps", body="bug here", url="u1"),
        _ext_thread("b.py", 20, author="some-human", body="human note", url="u2"),
    ])
    monkeypatch.setattr(git_ops, "gh_api", lambda *a, **k: page)
    found = fetch_external_findings(tmp_path, "o/r", 7, bot_logins=["greptile-apps[bot]"])
    assert [c.path for c in found] == ["a.py"]
    assert found[0].url == "u1"
    assert found[0].line == 10


def test_fetch_external_findings_ignores_original_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # originalLine is a base-commit coordinate; we must not store it in
    # ExternalComment.line (head-commit space) — the downstream filter uses
    # file-level matching when line is None.
    page = _ext_page([_ext_thread("a.py", None, original=42, author="greptile-apps",
                                  body="x", url="u1")])
    monkeypatch.setattr(git_ops, "gh_api", lambda *a, **k: page)
    found = fetch_external_findings(tmp_path, "o/r", 7, bot_logins=["greptile-apps"])
    assert found[0].line is None


def test_fetch_external_findings_empty_bots_makes_no_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("gh_api must not be called when no bots are configured")

    monkeypatch.setattr(git_ops, "gh_api", _boom)
    assert fetch_external_findings(tmp_path, "o/r", 7, bot_logins=[]) == []


# --- build_external_dedup_candidates --------------------------------------


def _item(item_id: int, file: str, line: int | None, desc: str = "d") -> dict[str, Any]:
    return {"id": item_id, "file": file, "line": line, "description": desc}


def _ext(path: str, line: int | None, url: str = "u") -> ExternalComment:
    return ExternalComment(path=path, line=line, body="b", url=url, author="greptile")


def test_candidates_pair_within_window() -> None:
    pairs = build_external_dedup_candidates([_item(1, "a.py", 10)], [_ext("a.py", 15)], line_window=10)
    assert len(pairs) == 1
    assert pairs[0].item_id == 1


def test_candidates_skip_outside_window() -> None:
    pairs = build_external_dedup_candidates([_item(1, "a.py", 10)], [_ext("a.py", 100)], line_window=10)
    assert pairs == []


def test_candidates_skip_different_file() -> None:
    pairs = build_external_dedup_candidates([_item(1, "a.py", 10)], [_ext("b.py", 10)])
    assert pairs == []


def test_candidates_file_level_fallback_when_line_unknown() -> None:
    # Item has no line -> same-file comment still pairs (LLM adjudicates).
    pairs = build_external_dedup_candidates([_item(1, "a.py", None)], [_ext("a.py", 999)])
    assert len(pairs) == 1


# --- parsed_issues_from_items honors the disposition ----------------------


def test_parsed_issues_skips_deduped_external() -> None:
    items = [
        {"file": "a.py", "line": 1, "description": "keep me", "severity": "high"},
        {"file": "b.py", "line": 2, "description": "drop me", "severity": "high",
         "disposition": "deduped-vs-external", "external_ref": "u1"},
    ]
    parsed = parsed_issues_from_items(items)
    assert [p.path for p in parsed] == ["a.py"]


# --- phase_dedup_external (real path: real files, mocked GitHub + agent) ---


@pytest.mark.asyncio
async def test_phase_dedup_external_suppresses_only_high_confidence_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Any, silence_console: Any
) -> None:
    silence_console("daydream.phases")
    from daydream import phases

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    items_file = deep_dir / "merged-items.json"
    items = [
        {"id": 1, "file": "a.py", "line": 10, "description": "off-by-one in loop", "severity": "high"},
        {"id": 2, "file": "b.py", "line": 20, "description": "unique daydream finding", "severity": "high"},
    ]
    items_file.write_text(json.dumps({"items": items, "held": []}))

    # greptile already flagged item 1's location; nothing near item 2.
    external = [ExternalComment(path="a.py", line=11, body="loop overruns by one",
                               url="https://gh/c/1", author="greptile-apps")]
    from daydream import reconcile

    # The phase late-imports fetch_external_findings from reconcile, so patch it there.
    monkeypatch.setattr(reconcile, "fetch_external_findings", lambda *a, **k: external)

    verdicts = {
        "verdicts": [
            {"item_id": 1, "external_ref": "https://gh/c/1", "duplicate": True,
             "confidence": "high", "reason": "same off-by-one"},
        ]
    }
    backend = ScriptedBackend(events=(ResultEvent(structured_output=verdicts, continuation=None),))

    suppressed = await phases.phase_dedup_external(
        backend,
        make_work(tmp_path),
        merged_items_path=items_file,
        deep_dir=deep_dir,
        repo_slug="o/r",
        pr_number=7,
        bot_logins=["greptile-apps[bot]"],
    )

    assert suppressed == 1
    written = json.loads(items_file.read_text())["items"]
    by_id = {i["id"]: i for i in written}
    assert by_id[1]["disposition"] == "deduped-vs-external"
    assert by_id[1]["external_ref"] == "https://gh/c/1"
    assert "disposition" not in by_id[2]

    sidecar = json.loads((deep_dir / "external-dedup.json").read_text())
    assert sidecar["suppressed"][0]["id"] == 1

    # The suppressed item never reaches the postable/artifact issue list.
    parsed = parsed_issues_from_items(written)
    assert [p.path for p in parsed] == ["b.py"]


@pytest.mark.asyncio
async def test_phase_dedup_external_keeps_low_confidence_and_non_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Any, silence_console: Any
) -> None:
    silence_console("daydream.phases")
    from daydream import phases, reconcile

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    items_file = deep_dir / "merged-items.json"
    items = [{"id": 1, "file": "a.py", "line": 10, "description": "maybe related", "severity": "high"}]
    items_file.write_text(json.dumps({"items": items, "held": []}))

    external = [ExternalComment(path="a.py", line=10, body="different concern",
                               url="u1", author="greptile-apps")]
    monkeypatch.setattr(reconcile, "fetch_external_findings", lambda *a, **k: external)

    # duplicate=True but only medium confidence -> must NOT suppress.
    verdicts = {"verdicts": [{"item_id": 1, "external_ref": "u1", "duplicate": True,
                              "confidence": "medium", "reason": "unsure"}]}
    backend = ScriptedBackend(events=(ResultEvent(structured_output=verdicts, continuation=None),))

    suppressed = await phases.phase_dedup_external(
        backend, make_work(tmp_path), merged_items_path=items_file, deep_dir=deep_dir,
        repo_slug="o/r", pr_number=7, bot_logins=["greptile-apps"],
    )
    assert suppressed == 0
    assert "disposition" not in json.loads(items_file.read_text())["items"][0]


@pytest.mark.asyncio
async def test_phase_dedup_external_no_candidates_skips_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Any, silence_console: Any
) -> None:
    silence_console("daydream.phases")
    from daydream import phases, reconcile

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    items_file = deep_dir / "merged-items.json"
    items_file.write_text(json.dumps({"items": [{"id": 1, "file": "a.py", "line": 10,
                                                 "description": "x"}], "held": []}))
    # Competitor comment on an unrelated file -> no candidates.
    monkeypatch.setattr(reconcile, "fetch_external_findings",
                        lambda *a, **k: [ExternalComment(path="z.py", line=1, body="b", url="u", author="g")])

    class _NoExecute(ScriptedBackend):
        async def execute(self, *a: Any, **k: Any):  # type: ignore[override]
            raise AssertionError("adjudicator must not run when there are no candidates")
            yield  # pragma: no cover

    suppressed = await phases.phase_dedup_external(
        _NoExecute(), make_work(tmp_path), merged_items_path=items_file, deep_dir=deep_dir,
        repo_slug="o/r", pr_number=7, bot_logins=["greptile-apps"],
    )
    assert suppressed == 0
    assert json.loads((deep_dir / "external-dedup.json").read_text()) == {"suppressed": []}


# --- dedup-external FlowStep, driven end-to-end via runner.run -------------
#
# The tests above call ``phase_dedup_external`` directly (mocking
# ``reconcile.fetch_external_findings``) or call ``_step_dedup_external``
# directly (mocking ``phase_dedup_external``). Neither exercises the real
# ``FlowStep`` dispatch into the real phase, nor the real GraphQL fetch. This
# test registers a tiny fork flow that seeds merged-items.json and then runs
# the *real* built-in ``dedup-external`` step through ``runner.run``, faking
# only the ``gh`` subprocess boundary (``fake_gh``) and the agent seam.

_SEED_DEDUP_FLOW_EXT = """
import json

from daydream.extensions import FlowStep


async def _seed(ctx):
    deep_dir = ctx.work.repo / ".daydream" / "deep"
    deep_dir.mkdir(parents=True, exist_ok=True)
    items_file = deep_dir / "merged-items.json"
    items = [
        {"id": 1, "file": "api.py", "line": 1, "description": "off-by-one in loop", "severity": "high"},
        {"id": 2, "file": "App.tsx", "line": 1, "description": "unique daydream finding", "severity": "high"},
    ]
    items_file.write_text(json.dumps({"items": items, "held": []}))
    ctx.data["items_file"] = items_file
    ctx.data["dd"] = deep_dir
    ctx.data["merged_report"] = deep_dir / "merged-report.md"


def register(r):
    r.register_phase(FlowStep(name="seed-dedup", run=_seed))
    r.set_flow("dedup-real-path", ["seed-dedup", "dedup-external"])
"""


def _serve_open_pr(fake_gh: FakeGh, target: Path) -> None:
    fake_gh.serve_pr_view(
        {
            "number": 7,
            "state": "OPEN",
            "headRefName": "feature",
            "baseRefName": "main",
            "headRefOid": git_ops.head_sha(target),
            "baseRefOid": git_ops.merge_base(target, "main"),
            "url": "https://github.com/acme/widgets/pull/7",
            "body": "",
        }
    )


async def test_dedup_external_step_wires_to_real_phase_and_fetch_via_runner_run(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    fake_gh: FakeGh,
    install_backend: Any,
    make_config: Any,
) -> None:
    """Real ``FlowStep`` dispatch into the real phase, real fetch, faked ``gh``.

    Only the ``gh`` subprocess boundary is faked (via ``fake_gh``, which
    intercepts ``subprocess.run`` inside ``git_ops``); ``reconcile.fetch_external_findings``,
    ``phases.phase_dedup_external``, and ``_step_dedup_external`` all run for real.
    """
    ext_dir.write_module(_SEED_DEDUP_FLOW_EXT)
    _serve_open_pr(fake_gh, multi_stack_target)
    # greptile already flagged api.py near the item-1 location.
    page = _ext_page([_ext_thread("api.py", 3, author="greptile-apps",
                                  body="loop overruns by one", url="https://gh/c/1")])
    fake_gh.set_response("graphql_threads", value=page)

    verdicts = {
        "verdicts": [
            {"item_id": 1, "external_ref": "https://gh/c/1", "duplicate": True,
             "confidence": "high", "reason": "same off-by-one"},
        ]
    }
    install_backend(ScriptedBackend(events=(ResultEvent(structured_output=verdicts, continuation=None),)))

    rc = await runner.run(
        make_config(
            multi_stack_target,
            pr_number=7,
            flow_name="dedup-real-path",
            file_config=DaydreamFileConfig(external_review_bots=["greptile-apps[bot]"]),
        )
    )

    assert rc == 0
    items_file = multi_stack_target / ".daydream" / "deep" / "merged-items.json"
    by_id = {item["id"]: item for item in json.loads(items_file.read_text())["items"]}
    assert by_id[1]["disposition"] == "deduped-vs-external"
    assert by_id[1]["external_ref"] == "https://gh/c/1"
    assert "disposition" not in by_id[2]

    sidecar = json.loads((multi_stack_target / ".daydream" / "deep" / "external-dedup.json").read_text())
    assert sidecar["suppressed"][0]["id"] == 1


# --- _spine_dedup_external (enable gate) -------------------------------------


def _make_flow_ctx(tmp_path: Path, **run_config_kwargs: Any) -> Any:
    """Build a minimal FlowContext for spine-gate unit tests."""
    from daydream.extensions import Registry
    from daydream.flows.engine import FlowContext
    from daydream.runner import RunConfig
    from daydream.workspace import WorkContext

    config = RunConfig(target=str(tmp_path), **run_config_kwargs)
    work = WorkContext(
        repo=tmp_path,
        source=tmp_path,
        base_branch="main",
        base_sha="",
        head_branch=None,
        head_sha="",
        is_ephemeral=False,
        run_id="test",
    )
    return FlowContext(config=config, work=work, registry=Registry(), data={})


def test_spine_dedup_external_off_by_default(tmp_path: Path) -> None:
    """No ``external_review_bots`` configured → step is disabled."""
    from daydream.deep.orchestrator import _spine_dedup_external

    assert _spine_dedup_external(_make_flow_ctx(tmp_path)) is False


def test_spine_dedup_external_on_when_bots_configured(tmp_path: Path) -> None:
    """Non-empty ``external_review_bots`` on a fresh run enables the step."""
    from daydream.config_file import DaydreamFileConfig
    from daydream.deep.orchestrator import _spine_dedup_external

    fc = DaydreamFileConfig(external_review_bots=["greptile-apps[bot]"])
    assert _spine_dedup_external(_make_flow_ctx(tmp_path, file_config=fc)) is True


def test_spine_dedup_external_off_on_fix_resume(tmp_path: Path) -> None:
    """``--start-at fix`` resume must not re-fetch competitor comments."""
    from daydream.config_file import DaydreamFileConfig
    from daydream.deep.orchestrator import _spine_dedup_external

    fc = DaydreamFileConfig(external_review_bots=["greptile-apps[bot]"])
    assert _spine_dedup_external(_make_flow_ctx(tmp_path, file_config=fc, start_at="fix")) is False


# --- _step_dedup_external (orchestrator wiring) ------------------------------


def _make_step_ctx(tmp_path: Path, items_file: Path, deep_dir: Path,
                   bot_logins: list[str], pr_number: int | None = None) -> Any:
    """Build a FlowContext wired with the data keys _step_dedup_external reads."""
    from daydream.config_file import DaydreamFileConfig
    from daydream.extensions import Registry
    from daydream.flows.engine import FlowContext
    from daydream.runner import RunConfig
    from daydream.workspace import WorkContext

    fc = DaydreamFileConfig(external_review_bots=bot_logins)
    config = RunConfig(target=str(tmp_path), file_config=fc, pr_number=pr_number,
                       non_interactive=True, cleanup=False, archive=False)
    work = WorkContext(
        repo=tmp_path,
        source=tmp_path,
        base_branch="main",
        base_sha="",
        head_branch=None,
        head_sha="",
        is_ephemeral=False,
        run_id="test",
    )
    ctx = FlowContext(config=config, work=work, registry=Registry(),
                      data={"items_file": items_file, "dd": deep_dir})
    return ctx


def _fake_pr() -> Any:
    """Minimal PRInfo-like object with the fields _step_dedup_external uses."""
    from daydream.pr_review import PRInfo

    return PRInfo(number=7, head_sha="abc", base_sha="def", base_ref="main",
                  owner="myorg", repo="myrepo", url="https://gh/pr/7")


@pytest.mark.asyncio
async def test_step_dedup_external_git_error_warns_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, silence_console: Any,
) -> None:
    """GitError during PR resolution → warns, does not call phase_dedup_external."""
    silence_console("daydream.deep.orchestrator")
    from daydream import pr_review
    from daydream.deep.orchestrator import _step_dedup_external
    from daydream.git_ops import GitError

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    items_file = deep_dir / "merged-items.json"
    items_file.write_text(json.dumps({"items": [], "held": []}))

    def _raise_git_error(*a: Any, **k: Any) -> None:
        raise GitError("no remote")

    monkeypatch.setattr(pr_review, "find_open_pr", _raise_git_error)

    called = []

    async def _no_call(*a: Any, **k: Any) -> int:
        called.append(True)
        return 0

    monkeypatch.setattr("daydream.deep.orchestrator.phase_dedup_external", _no_call)

    ctx = _make_step_ctx(tmp_path, items_file, deep_dir, bot_logins=["greptile-apps[bot]"])
    result = await _step_dedup_external(ctx)

    assert result is None  # warn-and-continue: step does not Stop the flow
    assert called == []    # phase_dedup_external must not be invoked


@pytest.mark.asyncio
async def test_step_dedup_external_no_pr_warns_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, silence_console: Any,
) -> None:
    """No resolvable PR → warns, does not call phase_dedup_external."""
    silence_console("daydream.deep.orchestrator")
    from daydream import pr_review
    from daydream.deep.orchestrator import _step_dedup_external

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    items_file = deep_dir / "merged-items.json"
    items_file.write_text(json.dumps({"items": [], "held": []}))

    monkeypatch.setattr(pr_review, "find_open_pr", lambda *a, **k: None)

    called = []

    async def _no_call(*a: Any, **k: Any) -> int:
        called.append(True)
        return 0

    monkeypatch.setattr("daydream.deep.orchestrator.phase_dedup_external", _no_call)

    ctx = _make_step_ctx(tmp_path, items_file, deep_dir, bot_logins=["greptile-apps[bot]"])
    result = await _step_dedup_external(ctx)

    assert result is None
    assert called == []


@pytest.mark.asyncio
async def test_step_dedup_external_open_pr_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, silence_console: Any,
) -> None:
    """No pinned pr_number → uses find_open_pr and delegates to phase_dedup_external."""
    silence_console("daydream.deep.orchestrator")
    from daydream import pr_review
    from daydream.deep.orchestrator import _step_dedup_external

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    items_file = deep_dir / "merged-items.json"
    items_file.write_text(json.dumps({"items": [], "held": []}))

    pr = _fake_pr()
    monkeypatch.setattr(pr_review, "find_open_pr", lambda *a, **k: pr)

    calls: list[dict[str, Any]] = []

    async def _record(*a: Any, **k: Any) -> int:
        calls.append({"repo_slug": k.get("repo_slug"), "pr_number": k.get("pr_number"),
                       "bot_logins": k.get("bot_logins")})
        return 0

    monkeypatch.setattr("daydream.deep.orchestrator.phase_dedup_external", _record)

    ctx = _make_step_ctx(tmp_path, items_file, deep_dir, bot_logins=["greptile-apps[bot]"])
    result = await _step_dedup_external(ctx)

    assert result is None
    assert len(calls) == 1
    assert calls[0]["repo_slug"] == "myorg/myrepo"
    assert calls[0]["pr_number"] == 7
    assert calls[0]["bot_logins"] == ["greptile-apps[bot]"]


@pytest.mark.asyncio
async def test_step_dedup_external_pinned_pr_number_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, silence_console: Any,
) -> None:
    """Pinned ``--pr-number`` → uses find_pr_by_number (not find_open_pr)."""
    silence_console("daydream.deep.orchestrator")
    from daydream import pr_review
    from daydream.deep.orchestrator import _step_dedup_external

    deep_dir = tmp_path / ".daydream" / "deep"
    deep_dir.mkdir(parents=True)
    items_file = deep_dir / "merged-items.json"
    items_file.write_text(json.dumps({"items": [], "held": []}))

    pr = _fake_pr()
    find_by_number_calls: list[int] = []

    def _find_by_number(target_dir: Path, number: int) -> Any:
        find_by_number_calls.append(number)
        return pr

    monkeypatch.setattr(pr_review, "find_pr_by_number", _find_by_number)
    # find_open_pr must not be called on the pinned path.
    def _must_not_be_called(*a: Any, **k: Any) -> None:
        raise AssertionError("find_open_pr must not be called on the pinned-pr-number path")

    monkeypatch.setattr(pr_review, "find_open_pr", _must_not_be_called)

    async def _noop(*a: Any, **k: Any) -> int:
        return 0

    monkeypatch.setattr("daydream.deep.orchestrator.phase_dedup_external", _noop)

    ctx = _make_step_ctx(tmp_path, items_file, deep_dir,
                         bot_logins=["greptile-apps[bot]"], pr_number=42)
    result = await _step_dedup_external(ctx)

    assert result is None
    assert find_by_number_calls == [42]
