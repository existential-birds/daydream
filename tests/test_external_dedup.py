"""Tests for external-bot finding dedup.

Covers the four layers of the feature:
  1. ``reconcile.fetch_external_findings`` — competitor-comment inventory + author filter.
  2. ``deep.dedup.build_external_dedup_candidates`` — location pre-filter, and
     ``batch_external_dedup_pairs`` — the prompt-sized sharding of the pair set.
  3. ``phases.phase_dedup_external`` — the adjudicated suppression (real path: real
     temp files + merged-items.json, mocking only the GitHub fetch and the agent seam).
  4. ``pr_review.parsed_issues_from_items`` — the disposition is honored so suppressed
     items never reach the PR or the findings artifact.
  5. ``deep.orchestrator._spine_dedup_external`` — enable gate (opt-in + resume guard).
  6. ``deep.orchestrator._step_dedup_external`` — PR resolution branches and GitError
     handling.
  7. The full ``external-dedup`` FlowStep, driven end-to-end through
     ``runner.run`` with only the ``gh`` subprocess boundary faked (``fake_gh``),
     exercising the real step→phase wiring and the real GraphQL fetch.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream import git_ops, runner
from daydream.backends import AgentEvent, ResultEvent
from daydream.config_file import DaydreamFileConfig
from daydream.deep.dedup import (
    _EXTERNAL_PAIRS_PER_BATCH,
    batch_external_dedup_pairs,
    build_external_dedup_candidates,
)
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


# --- batch_external_dedup_pairs -------------------------------------------


def test_batch_external_pairs_covers_every_pair_in_order() -> None:
    comments = [_ext("a.py", None, url=f"https://gh/c/{i:03d}") for i in range(6)]
    items = [_item(i, "a.py", None) for i in range(1, 31)]
    pairs = build_external_dedup_candidates(items, comments)
    assert len(pairs) == 180  # 30 items x 6 comments

    batches = batch_external_dedup_pairs(pairs)

    expected_full, remainder = divmod(len(pairs), _EXTERNAL_PAIRS_PER_BATCH)
    assert len(batches) == expected_full + (1 if remainder else 0)
    assert [len(b) for b in batches[:-1]] == [_EXTERNAL_PAIRS_PER_BATCH] * expected_full
    assert len(batches[-1]) == (remainder or _EXTERNAL_PAIRS_PER_BATCH)
    # Nothing lost, nothing duplicated, original order preserved.
    assert [pair for batch in batches for pair in batch] == pairs


def test_batch_external_pairs_single_batch_when_under_the_bound() -> None:
    pairs = build_external_dedup_candidates([_item(1, "a.py", None)], [_ext("a.py", None)])
    assert batch_external_dedup_pairs(pairs) == [pairs]


def test_batch_external_pairs_rejects_non_positive_bound() -> None:
    with pytest.raises(ValueError):
        batch_external_dedup_pairs([], max_per_batch=0)


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


# --- external-dedup FlowStep, driven end-to-end via runner.run -------------
#
# The tests above call ``phase_dedup_external`` directly (mocking
# ``reconcile.fetch_external_findings``) or call ``_step_dedup_external``
# directly (mocking ``phase_dedup_external``). Neither exercises the real
# ``FlowStep`` dispatch into the real phase, nor the real GraphQL fetch. This
# test registers a tiny fork flow that seeds merged-items.json and then runs
# the *real* built-in ``external-dedup`` step through ``runner.run``, faking
# only the ``gh`` subprocess boundary (``fake_gh``) and the agent seam.

_SEED_DEDUP_FLOW_EXT = """
import json

from daydream.extensions import FlowStep


async def _seed(ctx):
    deep_dir = ctx.work.repo / ".daydream" / "deep"
    deep_dir.mkdir(parents=True, exist_ok=True)
    items_file = deep_dir / "merged-items.json"
    items = [
        {"id": 1, "file": "api.py", "line": 1, "description": "off-by-one in loop", "severity": "high",
         "lens": "per-stack"},
        {"id": 2, "file": "App.tsx", "line": 1, "description": "unique daydream finding", "severity": "high",
         "lens": "per-stack"},
    ]
    items_file.write_text(json.dumps({"items": items, "held": []}))
    ctx.data["items_file"] = items_file
    ctx.data["dd"] = deep_dir


def register(r):
    r.register_phase(FlowStep(name="seed-dedup", run=_seed))
    r.set_flow("dedup-real-path", ["seed-dedup", "external-dedup"])
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

    # The step re-renders the deep-dir report from the surviving items even
    # though this flow never ran ``load-items``, so ctx.data has no
    # ``merged_report`` repo-root copy to update.
    report = (multi_stack_target / ".daydream" / "deep" / "review-output.md").read_text()
    assert "off-by-one in loop" not in report
    assert "unique daydream finding" in report


# --- sharded adjudication (>1 batch) --------------------------------------
#
# Item 1 of ``_SEED_DEDUP_FLOW_EXT`` pairs with every ``api.py`` comment, so a
# chatty competitor bot is the way to force more than one adjudicator shard.

_SHARD_COMMENT_COUNT = _EXTERNAL_PAIRS_PER_BATCH + 3
# The last URL of each batch: batch 0 ends at index _EXTERNAL_PAIRS_PER_BATCH-1
# because build_external_dedup_candidates sorts by (item_id, external_url) and
# the zero-padded URLs sort numerically.
_FIRST_SHARD_MARKER = f"https://gh/c/{_EXTERNAL_PAIRS_PER_BATCH - 1:03d}"
_LAST_SHARD_MARKER = f"https://gh/c/{_SHARD_COMMENT_COUNT - 1:03d}"


def _shard_verdict(external_ref: str) -> dict[str, Any]:
    return {
        "verdicts": [
            {"item_id": 1, "external_ref": external_ref, "duplicate": True,
             "confidence": "high", "reason": "same off-by-one"},
        ]
    }


class _PerShardBackend(ScriptedBackend):
    """Routes each shard's turn by a URL marker unique to that shard's prompt.

    A mapped value is either the structured output to return or an exception to
    raise. ``delays`` lets a test invert completion order relative to batch
    order; ``completed`` records the order shards actually finished in.
    """

    def __init__(self, by_marker: dict[str, Any], *, delays: dict[str, float] | None = None) -> None:
        super().__init__()
        self._by_marker = by_marker
        self._delays = delays or {}
        self.completed: list[str] = []

    async def execute(  # type: ignore[override]
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        self.calls.append({"cwd": cwd, "prompt": prompt, "output_schema": output_schema,
                           "continuation": continuation, "agents": agents, "max_turns": max_turns,
                           "read_only": read_only, "persist_session": persist_session})
        matches = [marker for marker in self._by_marker if marker in prompt]
        assert len(matches) == 1, f"prompt matched {matches}, expected exactly one shard marker"
        marker = matches[0]
        delay = self._delays.get(marker, 0.0)
        if delay:
            await anyio.sleep(delay)
        self.completed.append(marker)
        outcome = self._by_marker[marker]
        if isinstance(outcome, BaseException):
            raise outcome
        yield ResultEvent(structured_output=outcome, continuation=None)


def _serve_chatty_bot(fake_gh: FakeGh, count: int = _SHARD_COMMENT_COUNT) -> None:
    fake_gh.set_response(
        "graphql_threads",
        value=_ext_page([
            _ext_thread("api.py", 3, author="greptile-apps", body=f"concern {i}",
                        url=f"https://gh/c/{i:03d}")
            for i in range(count)
        ]),
    )


async def test_dedup_external_shards_adjudicate_every_pair_via_runner_run(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    fake_gh: FakeGh,
    install_backend: Any,
    make_config: Any,
) -> None:
    """Every candidate pair is adjudicated, across as many shards as it takes.

    The suppressing verdict is returned only by the *second* shard and cites a
    URL that exists only in that shard's batch, so the assertion can only pass
    if the tail of the candidate set really reached an adjudicator.
    """
    ext_dir.write_module(_SEED_DEDUP_FLOW_EXT)
    _serve_open_pr(fake_gh, multi_stack_target)
    _serve_chatty_bot(fake_gh)

    backend = _PerShardBackend({
        _FIRST_SHARD_MARKER: {"verdicts": []},
        _LAST_SHARD_MARKER: _shard_verdict(_LAST_SHARD_MARKER),
    })
    install_backend(backend)

    rc = await runner.run(
        make_config(
            multi_stack_target,
            pr_number=7,
            flow_name="dedup-real-path",
            file_config=DaydreamFileConfig(external_review_bots=["greptile-apps[bot]"]),
        )
    )

    assert rc == 0
    prompts = backend.prompts
    assert len(prompts) == 2
    # The union of the delivered prompts covers every candidate pair.
    adjudicated = {
        f"https://gh/c/{i:03d}"
        for i in range(_SHARD_COMMENT_COUNT)
        if any(f"external_ref=https://gh/c/{i:03d}" in p for p in prompts)
    }
    assert len(adjudicated) == _SHARD_COMMENT_COUNT
    assert sum(p.count("external_ref=https://gh/c/") for p in prompts) == _SHARD_COMMENT_COUNT

    by_id = {item["id"]: item
             for item in json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())
             ["items"]}
    assert by_id[1]["disposition"] == "deduped-vs-external"
    assert by_id[1]["external_ref"] == _LAST_SHARD_MARKER
    assert "disposition" not in by_id[2]


async def test_dedup_external_failed_shard_warns_and_suppresses_nothing_via_runner_run(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    fake_gh: FakeGh,
    install_backend: Any,
    make_config: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shard whose agent call raises loses only its own pairs, and says so.

    The other shard succeeds, which is what makes this the warn path rather
    than the propagate path.
    """
    ext_dir.write_module(_SEED_DEDUP_FLOW_EXT)
    _serve_open_pr(fake_gh, multi_stack_target)
    _serve_chatty_bot(fake_gh)

    warnings: list[str] = []
    monkeypatch.setattr("daydream.phases.print_warning", lambda _console, message: warnings.append(message))

    # The failing shard is the one that would have suppressed item 1: its
    # verdict is lost, so nothing is suppressed at all.
    backend = _PerShardBackend({
        _FIRST_SHARD_MARKER: {"verdicts": []},
        _LAST_SHARD_MARKER: RuntimeError("adjudicator exploded"),
    })
    install_backend(backend)

    rc = await runner.run(
        make_config(
            multi_stack_target,
            pr_number=7,
            flow_name="dedup-real-path",
            file_config=DaydreamFileConfig(external_review_bots=["greptile-apps[bot]"]),
        )
    )

    assert rc == 0
    unadjudicated = _SHARD_COMMENT_COUNT - _EXTERNAL_PAIRS_PER_BATCH
    assert any(
        f"leaving {unadjudicated} candidate pair(s) unadjudicated" in w and "adjudicator exploded" in w
        for w in warnings
    ), warnings

    items = json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())["items"]
    assert all("disposition" not in item for item in items)
    sidecar = json.loads((multi_stack_target / ".daydream" / "deep" / "external-dedup.json").read_text())
    assert sidecar == {"suppressed": []}


async def test_dedup_external_all_shards_failed_propagates_first_batch_error(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    fake_gh: FakeGh,
    install_backend: Any,
    make_config: Any,
) -> None:
    """No shard succeeded → the adjudication accomplished nothing, so it raises.

    The lowest-numbered failed batch's exception is the one that propagates, so
    the error a run reports is deterministic.
    """
    ext_dir.write_module(_SEED_DEDUP_FLOW_EXT)
    _serve_open_pr(fake_gh, multi_stack_target)
    _serve_chatty_bot(fake_gh)

    install_backend(_PerShardBackend({
        _FIRST_SHARD_MARKER: RuntimeError("batch 0 exploded"),
        _LAST_SHARD_MARKER: RuntimeError("batch 1 exploded"),
    }))

    with pytest.raises(RuntimeError, match="batch 0 exploded"):
        await runner.run(
            make_config(
                multi_stack_target,
                pr_number=7,
                flow_name="dedup-real-path",
                file_config=DaydreamFileConfig(external_review_bots=["greptile-apps[bot]"]),
            )
        )

    items = json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())["items"]
    assert all("disposition" not in item for item in items)


async def test_dedup_external_single_batch_failure_propagates(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    fake_gh: FakeGh,
    install_backend: Any,
    make_config: Any,
) -> None:
    """One candidate pair → one batch; its failure propagates as it always has."""
    ext_dir.write_module(_SEED_DEDUP_FLOW_EXT)
    _serve_open_pr(fake_gh, multi_stack_target)
    fake_gh.set_response("graphql_threads", value=_ext_page([
        _ext_thread("api.py", 3, author="greptile-apps", body="loop overruns by one",
                    url="https://gh/c/000"),
    ]))
    install_backend(ScriptedBackend(script=[(RuntimeError("adjudicator exploded"),)]))

    with pytest.raises(RuntimeError, match="adjudicator exploded"):
        await runner.run(
            make_config(
                multi_stack_target,
                pr_number=7,
                flow_name="dedup-real-path",
                file_config=DaydreamFileConfig(external_review_bots=["greptile-apps[bot]"]),
            )
        )


async def test_dedup_external_suppression_follows_batch_order_not_completion_order(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    fake_gh: FakeGh,
    install_backend: Any,
    make_config: Any,
) -> None:
    """Both shards vote to suppress item 1; the first batch's verdict must win.

    The second shard is scripted to finish first, so a completion-ordered
    assembly would write its ``external_ref`` instead.
    """
    ext_dir.write_module(_SEED_DEDUP_FLOW_EXT)
    _serve_open_pr(fake_gh, multi_stack_target)
    _serve_chatty_bot(fake_gh)

    backend = _PerShardBackend(
        {
            _FIRST_SHARD_MARKER: _shard_verdict(_FIRST_SHARD_MARKER),
            _LAST_SHARD_MARKER: _shard_verdict(_LAST_SHARD_MARKER),
        },
        delays={_FIRST_SHARD_MARKER: 0.2},
    )
    install_backend(backend)

    rc = await runner.run(
        make_config(
            multi_stack_target,
            pr_number=7,
            flow_name="dedup-real-path",
            file_config=DaydreamFileConfig(external_review_bots=["greptile-apps[bot]"]),
        )
    )

    assert rc == 0
    assert backend.completed == [_LAST_SHARD_MARKER, _FIRST_SHARD_MARKER]
    by_id = {item["id"]: item
             for item in json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())
             ["items"]}
    assert by_id[1]["external_ref"] == _FIRST_SHARD_MARKER


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
    # Returns None (warn-and-continue): the step never Stops the flow.
    await _step_dedup_external(ctx)
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
    await _step_dedup_external(ctx)
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
    await _step_dedup_external(ctx)
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
    await _step_dedup_external(ctx)
    assert find_by_number_calls == [42]
