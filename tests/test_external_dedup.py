"""Tests for external-bot finding dedup.

Covers the four layers of the feature:
  1. ``reconcile.fetch_external_findings`` — competitor-comment inventory + author filter.
  2. ``deep.dedup.build_external_dedup_candidates`` — location pre-filter.
  3. ``phases.phase_dedup_external`` — the adjudicated suppression (real path: real
     temp files + merged-items.json, mocking only the GitHub fetch and the agent seam).
  4. ``pr_review.parsed_issues_from_items`` — the disposition is honored so suppressed
     items never reach the PR or the findings artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.backends import ResultEvent
from daydream.deep.dedup import build_external_dedup_candidates
from daydream.pr_review import parsed_issues_from_items
from daydream.reconcile import ExternalComment, fetch_external_findings
from tests.harness.backend import ScriptedBackend

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


def test_fetch_external_findings_falls_back_to_original_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    page = _ext_page([_ext_thread("a.py", None, original=42, author="greptile-apps",
                                  body="x", url="u1")])
    monkeypatch.setattr(git_ops, "gh_api", lambda *a, **k: page)
    found = fetch_external_findings(tmp_path, "o/r", 7, bot_logins=["greptile-apps"])
    assert found[0].line == 42


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
