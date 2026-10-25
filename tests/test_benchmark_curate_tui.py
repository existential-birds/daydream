"""Real-path tests for the resumable terminal curation client (curate_tui).

Drives ``run_curate_tui`` over the real service path (``_seed_ready_case`` +
``fake_gh``), mocking only the editor subprocess and pager. Every action
``[a/e/n/x/c/r/d/z/i/q]`` is pinned by a test that asserts the persisted case
YAML/state via service reads.
"""
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from daydream.benchmark import curate_tui as tui, curation as cu
from daydream.benchmark.curate_tui import (
    parse_indices,
    render_case,
    render_index_table,
    run_curate_tui,
)
from daydream.benchmark.storage import atomic_write_json, load_json_strict, load_yaml_strict
from daydream.benchmark.workspace import validate_workspace
from tests.harness.fake_gh import FakeGh
from tests.test_benchmark_curation import _reanchor_frozen_inline, _seed_ready_case, _seed_ready_case_mixed


def _scripted(*lines: Any) -> Any:
    """A ``read_line`` callable that yields *lines* then raises StopIteration."""
    it = iter(lines)
    return lambda _prompt: next(it)


def _install_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    editor = tmp_path / "editor"
    editor.write_text(script)
    editor.chmod(0o755)
    monkeypatch.setenv("VISUAL", str(editor))
    monkeypatch.delenv("EDITOR", raising=False)


def test_parse_indices_accepts_commas_and_ranges() -> None:
    assert parse_indices("1,3-5", 5) == [0, 2, 3, 4]   # 1-based in, 0-based out
    assert parse_indices("2", 5) == [1]
    assert parse_indices("5-1", 5) == [0, 1, 2, 3, 4]   # reversed range normalizes
    for bad in ("0", "6", "1,1", "1-1", "abc", "1,,2", "1-2,3-4", ""):
        try:
            parse_indices(bad, 5)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} must raise ValueError")


def test_render_case_pages_all_evidence_kinds(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _ = _seed_ready_case_mixed(tmp_path, fake_gh)
    view = cu.get_case(ws, case_id)
    assert {row["source_id"]: (row["kind"], row["candidate_index"]) for row in view["evidence"]} == {
        "github:inline_comment:1": ("inline_comment", 0), "github:inline_comment:2": ("inline_comment", None),
        "github:review:100": ("review", None), "github:review:101": ("review", 1),
        "github:issue_comment:200": ("issue_comment", None),
    }
    assert next(row for row in view["evidence"] if row["source_id"] == "github:review:100")["state"] == "APPROVED"
    assert cu.list_cases(ws)[0]["evidence_count"] == 5
    out = render_case(view)
    assert "APPROVED" in out and "carol" in out          # pure approval review paged
    assert "issue_comment" in out and "dave" in out      # conversation comment paged
    assert "reply text" in out and "bob" in out          # inline reply paged
    assert "inline_comment" in out and "please fix" in out  # root candidate still visible


def test_render_case_shows_authoring_commit_and_fixed_reason(tmp_path: Path, fake_gh: FakeGh) -> None:
    """The evidence row surfaces the strict authoring commit (short form) and the
    fixed not-exact reason whenever exact acceptance is unavailable, next to the
    re-anchored commit_id -- the two commits are never conflated."""

    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    _reanchor_frozen_inline(ws, case_id, authoring_commit="b" * 40)
    view = cu.get_case(ws, case_id)
    projection = next(c for c in view["candidates"] if not c["exact_acceptable"])["evidence"]
    assert (projection["authoring_commit_id"], projection["commit_id"], projection["not_exact_reason"]) == (
        "b" * 40, head_sha, "re-anchored",
    )
    out = render_case(view)
    assert f"auth:{'b' * 12}" in out       # authoring commit short form, labeled
    assert "[re-anchored]" in out          # fixed reason shown for the non-exact candidate
    assert "feature.py:2" in out           # anchor display unchanged


def test_action_exclude_evidence_persists_supported_reason(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    source_id = cu.get_case(ws, case_id)["evidence"][0]["source_id"]

    run_curate_tui(ws, case_id, read_line=_scripted("x", "1", "fixed_before_snapshot", "", "q"))

    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["curation"]["exclusions"] == [{"source_id": source_id, "reason": "fixed_before_snapshot", "note": None}]
    assert raw["curation"]["state"] == "draft"
    out = capsys.readouterr().out
    assert "excluded 1 evidence source(s)" in out
    assert "fixed_before_snapshot" in out


def test_action_accept_non_exact_candidate_offers_edit_path(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    raw = load_yaml_strict(path)
    raw["candidates"][0]["exact_acceptable"] = False
    raw["candidates"][0]["not_exact_reason"] = "needs rework"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    after_rewrite = path.read_bytes()

    run_curate_tui(ws, case_id, read_line=_scripted("a", "1", "q"))
    assert path.read_bytes() == after_rewrite
    assert "not exactly acceptable" in capsys.readouterr().out

def test_action_new_via_real_editor_persists_authored(tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3)
    log = tmp_path / "editor.log"
    _install_editor(tmp_path, monkeypatch,
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import stat\n"
        "import sys\n"
        "buf = sys.argv[1]\n"
        "with open(os.environ['LOG'], 'w') as fh:\n"
        "    fh.write(f'{stat.S_IMODE(os.stat(buf).st_mode):o} {buf}\\n')\n"
        "with open(buf, 'w') as fh:\n"
        "    fh.write('findings:\\n  - title: New concern\\n    body: fresh wording\\n"
        "    severity: medium\\n    location: null\\n    source_ids: []\\n')\n"
    )
    monkeypatch.setenv("LOG", str(log))

    run_curate_tui(ws, case_id, read_line=_scripted("n", "q"))
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    f = raw["curation"]["findings"][0]
    assert f["title"] == "New concern" and f["provenance"]["kind"] == "authored"
    assert raw["curation"]["state"] == "draft"
    mode, buf = log.read_text().strip().split(" ", 1)
    assert mode == "600"                        # editor buffer was mode 0600
    assert not Path(buf).exists()               # buffer removed after the edit

@pytest.mark.parametrize("editor_script",
    [
        pytest.param("#!/bin/sh\nexit 3\n", id="nonzero-exit"),
        pytest.param(
            "#!/bin/sh\ncat > \"$1\" <<'EOF'\ntitle: [unclosed\nEOF\n",
            id="malformed-buffer",
        ),
    ],
)
def test_editor_failure_leaves_state_unchanged(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    editor_script: str,
) -> None:
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    _install_editor(tmp_path, monkeypatch, editor_script)

    run_curate_tui(ws, case_id, read_line=_scripted("n", "q"))
    assert path.read_bytes() == before
    assert "Traceback" not in capsys.readouterr().err

def test_action_edit_replaces_seeded_finding(tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    current = next(c for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])
    _install_editor(tmp_path, monkeypatch,
        "#!/bin/sh\ncat > \"$1\" <<'EOF'\nfindings:\n"
        "  - title: Reworked\n    body: edited wording\n"
        "    severity: low\n    location: null\n"
        f"    source_ids: [{current['source_id']}]\nEOF\n"
    )

    run_curate_tui(ws, case_id, read_line=_scripted("a", "1", "e", "1", "q"))
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    f = raw["curation"]["findings"][0]
    assert f["title"] == "Reworked" and f["provenance"]["kind"] == "edited"

def test_action_edit_authors_edited_finding_from_non_candidate_evidence(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, case_id, _h = _seed_ready_case_mixed(tmp_path, fake_gh)
    _install_editor(tmp_path, monkeypatch,
        "#!/bin/sh\ncat > \"$1\" <<'EOF'\nfindings:\n  - title: From approval\n"
        "    body: edited wording\n    severity: null\n    location: null\n"
        "    source_ids: [github:review:100]\nEOF\n"
    )
    run_curate_tui(ws, case_id, read_line=_scripted("e", "a", "4", "q"))
    f = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["findings"][0]
    assert f["title"] == "From approval" and f["provenance"]["kind"] == "edited"
    assert f["provenance"]["source_ids"] == ["github:review:100"]

def test_edit_author_prefills_selected_evidence_source_ids(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The [e]->a author selector must pin the selected evidence's source_ids
    into the editor buffer before it opens, so a wrong/empty/off-by-one prefill
    cannot slip past the callers that rewrite source_ids in their heredocs."""
    ws, case_id, _h = _seed_ready_case_mixed(tmp_path, fake_gh)
    log = tmp_path / "prefill.log"
    _install_editor(tmp_path, monkeypatch,
        "#!/bin/sh\n"
        "cat > \"$LOG\" < \"$1\"\n"
        "cat > \"$1\" <<'EOF'\nfindings:\n  - title: From selected\n    body: pinned\n"
        "    severity: null\n    location: null\n    source_ids: [github:review:100]\nEOF\n"
    )
    monkeypatch.setenv("LOG", str(log))

    run_curate_tui(ws, case_id, read_line=_scripted("e", "a", "4", "q"))
    f = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["findings"][0]
    assert f["title"] == "From selected" and f["provenance"]["kind"] == "edited"
    prefill = log.read_text()
    assert "source_ids" in prefill and "- github:review:100" in prefill
    assert "github:review:100" in prefill and "github:issue_comment:200" not in prefill

def test_action_edit_splits_one_evidence_into_two_findings(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, case_id, _h = _seed_ready_case_mixed(tmp_path, fake_gh)
    _install_editor(tmp_path, monkeypatch,
        "#!/bin/sh\ncat > \"$1\" <<'EOF'\nfindings:\n  - title: Part A\n    body: a\n"
        "    severity: high\n    location: {path: feature.py, start_line: 1, end_line: 1}\n"
        "    source_ids: [github:inline_comment:1]\n"
        "  - title: Part B\n    body: b\n    severity: null\n    location: null\n"
        "    source_ids: [github:inline_comment:1]\nEOF\n"
    )
    run_curate_tui(ws, case_id, read_line=_scripted("e", "a", "1", "q"))
    fs = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["findings"]
    assert len(fs) == 2 and {f["provenance"]["kind"] for f in fs} == {"edited"}
    assert [(f["title"], f["severity"], f["location"]) for f in fs] == [
        ("Part A", "high", {"path": "feature.py", "start_line": 1, "end_line": 1}), ("Part B", None, None),
    ]


def test_action_edit_merges_range_into_one_finding(tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, case_id, _f = _seed_ready_case_mixed(tmp_path, fake_gh)
    _install_editor(tmp_path, monkeypatch,
        "#!/bin/sh\ncat > \"$1\" <<'EOF'\nfindings:\n  - title: Merged\n    body: combined\n"
        "    severity: medium\n    location: {path: feature.py, start_line: 2, end_line: 2}\n"
        "    source_ids: [github:inline_comment:1, github:review:100, github:issue_comment:200]\nEOF\n"
    )
    run_curate_tui(ws, case_id, read_line=_scripted("e", "a", "1,4,5", "q"))
    f = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["findings"][0]
    assert f["provenance"]["source_ids"] == ["github:inline_comment:1", "github:review:100", "github:issue_comment:200"]
    assert (f["title"], f["body"], f["severity"], f["location"], f["provenance"]["kind"]) == (
        "Merged", "combined", "medium", {"path": "feature.py", "start_line": 2, "end_line": 2}, "edited",
    )


def test_action_exclude_evidence_other_requires_note(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    src = cu.get_case(ws, case_id)["candidates"][0]["source_id"]

    run_curate_tui(ws, case_id, read_line=_scripted("x", "1", "other", "stale link", "q"))
    ex = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["exclusions"][0]
    assert ex == {"source_id": src, "reason": "other", "note": "stale link"}

def test_action_exclude_evidence_rejects_stray_note(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    run_curate_tui(ws, case_id, read_line=_scripted("x", "1", "duplicate", "a stray note", "q"))
    assert path.read_bytes() == before                      # service rejects the note
    assert "Traceback" not in capsys.readouterr().err

def test_action_exclude_range_excludes_all_selected(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _h = _seed_ready_case_mixed(tmp_path, fake_gh)
    run_curate_tui(ws, case_id, read_line=_scripted("x", "1,4", "duplicate", "q"))
    ex = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["exclusions"]
    assert {e["source_id"] for e in ex} == {"github:inline_comment:1", "github:review:100"}
    assert all(e["reason"] == "duplicate" for e in ex)

def test_action_exclude_range_invalid_mutates_nothing(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, _h = _seed_ready_case_mixed(tmp_path, fake_gh)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    run_curate_tui(ws, case_id, read_line=_scripted("x", "2,2", "q"))   # repeated index
    assert path.read_bytes() == before
    assert "Traceback" not in capsys.readouterr().err


def test_no_comment_clean_then_ready_marks_case_ready(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=2)   # no-comment, empty gold
    def inputs() -> Iterator[str]:
        yield "c"
        yield "y"
        cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
        assert cur["clean_attested"] is True and cur["gold_status"] == "clean"
        assert cur["state"] == "draft" and cur["snapshot_attested"] is False
        yield "r"
        yield "y"
        yield "q"

    reads = inputs()
    run_curate_tui(ws, case_id, read_line=lambda _prompt: next(reads))
    cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
    assert cur["state"] == "ready" and cur["snapshot_attested"] is True
    assert cur["clean_attested"] is True and cur["gold_status"] == cur["gold_mode"] == "clean"
    assert validate_workspace(ws) == (0, "ready")
    out = capsys.readouterr().out
    assert "as reviewed clean with zero expected findings" in out
    assert "attested" in out and f"mark {case_id} ready?" in out


def test_stale_case_shows_marker_and_re_attests(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    raw = load_yaml_strict(path)
    raw["curation"]["state"] = "stale"
    raw["curation"]["snapshot_attested"] = True
    path.write_text(yaml.safe_dump(raw, sort_keys=False))   # force stale + attested

    assert "stale" in render_case(cu.get_case(ws, case_id))  # stale marker rendered

    run_curate_tui(ws, case_id, read_line=_scripted("a", "1", "r", "y", "q"))
    cur = load_yaml_strict(path)["curation"]
    assert cur["state"] == "ready" and cur["snapshot_attested"] is True
    out = capsys.readouterr().out
    assert f"valid against head {head_sha}" in out          # stale re-ran the SHA confirm

def test_case_exclude_and_reinclude(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3)
    path = ws / "cases" / f"{case_id}.yaml"

    run_curate_tui(ws, case_id, read_line=_scripted("z", "not_suitable", "q"))
    cur = load_yaml_strict(path)["curation"]
    assert cur["state"] == "excluded"
    assert cur["case_exclusion"] == {"reason": "not_suitable", "note": None}

    run_curate_tui(ws, case_id, read_line=_scripted("i", "q"))     # resume re-include
    cur = load_yaml_strict(path)["curation"]
    assert cur["state"] == "draft" and cur["case_exclusion"] is None

def test_case_exclude_other_requires_note(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    run_curate_tui(ws, case_id, read_line=_scripted("z", "other", "", "q"))  # empty note
    assert path.read_bytes() == before
    out = capsys.readouterr()
    assert "case exclusion reason 'other' requires a note" in out.out
    assert "Traceback" not in out.err

def test_defer_is_ui_local_no_mutation(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    rc = run_curate_tui(ws, case_id, read_line=_scripted("d"))   # defer in a case
    assert rc == 0
    assert path.read_bytes() == before                       # nothing persisted
    assert "deferred" in capsys.readouterr().out


def test_ctrl_c_preserves_prior_actions_and_cleans_temp(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    src = next(c["source_id"] for c in cu.get_case(ws, case_id)["candidates"])

    reads = iter(["a", "1", None, "q"])   # a,1 = accept; None=(Ctrl-C); q quits after
    def interrupted(_prompt: Any) -> Any:
        nxt = next(reads)
        if nxt is None:
            raise KeyboardInterrupt
        return nxt
    run_curate_tui(ws, case_id, read_line=interrupted)

    raw = load_yaml_strict(path)
    assert len(raw["curation"]["findings"]) == 1          # prior action persisted
    assert raw["curation"]["findings"][0]["provenance"]["source_ids"] == [src]
    out = capsys.readouterr()
    assert "interrupted" in out.out
    assert "Traceback" not in out.err

def test_corrupt_workspace_returns_1_no_traceback(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)
    (ws / "cases" / f"{case_id}.yaml").unlink()          # absent case file
    rc = run_curate_tui(ws, read_line=_scripted("q"))
    assert rc == 1
    assert "Traceback" not in capsys.readouterr().err


def test_queue_accept_decline_reopen_approve_persists_rendered_state(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A curator may reject an approval, quit, and approve the same retained finding later."""
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    paged: list[str] = []
    monkeypatch.setattr(tui, "_launch_pager", paged.append)

    def inputs() -> Iterator[str]:
        yield "bogus-id"
        yield "9"
        yield "a"  # refresh the queue before selecting the real row
        yield "1"
        yield "z9"
        yield "1"  # page the captured evidence
        yield "a"
        yield "999"
        assert path.read_bytes() == before
        yield "a"
        yield "1"
        cur = load_yaml_strict(path)["curation"]
        assert cur["state"] == "draft" and cur["snapshot_attested"] is False
        assert len(cur["findings"]) == 1
        assert cur["findings"][0]["provenance"] == {
            "kind": "historical", "source_ids": ["github:inline_comment:1"],
        }
        yield "r"
        yield "n"
        cur = load_yaml_strict(path)["curation"]
        assert cur["state"] == "draft" and cur["snapshot_attested"] is False
        assert "task_spec_sha256" not in cur and "task_spec_approved_at" not in cur
        yield "q"

    reads = inputs()
    assert run_curate_tui(ws, read_line=lambda _prompt: next(reads)) == 0
    assert paged == ["please fix"]
    declined_bytes = path.read_bytes()
    assert run_curate_tui(ws, case_id, read_line=_scripted("r", "y", "q")) == 0
    cur = load_yaml_strict(path)["curation"]
    assert path.read_bytes() != declined_bytes
    assert cur["state"] == "ready" and cur["snapshot_attested"] is True
    assert cur["task_spec_sha256"] and cur["task_spec_approved_at"]
    assert len(cur["findings"]) == 1
    assert cur["findings"][0]["provenance"]["source_ids"] == ["github:inline_comment:1"]
    approved_bytes = path.read_bytes()
    assert run_curate_tui(ws, read_line=_scripted("q")) == 0
    assert path.read_bytes() == approved_bytes
    table = render_index_table(cu.list_cases(ws))
    assert f"{case_id} | 101 | {head_sha[:12]} | 2 | 5 | 1 | ready | historical | 1" in table
    out = capsys.readouterr().out
    assert "no case at row 9" in out and "unknown case bogus-id" in out and "unknown action: z9" in out
    assert case_id in out and "alice" in out and "inline_comment" in out and "feature.py:2" in out
    assert "please fix" in out and "draft" in out and "historical" in out
    assert "Task Spec" in out and f"valid against head {head_sha}" in out
    assert f"mark {case_id} ready?" in out
    assert "approve this Task Spec and attest" in out.lower() or "Approve this Task Spec" in out


def _add_late_finding(cu_mod: Any, ws: Path, case_id: str) -> None:
    """A real post-render curator action: one authored finding, no sources."""
    cu_mod.add_finding(ws, case_id, title="late concern", body="added after render",
                       severity="low", location=None, source_ids=[])


def test_render_case_shows_prioritized_sections_and_legend(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _h = _seed_ready_case_mixed(tmp_path, fake_gh)
    view = cu.get_case(ws, case_id)
    out = tui.render_case(view)
    assert "-- review_first --" in out
    assert "-- context --" in out
    assert "-- decided --" not in out
    assert "-- withdrawn --" not in out
    assert "-- likely_actioned --" not in out
    assert out.index("-- review_first --") < out.index("-- context --")
    assert "resolved" in out or "reasons:" in out
    assert "legend:" in out.lower()
    binding = tui._view_binding(view)
    for n, sid in enumerate(binding, start=1):
        assert f"  {n}. " in out and sid in out

def test_number_action_resolves_through_captured_binding_not_fresh_order(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, _h = _seed_ready_case_mixed(tmp_path, fake_gh)
    view = cu.get_case(ws, case_id)
    binding = tui._view_binding(view)                     # captured entry->source_id map
    raw_order = [ev["source_id"] for ev in view["evidence"]]
    by_sid = {ev["source_id"]: ev for ev in view["evidence"]}
    # the captured binding reorders the raw evidence (band ranking), so a number
    # action re-derived from fresh order would page a different record; pick the
    # first number where the two orders diverge
    diverging = next((n for n, sid in enumerate(binding, start=1) if raw_order[n - 1] != sid), None)
    assert diverging is not None
    bound_sid = binding[diverging - 1]
    paged: dict[str, str] = {}
    monkeypatch.setattr("daydream.benchmark.curate_tui._launch_pager", lambda body: paged.update(body=body))
    tui.run_curate_tui(ws, case_id, read_line=_scripted(str(diverging), "99", "q"))
    assert paged["body"] == by_sid[bound_sid]["body"]
    assert paged["body"] != by_sid[raw_order[diverging - 1]]["body"]
    assert "no evidence number 99" in capsys.readouterr().out

def test_stale_binding_prompts_rerender_instead_of_reinterpreting(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, _h = _seed_ready_case_mixed(tmp_path, fake_gh)
    path = ws / "cases" / f"{case_id}.yaml"
    view = cu.get_case(ws, case_id)
    binding = tui._view_binding(view)
    _add_late_finding(cu, ws, case_id)
    assert tui._binding_stale(ws, case_id, binding) is True
    assert tui._binding_stale(ws, case_id, tui._view_binding(cu.get_case(ws, case_id))) is False
    assert len(load_yaml_strict(path)["curation"]["findings"]) == 1
    ws2, case_id2, _h2 = _seed_ready_case_mixed(tmp_path, fake_gh)
    path2 = ws2 / "cases" / f"{case_id2}.yaml"
    paged: list[str] = []
    monkeypatch.setattr("daydream.benchmark.curate_tui._launch_pager", paged.append)
    late = [0]

    def mutate_then_answer(_prompt: str) -> str:
        # every number keystroke adds a *distinct* case mutation, so no
        # duplicate finding_id can abort the session and mask the rerender bound
        if late[0] >= 2:
            return "q"
        late[0] += 1
        cu.add_finding(ws2, case_id2, title=f"late concern {late[0]}",
                       body=f"added after render #{late[0]}", severity="low",
                       location=None, source_ids=[])
        return "1"

    rc = run_curate_tui(ws2, case_id2, read_line=mutate_then_answer)
    out = capsys.readouterr().out
    assert rc == 0
    assert "view changed" in out
    # the number action executes against the freshest displayed binding even
    # while the case keeps mutating — no unbounded re-prompt spin
    assert len(paged) == 2
    cur = load_yaml_strict(path2)["curation"]
    assert len(cur["findings"]) == 2 and not cur.get("exclusions")

def test_accept_non_candidate_and_context_is_rejected_without_write(tmp_path: Path, fake_gh: FakeGh,) -> None:
    ws, case_id, _h = _seed_ready_case_mixed(tmp_path, fake_gh)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    view = cu.get_case(ws, case_id)
    ctx_sid = next(e["source_id"] for e in view["prioritized_evidence"]["entries"] if e["band"] == "context")
    with pytest.raises(cu.CurationError):
        cu.accept_candidate(ws, case_id, ctx_sid)
    assert path.read_bytes() == before

def test_low_priority_exact_candidate_still_acceptable(tmp_path: Path, fake_gh: FakeGh) -> None:
    """Prioritization is advisory: a candidate whose facts/signals sink it to
    possibly_actioned is still acceptable through the unchanged service path."""
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    cand_sid = raw["candidates"][0]["source_id"]
    import_path = ws / raw["source"]["import_file"]
    imp = load_json_strict(import_path)
    for rec in imp["evidence"]:
        if rec["source_id"] == cand_sid:
            rec["resolved"] = True
    atomic_write_json(import_path, imp)

    view = cu.get_case(ws, case_id)
    assert view["prioritized_evidence"]["by_source"][cand_sid]["band"] == "possibly_actioned"
    assert view["prioritized_evidence"]["by_source"][cand_sid]["reasons"] == ["resolved"]

    cu.accept_candidate(ws, case_id, cand_sid)   # must not raise
    cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
    assert cur["findings"][0]["provenance"]["kind"] == "historical"
    assert cur["findings"][0]["provenance"]["source_ids"] == [cand_sid]
