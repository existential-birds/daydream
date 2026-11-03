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

from daydream.benchmark import curate_tui as tui, curation as cu
from daydream.benchmark.curate_tui import (
    parse_indices,
    render_index_table,
    run_curate_tui,
)
from daydream.benchmark.storage import load_yaml_strict
from daydream.benchmark.workspace import validate_workspace
from tests.harness.fake_gh import FakeGh
from tests.test_benchmark_curation import _seed_ready_case, _seed_ready_case_mixed


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


def _action_new_via_real_editor_persists_authored(tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
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

def _editor_failure_leaves_state_unchanged(
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

def _action_edit_replaces_seeded_finding(tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch) -> None:
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

def _action_edit_splits_one_evidence_into_two_findings(
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


def _action_exclude_evidence_other_requires_note(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    src = cu.get_case(ws, case_id)["candidates"][0]["source_id"]

    run_curate_tui(ws, case_id, read_line=_scripted("x", "1", "other", "stale link", "q"))
    ex = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["exclusions"][0]
    assert ex == {"source_id": src, "reason": "other", "note": "stale link"}

def _action_exclude_evidence_rejects_stray_note(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    run_curate_tui(ws, case_id, read_line=_scripted("x", "1", "duplicate", "a stray note", "q"))
    assert path.read_bytes() == before                      # service rejects the note
    assert "Traceback" not in capsys.readouterr().err

def _action_exclude_range_invalid_mutates_nothing(
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


def _case_exclude_and_reinclude(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3)
    path = ws / "cases" / f"{case_id}.yaml"

    run_curate_tui(ws, case_id, read_line=_scripted("z", "not_suitable", "q"))
    cur = load_yaml_strict(path)["curation"]
    assert cur["state"] == "excluded"
    assert cur["case_exclusion"] == {"reason": "not_suitable", "note": None}

    run_curate_tui(ws, case_id, read_line=_scripted("i", "q"))     # resume re-include
    cur = load_yaml_strict(path)["curation"]
    assert cur["state"] == "draft" and cur["case_exclusion"] is None

def _case_exclude_other_requires_note(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str]) -> None:
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    run_curate_tui(ws, case_id, read_line=_scripted("z", "other", "", "q"))  # empty note
    assert path.read_bytes() == before
    out = capsys.readouterr()
    assert "case exclusion reason 'other' requires a note" in out.out
    assert "Traceback" not in out.err

def _corrupt_workspace_returns_1_no_traceback(tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
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


def _inexact_accept_interrupt_batch_exclude_then_defer(
    tmp_path: Path, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
) -> None:
    """An inexact accept is refused, ^C keeps the loop, a range shares one reason, and defer ends."""
    ws, case_id, _h = _seed_ready_case_mixed(tmp_path, fake_gh)
    view = cu.get_case(ws, case_id)
    binding = tui._view_binding(view)
    inexact = next(c["source_id"] for c in view["candidates"] if not c["exact_acceptable"])
    actions = iter(["a", str(binding.index(inexact) + 1), None, "x", "1-2", "duplicate", "d"])

    def read_line(_prompt: str) -> str:
        answer = next(actions)
        if answer is None:
            raise KeyboardInterrupt
        return answer

    capsys.readouterr()
    assert run_curate_tui(ws, case_id, read_line=read_line) == 0
    out = capsys.readouterr().out
    assert f"{inexact} is not exactly acceptable" in out and "interrupted" in out
    assert "excluded 2 evidence source(s)" in out and f"case {case_id} deferred" in out
    exclusions = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["exclusions"]
    assert [row["source_id"] for row in exclusions] == binding[:2]
    assert {row["note"] for row in exclusions} == {None}


def test_editor_and_exclusion_actions_persist_through_the_service(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    for name, check in (("new", _action_new_via_real_editor_persists_authored),
                        ("edit", _action_edit_replaces_seeded_finding),
                        ("split", _action_edit_splits_one_evidence_into_two_findings)):
        (tmp_path / name).mkdir()
        check(tmp_path / name, fake_gh, monkeypatch)
    (tmp_path / "batch").mkdir()
    _inexact_accept_interrupt_batch_exclude_then_defer(tmp_path / "batch", fake_gh, capsys)
    for name, plain in (("exclude", _action_exclude_evidence_other_requires_note),
                        ("case", _case_exclude_and_reinclude)):
        (tmp_path / name).mkdir()
        plain(tmp_path / name, fake_gh)


def test_invalid_actions_and_editor_failures_mutate_nothing(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Every rejected input keeps the case bytes and never surfaces a traceback."""
    for name, script in (("nonzero-exit", "#!/bin/sh\nexit 3\n"),
                         ("malformed-buffer", "#!/bin/sh\ncat > \"$1\" <<'EOF'\ntitle: [unclosed\nEOF\n")):
        (tmp_path / name).mkdir()
        _editor_failure_leaves_state_unchanged(tmp_path / name, fake_gh, monkeypatch, capsys, script)
    for name, check in (("stray", _action_exclude_evidence_rejects_stray_note),
                        ("range", _action_exclude_range_invalid_mutates_nothing),
                        ("note", _case_exclude_other_requires_note),
                        ("corrupt", _corrupt_workspace_returns_1_no_traceback)):
        (tmp_path / name).mkdir()
        check(tmp_path / name, fake_gh, capsys)
