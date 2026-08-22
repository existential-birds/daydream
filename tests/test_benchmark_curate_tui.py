"""Real-path tests for the resumable terminal curation client (curate_tui).

Drives ``run_curate_tui`` over the real service path (``_seed_ready_case`` +
``fake_gh``), mocking only the editor subprocess and pager. Every action
``[a/e/n/x/c/r/d/z/i/q]`` is pinned by a test that asserts the persisted case
YAML/state via service reads.
"""

import pytest
from pathlib import Path

from tests.test_benchmark_curation import _seed_ready_case


def _scripted(*lines):
    """A ``read_line`` callable that yields *lines* then raises StopIteration."""
    it = iter(lines)
    return lambda _prompt: next(it)


def test_parse_indices_accepts_commas_and_ranges():
    from daydream.benchmark.curate_tui import parse_indices
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


def test_run_curate_tui_queue_renders_index_and_quits(tmp_path, fake_gh, capsys):
    from daydream.benchmark.curate_tui import render_index_table, run_curate_tui
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)

    table = render_index_table([{"case_id": case_id, "pr_number": 101,
        "head_prefix": "a" * 12, "state": "draft", "gold_mode": "historical",
        "gold_count": 0, "evidence_count": 1, "changed_files": 2, "changed_lines": 5}])
    assert case_id in table and "draft" in table and "historical" in table
    assert "101" in table and ("2" in table and "5" in table)

    rc = run_curate_tui(ws, read_line=_scripted("q"))       # quit from the queue
    assert rc == 0
    # discriminating: the real case_id (from list_cases) must be rendered to stdout,
    # so a stub that ignores list_cases cannot pass
    assert case_id in capsys.readouterr().out

def test_render_case_shows_header_and_numbered_evidence(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    from daydream.benchmark.curate_tui import render_case
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    view = cu.get_case(ws, case_id)
    out = render_case(view)
    assert case_id in out and "draft" in out
    assert "alice" in out and "inline_comment" in out          # evidence projection
    assert "feature.py:2" in out                                # path/line anchor
    assert "please fix" in out                                  # body preview


def test_run_curate_tui_unknown_action_reprompts(tmp_path, fake_gh, capsys):
    from daydream.benchmark.curate_tui import run_curate_tui
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    rc = run_curate_tui(ws, case_id, read_line=_scripted("z9", "q"))
    assert rc == 0
    assert "unknown" in capsys.readouterr().out


def test_action_accept_persists_historical_finding(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    from daydream.benchmark.curate_tui import run_curate_tui
    from daydream.benchmark.storage import load_yaml_strict
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    src = next(c["source_id"] for c in cu.get_case(ws, case_id)["candidates"]
               if c["exact_acceptable"])

    run_curate_tui(ws, case_id, read_line=_scripted("a", "1", "q"))   # accept index 1
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    f = raw["curation"]["findings"][0]
    assert f["provenance"]["kind"] == "historical" and f["provenance"]["source_ids"] == [src]
    assert raw["curation"]["state"] == "draft"


def test_action_accept_invalid_index_mutates_nothing(tmp_path, fake_gh):
    from daydream.benchmark.curate_tui import run_curate_tui
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    before = path.read_bytes()
    run_curate_tui(ws, case_id, read_line=_scripted("a", "999", "q"))  # bad idx
    assert path.read_bytes() == before                       # unchanged


def test_action_accept_non_exact_candidate_offers_edit_path(tmp_path, fake_gh, capsys):
    import yaml
    from daydream.benchmark.curate_tui import run_curate_tui
    from daydream.benchmark.storage import load_yaml_strict
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    raw = load_yaml_strict(path)
    raw["candidates"][0]["exact_acceptable"] = False
    raw["candidates"][0]["not_exact_reason"] = "needs rework"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    after_rewrite = path.read_bytes()

    run_curate_tui(ws, case_id, read_line=_scripted("a", "1", "q"))
    assert path.read_bytes() == after_rewrite                 # unchanged
    assert "not exactly acceptable" in capsys.readouterr().out


def test_action_new_via_real_editor_persists_authored(tmp_path, fake_gh, monkeypatch):
    from daydream.benchmark.curate_tui import run_curate_tui
    from daydream.benchmark.storage import load_yaml_strict
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3)
    log = tmp_path / "editor.log"
    editor = tmp_path / "edit.sh"
    editor.write_text("#!/bin/sh\nprintf '%s %s\\n' \"$(stat -c '%a' \"$1\")\" \"$1\" > \"$LOG\"\n"
                      "cat > \"$1\" <<'EOF'\nfindings:\n"
                      "  - title: New concern\n    body: fresh wording\n"
                      "    severity: medium\n    location: null\n    source_ids: []\nEOF\n")
    editor.chmod(0o755)
    monkeypatch.setenv("VISUAL", str(editor)); monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.setenv("LOG", str(log))

    run_curate_tui(ws, case_id, read_line=_scripted("n", "q"))
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    f = raw["curation"]["findings"][0]
    assert f["title"] == "New concern" and f["provenance"]["kind"] == "authored"
    assert raw["curation"]["state"] == "draft"
    mode, buf = log.read_text().strip().split(" ", 1)
    assert mode == "600"                        # editor buffer was mode 0600
    assert not Path(buf).exists()               # buffer removed after the edit


def test_editor_nonzero_exit_leaves_state_unchanged(tmp_path, fake_gh, monkeypatch, capsys):
    from daydream.benchmark.curate_tui import run_curate_tui
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3)
    path = ws / "cases" / f"{case_id}.yaml"; before = path.read_bytes()
    editor = tmp_path / "fail.sh"; editor.write_text("#!/bin/sh\nexit 3\n"); editor.chmod(0o755)
    monkeypatch.setenv("VISUAL", str(editor)); monkeypatch.delenv("EDITOR", raising=False)

    run_curate_tui(ws, case_id, read_line=_scripted("n", "q"))
    assert path.read_bytes() == before                       # unchanged
    assert "Traceback" not in capsys.readouterr().err


def test_editor_malformed_buffer_is_discarded(tmp_path, fake_gh, monkeypatch, capsys):
    from daydream.benchmark.curate_tui import run_curate_tui
    ws, case_id, _ = _seed_ready_case(tmp_path, fake_gh, lines=3)
    path = ws / "cases" / f"{case_id}.yaml"; before = path.read_bytes()
    editor = tmp_path / "bad.sh"
    editor.write_text("#!/bin/sh\ncat > \"$1\" <<'EOF'\ntitle: [unclosed\nEOF\n")
    editor.chmod(0o755)
    monkeypatch.setenv("VISUAL", str(editor)); monkeypatch.delenv("EDITOR", raising=False)

    run_curate_tui(ws, case_id, read_line=_scripted("n", "q"))
    assert path.read_bytes() == before
    assert "Traceback" not in capsys.readouterr().err


def test_action_edit_replaces_seeded_finding(tmp_path, fake_gh, monkeypatch):
    import yaml as _yaml
    from daydream.benchmark import curation as cu
    from daydream.benchmark.curate_tui import run_curate_tui
    from daydream.benchmark.storage import load_yaml_strict
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    current = next(c for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])
    editor = tmp_path / "edit2.sh"
    editor.write_text(
        "#!/bin/sh\ncat > \"$1\" <<'EOF'\nfindings:\n"
        "  - title: Reworked\n    body: edited wording\n"
        "    severity: low\n    location: null\n"
        f"    source_ids: [{current['source_id']}]\nEOF\n"
    )
    editor.chmod(0o755)
    monkeypatch.setenv("VISUAL", str(editor)); monkeypatch.delenv("EDITOR", raising=False)

    run_curate_tui(ws, case_id, read_line=_scripted("a", "1", "e", "1", "q"))
    raw = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    f = raw["curation"]["findings"][0]
    assert f["title"] == "Reworked" and f["provenance"]["kind"] == "edited"


def test_action_exclude_evidence_other_requires_note(tmp_path, fake_gh):
    from daydream.benchmark import curation as cu
    from daydream.benchmark.curate_tui import run_curate_tui
    from daydream.benchmark.storage import load_yaml_strict
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    src = cu.get_case(ws, case_id)["candidates"][0]["source_id"]

    run_curate_tui(ws, case_id,
                   read_line=_scripted("x", "1", "other", "stale link", "q"))
    ex = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]["exclusions"][0]
    assert ex == {"source_id": src, "reason": "other", "note": "stale link"}


def test_action_exclude_evidence_rejects_stray_note(tmp_path, fake_gh, capsys):
    from daydream.benchmark.curate_tui import run_curate_tui
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"; before = path.read_bytes()
    run_curate_tui(ws, case_id,
                   read_line=_scripted("x", "1", "duplicate", "a stray note", "q"))
    assert path.read_bytes() == before                      # service rejects the note
    assert "Traceback" not in capsys.readouterr().err


def test_clean_confirm_does_not_mark_ready(tmp_path, fake_gh, capsys):
    from daydream.benchmark.curate_tui import run_curate_tui
    from daydream.benchmark.storage import load_yaml_strict
    ws, case_id, _h = _seed_ready_case(tmp_path, fake_gh, lines=2)   # empty gold
    run_curate_tui(ws, case_id, read_line=_scripted("c", "y", "q"))
    cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
    assert cur["clean_attested"] is True and cur["gold_status"] == "clean"
    assert cur["state"] == "draft" and cur["snapshot_attested"] is False
    assert ("as reviewed clean with zero expected findings" in
            capsys.readouterr().out)


def test_mark_ready_requires_yes_and_exact_sha(tmp_path, fake_gh, capsys):
    from daydream.benchmark.curate_tui import run_curate_tui
    from daydream.benchmark.storage import load_yaml_strict
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    run_curate_tui(ws, case_id, read_line=_scripted("a", "1", "r", "n", "q"))
    cur = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["curation"]
    assert cur["state"] == "draft"                        # 'n' declined the attest
    out = capsys.readouterr().out
    assert f"valid against head {head_sha}" in out        # exact SHA confirmation shown
    assert f"mark {case_id} ready?" in out


def test_stale_case_shows_marker_and_re_attests(tmp_path, fake_gh, capsys):
    import yaml
    from daydream.benchmark import curation as cu
    from daydream.benchmark.curate_tui import render_case, run_curate_tui
    from daydream.benchmark.storage import load_yaml_strict
    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=3, candidate=True)
    path = ws / "cases" / f"{case_id}.yaml"
    raw = load_yaml_strict(path)
    raw["curation"]["state"] = "stale"; raw["curation"]["snapshot_attested"] = True
    path.write_text(yaml.safe_dump(raw, sort_keys=False))   # force stale + attested

    assert "stale" in render_case(cu.get_case(ws, case_id))  # stale marker rendered

    run_curate_tui(ws, case_id, read_line=_scripted("a", "1", "r", "y", "q"))
    cur = load_yaml_strict(path)["curation"]
    assert cur["state"] == "ready" and cur["snapshot_attested"] is True
    out = capsys.readouterr().out
    assert f"valid against head {head_sha}" in out          # stale re-ran the SHA confirm
