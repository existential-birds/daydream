"""Tests for the ``corpus`` namespace dispatch.

The data-pipeline verbs (``harvest``, ``build``/``build-corpus``, ``label``,
``hydrate-hub``)
live under a ``corpus`` parent verb. ``main()`` recognizes ``corpus`` and
dispatches the sub-verb to the existing handlers; a bare ``daydream corpus``
prints help and exits 2. The old top-level forms are removed — ``daydream
harvest`` is no longer a verb, so it falls through to the ``review`` shim and
is rejected as an invalid target.

These tests drive ``cli.main`` through ``sys.argv`` (the production
entrypoint), mocking only the handler/backend seam, and assert on the exit
code and on whether the handler was actually invoked — not on mere dispatch.
"""
import json
from pathlib import Path
from typing import Any

import pytest

from daydream import cli


def _run_main(argv: list[str]) -> int:
    """Drive ``cli.main`` with ``argv`` and return its exit code."""
    import sys

    saved = sys.argv
    sys.argv = ["daydream", *argv]
    try:
        cli.main()
    except SystemExit as exc:  # main() always exits via sys.exit
        return int(exc.code or 0)
    finally:
        sys.argv = saved
    return 0


def test_corpus_harvest_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def _fake_run_harvest(_config: Any) -> dict[str, Any]:
        called["hit"] = True
        return {"errors": 0, "annotated": 0, "skipped": 0, "total": 0}

    monkeypatch.setattr("daydream.training.harvest.run_harvest", _fake_run_harvest)
    assert _run_main(["corpus", "harvest", "--dry-run"]) == 0
    assert called["hit"]


def test_corpus_build_and_label_route(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "x.jsonl"
    assert _run_main(["corpus", "build", "--out", str(out), "--dry-run"]) == 0

    label_called = {}

    def _fake_label(argv: list[str]) -> int:
        label_called["argv"] = argv
        return 0

    # Patch the dict entry directly: _handle_corpus_command dispatches from
    # _CORPUS_SUBVERBS, which captured the original function reference at import
    # time. Patching cli._handle_label_command replaces the module attribute but
    # leaves the dict value unchanged, so the real handler still runs.
    monkeypatch.setitem(cli._CORPUS_SUBVERBS, "label", _fake_label)
    assert _run_main(["corpus", "label", "sess-0001", "--outcome", "accepted"]) == 0
    assert label_called["argv"] == ["sess-0001", "--outcome", "accepted"]


def test_bare_corpus_prints_help_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run_main(["corpus"]) == 2
    captured = capsys.readouterr()
    # CI terminals wrap help output at 80 cols, so assert per token, not the
    # full usage line.
    assert "calibrate-reward" in captured.out
    assert "build-v2" in captured.out
    assert "hydrate-hub" in captured.out
    assert "harvest" in captured.out
    assert "build" in captured.out
    assert "build-v2" in captured.out
    assert "label" in captured.out
    assert "calibrate-reward" in captured.out


# ---------------------------------------------------------------------------
# Task 8 (#1080): build-v2 operator inputs — pinned license policy, exact-slug
# copyleft opt-ins, and refusal of URL-shaped identities. Real-path: the
# handler runs the real projector over a real fixture bundle pair; only the
# policy file is authored by the test.
# ---------------------------------------------------------------------------


def _run_build_v2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra_args: list[str]
) -> tuple[int, str]:
    """Drive ``daydream corpus build-v2`` through ``cli.main`` over the
    standard fixture bundle pair (from tests.test_corpus_v2) and return
    (exit code, captured stdout+stderr)."""
    from tests.test_corpus_v2 import _write_annotations_snapshot, _write_bundle

    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir, dispositions=["accepted"])
    out_dir = tmp_path / "corpus-out"
    rc = _run_main([
        "corpus", "build-v2",
        "--bundle-root", str(bundle_dir),
        "--annotation-bundle-root", str(snap.parent),
        "--out", str(out_dir / "corpus-v2.jsonl"),
        *extra_args,
    ])
    captured = capsys.readouterr()
    return rc, captured.out + captured.err


def test_build_v2_accepts_license_policy_and_repeatable_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = tmp_path / "license-policy.json"
    policy.write_text(json.dumps({"policy_version": "1", "spdx_decisions": {"MIT": "accepted"}}))
    rc, _out = _run_build_v2(tmp_path, capsys, [
        "--license-policy", str(policy),
        "--allow-copyleft", "a/b", "--allow-copyleft", "c/d",
    ])
    assert rc == 0
    lineage = json.loads((tmp_path / "corpus-out" / "lineage.json").read_text())
    assert lineage["license_policy"]["policy_version"] == "1"
    assert lineage["copyleft_opt_ins"] == ["a/b", "c/d"]


def test_build_v2_requires_license_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out = _run_build_v2(tmp_path, capsys, [])
    assert rc == 1
    assert "license-policy" in out


def test_build_v2_refuses_unknown_policy_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Refused before any build work: no output directory is created.
    policy_path = tmp_path / "bad-policy.json"
    policy_path.write_text(json.dumps({"policy_version": "", "spdx_decisions": {}}))
    rc, out = _run_build_v2(tmp_path, capsys, ["--license-policy", str(policy_path)])
    assert rc == 1
    assert "policy_version" in out
    assert not (tmp_path / "corpus-out").exists()


def test_build_v2_refuses_raw_authenticated_url_as_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A raw remote URL is never a repo identity: any URL-shaped --repo-slug
    # value is refused before it can reach BuildCorpusV2Config.
    rc, out = _run_build_v2(tmp_path, capsys, [
        "--repo-slug", "https://user:token@github.com/owner/repo",
    ])
    assert rc == 1
    assert "Unsupported --repo-slug" in out


def test_bare_harvest_is_unknown_verb_treated_as_review_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 'harvest' is no longer a verb; _first_verb falls through to review,
    # which then rejects the unknown '--dry-run' flag (argparse error → exit 2).
    assert _run_main(["harvest", "--dry-run"]) == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err
    assert "--dry-run" in captured.err
