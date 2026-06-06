"""Tests for the ``corpus`` namespace dispatch.

The data-pipeline verbs (``harvest``, ``build``/``build-corpus``, ``label``)
live under a ``corpus`` parent verb. ``main()`` recognizes ``corpus`` and
dispatches the sub-verb to the existing handlers; a bare ``daydream corpus``
prints help and exits 2. The old top-level forms are removed — ``daydream
harvest`` is no longer a verb, so it falls through to the ``review`` shim and
is rejected as an invalid target.

These tests drive ``cli.main`` through ``sys.argv`` (the production
entrypoint), mocking only the handler/backend seam, and assert on the exit
code and on whether the handler was actually invoked — not on mere dispatch.
"""

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
    monkeypatch.setattr(
        "daydream.training.harvest.run_harvest",
        lambda c: called.setdefault("hit", True) or {"errors": 0, "annotated": 0, "skipped": 0, "total": 0},
    )
    assert _run_main(["corpus", "harvest", "--dry-run"]) == 0
    assert called["hit"]


def test_corpus_build_and_label_route(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    out = tmp_path / "x.jsonl"
    assert _run_main(["corpus", "build", "--out", str(out), "--dry-run"]) == 0

    label_called = {}

    def _fake_label(argv: list[str]) -> int:
        label_called["argv"] = argv
        return 0

    monkeypatch.setattr(cli, "_handle_label_command", _fake_label)
    assert _run_main(["corpus", "label", "sess-0001", "--outcome", "accepted"]) == 0
    assert label_called["argv"] == ["sess-0001", "--outcome", "accepted"]


def test_bare_corpus_prints_help_exits_2() -> None:
    assert _run_main(["corpus"]) == 2


def test_bare_harvest_is_unknown_verb_treated_as_review_target() -> None:
    # 'harvest' is no longer a verb; _first_verb falls through to review,
    # which then rejects the unknown '--dry-run' flag (argparse error → exit 2).
    assert _run_main(["harvest", "--dry-run"]) != 0
