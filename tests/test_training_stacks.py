"""Tests for the Stage-0 corpus-side dataset loader (M17, M22)."""

import json
from pathlib import Path

import pytest

from daydream.training import stacks
from daydream.training.exclusion import load_exclusion_list


def _rec(repo: str, label: str = "accepted", pr_number: int | None = 1) -> dict[str, object]:
    """One corpus record dict shaped like run_build_corpus's schema/v1 output."""
    return {
        "repo_slug": repo,
        "pr_number": pr_number,
        "label": label,
        "skill": "review",
    }


def test_load_corpus_fails_closed_on_excluded_repo(tmp_path: Path) -> None:
    assert "getsentry/sentry" in {s.casefold() for s in load_exclusion_list()}
    p = tmp_path / "corpus.jsonl"
    p.write_text(json.dumps(_rec(repo="getsentry/sentry")), encoding="utf-8")
    with pytest.raises(ValueError, match="excluded"):
        stacks.load_dataset(p)


def test_load_corpus_fails_closed_on_unopted_copyleft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The on-disk copyleft list may be empty; monkeypatch the module's loader
    # (the single source of lists) to simulate a populated C8 list.
    monkeypatch.setattr(stacks, "load_copyleft_list", lambda: frozenset({"example/copyleft-proj"}))
    p = tmp_path / "corpus.jsonl"
    p.write_text(json.dumps(_rec(repo="example/copyleft-proj", label="rejected")), encoding="utf-8")
    with pytest.raises(ValueError, match="copyleft"):
        stacks.load_dataset(p)  # allow_list empty by default


def test_load_corpus_casefolded_exclusion_match(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    p.write_text(json.dumps(_rec(repo="GetSentry/Sentry")), encoding="utf-8")
    with pytest.raises(ValueError, match="excluded"):
        stacks.load_dataset(p)


def test_load_corpus_passes_prless_records(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    p.write_text(json.dumps(_rec(repo="acme/widget", pr_number=None)), encoding="utf-8")
    out = stacks.load_dataset(p)
    assert len(out) == 1
    assert out[0]["pr_number"] is None


def test_load_corpus_returns_all_clean_records(tmp_path: Path) -> None:
    recs = [_rec(repo="acme/one"), _rec(repo="acme/two", label="rejected")]
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    loaded = stacks.load_dataset(p)
    # Records carry the M23 legacy_policy tag on admission; everything else
    # round-trips exactly.
    expected = [{**r, "legacy_policy": True} for r in recs]
    assert loaded == expected


def test_load_corpus_names_all_offenders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stacks, "load_copyleft_list", lambda: frozenset({"example/copyleft-proj"}))
    recs = [_rec(repo="acme/ok"), _rec(repo="example/one"), _rec(repo="example/copyleft-proj")]
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        stacks.load_dataset(p)
    msg = str(excinfo.value).casefold()
    # The exclusion (C5) gate is evaluated first; only when it passes are
    # copyleft offenders named. Here only the copyleft class fires.
    assert "example/copyleft-proj" in msg
    assert "acme/ok" not in msg
    assert "example/one" not in msg


def test_exclusion_gate_runs_before_copyleft_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stacks, "load_copyleft_list", lambda: frozenset({"example/one"}))
    recs = [_rec(repo="example/one"), _rec(repo="getsentry/sentry")]
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    with pytest.raises(ValueError, match="excluded"):
        stacks.load_dataset(p)


def test_load_dataset_tags_legacy_policy_rows(tmp_path: Path) -> None:
    """M23: rows without a labeler policy version are tagged legacy_policy=True.

    The tag is metadata the loader adds on admission (the loader never
    drop-and-warns); current-policy SFT prefers native-profile traces, and the
    tag is what lets downstream selection distinguish the two classes.
    """
    legacy = _rec(repo="acme/legacy")
    native = {**_rec(repo="acme/native"), "labeler_policy_version": "0.9.8-policy-r1"}
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (legacy, native)), encoding="utf-8")
    out = stacks.load_dataset(p)
    assert out[0]["legacy_policy"] is True
    assert out[1]["legacy_policy"] is False


def test_load_dataset_empty_policy_version_is_legacy(tmp_path: Path) -> None:
    p = tmp_path / "corpus.jsonl"
    p.write_text(json.dumps({**_rec(repo="acme/x"), "labeler_policy_version": None}), encoding="utf-8")
    out = stacks.load_dataset(p)
    assert out[0]["legacy_policy"] is True
