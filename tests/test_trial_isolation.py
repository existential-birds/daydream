"""Tests for isolated per-trial corpus dirs."""

from __future__ import annotations

import json

from daydream.benchmark.trial_isolation import (
    init_trial_corpus,
    trial_corpus_dir,
    trial_tool_label,
    trials_root,
)


def test_trial_corpus_dir_path(tmp_path):
    d = trial_corpus_dir(tmp_path, "daydream", 3)
    assert d == tmp_path / ".daydream-bench" / "trials" / "daydream" / "trial-03"


def test_trials_root_path(tmp_path):
    assert trials_root(tmp_path, "daydream-glm") == tmp_path / ".daydream-bench" / "trials" / "daydream-glm"


def test_trial_tool_label_format():
    assert trial_tool_label("daydream", 0) == "daydream-t00"
    assert trial_tool_label("daydream", 12) == "daydream-t12"
    assert trial_tool_label("daydream-glm", 1) == "daydream-glm-t01"


def test_init_trial_corpus_copies_data(tmp_path):
    canonical = tmp_path / "results" / "benchmark_data.json"
    canonical.parent.mkdir(parents=True)
    payload = {"https://x/pull/1": {"golden_comments": [], "reviews": []}}
    canonical.write_text(json.dumps(payload), encoding="utf-8")

    trial_dir = trial_corpus_dir(tmp_path, "daydream", 0)
    dest = init_trial_corpus(canonical, trial_dir)

    assert dest == trial_dir / "results" / "benchmark_data.json"
    assert dest.exists()
    assert json.loads(dest.read_text()) == payload


def test_init_trial_corpus_is_independent_of_canonical(tmp_path):
    """Mutating the trial copy must not touch the canonical corpus."""
    canonical = tmp_path / "results" / "benchmark_data.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_text(json.dumps({"a": 1}), encoding="utf-8")

    dest = init_trial_corpus(canonical, trial_corpus_dir(tmp_path, "daydream", 0))
    dest.write_text(json.dumps({"a": 2}), encoding="utf-8")

    assert json.loads(canonical.read_text()) == {"a": 1}
