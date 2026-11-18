"""Tests for the projection share-cap stage and its build wiring."""
import json
from pathlib import Path

import pytest

from daydream.training.corpus_projection.projector import build_frozen_corpus
from daydream.training.corpus_projection.selection import _apply_share_caps
from tests.harness.record_projection import add_projection_run, projection_config, seed_projection_store
from tests.harness.scripts import cli_main
from tests.test_corpus_projection import _read_jsonl


def test_empty_population_with_configured_caps_stays_empty() -> None:
    # Issue #6: a zero-record build (no decisive findings) previously completed
    # with an empty corpus; configuring a share cap must not turn it into a hard failure.
    kept, exclusions = _apply_share_caps([], {"repo": 0.5, "profile": 0.5})
    assert kept == []
    assert exclusions == {}


def test_real_cli_applies_caps_and_pins_matching_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    store = seed_projection_store(tmp_path, dispositions=("accepted", "accepted", "rejected"))
    add_projection_run(store, run_id="sess-b", dispositions=("accepted", "accepted"), repo_slug="owner/repo-b")
    add_projection_run(store, run_id="sess-c", dispositions=("accepted", "accepted"), profile="quick-review")
    add_projection_run(store, run_id="sess-d", dispositions=("accepted",), profile="quick-review",
                       repo_slug="owner/repo-b")
    config = projection_config(store, tmp_path, max_repo_share=0.6, max_profile_share=0.7)
    args = [
        "corpus", "build", "--store", str(store.root), "--snapshot-id", config.snapshot_id,
        "--out", str(config.out_dir / "corpus.jsonl"),
        "--max-repo-share", "0.6", "--max-profile-share", "0.7",
    ]
    assert cli_main(args + ["--dry-run"]) == 0
    dry_text = capsys.readouterr().out
    assert not config.out_dir.exists()
    assert cli_main(args) == 0
    records = _read_jsonl(config.out_dir / "corpus.jsonl")
    assert [(row["session_id"], row["item_uid"]) for row in records] == [
        ("sess-b", "item:0"), ("sess-a", "item:1"), ("sess-c", "item:1"), ("sess-a", "item:0"), ("sess-d", "item:0"),
    ]
    assert "5 records" in dry_text
    lineage = json.loads((config.out_dir / "lineage.json").read_text())
    assert lineage["snapshot_id"] == config.snapshot_id
    report = lineage["share_caps"]
    assert report["version"] == 2
    assert report["configured"] == {"repo": 0.6, "profile": 0.7}
    assert report["applied"] == {
        "repo": {"owner/repo-a": {"count": 3, "share": 0.6}, "owner/repo-b": {"count": 2, "share": 0.4}},
        "profile": {"deep-review": {"count": 3, "share": 0.6}, "quick-review": {"count": 2, "share": 0.4}},
    }
    assert report["exclusions"] == {"repo:owner/repo-a": 2, "profile:deep-review": 1}
    assert lineage["exclusions_by_reason"] == {
        "non-decisive-adjudication": 0, "share-cap:repo:owner/repo-a": 2, "share-cap:profile:deep-review": 1,
    }
    build_frozen_corpus(projection_config(
        store, tmp_path, out_dir=tmp_path / "direct", max_repo_share=0.6, max_profile_share=0.7,
    ))
    assert json.loads((tmp_path / "direct" / "lineage.json").read_text())["share_caps"] == report


def test_cli_refuses_invalid_share_before_creating_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    store = seed_projection_store(tmp_path)
    config = projection_config(store, tmp_path)
    assert cli_main(["corpus", "build", "--store", str(store.root), "--snapshot-id", config.snapshot_id,
                     "--out", str(config.out_dir / "corpus.jsonl"), "--max-profile-share", "1.5"]) == 1
    output = capsys.readouterr().out
    assert "Invalid --max-profile-share" in output
    assert "Must be in (0, 1]." in output
    assert not config.out_dir.exists()


def test_build_caps_fail_closed_without_output(tmp_path: Path) -> None:
    # Two mono-repo, mono-profile records: 0.1 keeps none at entry, so profile refuses before
    # the earlier repo cap can; 0.6 keeps one, whose 100% share can no longer be trimmed.
    # Both terminal states fail closed instead of emitting it.
    for cap, repo_cap in ((0.1, 0.5), (0.6, None)):
        root = tmp_path / str(cap)
        with pytest.raises(ValueError, match="max_profile_share"):
            build_frozen_corpus(projection_config(
                seed_projection_store(root), root, max_repo_share=repo_cap, max_profile_share=cap,
            ))
        assert not (root / "out").exists()
