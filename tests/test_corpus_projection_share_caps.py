"""Tests for the projection share-cap stage and its build wiring."""
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.training.corpus_projection.projector import build_frozen_corpus
from daydream.training.corpus_projection.selection import _apply_share_caps
from tests.harness.record_projection import add_projection_run, projection_config, seed_projection_store
from tests.harness.scripts import cli_main
from tests.test_corpus_projection import _read_jsonl


def _mk_record(rid: str, stack: str | None, repo: str, profile: str | None) -> dict[str, Any]:
    return {"record_id": rid, "tier": "silver", "stack": stack, "profile": {"profile_name": profile},
        "lineage": {"repo_slug": repo, "split": "train"},
    }


class TestApplyShareCaps:


    def test_sole_remaining_value_above_cap_fails_closed(self) -> None:
        # Mono-value boundary consistency (issues #5/#7): a cap that floors to
        # zero on the entry population raises fail-closed, and the identical
        # terminal state via iteration (one over-share value left, trimming it
        # would empty the population) must raise too — never emit a single
        # record at 100% share over its cap with exit 0.
        for cap in (0.2, 0.4):
            records = [_mk_record(f"r{i:03d}", "python", "owner/repo-a", "deep") for i in range(4)]
            with pytest.raises(ValueError, match="max_stack_share"):
                # 0.4 * 4 = 1.6 → keep 1 → the sole survivor is then 100%
                # of a 1-record population and can no longer be trimmed.
                _apply_share_caps(records, max_stack_share=cap, max_repo_share=None, max_profile_share=None)

    def test_empty_population_with_configured_caps_stays_empty(self) -> None:
        # Issue #6: a zero-record build (no decisive findings) previously completed
        # with an empty corpus; configuring a share cap must not turn it into a hard failure.
        kept, exclusions = _apply_share_caps([], max_stack_share=0.5, max_repo_share=0.5, max_profile_share=0.5)
        assert kept == []
        assert exclusions == {}

def _store_with_caps_population(tmp_path: Path) -> Any:
    store = seed_projection_store(tmp_path, dispositions=("accepted", "accepted", "rejected"))
    add_projection_run(store, run_id="sess-b", dispositions=("accepted", "accepted"), stack="rust",
                       profile="quick-review", repo_slug="owner/repo-b")
    return store


def test_real_cli_applies_caps_and_pins_matching_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    store = seed_projection_store(tmp_path, dispositions=("accepted", "accepted", "rejected"))
    add_projection_run(store, run_id="sess-b", dispositions=("accepted", "accepted"), repo_slug="owner/repo-b")
    add_projection_run(store, run_id="sess-c", dispositions=("accepted", "accepted"), stack="rust",
                       profile="quick-review")
    add_projection_run(store, run_id="sess-d", dispositions=("accepted",), stack="rust", repo_slug="owner/repo-b")
    config = projection_config(store, tmp_path, max_stack_share=0.5, max_repo_share=0.6, max_profile_share=0.7)
    args = [
        "corpus", "build", "--store", str(store.root), "--snapshot-id", config.snapshot_id,
        "--out", str(config.out_dir / "corpus.jsonl"),
        "--max-stack-share", "0.5", "--max-repo-share", "0.6", "--max-profile-share", "0.7",
    ]
    assert cli_main(args + ["--dry-run"]) == 0
    dry_text = capsys.readouterr().out
    assert not config.out_dir.exists()
    assert cli_main(args) == 0
    records = _read_jsonl(config.out_dir / "corpus.jsonl")
    assert [(row["session_id"], row["item_uid"]) for row in records] == [
        ("sess-b", "item:0"), ("sess-c", "item:1"),
    ]
    assert "2 records" in dry_text
    lineage = json.loads((config.out_dir / "lineage.json").read_text())
    assert lineage["snapshot_id"] == config.snapshot_id
    report = lineage["share_caps"]
    assert report["version"] == 1
    assert report["configured"] == {"stack": 0.5, "repo": 0.6, "profile": 0.7}
    assert report["applied"] == {
        "stack": {"python": {"count": 1, "share": 0.5}, "rust": {"count": 1, "share": 0.5}},
        "repo": {"owner/repo-a": {"count": 1, "share": 0.5}, "owner/repo-b": {"count": 1, "share": 0.5}},
        "profile": {"deep-review": {"count": 1, "share": 0.5}, "quick-review": {"count": 1, "share": 0.5}},
    }
    assert report["exclusions"] == {"stack:python": 3, "repo:owner/repo-a": 1, "profile:deep-review": 2}
    assert lineage["exclusions_by_reason"] == {
        "non-decisive-adjudication": 0, "share-cap:stack:python": 3,
        "share-cap:repo:owner/repo-a": 1, "share-cap:profile:deep-review": 2,
    }
    build_frozen_corpus(projection_config(
        store, tmp_path, out_dir=tmp_path / "direct", max_stack_share=0.5, max_repo_share=0.6, max_profile_share=0.7,
    ))
    assert json.loads((tmp_path / "direct" / "lineage.json").read_text())["share_caps"] == report


def test_cli_refuses_invalid_share_before_creating_output(tmp_path: Path) -> None:
    dimension = "profile"
    value = "1.5"
    store = _store_with_caps_population(tmp_path)
    config = projection_config(store, tmp_path)
    assert cli_main(["corpus", "build", "--store", str(store.root), "--snapshot-id", config.snapshot_id,
                     "--out", str(config.out_dir / "corpus.jsonl"), f"--max-{dimension}-share", value]) == 1
    assert not config.out_dir.exists()


def test_build_caps_fail_closed_without_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_profile_share"):
        build_frozen_corpus(projection_config(seed_projection_store(tmp_path / "mono"), tmp_path / "mono",
                                              max_profile_share=0.1))
    assert not (tmp_path / "mono" / "out").exists()
