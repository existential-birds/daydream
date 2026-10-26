"""Tests for the projection share-cap stage and its build wiring."""
import json
from collections.abc import Callable, Sequence
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


def _assert_shares_within(records: Sequence[Any], dims: Sequence[tuple[str, Callable[[Any], str], float]]) -> None:
    """Assert every per-value share of ``records`` is within each dimension's limit (M4)."""
    total = len(records)
    assert total > 0, "cap stage must never silently empty the corpus"
    for key, getter, limit in dims:
        counts: dict[str, int] = {}
        for r in records:
            counts[getter(r)] = counts.get(getter(r), 0) + 1
        for value, count in counts.items():
            assert count / total <= limit + 1e-9, f"{key}={value}: {count}/{total} > {limit}"


class TestApplyShareCaps:


    def test_sequential_passes_reconverge_correlated_dimensions(self) -> None:
        records = [
            _mk_record("r000", None, "owner/repo-a", "deep"),
            _mk_record("r001", None, "owner/repo-a", "quick"),
            _mk_record("r002", None, "owner/repo-a", "quick"),
            _mk_record("r003", None, "owner/repo-a", "deep"),
            _mk_record("r004", None, "owner/repo-a", "deep"),
            _mk_record("r005", "rust", "owner/repo-a", "deep"),
            _mk_record("r006", "rust", "owner/repo-a", "quick"),
            _mk_record("r007", "rust", "owner/repo-b", "deep"),
            _mk_record("r008", "rust", "owner/repo-b", "quick"),
            _mk_record("r009", "rust", "owner/repo-b", "quick"),
        ]
        kept, exclusions = _apply_share_caps(
            records, max_stack_share=0.5, max_repo_share=0.6, max_profile_share=0.6,
        )
        assert [row["record_id"] for row in kept] == ["r000", "r001", "r007", "r008"]
        assert exclusions == {"repo:owner/repo-a": 3, "stack:(none)": 2, "profile:quick": 1}
        _assert_shares_within(kept, [
            ("stack", lambda row: str(row["stack"]), 0.5),
            ("repo", lambda row: str(row["lineage"]["repo_slug"]), 0.6),
            ("profile", lambda row: str(row["profile"]["profile_name"]), 0.6),
        ])
        reversed_kept, reversed_exclusions = _apply_share_caps(
            list(reversed(records)), max_stack_share=0.5, max_repo_share=0.6, max_profile_share=0.6,
        )
        assert reversed_kept == kept and reversed_exclusions == exclusions
        stable, stable_exclusions = _apply_share_caps(
            kept, max_stack_share=0.5, max_repo_share=0.6, max_profile_share=0.6,
        )
        assert stable == kept and stable_exclusions == {}


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
        # Issue #6: a zero-record build (no decisive findings, or tier caps
        # that trimmed everything) previously completed with an empty corpus;
        # configuring a share cap must not turn it into a hard failure.
        kept, exclusions = _apply_share_caps([], max_stack_share=0.5, max_repo_share=0.5, max_profile_share=0.5)
        assert kept == []
        assert exclusions == {}

    def test_fixed_point_fails_closed_when_caps_conflict(self) -> None:
        # Finding-1 counterexample: the repo pass re-trims by lowest record_id
        # and would push stack A to 100% under single sequential passes. The
        # fixed point re-runs the stack pass, but the greedy lowest-id policy
        # cannot satisfy both caps here — it must fail closed rather than emit
        # a population that silently violates an earlier dimension's cap.
        records = [_mk_record("r001", "A", "owner/r1", "deep"), _mk_record("r002", "A", "owner/r1", "deep"),
            _mk_record("r003", "A", "owner/r1", "deep"), _mk_record("r004", "B", "owner/r1", "deep"),
            _mk_record("r005", "B", "owner/r1", "deep"), _mk_record("r006", "B", "owner/r1", "deep"),
            _mk_record("r007", "B", "owner/r1", "deep"), _mk_record("r008", "B", "owner/r1", "deep"),
            _mk_record("r009", "A", "owner/r2", "deep"), _mk_record("r010", "A", "owner/r2", "deep"),
        ]
        with pytest.raises(ValueError, match="max_stack_share"):
            _apply_share_caps(records, max_stack_share=0.5, max_repo_share=0.5, max_profile_share=None)


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
    direct = build_frozen_corpus(projection_config(
        store, tmp_path, out_dir=tmp_path / "direct", max_stack_share=0.5, max_repo_share=0.6, max_profile_share=0.7,
    ))
    assert direct["share_caps"] == report


@pytest.mark.parametrize(("dimension", "value"), [("stack", "0"), ("repo", "-0.1"), ("profile", "1.5")])
def test_cli_refuses_invalid_share_before_creating_output(tmp_path: Path, dimension: str, value: str) -> None:
    store = _store_with_caps_population(tmp_path)
    config = projection_config(store, tmp_path)
    assert cli_main(["corpus", "build", "--store", str(store.root), "--snapshot-id", config.snapshot_id,
                     "--out", str(config.out_dir / "corpus.jsonl"), f"--max-{dimension}-share", value]) == 1
    assert not config.out_dir.exists()


def test_build_caps_account_for_final_population_and_fail_closed(tmp_path: Path) -> None:
    store = _store_with_caps_population(tmp_path)
    config = projection_config(store, tmp_path, max_stack_share=0.6, max_repo_share=0.6, max_profile_share=0.6,
                               caps={"gold": 4})
    summary = build_frozen_corpus(config)
    records = _read_jsonl(config.out_dir / "corpus.jsonl")
    _assert_shares_within(records, [("stack", lambda r: str(r["stack"]), 0.6),
                                  ("repo", lambda r: str(r["lineage"]["repo_slug"]), 0.6),
                                  ("profile", lambda r: str(r["profile"]["profile_name"]), 0.6)])
    assert summary["emitted"] + sum(summary["exclusions_by_reason"].values()) == 5
    with pytest.raises(ValueError, match="max_profile_share"):
        build_frozen_corpus(projection_config(seed_projection_store(tmp_path / "mono"), tmp_path / "mono",
                                              max_profile_share=0.1))
    assert not (tmp_path / "mono" / "out").exists()
