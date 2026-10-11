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
    def test_over_share_group_is_capped_to_limit(self) -> None:
        records = [_mk_record(f"r{i:03d}", "python", "owner/repo-a", "deep") for i in range(10)]
        records.append(_mk_record("r900", "rust", "owner/repo-b", "deep"))
        kept, exclusions = _apply_share_caps(records, max_stack_share=0.5, max_repo_share=None, max_profile_share=None)
        # Strict output-share semantics: iterate until python's share of the
        # final population is <= 0.5. With one rust record anchoring the
        # population, python converges to 1 kept (1/2) — 9 excluded, lowest
        # record_id kept.
        assert len(kept) == 2
        py = [r for r in kept if r["stack"] == "python"]
        assert [r["record_id"] for r in py] == ["r000"]
        assert [r["record_id"] for r in kept if r["stack"] == "rust"] == ["r900"]
        assert exclusions == {"stack:python": 9}

    def test_uniform_corpus_at_cap_keeps_everything(self) -> None:
        records = [_mk_record(f"r{i:03d}", "python" if i < 4 else "rust", "owner/repo", "deep") for i in range(10)]
        kept, exclusions = _apply_share_caps(records, max_stack_share=0.6, max_repo_share=None, max_profile_share=None)
        assert len(kept) == 10
        assert exclusions == {}

    def test_final_shares_respect_limits_after_sequential_passes(self) -> None:
        # Profiles must vary (a single profile value is trivially 100% of any
        # positive population and could never satisfy a <1.0 cap); the contract
        # under test is that every final share <= its limit over the final
        # population after the sequential stack → repository → profile passes.
        records = [_mk_record(f"r{i:03d}", "python" if i < 9 else "rust", "owner/repo-a" if i < 9 else "owner/repo-b",
                "deep" if i % 2 else "quick",
            )
            for i in range(10)
        ]
        kept, _ = _apply_share_caps(records, max_stack_share=0.6, max_repo_share=None, max_profile_share=0.5)
        _assert_shares_within(kept, [
            ("stack", lambda r: str(r["stack"]), 0.6), ("profile", lambda r: str(r["profile"]["profile_name"]), 0.5),
        ])

    def test_sequential_passes_reconverge_correlated_dimensions(self) -> None:
        # F1-shaped correlated fixture: python is concentrated in repo-a while
        # rust is split repo-a x2 + repo-b x3, and python sits exactly at the
        # 0.5 stack cap on entry (5/10). The repository pass (a later
        # dimension) then trims repo-a by lowest record_id — removing rust
        # records and pushing python back over its stack cap (4/7 = 0.571). A
        # single sequential pass would leave that drift in the output (it
        # breaks the M4 contract); the cap stage must re-run the dimension
        # sequence to a fixed point and reconverge every dimension within its
        # limit of the final population.
        records = [_mk_record(f"r{i:03d}", "python", "owner/repo-a", "deep") for i in range(5)]
        records += [
            _mk_record("r005", "rust", "owner/repo-a", "deep"), _mk_record("r006", "rust", "owner/repo-a", "deep"),
            _mk_record("r007", "rust", "owner/repo-b", "quick"), _mk_record("r008", "rust", "owner/repo-b", "quick"),
            _mk_record("r009", "rust", "owner/repo-b", "quick"),
        ]
        kept, exclusions = _apply_share_caps(records, max_stack_share=0.5, max_repo_share=0.6, max_profile_share=None)
        # The fixed point rebuilds a population where every dimension is back
        # within its limit — three python + three rust across both repos, all
        # shares 0.5. The drift (python at 0.571 after the repo pass) is
        # repaired by re-running the stack pass after the repository pass's
        # exclusions.
        assert [r["record_id"] for r in kept] == ["r000", "r001", "r002", "r007", "r008", "r009"]
        _assert_shares_within(kept, [
            ("stack", lambda r: str(r["stack"]), 0.5), ("repo", lambda r: str(r["lineage"]["repo_slug"]), 0.6),
        ])
        # The repository pass excluded 3; a later fixed-point pass re-ran the
        # stack pass (1 further exclusion) to repair the drift.
        assert exclusions == {"repo:owner/repo-a": 3, "stack:python": 1}

    def test_order_invariance_identical_kept_population(self) -> None:
        records = [_mk_record(f"r{i:03d}", "python" if i % 3 else "rust", "owner/repo-a" if i % 2 else "owner/repo-b",
                "deep" if i % 2 else "quick",
            )
            for i in range(12)
        ]
        a_kept, a_excl = _apply_share_caps(list(records), max_stack_share=0.5, max_repo_share=0.9, max_profile_share=0.6
        )
        shuffled = list(reversed(records))
        b_kept, b_excl = _apply_share_caps(shuffled, max_stack_share=0.5, max_repo_share=0.9, max_profile_share=0.6)
        assert [r["record_id"] for r in a_kept] == [r["record_id"] for r in b_kept]
        assert a_excl == b_excl

    def test_none_dimension_value_is_its_own_bucket_and_cappable(self) -> None:
        records = [_mk_record(f"r{i:03d}", None, "owner/repo-a", None) for i in range(8)]
        records.append(_mk_record("r800", "rust", "owner/repo-b", "deep"))
        kept, exclusions = _apply_share_caps(records, max_stack_share=0.5, max_repo_share=None, max_profile_share=None)
        # None bucket capped like any value: final pop 5, at most 2-3 None-stack kept.
        none_kept = [r for r in kept if r["stack"] is None]
        total = len(kept)
        assert total > 0
        assert len(none_kept) / total <= 0.5 + 1e-9
        assert "stack:(none)" in exclusions


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


@pytest.mark.parametrize("dimension", ["stack", "repo", "profile"])
def test_real_cli_applies_caps_and_pins_matching_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], dimension: str,
) -> None:
    store = _store_with_caps_population(tmp_path)
    config = projection_config(store, tmp_path, **{f"max_{dimension}_share": 0.5})
    args = ["corpus", "build", "--store", str(store.root), "--snapshot-id", config.snapshot_id,
            "--out", str(config.out_dir / "corpus.jsonl"),
            f"--max-{dimension}-share", "0.5"]
    assert cli_main(args + ["--dry-run"]) == 0
    dry_text = capsys.readouterr().out
    assert not config.out_dir.exists()
    assert cli_main(args) == 0
    records = _read_jsonl(config.out_dir / "corpus.jsonl")
    assert len(records) == 4
    assert "4" in dry_text
    getter = {"stack": lambda r: str(r["stack"]), "repo": lambda r: str(r["lineage"]["repo_slug"]),
              "profile": lambda r: str(r["profile"]["profile_name"])}[dimension]
    _assert_shares_within(records, [(dimension, getter, 0.5)])
    lineage = json.loads((config.out_dir / "lineage.json").read_text())
    assert lineage["share_caps"]["configured"][dimension] == 0.5
    assert lineage["share_caps"]["version"] == 1
    assert any(key.startswith(f"share-cap:{dimension}:") for key in lineage["exclusions_by_reason"])
    direct = build_frozen_corpus(projection_config(store, tmp_path, out_dir=tmp_path / "direct",
                                                  **{f"max_{dimension}_share": 0.5}))
    assert direct["share_caps"] == lineage["share_caps"]


@pytest.mark.parametrize(("dimension", "value"), [(dimension, value) for dimension in ("stack", "repo", "profile")
                                                  for value in ("0", "-0.1", "1.5")])
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
