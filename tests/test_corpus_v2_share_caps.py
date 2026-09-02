"""Tests for the pure share-cap selection stage (_apply_share_caps)."""

from collections.abc import Callable
from typing import Any

import pytest


def _mk_record(rid: str, stack: str | None, repo: str, profile: str | None) -> dict[str, Any]:
    return {
        "record_id": rid,
        "tier": "silver",
        "stack": stack,
        "profile": {"profile_name": profile},
        "lineage": {"repo_slug": repo, "split": "train"},
    }


class TestApplyShareCaps:
    def test_over_share_group_is_capped_to_limit(self) -> None:
        from daydream.training.corpus_v2.projector import _apply_share_caps

        records = [_mk_record(f"r{i:03d}", "python", "owner/repo-a", "deep") for i in range(10)]
        records.append(_mk_record("r900", "rust", "owner/repo-b", "deep"))
        kept, exclusions = _apply_share_caps(
            records, max_stack_share=0.5, max_repo_share=None, max_profile_share=None
        )
        # Strict output-share semantics: iterate until python's share of the
        # final population is <= 0.5. With one rust record anchoring the
        # population, python converges to 1 kept (1/2) — 9 excluded, lowest
        # record_id kept.
        assert len(kept) == 2
        py = [r for r in kept if r["stack"] == "python"]
        assert [r["record_id"] for r in py] == ["r000"]
        assert [r["record_id"] for r in kept if r["stack"] == "rust"] == ["r900"]
        assert exclusions == {"stack:python": 9}

    def test_final_shares_respect_limits_after_sequential_passes(self) -> None:
        from daydream.training.corpus_v2.projector import _apply_share_caps

        # Profiles must vary (a single profile value is trivially 100% of any
        # positive population and could never satisfy a <1.0 cap); the contract
        # under test is that every final share <= its limit over the final
        # population after the sequential stack → repository → profile passes.
        records = [
            _mk_record(
                f"r{i:03d}",
                "python" if i < 9 else "rust",
                "owner/repo-a" if i < 9 else "owner/repo-b",
                "deep" if i % 2 else "quick",
            )
            for i in range(10)
        ]
        kept, _ = _apply_share_caps(
            records, max_stack_share=0.6, max_repo_share=None, max_profile_share=0.5
        )
        total = len(kept)
        assert total > 0
        checks: list[tuple[str, float, Callable[[dict[str, Any]], str]]] = [
            ("stack", 0.6, lambda r: str(r["stack"])),
            ("profile", 0.5, lambda r: str(r["profile"]["profile_name"])),
        ]
        for key, limit, getter in checks:
            counts: dict[str, int] = {}
            for r in kept:
                counts[getter(r)] = counts.get(getter(r), 0) + 1
            for value, count in counts.items():
                assert count / total <= limit + 1e-9, f"{key}={value} share {count}/{total}"

    def test_order_invariance_identical_kept_population(self) -> None:
        from daydream.training.corpus_v2.projector import _apply_share_caps

        records = [
            _mk_record(
                f"r{i:03d}",
                "python" if i % 3 else "rust",
                "owner/repo-a",
                "deep" if i % 2 else "quick",
            )
            for i in range(12)
        ]
        a_kept, a_excl = _apply_share_caps(
            list(records), max_stack_share=0.5, max_repo_share=0.9, max_profile_share=0.6
        )
        shuffled = list(reversed(records))
        b_kept, b_excl = _apply_share_caps(
            shuffled, max_stack_share=0.5, max_repo_share=0.9, max_profile_share=0.6
        )
        assert [r["record_id"] for r in a_kept] == [r["record_id"] for r in b_kept]
        assert a_excl == b_excl

    def test_none_dimension_value_is_its_own_bucket_and_cappable(self) -> None:
        from daydream.training.corpus_v2.projector import _apply_share_caps

        records = [_mk_record(f"r{i:03d}", None, "owner/repo-a", None) for i in range(8)]
        records.append(_mk_record("r800", "rust", "owner/repo-b", "deep"))
        kept, exclusions = _apply_share_caps(
            records, max_stack_share=0.5, max_repo_share=None, max_profile_share=None
        )
        # None bucket capped like any value: final pop 5, at most 2-3 None-stack kept.
        none_kept = [r for r in kept if r["stack"] is None]
        total = len(kept)
        assert total > 0
        assert len(none_kept) / total <= 0.5 + 1e-9
        assert "stack:(none)" in exclusions

    def test_degenerate_cap_that_empties_population_fails_closed(self) -> None:
        from daydream.training.corpus_v2.projector import _apply_share_caps

        records = [_mk_record(f"r{i:03d}", "python", "owner/repo-a", "deep") for i in range(4)]
        with pytest.raises(ValueError, match="max_stack_share"):
            # 0.2 * 4 = 0.8 → keep 0 → population would collapse to zero.
            _apply_share_caps(
                records, max_stack_share=0.2, max_repo_share=None, max_profile_share=None
            )
