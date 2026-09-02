"""Tests for the corpus-v2 share-cap stage and its build wiring."""
from collections.abc import Callable
from pathlib import Path
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


class TestBuildWiring:
    def _share_cfg(self, out: Path, bundle: Path, snap: Path, **share: Any) -> Any:
        from tests.test_corpus_v2 import _cfg

        return _cfg(out, bundle, snap, **share)

    def _build(self, tmp_path: Path, **share: Any) -> tuple[Path, dict[str, Any]]:
        from daydream.training.corpus_v2.projector import run_build_corpus_v2
        from tests.test_corpus_v2 import (
            _admit_second_batch,
            _write_annotations_snapshot,
            _write_bundle,
        )

        bundle = _write_bundle(tmp_path)
        # 3 accepted findings on sess-a (same stack/repo/profile) → over any
        # small share cap once a second dimension value exists; the 2-record
        # fixture is exercised for *reporting* here, M4's strict share math
        # is covered by TestApplyShareCaps.
        snap = _write_annotations_snapshot(
            bundle, session_id="sess-a",
            dispositions=["accepted", "accepted", "rejected"],
        )
        # Two admitted sessions so the emitted population has a second
        # stack/repo/profile value and the share cap is actually satisfiable
        # (the flip needs the annotation lineage to exist first).
        _admit_second_batch(bundle, "owner/repo-b", spdx_id="MIT")
        _write_annotations_snapshot(
            bundle, session_id="sess-b", dispositions=["accepted", "accepted"], stack="rust",
        )
        out = tmp_path / "out"
        summary = run_build_corpus_v2(
            self._share_cfg(out, bundle, snap, **share)
        )
        return out, summary

    def test_share_caps_exceed_and_exclusions_recorded(self, tmp_path: Path) -> None:
        import json

        out, summary = self._build(tmp_path, max_stack_share=0.5)
        emitted = [
            json.loads(line)
            for line in (out / "corpus.jsonl").read_text().splitlines() if line
        ]
        assert emitted, "cap stage must never silently empty the corpus"
        stack_counts: dict[str, int] = {}
        for r in emitted:
            stack_counts[str(r["stack"])] = stack_counts.get(str(r["stack"]), 0) + 1
        total = len(emitted)
        for value, count in stack_counts.items():
            assert count / total <= 0.5 + 1e-9, f"stack {value} exceeded cap"
        assert summary["share_caps"]["configured"]["stack"] == 0.5
        assert summary["share_caps"]["version"] == 1
        assert summary["exclusions_by_reason"]  # tier/np keys present

    def test_lineage_and_summary_share_caps_cannot_drift(self, tmp_path: Path) -> None:
        import json

        out, summary = self._build(tmp_path, max_repo_share=0.5)
        lineage = json.loads((out / "lineage.json").read_text())
        assert lineage["share_caps"] == summary["share_caps"]
        assert lineage["share_caps"]["version"] == 1
        assert lineage["share_caps"]["configured"]["repo"] == 0.5
        # final per-value counts + shares are reported against the final population
        assert "applied" in lineage["share_caps"] and "exclusions" in lineage["share_caps"]

    def test_no_caps_configured_report_block_absent(self, tmp_path: Path) -> None:
        import json

        out, summary = self._build(tmp_path)
        lineage = json.loads((out / "lineage.json").read_text())
        assert "share_caps" not in lineage
        assert "share_caps" not in summary

    def test_zero_population_cap_fails_closed(self, tmp_path: Path) -> None:
        from daydream.training.corpus_v2.projector import run_build_corpus_v2
        from tests.test_corpus_v2 import _write_annotations_snapshot, _write_bundle

        bundle = _write_bundle(tmp_path)
        snap = _write_annotations_snapshot(bundle, session_id="sess-a",
                                           dispositions=["accepted", "accepted", "accepted"])
        with pytest.raises(ValueError, match="max_profile_share"):
            run_build_corpus_v2(
                self._share_cfg(tmp_path / "out2", bundle, snap, max_profile_share=0.1)
            )
        # fail-closed: nothing written
        assert not (tmp_path / "out2" / "_SUCCESS").exists()
