"""Tests for the corpus-v2 share-cap stage and its build wiring."""
import hashlib
import json
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

    def test_sequential_passes_reconverge_correlated_dimensions(self) -> None:
        from daydream.training.corpus_v2.projector import _apply_share_caps

        # F1-shaped correlated fixture: python is concentrated in repo-a while
        # rust is split repo-a x2 + repo-b x3, and python sits exactly at the
        # 0.5 stack cap on entry (5/10). The repository pass (a later
        # dimension) then trims repo-a by lowest record_id — removing rust
        # records and pushing python back over its stack cap (4/7 = 0.571). A
        # single sequential pass would leave that drift in the output (it
        # breaks the M4 contract); the cap stage must re-run the dimension
        # sequence to a fixed point and reconverge every dimension within its
        # limit of the final population.
        records = [
            _mk_record(f"r{i:03d}", "python", "owner/repo-a", "deep") for i in range(5)
        ]
        records += [
            _mk_record("r005", "rust", "owner/repo-a", "deep"),
            _mk_record("r006", "rust", "owner/repo-a", "deep"),
            _mk_record("r007", "rust", "owner/repo-b", "quick"),
            _mk_record("r008", "rust", "owner/repo-b", "quick"),
            _mk_record("r009", "rust", "owner/repo-b", "quick"),
        ]
        kept, exclusions = _apply_share_caps(
            records, max_stack_share=0.5, max_repo_share=0.6, max_profile_share=None
        )
        # The fixed point rebuilds a population where every dimension is back
        # within its limit — three python + three rust across both repos, all
        # shares 0.5. The drift (python at 0.571 after the repo pass) is
        # repaired by re-running the stack pass after the repository pass's
        # exclusions.
        assert [r["record_id"] for r in kept] == [
            "r000", "r001", "r002", "r007", "r008", "r009",
        ]
        total = len(kept)
        stack: dict[str, int] = {}
        repo: dict[str, int] = {}
        for r in kept:
            stack[str(r["stack"])] = stack.get(str(r["stack"]), 0) + 1
            repo[str(r["lineage"]["repo_slug"])] = repo.get(str(r["lineage"]["repo_slug"]), 0) + 1
        for value, count in stack.items():
            assert count / total <= 0.5 + 1e-9, f"stack={value} share {count}/{total}"
        for value, count in repo.items():
            assert count / total <= 0.6 + 1e-9, f"repo={value} share {count}/{total}"
        # The repository pass excluded 3; a later fixed-point pass re-ran the
        # stack pass (1 further exclusion) to repair the drift.
        assert exclusions == {"repo:owner/repo-a": 3, "stack:python": 1}

    def test_order_invariance_identical_kept_population(self) -> None:
        from daydream.training.corpus_v2.projector import _apply_share_caps

        records = [
            _mk_record(
                f"r{i:03d}",
                "python" if i % 3 else "rust",
                "owner/repo-a" if i % 2 else "owner/repo-b",
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

    def test_sole_remaining_value_above_cap_fails_closed(self) -> None:
        from daydream.training.corpus_v2.projector import _apply_share_caps

        # Mono-value boundary consistency (issues #5/#7): a cap that floors to
        # zero on the entry population raises fail-closed, and the identical
        # terminal state via iteration (one over-share value left, trimming it
        # would empty the population) must raise too — never emit a single
        # record at 100% share over its cap with exit 0.
        for cap in (0.2, 0.4):
            records = [
                _mk_record(f"r{i:03d}", "python", "owner/repo-a", "deep") for i in range(4)
            ]
            with pytest.raises(ValueError, match="max_stack_share"):
                # 0.4 * 4 = 1.6 → keep 1 → the sole survivor is then 100%
                # of a 1-record population and can no longer be trimmed.
                _apply_share_caps(
                    records, max_stack_share=cap, max_repo_share=None, max_profile_share=None
                )

    def test_empty_population_with_configured_caps_stays_empty(self) -> None:
        from daydream.training.corpus_v2.projector import _apply_share_caps

        # Issue #6: a zero-record build (no decisive findings, or tier caps
        # that trimmed everything) previously completed with an empty corpus;
        # configuring a share cap must not turn it into a hard failure.
        kept, exclusions = _apply_share_caps(
            [], max_stack_share=0.5, max_repo_share=0.5, max_profile_share=0.5
        )
        assert kept == []
        assert exclusions == {}

    def test_fixed_point_fails_closed_when_caps_conflict(self) -> None:
        from daydream.training.corpus_v2.projector import _apply_share_caps

        # Finding-1 counterexample: the repo pass re-trims by lowest record_id
        # and would push stack A to 100% under single sequential passes. The
        # fixed point re-runs the stack pass, but the greedy lowest-id policy
        # cannot satisfy both caps here — it must fail closed rather than emit
        # a population that silently violates an earlier dimension's cap.
        records = [
            _mk_record("r001", "A", "owner/r1", "deep"),
            _mk_record("r002", "A", "owner/r1", "deep"),
            _mk_record("r003", "A", "owner/r1", "deep"),
            _mk_record("r004", "B", "owner/r1", "deep"),
            _mk_record("r005", "B", "owner/r1", "deep"),
            _mk_record("r006", "B", "owner/r1", "deep"),
            _mk_record("r007", "B", "owner/r1", "deep"),
            _mk_record("r008", "B", "owner/r1", "deep"),
            _mk_record("r009", "A", "owner/r2", "deep"),
            _mk_record("r010", "A", "owner/r2", "deep"),
        ]
        with pytest.raises(ValueError, match="max_stack_share"):
            _apply_share_caps(
                records, max_stack_share=0.5, max_repo_share=0.5, max_profile_share=None
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

    def test_sequential_passes_converge_all_final_shares_within_limits(
        self, tmp_path: Path
    ) -> None:
        # Real build with ALL THREE caps set tight (0.6) over a correlated
        # 5-record fixture (python/repo-a vs rust/repo-b) that exercises the
        # exclusion path: re-profiling every python row plus the first rust
        # row to deep-review puts four of the five records on one profile
        # value (4/5 = 0.8 > 0.6), so the profile pass must trim at entry — a
        # fixture landing every dimension exactly on the cap would never run
        # the trim branch and could not catch drift from a later pass. Assert
        # every final per-value share in the emitted corpus is <= its limit
        # (the M4 contract) against the post-trim final population, and that
        # the cap stage actually excluded records (its report names them
        # under ``exclusions_by_reason`` as ``share-cap:*``).
        import hashlib

        from daydream.training.corpus_v2.projector import run_build_corpus_v2
        from tests.test_corpus_v2 import (
            _admit_second_batch,
            _write_annotations_snapshot,
            _write_bundle,
        )

        bundle = _write_bundle(tmp_path)
        snap = _write_annotations_snapshot(
            bundle, session_id="sess-a", n_siblings=6,
            dispositions=["accepted", "accepted", "accepted"], stack="python",
        )
        _admit_second_batch(bundle, "owner/repo-b", spdx_id="MIT")
        _write_annotations_snapshot(
            bundle, session_id="sess-b", n_siblings=4,
            dispositions=["accepted", "accepted"], stack="rust",
        )
        # Re-profile only the last accepted row to quick-review so deep-review
        # holds 4/5 of the emitted population (0.8 > 0.6) and the trim branch
        # executes.
        rows = [json.loads(line) for line in snap.read_text().splitlines() if line]
        rows[-1]["profile"]["profile_name"] = "quick-review"
        snap.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        ann_dir = snap.parent
        rel = sorted(
            p.relative_to(ann_dir).as_posix() for p in ann_dir.rglob("*")
            if p.is_file() and p.name != "SHA256SUMS"
        )
        (ann_dir / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256((ann_dir / p).read_bytes()).hexdigest()}  {p}\n" for p in rel
        ))

        out = tmp_path / "out"
        summary = run_build_corpus_v2(self._share_cfg(
            out, bundle, snap,
            max_stack_share=0.6, max_repo_share=0.6, max_profile_share=0.6,
        ))
        emitted = [
            json.loads(line)
            for line in (out / "corpus.jsonl").read_text().splitlines() if line
        ]
        total = len(emitted)
        assert total > 0
        dims: list[tuple[str, Callable[[dict[str, Any]], str], float]] = [
            ("stack", lambda r: str(r["stack"]), 0.6),
            ("repo", lambda r: str(r["lineage"]["repo_slug"]), 0.6),
            ("profile", lambda r: str(r["profile"]["profile_name"]), 0.6),
        ]
        for key, getter, limit in dims:
            counts: dict[str, int] = {}
            for r in emitted:
                counts[getter(r)] = counts.get(getter(r), 0) + 1
            for value, count in counts.items():
                assert count / total <= limit + 1e-9, f"{key}={value}: {count}/{total} > {limit}"
        # The exclusion path executed: the over-share profile value was
        # trimmed at entry and reported as a share-cap exclusion.
        assert any(
            key.startswith("share-cap:") for key in summary["exclusions_by_reason"]
        ), summary["exclusions_by_reason"]

    def test_lineage_and_summary_share_caps_cannot_drift(self, tmp_path: Path) -> None:
        import json

        out, summary = self._build(tmp_path, max_repo_share=0.5)
        lineage = json.loads((out / "lineage.json").read_text())
        assert lineage["share_caps"] == summary["share_caps"]
        assert lineage["share_caps"]["version"] == 1
        assert lineage["share_caps"]["configured"]["repo"] == 0.5
        # one shared spelling across configured/applied/exclusion keys
        assert "repo" in lineage["share_caps"]["applied"]
        assert "repository" not in lineage["share_caps"]["applied"]
        assert any(
            key.startswith("share-cap:repo:") for key in lineage["exclusions_by_reason"]
        )
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


# ---------------------------------------------------------------------------
# Task 5: CLI wiring — build-v2 accepts share-cap flags (M2, M9)
# ---------------------------------------------------------------------------


class TestCliShareFlags:
    """CLI-level share-cap wiring: parser acceptance, fail-closed range
    validation before any build work, and dry-run parity over the real
    projection path."""

    def _base_argv(self, tmp_path: Path) -> list[str]:
        from tests.test_corpus_v2 import (
            _admit_second_batch,
            _policy_file,
            _write_annotations_snapshot,
            _write_bundle,
        )

        bundle_dir = _write_bundle(tmp_path)
        snap = _write_annotations_snapshot(bundle_dir)
        # A second admitted session with a distinct stack/repo so every
        # configured share cap is satisfiable (a lone value is 100% of the
        # population and can never satisfy a <1.0 cap); re-profile the tail
        # rows to a second profile value so the profile dimension has variety
        # too. The annotation bundle's SHA256SUMS must be refreshed after the
        # re-profile (same mechanics as the build-wiring fixtures).
        _admit_second_batch(bundle_dir, "owner/repo-b", spdx_id="MIT")
        snap = _write_annotations_snapshot(
            bundle_dir, session_id="sess-b", dispositions=["accepted", "accepted"],
            stack="rust",
        )
        rows = [json.loads(line) for line in snap.read_text().splitlines() if line]
        for row in rows[2:]:
            row["profile"]["profile_name"] = "quick-review"
        snap.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        ann_dir = snap.parent
        rel = sorted(
            p.relative_to(ann_dir).as_posix() for p in ann_dir.rglob("*")
            if p.is_file() and p.name != "SHA256SUMS"
        )
        (ann_dir / "SHA256SUMS").write_text("".join(
            f"{hashlib.sha256((ann_dir / p).read_bytes()).hexdigest()}  {p}\n" for p in rel
        ))
        return [
            "--bundle-root", str(bundle_dir),
            "--annotation-bundle-root", str(snap.parent),
            "--license-policy", str(_policy_file(tmp_path)),
            "--out", str(tmp_path / "out" / "corpus.jsonl"),
        ]

    def test_share_flags_accepted_by_parser(self) -> None:
        from daydream.cli import _build_build_corpus_v2_parser

        args = _build_build_corpus_v2_parser().parse_args(
            ["--bundle-root", "/b", "--annotation-bundle-root", "/a",
             "--license-policy", "/l", "--out", "/o/corpus.jsonl",
             "--max-stack-share", "0.5", "--max-repo-share", "0.6",
             "--max-profile-share", "0.7"]
        )
        assert args.max_stack_share == 0.5
        assert args.max_repo_share == 0.6
        assert args.max_profile_share == 0.7

    @pytest.mark.parametrize(
        ("flag", "value"),
        [("--max-stack-share", "1.5"), ("--max-stack-share", "0"),
         ("--max-repo-share", "1.5"), ("--max-repo-share", "-0.1"),
         ("--max-profile-share", "1.5"), ("--max-profile-share", "0")],
    )
    def test_share_out_of_range_refuses_before_build(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str, value: str
    ) -> None:
        from daydream.cli import _handle_build_corpus_v2_command

        rc = _handle_build_corpus_v2_command(self._base_argv(tmp_path) + [flag, value])
        assert rc == 1
        assert f"Invalid {flag}" in capsys.readouterr().out
        # refused before any build work: no output written
        assert not (tmp_path / "out" / "_SUCCESS").exists()

    def test_dry_run_writes_nothing_but_reports_capped_population(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from daydream.cli import _handle_build_corpus_v2_command
        from daydream.training.corpus_v2 import BuildCorpusV2Config, run_build_corpus_v2
        from tests.test_corpus_v2 import _policy_file

        argv = self._base_argv(tmp_path) + [
            "--dry-run", "--max-stack-share", "0.5", "--max-repo-share", "0.6",
            "--max-profile-share", "0.7",
        ]
        rc = _handle_build_corpus_v2_command(argv)
        assert rc == 0
        # dry run writes nothing into the real output directory
        assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
        out_text = capsys.readouterr().out

        # the printed count names the projected (share-capped) population:
        # run the same projection directly and compare the emitted counts.
        bundle_dir = tmp_path / "curated" / "cur-0123456789abcdef"
        snap = bundle_dir.parent / (bundle_dir.name + "-annotations")
        direct = run_build_corpus_v2(BuildCorpusV2Config(
            out_dir=tmp_path / "direct",
            bundle_dir=bundle_dir,
            annotation_bundle_dir=snap,
            license_policy_path=_policy_file(tmp_path),
            max_stack_share=0.5, max_repo_share=0.6, max_profile_share=0.7,
        ))
        assert str(direct["emitted"]) in out_text

    def test_dry_run_parity_for_capped_build(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from daydream.cli import _handle_build_corpus_v2_command
        from tests.test_corpus_v2 import _policy_file

        argv = self._base_argv(tmp_path) + ["--dry-run", "--max-stack-share", "0.5"]
        rc = _handle_build_corpus_v2_command(argv)
        assert rc == 0
        capsys.readouterr()
        # the real build with the same caps succeeds over the same inputs
        bundle_dir = tmp_path / "curated" / "cur-0123456789abcdef"
        snap = bundle_dir.parent / (bundle_dir.name + "-annotations")
        rc2 = _handle_build_corpus_v2_command([
            "--bundle-root", str(bundle_dir),
            "--annotation-bundle-root", str(snap),
            "--license-policy", str(_policy_file(tmp_path)),
            "--out", str(tmp_path / "real" / "corpus.jsonl"),
            "--max-stack-share", "0.5",
        ])
        assert rc2 == 0
        assert (tmp_path / "real" / "_SUCCESS").is_file()
        lineage = json.loads((tmp_path / "real" / "lineage.json").read_text())
        assert lineage["share_caps"]["configured"] == {"stack": 0.5}
