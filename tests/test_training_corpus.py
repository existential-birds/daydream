"""Tests for daydream.training.corpus — the build-corpus projection.

Drives :func:`run_build_corpus` against a real SQLite index built with the
production ``upsert_run`` + ``append_label_observation`` helpers. No mocking of
SQLite, the archive layer, or the filesystem. The projection reads the
``as_of``-pinned annotation per run (silver) rather than the denormalized
``runs.outcome_labels`` cache, so the seeding helper writes a real bitemporal
annotation row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import jsonschema

from daydream.archive.index import append_label_observation, upsert_run
from daydream.archive.manifest import Manifest
from daydream.training.corpus import (
    BuildCorpusConfig,
    CorpusFilters,
    _annotation_reward,
    _build_query,
    _build_record,
    _is_admitted,
    _is_admitted_outcome_gold,
    run_build_corpus,
)
from daydream.training.labeler_versions import LABELER_POLICY_VERSION
from daydream.training.reward import PosteriorBreakdown, RewardBreakdown
from tests.fixtures.training.build_archive import build_fixture_archive

SCHEMA_PATH = Path(__file__).parent.parent / "daydream" / "training" / "schema" / "v1.json"


def _seed_run_with_annotation(
    archive_dir: Path,
    session_id: str,
    *,
    label: str | None = None,
    reward_json: str | None = None,
    composite_reward: float | None = None,
    reward_version: str = "r1",
    rubric_json: str | None = None,
    has_posterior: bool | None = None,
    labeler_policy_version: str | None = LABELER_POLICY_VERSION,
    observed_at: str,
    valid_at: str,
    pipeline_status: str = "unknown",
) -> Path:
    """Index a run and append one bitemporal annotation carrying label + reward.

    ``upsert_run`` registers the manifest row that build-corpus walks; the run
    directory holds the minimal bronze artifacts (``trajectory.json`` +
    ``manifest.json``) the projection materializes per row. The annotation is
    written via :func:`append_label_observation` with the reward/valid_at kwargs
    (present on HEAD from Tasks 1-2) so the pinned silver row — not the
    denormalized cache — is the source of truth for label/reward.

    Args:
        archive_dir: Archive root (the ``archive_dir`` fixture's tmpdir).
        session_id: Session UUID for the run + annotation.
        label: Outcome label to record on the annotation, or ``None`` for an
            empty label list.
        reward_json: Serialised ``RewardBreakdown.to_dict()`` JSON, or ``None``.
        composite_reward: Cached composite scalar mirrored onto the row.
        reward_version: Version tag stamped on the annotation.
        rubric_json: Serialised rubric dict stored on the annotation, or ``None``.
        observed_at: ISO-8601 transaction time. The test patches this onto the
            row directly because :func:`append_label_observation` stamps wall
            clock; we override it post-write for deterministic ``as_of`` pins.
        valid_at: ISO-8601 valid time (e.g. PR merge timestamp).

    Returns:
        The run directory holding the bronze artifacts.
    """
    run_dir = archive_dir / "runs" / session_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trajectory.json").write_text(json.dumps({"steps": []}), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest": {"session_id": session_id},
                "code_context": {"base_sha": "base123", "changed_files": ["app.py"]},
                "git": {"head_sha": "head456", "base_branch": "main", "branch": "feat"},
            }
        ),
        encoding="utf-8",
    )
    upsert_run(
        archive_dir,
        Manifest(
            session_id=session_id,
            archived_at="2026-01-01T00:00:00Z",
            run_flow="normal",
            backend="claude",
            skill="beagle-python:review-python",
            repo_slug="org/repo",
            base_sha="base123",
            head_sha="head456",
            base_branch="main",
            branch="feat",
            grounding_rate=1.0,
            changed_files=["app.py"],
            archive_path=str(run_dir),
            pipeline_status=pipeline_status,
        ),
    )
    append_label_observation(
        archive_dir,
        session_id,
        labels=[label] if label is not None else [],
        pr_state="merged" if label is not None else None,
        labeler_version="v1",
        evidence_sha=None,
        valid_at=valid_at,
        reward_version=reward_version,
        reward_json=reward_json,
        composite_reward=composite_reward,
        rubric_json=rubric_json,
        # Default: a labeled seed is an evidenced (pr_review) one, which is what
        # every caller means. A local_branch row passes has_posterior=False.
        has_posterior=(label is not None) if has_posterior is None else has_posterior,
    )
    # Stamp the policy axis: current-policy seeds (default
    # LABELER_POLICY_VERSION) are admitted as outcome gold; None models a
    # legacy row stamped before the reply-classifier policy axis existed.
    import sqlite3

    conn = sqlite3.connect(str(archive_dir / "index.db"))
    try:
        conn.execute(
            "UPDATE label_observations SET labeler_policy_version = ? WHERE session_id = ?",
            (labeler_policy_version, session_id),
        )
        conn.commit()
    finally:
        conn.close()
    # append_label_observation stamps observed_at with wall clock; overwrite it
    # so as_of pins in tests are deterministic.
    import sqlite3

    conn = sqlite3.connect(str(archive_dir / "index.db"))
    try:
        conn.execute(
            "UPDATE label_observations SET observed_at = ?, valid_at = COALESCE(?, valid_at) "
            "WHERE session_id = ?",
            (observed_at, valid_at, session_id),
        )
        conn.commit()
    finally:
        conn.close()
    return run_dir


def _cfg(tmp_path: Path, **overrides: Any) -> BuildCorpusConfig:
    """Build a BuildCorpusConfig pointing at ``tmp_path`` with sensible defaults."""
    base: dict[str, Any] = {
        "out_path": tmp_path / "out.jsonl",
        "filters": CorpusFilters(),
        "archive_dir": tmp_path,
    }
    base.update(overrides)
    return BuildCorpusConfig(**base)


def _load_schema() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def test_build_corpus_reads_as_of_annotation_and_embeds_reward(tmp_path: Path, archive_dir: Any) -> None:
    _seed_run_with_annotation(archive_dir, "s1", label="accepted",
                              reward_json='{"composite":0.7}', composite_reward=0.7,
                              observed_at="2026-03-01T00:00:00+00:00",
                              valid_at="2026-03-01T00:00:00+00:00")
    out = tmp_path / "corpus.jsonl"
    run_build_corpus(BuildCorpusConfig(out_path=out, archive_dir=archive_dir,
                                       filters=CorpusFilters(), as_of="2026-04-01T00:00:00+00:00"))
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["outcome_label"] == "accepted"
    assert rec["composite_reward"] == 0.7 and rec["reward"]["composite"] == 0.7


def test_record_with_reward_validates_against_schema(tmp_path: Path, archive_dir: Any) -> None:
    import jsonschema
    schema = json.loads(Path("daydream/training/schema/v1.json").read_text())
    _seed_run_with_annotation(archive_dir, "s1", label="accepted",
                              reward_json='{"composite":0.7,"axes_present":{}}', composite_reward=0.7,
                              observed_at="2026-03-01T00:00:00+00:00", valid_at="2026-03-01T00:00:00+00:00")
    out = tmp_path / "c.jsonl"
    run_build_corpus(BuildCorpusConfig(out_path=out, archive_dir=archive_dir,
                                       filters=CorpusFilters(), as_of="2026-04-01T00:00:00+00:00"))
    jsonschema.validate(json.loads(out.read_text().splitlines()[0]), schema)


# Migrated from tests/test_training_export.py — the §9 fixture matrix drives
# run_build_corpus end-to-end against a real SQLite index (silver annotations seeded by build_fixture_archive).
def test_export_emits_valid_jsonl(tmp_path: Path) -> None:
    """Every emitted line is valid JSON and validates against schema v1."""
    build_fixture_archive(tmp_path)
    summary = run_build_corpus(_cfg(tmp_path))
    assert summary["emitted"] > 0

    schema = _load_schema()
    lines = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines, "expected at least one emitted record"
    for line in lines:
        record = json.loads(line)
        jsonschema.validate(record, schema)


def test_export_deterministic_output(tmp_path: Path) -> None:
    """Two runs against the same archive produce byte-identical JSONL."""
    build_fixture_archive(tmp_path)
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    run_build_corpus(_cfg(tmp_path, out_path=out_a))
    run_build_corpus(_cfg(tmp_path, out_path=out_b))

    digest_a = hashlib.sha256(out_a.read_bytes()).hexdigest()
    digest_b = hashlib.sha256(out_b.read_bytes()).hexdigest()
    assert digest_a == digest_b


def test_export_schema_json_emitted_next_to_out(tmp_path: Path) -> None:
    """``schema.json`` lands next to ``out_path`` and matches the source."""
    build_fixture_archive(tmp_path)
    config = _cfg(tmp_path)
    run_build_corpus(config)

    schema_dst = config.out_path.parent / "schema.json"
    assert schema_dst.exists()
    assert schema_dst.read_text(encoding="utf-8") == SCHEMA_PATH.read_text(encoding="utf-8")


def test_export_dry_run_writes_nothing(tmp_path: Path) -> None:
    """``dry_run=True`` skips file writes and reports ``emitted=0``."""
    build_fixture_archive(tmp_path)
    config = _cfg(tmp_path, dry_run=True)
    summary = run_build_corpus(config)

    assert not config.out_path.exists()
    assert summary["emitted"] == 0
    assert summary["after_filters"] > 0  # filter pipeline still ran


def test_export_missing_trajectory_logged_and_skipped(tmp_path: Path) -> None:
    """A row with no ``trajectory.json`` is skipped (not crashed) and absent."""
    build_fixture_archive(tmp_path)
    (tmp_path / "runs" / "aaa-python-accepted" / "trajectory.json").unlink()

    config = _cfg(tmp_path)
    summary = run_build_corpus(config)
    assert summary["emitted"] > 0  # other rows still made it

    emitted_ids = {
        json.loads(line)["session_id"]
        for line in config.out_path.read_text(encoding="utf-8").splitlines()
    }
    assert "aaa-python-accepted" not in emitted_ids


def test_export_emit_schema_only_writes_schema_no_records(tmp_path: Path) -> None:
    """``emit_schema_only=True`` writes only ``schema.json``; no JSONL."""
    config = _cfg(tmp_path, emit_schema_only=True)
    summary = run_build_corpus(config)

    assert (config.out_path.parent / "schema.json").exists()
    assert not config.out_path.exists()
    assert summary == {
        "total_runs_in_index": 0,
        "after_filters": 0,
        "after_stratify": 0,
        "emitted": 0,
    }


def test_min_reward_admits_non_accepted_run(tmp_path: Path, archive_dir: Path) -> None:
    """A rejected run with intrinsic reward >= min_reward is admitted (C9 alt path)."""
    _seed_run_with_annotation(archive_dir, "s1", label="rejected",
                              reward_json='{"composite":0.8}', composite_reward=0.8,
                              observed_at="2026-03-01T00:00:00+00:00",
                              valid_at="2026-03-01T00:00:00+00:00")
    out = tmp_path / "out.jsonl"
    run_build_corpus(BuildCorpusConfig(out_path=out, archive_dir=archive_dir,
                                       filters=CorpusFilters(min_reward=0.5),
                                       as_of="2026-04-01T00:00:00+00:00"))
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["session_id"] for r in recs] == ["s1"]
    assert recs[0]["outcome_label"] == "rejected"
    assert recs[0]["composite_reward"] == 0.8


def test_unlabeled_run_at_as_of_is_dropped(tmp_path: Path, archive_dir: Path) -> None:
    """A run with no annotation at the pin is unlabeled and not admitted (C9)."""
    _seed_run_with_annotation(archive_dir, "s1", label="accepted",
                              reward_json='{"composite":0.7}', composite_reward=0.7,
                              observed_at="2026-05-01T00:00:00+00:00",
                              valid_at="2026-05-01T00:00:00+00:00")
    out = tmp_path / "out.jsonl"
    # Pin BEFORE the annotation's observed_at — no in-time annotation resolves.
    run_build_corpus(BuildCorpusConfig(out_path=out, archive_dir=archive_dir,
                                       filters=CorpusFilters(), as_of="2026-04-01T00:00:00+00:00"))
    assert out.read_text() == ""


# CLI surface — exercise the build-corpus handler (wired to run_build_corpus) without spawning a subprocess.
def test_cli_build_corpus_end_to_end(tmp_path: Path, archive_dir: Path) -> None:
    """Handler exits 0, writes JSONL + schema.json, every line parses."""
    from daydream.cli import _handle_build_corpus_command

    build_fixture_archive(archive_dir)

    out_path = tmp_path / "out.jsonl"
    rc = _handle_build_corpus_command(["--out", str(out_path)])

    assert rc == 0
    assert out_path.exists()
    assert (out_path.parent / "schema.json").exists()
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines, "expected at least one emitted record"
    for line in lines:
        json.loads(line)  # raises on malformed JSON


def test_cli_build_corpus_dry_run_via_handler(tmp_path: Path, archive_dir: Path) -> None:
    """``--dry-run`` returns 0 but writes no JSONL file."""
    from daydream.cli import _handle_build_corpus_command

    build_fixture_archive(archive_dir)

    out_path = tmp_path / "out.jsonl"
    rc = _handle_build_corpus_command(["--out", str(out_path), "--dry-run"])

    assert rc == 0
    assert out_path.exists() is False


def test_cli_build_corpus_allow_copyleft_flag_parsing() -> None:
    """``--allow-copyleft`` accumulates into a list on the namespace."""
    from daydream.cli import _build_build_corpus_parser

    parser = _build_build_corpus_parser()
    args = parser.parse_args(
        [
            "--out",
            "/tmp/x.jsonl",
            "--allow-copyleft",
            "gnu/coreutils",
            "--allow-copyleft",
            "fsf/bash",
        ]
    )
    assert args.allow_copyleft == ["gnu/coreutils", "fsf/bash"]


def test_cli_build_corpus_invalid_max_stack_share_returns_1() -> None:
    """``--max-stack-share`` outside (0, 1] is rejected with exit code 1."""
    from daydream.cli import _handle_build_corpus_command

    rc = _handle_build_corpus_command(["--out", "/tmp/x.jsonl", "--max-stack-share", "1.5"])
    assert rc == 1


def test_cli_build_corpus_passes_as_of_and_min_reward(tmp_path: Path, archive_dir: Path) -> None:
    """``--as-of`` and ``--min-reward`` thread through into the config."""
    from daydream.cli import _build_build_corpus_parser

    parser = _build_build_corpus_parser()
    args = parser.parse_args(
        ["--out", str(tmp_path / "x.jsonl"), "--as-of", "2026-04-01T00:00:00+00:00", "--min-reward", "0.5"]
    )
    assert args.as_of == "2026-04-01T00:00:00+00:00"
    assert args.min_reward == 0.5


# C3 — typed population separation: pin the intrinsic-only comparison and the posterior_cost discriminator.
# Post-C5 contract: stored composite_reward IS the pure intrinsic composite; posterior rides along as a sibling.
def _ann_with_posterior_reward_json() -> dict[str, Any]:
    """Annotation row for a labeled (PR-outcome) run.

    ``reward_json`` is a real ``PosteriorBreakdown.to_dict()`` — it carries
    ``posterior_cost`` (the population discriminator), while ``composite`` /
    ``composite_reward`` remain the pure intrinsic score (C5: the posterior is
    a sibling, never folded into the composite).
    """
    breakdown = PosteriorBreakdown(
        correctness_per_finding=[1.0],
        grounding=0.9,
        format_valid=True,
        length_penalty=0.1,
        composite=0.6,
        axes_present={"correctness": True, "grounding": True, "length": True},
        reward_version="2026.05.28-1",
        false_positive_penalty=1.0,
        posterior_cost=0.5,
        outcome_prior=0.5,
        outcome_prior_n=12,
    )
    reward_dict = breakdown.to_dict()
    return {
        "session_id": "s-labeled",
        "labels": json.dumps(["rejected"]),
        "reward_json": json.dumps(reward_dict),
        "composite_reward": reward_dict["composite"],
        "valid_at": "2026-04-01T00:00:00+00:00",
    }


def _ann_with_intrinsic_reward_json() -> dict[str, Any]:
    """Annotation row for an unlabeled (no PR-outcome) run.

    ``reward_json`` is a real ``RewardBreakdown.to_dict()`` — it has no
    ``posterior_cost`` key, so the absence of that key on the emitted record is
    what marks the row as intrinsic-only.
    """
    breakdown = RewardBreakdown(
        correctness_per_finding=[1.0],
        grounding=0.9,
        format_valid=True,
        length_penalty=0.1,
        composite=0.6,
        axes_present={"correctness": True, "grounding": True, "length": True},
        reward_version="2026.05.28-1",
    )
    reward_dict = breakdown.to_dict()
    return {
        "session_id": "s-intrinsic",
        "labels": json.dumps([]),
        "reward_json": json.dumps(reward_dict),
        "composite_reward": reward_dict["composite"],
        "valid_at": "2026-04-01T00:00:00+00:00",
    }


def test_is_admitted_min_reward_compares_intrinsic_only() -> None:
    """``min_reward`` compares against the stored intrinsic composite (C5).

    The labeled row carries a ``posterior_cost`` of 0.5 in its breakdown, yet
    its stored ``composite_reward`` is the pure intrinsic 0.6 (the posterior is
    never folded in). So ``min_reward=0.6`` admits it on the intrinsic threshold
    even though its label (``rejected``) is not in ``labels``. If the stored
    scalar were intrinsic-minus-posterior (0.1), this would NOT admit — that is
    the mixing bug C5 prevents and this test pins.
    """
    assert (
        _is_admitted(
            label="rejected",
            composite_reward=0.6,
            filters=CorpusFilters(min_reward=0.6, include_all_labels=False, labels=()),
        )
        is True
    )


def test_default_build_excludes_accepted_without_posterior_evidence(tmp_path: Path) -> None:
    """A ``local_branch`` "accepted" must not enter a default accepted-only corpus.

    The C9 default admits ``labels=("accepted",)``. A local-commit outcome
    carries exactly that label but no posterior evidence — a local commit is not
    a maintainer acting in a PR. Admitting on the bare label silently mixed the
    two evidence tiers in the training set (156 such rows in the real archive
    against 18 evidenced accepts). Only the evidenced run may be projected.
    """
    _seed_run_with_annotation(
        tmp_path, "s-evidenced", label="accepted", composite_reward=0.9,
        rubric_json=json.dumps({"posterior_source": "pr_review"}),
        has_posterior=True,
        observed_at="2026-05-01T00:00:00+00:00", valid_at="2026-04-01T00:00:00+00:00",
    )
    _seed_run_with_annotation(
        tmp_path, "s-local-tier", label="accepted", composite_reward=0.9,
        rubric_json=json.dumps({"posterior_source": "local_branch"}),
        has_posterior=False,
        observed_at="2026-05-01T00:00:00+00:00", valid_at="2026-04-01T00:00:00+00:00",
    )
    out = tmp_path / "corpus.jsonl"
    run_build_corpus(_cfg(tmp_path, out_path=out))

    emitted = [json.loads(line)["session_id"] for line in out.read_text().splitlines()]
    assert emitted == ["s-evidenced"]


def test_min_reward_still_admits_unevidenced_run_on_intrinsic_score(tmp_path: Path) -> None:
    """The intrinsic ``min_reward`` path is unchanged by the posterior gate.

    ``min_reward`` is an explicit opt-in to intrinsic-only admission (C5), so it
    still admits a run with no posterior evidence. The gate constrains only the
    *label* path, which is the one that claims maintainer evidence.
    """
    _seed_run_with_annotation(
        tmp_path, "s-local-tier", label="accepted", composite_reward=0.9,
        rubric_json=json.dumps({"posterior_source": "local_branch"}),
        has_posterior=False,
        observed_at="2026-05-01T00:00:00+00:00", valid_at="2026-04-01T00:00:00+00:00",
    )
    out = tmp_path / "corpus.jsonl"
    run_build_corpus(_cfg(tmp_path, out_path=out, filters=CorpusFilters(min_reward=0.5)))

    emitted = [json.loads(line)["session_id"] for line in out.read_text().splitlines()]
    assert emitted == ["s-local-tier"]


def test_build_record_emits_posterior_discriminator_only_for_labeled(tmp_path: Path) -> None:
    """``posterior_cost`` in ``record["reward"]`` is the population discriminator.

    ``reward_json`` is parsed via ``_annotation_reward`` and written verbatim
    (no transform), so a labeled annotation built from
    ``PosteriorBreakdown.to_dict()`` carries ``posterior_cost`` while an
    unlabeled one built from ``RewardBreakdown.to_dict()`` does not.
    """
    manifest_row = {"session_id": "s", "archive_path": str(tmp_path)}

    labeled_reward, labeled_composite = _annotation_reward(_ann_with_posterior_reward_json(), "s")
    rec_labeled = _build_record(
        manifest_row,
        trajectory={},
        stack=None,
        manifest=None,
        reward=labeled_reward,
        composite_reward=labeled_composite,
    )
    intrinsic_reward, intrinsic_composite = _annotation_reward(_ann_with_intrinsic_reward_json(), "s")
    rec_intrinsic = _build_record(
        manifest_row,
        trajectory={},
        stack=None,
        manifest=None,
        reward=intrinsic_reward,
        composite_reward=intrinsic_composite,
    )

    assert "posterior_cost" in rec_labeled["reward"]
    assert "posterior_cost" not in rec_intrinsic.get("reward", {})
    # The discriminator does not leak into the intrinsic composite scalar.
    assert rec_labeled["composite_reward"] == 0.6
    assert rec_intrinsic["composite_reward"] == 0.6


def test_corpus_can_filter_on_pipeline_status(tmp_path: Path, archive_dir: Any) -> None:
    """The status gate (default 'complete') must be able to exclude failed pipelines.

    ``BuildCorpusConfig`` gains a ``pipeline_status`` knob that flows into the
    WHERE clause as ``pipeline_status = ?``, mirroring the existing
    ``status = ?`` construction — the ''authoritative'' pipeline-outcome gate
    the spec's consumer migration calls for.
    """
    cfg = BuildCorpusConfig(out_path=tmp_path / "out.jsonl",
                            filters=CorpusFilters(status="complete"),
                            pipeline_status="succeeded", archive_dir=archive_dir)
    where, params = _build_query(filters=cfg.filters)
    # The status gate itself is unchanged...
    assert "status = ?" in where and "complete" in params
    # ...and with the config-level knob applied the query gains the pipeline gate.
    from dataclasses import replace
    effective = replace(cfg.filters, pipeline_status=cfg.pipeline_status)
    where2, params2 = _build_query(filters=effective)
    assert "pipeline_status = ?" in where2 and "succeeded" in params2


def test_build_corpus_pipeline_status_gate_excludes_failed_pipelines(tmp_path: Path, archive_dir: Any) -> None:
    """Real-path: a corpus built with pipeline_status='succeeded' drops failed runs.

    Seeds one succeeded and one failed pipeline run in the index; the emitted
    JSONL must contain only the succeeded run.
    """
    _seed_run_with_annotation(archive_dir, "ok-run", label="accepted",
                              observed_at="2026-03-01T00:00:00+00:00",
                              valid_at="2026-03-01T00:00:00+00:00",
                              pipeline_status="succeeded")
    _seed_run_with_annotation(archive_dir, "bad-run", label="accepted",
                              observed_at="2026-03-01T00:00:00+00:00",
                              valid_at="2026-03-01T00:00:00+00:00",
                              pipeline_status="failed")
    out = tmp_path / "corpus.jsonl"
    run_build_corpus(BuildCorpusConfig(
        out_path=out, archive_dir=archive_dir,
        filters=CorpusFilters(status="complete"), pipeline_status="succeeded",
        as_of="2026-04-01T00:00:00+00:00",
    ))
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["session_id"] == "ok-run"
    # Without the knob, both runs are admitted (backward compat preserved).
    out2 = tmp_path / "corpus-all.jsonl"
    run_build_corpus(BuildCorpusConfig(
        out_path=out2, archive_dir=archive_dir,
        filters=CorpusFilters(status="complete"),
        as_of="2026-04-01T00:00:00+00:00",
    ))
    lines2 = [line for line in out2.read_text().splitlines() if line.strip()]
    assert len(lines2) == 2


def test_cli_build_corpus_default_excludes_failed_pipelines(tmp_path: Path, archive_dir: Any) -> None:
    """Finding #8: the corpus build CLI gates on pipeline_status='succeeded' by
    default, so a default build drops merge-failed runs (archived status=complete
    but pipeline_status=failed) rather than admitting them into training data.

    The config-level knob already exists; this pins that the CLI *wires* it with
    a safe default instead of leaving it opt-in only.
    """
    from daydream.cli import _handle_build_corpus_command

    _seed_run_with_annotation(archive_dir, "ok-run", label="accepted",
                              observed_at="2026-03-01T00:00:00+00:00",
                              valid_at="2026-03-01T00:00:00+00:00",
                              pipeline_status="succeeded")
    _seed_run_with_annotation(archive_dir, "bad-run", label="accepted",
                              observed_at="2026-03-01T00:00:00+00:00",
                              valid_at="2026-03-01T00:00:00+00:00",
                              pipeline_status="failed")
    out = tmp_path / "corpus.jsonl"
    # No --pipeline-status passed: the CLI default ('succeeded') must apply and
    # drop the merge-failed run.
    rc = _handle_build_corpus_command(["--out", str(out)])
    assert rc == 0
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["session_id"] == "ok-run"

    # An explicit override proves the flag threads through to the SQL gate.
    out2 = tmp_path / "corpus-failed.jsonl"
    rc2 = _handle_build_corpus_command(
        ["--out", str(out2), "--pipeline-status", "failed"]
    )
    assert rc2 == 0
    lines2 = [line for line in out2.read_text().splitlines() if line.strip()]
    assert len(lines2) == 1
    assert json.loads(lines2[0])["session_id"] == "bad-run"


LEGACY: dict[str, Any] = {"label": "accepted", "has_posterior": True, "labeler_policy_version": None,
                          "decisive_mix": False, "decisive_only": True}
MIXED: dict[str, Any]  = {**LEGACY, "labeler_policy_version": LABELER_POLICY_VERSION, "decisive_mix": True}
AMBIG: dict[str, Any]  = {
    **LEGACY, "labeler_policy_version": LABELER_POLICY_VERSION, "decisive_mix": False, "decisive_only": False}
GOOD: dict[str, Any]   = {**LEGACY, "labeler_policy_version": LABELER_POLICY_VERSION}


def test_gold_admission_rejects_legacy_observations() -> None:
    """Legacy reply-count/merge-presence rows are excluded from outcome-bearing gold (M16/M22)."""
    assert _is_admitted_outcome_gold(**LEGACY) is False


def test_gold_admission_rejects_mixed_and_ambiguous() -> None:
    assert _is_admitted_outcome_gold(**MIXED) is False
    assert _is_admitted_outcome_gold(**AMBIG) is False


def test_gold_admission_accepts_current_clean_evidence() -> None:
    assert _is_admitted_outcome_gold(**GOOD) is True


def test_is_admitted_label_path_respects_filters_labels() -> None:
    """The label path requires both filters.labels membership and the gold guard.

    A ``labels=()`` filter admits nothing on the label path even when the row
    is current-policy gold — and a clean gold row still admits under the
    default ``("accepted",)`` filter, so the guard never narrows the default.
    """
    assert _is_admitted("accepted", None, CorpusFilters(labels=()), **
                        {k: v for k, v in GOOD.items() if k != "label"}) is False
    assert _is_admitted(**GOOD, composite_reward=None, filters=CorpusFilters()) is True


def test_min_reward_path_unaffected() -> None:
    """The intrinsic min_reward path keeps its existing contract (no creep)."""
    assert _is_admitted("rejected", 1.0, CorpusFilters(min_reward=0.5)) is True


def test_min_reward_path_drops_legacy_and_unevidenced_gold_label(tmp_path: Path, archive_dir: Any) -> None:
    """The min_reward back door must not admit a failed-guard ``accepted`` as gold (#980).

    The intrinsic ``min_reward`` path keeps admitting rows, but an outcome-gold
    ``accepted`` whose evidence fails the gold-admission guard (legacy NULL
    policy, or unevidenced local_branch) is admitted only as an unlabeled
    intrinsic row — its label is dropped so it never re-enters the accepted/gold
    population. A current-policy, evidenced, decisive-only ``accepted`` admitted
    via ``min_reward`` keeps its label.
    """
    decisive = json.dumps(
        {"posterior_source": "pr_review", "per_finding_outcomes": ["accepted", "accepted"]}
    )
    kw_composite_reward = 0.9
    kw_observed_at = "2026-03-01T00:00:00+00:00"
    kw_valid_at = "2026-03-01T00:00:00+00:00"
    # Legacy NULL-policy mislabeled accept: admitted intrinsically, label dropped.
    _seed_run_with_annotation(
        archive_dir, "s-legacy", label="accepted", rubric_json=decisive,
        has_posterior=True, labeler_policy_version=None,
        composite_reward=kw_composite_reward, observed_at=kw_observed_at,
        valid_at=kw_valid_at,
    )
    # Unevidenced local_branch accept: admitted intrinsically, label dropped.
    _seed_run_with_annotation(
        archive_dir, "s-local", label="accepted", rubric_json=decisive,
        has_posterior=False,
        composite_reward=kw_composite_reward, observed_at=kw_observed_at,
        valid_at=kw_valid_at,
    )
    # Current-policy evidenced accept: keeps its gold label.
    _seed_run_with_annotation(
        archive_dir, "s-good", label="accepted", rubric_json=decisive,
        has_posterior=True,
        composite_reward=kw_composite_reward, observed_at=kw_observed_at,
        valid_at=kw_valid_at,
    )
    out = tmp_path / "corpus.jsonl"
    run_build_corpus(BuildCorpusConfig(
        out_path=out, archive_dir=archive_dir,
        filters=CorpusFilters(min_reward=0.5),
        as_of="2026-04-01T00:00:00+00:00",
    ))
    labels = {json.loads(line)["session_id"]: json.loads(line)["outcome_label"]
              for line in out.read_text().splitlines()}
    assert labels["s-good"] == "accepted"
    assert labels["s-legacy"] is None
    assert labels["s-local"] is None


def test_gold_admission_gates_legacy_and_ambiguous_rows_in_build(tmp_path: Path, archive_dir: Any) -> None:
    """Real-path gold guard: legacy NULL-policy and rubric-ambiguous rows are excluded (M16/M22).

    The pure ``_is_admitted_outcome_gold`` tests above drive the guard in
    isolation; this drives the same gate through ``run_build_corpus`` so the
    ``_rubric_decisive_only`` derivation from the persisted ``rubric_json`` —
    including its FALSE branch on ambiguous ``per_finding_outcomes`` and the
    legacy ``labeler_policy_version=None`` row — is exercised end-to-end.
    """
    decisive_rubric = json.dumps(
        {"posterior_source": "pr_review", "per_finding_outcomes": ["accepted", "accepted"]}
    )
    ambiguous_rubric = json.dumps(
        {"posterior_source": "pr_review", "per_finding_outcomes": ["accepted", "ambiguous"]}
    )
    kw_composite_reward = 0.9
    kw_has_posterior = True
    kw_observed_at = "2026-05-01T00:00:00+00:00"
    kw_valid_at = "2026-04-01T00:00:00+00:00"
    # Current-policy + decisive rubric: admitted as gold.
    _seed_run_with_annotation(
        tmp_path, "s-good", label="accepted", rubric_json=decisive_rubric,
        has_posterior=kw_has_posterior, composite_reward=kw_composite_reward,
        observed_at=kw_observed_at, valid_at=kw_valid_at,
    )
    # Current-policy + ambiguous rubric: ``_rubric_decisive_only`` returns False, so no gold.
    _seed_run_with_annotation(
        tmp_path, "s-ambiguous", label="accepted", rubric_json=ambiguous_rubric,
        has_posterior=kw_has_posterior, composite_reward=kw_composite_reward,
        observed_at=kw_observed_at, valid_at=kw_valid_at,
    )
    # Legacy NULL policy + decisive rubric: the policy gate rejects it.
    _seed_run_with_annotation(
        tmp_path, "s-legacy", label="accepted", rubric_json=decisive_rubric,
        labeler_policy_version=None,
        has_posterior=kw_has_posterior, composite_reward=kw_composite_reward,
        observed_at=kw_observed_at, valid_at=kw_valid_at,
    )
    out = tmp_path / "corpus.jsonl"
    run_build_corpus(_cfg(tmp_path, out_path=out))

    emitted = [json.loads(line)["session_id"] for line in out.read_text().splitlines()]
    assert emitted == ["s-good"]
