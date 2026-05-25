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
from typing import Any

import jsonschema

from daydream.archive.index import append_label_observation, upsert_run
from daydream.archive.manifest import Manifest
from daydream.training.corpus import BuildCorpusConfig, CorpusFilters, run_build_corpus
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
    observed_at: str,
    valid_at: str,
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
    )
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
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_build_corpus_reads_as_of_annotation_and_embeds_reward(tmp_path, archive_dir):
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


# ---------------------------------------------------------------------------
# Migrated from tests/test_training_export.py — the §9 fixture matrix drives
# run_build_corpus end-to-end against a real SQLite index (now with silver
# annotations seeded by build_fixture_archive).
# ---------------------------------------------------------------------------


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


def test_export_record_fields_present(tmp_path: Path) -> None:
    """Every required schema field appears on the first emitted record."""
    build_fixture_archive(tmp_path)
    run_build_corpus(_cfg(tmp_path))

    schema = _load_schema()
    first_line = (tmp_path / "out.jsonl").read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    for field in schema["required"]:
        assert field in record, f"missing required field: {field}"


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


# ---------------------------------------------------------------------------
# CLI surface — exercise the existing export-jsonl handler (still wired to
# run_build_corpus until Task 11 retires it) without spawning a subprocess.
# ---------------------------------------------------------------------------


def test_cli_export_jsonl_end_to_end(tmp_path: Path, archive_dir: Path) -> None:
    """Handler exits 0, writes JSONL + schema.json, every line parses."""
    from daydream.cli import _handle_export_command

    build_fixture_archive(archive_dir)

    out_path = tmp_path / "out.jsonl"
    rc = _handle_export_command(["--out", str(out_path)])

    assert rc == 0
    assert out_path.exists()
    assert (out_path.parent / "schema.json").exists()
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert lines, "expected at least one emitted record"
    for line in lines:
        json.loads(line)  # raises on malformed JSON


def test_cli_export_jsonl_dry_run_via_handler(tmp_path: Path, archive_dir: Path) -> None:
    """``--dry-run`` returns 0 but writes no JSONL file."""
    from daydream.cli import _handle_export_command

    build_fixture_archive(archive_dir)

    out_path = tmp_path / "out.jsonl"
    rc = _handle_export_command(["--out", str(out_path), "--dry-run"])

    assert rc == 0
    assert out_path.exists() is False


def test_cli_export_jsonl_allow_copyleft_flag_parsing() -> None:
    """``--allow-copyleft`` accumulates into a list on the namespace."""
    from daydream.cli import _build_export_jsonl_parser

    parser = _build_export_jsonl_parser()
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


def test_cli_export_jsonl_invalid_max_stack_share_returns_1() -> None:
    """``--max-stack-share`` outside (0, 1] is rejected with exit code 1."""
    from daydream.cli import _handle_export_command

    rc = _handle_export_command(["--out", "/tmp/x.jsonl", "--max-stack-share", "1.5"])
    assert rc == 1
