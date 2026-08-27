"""Task 9 (R12): attributable review-profile provenance across stores.

The resolved review profile (validated object + source kind + digest) must
travel end to end: trajectory run metadata (top-level ``extra``), archive
manifest (optional fields, omitted on legacy), and the SQLite run projection
(with an additive migration for legacy DBs).
"""
import sqlite3
from pathlib import Path

from daydream.archive import manifest as m
from daydream.backends import ResultEvent, TextEvent
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder


async def test_trajectory_build_extra_carries_profile_provenance(tmp_path: Path) -> None:
    rec = TrajectoryRecorder(
        path=tmp_path / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="s1",
    )
    rec.record_profile(schema_version=1, name="p", source_kind="default", digest="abc")
    async with rec:
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="first chunk"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = rec.build_trajectory()  # the in-memory Trajectory; extra carries profile fields
    extra = traj.extra
    assert extra is not None
    assert extra["profile_schema_version"] == 1
    assert extra["profile_name"] == "p"
    assert extra["profile_source_kind"] == "default"
    assert extra["profile_digest"] == "abc"


def test_manifest_to_dict_carries_profile_provenance_and_omits_when_none() -> None:
    # A new manifest with profile fields serializes them; a legacy-shaped manifest
    # (no profile fields) omits them entirely (optional on legacy, R12).
    man = m.Manifest(  # fields per Manifest.__init__/to_dict; executor fills the rest
        schema_version="1",
        session_id="s",
        archived_at="2026-08-23T00:00:00Z",
        status="complete",
        profile_schema_version=1,
        profile_name="p",
        profile_source_kind="default",
        profile_digest="abc",
    )
    d = man.to_dict()
    assert d["profile_schema_version"] == 1 and d["profile_digest"] == "abc"
    assert d["profile_name"] == "p" and d["profile_source_kind"] == "default"


def test_legacy_manifest_without_profile_fields_still_serializes() -> None:
    # Legacy manifests are read from disk (json) and never rewritten (R12);
    # to_dict on a profile-field-None Manifest omits them.
    man = m.Manifest(  # profile_* all None (legacy shape)
        schema_version="1",
        session_id="s",
        archived_at="2026-01-01T00:00:00Z",
        status="complete",
    )
    d = man.to_dict()
    assert "profile_digest" not in d and "profile_name" not in d
    assert "profile_schema_version" not in d and "profile_source_kind" not in d


def test_sqlite_projection_has_profile_columns_and_migration() -> None:
    from daydream.archive import _schema

    ddl = _schema._CREATE_TABLE  # the runs CREATE TABLE constant
    assert "profile_digest TEXT" in ddl
    conn = sqlite3.connect(":memory:")
    conn.execute(_schema._CREATE_TABLE)  # current DDL
    _schema._migrate_schema(conn)  # no-op on current, adds on legacy
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    assert {"profile_schema_version", "profile_name", "profile_source_kind", "profile_digest"} <= columns
