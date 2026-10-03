"""Profile object, source, and digest propagate through trajectories, manifests, and SQLite.

Legacy manifests omit optional fields; legacy databases receive an additive migration.
"""
import sqlite3
from pathlib import Path

from daydream.archive import _schema
from daydream.backends import ResultEvent, TextEvent
from daydream.run_snapshot import RunProfileIdentity
from daydream.trajectory import DaydreamPhase
from tests.harness.trajectory import make_recorder
from tests.test_archive import _build, _manifest_identity


async def test_trajectory_build_extra_carries_profile_provenance(tmp_path: Path) -> None:
    rec = make_recorder(tmp_path, path=tmp_path / "trajectory.json", agent_model_name="opus", session_id="s1",)
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


def test_manifest_carries_profile_provenance(tmp_path: Path) -> None:
    profile = RunProfileIdentity(schema_version=1, name="p", source_kind="default", digest="abc")
    d = _build(tmp_path, identity=_manifest_identity(profile=profile))
    assert d["profile_schema_version"] == 1 and d["profile_digest"] == "abc"
    assert d["profile_name"] == "p" and d["profile_source_kind"] == "default"


def test_manifest_without_profile_omits_optional_fields(tmp_path: Path) -> None:
    d = _build(tmp_path)
    assert "profile_digest" not in d and "profile_name" not in d
    assert "profile_schema_version" not in d and "profile_source_kind" not in d

def test_sqlite_projection_has_profile_columns_and_migration() -> None:

    ddl = _schema._CREATE_TABLE  # the runs CREATE TABLE constant
    assert "profile_digest TEXT" in ddl
    conn = sqlite3.connect(":memory:")
    conn.execute(_schema._CREATE_TABLE)  # current DDL
    _schema._migrate_schema(conn)  # no-op on current, adds on legacy
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    assert {"profile_schema_version", "profile_name", "profile_source_kind", "profile_digest"} <= columns
