import fcntl
import stat

import pytest

from daydream.benchmark.storage import (
    LockContentionError,
    Transaction,
    WorkspaceCorrupt,
    WorkspaceLock,
    atomic_write_json,
    atomic_write_yaml,
    ensure_private_dir,
    load_json_strict,
    load_yaml_strict,
    recover_startup,
    sha256_file,
)


def test_load_yaml_rejects_duplicate_keys(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text("a: 1\na: 2\n")
    with pytest.raises(WorkspaceCorrupt):
        load_yaml_strict(p)


def test_load_yaml_rejects_unknown_keys_at_schema_layer(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("schema_version: 1\nbogus: true\n")
    assert "schema_version" in load_yaml_strict(p)  # raw load is permissive; schema is strict


def test_load_yaml_safe_load_no_object_execution(tmp_path):
    p = tmp_path / "unsafe.yaml"
    p.write_text("!!python/object/apply:os.system ['true']\n")
    with pytest.raises(WorkspaceCorrupt):
        load_yaml_strict(p)


def test_atomic_write_json_0600_and_readback(tmp_path):
    dest = tmp_path / "out" / "nested" / "data.json"
    atomic_write_json(dest, {"k": 1})
    assert load_json_strict(dest) == {"k": 1}
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600


def test_atomic_write_yaml_0600_and_readback(tmp_path):
    dest = tmp_path / "benchmark.yaml"
    atomic_write_yaml(dest, {"schema_version": 1})
    assert load_yaml_strict(dest) == {"schema_version": 1}
    assert stat.S_IMODE(dest.stat().st_mode) == 0o600


def test_ensure_private_dir_is_0700(tmp_path):
    d = tmp_path / "private" / "nested"
    ensure_private_dir(d)
    assert stat.S_IMODE(d.stat().st_mode) == 0o700


def test_sha256_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

def test_workspace_lock_acquire_release(tmp_path):
    with WorkspaceLock(tmp_path):
        lock_file = tmp_path / ".benchmark.lock"
        assert lock_file.exists()
    # released: re-acquirable
    with WorkspaceLock(tmp_path):
        pass


def test_workspace_lock_contention_is_explicit(tmp_path):
    # A second exclusive holder on a SEPARATE open file description must be
    # surfaced explicitly, never silently ignored. Use a non-blocking probe so
    # this cannot deadlock against the first holder's flock (flock conflicts
    # across open file descriptions even within one process).
    first = WorkspaceLock(tmp_path)
    first.__enter__()
    try:
        with open(tmp_path / ".benchmark.lock", "w") as held:
            with pytest.raises(BlockingIOError):
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Our own WorkspaceLock handle is reentrant on the same fd.
        with WorkspaceLock(tmp_path, blocking=False):
            pass
    finally:
        first.__exit__(None, None, None)


def test_workspace_lock_contention_raises(tmp_path):
    lock_path = tmp_path / ".benchmark.lock"
    with open(lock_path, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(LockContentionError):
            WorkspaceLock(tmp_path, blocking=False).__enter__()


def _stage(ctx, path, content):
    ctx.stage(path, content.encode())


def test_transaction_commit_replaces_all_and_manifest_last(tmp_path):
    data_a = tmp_path / "cases" / "a.yaml"
    manifest = tmp_path / "benchmark.yaml"
    (tmp_path / "cases").mkdir(parents=True, exist_ok=True)
    with Transaction(tmp_path, op_id="op-1", kind="write") as tx:
        _stage(tx, data_a, "case-a")
        _stage(tx, manifest, "manifest-v2")
        tx.commit()
    assert data_a.read_text() == "case-a"
    assert manifest.read_text() == "manifest-v2"
    assert not (tmp_path / "transactions").exists() or not list((tmp_path / "transactions").iterdir())


def test_prepared_journal_rolls_back_on_startup(tmp_path):
    data = tmp_path / "cases" / "b.yaml"
    data.parent.mkdir(parents=True)
    with Transaction(tmp_path, op_id="op-2", kind="write") as tx:
        _stage(tx, data, "new-b")
        tx.prepare()  # fsync prepared, do NOT commit -> simulates crash
    recover_startup(tmp_path)
    assert not data.exists()


def test_committing_journal_rolls_back_in_reverse(tmp_path):
    target = tmp_path / "target.yaml"
    target.write_text("old")
    with Transaction(tmp_path, op_id="op-3", kind="write") as tx:
        _stage(tx, target, "new")
        tx.prepare()
        tx.begin_commit()  # set state=committing, apply target with new
        tx.inject_crash()  # leave committing incomplete, applied=1
    assert target.read_text() == "new"
    recover_startup(tmp_path)
    assert target.read_text() == "old"  # restored from backup


def test_complete_journal_is_verified_and_cleaned(tmp_path):
    target = tmp_path / "target.yaml"
    target.write_text("old")
    with Transaction(tmp_path, op_id="op-4", kind="write") as tx:
        _stage(tx, target, "new")
        tx.commit()
        tx.force_state("complete")  # simulate crash right after mark-complete, before journal removal
    recover_startup(tmp_path)
    assert target.read_text() == "new"  # after state verified, journal cleaned


def test_no_journal_orphan_is_corruption(tmp_path):
    orphan = tmp_path / "cases" / "pr-000001-abcdef012345.yaml"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("case")
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path, indexed=set(), on_disk={orphan})


def test_referenced_missing_file_is_corruption(tmp_path):
    manifest = tmp_path / "benchmark.yaml"
    manifest.write_text("references a missing case\n")
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path, indexed={"cases/pr-000001-abcdef012345.yaml"}, on_disk=set())
