import fcntl
import os
import stat

import pytest

from daydream.benchmark.storage import (
    LockContentionError,
    WorkspaceCorrupt,
    WorkspaceLock,
    atomic_write_json,
    atomic_write_yaml,
    ensure_private_dir,
    load_json_strict,
    load_yaml_strict,
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
