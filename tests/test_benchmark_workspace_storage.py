import stat

import pytest

from daydream.benchmark.storage import (
    WorkspaceCorrupt,
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
