import fcntl
import os
import stat
from pathlib import Path
from typing import Any, Literal

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
from tests.harness.transaction_faults import TransactionFaultDriver


def _stage(ctx: Any, path: Any, content: str) -> None:
    ctx.stage(path, content.encode())


def test_transaction_commit_replaces_all_and_manifest_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real locked transaction replaces typed files privately, retires an attested blob and publishes last."""
    data = tmp_path / "cases" / "nested" / "data.json"
    manifest = tmp_path / "benchmark.yaml"
    atomic_write_json(data, {"k": 1})
    atomic_write_yaml(manifest, {"schema_version": 1})
    assert load_json_strict(data) == {"k": 1} and load_yaml_strict(manifest) == {"schema_version": 1}
    assert stat.S_IMODE(data.stat().st_mode) == stat.S_IMODE(manifest.stat().st_mode) == 0o600
    assert stat.S_IMODE(data.parent.stat().st_mode) == stat.S_IMODE(data.parent.parent.stat().st_mode) == 0o700
    bundle = tmp_path / "snapshots" / "old.bundle"
    ensure_private_dir(bundle.parent)
    bundle.write_bytes(b"hello")
    digest = sha256_file(bundle)
    assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    replacements: list[Path] = []
    real_replace = os.replace
    def observe_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination) in (data, manifest):
            replacements.append(Path(destination))
        real_replace(source, destination)
    monkeypatch.setattr(os, "replace", observe_replace)
    with WorkspaceLock(tmp_path):
        assert (tmp_path / ".benchmark.lock").is_file()
        with Transaction(tmp_path, op_id="op-1", kind="write") as tx:
            tx.stage(data, b'{"k": 2}\n')  # the public API admits an absolute contained target
            tx.retire("snapshots/old.bundle", expected_sha256=digest)
            tx.stage("benchmark.yaml", b"schema_version: 2\n")
            tx.commit()
    assert replacements == [data, manifest]
    assert load_json_strict(data) == {"k": 2} and load_yaml_strict(manifest) == {"schema_version": 2}
    assert stat.S_IMODE(data.stat().st_mode) == stat.S_IMODE(manifest.stat().st_mode) == 0o600
    assert stat.S_IMODE(bundle.parent.stat().st_mode) == 0o700 and not bundle.exists()
    assert not list((tmp_path / "transactions").iterdir())
    with WorkspaceLock(tmp_path):
        recover_startup(tmp_path)
    assert load_json_strict(data) == {"k": 2} and load_yaml_strict(manifest) == {"schema_version": 2}


def test_transaction_retirement_recovers_with_other_targets(tmp_path: Path) -> None:
    for boundary in (0, "manifest"):
        _retire_and_recover(tmp_path / f"ws-{boundary}", boundary)


def _retire_and_recover(root: Path, boundary: int | Literal["manifest"]) -> None:
    bundle = root / "snapshots" / "old.bundle"
    case = root / "cases" / "case.yaml"
    manifest = root / "benchmark.yaml"
    bundle.parent.mkdir(parents=True)
    case.parent.mkdir(parents=True)
    bundle.write_bytes(b"old bundle")
    case.write_bytes(b"old case")
    manifest.write_bytes(b"old manifest")
    faults = TransactionFaultDriver(root, op_id="retire-crash", kind="import")
    with faults.transaction as tx:
        tx.retire("snapshots/old.bundle", expected_sha256=sha256_file(bundle))
        tx.stage("cases/case.yaml", b"new case")
        tx.stage("benchmark.yaml", b"new manifest")
        faults.halt_at("manifest" if boundary == "manifest" else f"target-{boundary}")
    doc = load_json_strict(root / "transactions" / "retire-crash" / "journal.json")
    applied = 3 if boundary == "manifest" else boundary
    assert doc["state"] == ("complete" if boundary == "manifest" else "prepared" if applied == 0 else "committing"
    )
    assert doc["applied_count"] == applied
    if applied:
        assert not bundle.exists()  # retirement was genuinely performed before recovery
    if applied >= 2:
        assert case.read_bytes() == b"new case"
    if applied >= 3:
        assert manifest.read_bytes() == b"new manifest"
    recover_startup(root)
    if boundary == "manifest":
        assert not bundle.exists()
        assert case.read_bytes() == b"new case"
        assert manifest.read_bytes() == b"new manifest"
    else:
        assert bundle.read_bytes() == b"old bundle"
        assert case.read_bytes() == b"old case"
        assert manifest.read_bytes() == b"old manifest"
    recover_startup(root)  # retirement recovery is idempotent after cleanup

def test_complete_journal_rejects_drifted_targets_at_startup(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    target.write_text("old")
    faults = TransactionFaultDriver(tmp_path, op_id="complete-altered", kind="write")
    with faults.transaction as tx:
        _stage(tx, target, "new")
        faults.halt_at("manifest")
    target.write_text("altered")
    with pytest.raises(WorkspaceCorrupt, match="digest mismatch"):
        recover_startup(tmp_path)
    assert (tmp_path / "transactions" / "complete-altered" / "journal.json").exists()
    target.unlink()
    (tmp_path / "transactions" / "complete-altered" / "journal.json").unlink()
    tmp_path = tmp_path / "retired-ws"
    retired = tmp_path / "snapshots" / "old.bundle"
    retired.parent.mkdir(parents=True)
    retired.write_bytes(b"old")
    faults = TransactionFaultDriver(tmp_path, op_id="complete-retired", kind="write")
    with faults.transaction as tx:
        tx.retire("snapshots/old.bundle", expected_sha256=sha256_file(retired))
        tx.stage("benchmark.yaml", b"new manifest")
        faults.halt_at("manifest")
    retired.write_bytes(b"returned")
    with pytest.raises(WorkspaceCorrupt, match="retired target"):
        recover_startup(tmp_path)
    assert (tmp_path / "transactions" / "complete-retired" / "journal.json").exists()

def test_crash_injection_at_every_boundary_restores_before_or_after(tmp_path: Path) -> None:
    # Assert each durable state before recovery, including complete-before-cleanup,
    # so every boundary exercises its actual recovery branch.
    for boundary in ("staged", "backup", "journal", "data", "manifest"):
        target = tmp_path / f"t-{boundary}.yaml"
        target.write_text("before")
        faults = TransactionFaultDriver(tmp_path, op_id=f"op-{boundary}", kind="write")
        with faults.transaction as tx:
            tx.stage(target.relative_to(tmp_path), b"after")
            faults.halt_at(boundary)
        op_dir = tmp_path / "transactions" / f"op-{boundary}"
        journal = op_dir / "journal.json"
        if boundary in ("staged", "backup"):
            assert not journal.exists()
            assert target.read_text() == "before"
        else:
            doc = load_json_strict(journal)
            expected_state = {"journal": "prepared", "data": "committing", "manifest": "complete"}[boundary]
            assert doc["state"] == expected_state
            assert doc["applied_count"] == (0 if boundary == "journal" else 1)
            assert target.read_text() == ("before" if boundary == "journal" else "after")
            if boundary == "manifest":
                assert (op_dir / "backup-0000.bin").exists()
        recover_startup(tmp_path)
        if boundary in ("staged", "backup"):
            assert target.read_text() == "before"
        elif boundary in ("journal", "data"):
            assert target.read_text() == "before"
        else:  # manifest
            assert target.read_text() == "after"
        assert not (tmp_path / "transactions").exists() or not list((tmp_path / "transactions").iterdir())

def _write_corrupt_journal(tmp_path: Path, op_id: Any, mutate: Any) -> None:
    """Build a valid prepared transaction, then corrupt its journal doc."""
    target = tmp_path / "target.yaml"
    target.write_text("old")  # pre-existing target so "untouched" is observable
    with Transaction(tmp_path, op_id=op_id, kind="write") as tx:
        _stage(tx, target, "old")
        tx.prepare()
    jf = tmp_path / "transactions" / op_id / "journal.json"
    doc = load_json_strict(jf)
    mutate(doc)
    atomic_write_json(jf, doc)


# One entry per journal validation in recover_startup; ``escaped_name`` marks the case that
# must also leave the workspace parent untouched (path confinement, not just fail-closed).
_JOURNAL_CORRUPTIONS: list[tuple[str, Any, str | None]] = [
    ("op-1", lambda d: d.__setitem__("state", "bogus"), None),
    ("op-2", lambda d: d.__setitem__("op_id", "other-dir"), None),
    ("op-3", lambda d: d["targets"][0].__setitem__("rel", "../../escaped-by-journal.bin"), "escaped-by-journal.bin"),
    ("op-4", lambda d: d["targets"][0].__setitem__("stage", "sub/stage-0000.bin"), None),
    ("op-5", lambda d: d["targets"].append(dict(d["targets"][0])), None),
    ("op-6", lambda d: d.__setitem__("applied_count", 99), None),
    ("op-7", lambda d: d.__setitem__("replacement_order", ["not-a-target"]), None),
]


def test_journal_corruption_and_conflicts_fail_closed(tmp_path: Path) -> None:
    for op_id, mutate, escaped_name in _JOURNAL_CORRUPTIONS:
        root = tmp_path / op_id / "ws"
        root.mkdir(parents=True)
        outside = None if escaped_name is None else root.parent / escaped_name
        _write_corrupt_journal(root, op_id, mutate)
        with pytest.raises(WorkspaceCorrupt):
            recover_startup(root)
        assert (root / "target.yaml").read_text() == "old"  # target untouched
        if outside is not None:
            assert not outside.exists()  # no path outside the workspace changed
    _cross_transaction_conflict(tmp_path / "conflict")


def _cross_transaction_conflict(tmp_path: Path) -> None:
    tmp_path.mkdir()
    target = tmp_path / "target.yaml"
    target.write_text("before")
    for i in (1, 2):
        with Transaction(tmp_path, op_id=f"op-{i}", kind="write") as tx:
            _stage(tx, target, f"after-{i}")
            tx.prepare()
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)
    assert target.read_text() == "before"  # neither journal applied; no last-writer-wins

def test_disjoint_journals_recover_each_durable_state_and_restart(tmp_path: Path) -> None:
    """One startup recovers every disjoint durable journal and preserves user content."""
    prepared = tmp_path / "prepared.yaml"
    committing = tmp_path / "committing.yaml"
    complete = tmp_path / "complete.yaml"
    prepared.write_text("prepared-before")
    committing.write_text("committing-before")
    complete.write_text("complete-before")
    os.chmod(committing, 0o600)
    os.chmod(complete, 0o600)
    prepared_faults = TransactionFaultDriver(tmp_path, op_id="op-a", kind="init")
    with prepared_faults.transaction as tx:
        tx.create_dir("cases")
        tx.stage("prepared.yaml", b"prepared-after")
        tx.stage("cases/new.yaml", b"new")
        prepared_faults.halt_at("target-0")
    kept = tmp_path / "cases" / "keep.yaml"
    kept.write_text("keep")
    with Transaction(tmp_path, op_id="op-b", kind="write") as tx:
        tx.stage("committing.yaml", b"committing-after")
        tx.begin_commit()
    complete_faults = TransactionFaultDriver(tmp_path, op_id="op-c", kind="write")
    with complete_faults.transaction as tx:
        tx.stage("complete.yaml", b"complete-after")
        complete_faults.halt_at("manifest")
    journals = [load_json_strict(tmp_path / "transactions" / op / "journal.json")
                for op in ("op-a", "op-b", "op-c")]
    assert [(doc["state"], doc["applied_count"]) for doc in journals] == [
        ("prepared", 0), ("committing", 1), ("complete", 1),
    ]
    assert (prepared.read_text(), committing.read_text(), complete.read_text()) == (
        "prepared-before", "committing-after", "complete-after",
    )
    assert not (tmp_path / "cases" / "new.yaml").exists()

    recover_startup(tmp_path)

    assert (prepared.read_text(), committing.read_text(), complete.read_text()) == (
        "prepared-before", "committing-before", "complete-after",
    )
    assert not (tmp_path / "cases" / "new.yaml").exists()
    assert kept.read_text() == "keep"
    assert stat.S_IMODE(kept.parent.stat().st_mode) == 0o700
    assert [stat.S_IMODE(path.stat().st_mode) for path in (committing, complete)] == [0o600, 0o600]
    assert list((tmp_path / "transactions").iterdir()) == []
    recover_startup(tmp_path)
    assert (prepared.read_text(), committing.read_text(), complete.read_text()) == (
        "prepared-before", "committing-before", "complete-after",
    )
    assert not (tmp_path / "cases" / "new.yaml").exists() and kept.read_text() == "keep"
    assert list((tmp_path / "transactions").iterdir()) == []

def test_recovery_refuses_unidentifiable_residue_and_unsafe_targets(tmp_path: Path) -> None:
    """Symlinked, foreign, or empty residue and escaping/unsafe targets fail closed untouched."""
    for check in (_never_follows_symlink, _removes_only_positive_residue, _empty_prejournal_dir_left_untouched,
                  _unsafe_targets_and_lock_contention):
        root = tmp_path / check.__name__
        root.mkdir()
        check(root)


def _never_follows_symlink(tmp_path: Path) -> None:
    # A journaled operation ensures recovery reaches _empty_transactions.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.txt").write_text("keep me")
    with Transaction(tmp_path, op_id="op-ok", kind="write") as tx:
        _stage(tx, tmp_path / "target.yaml", "new")
        tx.prepare()
    (tmp_path / "transactions" / "op-link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(WorkspaceCorrupt):  # symlink residue is unidentifiable -> fail closed
        recover_startup(tmp_path)
    assert (outside / "precious.txt").read_text() == "keep me"  # never followed/deleted
    assert (tmp_path / "transactions" / "op-link").is_symlink()  # never removed

def _removes_only_positive_residue(tmp_path: Path) -> None:
    # Mix a journal, recognized staging residue, and an unrecognized foreign file.
    with Transaction(tmp_path, op_id="op-j", kind="write") as tx:
        _stage(tx, tmp_path / "a.yaml", "x")
        tx.prepare()
    residue = tmp_path / "transactions" / "op-residue"
    residue.mkdir(parents=True)
    (residue / "stage-0000.bin").write_bytes(b"stale")
    foreign = tmp_path / "transactions" / "op-foreign"
    foreign.mkdir()
    (foreign / "note.txt").write_text("keep")
    with pytest.raises(WorkspaceCorrupt):  # foreign op dir is unidentifiable
        recover_startup(tmp_path)
    assert (foreign / "note.txt").read_text() == "keep"  # foreign left untouched

def _empty_prejournal_dir_left_untouched(tmp_path: Path) -> None:
    """An empty directory is not enough evidence to identify transaction residue."""
    op = tmp_path / "transactions" / "op-empty"
    op.mkdir(parents=True)
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)
    assert op.is_dir()  # left untouched

def _unsafe_targets_and_lock_contention(tmp_path: Path) -> None:
    (tmp_path / "cases").mkdir(parents=True)
    f = tmp_path / "cases" / "pr-000001-abcdef012345.yaml"
    f.write_text("case")
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path, indexed={str(f)}, on_disk={f})
    f.unlink()
    manifest = tmp_path / "benchmark.yaml"
    manifest.write_text("references a missing case\n")
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path, indexed={"cases/pr-000001-abcdef012345.yaml"}, on_disk=set())
    bundle = tmp_path / "snapshots" / "old.bundle"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"old bundle")
    with Transaction(tmp_path, op_id="retire-wrong", kind="import") as tx:
        with pytest.raises(WorkspaceCorrupt, match="digest mismatch"):
            tx.retire("snapshots/old.bundle", expected_sha256="f" * 64)
    assert bundle.read_bytes() == b"old bundle"
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir()
    (tmp_path / "links").symlink_to(outside, target_is_directory=True)
    with Transaction(tmp_path, op_id="op-sym", kind="write") as tx:
        with pytest.raises(WorkspaceCorrupt):
            tx.stage("links/x.yaml", b"x")
    assert not (outside / "x.yaml").exists()
    (tmp_path / "broken.json").write_text("{")
    with pytest.raises(WorkspaceCorrupt, match="invalid JSON"):
        load_json_strict(tmp_path / "broken.json")
    with open(tmp_path / ".benchmark.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(LockContentionError):
            WorkspaceLock(tmp_path, blocking=False).__enter__()
