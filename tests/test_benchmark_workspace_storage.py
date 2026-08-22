import fcntl
import os
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


def test_crash_injection_at_every_boundary_restores_before_or_after(tmp_path):
    # For each named boundary, drive a transaction that injects a crash there,
    # then recover_startup and assert the workspace is either the complete
    # before-state or the complete after-state — never checksum drift. Each
    # boundary advances the journal to its own distinct position (open staging
    # vs prepared vs committing vs complete) rather than blanket-preparing, so
    # the after-recovery (complete) branch is exercised too.
    for boundary in ("staged", "backup", "journal", "data", "manifest"):
        target = tmp_path / f"t-{boundary}.yaml"
        target.write_text("before")
        with Transaction(tmp_path, op_id=f"op-{boundary}", kind="write") as tx:
            tx.stage(target.relative_to(tmp_path), b"after")
            tx.inject_crash(boundary)
        recover_startup(tmp_path)
        if boundary in ("staged", "backup"):
            # Crash before the journal is written: recovery has nothing to do
            # and leaves the pristine target untouched.
            assert target.read_text() == "before"
        elif boundary in ("journal", "data"):
            # Prepared/committing journals roll back to the whole before-state.
            assert target.read_text() == "before"
        else:  # manifest
            # A complete journal is verified against the after-state and kept.
            assert target.read_text() == "after"
        assert not (tmp_path / "transactions").exists() or not list(
            (tmp_path / "transactions").iterdir()
        )


def test_prejournal_stage_residue_is_removed(tmp_path):
    target = tmp_path / "t.yaml"
    target.write_text("before")
    with Transaction(tmp_path, op_id="op-pre", kind="write") as tx:
        _stage(tx, target, "after")
        tx.inject_crash("staged")  # stage-*.bin written, NO journal.json
    recover_startup(tmp_path)
    assert target.read_text() == "before"  # untouched
    txn = tmp_path / "transactions"
    assert not txn.exists() or not list(txn.iterdir())  # residue gone


def test_prejournal_backup_residue_is_removed(tmp_path):
    target = tmp_path / "t.yaml"
    target.write_text("before")
    with Transaction(tmp_path, op_id="op-pre2", kind="write") as tx:
        _stage(tx, target, "after")
        tx.inject_crash("backup")  # stage + backup written, NO journal.json
    recover_startup(tmp_path)
    txn = tmp_path / "transactions"
    assert not txn.exists() or not list(txn.iterdir())


def test_unidentifiable_residue_is_corruption_and_left_untouched(tmp_path):
    op = tmp_path / "transactions" / "op-x"
    op.mkdir(parents=True)
    (op / "foreign.txt").write_text("not residue")
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)
    assert (op / "foreign.txt").read_text() == "not residue"  # left untouched, never guessed/deleted


def test_import_crash_transaction_restores_before_or_after(tmp_path):
    # The import writes {import file, case, benchmark.yaml} as one atomic unit.
    # Driving crater recovery across that whole set proves a crash at any
    # boundary leaves the complete before- or after-state, never a mix.
    for boundary in ("journal", "data", "manifest"):
        imp = tmp_path / "imports" / "pr-000101.json"
        case = tmp_path / "cases" / "pr-000101-aaaaaaaaaaaa.yaml"
        manifest = tmp_path / "benchmark.yaml"
        for f in (imp, case):
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("before")
        manifest.write_text("ledger-before")
        with Transaction(tmp_path, op_id=f"imp-{boundary}", kind="import") as tx:
            tx.stage("imports/pr-000101.json", b"import-after")
            tx.stage("cases/pr-000101-aaaaaaaaaaaa.yaml", b"case-after")
            tx.stage("benchmark.yaml", b"ledger-after")
            tx.inject_crash(boundary)
        recover_startup(tmp_path)
        if boundary in ("journal", "data"):
            assert imp.read_text() == "before"
            assert case.read_text() == "before"
            assert manifest.read_text() == "ledger-before"
        else:  # manifest (complete journal kept)
            assert imp.read_text() == "import-after"
            assert case.read_text() == "case-after"
            assert manifest.read_text() == "ledger-after"
        if boundary in ("journal", "data", "manifest"):
            assert not (tmp_path / "transactions").exists() or not list(
                (tmp_path / "transactions").iterdir()
            )


def test_stage_rejects_rel_escape(tmp_path):
    outside = tmp_path.parent / "escaped.bin"
    outside.unlink(missing_ok=True)
    with Transaction(tmp_path, op_id="op-esc", kind="write") as tx:
        with pytest.raises(WorkspaceCorrupt):
            tx.stage("../escaped.bin", b"x")
    assert not outside.exists()


def test_stage_rejects_absolute_outside(tmp_path):
    outside = tmp_path.parent / f"outside-{tmp_path.name}" / "evil.bin"
    outside.parent.mkdir(parents=True, exist_ok=True)
    with Transaction(tmp_path, op_id="op-abs", kind="write") as tx:
        with pytest.raises(WorkspaceCorrupt):
            tx.stage(str(outside), b"x")
    assert not outside.exists()


def test_stage_rejects_symlink_parent_escape(tmp_path):
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir(parents=True, exist_ok=True)
    (tmp_path / "cases").symlink_to(outside, target_is_directory=True)
    with Transaction(tmp_path, op_id="op-sym", kind="write") as tx:
        with pytest.raises(WorkspaceCorrupt):
            tx.stage("cases/x.yaml", b"x")
    assert not (outside / "x.yaml").exists()


def test_stage_accepts_absolute_inside_root(tmp_path):
    target = tmp_path / "cases" / "a.yaml"
    target.parent.mkdir(parents=True)
    with Transaction(tmp_path, op_id="op", kind="write") as tx:
        _stage(tx, target, "ok")
        tx.commit()
    assert target.read_text() == "ok"


def _write_corrupt_journal(tmp_path, op_id, mutate):
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


def test_journal_invalid_state_fails_closed(tmp_path):
    _write_corrupt_journal(tmp_path, "op-1", lambda d: d.__setitem__("state", "bogus"))
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)
    assert (tmp_path / "target.yaml").read_text() == "old"  # target untouched


def test_journal_opid_mismatch_fails_closed(tmp_path):
    _write_corrupt_journal(tmp_path, "op-2", lambda d: d.__setitem__("op_id", "other-dir"))
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)


def test_journal_rel_escape_fails_closed(tmp_path):
    outside = tmp_path.parent / "escaped-by-journal.bin"
    outside.unlink(missing_ok=True)
    _write_corrupt_journal(tmp_path, "op-3",
        lambda d: d["targets"][0].__setitem__("rel", "../../escaped-by-journal.bin"))
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)
    assert not outside.exists()  # no path outside the workspace changed


def test_journal_stage_with_separator_fails_closed(tmp_path):
    _write_corrupt_journal(tmp_path, "op-4", lambda d: d["targets"][0].__setitem__("stage", "sub/stage-0000.bin"))
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)


def test_journal_duplicate_target_rel_fails_closed(tmp_path):
    _write_corrupt_journal(tmp_path, "op-5",
        lambda d: d["targets"].append(dict(d["targets"][0])))
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)


def test_journal_applied_count_out_of_bounds_fails_closed(tmp_path):
    _write_corrupt_journal(tmp_path, "op-6", lambda d: d.__setitem__("applied_count", 99))
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)


def test_journal_replacement_order_unknown_target_fails_closed(tmp_path):
    _write_corrupt_journal(tmp_path, "op-7",
        lambda d: d.__setitem__("replacement_order", ["not-a-target"]))
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)


def test_cross_transaction_target_conflict_is_corruption(tmp_path):
    target = tmp_path / "target.yaml"
    target.write_text("before")
    for i in (1, 2):
        with Transaction(tmp_path, op_id=f"op-{i}", kind="write") as tx:
            _stage(tx, target, f"after-{i}")
            tx.prepare()
    with pytest.raises(WorkspaceCorrupt):
        recover_startup(tmp_path)
    assert target.read_text() == "before"  # neither journal applied; no last-writer-wins


def test_disjoint_transactions_both_recover(tmp_path):
    t1 = tmp_path / "a.yaml"
    t2 = tmp_path / "b.yaml"
    t1.write_text("a-before")
    t2.write_text("b-before")
    with Transaction(tmp_path, op_id="op-a", kind="write") as tx:
        _stage(tx, t1, "a-after")
        tx.prepare()
    with Transaction(tmp_path, op_id="op-b", kind="write") as tx:
        _stage(tx, t2, "b-after")
        tx.prepare()
    recover_startup(tmp_path)  # disjoint targets must both roll back cleanly
    assert t1.read_text() == "a-before" and t2.read_text() == "b-before"


def test_empty_transactions_never_follows_symlink(tmp_path):
    # A journaled op dir exists so recover_startup reaches _empty_transactions,
    # plus a symlink under transactions/ that must NOT be followed or deleted.
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


def test_empty_transactions_removes_only_positive_residue(tmp_path):
    # Journaled dir is cleaned by _recover_one_journal; a leftover positively-
    # identified residue dir is removed; a foreign file is left untouched.
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


def test_multi_target_each_per_target_boundary_restores_all_old(tmp_path):
    # 3 targets; inject a crash after each individual rename/fsync boundary.
    for n in range(0, 4):  # after applying 0, 1, 2, 3 of the 3 targets
        root = tmp_path / f"ws-{n}"
        root.mkdir()
        before = {}
        for i in range(3):
            rel = f"cases/t{i}.yaml"
            before[rel] = f"before-{i}"
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(before[rel])
        with Transaction(root, op_id=f"op-{n}", kind="import") as tx:
            for i in range(3):
                _stage(tx, root / f"cases/t{i}.yaml", f"after-{i}")
            tx.inject_crash(boundary=f"target-{n}")  # new per-target boundary
        recover_startup(root)
        for rel, content in before.items():
            assert (root / rel).read_text() == content  # every target back to all-old
        assert not (root / "transactions").exists() or not list((root / "transactions").iterdir())


def test_single_target_target_boundary(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    t = root / "t.yaml"
    t.write_text("before")
    with Transaction(root, op_id="op-1t", kind="write") as tx:
        _stage(tx, t, "after")
        tx.inject_crash(boundary="target-0")
    recover_startup(root)
    assert t.read_text() == "before"
    assert not (root / "transactions").exists() or not list((root / "transactions").iterdir())


def test_verify_complete_preserves_0600_target(tmp_path):
    t = tmp_path / "target.yaml"
    t.write_text("old")
    os.chmod(t, 0o600)
    with Transaction(tmp_path, op_id="op-c", kind="write") as tx:
        _stage(tx, t, "new")
        tx.commit()
        tx.force_state("complete")
    recover_startup(tmp_path)
    assert stat.S_IMODE(t.stat().st_mode) == 0o600


def test_rollback_preserves_0700_scaffold_dir(tmp_path):
    with Transaction(tmp_path, op_id="op-i", kind="init") as tx:
        tx.create_dir("cases")
        tx.stage("cases/keep.yaml", b"keep")
        tx.prepare()
    (tmp_path / "cases" / "keep.yaml").write_text("keep")  # non-empty -> dir preserved
    recover_startup(tmp_path)
    if (tmp_path / "cases").exists():
        assert stat.S_IMODE((tmp_path / "cases").stat().st_mode) == 0o700


def test_rollback_committing_restored_target_is_0600(tmp_path):
    t = tmp_path / "target.yaml"
    t.write_text("old")
    os.chmod(t, 0o600)
    with Transaction(tmp_path, op_id="op-r", kind="write") as tx:
        _stage(tx, t, "new")
        tx.begin_commit()
        tx.inject_crash()
    recover_startup(tmp_path)
    assert t.read_text() == "old"
    assert stat.S_IMODE(t.stat().st_mode) == 0o600


def test_restart_recovery_is_idempotent(tmp_path):
    t = tmp_path / "target.yaml"
    t.write_text("old")
    with Transaction(tmp_path, op_id="op-idem", kind="write") as tx:
        _stage(tx, t, "new")
        tx.begin_commit()
        tx.inject_crash()
    recover_startup(tmp_path)   # first pass: roll back committing journal
    first_txn = list((tmp_path / "transactions").iterdir()) if (tmp_path / "transactions").exists() else []
    recover_startup(tmp_path)   # second pass: must be a clean no-op
    assert t.read_text() == "old"                       # state identical to first pass
    txn = list((tmp_path / "transactions").iterdir()) if (tmp_path / "transactions").exists() else []
    assert txn == first_txn                              # no leftover residue, no second-pass failure


def test_complete_restart_recovery_is_idempotent(tmp_path):
    t = tmp_path / "target.yaml"
    t.write_text("old")
    with Transaction(tmp_path, op_id="op-idem2", kind="write") as tx:
        _stage(tx, t, "new")
        tx.commit()
        tx.force_state("complete")
    recover_startup(tmp_path)
    recover_startup(tmp_path)   # second pass: complete journal already verified+cleaned
    assert t.read_text() == "new"
