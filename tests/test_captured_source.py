"""Real Git/filesystem proofs at the accepted-source owner.

Snapshot tests cover copy independence and index preservation. These cases add
net source identity, stable acceptance and later integrity, which a copy alone
cannot establish; source consumers exercise dispatch and coverage separately.
"""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream.git_ops import immutable, process, queries
from daydream.git_ops.captured_source import capture_source
from daydream.git_ops.models import GitError, PathAbsentError, SnapshotPreparationError
from daydream.git_ops.source import FrozenSourceRejected
from tests.harness.git_helpers import commit, git, init_repo, write_and_stage


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "source"
    init_repo(repo)
    write_and_stage(repo, "subject.py", "value = 0\n")
    write_and_stage(repo, "helper.py", "helper = 0\n")
    write_and_stage(repo, "unchanged/helper.py", "support = 0\n")
    write_and_stage(repo, ".gitignore", "ignored/\n")
    base = commit(repo, "base")
    write_and_stage(repo, "subject.py", "value = 1\n")
    head = commit(repo, "feature")
    return repo, base, head


def test_capture_net_final_bytes_and_population_identity(source: tuple[Path, str, str], tmp_path: Path) -> None:
    """Staging cannot preserve an undone change; dirty excluded helpers bind native context."""
    repo, base, head = source
    write_and_stage(repo, "subject.py", "value = 2\n")
    (repo / "subject.py").write_text("value = 0\n")
    original_index = (repo / ".git/index").read_bytes()
    empty = capture_source(repo, tmp_path / "empty", base_sha=base, head_sha=head, local=True)
    assert empty.diff == ""
    assert empty.files == ()
    (repo / "helper.py").write_text("helper = 2\n")
    (repo / "scratch.py").write_text("scratch = 3\n")
    (repo / "ignored").mkdir()
    (repo / "ignored/secret.py").write_text("excluded = 4\n")
    (repo / ".daydream").mkdir()
    (repo / ".daydream/old.json").write_text("private evidence")
    (repo / ".review-output.md").write_text("old report")
    first = capture_source(repo, tmp_path / "first", base_sha=base, head_sha=head, local=True, exclude=["helper.py"])
    assert first.head_sha == head
    assert first.files == ("scratch.py",)
    assert first.read("new", "subject.py")[1] == b"value = 0\n"
    assert first.read("new", "helper.py")[1] == b"helper = 2\n"
    assert first.read("new", "scratch.py")[1] == b"scratch = 3\n"
    assert not (first.root / "ignored").exists()
    assert not (first.root / ".daydream").exists()
    assert not (first.root / ".review-output.md").exists()
    with pytest.raises(PathAbsentError):
        first.read("old", "scratch.py")
    (repo / "helper.py").write_text("helper = 9\n")
    second = capture_source(repo, tmp_path / "second", base_sha=base, head_sha=head, local=True, exclude=["helper.py"])
    assert first.diff == second.diff
    assert first.source_id != second.source_id
    assert first.read("new", "helper.py")[1] == b"helper = 2\n"
    assert first.revision_for("new") == first.source_id
    assert first.revision_for("old") == base
    assert (repo / ".git/index").read_bytes() == original_index
    assert git(repo, "rev-parse", "HEAD") == head
    population = capture_source(repo, tmp_path / "population", base_sha=base, head_sha=head, local=True,
                                exclude=["helper.py", "scratch.py", "other.txt"])
    assert population.source_id != second.source_id
    (repo / "other.txt").write_text("other")
    added = capture_source(repo, tmp_path / "added", base_sha=base, head_sha=head, local=True,
                           exclude=["helper.py", "scratch.py", "other.txt"])
    assert added.diff == population.diff == ""
    assert added.source_id != population.source_id


@pytest.mark.parametrize(("path", "edited"), [
    ("subject.py", False), ("subject.py", True), ("added.py", False), ("added.bin", False),
], ids=["staged-final", "edited-after-staging", "added-staged", "large-staged"])
def test_capture_uses_working_bytes_after_staging(
    source: tuple[Path, str, str], tmp_path: Path, path: str, edited: bool,
) -> None:
    """Final bytes may reuse staged loose blobs; later edits still supersede the index.

    The edited-after-staging row misses reuse of a copied non-endpoint object.
    A large compressed staged blob also exercises incremental existing-object reads.
    """
    repo, base, head = source
    staged = b"\0" + b"x" * (2 * 1024**2 + 17) if path.endswith(".bin") else b"value = 2\n"
    working = b"value = 3\n" if edited else staged
    write_and_stage(repo, path, staged)
    (repo / path).write_bytes(working)
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True)
    assert path in captured.files
    if not path.endswith(".bin"):
        assert "+" + working.decode() in captured.diff
    if edited:
        assert "+value = 2\n" not in captured.diff
    if path == "subject.py":
        assert captured.read("old", path)[1] == b"value = 0\n"
    else:
        with pytest.raises(PathAbsentError):
            captured.read("old", path)
    assert captured.read("new", path)[1] == working
    assert (captured.root / path).read_bytes() == working


def test_committed_capture_ignores_original_checkout_bytes(source: tuple[Path, str, str], tmp_path: Path) -> None:
    """Pinned comments inspect raw PR objects despite dirtiness or another checked-out HEAD."""
    repo, base, head = source
    git(repo, "checkout", "--detach", base)
    (repo / "subject.py").write_bytes(b"dirty\r\n")
    (repo / "scratch.py").write_text("local only")
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=False)
    assert captured.head_sha == head
    assert captured.source_id is None
    assert captured.revision_for("new") == head
    assert captured.files == ("subject.py",)
    assert captured.read("new", "subject.py")[1] == b"value = 1\n"
    assert not (captured.root / "scratch.py").exists()
    assert (repo / "subject.py").read_bytes() == b"dirty\r\n"
    assert git(repo, "rev-parse", "HEAD") == base
    shutil.rmtree(repo)
    captured.validate()
    assert captured.read("old", "subject.py")[1] == b"value = 0\n"


@pytest.mark.parametrize("change", ["bytes", "parent", "symlink", "missing"])
@pytest.mark.parametrize("persistent", [False, True])
def test_capture_retries_actual_materialization_instability(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool,
    change: str,
) -> None:
    """Concurrent writes around actual copying retry finitely and never accept torn source."""
    repo, base, head = source
    destination = tmp_path / "captured"
    relative = "unchanged/helper.py" if change == "parent" else "subject.py"
    changed = "helper.py" if change in {"symlink", "missing"} else relative
    if change in {"symlink", "missing"}:
        (repo / changed).unlink()
        if change == "symlink":
            (repo / changed).symlink_to("subject.py")
    original = os.fdopen
    writes = 0

    def changing_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        stream = original(fd, *args, **kwargs)
        write = stream.write

        def changing_write(body: bytes) -> int:
            nonlocal writes
            written = int(write(body))
            target = destination / relative
            if target.exists() and os.fstat(fd).st_ino == target.stat().st_ino:
                writes += 1
                if persistent or writes == 1:
                    if change == "parent":
                        (repo / "unchanged").rename(tmp_path / f"prior-parent-{writes}")
                        (repo / "unchanged").mkdir()
                    if change in {"symlink", "missing"}:
                        (repo / changed).unlink(missing_ok=True)
                    (repo / changed).write_bytes(f"value = {writes + 10}\n".encode())
            return written

        if args and args[0] == "wb":
            stream.write = changing_write
        return stream

    monkeypatch.setattr(os, "fdopen", changing_fdopen)
    if persistent:
        with pytest.raises(SnapshotPreparationError, match="stop concurrent writers and rerun"):
            capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
        assert writes == 3
        assert not destination.exists()
    else:
        captured = capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
        assert writes == 2
        assert captured.read("new", changed)[1] == b"value = 11\n"


@pytest.mark.parametrize("fault", ["missing-spool", "cancel-replaced-storage", "spool-disappears"])
def test_capture_private_write_failure_preserves_subject_and_primary_error(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Private write failure cannot turn a present source into an accepted deletion or erase cancellation."""
    import asyncio

    repo, base, head = source
    destination = tmp_path / "captured"
    expected_body = b"value = 2\n" if fault == "spool-disappears" else b"value = 1\n"
    (repo / "subject.py").write_bytes(expected_body)
    original = os.open
    writes = 0

    def failing_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal writes
        if path == "subject.py" and flags & os.O_CREAT:
            writes += 1
            if fault == "cancel-replaced-storage":
                (destination / ".git/objects").rename(destination / ".git/preserved-objects")
                raise asyncio.CancelledError("cancel during private write")
            if fault != "spool-disappears":
                raise FileNotFoundError("private spool disappeared")
        if fault == "spool-disappears" and isinstance(path, str) and "." in path and len(path) > 65:
            (destination / "subject.py").rename(destination / "preserved-subject.py")
            (destination / "witness").write_bytes(b"competing bytes")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", failing_open)
    expected = (asyncio.CancelledError if fault == "cancel-replaced-storage" else
                GitError if fault == "spool-disappears" else SnapshotPreparationError)
    with pytest.raises(expected):
        capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
    assert writes == (3 if fault == "missing-spool" else 1)
    assert (repo / "subject.py").read_bytes() == expected_body
    assert destination.exists() == (fault != "missing-spool")
    if fault == "spool-disappears":
        assert (destination / "preserved-subject.py").read_bytes() == expected_body
        assert (destination / "witness").read_bytes() == b"competing bytes"


def test_capture_retries_object_population_change_during_copy(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physical object mutation during copying retries before any source is accepted."""
    repo, base, head = source
    original = immutable._copy_objects
    copies = 0

    def changing_copy(common: Path, target: Path) -> None:
        nonlocal copies
        original(common, target)
        copies += 1
        if copies == 1:
            # Mutate physical object population after its copy to exercise the
            # source-object copy boundary, independently of working-file reads.
            evidence = tmp_path / "new-object.txt"
            evidence.write_text("concurrent object")
            git(repo, "hash-object", "-w", "--no-filters", "--", str(evidence))

    monkeypatch.setattr(immutable, "_copy_objects", changing_copy)
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True)
    assert copies == 2
    assert captured.read("new", "subject.py")[1] == b"value = 1\n"


def test_capture_retries_working_population_change_during_enumeration(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file appearing after real enumeration must join accepted targets and native search."""
    repo, base, head = source
    enumerate_untracked = queries._list_untracked_z
    first = True

    def growing_population(root: Path) -> list[str]:
        nonlocal first
        paths = enumerate_untracked(root)
        if first:
            first = False
            (root / "appeared.py").write_text("appeared = 1\n")
        return paths

    monkeypatch.setattr(queries, "_list_untracked_z", growing_population)
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True)
    assert set(captured.files) == {"appeared.py", "subject.py"}
    assert captured.read("new", "appeared.py")[1] == b"appeared = 1\n"


@pytest.mark.parametrize("corruption", [
    "bytes", "missing", "population", "population-exclude", "population-gitignore", "population-link", "metadata",
    "storage", "index", "skip", "assume", "stages", "intent", "support-bytes", "support-mode", "support-missing",
    "support-parent",
])
def test_capture_integrity_never_falls_back_to_original(
    source: tuple[Path, str, str], tmp_path: Path, corruption: str,
) -> None:
    """Required damaged capture evidence stays a failure even while original bytes remain available."""
    repo, base, head = source
    if corruption == "intent":
        write_and_stage(repo, "empty.bin", b"")
        head = commit(repo, "empty file")
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True,
                              exclude=["unchanged"])
    captured.validate()  # A preceding success must never authorize later corruption.
    if corruption.startswith("support-"):
        helper = captured.root / "unchanged/helper.py"
        if corruption == "support-bytes":
            info = helper.stat()
            helper.write_bytes(b"support = 9\n")
            os.utime(helper, ns=(info.st_atime_ns, info.st_mtime_ns))
        elif corruption == "support-mode":
            helper.chmod(0o755)
        elif corruption == "support-missing":
            helper.unlink()
        else:
            (captured.root / "unchanged").rename(captured.root / "moved-support")
            (captured.root / "unchanged").symlink_to(repo / "unchanged", target_is_directory=True)
    elif corruption == "bytes":
        (captured.root / "subject.py").write_text("tampered")
    elif corruption == "missing":
        (captured.root / "subject.py").unlink()
    elif corruption == "population":
        (captured.root / "injected.py").write_text("tampered")
    elif corruption == "population-exclude":
        (captured.root / ".git/info").mkdir()
        (captured.root / ".git/info/exclude").write_text("injected.py\n")
        (captured.root / "injected.py").write_text("tampered")
    elif corruption == "population-gitignore":
        (captured.root / "generated").mkdir()
        (captured.root / "generated/.gitignore").write_text("*\n")
        (captured.root / "generated/injected.py").write_text("tampered")
    elif corruption == "population-link":
        (captured.root / "injected.py").symlink_to(captured.root / "subject.py")
    elif corruption == "metadata":
        (captured.root / ".git/HEAD").write_text(base + "\n")
    elif corruption == "index":
        old_oid = git(captured.root, "rev-parse", f"{base}:subject.py")
        git(captured.root, "update-index", "--cacheinfo", f"100644,{old_oid},subject.py")
    elif corruption in {"skip", "assume"}:
        flag = "--skip-worktree" if corruption == "skip" else "--assume-unchanged"
        git(captured.root, "update-index", flag, "subject.py")
    elif corruption == "stages":
        oid = git(captured.root, "rev-parse", "HEAD:subject.py")
        git(captured.root, "update-index", "--force-remove", "subject.py")
        process._run_git(captured.root, ["update-index", "--index-info"], retries=0,
                         input_text=f"100644 {oid} 1\tsubject.py\n", error_context="test index stages failed")
    elif corruption == "intent":
        inventory = git(captured.root, "ls-files", "-t", "-v", "--stage", "-z")
        git(captured.root, "update-index", "--force-remove", "empty.bin")
        git(captured.root, "add", "--intent-to-add", "--", "empty.bin")
        assert git(captured.root, "ls-files", "-t", "-v", "--stage", "-z") == inventory
    else:
        old_objects = captured.root / ".git/old-objects"
        (captured.root / ".git/objects").rename(old_objects)
        shutil.copytree(old_objects, captured.root / ".git/objects")
    with pytest.raises(GitError) as failure:
        captured.validate()
    if corruption.startswith("population"):
        assert ("generated/.gitignore" if corruption == "population-gitignore" else "injected.py") in str(failure.value)
    elif not corruption.startswith("support-"):
        with pytest.raises(GitError):
            captured.read("new", "subject.py")
    assert (repo / "subject.py").read_bytes() == b"value = 1\n"


def test_ignored_check_output_leaves_accepted_source_valid(tmp_path: Path) -> None:
    """A reviewer's real test run writes the ignored caches capture itself would exclude."""
    repo = tmp_path / "source"
    init_repo(repo)
    write_and_stage(repo, ".gitignore", "__pycache__/\n.pytest_cache/\nbuild/\n")
    write_and_stage(repo, "tests/test_subject.py", "def test_subject() -> None:\n    assert True\n")
    base = commit(repo, "base")
    write_and_stage(repo, "subject.py", "value = 1\n")
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=commit(repo, "feature"), local=True)
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONDONTWRITEBYTECODE"}
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=captured.root, env=environment, check=True,
                   capture_output=True)
    assert list(captured.root.glob("tests/__pycache__/*.pyc"))
    assert (captured.root / ".pytest_cache/.gitignore").is_file()
    (captured.root / "build").mkdir()
    (captured.root / "build/subject.py").symlink_to(captured.root / "subject.py")
    captured.validate()
    assert captured.read("new", "subject.py")[1] == b"value = 1\n"


def test_capture_preserves_metadata_without_native_symlink_or_gitlink_expansion(
    source: tuple[Path, str, str], tmp_path: Path,
) -> None:
    """Working gitlink HEAD, unlike its index entry, binds metadata without exposing nested contents."""
    repo, _, _ = source
    module_origin = tmp_path / "module-origin"
    init_repo(module_origin)
    write_and_stage(module_origin, "secret.py", "nested = 0\n")
    module_base = commit(module_origin, "module base")
    write_and_stage(module_origin, "secret.py", "nested = 1\n")
    module_head = commit(module_origin, "module feature")
    git(repo, "-c", "protocol.file.allow=always", "submodule", "add", "--", str(module_origin), "module")
    git(repo / "module", "checkout", "--detach", module_base)
    git(repo, "add", "module")
    write_and_stage(repo, ".gitignore", "ignored/\nlinked.py\n")
    base = commit(repo, "gitlink base")
    write_and_stage(repo, "subject.py", "value = 2\n")
    head = commit(repo, "feature after gitlink base")
    (repo / "linked.py").symlink_to(tmp_path / "private.py")
    (tmp_path / "private.py").write_text("outside")
    (repo / ".daydream").symlink_to(tmp_path / "private.py")
    git(repo, "add", "--force", "linked.py")
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{module_base},uninitialized")
    (repo / "uninitialized").mkdir()
    (repo / "uninitialized/secret.py").write_text("uninitialized nested bytes")
    unchanged = capture_source(repo, tmp_path / "unchanged", base_sha=base, head_sha=head, local=True)
    assert set(unchanged.files) == {"linked.py", "subject.py", "uninitialized"}
    git(repo / "module", "checkout", "--detach", module_head)
    (repo / "module/secret.py").write_text("dirty nested bytes")
    original_index = (repo / ".git/index").read_bytes()
    original_refs = git(repo, "show-ref")
    module_index = (repo / ".git/modules/module/index").read_bytes()
    module_refs = git(repo / "module", "show-ref")
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True)
    assert set(captured.files) == {"linked.py", "module", "subject.py", "uninitialized"}
    assert f"-Subproject commit {module_base}\n+Subproject commit {module_head}\n" in captured.diff
    assert captured.source_id != unchanged.source_id
    assert not (captured.root / ".daydream").exists()
    assert not (captured.root / "linked.py").exists()
    assert not (captured.root / "module").exists()
    assert not (captured.root / "uninitialized").exists()
    assert "120000" in git(captured.root, "ls-tree", "HEAD", "linked.py")
    assert f"160000 commit {module_head}" in git(captured.root, "ls-tree", "HEAD", "module")
    assert f"160000 commit {module_base}" in git(captured.root, "ls-tree", "HEAD", "uninitialized")
    for path in ("linked.py", "module", "uninitialized", "../private.py"):
        with pytest.raises(FrozenSourceRejected):
            captured.read("new", path)
    with pytest.raises(FrozenSourceRejected):
        captured.read("old", "module")
    captured.validate()
    (captured.root / "linked.py").mkdir()  # Ignore rules cannot hide files beneath accepted metadata paths.
    (captured.root / "linked.py/injected.py").write_text("shadow")
    with pytest.raises(GitError):
        captured.validate()
    shutil.rmtree(captured.root / "linked.py")
    (captured.root / "linked.py").write_text("shadow")  # Nor the accepted metadata path itself.
    with pytest.raises(GitError):
        captured.validate()
    assert git(repo, "rev-parse", "HEAD") == head
    assert git(repo, "show-ref") == original_refs
    assert (repo / ".git/index").read_bytes() == original_index
    assert git(repo / "module", "rev-parse", "HEAD") == module_head
    assert git(repo / "module", "show-ref") == module_refs
    assert (repo / ".git/modules/module/index").read_bytes() == module_index
    assert (repo / "module/secret.py").read_text() == "dirty nested bytes"
    assert (repo / "uninitialized/secret.py").read_text() == "uninitialized nested bytes"
    (repo / ".daydream").unlink()
    (repo / ".daydream").symlink_to(tmp_path / "different-private-target")
    repeated = capture_source(repo, tmp_path / "repeated", base_sha=base, head_sha=head, local=True)
    assert repeated.source_id == captured.source_id


def test_capture_raw_duplicate_and_empty_blobs(source: tuple[Path, str, str], tmp_path: Path) -> None:
    """Raw batching preserves binary duplicate and zero-length payload boundaries."""
    repo, base, _ = source
    payload = b"\0duplicate\xff\n" * 100_003
    write_and_stage(repo, "a.bin", payload)
    write_and_stage(repo, "b.bin", payload)
    write_and_stage(repo, "empty.bin", b"")
    head = commit(repo, "raw files")
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=False)
    for name in ("a.bin", "b.bin"):
        assert (captured.root / name).read_bytes() == payload
        assert captured.read("new", name)[1] == payload
    assert (captured.root / "empty.bin").read_bytes() == b""
    captured.validate()


def test_capture_rename_deletion_binary_and_mode_targets(source: tuple[Path, str, str], tmp_path: Path) -> None:
    """Full native diff metadata survives independently of whether text is reviewable."""
    repo, base, head = source
    (repo / "renamed.py").write_bytes((repo / "subject.py").read_bytes())
    (repo / "subject.py").unlink()
    (repo / "helper.py").unlink()
    (repo / "binary.dat").write_bytes(b"\0binary\xff")
    (repo / ".gitignore").chmod(0o755)
    captured = capture_source(repo, tmp_path / "captured", base_sha=head, head_sha=head, local=True)
    assert set(captured.files) == {".gitignore", "binary.dat", "helper.py", "renamed.py"}
    assert "rename from subject.py" in captured.diff
    assert "rename to renamed.py" in captured.diff
    assert "deleted file mode" in captured.diff
    assert "Binary files" in captured.diff
    assert "old mode 100644\nnew mode 100755" in captured.diff
    assert captured.read("old", "subject.py")[1] == b"value = 1\n"
    with pytest.raises(PathAbsentError):
        captured.read("new", "subject.py")


def test_capture_cleanup_preserves_replaced_storage(source: tuple[Path, str, str], tmp_path: Path) -> None:
    """Cleanup authority follows pinned private directories, not a reusable pathname."""
    repo, base, head = source
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True)
    owned = tmp_path / "owned"
    captured.root.rename(owned)
    captured.root.mkdir()
    (captured.root / "competing.txt").write_text("preserve")
    with pytest.raises(GitError):
        captured.remove()
    assert (captured.root / "competing.txt").read_text() == "preserve"
    assert (owned / "subject.py").read_text() == "value = 1\n"
    another = capture_source(repo, tmp_path / "another", base_sha=base, head_sha=head, local=True)
    (another.root / "subject.py").write_text("corrupt but owned")
    another.remove()
    assert not another.root.exists()


@pytest.mark.parametrize("state", ["cone", "noncone", "omitted", "present", "deleted", "deleted-directory"])
def test_local_capture_requires_materialized_skip_worktree_contents(
    source: tuple[Path, str, str], tmp_path: Path, state: str,
) -> None:
    """Sparse omission must fail preparation without erasing real working edits or deletions."""
    repo, _, _ = source
    (repo / "visible").mkdir()
    (repo / "support").mkdir()
    git(repo, "mv", "subject.py", "visible/subject.py")
    git(repo, "mv", "helper.py", "support/helper.py")
    base = commit(repo, "organized source")
    write_and_stage(repo, "visible/subject.py", "value = 2\n")
    head = commit(repo, "feature")
    if state == "cone":
        git(repo, "sparse-checkout", "init", "--cone", "--sparse-index")
        git(repo, "sparse-checkout", "set", "visible")
    elif state == "noncone":
        git(repo, "sparse-checkout", "set", "--no-cone", "/visible/", "/.gitignore")
    else:
        if not state.startswith("deleted"):
            git(repo, "update-index", "--skip-worktree", "support/helper.py")
        if state == "present":
            (repo / "support/helper.py").write_text("helper = 4\n")
        elif state == "deleted-directory":
            shutil.rmtree(repo / "support")
        else:
            (repo / "support/helper.py").unlink()
    flags = git(repo, "ls-files", "-t", "--", "support/helper.py")
    assert flags == ("H" if state.startswith("deleted") else "S") + " support/helper.py"
    original_index = (repo / ".git/index").read_bytes()
    original_refs = git(repo, "show-ref")
    destination = tmp_path / "captured"
    if state in {"cone", "noncone", "omitted"}:
        with pytest.raises(SnapshotPreparationError, match="skip-worktree.*materialize.*rerun"):
            capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
        assert not destination.exists()
    else:
        captured = capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
        assert set(captured.files) == {"visible/subject.py", "support/helper.py"}
        if state == "present":
            assert captured.read("new", "support/helper.py")[1] == b"helper = 4\n"
            assert (captured.root / "support/helper.py").read_bytes() == b"helper = 4\n"
        else:
            with pytest.raises(PathAbsentError):
                captured.read("new", "support/helper.py")
            assert "deleted file mode" in captured.diff
        captured.validate()
    assert (repo / ".git/index").read_bytes() == original_index
    assert git(repo, "show-ref") == original_refs
    assert (repo / "visible/subject.py").read_bytes() == b"value = 2\n"
    if state == "present":
        assert (repo / "support/helper.py").read_bytes() == b"helper = 4\n"
    else:
        assert not (repo / "support/helper.py").exists()


def test_capture_allows_native_index_stat_refresh(source: tuple[Path, str, str], tmp_path: Path) -> None:
    """Allowed native Git inspection may refresh index stat/cache bytes without changing source."""
    repo, base, head = source
    original_index = (repo / ".git/index").read_bytes()
    original_refs = git(repo, "show-ref")
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True)
    frozen_index = (captured.root / ".git/index").read_bytes()
    inventory = git(captured.root, "ls-files", "-t", "-v", "--stage", "-z")
    assert git(captured.root, "status", "--porcelain", env={"GIT_OPTIONAL_LOCKS": "1"}) == ""
    assert (captured.root / ".git/index").read_bytes() != frozen_index
    assert git(captured.root, "ls-files", "-t", "-v", "--stage", "-z") == inventory
    assert captured.read("new", "subject.py")[1] == b"value = 1\n"
    captured.validate()
    assert (repo / ".git/index").read_bytes() == original_index
    assert git(repo, "show-ref") == original_refs
    assert (repo / "subject.py").read_bytes() == b"value = 1\n"


@pytest.mark.parametrize("state", ["removed-ignored", "removed-visible", "tracked-ignored"])
def test_capture_current_population_excludes_only_untracked_ignored_bytes(
    source: tuple[Path, str, str], tmp_path: Path, state: str,
) -> None:
    """Index removal revokes ignored after-content without losing old-side evidence or eligible working files."""
    repo, base, head = source
    if state != "removed-visible":
        (repo / ".gitignore").write_text("helper.py\n")
    before = capture_source(repo, tmp_path / "before", base_sha=base, head_sha=head, local=True)
    if state != "tracked-ignored":
        git(repo, "rm", "--cached", "--", "helper.py")
    original_index = (repo / ".git/index").read_bytes()
    original_refs = git(repo, "show-ref")
    unchanged = capture_source(repo, tmp_path / "unchanged", base_sha=base, head_sha=head, local=True)
    if state == "removed-ignored":
        assert unchanged.source_id != before.source_id
        assert "helper.py" in unchanged.files
        assert "diff --git a/helper.py b/helper.py\ndeleted file mode 100644\n" in unchanged.diff
        assert unchanged.read("old", "helper.py")[1] == b"helper = 0\n"
        with pytest.raises(PathAbsentError):
            unchanged.read("new", "helper.py")
        assert not (unchanged.root / "helper.py").exists()
    else:
        assert unchanged.source_id == before.source_id
        assert unchanged.diff == before.diff
        assert unchanged.read("new", "helper.py")[1] == b"helper = 0\n"
    (repo / "helper.py").write_text("helper = 4\n")
    dirty = capture_source(repo, tmp_path / "dirty", base_sha=base, head_sha=head, local=True)
    if state == "removed-ignored":
        assert dirty.source_id == unchanged.source_id
        assert dirty.diff == unchanged.diff
        with pytest.raises(PathAbsentError):
            dirty.read("new", "helper.py")
        assert not (dirty.root / "helper.py").exists()
    else:
        assert dirty.source_id != unchanged.source_id
        assert "helper.py" in dirty.files
        assert dirty.read("new", "helper.py")[1] == b"helper = 4\n"
        assert (dirty.root / "helper.py").read_bytes() == b"helper = 4\n"
    assert (repo / ".git/index").read_bytes() == original_index
    assert git(repo, "show-ref") == original_refs
    assert git(repo, "rev-parse", "HEAD") == head
    assert (repo / "helper.py").read_bytes() == b"helper = 4\n"

@pytest.mark.parametrize("fault", [
    "valid", "missing", "wrong-oid", "wrong-type", "truncated", "terminator", "extra", "nonzero",
    "stderr", "timeout", "setup", "sink", "cancel",
])
def test_streamed_blob_protocol_reaps_child_on_every_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Real duplex pipes prove framing, stderr/deadline bounds and cleanup even before reader setup completes.

    Existing buffered subprocess tests have no request writes, object framing or
    failing consumer, so cannot catch these lifecycle/duplex regressions.
    """
    import asyncio
    import hashlib
    import subprocess
    import sys

    from daydream.git_ops.captured_source import _BlobReader

    body = b"raw\r\n\0\xff" * 20_003
    oid = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    script = (
        "import os,sys,time\n"
        "fault=sys.argv[1]\n"
        "oid=sys.stdin.buffer.readline().strip()\n"
        "body=b'raw\\r\\n\\0\\xff'*20003\n"
        "if fault=='timeout': time.sleep(60)\n"
        "if fault=='stderr': sys.stderr.buffer.write(b'x'*65537);sys.stderr.buffer.flush()\n"
        "header=oid+b' blob '+str(len(body)).encode()+b'\\n'\n"
        "if fault=='missing': header=oid+b' missing\\n'\n"
        "if fault=='wrong-oid': header=b'0'*40+b' blob '+str(len(body)).encode()+b'\\n'\n"
        "if fault=='wrong-type': header=oid+b' tree '+str(len(body)).encode()+b'\\n'\n"
        "for part in (header[:7],header[7:]):\n"
        " sys.stdout.buffer.write(part);sys.stdout.buffer.flush()\n"
        "if fault=='truncated': sys.stdout.buffer.write(body[:30]);sys.exit(0)\n"
        "sys.stdout.buffer.write(body);sys.stdout.buffer.write(b'x' if fault=='terminator' else b'\\n')\n"
        "if fault=='extra': sys.stdout.buffer.write(b'extra')\n"
        "sys.stdout.buffer.flush()\n"
        "if fault=='nonzero': sys.exit(7)\n"
        "sys.stdin.buffer.read()\n"
    )
    children: list[subprocess.Popen[Any]] = []
    launch = subprocess.Popen

    def launch_child(argv: Any, *args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        child = launch([sys.executable, "-I", "-c", script, fault], *args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch_child)
    if fault == "setup":
        def unavailable_nonblocking(fd: int, blocking: bool) -> None:
            raise RuntimeError("pipe setup failed")
        monkeypatch.setattr(os, "set_blocking", unavailable_nonblocking)
    sink = tmp_path / "accepted-bytes"
    def consume(chunk: bytes) -> None:
        with sink.open("ab") as output:
            output.write(chunk)
        if fault == "sink":
            raise OSError("destination write failed")
        if fault == "cancel":
            raise asyncio.CancelledError("cancel during actual write")

    def read() -> None:
        with process._stream_git(tmp_path, ["cat-file", "--batch"],
                                 timeout=0.2 if fault == "timeout" else 5) as transport:
            _BlobReader(transport, "sha1", total=len(body), count=1).read(oid, consume)

    if fault == "valid":
        read()
        assert sink.read_bytes() == body
    else:
        expected = (asyncio.CancelledError if fault == "cancel" else RuntimeError if fault == "setup"
                    else GitError)
        with pytest.raises(expected):
            read()
    assert len(children) == 1
    child = children[0]
    assert child.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(child.pid, 0)
    assert all(stream is not None and stream.closed for stream in (child.stdin, child.stdout, child.stderr))


def test_capture_verifies_both_separately_admitted_sides(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two legal endpoint populations may exceed a single-side payload budget together."""
    repo, _, _ = source
    old, new = b"\0" + b"a" * 750, b"\0" + b"b" * 750
    write_and_stage(repo, "both.bin", old)
    base = commit(repo, "old binary")
    write_and_stage(repo, "both.bin", new)
    head = commit(repo, "new binary")
    monkeypatch.setattr(immutable, "_MAX_SOURCE_BYTES", 1024)
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=False)
    captured.validate()
    assert captured.read("old", "both.bin")[1] == old
    assert captured.read("new", "both.bin")[1] == new


@pytest.mark.parametrize("kind", [
    "blob", "tree", "existing-tree", "native", "parent", "parent-open", "parent-create",
    "parent-source", "parent-cancel",
    "leaf-source", "leaf-cancel", "read-truncated", "read-cancel", "read-symlink", "read-missing",
])
def test_capture_preserves_competing_object_bytes(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    """Loose-object publication must neither replace a competing leaf nor erase it during refusal cleanup."""
    import asyncio

    repo, base, head = source
    expected_source = "value = 1\n" if kind == "existing-tree" else "value = 345\n"
    (repo / "subject.py").write_text(expected_source)
    private_io = kind in {
        "parent-source", "parent-cancel", "leaf-source", "leaf-cancel", "read-truncated", "read-cancel",
        "read-symlink", "read-missing",
    }
    if kind == "parent-create" or private_io:
        (repo / "fresh").mkdir()
        (repo / "fresh/new.py").write_text(expected_source)
    tree_oid = git(repo, "rev-parse", "HEAD^{tree}")
    destination = tmp_path / "captured"
    expected = ((repo / ".git/objects" / tree_oid[:2] / tree_oid[2:]).read_bytes()
                if kind == "existing-tree" else b"competing object bytes")
    link = os.link
    competing: list[Path] = []
    create_failures: list[FileNotFoundError] = []
    cancellation = asyncio.CancelledError("cancel during actual private write")

    def preserve_parent(parent: Path) -> None:
        preserved = parent.with_name("preserved-parent")
        parent.rename(preserved)
        witness = preserved / "witness"
        witness.write_bytes(expected)
        competing.append(witness)

    def replace_leaf(leaf: Path) -> Path:
        prior = leaf.with_name("prior-new.py")
        leaf.rename(prior)
        leaf.write_bytes(expected)
        competing.append(leaf)
        return prior

    def competing_link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        parent = next((destination / ".git/objects").glob("*/" + str(src)))
        if kind == "parent":
            link(src, dst, *args, **kwargs)
            if not competing:
                preserve_parent(parent.parent)
            return
        target = parent.with_name(str(dst))
        target.write_bytes(b"competing object bytes")
        competing.append(target)
        link(src, dst, *args, **kwargs)

    opened = os.open
    fdopen = os.fdopen
    stat = os.stat

    def competing_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if (path == "new.py" and (destination / ".git/index").is_file() and not competing
                and kwargs.get("dir_fd") is not None
                and os.fstat(kwargs["dir_fd"]).st_ino == (destination / "fresh").stat().st_ino):
            leaf = destination / "fresh/new.py"
            leaf.rename(leaf.with_name("prior-new.py"))
            witness = leaf.with_name("witness")
            witness.write_bytes(expected)
            competing.append(witness)
            if kind == "read-symlink":
                leaf.symlink_to("witness")
        return stat(path, *args, **kwargs)

    def competing_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
        stream = fdopen(fd, *args, **kwargs)
        write = stream.write
        read = stream.read

        def competing_write(body: bytes) -> int:
            written = int(write(body))
            leaf = destination / "fresh/new.py"
            if leaf.exists() and os.fstat(fd).st_ino == leaf.stat().st_ino and not competing:
                if kind.startswith("parent"):
                    preserve_parent(leaf.parent)
                else:
                    replace_leaf(leaf)
                if kind.endswith("cancel"):
                    raise cancellation
                (repo / "fresh/new.py").write_text("value = 67890\n")
            return written

        def competing_read(size: int = -1) -> bytes:
            leaf = destination / "fresh/new.py"
            if ((destination / ".git/index").is_file() and leaf.exists()
                    and os.fstat(fd).st_ino == leaf.stat().st_ino and not competing):
                prior = replace_leaf(leaf)
                if kind == "read-truncated":
                    prior.write_bytes(b"")
                else:
                    read(size)
                    raise cancellation
            return bytes(read(size))

        if args and args[0] == "wb" and not kind.startswith("read"):
            stream.write = competing_write
        elif args and args[0] == "rb" and kind.startswith("read"):
            stream.read = competing_read
        return stream

    def competing_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if kind == "parent-create" and path == "new.py" and flags & os.O_CREAT and not competing:
            parent = destination / "fresh"
            parent.rmdir()
            parent.mkdir()
            witness = parent / "witness"
            witness.write_bytes(expected)
            competing.append(witness)
            try:
                return opened(path, flags, *args, **kwargs)
            except FileNotFoundError as exc:
                create_failures.append(exc)
                raise
        if (kind == "parent-open" and flags & os.O_DIRECTORY and isinstance(path, str) and len(path) == 2
                and (destination / "subject.py").exists() and not competing):
            preserve_parent(destination / ".git/objects" / path)
        if (kind == "existing-tree" and path == tree_oid[2:] and not flags & os.O_CREAT
                and (destination / "subject.py").exists() and not competing):
            target = destination / ".git/objects" / tree_oid[:2] / tree_oid[2:]
            target.rename(target.with_name("preserved-object"))
            target.write_bytes(expected)
            competing.append(target)
        if ((kind == "tree" and isinstance(path, str) and len(path) == 38
             and (destination / "subject.py").exists() or kind == "native" and path == "subject.py")
                and flags & os.O_CREAT):
            fd = opened(path, flags, *args, **kwargs)
            os.write(fd, b"competing object bytes")
            os.close(fd)
            if kind == "native":
                competing.append(destination / "subject.py")
            else:
                competing.extend((destination / ".git/objects").glob("*/" + path))
        return opened(path, flags, *args, **kwargs)

    if kind in {"read-symlink", "read-missing"}:
        monkeypatch.setattr(os, "stat", competing_stat)
    elif private_io:
        monkeypatch.setattr(os, "fdopen", competing_fdopen)
    elif kind in {"blob", "parent"}:
        monkeypatch.setattr(os, "link", competing_link)
    else:
        monkeypatch.setattr(os, "open", competing_open)
    failure = asyncio.CancelledError if kind.endswith("cancel") else GitError
    message = None if kind.startswith("parent") or private_io else "conflicts|replaced"
    with pytest.raises(failure, match=message) as caught:
        capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
    assert len(competing) == 1 and competing[0].read_bytes() == expected
    assert (repo / "subject.py").read_text() == expected_source
    if kind == "parent-create" or private_io:
        assert len(create_failures) == (1 if kind == "parent-create" else 0)
        assert (repo / "fresh/new.py").read_text() == (
            "value = 67890\n" if kind.endswith("source") else expected_source)
    if kind.startswith(("leaf", "read")):
        assert (destination / "fresh/prior-new.py").read_bytes() == (
            b"" if kind == "read-truncated" else expected_source.encode())
    if kind == "read-symlink":
        assert (destination / "fresh/new.py").is_symlink()
    if kind.endswith("cancel"):
        assert caught.value is cancellation


@pytest.mark.parametrize("phase", ["compression", "comparison", "comparison-stat", "compression-symlink"])
def test_capture_rejects_identical_byte_spool_replacement(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    """Equal content cannot authorize a different private spool inode during local acceptance."""
    repo, base, head = source
    expected = b"value = 4321\n" if phase.startswith("compression") else b"value = 1\n"
    (repo / "subject.py").write_bytes(expected)
    destination = tmp_path / "captured"
    opened, stat = os.open, os.stat
    replaced = comparison_started = False
    open_errors: list[int | None] = []

    def replace_private() -> None:
        nonlocal replaced
        replaced = True
        target = destination / "subject.py"
        body = target.read_bytes() if phase != "comparison-stat" else b"competing changed spool bytes"
        target.rename(destination / "prior-subject")
        if phase == "compression-symlink":
            target.symlink_to("prior-subject")
        else:
            target.write_bytes(body)

    def replace_spool(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal comparison_started
        parent = kwargs.get("dir_fd")
        if (path == "subject.py" and not flags & os.O_CREAT and parent is not None
                and os.fstat(parent).st_ino == repo.stat().st_ino and (destination / "subject.py").exists()):
            comparison_started = True
        private_parent = (parent is not None and destination.exists()
                          and os.fstat(parent).st_ino == destination.stat().st_ino)
        if (path == "subject.py" and not flags & os.O_CREAT and private_parent
                and not replaced and phase != "comparison-stat"):
            replace_private()
        try:
            return opened(path, flags, *args, **kwargs)
        except OSError as exc:
            if phase == "compression-symlink" and private_parent and path == "subject.py":
                open_errors.append(exc.errno)
            raise

    def replace_at_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        parent = kwargs.get("dir_fd")
        target = destination / "subject.py"
        private_parent = (parent is not None and destination.exists()
                          and os.fstat(parent).st_ino == destination.stat().st_ino)
        if path == "subject.py" and target.exists() and private_parent and comparison_started and not replaced:
            replace_private()
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_spool)
    if phase == "comparison-stat":
        monkeypatch.setattr(os, "stat", replace_at_stat)
    with pytest.raises(GitError, match="spool (was replaced|is unavailable)"):
        capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
    assert replaced
    assert (destination / "subject.py").read_bytes() == (
        b"competing changed spool bytes" if phase == "comparison-stat" else expected)
    assert (destination / "prior-subject").read_bytes() == expected
    if phase == "compression-symlink":
        assert (destination / "subject.py").is_symlink()
        assert open_errors == [errno.ELOOP]



def test_physical_copy_rechecks_actual_aggregate_byte_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Files growing after enumeration cannot exceed the physical-store budget while being copied.

    Existing physical-store refusal covers a stable oversized store; it misses
    two individually legal leaves whose reopened sizes exceed the total together.
    """
    common, destination = tmp_path / "git", tmp_path / "copied"
    destination.mkdir()
    paths = []
    for char in ('a', 'b'):
        path = common / 'objects' / (char * 2) / (char * 38)
        path.parent.mkdir(parents=True)
        path.write_bytes(b'x')
        paths.append(path)
    stat = os.stat

    def growing_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if path in ('a' * 38, 'b' * 38) and 'dir_fd' in kwargs:
            original = next(p for p in paths if p.name == path)
            if original.stat().st_size == 1:
                original.write_bytes(b'x' * 800)
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(immutable, '_MAX_OBJECT_BYTES', 1000)
    monkeypatch.setattr(os, 'stat', growing_stat)
    with pytest.raises(GitError, match='object population exceeds'):
        immutable._copy_objects(common, destination)
    assert sum(path.stat().st_size for path in destination.glob('*/*')) <= 1000
