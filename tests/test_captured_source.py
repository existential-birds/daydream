"""Real Git/filesystem proofs at the accepted-source owner.

Snapshot tests cover copy independence and index preservation. These cases add
net source identity, stable acceptance and later integrity, which a copy alone
cannot establish; source consumers exercise dispatch and coverage separately.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from daydream.git_ops import immutable, queries
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


def test_capture_uses_working_bytes_after_staging(source: tuple[Path, str, str], tmp_path: Path) -> None:
    """The net patch and native files share accepted final bytes rather than staged bytes."""
    repo, base, head = source
    write_and_stage(repo, "subject.py", "value = 2\n")
    (repo / "subject.py").write_text("value = 3\n")
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True)
    assert "+value = 3\n" in captured.diff
    assert "+value = 2\n" not in captured.diff
    assert captured.read("old", "subject.py")[1] == b"value = 0\n"
    assert captured.read("new", "subject.py")[1] == b"value = 3\n"
    assert (captured.root / "subject.py").read_bytes() == b"value = 3\n"


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


@pytest.mark.parametrize("persistent", [False, True])
def test_capture_retries_actual_materialization_instability(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool,
) -> None:
    """Concurrent writes around actual copying retry finitely and never accept torn source."""
    repo, base, head = source
    destination = tmp_path / "captured"
    original = Path.write_bytes
    writes = 0

    def changing_write(path: Path, body: bytes) -> int:
        nonlocal writes
        written = original(path, body)
        if path == destination / "subject.py":
            writes += 1
            if persistent or writes == 1:
                original(repo / "subject.py", f"value = {writes + 10}\n".encode())
        return written

    monkeypatch.setattr(Path, "write_bytes", changing_write)
    if persistent:
        with pytest.raises(SnapshotPreparationError, match="stop concurrent writers and rerun"):
            capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
        assert writes == 3
        assert not destination.exists()
    else:
        captured = capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
        assert writes == 2
        assert captured.read("new", "subject.py")[1] == b"value = 11\n"


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


@pytest.mark.parametrize("corruption", ["bytes", "missing", "population", "metadata", "storage"])
def test_capture_integrity_never_falls_back_to_original(
    source: tuple[Path, str, str], tmp_path: Path, corruption: str,
) -> None:
    """Required damaged capture evidence stays a failure even while original bytes remain available."""
    repo, base, head = source
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True)
    if corruption == "bytes":
        (captured.root / "subject.py").write_text("tampered")
    elif corruption == "missing":
        (captured.root / "subject.py").unlink()
    elif corruption == "population":
        (captured.root / "injected.py").write_text("tampered")
    elif corruption == "metadata":
        (captured.root / ".git/HEAD").write_text(base + "\n")
    else:
        old_objects = captured.root / ".git/old-objects"
        (captured.root / ".git/objects").rename(old_objects)
        shutil.copytree(old_objects, captured.root / ".git/objects")
    with pytest.raises(GitError):
        captured.validate()
    if corruption in {"bytes", "missing", "metadata", "storage"}:
        with pytest.raises(GitError):
            captured.read("new", "subject.py")
    assert (repo / "subject.py").read_bytes() == b"value = 1\n"


def test_capture_preserves_metadata_without_native_symlink_or_gitlink_expansion(
    source: tuple[Path, str, str], tmp_path: Path,
) -> None:
    """Unsupported text entries retain patch identity without exposing linked native contents."""
    repo, base, head = source
    (repo / "linked.py").symlink_to(tmp_path / "private.py")
    (tmp_path / "private.py").write_text("outside")
    (repo / ".daydream").symlink_to(tmp_path / "private.py")
    git(repo, "add", "linked.py")
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},module")
    (repo / "module").mkdir()
    (repo / "module/secret.py").write_text("nested")
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True)
    assert not (captured.root / ".daydream").exists()
    assert not (captured.root / "linked.py").exists()
    assert not (captured.root / "module").exists()
    assert "120000" in git(captured.root, "ls-tree", "HEAD", "linked.py")
    assert "160000" in git(captured.root, "ls-tree", "HEAD", "module")
    for path in ("linked.py", "module", "../private.py"):
        with pytest.raises(FrozenSourceRejected):
            captured.read("new", path)
    captured.validate()
    (repo / ".daydream").unlink()
    (repo / ".daydream").symlink_to(tmp_path / "different-private-target")
    repeated = capture_source(repo, tmp_path / "repeated", base_sha=base, head_sha=head, local=True)
    assert repeated.source_id == captured.source_id


def test_capture_raw_duplicate_and_empty_blobs(source: tuple[Path, str, str], tmp_path: Path) -> None:
    """Raw batching preserves binary duplicate and zero-length payload boundaries."""
    repo, base, _ = source
    write_and_stage(repo, "a.bin", b"\0duplicate\xff\n")
    write_and_stage(repo, "b.bin", b"\0duplicate\xff\n")
    write_and_stage(repo, "empty.bin", b"")
    head = commit(repo, "raw files")
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=False)
    for name in ("a.bin", "b.bin"):
        assert (captured.root / name).read_bytes() == b"\0duplicate\xff\n"
        assert captured.read("new", name)[1] == b"\0duplicate\xff\n"
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
