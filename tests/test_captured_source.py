"""Real Git/filesystem proofs at the accepted-source owner.

Snapshot tests cover copy independence and index preservation. These cases add
net source identity, stable acceptance and later integrity, which a copy alone
cannot establish; source consumers exercise dispatch and coverage separately.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream.git_ops import immutable, process, snapshot
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


def test_capture_uses_working_bytes_of_a_staged_addition(source: tuple[Path, str, str], tmp_path: Path) -> None:
    """A staged added file is captured from its working bytes and has no old side."""
    repo, base, head = source
    write_and_stage(repo, "added.py", b"value = 2\n")
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True)
    assert "added.py" in captured.files
    assert "+value = 2\n" in captured.diff
    with pytest.raises(PathAbsentError):
        captured.read("old", "added.py")
    assert captured.read("new", "added.py")[1] == b"value = 2\n"
    assert (captured.root / "added.py").read_bytes() == b"value = 2\n"


def test_capture_of_a_shallow_clone_stays_copyable(source: tuple[Path, str, str], tmp_path: Path) -> None:
    """Shallow history ends at a boundary; the capture keeps it, so read-only copies still clone."""
    origin, _, head = source
    write_and_stage(origin, "subject.py", "value = 2\n")
    latest = commit(origin, "latest")
    shallow = tmp_path / "shallow"
    git(tmp_path, "clone", "--quiet", "--depth=2", origin.as_uri(), str(shallow))
    assert git(shallow, "rev-parse", "--is-shallow-repository") == "true"
    captured = capture_source(shallow, tmp_path / "captured", base_sha=head, head_sha=latest, local=True)
    copy = snapshot.prepare_independent_snapshot(captured.root, tmp_path / "copy", include_untracked=True)
    assert (copy.repo / "subject.py").read_text() == "value = 2\n"


@pytest.mark.parametrize("change", ["bytes", "parent", "symlink"])
@pytest.mark.parametrize("persistent", [False, True])
def test_capture_retries_actual_materialization_instability(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persistent: bool,
    change: str,
) -> None:
    """Concurrent writes around actual copying retry finitely and never accept torn source."""
    repo, base, head = source
    destination = tmp_path / "captured"
    relative = "unchanged/helper.py" if change == "parent" else "subject.py"
    changed = "helper.py" if change == "symlink" else relative
    if change == "symlink":
        (repo / changed).unlink()
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
                    if change == "symlink":
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


@pytest.mark.parametrize("fault", ["cancel-replaced-storage", "missing-spool"])
def test_capture_private_write_failure_preserves_subject_and_primary_error(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Private write failure cannot turn a present source into an accepted deletion or erase cancellation."""
    import asyncio

    repo, base, head = source
    destination = tmp_path / "captured"
    original = os.open
    writes = 0

    def failing_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        nonlocal writes
        if path == "subject.py" and flags & os.O_CREAT:
            writes += 1
            if fault == "cancel-replaced-storage":
                (destination / ".git/objects").rename(destination / ".git/preserved-objects")
                raise asyncio.CancelledError("cancel during private write")
            raise FileNotFoundError("private spool disappeared")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", failing_open)
    expected = asyncio.CancelledError if fault == "cancel-replaced-storage" else SnapshotPreparationError
    with pytest.raises(expected):
        capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
    assert writes == (3 if fault == "missing-spool" else 1)
    assert (repo / "subject.py").read_bytes() == b"value = 1\n"
    assert destination.exists() == (fault != "missing-spool")


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


@pytest.mark.parametrize("corruption", ["missing", "population-gitignore", "metadata", "storage", "skip"])
def test_capture_integrity_never_falls_back_to_original(
    source: tuple[Path, str, str], tmp_path: Path, corruption: str,
) -> None:
    """Required damaged capture evidence stays a failure even while original bytes remain available."""
    repo, base, head = source
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True,
                              exclude=["unchanged"])
    captured.validate()  # A preceding success must never authorize later corruption.
    if corruption == "missing":
        (captured.root / "subject.py").unlink()
    elif corruption == "population-gitignore":
        (captured.root / "generated").mkdir()
        (captured.root / "generated/.gitignore").write_text("*\n")
        (captured.root / "generated/injected.py").write_text("tampered")
    elif corruption == "metadata":
        (captured.root / ".git/HEAD").write_text(base + "\n")
    elif corruption == "skip":
        git(captured.root, "update-index", "--skip-worktree", "subject.py")
    else:
        old_objects = captured.root / ".git/old-objects"
        (captured.root / ".git/objects").rename(old_objects)
        shutil.copytree(old_objects, captured.root / ".git/objects")
    with pytest.raises(GitError) as failure:
        captured.validate()
    if corruption == "population-gitignore":
        assert "generated/.gitignore" in str(failure.value)
    else:
        with pytest.raises(GitError):
            captured.read("new", "subject.py")
    assert (repo / "subject.py").read_bytes() == b"value = 1\n"


def test_capture_integrity_rejects_tampered_content_until_restored(
    source: tuple[Path, str, str], tmp_path: Path,
) -> None:
    """Tampered subject bytes, a same-length support rewrite with its mtime restored (defeating
    stat-only checks), and an injected file each fail validation; restoring the bytes revalidates."""
    repo, base, head = source
    captured = capture_source(repo, tmp_path / "captured", base_sha=base, head_sha=head, local=True,
                              exclude=["unchanged"])
    captured.validate()
    subject, helper = captured.root / "subject.py", captured.root / "unchanged/helper.py"
    injected = captured.root / "injected.py"

    def restore_bytes(path: Path, original: bytes, info: os.stat_result) -> None:
        path.write_bytes(original)
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))

    for name in ("bytes", "support-bytes", "population"):
        if name == "population":
            injected.write_text("tampered")
        else:
            target = subject if name == "bytes" else helper
            original, info = target.read_bytes(), target.stat()
            restore_bytes(target, b"value = 9\n" if name == "bytes" else b"support = 9\n", info)
        with pytest.raises(GitError) as failure:
            captured.validate()
        if name == "population":
            assert "injected.py" in str(failure.value), name
            injected.unlink()
        else:
            if name == "bytes":
                with pytest.raises(GitError):
                    captured.read("new", "subject.py")
            restore_bytes(target, original, info)
        captured.validate()
    assert (repo / "subject.py").read_bytes() == b"value = 1\n"


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


@pytest.mark.parametrize("state", ["noncone", "deleted"])
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
    if state == "noncone":
        git(repo, "sparse-checkout", "set", "--no-cone", "/visible/", "/.gitignore")
    else:
        (repo / "support/helper.py").unlink()
    flags = git(repo, "ls-files", "-t", "--", "support/helper.py")
    assert flags == ("H" if state == "deleted" else "S") + " support/helper.py"
    original_index = (repo / ".git/index").read_bytes()
    original_refs = git(repo, "show-ref")
    destination = tmp_path / "captured"
    if state == "noncone":
        with pytest.raises(SnapshotPreparationError, match="skip-worktree.*materialize.*rerun"):
            capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
        assert not destination.exists()
    else:
        captured = capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
        assert set(captured.files) == {"visible/subject.py", "support/helper.py"}
        with pytest.raises(PathAbsentError):
            captured.read("new", "support/helper.py")
        assert "deleted file mode" in captured.diff
        captured.validate()
    assert (repo / ".git/index").read_bytes() == original_index
    assert git(repo, "show-ref") == original_refs
    assert (repo / "visible/subject.py").read_bytes() == b"value = 2\n"
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


def test_streamed_blob_framing_faults_reap_the_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for fault in ("wrong-oid", "truncated", "terminator", "extra"):
        with monkeypatch.context() as patch:
            _assert_streamed_blob_child_reaped(tmp_path / fault, patch, fault)


def test_streamed_blob_process_faults_reap_the_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for fault in ("nonzero", "stderr", "timeout", "setup", "sink"):
        with monkeypatch.context() as patch:
            _assert_streamed_blob_child_reaped(tmp_path / fault, patch, fault)


def _assert_streamed_blob_child_reaped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    """Real duplex pipes prove framing, stderr/deadline bounds and cleanup even before reader setup completes.

    Existing buffered subprocess tests have no request writes, object framing or
    failing consumer, so cannot catch these lifecycle/duplex regressions.
    """
    import asyncio
    import hashlib
    import subprocess
    import sys

    from daydream.git_ops.captured_source import _BlobReader

    tmp_path.mkdir()
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


@pytest.mark.parametrize("kind", [
    "blob", "existing-tree", "parent-open", "leaf-cancel", "read-truncated", "read-cancel", "read-symlink",
])
def test_capture_preserves_competing_object_bytes(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    """Loose-object publication must neither replace a competing leaf nor erase it during refusal cleanup."""
    import asyncio

    repo, base, head = source
    expected_source = "value = 1\n" if kind == "existing-tree" else "value = 345\n"
    (repo / "subject.py").write_text(expected_source)
    private_io = kind in {"leaf-cancel", "read-truncated", "read-cancel", "read-symlink"}
    if private_io:
        (repo / "fresh").mkdir()
        (repo / "fresh/new.py").write_text(expected_source)
    tree_oid = git(repo, "rev-parse", "HEAD^{tree}")
    destination = tmp_path / "captured"
    expected = ((repo / ".git/objects" / tree_oid[:2] / tree_oid[2:]).read_bytes()
                if kind == "existing-tree" else b"competing object bytes")
    link = os.link
    competing: list[Path] = []
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
                replace_leaf(leaf)
                raise cancellation
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
        if (kind == "parent-open" and flags & os.O_DIRECTORY and isinstance(path, str) and len(path) == 2
                and (destination / "subject.py").exists() and not competing):
            preserve_parent(destination / ".git/objects" / path)
        if (kind == "existing-tree" and path == tree_oid[2:] and not flags & os.O_CREAT
                and (destination / "subject.py").exists() and not competing):
            target = destination / ".git/objects" / tree_oid[:2] / tree_oid[2:]
            target.rename(target.with_name("preserved-object"))
            target.write_bytes(expected)
            competing.append(target)
        return opened(path, flags, *args, **kwargs)

    if kind == "read-symlink":
        monkeypatch.setattr(os, "stat", competing_stat)
    elif private_io:
        monkeypatch.setattr(os, "fdopen", competing_fdopen)
    elif kind == "blob":
        monkeypatch.setattr(os, "link", competing_link)
    else:
        monkeypatch.setattr(os, "open", competing_open)
    failure = asyncio.CancelledError if kind.endswith("cancel") else GitError
    message = None if kind.startswith("parent") or private_io else "conflicts|replaced"
    with pytest.raises(failure, match=message) as caught:
        capture_source(repo, destination, base_sha=base, head_sha=head, local=True)
    assert len(competing) == 1 and competing[0].read_bytes() == expected
    assert (repo / "subject.py").read_text() == expected_source
    if private_io:
        assert (repo / "fresh/new.py").read_text() == expected_source
    if kind.startswith(("leaf", "read")):
        assert (destination / "fresh/prior-new.py").read_bytes() == (
            b"" if kind == "read-truncated" else expected_source.encode())
    if kind == "read-symlink":
        assert (destination / "fresh/new.py").is_symlink()
    if kind.endswith("cancel"):
        assert caught.value is cancellation


@pytest.mark.parametrize("phase", ["compression", "comparison-stat"])
def test_capture_rejects_identical_byte_spool_replacement(
    source: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    """Equal content cannot authorize a different private spool inode during local acceptance."""
    repo, base, head = source
    expected = b"value = 4321\n" if phase == "compression" else b"value = 1\n"
    (repo / "subject.py").write_bytes(expected)
    destination = tmp_path / "captured"
    opened, stat = os.open, os.stat
    replaced = comparison_started = False

    def replace_private() -> None:
        nonlocal replaced
        replaced = True
        target = destination / "subject.py"
        body = target.read_bytes() if phase != "comparison-stat" else b"competing changed spool bytes"
        target.rename(destination / "prior-subject")
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
        return opened(path, flags, *args, **kwargs)

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
