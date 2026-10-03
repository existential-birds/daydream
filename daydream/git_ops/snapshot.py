"""Independent repository storage with validated environment and copied worktree state."""

from __future__ import annotations

import os
import re
import shutil
import stat
from pathlib import Path

from daydream.git_ops import mutations, process, queries
from daydream.git_ops.models import GitError, IndependentSnapshot, SnapshotPreparationError


def _require_disjoint_snapshot_paths(source: Path, destination: Path) -> None:
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise SnapshotPreparationError("snapshot paths must be disjoint")


def _snapshot_leaf(root: Path, rel: str) -> Path:
    parts = rel.split("/")
    if not rel or any(part in {"", ".", "..", ".git"} for part in parts) or "\0" in rel:
        raise SnapshotPreparationError("invalid snapshot-relative path")
    parent = root
    for part in parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            raise SnapshotPreparationError("snapshot path has a symlinked parent")
    return root / rel


def _remove_snapshot_leaf(path: Path) -> None:
    # Called only on a validated leaf in a newly created standalone snapshot.
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _snapshot_has_nondirectory_ancestor(root: Path, rel: str) -> bool:
    """Return whether a copied ancestor already makes *rel* unreachable."""
    ancestor = root
    for part in rel.split("/")[:-1]:
        ancestor /= part
        try:
            mode = ancestor.lstat().st_mode
        except FileNotFoundError:
            return False
        except NotADirectoryError:
            return True
        if not stat.S_ISDIR(mode):
            return True
    return False


def _copy_snapshot_leaf(source: Path, destination: Path, rel: str) -> None:
    src = _snapshot_leaf(source, rel)
    dst = _snapshot_leaf(destination, rel)
    try:
        mode = src.lstat().st_mode
    except NotADirectoryError:
        # Parent paths are copied before children. A staged directory-to-file
        # change therefore makes former HEAD children unreachable only after
        # their authoritative replacement leaf has already been copied. With
        # untracked files excluded, however, the replacement is absent from the
        # path set and the old tracked destination child must become a deletion.
        if not _snapshot_has_nondirectory_ancestor(destination, rel):
            _remove_snapshot_leaf(dst)
        return
    except FileNotFoundError:
        _remove_snapshot_leaf(dst)
        return
    if stat.S_ISLNK(mode):
        _remove_snapshot_leaf(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.readlink(src), dst)
    elif stat.S_ISDIR(mode):
        # Gitlinks contain no file bytes to copy. A former file now represented
        # by a directory must not retain the old committed file in the snapshot.
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        dst.mkdir(parents=True, exist_ok=True)
    elif stat.S_ISREG(mode):
        if dst.is_symlink() or dst.is_dir():
            _remove_snapshot_leaf(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)
    else:
        raise SnapshotPreparationError("snapshot path is not a regular file, symlink, or directory")


_SNAPSHOT_GIT_REDIRECTS = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CEILING_DIRECTORIES",
        "GIT_PREFIX",
        "GIT_NAMESPACE",
        "GIT_SHALLOW_FILE",
        "GIT_GRAFT_FILE",
        "GIT_REPLACE_REF_BASE",
        "GIT_REFERENCE_BACKEND",
        "GIT_TEMPLATE_DIR",
        "GIT_EXTERNAL_DIFF",
        "GIT_DIFF_OPTS",
    }
)


def _snapshot_git_exec_path_is_safe() -> bool:
    """Admit only the exact helper path Git itself exports to hook processes."""
    inherited = os.environ.get("GIT_EXEC_PATH")
    if inherited is None:
        return True
    if not inherited:
        return False
    query_env = dict(os.environ)
    del query_env["GIT_EXEC_PATH"]
    try:
        proc = process._run_git(
            Path.cwd(),
            ["--exec-path"],
            timeout=5,
            retries=0,
            env_cmd=query_env,
        )
    except (GitError, OSError, UnicodeError):
        return False
    return proc.returncode == 0 and proc.stdout == f"{inherited}\n"


def _snapshot_git_environment_violations() -> tuple[str, ...]:
    """Return names of inherited Git settings that make snapshots unsafe."""
    violations = {name for name in os.environ if name in _SNAPSHOT_GIT_REDIRECTS or name.startswith("GIT_TRACE")}
    config_names = {name for name in os.environ if name.startswith("GIT_CONFIG")}
    if config_names:
        config_is_safe = True
        raw_count = os.environ.get("GIT_CONFIG_COUNT", "")
        if re.fullmatch(r"[0-9]{1,3}", raw_count) is None or int(raw_count) > 256:
            config_is_safe = False
        else:
            expected = {"GIT_CONFIG_COUNT"}
            for index in range(int(raw_count)):
                key_name = f"GIT_CONFIG_KEY_{index}"
                value_name = f"GIT_CONFIG_VALUE_{index}"
                expected.update((key_name, value_name))
                key = os.environ.get(key_name, "").lower()
                value = os.environ.get(value_name)
                if value is None:
                    config_is_safe = False
                    continue
                # Signing policy and ignore-pattern paths cannot redirect storage or
                # execute Git helpers. Preserve them verbatim, including true signing.
                # All other config (especially includes, filters and hooks) fails closed.
                if key == "commit.gpgsign":
                    if value.lower() not in {
                        "true",
                        "false",
                        "yes",
                        "no",
                        "on",
                        "off",
                        "1",
                        "0",
                    }:
                        config_is_safe = False
                elif key != "core.excludesfile":
                    config_is_safe = False
            if config_names != expected:
                config_is_safe = False
        if not config_is_safe:
            violations.update(config_names)
    # Preserve the old fail-fast boundary: once refusal is certain, do not run
    # a diagnostic Git subprocess under already-dangerous inherited settings.
    if not violations and not _snapshot_git_exec_path_is_safe():
        violations.add("GIT_EXEC_PATH")
    return tuple(sorted(violations))


_SAFE_GIT_ENVIRONMENT_NAME_RE = re.compile(r"GIT_[A-Z0-9_]+")
_GIT_ENVIRONMENT_DIAGNOSTIC_BUDGET = 320
_GIT_ENVIRONMENT_DIAGNOSTIC_NAME_LIMIT = 20


def _format_snapshot_git_environment_violations(names: tuple[str, ...]) -> str:
    """Format bounded variable-name evidence without exposing arbitrary keys."""
    safe_names: list[str] = []
    invalid_count = 0
    for name in names:
        if len(name) <= 128 and _SAFE_GIT_ENVIRONMENT_NAME_RE.fullmatch(name) is not None:
            safe_names.append(name)
        else:
            invalid_count += 1

    displayed: list[str] = []
    used = 0
    for name in safe_names:
        addition = len(name) + (2 if displayed else 0)
        if (
            len(displayed) >= _GIT_ENVIRONMENT_DIAGNOSTIC_NAME_LIMIT
            or used + addition > _GIT_ENVIRONMENT_DIAGNOSTIC_BUDGET
        ):
            break
        displayed.append(name)
        used += addition
    parts = [", ".join(displayed)] if displayed else []
    omitted_count = len(safe_names) - len(displayed)
    if omitted_count:
        parts.append(f"and {omitted_count} more")
    if invalid_count:
        noun = "name" if invalid_count == 1 else "names"
        parts.append(f"{invalid_count} invalid Git variable {noun}")
    return "; ".join(parts)


def prepare_independent_snapshot(
    source: Path,
    destination: Path,
    *,
    include_untracked: bool,
) -> IndependentSnapshot:
    """Copy tracked worktree/index state into independent Git storage.

    This is a Git-storage boundary, not a host-filesystem sandbox. Parent
    symlinks fail closed before copying; leaf symlinks remain links. The caller
    owns the newly created destination and its cleanup on every failure path.

    Inherited Git repository/diff/trace and arbitrary configuration overrides
    are unsupported, even when empty. Only well-formed indexed signing-policy,
    ignore-file configuration, and Git's exact default executable-helper path
    pass through unchanged. Refuse before any snapshot subprocess or mutation
    rather than clearing caller settings (including hooks) or returning storage
    that only works in the parent environment.
    """
    environment_violations = _snapshot_git_environment_violations()
    if environment_violations:
        details = _format_snapshot_git_environment_violations(environment_violations)
        raise SnapshotPreparationError(
            "snapshot refuses inherited Git repository, configuration, diff, or trace "
            f"overrides; offending variables: {details}"
        )
    source = source.resolve(strict=True)
    destination = destination.resolve()
    _require_disjoint_snapshot_paths(source, destination)
    queries.assert_is_worktree(source)
    if destination.exists():
        raise SnapshotPreparationError("snapshot destination already exists")
    try:
        unborn = queries.is_unborn_head(source)
        if unborn:
            branch = queries.symbolic_head(source, strict=True)
            if branch is None:
                raise SnapshotPreparationError("unborn snapshot needs symbolic HEAD")
            queries.init_repository(destination, initial_branch=branch)
            paths = queries.ls_files(source, strict=True)
        else:
            mutations.clone(str(source), destination, no_local=True)
            mutations.checkout_detach(destination, queries.head_sha(source))
            mutations.update_refs(
                destination, {f"refs/heads/{name}": oid for name, oid in queries.list_local_branches(source).items()}
            )
            for remote in queries.list_remotes(destination, strict=True):
                mutations.remove_remote(destination, remote)
            refs = queries._snapshot_remote_refs(destination)
            if refs:
                commands = "".join(f"delete {ref}\n" for ref in refs)
                proc = process._run_git(destination, ["update-ref", "--stdin"], input_text=commands, retries=0)
                if proc.returncode != 0:
                    raise SnapshotPreparationError("cannot remove snapshot remote-tracking refs")
            paths = [*queries.ls_tree_files(source, "HEAD", strict=True), *queries.ls_files(source, strict=True)]
        _require_disjoint_snapshot_paths(queries.git_common_dir(source), queries.git_common_dir(destination))
        _require_disjoint_snapshot_paths(
            queries._snapshot_git_path(source, "objects"),
            queries._snapshot_git_path(destination, "objects"),
        )
        if queries.object_alternates(destination, strict=True) or queries.list_remotes(destination, strict=True):
            raise SnapshotPreparationError("snapshot retains alternates or remotes")
        if queries._snapshot_remote_refs(destination):
            raise SnapshotPreparationError("snapshot retains remote-tracking refs")
        if include_untracked:
            paths.extend(
                queries._path_names(
                    source,
                    ["ls-files", "--others", "--exclude-standard", "-z"],
                    strict=True,
                    timeout=5,
                )
            )
        unique_paths = sorted(set(paths))
        # Validate all parents before the first copy, including parents present
        # only in HEAD or only in the index (staged path-type changes).
        for rel in unique_paths:
            _snapshot_leaf(source, rel)
            _snapshot_leaf(destination, rel)
        for rel in unique_paths:
            _copy_snapshot_leaf(source, destination, rel)
        patch = queries.staged_patch(source)
        if patch:
            mutations.apply_staged_patch(destination, patch)
        outward = frozenset(
            destination / rel
            for rel in unique_paths
            if (destination / rel).is_symlink() and not (destination / rel).resolve().is_relative_to(destination)
        )
        return IndependentSnapshot(destination, outward)
    except (OSError, ValueError) as exc:
        raise SnapshotPreparationError(f"snapshot preparation failed: {type(exc).__name__}") from exc
