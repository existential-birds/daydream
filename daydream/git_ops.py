"""Single point of contact for all ``git`` and ``gh`` subprocess calls.

This module centralises every shell-out daydream performs against the local
repository or GitHub.  All callers should depend on the public API declared
here rather than spawning ``git`` / ``gh`` directly — that lets the worktree
isolation work evolve invariants (timeouts, working directory contracts,
error semantics) without rewriting the same incantations across the codebase.

Conventions:
    * Every command receives ``cwd=repo`` (no ``git -C`` shenanigans).
    * Read-only queries time out at 5 seconds, IO-bound at 30 seconds, and
      ``gh`` operations at 60 seconds.

Error-handling patterns:
    Functions in this module follow one of two documented patterns:

    **Hard failure (raise GitError)**: Used when the caller cannot proceed
    without the result. Examples: :func:`head_sha`, :func:`diff`, :func:`fetch`.
    These raise :class:`GitError` (or a subclass) on any non-zero exit.

    **Soft failure (return sentinel)**: Used when "data not available" is a
    valid, expected outcome the caller can handle inline. These return
    ``None``, ``False``, ``0``, or ``[]`` on non-zero exit instead of raising.
    Examples: :func:`remote_url`, :func:`merge_base`, :func:`gh_pr_view`.

    Each function's docstring specifies which pattern it follows under its
    **Raises** or **Returns** section.

The module is intentionally dependency-free: stdlib only.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# --- Errors ------------------------------------------------------------------


class GitError(Exception):
    """Base class for all git/gh failures raised by :mod:`daydream.git_ops`."""


class NotAWorktreeError(GitError):
    """Raised when an operation requires a worktree but the path is not one."""


class BranchNotFoundError(GitError):
    """Raised when an operation references a branch that does not exist."""


class WrongBranchError(GitError):
    """Raised when a worktree is checked out to an unexpected branch.

    Defined here for callers (notably the worktree isolation logic) to raise
    when invariant checks fail. Not raised by this module today.
    """


# --- Internal subprocess helpers --------------------------------------------


def _run_git(
    repo: Path,
    args: list[str],
    *,
    timeout: int = 5,
    capture_bytes: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run ``git`` in *repo* with hardened defaults.

    Args:
        repo: Repository working directory.
        args: Arguments after ``git``.
        timeout: Subprocess timeout in seconds.
        capture_bytes: When True, capture stdout/stderr as bytes (no decoding).

    Returns:
        The completed process. ``returncode`` is left to the caller to inspect.

    Raises:
        GitError: If the underlying subprocess machinery itself fails (for
            example timeout, missing binary, OS-level error).
    """
    try:
        return subprocess.run(  # noqa: S603 - arguments are not user-controlled
            ["git", *args],  # noqa: S607 - git is a trusted command
            cwd=repo,
            capture_output=True,
            text=not capture_bytes,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise GitError(f"git {' '.join(args)} failed: {type(exc).__name__}: {exc}") from exc


def _run_gh(
    repo: Path,
    args: list[str],
    *,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Run ``gh`` in *repo* with hardened defaults.

    Args:
        repo: Repository working directory.
        args: Arguments after ``gh``.
        timeout: Subprocess timeout in seconds.

    Returns:
        The completed process with text-decoded stdout/stderr.

    Raises:
        GitError: If the subprocess machinery fails (timeout, missing ``gh``,
            OS-level error).
    """
    try:
        return subprocess.run(  # noqa: S603 - arguments are not user-controlled
            ["gh", *args],  # noqa: S607 - gh is a trusted command
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise GitError(f"gh {' '.join(args)} failed: {type(exc).__name__}: {exc}") from exc


# --- Pre-flight --------------------------------------------------------------


def assert_is_worktree(repo: Path) -> None:
    """Verify *repo* is the root of a git worktree.

    Args:
        repo: Path that should resolve to a worktree top level.

    Raises:
        NotAWorktreeError: If *repo* is not inside a git worktree, or if it is
            inside one but is not itself the worktree's top-level directory
            (catches the "org dir holding repo subdirs" footgun).
    """
    if not repo.exists() or not repo.is_dir():
        raise NotAWorktreeError(f"{repo} is not a directory")

    proc = _run_git(repo, ["rev-parse", "--is-inside-work-tree"], timeout=5)
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise NotAWorktreeError(f"{repo} is not inside a git worktree")

    top_proc = _run_git(repo, ["rev-parse", "--show-toplevel"], timeout=5)
    if top_proc.returncode != 0:
        raise NotAWorktreeError(f"{repo} could not resolve its worktree top-level")

    top = Path(top_proc.stdout.strip()).resolve()
    if top != repo.resolve():
        raise NotAWorktreeError(
            f"{repo} is inside a worktree but its top-level is {top}; pass the worktree root instead",
        )


def is_inside_worktree(repo: Path) -> bool:
    """Return True iff :func:`assert_is_worktree` would succeed for *repo*.

    Args:
        repo: Candidate worktree path.

    Returns:
        True when *repo* is the top-level of a git worktree, otherwise False.
    """
    try:
        assert_is_worktree(repo)
    except NotAWorktreeError:
        return False
    return True


# --- Read-only queries -------------------------------------------------------


def head_sha(repo: Path) -> str:
    """Return the full SHA of ``HEAD`` in *repo*.

    Args:
        repo: Repository working directory.

    Returns:
        The 40-character SHA of the current ``HEAD`` commit.

    Raises:
        GitError: If ``git rev-parse HEAD`` fails (e.g. empty repository).
    """
    proc = _run_git(repo, ["rev-parse", "HEAD"], timeout=5)
    if proc.returncode != 0:
        raise GitError(f"cannot resolve HEAD in {repo}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def commit_message(repo: Path, ref: str = "HEAD") -> str | None:
    """Return the full commit message of *ref*, or ``None`` on failure.

    Soft-failure semantics: returns ``None`` when *ref* doesn't resolve
    (e.g. empty repository, invalid ref).

    Args:
        repo: Repository working directory.
        ref: Commit reference. Defaults to ``"HEAD"``.

    Returns:
        The commit message body, or ``None`` on any non-zero exit.
    """
    proc = _run_git(repo, ["log", "-1", "--format=%B", ref], timeout=5)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def remote_url(repo: Path, remote: str = "origin") -> str | None:
    """Return the URL configured for *remote*, or ``None`` when unset/missing.

    Soft-failure semantics: returns ``None`` on any non-zero ``git`` exit
    (mirrors :func:`default_branch` / :func:`merge_base` / :func:`current_branch`
    behavior for "the data isn't there" cases) and on subprocess machinery
    failures (timeout, missing binary).

    Args:
        repo: Repository working directory.
        remote: Remote name. Defaults to ``"origin"``.

    Returns:
        The configured URL, or ``None`` when the remote is not set.
    """
    try:
        proc = _run_git(repo, ["config", "--get", f"remote.{remote}.url"], timeout=5)
    except GitError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def current_branch(repo: Path) -> str | None:
    """Return the current branch name, or ``None`` when ``HEAD`` is detached.

    Args:
        repo: Repository working directory.

    Returns:
        The branch name (e.g. ``"main"``) or ``None`` for detached ``HEAD``.

    Raises:
        GitError: If the underlying subprocess fails to execute.
    """
    proc = _run_git(repo, ["branch", "--show-current"], timeout=5)
    if proc.returncode != 0:
        raise GitError(f"cannot read current branch in {repo}: {proc.stderr.strip()}")
    name = proc.stdout.strip()
    return name or None


def default_branch(repo: Path) -> str:
    """Resolve the repository's default branch name.

    Resolution order: ``origin/HEAD`` symbolic ref → local ``main`` → local
    ``master``. Raises if none of those exist.

    Args:
        repo: Repository working directory.

    Returns:
        The default branch name (e.g. ``"main"``).

    Raises:
        BranchNotFoundError: If no default branch can be detected.
    """
    sym = _run_git(repo, ["symbolic-ref", "refs/remotes/origin/HEAD"], timeout=5)
    if sym.returncode == 0 and sym.stdout.strip():
        return sym.stdout.strip().rsplit("/", 1)[-1]

    for candidate in ("main", "master"):
        check = _run_git(repo, ["rev-parse", "--verify", f"refs/heads/{candidate}"], timeout=5)
        if check.returncode == 0:
            return candidate

    raise BranchNotFoundError(f"no default branch (origin/HEAD, main, master) found in {repo}")


def branch_exists(repo: Path, ref: str) -> bool:
    """Check whether *ref* exists locally or as ``origin/<ref>``.

    Args:
        repo: Repository working directory.
        ref: Branch name to look up.

    Returns:
        True when ``refs/heads/<ref>`` or ``refs/remotes/origin/<ref>`` exists.
    """
    local = _run_git(repo, ["rev-parse", "--verify", f"refs/heads/{ref}"], timeout=5)
    if local.returncode == 0:
        return True
    remote = _run_git(repo, ["rev-parse", "--verify", f"refs/remotes/origin/{ref}"], timeout=5)
    return remote.returncode == 0


def merge_base(repo: Path, base: str, head: str = "HEAD") -> str | None:
    """Compute the merge-base between *head* and *base*, preferring upstream.

    Ports the algorithm in ``codex-rs/git-utils/src/branch.rs``: when
    ``<base>@{upstream}`` exists and is **ahead** of the local *base*
    (i.e. ``rev-list --left-right --count base...upstream`` reports
    ``right > 0``), the merge-base is computed against the upstream ref instead
    of the local branch.  This avoids stale merge-bases when the local copy of
    the base branch has been rewritten or simply not pulled.

    Returns ``None`` (rather than raising) on the codex-documented "soft"
    failure modes — empty repo, missing ``HEAD``, missing branch — so callers
    can treat them as "no merge-base available" without try/except plumbing.

    Args:
        repo: Repository working directory.
        base: Base branch name (e.g. ``"main"``).
        head: Ref whose merge-base to find. Defaults to ``"HEAD"``.

    Returns:
        The merge-base SHA, or ``None`` when one cannot be resolved.

    Raises:
        GitError: Only for unexpected subprocess failures (timeout, missing
            git binary). The documented soft-failure modes return ``None``.
    """
    head_proc = _run_git(repo, ["rev-parse", "--verify", head], timeout=5)
    if head_proc.returncode != 0:
        return None

    base_proc = _run_git(repo, ["rev-parse", "--verify", base], timeout=5)
    if base_proc.returncode != 0:
        return None
    preferred_ref = base

    upstream = _resolve_upstream_if_remote_ahead(repo, base)
    if upstream is not None:
        upstream_check = _run_git(repo, ["rev-parse", "--verify", upstream], timeout=5)
        if upstream_check.returncode == 0:
            preferred_ref = upstream

    mb = _run_git(repo, ["merge-base", head, preferred_ref], timeout=5)
    if mb.returncode != 0:
        return None
    out = mb.stdout.strip()
    return out or None


def _resolve_upstream_if_remote_ahead(repo: Path, branch: str) -> str | None:
    """Return ``<branch>@{upstream}`` iff it has commits the local branch lacks.

    Mirrors codex's ``resolve_upstream_if_remote_ahead``: parses the right-side
    count from ``rev-list --left-right --count <branch>...<upstream>`` and
    returns the upstream symbolic name when ``right > 0``.
    """
    upstream_name_proc = _run_git(
        repo,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", f"{branch}@{{upstream}}"],
        timeout=5,
    )
    if upstream_name_proc.returncode != 0:
        return None
    upstream = upstream_name_proc.stdout.strip()
    if not upstream:
        return None

    counts_proc = _run_git(
        repo,
        ["rev-list", "--left-right", "--count", f"{branch}...{upstream}"],
        timeout=5,
    )
    if counts_proc.returncode != 0:
        return None
    parts = counts_proc.stdout.strip().split()
    try:
        right = int(parts[1]) if len(parts) >= 2 else 0
    except ValueError:
        right = 0
    return upstream if right > 0 else None


def diff(repo: Path, base: str, head: str = "HEAD", *, exclude: list[str] | None = None) -> str:
    """Return the unified diff between *base* and *head*.

    Uses three-dot syntax (``base...head``) so the diff reflects changes on
    *head* since it diverged from *base*.

    Args:
        repo: Repository working directory.
        base: Base ref (e.g. ``"main"``).
        head: Comparison ref. Defaults to ``"HEAD"``.
        exclude: Optional pathspec excludes (each becomes ``:(exclude)<p>``).

    Returns:
        The diff text. Empty string when there are no changes.

    Raises:
        GitError: If ``git diff`` fails.
    """
    args = ["diff", f"{base}...{head}"]
    if exclude:
        args.append("--")
        args.append(".")
        args.extend(f":(exclude){p.rstrip('/')}" for p in exclude)
    proc = _run_git(repo, args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git diff {base}...{head} failed: {proc.stderr.strip()}")
    return proc.stdout


def diff_paths(
    repo: Path,
    base: str,
    head: str,
    paths: list[str],
    *,
    unified: int = 3,
    merge_base_diff: bool = False,
) -> str:
    """Diff a specific set of paths between *base* and *head*.

    Distinct from :func:`diff` because:
      * This uses two-dot range syntax (``base..head``) by default; :func:`diff`
        uses three-dot. Two-dot shows the working diff between the two refs;
        three-dot shows what *head* introduced relative to the merge-base. PR
        comment line resolution needs two-dot.
      * Always passes ``--unified=<unified>`` for explicit context.
      * Restricts output to *paths*.

    Args:
        repo: Repository root.
        base: Base ref.
        head: Head ref.
        paths: Pathspec list (relative to repo root).
        unified: Lines of context.
        merge_base_diff: When False (default), use ``base..head`` (direct diff);
            when True, use ``base...head`` (diff since merge-base).

    Returns:
        The diff text. Empty string when there are no changes for *paths*.

    Raises:
        GitError: If ``git diff`` fails.
    """
    sep = "..." if merge_base_diff else ".."
    range_arg = f"{base}{sep}{head}"
    args = ["diff", f"--unified={unified}", range_arg, "--", *paths]
    proc = _run_git(repo, args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git diff {range_arg} failed: {proc.stderr.strip()}")
    return proc.stdout


def log(repo: Path, base: str, head: str = "HEAD") -> str:
    """Return the one-line commit log for ``base..head``.

    Args:
        repo: Repository working directory.
        base: Base ref.
        head: Comparison ref. Defaults to ``"HEAD"``.

    Returns:
        Stripped ``--oneline`` log output. Empty string when no commits.

    Raises:
        GitError: If ``git log`` fails.
    """
    proc = _run_git(repo, ["log", f"{base}..{head}", "--oneline"], timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git log {base}..{head} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def daydream_commits(repo: Path, base: str, head: str = "HEAD") -> str | None:
    """Return oneline log of prior daydream commits in ``base..head``.

    Args:
        repo: Repository working directory.
        base: Base ref (e.g. ``"main"``).
        head: Comparison ref. Defaults to ``"HEAD"``.

    Returns:
        Stripped log output, or ``None`` if no daydream commits found.
    """
    proc = _run_git(
        repo,
        ["log", f"{base}..{head}", "--oneline", "--grep=Daydream-Run:"],
        timeout=30,
    )
    if proc.returncode != 0:
        return None
    output = proc.stdout.strip()
    return output or None


def show(repo: Path, ref: str, path: str) -> bytes:
    """Return the raw bytes of *path* at *ref* via ``git show``.

    Args:
        repo: Repository working directory.
        ref: Commit / branch / tag.
        path: Path within the repository.

    Returns:
        Raw bytes of the file at the given revision.

    Raises:
        GitError: If ``git show`` fails (e.g. path missing at that revision).
    """
    proc = _run_git(repo, ["show", f"{ref}:{path}"], timeout=30, capture_bytes=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else proc.stderr
        raise GitError(f"git show {ref}:{path} failed: {stderr.strip()}")
    return proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()


def grep(repo: Path, pattern: str) -> list[str]:
    """Return file paths matching *pattern* via ``git grep -l``.

    Args:
        repo: Repository working directory.
        pattern: Pattern passed to ``git grep -l``.

    Returns:
        List of matching file paths (one per line of stdout).

    Raises:
        GitError: If ``git grep`` exits with an unexpected status.  Exit code
            ``1`` is "no matches" and is treated as success (empty list).
    """
    proc = _run_git(repo, ["grep", "-l", "--", pattern], timeout=30)
    # git grep returns 1 when there are simply no matches.
    if proc.returncode not in (0, 1):
        raise GitError(f"git grep {pattern!r} failed: {proc.stderr.strip()}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def status_porcelain(repo: Path) -> str:
    """Return ``git status --porcelain`` output.

    Args:
        repo: Repository working directory.

    Returns:
        Porcelain-formatted status text. Empty when the tree is clean.

    Raises:
        GitError: If ``git status`` fails.
    """
    proc = _run_git(repo, ["status", "--porcelain"], timeout=10)
    if proc.returncode != 0:
        raise GitError(f"git status failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout


def upstream_ahead_count(repo: Path, branch: str) -> int:
    """Return the number of commits ``<branch>@{upstream}`` is ahead of *branch*.

    Args:
        repo: Repository working directory.
        branch: Branch name to compare against its tracked upstream.

    Returns:
        The right-side count from ``rev-list --left-right --count``. Returns
        ``0`` when *branch* has no configured upstream.
    """
    upstream_name_proc = _run_git(
        repo,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", f"{branch}@{{upstream}}"],
        timeout=5,
    )
    if upstream_name_proc.returncode != 0:
        return 0
    upstream = upstream_name_proc.stdout.strip()
    if not upstream:
        return 0
    counts_proc = _run_git(
        repo,
        ["rev-list", "--left-right", "--count", f"{branch}...{upstream}"],
        timeout=5,
    )
    if counts_proc.returncode != 0:
        return 0
    parts = counts_proc.stdout.strip().split()
    try:
        return int(parts[1]) if len(parts) >= 2 else 0
    except ValueError:
        return 0


def check_ignore(repo: Path, path: str) -> bool:
    """Return True iff ``git check-ignore`` says *path* is ignored.

    Soft-failure semantics: returns ``False`` on any subprocess error (timeout,
    missing binary, OS-level failure) to avoid blocking callers that use this
    for optional file-copy filtering.

    Args:
        repo: Repository working directory.
        path: Path (relative to *repo*) to check.

    Returns:
        True when *path* is gitignored, False otherwise or on error.
    """
    try:
        proc = _run_git(repo, ["check-ignore", "--quiet", path], timeout=5)
    except GitError:
        return False
    return proc.returncode == 0


# --- Mutating ----------------------------------------------------------------


def fetch(repo: Path, remote: str = "origin") -> None:
    """Run ``git fetch`` against *remote*.

    Args:
        repo: Repository working directory.
        remote: Remote name. Defaults to ``"origin"``.

    Raises:
        GitError: If the fetch fails.
    """
    proc = _run_git(repo, ["fetch", remote], timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git fetch {remote} failed in {repo}: {proc.stderr.strip()}")


def checkout_paths(repo: Path, paths: list[Path]) -> None:
    """Run ``git checkout -- <paths>`` to discard local changes for *paths*.

    Args:
        repo: Repository working directory.
        paths: Paths (relative to *repo*) to restore from the index. Pass
            ``[Path(".")]`` to restore the entire working tree.

    Raises:
        GitError: If the checkout fails.
    """
    if not paths:
        return
    args = ["checkout", "--", *(str(p) for p in paths)]
    proc = _run_git(repo, args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git checkout -- {paths} failed in {repo}: {proc.stderr.strip()}")


def clean_untracked(repo: Path) -> None:
    """Run ``git clean -fd`` to remove untracked files and directories.

    Args:
        repo: Repository working directory.

    Raises:
        GitError: If the clean fails.
    """
    proc = _run_git(repo, ["clean", "-fd"], timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git clean -fd failed in {repo}: {proc.stderr.strip()}")


def worktree_add(repo: Path, path: Path, ref: str, *, detach: bool = True) -> None:
    """Create a new worktree at *path* pointing at *ref*.

    Args:
        repo: The source repository.
        path: Filesystem path for the new worktree (must not already exist).
        ref: Commit / branch / tag to check out.
        detach: When True, pass ``--detach`` so the new worktree is detached.

    Raises:
        GitError: If ``git worktree add`` fails.
    """
    args = ["worktree", "add"]
    if detach:
        args.append("--detach")
    args.extend([str(path), ref])
    proc = _run_git(repo, args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git worktree add {path} {ref} failed: {proc.stderr.strip()}")


def worktree_remove(repo: Path, path: Path, *, force: bool = True) -> None:
    """Remove the worktree at *path*.

    Args:
        repo: The source repository (or any worktree linked to the same repo).
        path: Path of the worktree to remove.
        force: When True, pass ``--force`` to remove dirty worktrees.

    Raises:
        GitError: If ``git worktree remove`` fails.
    """
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(path))
    proc = _run_git(repo, args, timeout=30)
    if proc.returncode != 0:
        raise GitError(f"git worktree remove {path} failed: {proc.stderr.strip()}")


# --- gh wrappers -------------------------------------------------------------


def gh_pr_view(repo: Path, pr: int | None = None) -> dict | None:
    """Return ``gh pr view`` output as a dict, or ``None`` on failure.

    When *pr* is ``None``, ``gh pr view`` infers the PR from the currently
    checked-out branch. This mirrors the auto-detection flow used by the CLI
    when the user does not pass an explicit PR number.

    Args:
        repo: Repository working directory.
        pr: Pull request number, or ``None`` to let ``gh`` infer from the
            current branch.

    Returns:
        Parsed JSON dict, or ``None`` when no PR is found / the call fails.
    """
    args = ["pr", "view"]
    if pr is not None:
        args.append(str(pr))
    args.extend(
        [
            "--json",
            "number,title,body,state,headRefName,baseRefName,headRefOid,baseRefOid,url",
        ]
    )
    proc = _run_gh(repo, args, timeout=60)
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def gh_pr_list_for_branch(repo: Path, branch: str) -> list[dict]:
    """List open PRs whose head ref is *branch*.

    Args:
        repo: Repository working directory.
        branch: Head branch name.

    Returns:
        List of PR dicts (empty when no PRs match or the call fails).
    """
    proc = _run_gh(
        repo,
        [
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "number,headRefOid,baseRefOid,baseRefName,url,headRepository,headRepositoryOwner",
        ],
        timeout=60,
    )
    if proc.returncode != 0:
        return []
    try:
        rows = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return rows if isinstance(rows, list) else []


def gh_pr_diff(repo: Path, pr: int) -> str:
    """Return the unified diff for *pr* as text.

    Args:
        repo: Repository working directory.
        pr: Pull request number.

    Returns:
        The diff text.

    Raises:
        GitError: If ``gh pr diff`` fails.
    """
    proc = _run_gh(repo, ["pr", "diff", str(pr)], timeout=60)
    if proc.returncode != 0:
        raise GitError(f"gh pr diff {pr} failed: {proc.stderr.strip()}")
    return proc.stdout


def gh_repo_view(repo: Path) -> tuple[str, str] | None:
    """Return the ``(owner, name)`` slug for the current repository.

    Args:
        repo: Repository working directory.

    Returns:
        Tuple of ``(owner, name)``, or ``None`` when the call fails or the
        slug cannot be parsed.
    """
    proc = _run_gh(
        repo,
        ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        timeout=60,
    )
    if proc.returncode != 0:
        return None
    slug = proc.stdout.strip()
    if "/" not in slug:
        return None
    owner, _, name = slug.partition("/")
    if not owner or not name:
        return None
    return owner, name


def gh_api(
    repo: Path,
    endpoint: str,
    *,
    method: str = "GET",
    paginate: bool = False,
    input_data: Any | None = None,
) -> Any:
    """Call ``gh api <endpoint>`` and return parsed JSON.

    Args:
        repo: Repository working directory.
        endpoint: API path (e.g. ``"repos/owner/repo/pulls/1/comments"``).
        method: HTTP method. Defaults to ``"GET"``.
        paginate: When True, pass ``--paginate`` to walk all result pages.
        input_data: Optional JSON-serialisable payload. When provided, it is
            written to a temporary file and passed via ``--input <path>`` and
            the call uses ``--method <method>`` (gh's preferred form). On
            success the tempfile is removed; on failure it is preserved and
            its path is included in the raised :class:`GitError` so callers
            can inspect the exact request body that was sent.

    Returns:
        The parsed JSON value (object, list, or scalar).

    Raises:
        GitError: If the call fails or returns invalid JSON.
    """
    if input_data is None:
        args = ["api"]
        if method.upper() != "GET":
            args.extend(["-X", method.upper()])
        if paginate:
            args.append("--paginate")
        args.append(endpoint)
        proc = _run_gh(repo, args, timeout=60)
        if proc.returncode != 0:
            raise GitError(f"gh api {endpoint} failed: {proc.stderr.strip()}")
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise GitError(f"gh api {endpoint} returned invalid JSON: {exc}") from exc

    # input_data path: serialise to a tempfile and shell out via `--input`.
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115 - lifecycle managed below
        suffix=".json", mode="w", delete=False, encoding="utf-8"
    )
    tmp_path = Path(tmp.name)
    succeeded = False
    try:
        try:
            json.dump(input_data, tmp)
        finally:
            tmp.close()
        args = ["api", endpoint, "--method", method.upper(), "--input", str(tmp_path)]
        if paginate:
            args.append("--paginate")
        proc = _run_gh(repo, args, timeout=60)
        if proc.returncode != 0:
            raise GitError(
                f"gh api {endpoint} failed: {proc.stderr.strip()} "
                f"(request payload preserved at {tmp_path})"
            )
        try:
            result = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise GitError(
                f"gh api {endpoint} returned invalid JSON: {exc} "
                f"(request payload preserved at {tmp_path})"
            ) from exc
        succeeded = True
        return result
    finally:
        if succeeded:
            tmp_path.unlink(missing_ok=True)
