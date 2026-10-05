"""Executable evidence for deterministic post-fix regression claims.

Model prose cannot establish that a repository gate fails. Only an exact
published test recipe or a declared Makefile validation target may be run;
model-provided shell text is never executed.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.repository_paths import path_is_confined
from daydream.test_execution import TestRecipe, run_test_command

_VALIDATION_TARGETS = frozenset({"lint", "check", "test", "typecheck", "build", "validate", "verify"})
_CHECK_WORDS = re.compile(
    r"\b(?:ruff|lint\w*|mypy|typecheck|type.check|pytest|test suite|compiler|build|gate|check)\b", re.I,
)
_FAILURE_WORDS = re.compile(r"\b(?:fail\w*|red|error\w*|violat\w*|reject\w*)\b", re.I)
_MAKE_COMMAND = re.compile(r"\bmake\s+([a-zA-Z][a-zA-Z0-9_-]*)\b")
_TOOL_TARGETS = ((r"\b(?:ruff|lint\w*)\b", "lint"), (r"\b(?:mypy|typecheck|type.check)\b", "typecheck"),
                 (r"\b(?:pytest|test suite)\b", "test"), (r"\b(?:build|compiler)\b", "build"))


class CheckClaimUnavailable(RuntimeError):
    """A deterministic regression claim could not obtain complete host evidence."""


def _declared_make_target(repo: Path, target: str) -> bool:
    """Admit only conventional validation targets declared by this repository."""
    if target not in _VALIDATION_TARGETS:
        return False
    for name in ("GNUmakefile", "makefile", "Makefile"):
        path = repo / name
        if not path.is_file():
            continue
        if not path_is_confined(repo, name):
            return False
        # Make uses the first existing makefile; do not authorize a target from
        # a lower-priority file that make would never read.
        return bool(re.search(rf"(?m)^{re.escape(target)}\s*:(?![=])", path.read_text()))
    return False


def _command_for_claim(
    repo: Path, entry: dict[str, Any], recipe: TestRecipe | None,
) -> list[tuple[tuple[str, ...], Path]]:
    reason = str(entry.get("reason") or "")
    requested = entry.get("check_command")
    has_request = isinstance(requested, str) and bool(requested.strip())
    if (not has_request and entry.get("check_only") is not True
            and not (_CHECK_WORDS.search(reason) and _FAILURE_WORDS.search(reason))):
        return []
    canonical = recipe.command.value if recipe is not None else None
    canonical_argv = canonical if isinstance(canonical, tuple) else None
    package_cwd = repo / recipe.package.cwd_relative if recipe is not None else repo
    if not path_is_confined(repo, str(package_cwd), directory_scope=True, allow_absolute=True):
        raise CheckClaimUnavailable("Deterministic check unavailable: package working directory escapes repository")

    commands: list[tuple[tuple[str, ...], Path]] = []
    if has_request:
        try:
            argv = tuple(shlex.split(str(requested)))
        except ValueError as exc:
            raise CheckClaimUnavailable("Deterministic check unavailable: malformed check_command") from exc
        if canonical_argv is not None and argv == canonical_argv:
            commands.append((argv, package_cwd))
        elif len(argv) == 2 and argv[0] == "make" and _declared_make_target(repo, argv[1]):
            commands.append((argv, repo))
        else:
            raise CheckClaimUnavailable(
                f"Deterministic check unavailable: command is not repository-declared: {requested}"
            )

    matches = _MAKE_COMMAND.findall(reason)
    if matches:
        for target in dict.fromkeys(matches):
            if not _declared_make_target(repo, target):
                raise CheckClaimUnavailable(
                    f"Deterministic check unavailable: make {target} is not repository-declared"
                )
            commands.append((("make", target), repo))

    for pattern, target in _TOOL_TARGETS:
        if not re.search(pattern, reason, re.I):
            continue
        if _declared_make_target(repo, target):
            commands.append((("make", target), repo))
        elif canonical_argv is not None and any(re.search(pattern, word, re.I) for word in canonical_argv):
            commands.append((canonical_argv, package_cwd))
        else:
            raise CheckClaimUnavailable(f"Deterministic {target} check unavailable: no repository-declared command")
    if commands:
        return list(dict.fromkeys(commands))
    if canonical_argv is not None and entry.get("check_only") is True:
        return [(canonical_argv, package_cwd)]
    raise CheckClaimUnavailable("Deterministic check unavailable: no repository-declared validation command")


def _source_tree_key(repo: Path) -> str:
    """Include clean tracked modes and user symlinks as well as Git deltas."""
    tracked = git_ops.snapshot_worktree_paths(repo, git_ops.ls_files(repo, strict=True), allow_leaf_symlink=True)
    untracked = git_ops.snapshot_untracked_paths(repo, include_runtime_artifacts=False)
    ignored = git_ops.snapshot_ignored_owner_paths(repo)
    return git_ops.tree_key((*tracked, *untracked.values(), *ignored.values()))


async def substantiate_check_claims(
    repo: Path,
    verdicts: list[dict[str, Any]],
    *,
    recipe: TestRecipe | None,
    wall_budget_s: float = 60.0,
) -> list[dict[str, Any]]:
    """Confirm check-based regressions; refuted claims remain honestly unresolved.

    Semantic regressions retain their blocking verdict. Unavailable commands,
    timeouts, and incomplete output raise a verification failure rather than
    masquerading as either a confirmed regression or a clean check.
    """
    results: dict[tuple[tuple[str, ...], Path], dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for original in verdicts:
        entry = dict(original)
        if entry.get("verdict") != "regressed":
            output.append(entry)
            continue
        commands = _command_for_claim(repo, entry, recipe)
        if not commands:
            output.append(entry)
            continue
        checks = []
        for command in commands:
            argv, cwd = command
            evidence = results.get(command)
            if evidence is None:
                try:
                    head = git_ops.head_sha(repo)
                    index = git_ops.snapshot_index(repo)
                    before_key = _source_tree_key(repo)
                    result = await run_test_command(list(argv), cwd=cwd, wall_budget_s=wall_budget_s)
                    after_key = _source_tree_key(repo)
                    if (git_ops.head_sha(repo) != head or git_ops.snapshot_index(repo) != index
                            or after_key != before_key):
                        raise CheckClaimUnavailable(
                            f"Deterministic check unavailable: {shlex.join(argv)} mutated the source tree or index"
                        )
                except (OSError, git_ops.GitError) as exc:
                    raise CheckClaimUnavailable(
                        f"Deterministic check unavailable: {shlex.join(argv)}: {exc}"
                    ) from exc
                if result.incomplete:
                    raise CheckClaimUnavailable(
                        f"Deterministic check unavailable: {shlex.join(argv)} produced incomplete evidence "
                        f"(timed_out={result.timed_out}, output_truncated={result.output_truncated})"
                    )
                evidence = {
                    "command": list(argv), "cwd_relative": cwd.relative_to(repo).as_posix(),
                    "status": "passed" if result.passed else "failed",
                    "exit_status": result.exit_status, "output": result.merged_output,
                }
                results[command] = evidence
            checks.append(dict(evidence))
        failed = next((check for check in checks if check["status"] == "failed"), None)
        entry["check_evidence"] = {**(failed or checks[0]), "checks": checks}
        check_only = entry.get("check_only") is True
        if failed is None and not isinstance(entry.get("check_only"), bool):
            raise CheckClaimUnavailable(
                "Unsupported deterministic regression claim: repository checks passed, but sole-check/semantic "
                "classification is unavailable (missing check_only)"
            )
        if failed is None and check_only:
            entry["verdict"] = "unresolved"
            entry.pop("path", None)
            entry["reason"] = (
                f"Unsupported regression claim: {', '.join(shlex.join(cmd) for cmd, _ in commands)} "
                "passed under repository configuration. "
                f"Original finding remains unconfirmed. Verifier claim: {entry.get('reason') or ''}"
            )
        output.append(entry)
    return output
