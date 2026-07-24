"""Deterministic host enumeration of repository build/test/lint commands.

Make targets and package-manifest scripts are fully enumerable from disk, so
the host derives those command records itself instead of asking a model to
cite evidence for them and then re-deriving the same invocation to check the
citation. Only prose-sourced commands (a README, a CI workflow) still need a
model-authored ``literal-command`` evidence citation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from daydream.improve.command_contract import (
    HOST_EVIDENCE_KIND,
    path_is_confined,
)

_MAKE_DECLARATION = re.compile(
    r"^(?P<targets>[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:[ \t]+[A-Za-z0-9][A-Za-z0-9_.-]*)*)[ \t]*(?P<sep>::?)(?P<assign>=?)"
)
_SCRIPT_KEY = re.compile(r'^\s*"(?P<key>(?:[^"\\]|\\.)*)"\s*:')
_ID_SEPARATOR = re.compile(r"[^a-z0-9]+")
_COMMAND_NAME_WORDS = re.compile(r"[^a-z0-9]+")
_VERIFICATION_WORDS = frozenset(
    {
        "build",
        "check",
        "ci",
        "compile",
        "lint",
        "test",
        "tests",
        "typecheck",
        "validate",
        "validation",
        "verify",
        "verification",
    }
)
_MUTATING_WORDS = frozenset(
    {
        "bootstrap",
        "clean",
        "deploy",
        "format",
        "generate",
        "generation",
        "hook",
        "hooks",
        "init",
        "install",
        "postinstall",
        "postpublish",
        "preinstall",
        "prepare",
        "publish",
        "release",
        "report",
        "setup",
    }
)
_LOCKFILE_MANAGERS = (
    ("pnpm-lock.yaml", "pnpm"),
    ("yarn.lock", "yarn"),
    ("bun.lockb", "bun"),
    ("bun.lock", "bun"),
    ("package-lock.json", "npm"),
)


def package_invocation(package_manager: str, script: str) -> str:
    """Return the manager-specific invocation for a manifest script."""
    if package_manager == "npm":
        return f"npm run {script}"
    if package_manager == "bun":
        return f"bun run {script}"
    return f"{package_manager} {script}"


def _is_verification_name(name: str) -> bool:
    """Return whether a declared target or script is safe to offer as a gate.

    Declarations do not expose their runtime effects, so host discovery is
    deliberately conservative: require a verification-oriented name and
    exclude names that conventionally perform repository mutation.
    """
    words = {word for word in _COMMAND_NAME_WORDS.split(name.lower()) if word}
    return bool(words & _VERIFICATION_WORDS) and not bool(words & _MUTATING_WORDS)


def enumerate_repository_commands(
    repo: Path,
    *,
    directories: Sequence[str] = (".",),
    reserved_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return recon command records derived from Makefiles and manifests.

    ``directories`` are repository-relative working directories (``"."`` for
    the repository root). ``reserved_ids`` are ids already spoken for by
    model-supplied records, so a derived id never shadows one of those. Output
    is deterministic: directories in the given order, Makefile targets then
    manifest scripts, each in source order.

    Reads only. Every source file is confinement-checked before it is opened,
    so a directory or symlink pointing outside ``repo`` yields nothing.
    """
    records: list[dict[str, Any]] = []
    used_ids: set[str] = set(reserved_ids)
    for directory in directories:
        base = repo if directory == "." else repo / directory
        for record in (
            *_make_records(repo, base, directory),
            *_script_records(repo, base, directory),
        ):
            record["id"] = _unique_id(record["id"], used_ids)
            records.append(record)
    return records


def _make_records(
    repo: Path, base: Path, directory: str
) -> Iterable[dict[str, Any]]:
    source_path = _relative(directory, "Makefile")
    if not path_is_confined(repo, source_path):
        return
    text = _read_text(base / "Makefile")
    if text is None:
        return
    seen: set[str] = set()
    for index, line in enumerate(text.splitlines(), start=1):
        match = _MAKE_DECLARATION.match(line)
        if match is None or match.group("assign"):
            continue
        for target in match.group("targets").split():
            if target in seen or not _is_verification_name(target):
                continue
            seen.add(target)
            yield _record(
                command_id=f"make-{_slug(target)}",
                command=f"make {target}",
                purpose=f"Run the Make target {target}",
                observable_result=(
                    f"exit 0 and the {target} Make target completes"
                ),
                directory=directory,
                source_path=source_path,
                line=index,
                excerpt=line,
            )


def _script_records(
    repo: Path, base: Path, directory: str
) -> Iterable[dict[str, Any]]:
    source_path = _relative(directory, "package.json")
    if not path_is_confined(repo, source_path):
        return
    text = _read_text(base / "package.json")
    if text is None:
        return
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(manifest, dict):
        return
    scripts = manifest.get("scripts")
    if not isinstance(scripts, dict):
        return
    manager = _package_manager(base, manifest)
    lines = text.splitlines()
    for script, body in scripts.items():
        if (
            not isinstance(script, str)
            or not isinstance(body, str)
            or not _is_verification_name(script)
        ):
            continue
        line = _script_line(lines, script)
        yield _record(
            command_id=f"{manager}-{_slug(script)}",
            command=package_invocation(manager, script),
            purpose=f"Run the {script} package script",
            observable_result=(
                f"exit 0 and the {script} package script completes"
            ),
            directory=directory,
            source_path=source_path,
            line=line,
            excerpt=lines[line - 1],
        )


def _record(
    *,
    command_id: str,
    command: str,
    purpose: str,
    observable_result: str,
    directory: str,
    source_path: str,
    line: int,
    excerpt: str,
) -> dict[str, Any]:
    scope: dict[str, Any] = (
        {"kind": "whole-repository"}
        if directory == "."
        else {"kind": "in-scope-paths", "paths": [directory]}
    )
    return {
        "id": command_id,
        "purpose": purpose,
        "command": command,
        "working_directory": directory,
        "expected_success": {
            "exit_code": 0,
            "observable_result": observable_result,
        },
        "applicability": {
            "scope": scope,
            "preconditions": [],
            "rationale": (
                f"The host read this command declaration from {source_path}."
            ),
        },
        "evidence": {
            "kind": HOST_EVIDENCE_KIND,
            "source_path": source_path,
            "line_anchor": {"start_line": line, "end_line": line},
            "verbatim_excerpt": excerpt,
        },
    }


def _package_manager(base: Path, manifest: dict[str, Any]) -> str:
    declaration = manifest.get("packageManager")
    if isinstance(declaration, str):
        name = declaration.partition("@")[0]
        if name in {"npm", "pnpm", "yarn", "bun"}:
            return name
    for lockfile, manager in _LOCKFILE_MANAGERS:
        if (base / lockfile).is_file():
            return manager
    return "npm"


def _script_line(lines: list[str], script: str) -> int:
    encoded = json.dumps(script)[1:-1]
    for index, line in enumerate(lines, start=1):
        match = _SCRIPT_KEY.match(line)
        if match is not None and match.group("key") == encoded:
            return index
    for index, line in enumerate(lines, start=1):
        if '"scripts"' in line:
            return index
    return 1


def _relative(directory: str, name: str) -> str:
    return name if directory == "." else f"{directory}/{name}"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _slug(value: str) -> str:
    return _ID_SEPARATOR.sub("-", value.casefold()).strip("-") or "command"


def _unique_id(candidate: str, used: set[str]) -> str:
    unique = candidate
    suffix = 2
    while unique in used:
        unique = f"{candidate}-{suffix}"
        suffix += 1
    used.add(unique)
    return unique


__all__ = [
    "enumerate_repository_commands",
    "package_invocation",
]
