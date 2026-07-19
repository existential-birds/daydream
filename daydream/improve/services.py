"""Discover service roots for improve-flow monorepo audits."""

from __future__ import annotations

import fnmatch
import glob
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.config_file import DaydreamFileConfig, load_toml_or_empty

_CONVENTIONAL_ROOTS = ("apps", "services", "packages", "crates", "cmd")
_SERVICE_MANIFESTS = ("pyproject.toml", "package.json", "go.mod", "Cargo.toml", "mix.exs")
_PNPM_PACKAGES = re.compile(r"^packages:\s*$")
_PNPM_ITEM = re.compile(r"^\s+-\s+(.+?)\s*$")


@dataclass(frozen=True)
class Service:
    """A repository-relative service root and how it was discovered."""

    name: str
    root: Path
    source: str


def enumerate_services(repo_root: Path, file_config: DaydreamFileConfig) -> list[Service]:
    """Return deterministic service roots declared by config or inferred from layout."""
    repo_root = repo_root.resolve()
    if file_config.improve_service_roots:
        roots = _expand_globs(repo_root, file_config.improve_service_roots)
        return _build_services(repo_root, {root: "config" for root in roots})

    discovered: dict[Path, str] = {}
    signals = (
        ("package.json-workspaces", _package_workspace_patterns(repo_root)),
        ("pnpm-workspace", _pnpm_workspace_patterns(repo_root)),
        ("cargo-workspace", _cargo_workspace_patterns(repo_root)),
        ("go-work", _go_work_patterns(repo_root)),
    )
    for signal, patterns in signals:
        for root in _expand_globs(repo_root, patterns):
            discovered.setdefault(root, f"heuristic:{signal}")

    for root in _conventional_manifest_roots(repo_root):
        discovered.setdefault(root, "heuristic:conventional-root")

    return _build_services(repo_root, discovered)


def filter_scope(services: list[Service], scope: str) -> list[Service]:
    """Filter services by exact name, root path, or a glob over root paths."""
    normalized_scope = scope.removeprefix("./").rstrip("/")
    matched = [
        service
        for service in services
        if service.name == scope
        or service.root.as_posix() == normalized_scope
        or fnmatch.fnmatchcase(service.root.as_posix(), scope)
    ]
    if matched:
        return matched

    known = ", ".join(f"{service.name} ({service.root.as_posix()})" for service in services)
    raise ValueError(f"unknown service scope {scope!r}; known services: {known or '<none>'}")


def _build_services(repo_root: Path, discovered: dict[Path, str]) -> list[Service]:
    ordered = sorted(discovered, key=lambda root: root.as_posix())
    names = Counter(root.name or repo_root.name for root in ordered)
    used: set[str] = set()
    services: list[Service] = []

    for root in ordered:
        base = root.name or repo_root.name
        name = base
        if names[base] > 1:
            parent = root.parent.name or repo_root.parent.name
            name = f"{base}-{parent}"
        unique_name = name
        suffix = 2
        while unique_name in used:
            unique_name = f"{name}-{suffix}"
            suffix += 1
        used.add(unique_name)
        services.append(Service(name=unique_name, root=root, source=discovered[root]))

    return services


def _expand_globs(repo_root: Path, patterns: list[str]) -> list[Path]:
    roots: set[Path] = set()
    for pattern in patterns:
        if not pattern or Path(pattern).is_absolute():
            continue
        for match in glob.iglob(pattern, root_dir=repo_root, recursive=True):
            relative = Path(match)
            candidate = repo_root / relative
            if not candidate.is_dir():
                continue
            try:
                candidate.resolve().relative_to(repo_root)
            except (OSError, ValueError):
                continue
            roots.add(Path(relative.as_posix().rstrip("/")) or Path("."))
    return sorted(roots, key=lambda root: root.as_posix())


def _package_workspace_patterns(repo_root: Path) -> list[str]:
    data = _load_json_or_empty(repo_root / "package.json")
    workspaces = data.get("workspaces")
    if isinstance(workspaces, dict):
        workspaces = workspaces.get("packages")
    return _string_list(workspaces)


def _pnpm_workspace_patterns(repo_root: Path) -> list[str]:
    text = _read_text_or_empty(repo_root / "pnpm-workspace.yaml")
    patterns: list[str] = []
    in_packages = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not in_packages:
            in_packages = bool(_PNPM_PACKAGES.fullmatch(line))
            continue
        if not raw_line[:1].isspace():
            break
        item = _PNPM_ITEM.fullmatch(line)
        if item:
            patterns.append(_unquote(item.group(1)))
    return [pattern for pattern in patterns if pattern]


def _cargo_workspace_patterns(repo_root: Path) -> list[str]:
    data = load_toml_or_empty(repo_root / "Cargo.toml")
    workspace = data.get("workspace")
    if not isinstance(workspace, dict):
        return []
    return _string_list(workspace.get("members"))


def _go_work_patterns(repo_root: Path) -> list[str]:
    text = _read_text_or_empty(repo_root / "go.work")
    patterns: list[str] = []
    in_use_block = False
    for raw_line in text.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        if in_use_block:
            if line == ")":
                in_use_block = False
            else:
                patterns.append(_unquote(line))
            continue
        if not line.startswith("use"):
            continue
        value = line.removeprefix("use").strip()
        if value == "(":
            in_use_block = True
        elif value:
            patterns.append(_unquote(value))
    return [pattern for pattern in patterns if pattern]


def _conventional_manifest_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    for root_name in _CONVENTIONAL_ROOTS:
        parent = repo_root / root_name
        try:
            children = list(parent.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and any((child / manifest).is_file() for manifest in _SERVICE_MANIFESTS):
                roots.append(child.relative_to(repo_root))
    return sorted(roots, key=lambda root: root.as_posix())


def _load_json_or_empty(path: Path) -> dict[str, Any]:
    text = _read_text_or_empty(path)
    if not text:
        return {}
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text()
    except (OSError, UnicodeError):
        return ""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
