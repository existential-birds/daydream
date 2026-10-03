"""Discover complete sessions and validate normalized staging paths."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from daydream.archive.hydrate_types import StageError
from daydream.redaction import redact_text
from daydream.trajectory import RUN_DOCUMENT_NAME

_REQUIRED_SESSION_ARTIFACTS = frozenset(("manifest.json", RUN_DOCUMENT_NAME))


_DERIVED_ARCHIVE_ROOTS = frozenset(("annotations", "curated"))


# Never discover bronze raw ingestion or derived publication trees.
_RESERVED_CANONICAL_ROOTS = frozenset(("annotations", "bronze", "bundle", "bundles", "curated"))


@dataclass(frozen=True)
class _DiscoveredSession:
    """One source-layout session identified by its manifest path."""

    session_id: str
    source_root: str
    layout: str


@dataclass(frozen=True)
class _Discovery:
    """Validated snapshot discovery and its normalized staging file map."""

    sessions: tuple[_DiscoveredSession, ...]
    normalized_paths: tuple[tuple[str, str], ...]  # (normalized, source)
    run_shaped_manifests: int
    canonical_manifests: int
    legacy_manifests: int
    incomplete_manifests: tuple[str, ...]


def _source_root_for_path(relpath: str) -> str | None:
    """Return a discovery hint; only a matching manifest and required artifact set admit it as a source root."""
    parts = PurePosixPath(relpath).parts
    if not parts or parts[0] in _DERIVED_ARCHIVE_ROOTS:
        return None
    if parts[0] == "bundles":
        if (
            len(parts) >= 3
            and _is_bare_segment(parts[1])
            and parts[1] not in _RESERVED_CANONICAL_ROOTS
        ):
            return f"bundles/{parts[1]}"
        return None
    if len(parts) >= 2 and _is_bare_segment(parts[0]) and parts[0] not in _RESERVED_CANONICAL_ROOTS:
        return parts[0]
    return None


def _manifest_source_session(relpath: str) -> _DiscoveredSession | None:
    """Recognize the two supported manifest layouts without reading content."""
    parts = PurePosixPath(relpath).parts
    if len(parts) == 2 and parts[1] == "manifest.json":
        session_id = parts[0]
        if _is_bare_segment(session_id) and session_id not in _RESERVED_CANONICAL_ROOTS:
            return _DiscoveredSession(session_id, session_id, "canonical")
    if (
        len(parts) == 3
        and parts[0] == "bundles"
        and parts[2] == "manifest.json"
        and _is_bare_segment(parts[1])
        and parts[1] not in _RESERVED_CANONICAL_ROOTS
    ):
        return _DiscoveredSession(parts[1], f"bundles/{parts[1]}", "legacy")
    return None


def _discover_snapshot(relpaths: list[str]) -> _Discovery:
    """Discover manifest-backed sessions with trajectories; normalize both layouts and reject path collisions."""
    manifest_sessions: dict[str, _DiscoveredSession] = {}
    paths_by_root: dict[str, set[str]] = {}
    canonical_manifests = 0
    legacy_manifests = 0
    for relpath in relpaths:
        session = _manifest_source_session(relpath)
        if session is not None:
            if session.source_root in manifest_sessions:
                raise StageError(
                    redact_text(f"duplicate run-shaped manifest path for {session.source_root!r}")
                )
            manifest_sessions[session.source_root] = session
            if session.layout == "canonical":
                canonical_manifests += 1
            else:
                legacy_manifests += 1
        source_root = _source_root_for_path(relpath)
        if source_root is not None:
            paths_by_root.setdefault(source_root, set()).add(relpath)

    complete: list[_DiscoveredSession] = []
    incomplete: list[str] = []
    for source_root, session in sorted(manifest_sessions.items()):
        available = {
            PurePosixPath(path).relative_to(PurePosixPath(source_root)).as_posix()
            for path in paths_by_root.get(source_root, set())
        }
        missing = sorted(_REQUIRED_SESSION_ARTIFACTS - available)
        if missing:
            incomplete.append(f"{source_root} (missing {', '.join(missing)})")
        else:
            complete.append(session)

    session_id_counts = Counter(session.session_id for session in complete)
    duplicates = sorted(
        session_id for session_id, count in session_id_counts.items() if count > 1
    )
    if duplicates:
        raise StageError(
            redact_text(
                "multiple source layouts normalize to the same session id(s): "
                + ", ".join(repr(value) for value in duplicates)
            )
        )

    normalized_paths: dict[str, str] = {}
    for session in complete:
        source_root = session.source_root
        for source_path in sorted(paths_by_root[source_root]):
            suffix = PurePosixPath(source_path).relative_to(PurePosixPath(source_root)).as_posix()
            normalized = f"bundles/{session.session_id}/{suffix}"
            prior = normalized_paths.get(normalized)
            if prior is not None and prior != source_path:
                raise StageError(
                    redact_text(
                        f"source paths {prior!r} and {source_path!r} collide after normalization "
                        f"at {normalized!r}"
                    )
                )
            normalized_paths[normalized] = source_path

    return _Discovery(
        sessions=tuple(sorted(complete, key=lambda item: item.session_id)),
        normalized_paths=tuple(sorted(normalized_paths.items())),
        run_shaped_manifests=canonical_manifests + legacy_manifests,
        canonical_manifests=canonical_manifests,
        legacy_manifests=legacy_manifests,
        incomplete_manifests=tuple(incomplete),
    )


def _validate_relpath(relpath: str, root: Path) -> Path:
    """Require a relative path strictly within staging before writing; escaping paths raise StageError."""
    p = PurePosixPath(relpath)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise StageError(
            redact_text(
                f"refusing relpath {relpath!r}: traversal — path escapes the staging root"
            )
        )
    target = (root / relpath).resolve()
    if not target.is_relative_to(root.resolve()):
        raise StageError(
            redact_text(
                f"refusing relpath {relpath!r}: traversal — resolved path escapes the staging root"
            )
        )
    return target


def _is_bare_segment(value: str) -> bool:
    """Accept one nonempty path segment, excluding dot segments and either separator."""
    return bool(value) and value not in (".", "..") and "/" not in value and "\\" not in value
