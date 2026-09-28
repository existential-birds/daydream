"""Enumerated per-unit reuse-key payload for the deep review pipeline.

A re-review may restore a completed review unit byte-for-byte when the unit's
review **subject** and **contract** are unchanged. This module builds the pure
payload that decides that, and keeps the distinction structural rather than a
convention:

* ``components`` — the subject (assigned hunks, assigned/frontier file names and
  their worktree blob hashes, contributing record bytes) and the contract
  (cache format version, resolved ``ReviewProfile.digest``, resolved model,
  resolved effort, prompt-affecting run flags). ``unit_key`` hashes exactly
  ``{"format", "unit", "components"}``.
* ``grounding`` — the inputs the fix loop re-derives every iteration (the
  exploration pre-scan, intent, alternatives, the settled-decisions block).
  Their digests are *recorded and compared on every hit* but are never read by
  ``unit_key``, so a grounding move can never move a key (MH16).

A required component that is absent or unreadable becomes ``None`` and
``unit_key`` then returns ``None`` — a named miss (``absent_components``), never
a placeholder digest. Grounding absence is a value (the literal ``"absent"``),
not a miss.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

#: Bump whenever a payload shape changes so a stale entry can never be served.
REUSE_KEY_FORMAT: int = 1

#: A run-scoped artifact segment (``/runs/<id>/...``) inside any digest input.
_RUN_SEGMENT_RE = re.compile(r"/runs/[^/\s]+/")
#: A bare session/run UUID anywhere in a digest input.
_SESSION_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

_ABSENT = "absent"


@dataclass(frozen=True)
class PhaseIdentity:
    """The resolved contract identity of one model-bearing phase.

    ``backend`` is informational; ``model``, ``effort`` and ``profile_digest``
    are the contract rows keyed per unit.
    """

    backend: str
    model: str
    effort: str | None
    profile_digest: str


@overload
def normalize_run_scoped(
    text_or_bytes: str,
    *,
    session_id: str | None = None,
    run_root: str | Path | None = None,
) -> str: ...


@overload
def normalize_run_scoped(
    text_or_bytes: bytes,
    *,
    session_id: str | None = None,
    run_root: str | Path | None = None,
) -> bytes: ...


def normalize_run_scoped(
    text_or_bytes: str | bytes,
    *,
    session_id: str | None = None,
    run_root: str | Path | None = None,
) -> str | bytes:
    """Strip run-scoped identifiers from a digest input (A9).

    Removes the ``/runs/<id>/`` artifact segment, a known ``session_id``, a known
    ``run_root`` prefix and the current working directory. Normalization is
    digest-only: stored payload bytes are always restored verbatim.
    """
    if isinstance(text_or_bytes, bytes):
        text: str = text_or_bytes.decode("utf-8", "surrogateescape")
    else:
        text = text_or_bytes
    roots: list[Path] = []
    if run_root is not None:
        roots.append(Path(run_root))
    try:
        roots.append(Path.cwd())
    except OSError:  # pragma: no cover - cwd is always resolvable in practice
        pass
    for root in roots:
        root_text = str(root)
        if len(root_text) <= 1:
            continue
        text = text.replace(root_text.rstrip("/") + "/", "")
    if session_id:
        text = text.replace(session_id, "")
    text = _RUN_SEGMENT_RE.sub("/", text)
    text = _SESSION_RE.sub("", text)
    if isinstance(text_or_bytes, bytes):
        return text.encode("utf-8", "surrogateescape")
    return text


def _normalized_bytes(value: str | bytes) -> bytes:
    normalized = normalize_run_scoped(value)
    return normalized if isinstance(normalized, bytes) else normalized.encode("utf-8")


def digest_text(text: str) -> str:
    """SHA-256 over run-scoped-normalized text."""
    return hashlib.sha256(normalize_run_scoped(text).encode("utf-8")).hexdigest()


def digest_bytes(data: bytes) -> str:
    """SHA-256 over run-scoped-normalized bytes."""
    return hashlib.sha256(_normalized_bytes(data)).hexdigest()


def digest_file(path: Path) -> str | None:
    """SHA-256 of a file's run-scoped-normalized bytes, or ``None`` on failure."""
    try:
        return digest_bytes(Path(path).read_bytes())
    except OSError:
        return None


def blob_map_digest(worktree_root: str | Path, files: Sequence[str]) -> str | None:
    """Digest a canonical ``relative_path -> sha256(path + b"\\0" + bytes)`` map.

    Files are read from ``worktree_root``; any unreadable file makes the whole
    component ``None`` (a miss), never a placeholder. An empty file set hashes
    the empty map, which is a real value (a shard with no frontier still hits).
    """
    root = Path(worktree_root)
    entries: dict[str, str] = {}
    for relative in sorted(set(files)):
        try:
            payload = root.joinpath(relative).read_bytes()
        except OSError:
            return None
        entries[relative] = hashlib.sha256(
            relative.encode("utf-8") + b"\0" + payload
        ).hexdigest()
    return digest_text(json.dumps(entries, sort_keys=True, separators=(",", ":")))


def diff_blocks_digest(full_diff: str, files: Sequence[str]) -> str | None:
    """Digest the exact inline diff text a prompt would carry for ``files``.

    Returns ``None`` when the blocks are unavailable (no match, or over the
    inline byte budget) so the caller can fall back to the hunk index.
    """
    from daydream.deep.prompts import _diff_blocks_for_files

    blocks = _diff_blocks_for_files(full_diff, list(files))
    if blocks is None:
        return None
    return digest_text(blocks)


def hunk_slice_digest(hunk_index: Mapping[str, Any], files: Sequence[str]) -> str | None:
    """Digest the persisted hunk-index entries for exactly ``files``.

    A requested file missing from the index makes the slice ``None`` — the
    subject is not fully known, so the unit must miss rather than key a
    partial slice.
    """
    wanted = sorted(set(files))
    if not wanted:
        return None
    selected: dict[str, Any] = {}
    for name in wanted:
        if name not in hunk_index:
            return None
        selected[name] = hunk_index[name]
    return digest_text(json.dumps(selected, sort_keys=True, separators=(",", ":")))


def exploration_digest(exploration_dir: str | Path | None) -> str:
    """Digest the exploration pre-scan directory's content, excluding ``cache-key``.

    ``cache-key`` records how the directory was produced and must never enter a
    digest (spec Constraints). A missing directory or unreadable file is the
    ``"absent"`` sentinel.
    """
    if exploration_dir is None:
        return _ABSENT
    root = Path(exploration_dir)
    if not root.is_dir():
        return _ABSENT
    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "cache-key":
            continue
        try:
            entries[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return _ABSENT
    return digest_text(json.dumps(entries, sort_keys=True, separators=(",", ":")))


def grounding_digests(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return the recorded grounding map as ``name -> digest`` (``"absent"`` allowed)."""
    grounding = payload.get("grounding", {})
    return {
        name: (spec or {}).get("digest", _ABSENT)
        for name, spec in grounding.items()
    }


def absent_components(payload: Mapping[str, Any]) -> list[str]:
    """Names of required components that are absent (``None``), sorted."""
    components = payload.get("components", {})
    return sorted(name for name, value in components.items() if value is None)


def unit_key(payload: Mapping[str, Any]) -> str | None:
    """The content key, or ``None`` when any required component is absent.

    Hashes only ``{"format", "unit", "components"}``; ``grounding`` is never
    read here.
    """
    components = payload.get("components", {})
    if any(value is None for value in components.values()):
        return None
    keyed = {
        "format": payload.get("format"),
        "unit": payload.get("unit"),
        "components": components,
    }
    canonical = json.dumps(keyed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _diff_text(source: Any) -> str | None:
    """Read a diff source (text/path) into text, or ``None`` when unavailable."""
    if source is None:
        return None
    if isinstance(source, Mapping):
        return None
    if isinstance(source, Path):
        try:
            return source.read_text(encoding="utf-8")
        except OSError:
            return None
    if isinstance(source, (bytes, bytearray)):
        return bytes(source).decode("utf-8", "replace")
    return str(source)


def _hunk_slice_component(
    diff_path_or_hunks: Any, hunk_index: Mapping[str, Any], files: Sequence[str]
) -> str | None:
    """The assigned-hunk component: inline diff blocks, else persisted hunks."""
    if isinstance(diff_path_or_hunks, Mapping):
        return hunk_slice_digest(diff_path_or_hunks, files)
    text = _diff_text(diff_path_or_hunks)
    if text is not None:
        digest = diff_blocks_digest(text, files)
        if digest is not None:
            return digest
    return hunk_slice_digest(hunk_index, files)


def _digest_or_absent(text: str | None) -> dict[str, str]:
    if text is None:
        return {"digest": _ABSENT}
    return {"digest": digest_text(text)}


def shard_key_payload(
    *,
    stack_name: str,
    files: Sequence[str],
    frontier_files: Sequence[str],
    docs_only: bool,
    diff_path_or_hunks: Any,
    hunk_index: Mapping[str, Any],
    exploration_dir: str | Path | None,
    worktree_root: str | Path,
    identity: PhaseIdentity,
    intent_authoritative: bool,
    include_alternatives: bool,
    prior_commits: str | None,
    intent_text: str | None,
    alternatives_text: str | None,
) -> dict[str, Any]:
    """Build the per-stack (or structural) shard payload.

    ``exploration_present`` is a *contract* flag (whether this run has a
    pre-scan at all); the pre-scan's **content** is the ``exploration``
    *grounding* row.
    """
    ordered_files = sorted(set(files))
    ordered_frontier = sorted(set(frontier_files))
    exploration_present = exploration_dir is not None and Path(exploration_dir).is_dir()
    components: dict[str, Any] = {
        "format": REUSE_KEY_FORMAT,
        "hunk_slice": _hunk_slice_component(diff_path_or_hunks, hunk_index, ordered_files),
        "assigned_files": ordered_files,
        "assigned_blobs": blob_map_digest(worktree_root, ordered_files),
        "frontier_files": ordered_frontier,
        "frontier_blobs": blob_map_digest(worktree_root, ordered_frontier),
        "profile": identity.profile_digest,
        "model": identity.model,
        "effort": identity.effort if identity.effort else "default",
        "intent_authoritative": intent_authoritative,
        "include_alternatives": include_alternatives,
        "exploration_present": exploration_present,
        "docs_only": docs_only,
    }
    grounding = {
        "exploration": {"digest": exploration_digest(exploration_dir)},
        "intent": _digest_or_absent(intent_text),
        "alternatives": _digest_or_absent(alternatives_text),
        "settled_decisions": _digest_or_absent(prior_commits),
    }
    return {
        "format": REUSE_KEY_FORMAT,
        "unit": f"shard:{stack_name}",
        "components": components,
        "grounding": grounding,
    }


def phase_identity_for(ctx: Any, phase: str) -> PhaseIdentity:
    """Resolve ``phase``'s contract identity from the live flow context."""
    from daydream.review_profile import build_default_profile
    from daydream.runner import _resolved_backend_name, _resolved_reasoning_effort

    backend = ctx.backend_for(phase)
    profile = getattr(ctx, "review_profile", None)
    profile_digest = profile.digest if profile is not None else build_default_profile().digest
    return PhaseIdentity(
        backend=_resolved_backend_name(ctx.config, phase),
        model=getattr(backend, "model", ""),
        effort=_resolved_reasoning_effort(ctx.config, phase),
        profile_digest=profile_digest,
    )
