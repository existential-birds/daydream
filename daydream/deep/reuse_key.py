"""Pure content keys for completed deep review units.

Keys cover subject and contract: assigned hunks/files/blobs/records, format,
profile, model, effort, and prompt flags. Re-derived exploration, intent,
alternatives, and settled decisions are grounding: recorded and compared on
hits, but excluded from keys. Missing required components are None and prevent
a key; missing grounding is the valid literal "absent".
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

from daydream.trajectory import RUNS_DIRNAME

#: Bump whenever a payload shape changes so a stale entry can never be served.
REUSE_KEY_FORMAT: int = 2

#: A run-scoped artifact segment (``/runs/<id>/...``) inside any digest input.
#: The directory name comes from the layout surface, never a bare literal, so
#: it stays in lockstep with the recorder's own run-document layout.
_RUN_SEGMENT_RE = re.compile(rf"/{RUNS_DIRNAME}/[^/\s]+/")
#: A bare session/run UUID anywhere in a digest input.
_SESSION_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

_ABSENT = "absent"


@dataclass(frozen=True)
class PhaseIdentity:
    """Resolved phase contract; backend is informational, other fields participate in keys."""

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
    """Remove run IDs, artifact-run segments, run_root, and cwd from digest inputs.

    Stored payloads are restored verbatim; normalization affects only their keys.
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


def _canonical_json(value: Any) -> str:
    """The module's canonical JSON encoding for digest inputs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


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
    return digest_text(_canonical_json(entries))


def diff_blocks_digest(full_diff: str, files: Sequence[str]) -> str | None:
    """Digest the exact inline diff text a prompt would carry for ``files``.

    Returns ``None`` when the blocks are unavailable (no match, or over the
    inline byte budget) so the caller can fall back to the hunk index.
    """
    from daydream.deep.diff import _diff_blocks_for_files

    blocks = _diff_blocks_for_files(full_diff, list(files))
    if blocks is None:
        return None
    return digest_text(blocks)


def hunk_slice_digest(hunk_index: Mapping[str, Any], files: Sequence[str]) -> str | None:
    """Hash exactly the requested persisted hunks; an empty or incomplete slice is absent."""
    wanted = sorted(set(files))
    if not wanted:
        return None
    selected: dict[str, Any] = {}
    for name in wanted:
        if name not in hunk_index:
            return None
        selected[name] = hunk_index[name]
    return digest_text(_canonical_json(selected))


def exploration_digest(exploration_dir: str | Path | None) -> str:
    """Hash pre-scan files except cache-key; missing/unreadable content yields "absent"."""
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
    return digest_text(_canonical_json(entries))


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
    canonical = _canonical_json(keyed)
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


def digest_or_absent(text: str | None) -> dict[str, str]:
    """Grounding row for optional text: a content digest or the absent sentinel."""
    if text is None:
        return {"digest": _ABSENT}
    return {"digest": digest_text(text)}


def _identity_components(identity: PhaseIdentity) -> dict[str, Any]:
    """The ``profile``/``model``/``effort`` contract rows every unit keys on."""
    return {
        "profile": identity.profile_digest,
        "model": identity.model,
        "effort": identity.effort if identity.effort else "default",
    }


def _record_digests(
    contributing_records: Mapping[str, bytes | None],
) -> dict[str, str] | None:
    """Hash every contributing record, or None if any record is unreadable."""
    if any(data is None for data in contributing_records.values()):
        return None
    return {
        name: digest_bytes(data)
        for name, data in contributing_records.items()
        if data is not None
    }


def _unit_payload(
    unit: str,
    components: dict[str, Any],
    grounding: Mapping[str, Any],
) -> dict[str, Any]:
    """Wrap ``components``/``grounding`` in the four-key payload envelope.

    The format stamp lives on the envelope and is stamped into ``components`` here so
    the five unit builders cannot drift from it; no reader keys on the copy.
    """
    return {
        "format": REUSE_KEY_FORMAT,
        "unit": unit,
        "components": {"format": REUSE_KEY_FORMAT, **components},
        "grounding": dict(grounding),
    }


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
    from daydream.review_investigation import STAGED_REVIEW_CONTRACT

    ordered_files = sorted(set(files))
    ordered_frontier = sorted(set(frontier_files))
    exploration_present = exploration_dir is not None and Path(exploration_dir).is_dir()
    components: dict[str, Any] = {
        "hunk_slice": _hunk_slice_component(diff_path_or_hunks, hunk_index, ordered_files),
        "assigned_files": ordered_files,
        "assigned_blobs": blob_map_digest(worktree_root, ordered_files),
        "frontier_files": ordered_frontier,
        "frontier_blobs": blob_map_digest(worktree_root, ordered_frontier),
        **_identity_components(identity),
        "intent_authoritative": intent_authoritative,
        "include_alternatives": include_alternatives,
        "exploration_present": exploration_present,
        "docs_only": docs_only,
        "schema": _phases_schema_digest("PER_STACK_RECORD_SCHEMA"),
        "staged_review_contract": STAGED_REVIEW_CONTRACT,
    }
    grounding = {
        "exploration": {"digest": exploration_digest(exploration_dir)},
        "intent": digest_or_absent(intent_text),
        "alternatives": digest_or_absent(alternatives_text),
        "settled_decisions": digest_or_absent(prior_commits),
    }
    return _unit_payload(f"shard:{stack_name}", components, grounding)


def intent_key_payload(
    *,
    diff_text: str,
    commit_log: str,
    exploration_dir: str | Path | None,
    pr_description: str | None,
    branch_name: str,
    worktree_root: str | Path,
    identity: PhaseIdentity,
) -> dict[str, Any]:
    """Key diff, commit log, branch, and PR description; record exploration as grounding.

    worktree_root normalizes the diff's run paths and is never stored in the payload.
    """
    components: dict[str, Any] = {
        "diff": digest_text(normalize_run_scoped(diff_text, run_root=worktree_root)),
        "commit_log": digest_text(commit_log),
        "branch_name": digest_text(branch_name),
        "pr_description": (
            _ABSENT if pr_description is None else digest_text(pr_description)
        ),
        **_identity_components(identity),
    }
    return _unit_payload(
        "intent",
        components,
        {"exploration": {"digest": exploration_digest(exploration_dir)}},
    )


def wonder_key_payload(
    *,
    diff_text: str,
    horse_mode: bool,
    identity: PhaseIdentity,
    grounding: Mapping[str, Any],
) -> dict[str, Any]:
    """Key diff and horse_mode; intent and other loop grounding stay outside the key."""
    components: dict[str, Any] = {
        "diff": digest_text(diff_text),
        "horse_mode": horse_mode,
        **_identity_components(identity),
    }
    return _unit_payload("alternatives", components, grounding)


def _schema_digest(schema: Any) -> str:
    """Hash the strict schema contract so changed schemas invalidate cached output."""
    return digest_text(_canonical_json(schema))


def _phases_schema_digest(name: str) -> str:
    """Read phase schemas lazily to avoid the phase/reuse import cycle."""
    from daydream import phases

    return _schema_digest(getattr(phases, name))



def arbiter_key_payload(
    *,
    contributing_records: Mapping[str, bytes | None],
    structural_records: bytes | None,
    plan: Mapping[str, Any],
    precision_mode: bool,
    identity: PhaseIdentity,
    grounding: Mapping[str, Any],
) -> dict[str, Any]:
    """Key all input records, the group plan, precision mode, schema, and phase identity.

    One unreadable contributing file prevents reuse. An absent structural stack
    uses the valid "absent" sentinel. Grounding is recorded but not keyed; per-group
    completion identity remains the group markers' responsibility.
    """
    # A record that could not be read makes the whole component absent, so
    # ``unit_key`` returns ``None`` (a named miss) rather than keying the subset
    # that did read.
    components: dict[str, Any] = {
        "records": _record_digests(contributing_records),
        "structural": (
            _ABSENT if structural_records is None else digest_bytes(structural_records)
        ),
        "plan": dict(plan),
        "precision_mode": precision_mode,
        "schema": _phases_schema_digest("ARBITER_SCHEMA"),
        **_identity_components(identity),
    }
    return _unit_payload("arbiter", components, grounding)


def merge_key_payload(
    *,
    contributing_records: Mapping[str, bytes | None],
    structural_records_present: bool,
    failed_stacks: Sequence[str],
    identity: PhaseIdentity,
    grounding: Mapping[str, Any],
) -> dict[str, Any]:
    """Key all input records, structural presence, uncovered stacks, schema, and identity.

    One unreadable contributing file prevents reuse. Empty failed_stacks is valid;
    loop grounding is recorded but never keyed.
    """
    components: dict[str, Any] = {
        "records": _record_digests(contributing_records),
        "structural": structural_records_present,
        "failed_stacks": sorted(failed_stacks),
        "schema": _phases_schema_digest("MERGED_ITEMS_SCHEMA"),
        **_identity_components(identity),
    }
    return _unit_payload("merge", components, grounding)


def phase_identity_for(ctx: Any, phase: str) -> PhaseIdentity:
    """Resolve ``phase``'s contract identity from the live flow context."""
    from daydream.review_profile import build_default_profile
    from daydream.run_config import _resolved_backend_name, _resolved_reasoning_effort

    backend = ctx.backend_for(phase)
    profile = getattr(ctx, "review_profile", None)
    profile_digest = profile.digest if profile is not None else build_default_profile().digest
    return PhaseIdentity(
        backend=_resolved_backend_name(ctx.config, phase),
        model=getattr(backend, "model", ""),
        effort=_resolved_reasoning_effort(ctx.config, phase),
        profile_digest=profile_digest,
    )
