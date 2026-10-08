"""Shared atomic JSON/byte writers and tolerant model-output JSON extraction."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import suppress
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator


def _fsync_directory(path: Path) -> None:
    """Fsync the parent directory to persist a published rename."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_file(path: Path) -> None:
    """Fsync file content before publication."""
    with open(path, "rb", buffering=0) as f:
        os.fsync(f.fileno())


# Serialises the read-modify-restore fallback in ``_read_umask`` on platforms
# that do not expose the umask without mutating it.
_UMASK_LOCK = threading.Lock()


def _read_umask() -> int:
    """Read umask through /proc when available; otherwise serialize read/restore calls.

    The fallback briefly changes the process umask, so its lock prevents concurrent
    readers of this helper from leaving a corrupted value."""
    try:
        with open("/proc/self/status", "rb") as status:
            for line in status:
                if line.startswith(b"Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    with _UMASK_LOCK:
        current = os.umask(0)
        os.umask(current)
    return current


def umask_derived_mode() -> int:
    """Return 0o666 & ~umask; unlike a fixed mode, this respects restrictive user settings."""
    return 0o666 & ~_read_umask()


def _stage_bytes(path: Path, content: bytes, *, fsync: bool, mode: int | None) -> Path:
    """Write ``content`` to a sibling temp for ``path`` and return the temp path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        return Path(tmp)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def _publish_staged(path: Path, tmp: Path, *, dir_fsync: bool, mode: int | None) -> None:
    """Rename a staged temp into ``path``; remove it if the rename fails."""
    try:
        os.replace(tmp, path)
        if mode is not None:
            os.chmod(path, mode)
        if dir_fsync:
            _fsync_directory(path.parent)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_bytes(
    path: Path,
    content: bytes,
    *,
    fsync: bool = True,
    dir_fsync: bool = False,
    mode: int | None = None,
) -> None:
    """Publish bytes through an exclusive sibling temp and atomic rename.

    Create parents and clean failed temps best-effort. fsync flushes staged bytes;
    mode chmods both temp and final path; dir_fsync persists the parent rename."""
    tmp = _stage_bytes(path, content, fsync=fsync, mode=mode)
    _publish_staged(path, tmp, dir_fsync=dir_fsync, mode=mode)


def atomic_write_pair(
    first: tuple[Path, bytes],
    second: tuple[Path, bytes],
    *,
    fsync: bool = True,
    dir_fsync: bool = False,
    mode: int | None = None,
) -> None:
    """Stage both payloads before publishing either, then rename first and second.

    Each rename is atomic; a failure after the first can leave a mixed pair. Staging
    failures preserve both destinations, and failed temps are cleaned best-effort."""
    first_path, first_content = first
    second_path, second_content = second
    first_tmp = _stage_bytes(first_path, first_content, fsync=fsync, mode=mode)
    try:
        second_tmp = _stage_bytes(second_path, second_content, fsync=fsync, mode=mode)
    except BaseException:
        with suppress(OSError):
            os.unlink(first_tmp)
        raise
    try:
        _publish_staged(first_path, first_tmp, dir_fsync=dir_fsync, mode=mode)
    except BaseException:
        with suppress(OSError):
            os.unlink(second_tmp)
        raise
    _publish_staged(second_path, second_tmp, dir_fsync=dir_fsync, mode=mode)


def atomic_write_json(
    path: Path,
    data: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    default: Callable[[Any], Any] | None = None,
    trailing_newline: bool = False,
    fsync: bool = True,
    dir_fsync: bool = False,
    mode: int | None = None,
) -> None:
    """Serialize JSON, optionally add a newline, and publish via atomic_write_bytes."""
    text = json.dumps(data, indent=indent, sort_keys=sort_keys, default=default)
    if trailing_newline:
        text += "\n"
    atomic_write_bytes(path, text.encode("utf-8"), fsync=fsync, dir_fsync=dir_fsync, mode=mode)


def dataclass_payload(value: Any) -> dict[str, Any]:
    """Project a dataclass tree to dictionaries, converting tuple fields to JSON arrays."""
    return asdict(value, dict_factory=lambda fields: {
        name: list(item) if isinstance(item, tuple) else item
        for name, item in fields
    })


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read an object; absent, unreadable, undecodable, malformed, or non-object data yields {}."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def string_list(value: object) -> list[str]:
    """Return the non-empty strings in *value*, or ``[]`` when it is not a list."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def extract_json(text: str) -> Any:
    """Decode stripped/fence-unwrapped JSON, else return the largest valid object/array.

    The largest span favors structured output over incidental prose brackets such as
    metadata["sender"]. Clean JSON scalars are accepted; no parseable span yields None."""
    candidates = _json_candidates(text)
    if not candidates:
        return None
    return _largest_span(candidates)[1]


def validates_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Return whether ``value`` validates against ``schema`` (the strict full gate)."""
    return not any(Draft202012Validator(schema).iter_errors(value))


@dataclass(frozen=True)
class SchemaRejection:
    """Bounded validator metadata whose path belongs to the host schema, never output."""

    category: str
    schema_path: str
    error_count: int
    candidate_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def schema_rejection(value: Any, schema: dict[str, Any], *, candidate_count: int = 1) -> SchemaRejection | None:
    """Only a complete object candidate can qualify for a fresh strict-schema attempt."""
    if not isinstance(value, dict):
        return None
    errors = list(islice(Draft202012Validator(schema).iter_errors(value), 128))
    if not errors:
        return None
    first = errors[0]
    # absolute_schema_path contains only host-authored schema keys and indices;
    # unlike json_path, it cannot carry arbitrary unknown output property names.
    path = '/'.join(str(part) for part in first.absolute_schema_path)
    return SchemaRejection(str(first.validator), path[:512], min(len(errors), 128), min(candidate_count, 128))


@dataclass(frozen=True)
class SchemaAwareSelection:
    """One schema-aware extraction decision plus a content-free rejection trace.

    ``value`` is the selected candidate, or None when nothing was accepted.
    ``candidate_count`` counts every decoded span the scan enumerated. On rejection,
    ``rejected_type`` is the largest candidate's Python type name and
    ``rejected_reason`` is the first schema error as ``"<validator> at <json_path>"`` —
    never the jsonschema message, which embeds candidate content.
    """

    value: Any | None
    candidate_count: int
    rejected_type: str | None
    rejected_reason: str | None
    rejection: SchemaRejection | None = None
    schema_retry_eligible: bool = False


def _strip_json_fences(text: str) -> str:
    """Strip surrounding whitespace and markdown code fences.

    The candidate-set precondition for ``_json_candidates``: the opening fence
    line (which may carry a language tag like ``json``) and the closing fence
    line are dropped, leaving the payload.
    """
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.split("\n")[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _json_candidates(text: str) -> list[tuple[int, Any, int]]:
    """Enumerate every decodable JSON value in *text* as ``(start, value, span_len)``.

    The single source of truth for the candidate set behind both ``extract_json``
    and ``extract_json_by_schema``: the whole fence-stripped text when it parses,
    then every decodable ``{``/``[`` span, including spans nested inside a root
    that failed to decode. Duplicates are collapsed, so a whole text that parses
    and also matches a zero-offset scan yields one candidate, not two.
    """
    if not text or not text.strip():
        return []

    cleaned = _strip_json_fences(text)
    candidates: list[tuple[int, Any, int]] = []
    try:
        candidates.append((0, json.loads(cleaned), len(cleaned)))
    except ValueError:
        pass

    decoder = json.JSONDecoder()
    for start_char in ("{", "["):
        scan_from = 0
        while True:
            start_idx = cleaned.find(start_char, scan_from)
            if start_idx == -1:
                break
            try:
                parsed, end_idx = decoder.raw_decode(cleaned, start_idx)
            except ValueError:
                # Invalid or unbalanced: a nested span may still be valid JSON.
                scan_from = start_idx + 1
                continue
            if (start_idx, parsed, end_idx - start_idx) not in candidates:
                candidates.append((start_idx, parsed, end_idx - start_idx))
            # A parsed span's nested children can only be smaller, so skipping
            # past it never drops the winner and keeps well-formed scans near-linear.
            scan_from = end_idx
    return candidates


def _largest_span(candidates: list[tuple[int, Any, int]]) -> tuple[int, Any, int]:
    """Pick the longest candidate; a length tie goes to the earlier enumeration slot.

    ``_json_candidates`` emits object spans before array spans and each family in
    document order, so ``max``'s first-wins rule reproduces the long-standing
    preference for a structured object over an equal-length array (``prefix [1]
    then { }`` decodes to ``{}``) without either selector restating the rule."""
    return max(candidates, key=lambda item: item[2])


def extract_json_by_schema(
    text: str,
    *,
    schema: dict[str, Any],
    accept: Callable[[Any, dict[str, Any]], bool],
    require_complete_root: bool = False,
    rejection_guard: Callable[[Any], bool] | None = None,
) -> SchemaAwareSelection:
    """Return the *last* candidate in document order that ``accept`` admits.

    Candidates come from ``_json_candidates`` — the same enumeration
    ``extract_json`` uses. Selection is schema-driven rather than size-driven, so
    a trailing empty result outranks a larger incidental object. A whole text
    that parses but is rejected falls through to the scan.
    """
    if require_complete_root:
        try:
            root = json.loads(_strip_json_fences(text))
        except (ValueError, TypeError):
            return SchemaAwareSelection(None, 0, None, None)
        if accept(root, schema):
            return SchemaAwareSelection(root, 1, None, None)
        rejection = schema_rejection(root, schema)
        reason = f'{rejection.category} at {rejection.schema_path}' if rejection else None
        eligible = rejection is not None and (rejection_guard is None or rejection_guard(root))
        return SchemaAwareSelection(None, 1, type(root).__name__, reason, rejection, eligible)
    candidates = _json_candidates(text)
    if not candidates:
        return SchemaAwareSelection(None, 0, None, None)

    for _, value, _ in reversed(sorted(candidates, key=lambda item: item[0])):
        if accept(value, schema):
            return SchemaAwareSelection(value, len(candidates), None, None)

    # Nothing accepted: trace the largest span (extract_json's own tie-break) with
    # content-free rejection evidence so the caller can report a bounded reason.
    largest = _largest_span(candidates)
    reason = None
    for error in Draft202012Validator(schema).iter_errors(largest[1]):
        reason = f"{error.validator} at {error.json_path}"
        break
    rejection = schema_rejection(largest[1], schema, candidate_count=len(candidates))
    # A nested object decoded from an incomplete outer deliverable is not a
    # complete structured candidate. Only a whole decoded object qualifies.
    try:
        whole = json.loads(_strip_json_fences(text))
    except (ValueError, TypeError):
        rejection = None
    else:
        if not isinstance(whole, dict):
            rejection = None
    return SchemaAwareSelection(None, len(candidates), type(largest[1]).__name__, reason, rejection)
