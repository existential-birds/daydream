"""Shared atomic JSON/byte writers and tolerant model-output JSON extraction."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable


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
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Strip markdown code fences: ```json\n...\n``` or ```\n...\n```
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop the opening fence line (may include language tag like "json")
        lines = lines[1:]
        # Drop the closing fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Fast path — the entire text is valid JSON.
    try:
        return json.loads(cleaned)
    except ValueError:
        pass

    # Prefer the largest embedded payload over incidental prose brackets.
    best: Any = None
    best_len = 0
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
            span_len = end_idx - start_idx
            if span_len > best_len:
                best = parsed
                best_len = span_len
            # A parsed span's nested children can only be smaller, so skipping
            # past it never drops the winner and keeps well-formed scans near-linear.
            scan_from = end_idx

    return best
