"""Shared JSON utilities.

``extract_json`` is used by backends (structured-output extraction) and
``run_agent`` (raw-text fallback) to robustly pull JSON out of model output
that may be wrapped in reasoning prose or markdown code fences — common with
GLM and other OpenAI-compatible models. ``atomic_write_json`` is the shared
crash-safe JSON file writer (tempfile + ``os.replace``).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync of a directory entry (durable rename publication)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# Serialises the read-modify-restore fallback in ``_read_umask`` on platforms
# that do not expose the umask without mutating it.
_UMASK_LOCK = threading.Lock()


def _read_umask() -> int:
    """Return the process umask without leaving it observable as zero.

    Linux exposes the umask read-only in ``/proc/self/status``, so the common
    path never calls ``os.umask``: the historical ``os.umask(0)`` /
    ``os.umask(current)`` round trip briefly made concurrent file creation in
    this process inherit mode ``0o666`` and let two callers interleave into a
    corrupted value. The fallback for platforms without ``/proc`` still has to
    toggle the umask, so it is serialised under a lock to keep concurrent
    callers from observing a zeroed or corrupted value.
    """
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
    """Return the mode a plain ``open(path, "w")`` / ``Path.write_text`` applies.

    ``mkstemp`` always creates its temp as ``0600`` regardless of the process
    umask, so callers that previously relied on the umask-derived permission
    model (``0o666 & ~umask``) must pass this computed mode instead of a fixed
    ``0o644`` -- otherwise a restrictive umask (e.g. ``077``) is silently
    widened to world-readable.
    """
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
    """Atomically write ``content`` to ``path`` (same-dir temp + ``os.replace``).

    The shared crash-safe write primitive: the temp file is created exclusively
    in ``path``'s directory so the rename never crosses filesystems, parent
    directories are created as needed, and a failure removes the temp file
    best-effort before re-raising. ``fsync`` flushes the file before the rename;
    ``mode`` chmods the temp file before and the final path after the rename
    (umask-immune, covers a pre-existing destination); ``dir_fsync`` fsyncs the
    parent directory after the rename. Callers migrating from prior writers
    pass these knobs explicitly, so they are load-bearing.
    """
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
    """Atomically write a logical pair, staging both payloads before either lands.

    Both temps are written first and only then renamed (``first`` then
    ``second``), so a failure while writing the second payload cannot publish
    the first half of the pair. Every destination is always either its prior
    bytes or its completed new bytes, and on failure no temp survives.
    """
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
    """Atomically write ``data`` as JSON to ``path`` (tempfile + ``os.replace``).

    The temp file lives in ``path``'s directory so the rename never crosses
    filesystems; a crash mid-write leaves either the prior file or nothing.
    Parent directories are created as needed. On failure the temp file is
    removed best-effort and the original exception re-raised. ``fsync``,
    ``dir_fsync``, and ``mode`` forward to :func:`atomic_write_bytes`.
    """
    text = json.dumps(data, indent=indent, sort_keys=sort_keys, default=default)
    if trailing_newline:
        text += "\n"
    atomic_write_bytes(path, text.encode("utf-8"), fsync=fsync, dir_fsync=dir_fsync, mode=mode)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def extract_json(text: str) -> Any:
    """Extract a JSON object or array from possibly prose-wrapped model text.

    Tries, in priority order:

    1. Strip leading/trailing whitespace.
    2. Strip markdown code fences (```` ```json ... ``` ```` or bare ```` ``` ````).
    3. ``json.loads`` on the cleaned text (fast path for clean JSON).
    4. If that fails, scan for every balanced, parseable ``{...}`` or ``[...]``
       span with the JSON decoder and return the LARGEST one. The real
       structured-output payload is a substantial object/array, so size
       disambiguates it from incidental brackets in the surrounding prose — e.g.
       a ``metadata["sender"]`` code snippet, which parses as the tiny list
       ``["sender"]``. Returning the largest span avoids handing a caller that
       expects ``{"findings": [...]}`` a bogus bare list grabbed from prose.

    Returns the parsed value (dict, list, str, int, …) or ``None`` if no valid
    JSON was found. Never raises.
    """
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

    # Slow path — the text is prose with one or more embedded JSON spans. Find
    # EVERY balanced, parseable {...}/[...] span and return the LARGEST one.
    #
    # The largest span is the model's actual answer: a structured-output
    # response is a substantial object/array, whereas stray brackets in the
    # surrounding prose (e.g. a `metadata["sender"]` code snippet, which parses
    # as the one-element list `["sender"]`) are tiny. An earlier-bracket-wins
    # rule would return that incidental `["sender"]` and hand a bogus bare list
    # to a caller expecting `{"findings": [...]}`. Size disambiguates reliably:
    # the real payload dwarfs prose noise.
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
