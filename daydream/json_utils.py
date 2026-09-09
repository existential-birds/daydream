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


def atomic_write_bytes(
    path: Path,
    content: bytes,
    *,
    fsync: bool = True,
    dir_fsync: bool = False,
    mode: int | None = None,
) -> None:
    """Atomically write ``content`` to ``path`` (same-dir temp + ``os.replace``).

    The single shared crash-safe write primitive. The temp file is created
    exclusively in ``path``'s directory so the rename never crosses
    filesystems; a crash mid-write leaves either the prior file or nothing.
    Parent directories are created as needed. On failure the temp file is
    removed best-effort and the original exception re-raised.

    Knobs let callers preserve (or strengthen) their prior hardening:

    - ``fsync``: flush + fsync the file *before* the rename.
    - ``mode``: when given, the temp file is chmod'ed to this mode before the
      rename and the final path is chmod'ed again after it (the post-rename
      chmod is umask-immune and covers a pre-existing destination).
    - ``dir_fsync``: fsync the parent directory after the rename so the new
      name survives a crash.
    """
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
        os.replace(tmp, path)
        if mode is not None:
            os.chmod(path, mode)
        if dir_fsync:
            _fsync_directory(path.parent)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


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


def extract_json(text: str) -> Any:
    """Extract a JSON object or array from possibly prose-wrapped model text.

    Tries, in priority order:

    1. Strip leading/trailing whitespace.
    2. Strip markdown code fences (```` ```json ... ``` ```` or bare ```` ``` ````).
    3. ``json.loads`` on the cleaned text (fast path for clean JSON).
    4. If that fails, scan for every balanced, parseable ``{...}`` or ``[...]``
       span (depth-counting, string-aware) and return the LARGEST one. The real
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
    for start_char, end_char in (("{", "}"), ("[", "]")):
        scan_from = 0
        while True:
            start_idx = cleaned.find(start_char, scan_from)
            if start_idx == -1:
                break
            depth = 0
            in_string = False
            escape = False
            end_idx = -1
            for i in range(start_idx, len(cleaned)):
                ch = cleaned[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx == -1:
                # Unbalanced from here; advance one char and keep scanning for a
                # later valid span of this brace type.
                scan_from = start_idx + 1
                continue
            span_len = end_idx + 1 - start_idx
            try:
                parsed = json.loads(cleaned[start_idx : end_idx + 1])
            except ValueError:
                parsed = None
            if parsed is not None and span_len > best_len:
                best = parsed
                best_len = span_len
            if parsed is not None:
                # A parsed span's nested children can only be smaller, so
                # skipping past it never drops the winner and keeps the scan
                # near-linear on well-formed payloads.
                scan_from = end_idx + 1
            else:
                # Balanced but invalid: a nested {...}/[...] inside may still be
                # valid JSON, so re-enter the span instead of discarding it.
                scan_from = start_idx + 1

    return best
