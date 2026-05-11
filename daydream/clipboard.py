"""Best-effort system clipboard support via platform shell tools.

Detects the first available clipboard mechanism in priority order
(``pbcopy`` for macOS, ``xclip`` then ``xsel`` for Linux/X11, ``clip.exe``
for WSL/Windows) and pipes text into it via ``subprocess.run``. Returns
False when no mechanism is available or any subprocess call fails so
callers can degrade gracefully without raising.

Exports:
    clipboard_available: Check whether a clipboard mechanism is present.
    copy_to_clipboard: Copy ``text`` to the system clipboard.
"""

from __future__ import annotations

import shutil
import subprocess

# Detection order: macOS first, then Linux/X11, then WSL/Windows.
# Each entry is the argv to invoke; the first whose program exists on
# ``$PATH`` is used.
_CLIPBOARD_COMMANDS: tuple[list[str], ...] = (
    ["pbcopy"],
    ["xclip", "-selection", "clipboard"],
    ["xsel", "--clipboard", "--input"],
    ["clip.exe"],
)


def _detect_clipboard_command() -> list[str] | None:
    """Return the first available clipboard argv, or None if none found.

    Detection uses :func:`shutil.which` for each candidate's program name.
    The argv (with flags) is returned verbatim so the caller can invoke it
    via ``subprocess.run`` without re-deriving flags.
    """
    for argv in _CLIPBOARD_COMMANDS:
        if shutil.which(argv[0]) is not None:
            return list(argv)
    return None


def clipboard_available() -> bool:
    """Return True if a clipboard mechanism is present on this system."""
    return _detect_clipboard_command() is not None


def copy_to_clipboard(text: str) -> bool:
    """Copy *text* to the system clipboard using the first available tool.

    Detection order is macOS (``pbcopy``) → Linux/X11 (``xclip``,
    ``xsel``) → WSL/Windows (``clip.exe``). Returns False if no mechanism
    is available or the subprocess fails for any reason — callers should
    treat False as "tell the user to copy from the printed path".

    Args:
        text: Content to place on the clipboard.

    Returns:
        True on successful copy, False otherwise.
    """
    argv = _detect_clipboard_command()
    if argv is None:
        return False
    try:
        subprocess.run(  # noqa: S603 - args are not user-controlled
            argv,
            input=text,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return True
