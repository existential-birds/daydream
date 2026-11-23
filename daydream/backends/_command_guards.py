"""Shell-command rules shared by the adapters that guard a native shell tool.

Pure and import-free: Claude matches the dangerous patterns in its SDK hooks, and
Antigravity serializes them, with its background rule, into its per-run guard policy.
"""

from __future__ import annotations

import re

# Catastrophic commands denied in ALL phases (always-on guard, #177). These
# are the runaway-turn pathologies: full-filesystem-root scans that take hours,
# plus an unrecoverable wipe. Matched on the raw command via regex.
DANGEROUS_COMMAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*find\s+/(\s|$)"),       # find / ...  (root-anchored scan)
    re.compile(r"^\s*grep\b.*\s/\s*$"),      # grep ... /  (root is the sole trailing path)
    # Match recursive rm flags in any order plus a standalone / or /* target.
    # Subpaths such as /home are outside this catastrophic-wipe backstop.
    re.compile(r"^\s*rm\b(?=.*(?:^|\s)(?:-\w*[rR]\w*|--recursive)\b)(?=.*(?:^|\s)/\*?(?:\s|$)).*$"),
)

# An unquoted background control operator: a lone ``&`` that is not part of ``&&``,
# ``&>``, ``>&``, ``<&`` or ``|&``. The possessive loop consumes quoted strings,
# escapes and those operators whole, so only a bare ``&`` is left to match.
BACKGROUND_COMMAND_PATTERN: re.Pattern[str] = re.compile(
    r"""^(?:[^&'"\\|<>]|\\[\s\S]|'[^']*'|"(?:[^"\\]|\\[\s\S])*"|&&|&>|[|<>]&?+)*+&"""
)


def is_dangerous_command(cmd: str) -> bool:
    """Match catastrophic root scans/wipes; scoped paths and unmatched commands pass."""
    return any(pattern.search(cmd) for pattern in DANGEROUS_COMMAND_PATTERNS)
