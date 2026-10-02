"""Privacy-filtered exception diagnostics with bounded head and root-cause tail.

Render newest-first, redact before truncation, then remove terminal controls
except tabs/newlines. Formatting failures produce fixed markers; nothing is emitted.
"""

from __future__ import annotations

import re
import traceback
import unicodedata
from collections.abc import Mapping
from typing import Final

#: Fixed marker emitted when any formatting stage fails. Never rendered from
#: the failing formatter's own exception object.
VERBOSE_DIAGNOSTIC_UNAVAILABLE = "[VERBOSE_DIAGNOSTIC_UNAVAILABLE]"
#: Fixed marker inserted exactly once when a diagnostic exceeds the size bound.
VERBOSE_DIAGNOSTIC_TRUNCATED = "[VERBOSE_DIAGNOSTIC_TRUNCATED]"

#: Hard cap (UTF-8 bytes) for one formatted diagnostic.
_MAX_DIAGNOSTIC_BYTES: Final[int] = 65536
#: Retained-from-the-end budget for the root cause's tail after truncation.
_ROOT_CAUSE_TAIL_BYTES: Final[int] = 4096

#: ANSI CSI sequences (``ESC [ parameter-bytes intermediate-bytes final-byte``).
_CSI_SEQUENCE = re.compile(r"\x1b\[[0-9:;<=>?]*[ -/]*[@-~]")
#: Control characters the diagnostic is allowed to keep: line feeds and tabs.
_KEPT_CONTROLS = frozenset({"\n", "\t"})

#: Interstitial between an exception page and its direct cause page, adapted
#: from the interpreter's wording to the newest-first page order: the effect
#: ("the above exception") sits on top, the cause ("the following exception")
#: below it.
_DIRECT_CAUSE_LINK = (
    "\nThe above exception was directly caused by the following exception:\n\n"
)
#: Interstitial before an exception handled during another, likewise adapted to
#: the newest-first page order: the effect sits on top, the handled context
#: below it.
_CONTEXT_LINK = (
    "\nThe above exception occurred while handling the following exception:\n\n"
)


def exception_text(exc: BaseException) -> str | None:
    """Read untrusted exception text; None means its formatter failed.

    Callers must still redact and neutralize this text before displaying it.
    """
    try:
        return str(exc)
    except Exception:  # noqa: BLE001 - formatting must not replace the original failure
        return None


def _neutralize_control(value: str) -> str:
    """Drop ANSI CSI and remaining controls, preserving tabs/newlines for traceback structure."""
    value = _CSI_SEQUENCE.sub("", value)
    return "".join(
        char
        if char in _KEPT_CONTROLS or unicodedata.category(char) != "Cc"
        else ""
        for char in value
    )


def _chain_members(exc: BaseException) -> list[BaseException]:
    """Follow cause, then unsuppressed context, newest-first with identity deduplication."""
    members: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        members.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return members


def _render_chain(exc: BaseException) -> str:
    """Render bounded traceback pages newest-first, preserving notes and adapted chain links.

    Local variables are never captured. Width/depth/trace limits bound exception groups.
    """
    members = _chain_members(exc)
    pages: list[str] = []
    for index, member in enumerate(members):
        tbe = traceback.TracebackException.from_exception(
            member,
            capture_locals=False,
            limit=50,
            max_group_width=8,
            max_group_depth=4,
        )
        pages.append("".join(tbe.format(chain=False)))
        if index + 1 < len(members):
            following = members[index + 1]
            if member.__cause__ is following:
                pages.append(_DIRECT_CAUSE_LINK)
            else:
                pages.append(_CONTEXT_LINK)
    return "".join(pages)


def _assemble_diagnostic(value: str) -> str:
    """After redaction, neutralize controls and retain UTF-8-valid head/tail within the cap.

    Exactly one truncation marker separates the retained parts.
    """
    value = _neutralize_control(value)
    raw = value.encode("utf-8")
    if len(raw) <= _MAX_DIAGNOSTIC_BYTES:
        return value
    marker = VERBOSE_DIAGNOSTIC_TRUNCATED
    head_budget = _MAX_DIAGNOSTIC_BYTES - _ROOT_CAUSE_TAIL_BYTES - len(marker)
    # raw is valid UTF-8; only a code point split by either cut can be invalid.
    head = raw[:head_budget].decode("utf-8", errors="ignore")
    tail = raw[-_ROOT_CAUSE_TAIL_BYTES:].decode("utf-8", errors="ignore")
    assert len(head.encode("utf-8")) + len(tail.encode("utf-8")) + len(
        marker.encode("utf-8")
    ) <= _MAX_DIAGNOSTIC_BYTES
    return head + marker + tail


def sanitize_verbose_message(
    message: BaseException | str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Safely format, redact and neutralize one fatal panel message; failure yields empty.

    Exception conversion occurs inside the boundary because __str__ may raise.
    """
    from daydream.observability.privacy import PrivacyPolicy

    try:
        text = exception_text(message) if isinstance(message, BaseException) else message
        if text is None:
            return ""
        policy = PrivacyPolicy(environ=environ)
        return _neutralize_control(policy.text(text))
    except Exception:  # noqa: BLE001 - fail closed to an empty panel
        return ""


def format_verbose_exception(
    exc: BaseException,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Render a redacted, bounded chain with notes and no captured local variables.

    Redact complete text before any size cap, using the real environment unless
    overridden. Any formatting failure returns the fixed unavailable marker.
    """
    from daydream.observability.privacy import PrivacyPolicy

    try:
        # Materialize the top-level message up front: ``str()`` is the one
        # operation that can raise on a hostile value, and a raising message
        # must fail closed instead of letting the stdlib traceback renderer
        # substitute a placeholder line.
        if exception_text(exc) is None:
            return VERBOSE_DIAGNOSTIC_UNAVAILABLE
        formatted = _render_chain(exc)
        policy = PrivacyPolicy(environ=environ)
        # The COMPLETE rendered value crosses the privacy boundary first: a
        # credential can never straddle an artificial pre-redaction cut. Only
        # after redaction does the single final size cap apply.
        return _assemble_diagnostic(policy.text(formatted))
    except Exception:  # noqa: BLE001 - every stage fails closed to the marker
        return VERBOSE_DIAGNOSTIC_UNAVAILABLE
