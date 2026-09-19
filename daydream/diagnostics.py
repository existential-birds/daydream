"""Verbose fatal diagnostics: one privacy-first, size-bounded exception dump.

:func:`format_verbose_exception` renders an exception chain newest-first —
the exception passed to the call FIRST, then what it was caused by. The
interpreter itself narrates oldest-first (the cause page above, the effect
below); this module deliberately keeps the failing exception FIRST, per the
pinned newest-first contract, so it adapts the interstitial lines to that
order — redacts the rendered value with the observability
:class:`~.PrivacyPolicy` (the same policy that guards every traced span)
BEFORE any size bound, neutralizes terminal control characters while keeping
newlines and tabs, and finally caps the result at one bounded head, a single
explicit truncation marker, and the root cause's tail.

Every failure mode degrades to one fixed marker string, so a fatal diagnostic
can never leak a payload, crash twice, or render partial machinery. To keep
the fatal path from *introducing its own* I/O, the module imports only the
standard library plus :class:`~.PrivacyPolicy` and never writes to any log
sink, terminal, or tracer itself; note that :class:`~.PrivacyPolicy`
transitively imports ``daydream.trajectory`` (which imports
``daydream.ui``/``anyio``/``daydream.atif``), so the host CLI process already
has those loaded before the handler runs.

The only public seam::

    format_verbose_exception(exc, *, environ=None) -> str
"""

from __future__ import annotations

import re
import traceback
import unicodedata
from collections.abc import Mapping
from typing import Final

from daydream.observability.privacy import PrivacyPolicy

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


def _neutralize_control(value: str) -> str:
    """Drop ANSI CSI sequences and every remaining control character.

    Newlines and tabs are preserved so the traceback keeps its structure; every
    other control code (escapes, carriage returns, bells, ...) is removed so a
    hostile exception message cannot paint over the operator's terminal.
    """
    value = _CSI_SEQUENCE.sub("", value)
    return "".join(
        char
        if char in _KEPT_CONTROLS or unicodedata.category(char) != "Cc"
        else ""
        for char in value
    )


def _valid_utf8(chunk: bytes) -> bool:
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _utf8_prefix(raw: bytes, budget: int) -> bytes:
    """Longest UTF-8-valid prefix of *raw* whose byte length is <= *budget*."""
    cut = raw[:budget]
    while cut and not _valid_utf8(cut):
        cut = cut[:-1]
    return cut


def _utf8_suffix(raw: bytes, budget: int) -> bytes:
    """Longest UTF-8-valid suffix of *raw* whose byte length is <= *budget*."""
    cut = raw[-budget:]
    while cut and not _valid_utf8(cut):
        cut = cut[1:]
    return cut


def _chain_members(exc: BaseException) -> list[BaseException]:
    """Return the cause chain newest-first, de-duplicated by identity.

    The member passed in comes first; the walk follows ``__cause__`` (or, when
    absent, an un-suppressed ``__context__``) until the chain ends, mirroring
    the interpreter's notion of which exceptions belong to one traceback.
    """
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
    """Render *exc*'s chain newest-first, one traceback page per member.

    Per-member pages use :class:`traceback.TracebackException` exactly as the
    interpreter would (``capture_locals=False`` is mandatory — local-variable
    values must never appear; ``limit`` and the group caps keep the output
    bounded), but the pages are joined in the order the operator sees them:
    the exception itself first, then what caused it. The bridge lines are
    adapted from the interpreter's phrasing to this newest-first page order.
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
    """Neutralize controls and cap *value* at one bounded head + marker + tail.

    Runs AFTER the complete value has been redacted by the privacy policy: the
    head is the longest UTF-8-valid prefix within the head budget, the tail the
    longest UTF-8-valid suffix within the root-cause tail budget, and exactly
    one ``[VERBOSE_DIAGNOSTIC_TRUNCATED]`` marker bridges them. The budgets
    plus the marker always fit the hard cap; the assert protects that
    invariant from future budget edits.
    """
    value = _neutralize_control(value)
    raw = value.encode("utf-8")
    if len(raw) <= _MAX_DIAGNOSTIC_BYTES:
        return value
    marker = VERBOSE_DIAGNOSTIC_TRUNCATED
    head_budget = _MAX_DIAGNOSTIC_BYTES - _ROOT_CAUSE_TAIL_BYTES - len(marker)
    head = _utf8_prefix(raw, head_budget).decode("utf-8")
    tail = _utf8_suffix(raw, _ROOT_CAUSE_TAIL_BYTES).decode("utf-8")
    assert len(head.encode("utf-8")) + len(tail.encode("utf-8")) + len(
        marker.encode("utf-8")
    ) <= _MAX_DIAGNOSTIC_BYTES
    return head + marker + tail


def sanitize_verbose_message(
    message: BaseException | str,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Redact and control-neutralize one fatal message for the UI panel.

    Applies the same privacy and control boundary as
    :func:`format_verbose_exception` to a single-line message — the generic
    fatal branch's panel text. Accepts the exception object itself (not just
    its pre-materialized ``str``) so the ``str()`` conversion — the one
    operation that can raise on a hostile value — happens INSIDE the fail-
    closed try: an exception whose ``__str__`` raises degrades to ``""``
    instead of escaping the generic fatal handler. Fails closed to ``""``
    so a hostile message can never paint the panel with a payload.
    """
    try:
        text = str(message) if isinstance(message, BaseException) else message
        policy = PrivacyPolicy(environ=environ)
        return _neutralize_control(policy.text(text))
    except Exception:  # noqa: BLE001 - fail closed to an empty panel
        return ""


def format_verbose_exception(
    exc: BaseException,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the redacted, bounded, control-safe rendering of *exc*.

    Rendering keeps chaining enabled — direct causes and implicit contexts
    both appear, in the same order the interpreter would narrate them — never
    captures local-variable values, and prints exception notes. Redaction runs
    over the complete rendered text BEFORE any size bound and uses the real
    process environment unless *environ* supplies an explicit replacement.

    Any exception raised at any stage — including a value whose ``__str__``
    cannot run — fails closed to ``[VERBOSE_DIAGNOSTIC_UNAVAILABLE]``; the
    failing formatter's own exception object is never rendered into the output.
    """
    try:
        # Materialize the top-level message up front: ``str()`` is the one
        # operation that can raise on a hostile value, and a raising message
        # must fail closed instead of letting the stdlib traceback renderer
        # substitute a placeholder line.
        str(exc)
        formatted = _render_chain(exc)
        policy = PrivacyPolicy(environ=environ)
        # The COMPLETE rendered value crosses the privacy boundary first: a
        # credential can never straddle an artificial pre-redaction cut. Only
        # after redaction does the single final size cap apply.
        return _assemble_diagnostic(policy.text(formatted))
    except Exception:  # noqa: BLE001 - every stage fails closed to the marker
        return VERBOSE_DIAGNOSTIC_UNAVAILABLE
