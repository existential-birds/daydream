"""Conservative, versioned reply classifier for semantic gold labels.

Turns qualifying human reply text into a per-finding disposition:
``"accepted"``, ``"rejected"``, or ``"ambiguous"``. Pure functions only —
no I/O. Anything unclassifiable fails closed to ``"ambiguous"``.
"""

from __future__ import annotations

import re
from typing import Any

from daydream.training.labeler_versions import (
    REPLY_CLASSIFIER_VERSION as REPLY_CLASSIFIER_VERSION,
)

#: Logins that never qualify as reply authors (daydream's own accounts).
_DAYDREAM_AGENT_LOGINS = frozenset({"daydream-agent", "daydream-bot"})

_QUALIFYING_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

# Whole-phrase patterns. Matching is per-line, case-insensitive, with word
# boundaries — never bare substrings.
_ACCEPT_RULES: tuple[str, ...] = (
    r"fixed\s+in\s+\b[0-9a-f]{6,40}\b",
    r"\bapplied\b",
    r"\bgood\s+catch\b",
    r"\bagreed\b,?\s*(the\s+)?fix",
)

_REJECT_RULES: tuple[str, ...] = (
    r"\bnot\s+applicable\b",
    r"\balready\s+(?:handled|exists)\b",
    r"\bfalse\s+positive\b",
    r"\bintentional\b",
)

#: "won't fix" is only a rejection when a dispute phrase co-occurs.
_DISPUTE_RULES: tuple[str, ...] = (
    r"won'?t\s+fix",
    r"\bwrong\b",
    r"\bincorrect\b",
    r"\bnot\s+a\s+bug\b",
    r"\bthe\s+code\s+already\b",
)

_FACTUAL_DISAGREEMENT_RULES: tuple[str, ...] = (
    r"\bdisagree\b",
    r"\bthe\s+docs\s+say\b",
)

_NEGATION_TOKENS = re.compile(r"\b(?:not|never)\b|\bn't\b", re.IGNORECASE)


def _is_negated(text: str, start: int) -> bool:
    """True when a negation token appears before ``start`` in the same sentence."""
    prefix = text[:start]
    boundaries = [m.end() for m in re.finditer(r"[.!?;\n]", prefix)]
    sentence_start = boundaries[-1] if boundaries else 0
    return bool(_NEGATION_TOKENS.search(text[sentence_start:start]))


def _match_rules(text: str, rules: tuple[str, ...], *, guard_negation: bool) -> bool:
    for pattern in rules:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if guard_negation and _is_negated(text, m.start()):
                continue
            return True
    return False


def _dispute_present(text: str) -> bool:
    """Any dispute marker, un-negated, co-occurring in the body."""
    for pattern in _DISPUTE_RULES[1:]:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if not _is_negated(text, m.start()):
                return True
    # "won't fix" itself is a dispute trigger phrase but does not self-satisfy;
    # a *second*, distinct dispute marker is required.
    return False


def _direction(body: str) -> str:
    lines = body.splitlines() or [""]
    directions: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        has_accept = _match_rules(line, _ACCEPT_RULES, guard_negation=True)
        has_wontfix = bool(re.search(_DISPUTE_RULES[0], line, re.IGNORECASE))
        has_dispute = _dispute_present(line)
        has_reject = _match_rules(line, _REJECT_RULES, guard_negation=True) or (has_wontfix and has_dispute)
        has_factual = _match_rules(line, _FACTUAL_DISAGREEMENT_RULES, guard_negation=True)
        if has_accept:
            directions.add("accepted")
        if has_reject or has_factual:
            directions.add("rejected")
    if len(directions) == 1:
        return directions.pop()
    return "ambiguous"


def _user_str(reply: dict[str, Any], key: str) -> str:
    user = reply.get("user")
    if not isinstance(user, dict):
        return ""
    value = user.get(key)
    return value if isinstance(value, str) else ""


def _identity_gates_pass(reply: dict[str, Any]) -> bool:
    """Bot / daydream-agent / empty-login gates, independent of association."""
    if reply.get("is_self_reply"):
        return False
    if _user_str(reply, "type") == "Bot":
        return False
    login = _user_str(reply, "login")
    if not login or login.lower() in _DAYDREAM_AGENT_LOGINS:
        return False
    if login.lower().endswith("[bot]"):
        return False
    return True


def is_qualifying_author(
    reply: dict[str, Any],
    pr_author_logins: set[str] | frozenset[str],
    review_author_logins: set[str] | frozenset[str] = frozenset(),
) -> bool:
    """True when the reply's author is a human whose judgment counts (M6).

    Qualifies when the author is not a bot, has a non-empty login that is not
    a daydream agent, is not a marked self-reply, and is either a PR author,
    a formal-review author, or holds OWNER/MEMBER/COLLABORATOR association.
    """
    if reply.get("is_self_reply"):
        return False
    if not _identity_gates_pass(reply):
        return False
    login = _user_str(reply, "login")
    assoc = reply.get("author_association")
    if isinstance(assoc, str) and assoc in _QUALIFYING_ASSOCIATIONS:
        return True
    if login in pr_author_logins or login in review_author_logins:
        return True
    return False


def classify_reply(reply: dict[str, Any]) -> str:
    """Classify a single reply as ``"accepted"``/``"rejected"``/``"ambiguous"``.

    Non-qualifying authors and unparseable bodies are always ``"ambiguous"``.
    """
    if not _identity_gates_pass(reply):
        return "ambiguous"
    body = reply.get("body")
    if not isinstance(body, str) or not body.strip():
        return "ambiguous"
    return _direction(body)
