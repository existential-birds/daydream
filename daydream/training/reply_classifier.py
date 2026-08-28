"""Conservative, versioned reply classifier for semantic gold labels.

Turns qualifying human reply text into a per-finding disposition:
``"accepted"``, ``"rejected"``, or ``"ambiguous"``. Pure functions only —
no I/O. Anything unclassifiable fails closed to ``"ambiguous"``.
"""

from __future__ import annotations

import re
from typing import Any

REPLY_CLASSIFIER_VERSION = "reply-classifier-v1"

#: Logins that never qualify as reply authors (daydream's own accounts).
_DAYDREAM_AGENT_LOGINS = frozenset({"daydream-agent", "daydream-bot"})

_QUALIFYING_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

_DECISIVE = ("accepted", "rejected")

# (pattern, kind) pairs. Matching is per-line, case-insensitive, whole-phrase
# with word boundaries — never bare substrings.
_ACCEPT_RULES: tuple[tuple[str, str], ...] = (
    (r"fixed\s+in\s+\b[0-9a-f]{6,40}\b", "sha"),
    (r"\bapplied\b", "applied"),
    (r"\bgood\s+catch\b", "agree"),
    (r"\bagreed\b,?\s*(the\s+)?fix", "agree"),
)

_REJECT_RULES: tuple[tuple[str, str], ...] = (
    (r"\bnot\s+applicable\b", "na"),
    (r"\balready\s+(?:handled|exists)\b", "handled"),
    (r"\bfalse\s+positive\b", "fp"),
    (r"\bintentional\b", "intentional"),
)

#: "won't fix" is only a rejection when a dispute phrase co-occurs.
_DISPUTE_RULES: tuple[tuple[str, str], ...] = (
    (r"won'?t\s+fix", "wontfix"),
    (r"\bwrong\b", "wrong"),
    (r"\bincorrect\b", "incorrect"),
    (r"\bnot\s+a\s+bug\b", "notabug"),
    (r"\bthe\s+code\s+already\b", "codealready"),
)

_FACTUAL_DISAGREEMENT_RULES: tuple[tuple[str, str], ...] = (
    (r"\bdisagree\b", "disagree"),
    (r"\bthe\s+docs\s+say\b", "docs"),
)

_NEGATION_TOKENS = re.compile(r"\b(?:not|never)\b|\bn't\b", re.IGNORECASE)


def _sentence_containing(text: str, start: int, end: int) -> str:
    """Return the sentence (split on .!?; and newlines) containing [start, end)."""
    sentence_starts = [0] + [m.end() for m in re.finditer(r"[.!?;\n]", text) if m.end() <= start]
    start_idx = sentence_starts[-1]
    tail = re.search(r"[.!?;\n]", text[end:])
    end_idx = end + (tail.start() if tail else len(text[end:]))
    return text[start_idx:end_idx]


def _is_negated(text: str, start: int) -> bool:
    """True when a negation token appears before ``start`` in the same sentence."""
    sentence = _sentence_containing(text, start, start)
    return bool(_NEGATION_TOKENS.search(sentence[: _sentence_offset(sentence, text, start)]))


def _sentence_offset(sentence: str, text: str, start: int) -> int:
    # Offset of `start` within its sentence.
    idx = text.find(sentence)
    return start - idx if idx >= 0 else start


def _match_rules(
    text: str, rules: tuple[tuple[str, str], ...], *, guard_negation: bool
) -> bool:
    for pattern, _kind in rules:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if guard_negation and _is_negated(text, m.start()):
                continue
            return True
    return False


def _dispute_present(text: str) -> bool:
    """Any dispute marker, un-negated, co-occurring in the body."""
    for pattern, _kind in _DISPUTE_RULES[1:]:
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
        has_wontfix = bool(re.search(_DISPUTE_RULES[0][0], line, re.IGNORECASE))
        has_dispute = _dispute_present(line)
        has_reject = _match_rules(line, _REJECT_RULES, guard_negation=False) or (
            has_wontfix and has_dispute
        )
        has_factual = _match_rules(line, _FACTUAL_DISAGREEMENT_RULES, guard_negation=False)
        if has_accept:
            directions.add("accepted")
        if has_reject or has_factual:
            directions.add("rejected")
    if len(directions) == 1:
        return directions.pop()
    return "ambiguous"


def _login(reply: dict[str, Any]) -> str:
    user = reply.get("user")
    if not isinstance(user, dict):
        return ""
    login = user.get("login")
    return login if isinstance(login, str) else ""


def _user_type(reply: dict[str, Any]) -> str:
    user = reply.get("user")
    if not isinstance(user, dict):
        return ""
    utype = user.get("type")
    return utype if isinstance(utype, str) else ""


def _identity_gates_pass(reply: dict[str, Any]) -> bool:
    """Bot / daydream-agent / empty-login gates, independent of association."""
    if reply.get("is_self_reply"):
        return False
    if _user_type(reply) == "Bot":
        return False
    login = _login(reply)
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
    login = _login(reply)
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


def classify_replies(replies: list[dict[str, Any]]) -> str:
    """Aggregate reply classifications conservatively (M18/M22).

    Filters to qualifying authors, drops ambiguous votes, and returns the
    single decisive label, ``"ambiguous"`` on conflict, or ``"ambiguous"``
    when no decisive vote exists. Deterministic: identical input yields
    identical output.
    """
    votes: set[str] = set()
    for reply in replies:
        if not isinstance(reply, dict):
            continue
        if not _identity_gates_pass(reply):
            continue
        label = classify_reply(reply)
        if label in _DECISIVE:
            votes.add(label)
    if len(votes) == 1:
        return votes.pop()
    return "ambiguous"
