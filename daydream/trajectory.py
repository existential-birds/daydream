"""ATIF v1.7 trajectory recorder for daydream runs.

This module is the SOLE home for ATIF Pydantic model construction (D-19
module-bloat ban). Other modules (agent.py, phases.py, ui.py, runner.py,
backends/*) import only the public surface — never `daydream.atif.*`.

Lifecycle: ``runner.py`` opens ``async with TrajectoryRecorder(...) as
recorder`` once per run. ``agent.run_agent()`` opens an ``Invocation`` per
call against the recorder via ``get_current_recorder()``. Backends emit
``AgentEvent`` instances; the Invocation buffers them into ATIF Steps and
flushes to the parent Trajectory at scope exit. The Recorder writes the
Trajectory JSON on clean ``__aexit__``.

The recorder uses one ``ContextVar`` (``_RECORDER_VAR``) to expose the active
run, supports sibling recorders for parallel task groups, and redacts sensitive
values before trajectory data is persisted.
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypedDict

import daydream
from daydream.atif import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from daydream.json_utils import atomic_write_json
from daydream.timeutil import parse_iso_timestamp
from daydream.ui import create_console, print_error, print_warning

# Run-directory layout. Live + archive trajectories share an identical on-disk
# shape (<root>/runs/<session_id>/trajectory.json and .../trajectories/<descriptor>.json);
# live root is <target>/.daydream, archive root is per daydream.archive.
_RUNS_SUBDIR = "runs"
_TRAJECTORIES_SUBDIR = "trajectories"
_DAYDREAM_DIRNAME = ".daydream"

if TYPE_CHECKING:
    from daydream.backends import AgentEvent, CostEvent, ToolResultEvent

_console = create_console()
_INITIAL_TOTALS: dict[str, Any] = {"prompt": 0, "completion": 0, "cached": 0, "cost": 0.0, "any_cost_seen": False}  # noqa: E501 - module-level constant cloned via dict.copy() at recorder init

# Generic backend labels that should be replaced as soon as a real SDK
# model id arrives via MetricsEvent, CostEvent, or ResultEvent. Runner stamps
# the recorder
# with one of these (or empty) at init since the real model id isn't known
# until the first agent turn streams back.
_GENERIC_MODEL_LABELS: frozenset[str] = frozenset(
    {"claude", "codex", "osprey", "unknown", ""}
)


def _reasoning_extra(reasoning_tokens: int | None) -> dict[str, Any] | None:
    """Metrics ``extra`` carrier for reasoning_tokens (#192), or None when absent."""
    return {"reasoning_tokens": reasoning_tokens} if reasoning_tokens is not None else None


def _add(left: Any, right: Any) -> Any:
    """Sum two optional numbers, treating None as absent (not zero)."""
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _merge_metrics(existing: "Metrics", incoming: "Metrics") -> "Metrics":
    """Additively merge a later MetricsEvent into a Step's running metrics.

    A multi-turn invocation reports usage once per turn; every one of those
    turns is billed, so a Step spanning them carries their sum rather than the
    last turn's snapshot. reasoning_tokens rides in ``extra`` (#192) and sums
    the same way; other ``extra`` keys are carried forward.
    """
    merged_extra: dict[str, Any] | None = None
    if existing.extra is not None or incoming.extra is not None:
        merged_extra = {**(existing.extra or {}), **(incoming.extra or {})}
        reasoning = _add(
            (existing.extra or {}).get("reasoning_tokens"),
            (incoming.extra or {}).get("reasoning_tokens"),
        )
        if reasoning is not None:
            merged_extra["reasoning_tokens"] = reasoning
    return existing.model_copy(update={
        "prompt_tokens": _add(existing.prompt_tokens, incoming.prompt_tokens),
        "completion_tokens": _add(existing.completion_tokens, incoming.completion_tokens),
        "cached_tokens": _add(existing.cached_tokens, incoming.cached_tokens),
        "cost_usd": _add(existing.cost_usd, incoming.cost_usd),
        "extra": merged_extra,
    })


class _InvMetricsSum(TypedDict):
    """Per-message MetricsEvent totals for one invocation (issue #747).

    prompt/completion are always int, cached is int-or-absent, and only cost is
    fractional. Precise member types let the CostEvent reconcile subtract these
    values without the ``int()``/``float()`` coercions that a broad
    ``dict[str, float | int]`` annotation forced on every delta computation.
    """

    prompt: int
    completion: int
    cached: int
    cost: float


@dataclass(frozen=True)
class _CostDelta:
    """Per-dimension take-max residual of a CostEvent over per-event metrics.

    ``max(0, CostEvent.total - _inv_metrics_sum)`` (issue #747): Claude's
    authoritative session total exceeds the collapsed per-message single digits
    so the delta is the true repair magnitude; codex/pi re-state the already
    summed totals so every delta is 0 and the restatement never double-counts.
    Never negative, never subtraction.
    """

    prompt: int
    completion: int
    cached: int
    cost: float
    reasoning: int | None

    @property
    def nonzero(self) -> bool:
        """True when any residual dimension exists (delta-0 restatement folds nothing)."""
        return self.prompt != 0 or self.completion != 0 or self.cached != 0 or self.cost != 0.0

    def residual_metrics(self, *, cost_usd: float | None, include_reasoning: bool) -> Metrics:
        """Residual Metrics for this delta (the CostEvent residual-step mapping).

        ``reasoning_tokens`` (#192) is a subset of ``completion_tokens``, so it is
        carried only when it does not exceed this step's completion.
        """
        reasoning = None
        if (
            include_reasoning
            and self.reasoning is not None
            and self.reasoning <= self.completion
        ):
            reasoning = _reasoning_extra(self.reasoning)
        return Metrics(
            prompt_tokens=self.prompt,
            completion_tokens=self.completion,
            cached_tokens=self.cached,
            cost_usd=None if cost_usd is None else self.cost,
            extra=reasoning,
        )


# Redaction patterns (REDA-01..04). Redaction composes three stages in order:
# (1) auth-header + auth-scheme rules (redact_structured_text) so a
# shaped value under a header (e.g. `Authorization: Bearer sk-1234`) is consumed
# whole and never double-marked; (2) the flat rules below — URL-credential before
# bare API-key (so the captured credential isn't re-matched), PEM before env-var
# (so `VAR=<PEM>` collapses whole instead of leaking the key body), env-var before
# bare API-key (so `OPENAI_API_KEY=sk-1234` keeps its name per D-03); (3) structured
# key-value redaction (_redact_structured_key_values), which skips existing
# [REDACTED_*] markers so earlier stages' output is never clobbered.
_URL_CREDENTIAL_PATTERN = re.compile(r"(https?://)([^:@/\s]+):([^@/\s]+)@")
_API_KEY_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{6,}|ghp_[A-Za-z0-9]{6,}|ghs_[A-Za-z0-9]{6,}|xoxb-[A-Za-z0-9\-]{6,}|AKIA[A-Z0-9]{16})\b"
)
_JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\b"
)
_USERNAME_PATH_PATTERN = re.compile(r"(/Users/|/home/|[A-Z]:\\Users\\)([^/\\\s]+)")
# PEM private-key blocks (PKCS1/RSA, PKCS8, ENCRYPTED, OPENSSH, EC, DSA).
# Multi-line body collapsed before the bare API-key rule scans it. CERTIFICATE
# blocks are public material — not matched.
# The header fragment is shared (imported) by the benchmark's buffering
# anchors, so a variant addition needs one edit here, not four.
_PEM_HEADER = r"(?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY"
_PEM_KEY_PATTERN = re.compile(
    rf"-----BEGIN {_PEM_HEADER}-----"
    rf".*?-----END {_PEM_HEADER}-----",
    re.DOTALL,
)
#: Replacement marker for PEM private-key blocks. Shared (imported) by the
#: benchmark's buffering anchors so every redaction site emits one marker.
_PEM_KEY_REDACTED_MARKER = "[REDACTED_PEM_KEY]"

# Fixed-ASCII synthetic content for the incomplete-call marker emitted by
# ``Invocation.finish()`` for tool calls still in flight. Never formatted with
# tool data, so it is redaction-stable.
INCOMPLETE_CALL_CONTENT = "[interrupted: call did not complete before invocation ended]"
#: Replacement marker for credential values under sensitive keys (issue #455).
#: Distinct from every existing [REDACTED_*] marker so consumers can tell a
#: key-aware credential redaction from a flat regex hit.
_REDACTED_CREDENTIAL = "[REDACTED_CREDENTIAL]"
# Match env-var assignment where one of the underscore-separated SEGMENTS of
# the var name is a secret keyword. Substring matching (the original) over-
# redacted MONKEY_PATCH/KEYBOARD_LAYOUT/AUTHOR/TOKENIZED — segment-aware
# matching keeps the secret list precise without false positives.
# The separator whitespace is horizontal-only. With plain ``\s*`` an empty
# assignment (``API_KEY=`` at end of line) consumed the newline and swallowed
# the whole next line as its "value", deleting real content from redacted text.
# An assignment with no value on its own line carries no secret.
_ENV_VAR_PATTERN = re.compile(
    r"\b((?:[A-Z][A-Z0-9]*_)*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|API_?KEY|APIKEY|AUTH)(?:_[A-Z0-9]+)*)[^\S\n\r]*=[^\S\n\r]*([^\s\n\r;]+)"  # noqa: E501 - secret-segment alternation
)
_REDACTION_RULES: tuple[tuple[Any, str], ...] = (
    (_URL_CREDENTIAL_PATTERN, r"\1[REDACTED_USER]:[REDACTED_API_KEY]@"),
    (_PEM_KEY_PATTERN, _PEM_KEY_REDACTED_MARKER),
    (_ENV_VAR_PATTERN, r"\1=[REDACTED_ENV_VAR]"),
    (_API_KEY_PATTERN, "[REDACTED_API_KEY]"),
    (_JWT_PATTERN, "[REDACTED_JWT]"),
    (_USERNAME_PATH_PATTERN, r"\1[REDACTED_USER]"),
)


def redact_text(value: str) -> str:
    """Return redacted text, replacing the whole field if redaction fails."""
    try:
        for pattern, replacement in _REDACTION_RULES:
            value = pattern.sub(replacement, value)
    except Exception:  # noqa: BLE001 - fail closed at every host boundary
        return "[REDACTION_FAILED]"
    return value


def redact_value(value: Any, sensitive: bool = False) -> Any:
    """Recursively redact a value without mutating its argument (never raises).

    Key-aware (issue #455): ``str`` leaves under a sensitive key — their own
    key, or inherited from a sensitive ancestor container — are replaced with
    ``_REDACTED_CREDENTIAL``; other string leaves run through
    :func:`redact_structured_text`. String dict keys are run through
    :func:`redact_text` so secret-shaped keys are scrubbed too; containers
    are rebuilt fresh so the caller's object is never touched. Non-container,
    non-string leaves pass through unchanged. This is the canonical
    structured redactor for log-mode event payloads and the trajectory path
    alike; the recursion lives here so every consumer shares the same
    fail-closed boundary.
    """
    if isinstance(value, str):
        return _REDACTED_CREDENTIAL if sensitive else redact_structured_text(value)
    if isinstance(value, dict):
        return {
            (redact_text(k) if isinstance(k, str) else k): redact_value(
                v, sensitive or (_is_sensitive_key(k) if isinstance(k, str) else False)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item, sensitive) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, sensitive) for item in value)
    return value


def _safe_descriptor(raw: str) -> str:
    """Slugify a descriptor to filesystem-safe characters (D-06).

    Raises:
        ValueError: If *raw* produces an empty slug after sanitization.
    """
    slug = re.sub(r"[^a-z0-9-]", "-", raw.lower())
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    if not slug:
        raise ValueError(f"Descriptor {raw!r} produces empty slug after sanitization")
    return slug


def default_trajectory_path(target_dir: Path, session_id: str) -> Path:
    """Return the default trajectory path under ``<target>/.daydream/runs/<session_id>/``.

    The session_id segment guarantees uniqueness per run; the recorder
    creates the directory before its first write.
    """
    return (
        target_dir
        / _DAYDREAM_DIRNAME
        / _RUNS_SUBDIR
        / session_id
        / "trajectory.json"
    )


def maybe_fork(recorder: "TrajectoryRecorder | None", descriptor: str) -> Any:
    """Return a fork CM if *recorder* is set, otherwise a no-op context manager."""
    if recorder is not None:
        return recorder.fork(descriptor)
    return nullcontext()


def now_iso() -> str:
    """Return current UTC time as ISO 8601 with trailing 'Z'.

    The single source of truth for timestamps in daydream's trajectory
    recording. Used by ``AgentEvent`` dataclass ``field(default_factory=...)``
    in ``daydream/backends/__init__.py`` (Plan 02), by recorder Step
    construction here, and by Phase 4 partial-write paths.

    Banned alternatives: the deprecated naive-utc helper from ``datetime``
    (Pitfall 2: lacks tzinfo, deprecated in 3.12+); ad-hoc
    ``datetime.now().isoformat()`` (no ``Z`` suffix — Pydantic timestamp
    validator requires ``Z`` or ``+00:00``).

    Returns:
        Timestamp string parseable by ``Step.validate_timestamp``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").removesuffix("+00:00") + "Z"


class DaydreamPhase(str, Enum):
    """Phase label for ``Step.extra['daydream_phase']`` (MAP-08).

    Values match ATIF ``extra`` field literals exactly. Required keyword-only
    arg on ``run_agent()`` (D-05); every call site in ``phases.py`` passes a
    literal member.
    """

    REVIEW = "review"
    PARSE = "parse"
    FIX = "fix"
    TEST = "test"
    INTENT = "intent"
    ALTERNATIVES = "alternatives"
    DEEP = "deep"
    EXPLORATION = "exploration"
    VERIFY = "verify"
    RECON = "recon"
    AUDIT = "audit"
    VET = "vet"
    PLAN_WRITE = "plan_write"
    DIAGRAM = "diagram"


class DaydreamRunFlow(str, Enum):
    """Run-flow label for ``Step.extra['daydream_run_flow']`` (MAP-09).

    Set once at recorder construction (D-07); recorder stamps every Step.
    """

    NORMAL = "normal"
    TTT = "ttt"
    PR = "pr"
    DEEP = "deep"
    CUSTOM = "custom"
    IMPROVE = "improve"
    DIAGRAM = "diagram"


# Sensitive-key detection (issue #455): segment-aware, casing-agnostic.
# A key is sensitive when the whole normalized key is a member of
# _SENSITIVE_KEY_SUFFIXES, when it ends with "_" + member (compound members
# like private_key / secret_access_key), or when any underscore-separated
# segment is a member. Bare `key` is deliberately absent so keyStore/key_store
# stay clean (WR-03), and a secret term must appear as a full segment — never
# as a substring (tokenizer/passwordless pass through).
_SENSITIVE_KEY_SUFFIXES: frozenset[str] = frozenset({
    "api_key", "apikey", "auth", "authorization", "client_secret", "cookie",
    "credential", "credentials", "password", "passwd", "private_key",
    "secret", "secret_access_key", "set_cookie", "token",
})
# Insert a separator before an uppercase letter that follows a lowercase/digit
# (camelCase → snake_case boundary): apiKey → api_Key, dbPassword → db_Password.
_CAMEL_CASE_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Collapse any run of non-alphanumerics to a single underscore:
# client-secret → client_secret, Access_Token → Access_Token.
_NON_ALPHANUMERIC_KEY_PATTERN = re.compile(r"[^A-Za-z0-9]+")


def _normalize_sensitive_key(key: str) -> str:
    """Normalize *key* for sensitive-key matching (snake_case, lowercase).

    Insert camelCase boundaries, lowercase, collapse non-alphanumeric runs to
    ``_``, and strip edge separators: ``apiKey`` → ``api_key``,
    ``client-secret`` → ``client_secret``, ``Access_Token`` → ``access_token``,
    ``dbPassword`` → ``db_password``, ``AUTHORIZATION`` → ``authorization``,
    ``awsSecretAccessKey`` → ``aws_secret_access_key``.
    """
    return (
        _NON_ALPHANUMERIC_KEY_PATTERN.sub("_", _CAMEL_CASE_BOUNDARY_PATTERN.sub("_", key))
        .lower()
        .strip("_")
    )


def _is_sensitive_key(key: str) -> bool:
    """Return True when *key* names a credential-bearing value (issue #455).

    Segment-aware, casing-agnostic: True when the normalized key exactly
    equals a ``_SENSITIVE_KEY_SUFFIXES`` member, ends with ``_<member>``, or
    has some underscore-separated segment that is a member. The five WR-03
    negatives (``tokenizer``, ``passwordless``, ``monkeyPatch``, ``keyStore``,
    ``max_tokens``) all stay False — a secret term must appear as a full
    segment, never as a substring.
    """
    normalized = _normalize_sensitive_key(key)
    if normalized in _SENSITIVE_KEY_SUFFIXES:
        return True
    if any(normalized.endswith(f"_{member}") for member in _SENSITIVE_KEY_SUFFIXES):
        return True
    return any(segment in _SENSITIVE_KEY_SUFFIXES for segment in normalized.split("_"))


# Auth-header redaction (issue #455): case-insensitive, matches mid-line as
# well as line-start headers (curl -v / httpie output, YAML blocks, embedded
# headers). A negative lookbehind keeps the match from firing inside a word
# or a quoted JSON/YAML key; the value capture is a single token — with an
# optional Basic|Bearer|Token scheme prefix — so nothing past the token, and
# never past the end of the line, is ever consumed. A trailing lookahead
# keeps prose like ``The authorization: feature is enabled now`` out of this
# stage (the structured pair scan decides that case). Runs BEFORE the flat
# rules so a shaped value under a header is consumed whole here and never
# double-marked downstream.
_AUTHORIZATION_HEADER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_\"'])(Authorization|Proxy-Authorization|X-Api-Key|X-Auth-Token|Cookie|Set-Cookie):"
    r"[^\S\n\r]*(?:(Basic|Bearer|Token)[^\S\n\r]+)?([^\s,;\"']+)"
    r"(?=[^\S\n\r]*[,}\]\r\n]|$)",
    re.IGNORECASE,
)
# Structured key<: or =>value pairs in free text (JSON, Python-repr, YAML,
# assignment). The key may be wrapped in single or double quotes; values are
# single/double-quoted, a Bearer|Basic|Token scheme plus its opaque token, or
# a bare token, with backslash-escaped quotes honored inside quoted values.
# The bare-token class excludes the structural separators ',', ':' and '=' so
# replacing a value never swallows the separator that follows it
# ('{"token": null, "count": 3}' keeps its comma). A value that starts with
# a structural open-brace does not match (the pair is skipped so the scan
# continues to inner keys); block-style values are handled by the follow-up
# _redact_structured_blocks pass. A negative lookahead prevents re-matching a
# value that is immediately followed by a [REDACTED_*] marker (existing
# env-var/API-key/header redactions must not be clobbered).
_STRUCTURED_KEY_VALUE_PATTERN = re.compile(
    r"(['\"]?)([A-Za-z_][A-Za-z0-9_.\-]*)\1([^\S\n\r]*[:=][^\S\n\r]*)"
    r"(?:\"((?:\\.|[^\"\\])*)\"|'((?:\\.|[^'\\])*)'|(Basic|Bearer|Token)[^\S\n\r]+([^\s,;\"']+)|([^\s\"'\[\]{}():,=]++))"
    r"(?![^\S\n\r]*\[REDACTED)",
)


def _redact_structured_key_value(match: re.Match[str], text: str) -> str:
    """Replace one sensitive ``key<: or =>value`` pair (match->str transform).

    Re-emits the key's quote wrapper (group 1) and a single quote variable
    around the value so JSON/Python-repr text stays structurally valid: the
    value token becomes ``_REDACTED_CREDENTIAL``, never dropped. Bare-token
    values are redacted only when the pair is an ``=`` assignment or the value
    ends at a structural boundary, so prose like ``the token: is now
    available`` is left intact. Existing ``[REDACTED_*]`` markers and empty
    values are never re-matched.
    """
    key = match.group(2)
    if not _is_sensitive_key(key):
        return match.group(0)
    wrapped_key = f"{match.group(1)}{key}{match.group(1)}"
    sep = match.group(3)
    if match.group(6) is not None:
        # Bearer|Basic|Token <opaque>: keep the scheme, replace the token.
        token = match.group(7)
        if token is None or token == "" or "[REDACTED" in token:
            return match.group(0)
        return f"{wrapped_key}{sep}{match.group(6)} {_REDACTED_CREDENTIAL}"
    value = match.group(4) or match.group(5) or match.group(8)
    if value is None or value == "" or "[REDACTED" in value:
        return match.group(0)
    if match.group(8) is not None and not _bare_value_redactable(sep, text, match.end()):
        return match.group(0)
    quote = '"' if (match.group(4) is not None or match.group(8) is not None) else "'"
    return f"{wrapped_key}{sep}{quote}{_REDACTED_CREDENTIAL}{quote}"


def _bare_value_redactable(sep: str, text: str, end: int) -> bool:
    """Return True when a bare (unquoted) pair value should be redacted.

    ``=`` assignments are structural by nature and always redact; a ``:``
    value redacts only when it ends at a structural boundary — end of text,
    end of line, or one of ``, } ]`` — so prose like ``the token: is now
    available`` or ``The authorization: feature is enabled now`` survives
    while YAML-ish ``token: abc, other: 1`` keeps both its redaction and its
    separator.
    """
    if "=" in sep:
        return True
    return not text[end:] or re.match(r"[^\S\n\r]*[,}\]\r\n]", text[end:]) is not None


def _redact_structured_key_values(text: str) -> str:
    """Redact values of sensitive-key assignments in free text (fail-closed).

    Finds ``key<: or =>value`` pairs in JSON, Python-repr, YAML-like, and
    assignment text. For each pair whose key ``_is_sensitive_key``, replace
    the value token with ``_REDACTED_CREDENTIAL``; block-style values
    (multi-line YAML, brace-wrapped JSON) are consumed wholesale by the
    follow-up block pass. Existing ``[REDACTED_*]`` markers and empty values
    are never re-matched. Any exception degrades to ``"[REDACTION_FAILED]"``.
    """
    try:
        return _redact_structured_blocks(_redact_structured_pairs(text))
    except Exception:  # noqa: BLE001 - fail closed at every host boundary
        return "[REDACTION_FAILED]"


def _redact_structured_pairs(text: str) -> str:
    """Redact line-scoped ``key<: or =>value`` pairs, re-scanning every value.

    A plain ``re.sub`` resumes after each consumed match, so a sensitive pair
    nested inside a non-sensitive pair's value (``text: apiKey: opaque``,
    ``config: token=opaque``, ``{\"description\": \"use token=opaque here\"}``)
    would never be visited. The scan instead advances one character past any
    pair whose key is not sensitive, so nested pairs inside its value are
    still found and redacted.
    """
    out: list[str] = []
    pos = 0
    while True:
        match = _STRUCTURED_KEY_VALUE_PATTERN.search(text, pos)
        if match is None:
            break
        if _is_sensitive_key(match.group(2)):
            out.append(text[pos:match.start()])
            out.append(_redact_structured_key_value(match, text))
            pos = match.end()
        else:
            out.append(text[pos:match.start() + 1])
            pos = match.start() + 1
    out.append(text[pos:])
    return "".join(out)


_BLOCK_VALUE_PATTERN = re.compile(
    r"(['\"]?)([A-Za-z_][A-Za-z0-9_.\-]*)\1([^\S\n\r]*[:=][^\S\n\r]*)"
    r"(?=[\[{](?!REDACTED)|\n)"
)


def _redact_structured_blocks(text: str) -> str:
    """Redact block-style values under sensitive keys (fail-closed).

    Handles what the line-scoped pair scan cannot: block-style YAML
    (``apiKey:\\n  nested: <opaque>``) and brace-wrapped JSON
    (``\"apiKey\": {\\n  \"nested\": ...\\n}``) where the sensitive value
    spans lines. Each matched block is replaced wholesale with a quoted
    ``_REDACTED_CREDENTIAL`` marker so the output stays structurally valid.
    """
    try:
        out: list[str] = []
        pos = 0
        while True:
            match = _BLOCK_VALUE_PATTERN.search(text, pos)
            if match is None:
                break
            key = match.group(2)
            if not _is_sensitive_key(key):
                out.append(text[pos:match.start() + 1])
                pos = match.start() + 1
                continue
            block_end = _block_value_end(text, match.end())
            if block_end == match.end():
                # no indented block follows (empty value): skip and re-scan
                out.append(text[pos:match.start() + 1])
                pos = match.start() + 1
                continue
            out.append(text[pos:match.start()])
            quote = match.group(1) or ""
            out.append(f"{quote}{key}{quote}{match.group(3)}\"{_REDACTED_CREDENTIAL}\"")
            pos = block_end
        out.append(text[pos:])
        return "".join(out)
    except Exception:  # noqa: BLE001 - fail closed at every host boundary
        return "[REDACTION_FAILED]"


def _block_value_end(text: str, start: int) -> int:
    """Return the index just past the block opened at *start*.

    Brace/bracket blocks are consumed to their matching close (quote-aware);
    an unbalanced opener consumes to the end of the text (fail-safe).
    YAML-style blocks consume every following indented line, stopping at the
    first line that outdents back to the key's level — or return *start*
    unchanged when nothing is indented.
    """
    if start >= len(text):
        return start
    if text[start] in "[{":
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        depth = 0
        quote: str | None = None
        i = start
        while i < len(text):
            ch = text[i]
            if quote is not None:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in "\"'":
                quote = ch
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return len(text)
    # text[start] is the newline ending the key's line.
    prev = start
    i = start
    while i <= len(text):
        line_end = text.find("\n", i)
        if line_end == -1:
            line_end = len(text)
        line = text[i:line_end]
        if line and not line[0].isspace():
            break
        prev = line_end
        i = line_end + 1
    return prev


def _redact_header_value(match: re.Match[str]) -> str:
    """Replace one auth-header credential (match->str transform).

    Keeps the header name — and a ``Basic|Bearer|Token`` scheme when present —
    replacing the trailing opaque token with ``_REDACTED_CREDENTIAL``.
    """
    name = match.group(1)
    if match.group(2) is not None:
        return f"{name}: {match.group(2)} {_REDACTED_CREDENTIAL}"
    return f"{name}: {_REDACTED_CREDENTIAL}"


def redact_structured_text(s: str) -> str:
    """Redact structured free-text surfaces (fail-closed).

    Composes, in order: (1) auth-header + auth-scheme rules so a shaped value
    under a header (``Authorization: Bearer sk-1234``) is consumed whole and
    never double-marked; (2) the flat ``_REDACTION_RULES`` via
    :func:`redact_text`; (3) structured key-value redaction
    (``_redact_structured_key_values``), which skips existing
    ``[REDACTED_*]`` markers so earlier stages' output is never clobbered.
    Any exception in any stage degrades the whole field to
    ``"[REDACTION_FAILED]"`` — never raw pass-through.
    """
    try:
        s = _AUTHORIZATION_HEADER_PATTERN.sub(_redact_header_value, s)
        s = redact_text(s)
        return _redact_structured_key_values(s)
    except Exception:  # noqa: BLE001 - fail closed at every host boundary
        return "[REDACTION_FAILED]"


class Redactor:
    """Regex-driven redactor (REDA-01..06).

    Applies ``_REDACTION_RULES`` uniformly to all four ATIF text surfaces:
    ``Step.message``, ``Step.reasoning_content``, every
    ``ToolCall.arguments`` value, and every ``ObservationResult.content``
    string. Tool-call arguments and other native structures are walked
    recursively with key-aware sensitive-key detection
    (``redact_value`` / ``_is_sensitive_key``), so credentials under
    lower/mixed/camel-case keys are replaced even at nested depth while the
    surrounding shape is preserved; free-text surfaces compose the flat rules
    with auth-header and structured key-value stages
    (``redact_structured_text``). Per REDA-05 the failure mode is
    "redact-or-omit": any internal exception replaces the offending value
    with ``"[REDACTION_FAILED]"`` rather than letting the raw value through.
    """

    def _redact_optional_text(self, value: str | None) -> str | None:
        """Redact a possibly-None text field; degrade to [REDACTION_FAILED] on error."""
        if value is None:
            return None
        return redact_structured_text(value)

    def _redact_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Redact every value inside a ToolCall.arguments dict (native walk).

        Each value is walked recursively with key-aware sensitive-key
        detection (:func:`redact_value`) so nested credentials under
        lower/mixed/camel-case keys are replaced without a JSON round-trip;
        the output keeps its declared ``dict[str, Any]`` shape and preserves
        nested structure (CR-01). A failure on any one key degrades only that
        key to ``"[REDACTION_FAILED]"`` per REDA-05.
        """
        out: dict[str, Any] = {}
        for key, val in arguments.items():
            try:
                out[key] = redact_value(
                    val, _is_sensitive_key(key) if isinstance(key, str) else False
                )
            except Exception:  # noqa: BLE001 - REDA-05 redact-or-omit
                out[key] = "[REDACTION_FAILED]"
        return out

    def _redact_observation(self, observation: Observation | None) -> Observation | None:
        """Redact every string-valued ObservationResult.content in *observation*."""
        if observation is None:
            return None
        new_results: list[ObservationResult] = []
        for r in observation.results:
            new_content: Any = r.content
            if isinstance(r.content, str):
                try:
                    new_content = redact_structured_text(r.content)
                except Exception:  # noqa: BLE001 - REDA-05 redact-or-omit
                    new_content = "[REDACTION_FAILED]"
            elif isinstance(r.content, list):
                new_content = [
                    part.model_copy(update={"text": self._redact_optional_text(part.text)})
                    if part.type == "text"
                    else part
                    for part in r.content
                ]
            new_results.append(r.model_copy(update={"content": new_content}))
        return observation.model_copy(update={"results": new_results})

    def redact_step(self, step: Step) -> Step:
        """Return a redacted copy of *step* (REDA-04, REDA-05).

        Applies the redaction rules uniformly to ``message``,
        ``reasoning_content``, every ``ToolCall.arguments`` value, and
        every ``ObservationResult.content`` string. Internal exceptions
        degrade to ``"[REDACTION_FAILED]"`` for the offending field — never
        raw pass-through.

        Returns:
            A new Step instance whose text-bearing fields have been run
            through the redaction rules.
        """
        try:
            updates: dict[str, Any] = {}
            if isinstance(step.message, str):
                updates["message"] = self._redact_optional_text(step.message)
            elif isinstance(step.message, list):
                updates["message"] = [
                    part.model_copy(update={"text": self._redact_optional_text(part.text)})
                    if part.type == "text"
                    else part
                    for part in step.message
                ]
            if step.reasoning_content is not None:
                updates["reasoning_content"] = self._redact_optional_text(step.reasoning_content)
            if step.tool_calls is not None:
                redacted_calls = [
                    tc.model_copy(update={"arguments": self._redact_arguments(tc.arguments)})
                    for tc in step.tool_calls
                ]
                updates["tool_calls"] = redacted_calls
            if step.observation is not None:
                updates["observation"] = self._redact_observation(step.observation)
            if not updates:
                return step
            return step.model_copy(update=updates)
        except Exception as exc:  # noqa: BLE001 - REDA-05 redact-or-omit (top-level fallback)
            print_warning(_console, f"Redactor failure: {type(exc).__name__}")
            # Wipe every text-bearing surface — partial wipes leak secrets
            # if redaction failed mid-arguments / mid-observation.
            safe_updates: dict[str, Any] = {"message": "[REDACTION_FAILED]"}
            if step.reasoning_content is not None:
                safe_updates["reasoning_content"] = "[REDACTION_FAILED]"
            if step.tool_calls is not None:
                safe_updates["tool_calls"] = [
                    tc.model_copy(update={"arguments": {"_redaction": "[REDACTION_FAILED]"}})
                    for tc in step.tool_calls
                ]
            if step.observation is not None:
                safe_updates["observation"] = step.observation.model_copy(
                    update={
                        "results": [
                            r.model_copy(update={"content": "[REDACTION_FAILED]"})
                            for r in step.observation.results
                        ]
                    }
                )
            return step.model_copy(update=safe_updates)


# Recorder propagation uses a ContextVar (not a module-level dataclass, per
# PROJECT.md "propagated via ContextVar (not AgentState)"). Access via
# get_current_recorder() ONLY; never import _RECORDER_VAR directly. Test isolation
# goes through _reset_recorder_for_tests() (CORE-10 / D-17).
_RECORDER_VAR: ContextVar["TrajectoryRecorder | None"] = ContextVar(
    "_RECORDER_VAR", default=None,
)

# Signal-handler-safe stack of active recorders (root + forks). Python signal
# handlers fire in the main thread at bytecode boundaries — ContextVar.get()
# from that handler returns whatever context the interpreter happened to be
# in, which is non-deterministic relative to async tasks. The signal-handler
# path reads the top of this stack instead so SIGINT-flush is reliable.
_ACTIVE_RECORDERS: list["TrajectoryRecorder"] = []


def get_current_recorder() -> "TrajectoryRecorder | None":
    """Return the recorder for the current async context, or None if none active.

    The single public accessor for ``_RECORDER_VAR`` (D-10). ``agent.py`` reads
    this at the top of ``run_agent()`` and skips the entire Invocation lifecycle
    when None — direct test invocation of ``run_agent()`` without an active
    recorder is therefore a clean no-op (CORE-09).

    Signal handlers MUST use :func:`get_signal_recorder` instead — ContextVar
    reads inside a signal handler are not deterministic with respect to the
    async context where the recorder was set.

    Returns:
        The active ``TrajectoryRecorder`` instance, or ``None`` if no
        ``async with TrajectoryRecorder(...)`` block is on the stack.
    """
    return _RECORDER_VAR.get()


def get_signal_recorder() -> "TrajectoryRecorder | None":
    """Return the most recently entered recorder for signal-handler use.

    Signal handlers run in the main thread outside the asyncio task context,
    so ``ContextVar.get()`` returns non-deterministic values depending on
    where the interpreter was when the signal fired. This accessor reads
    from a module-level stack populated by ``TrajectoryRecorder.__aenter__``,
    which is set synchronously and remains valid across the entire run.

    Returns:
        The most recently entered (top-of-stack) ``TrajectoryRecorder``, or
        ``None`` if no recorder is active. For nested forks, the innermost
        recorder is returned — partial flushes cascade to ancestors via
        each recorder's own ``write_partial``.
    """
    return _ACTIVE_RECORDERS[-1] if _ACTIVE_RECORDERS else None


def _reset_recorder_for_tests() -> None:
    """Test-only: clear the recorder ContextVar and signal-handler stack.

    Use exclusively from the autouse ``_reset_trajectory_recorder`` fixture
    in ``tests/conftest.py`` (CORE-10, D-17). Production code MUST go through
    ``TrajectoryRecorder.__aenter__`` / ``__aexit__``.
    """
    _RECORDER_VAR.set(None)
    _ACTIVE_RECORDERS.clear()


def _result_extra(event: ToolResultEvent) -> dict[str, Any]:
    """Build ``ObservationResult.extra`` from a ToolResultEvent (issue #1126).

    Deterministic scalar-only metadata: ``is_error`` always, then each
    structured field only when the backend supplied it. Values are bool/int/
    float/fixed-ASCII strings — never tool output or arguments — so nothing
    free-text ever enters ``extra`` and redaction needs no special casing.
    """
    extra: dict[str, Any] = {"is_error": event.is_error}
    if event.exit_code is not None:
        extra["exit_code"] = event.exit_code
    if event.status:
        extra["status"] = event.status
    if event.duration_ms is not None:
        extra["duration_ms"] = event.duration_ms
    if event.cancelled:
        extra["cancelled"] = True
    if event.truncated:
        extra["truncated"] = True
    return extra


@dataclass
class Invocation:
    """Per-``run_agent()`` recording scope for one model conversation.

    Owns the Step buffer for one model conversation and the in-flight
    ``tool_call_id -> host-step`` map (CORE-06). ``parent`` linkage lives on
    ``TrajectoryRecorder`` (Phase 3, D-02), not on ``Invocation``.

    A ``TurnEndEvent`` closes the open Step, so a backend that emits one per
    turn produces N Steps. ``run_agent``'s normal loop does not forward
    TurnEndEvents, so in practice an invocation is usually a single Step
    spanning every turn; its ``metrics`` accumulate additively across the
    turns' MetricsEvents rather than holding the last turn's snapshot.
    ``ResultEvent`` also closes the open Step at the end of the invocation;
    ``finish()`` performs a final idempotent close so partial turns are not
    dropped.

    Tool-result-after-close: a ``ToolStartEvent`` followed by a
    ``TurnEndEvent`` and then a ``ToolResultEvent`` is legal — the result
    lands on the closed Step (the one that hosts the ``ToolStartEvent``) via
    ``model_copy`` so the observation stays attached to its originating turn.

    Attributes:
        recorder: Owning TrajectoryRecorder (shares step_id counter, Redactor).
        phase: DaydreamPhase label stamped on every Step (MAP-08, D-05).
        steps: Steps accumulated; flushed to ``recorder.steps`` at scope exit.
        _open_step_dict: In-progress agent-step state before flush.
        _in_flight_tools: tool_call_id -> ``{open_dict, closed_index}`` entry.
            While the host Step is open, ``open_dict`` is the live in-progress
            dict and ``closed_index`` is None; once closed, ``open_dict`` is
            None and ``closed_index`` is the index in ``self.steps`` of the
            closed Step. ToolResultEvents route to whichever is set.
    """

    recorder: "TrajectoryRecorder"
    phase: DaydreamPhase
    steps: list[Step] = field(default_factory=list)
    # Per-invocation timing boundaries (issue #203). Set in _InvocationCM;
    # surfaced via Trajectory.extra["subtrajectories"].
    started_at: str = ""
    ended_at: str = ""
    _open_step_dict: dict[str, Any] | None = None
    _in_flight_tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    _stop_reason: str | None = None
    _error_subtype: str | None = None
    # Per-invocation sum of the MetricsEvent values observed so far (issue
    # #747). The CostEvent handler reconciles each CostEvent's totals against
    # this sum (per-dimension take-max delta) instead of trusting the collapsed
    # per-message digits or blindly re-accumulating restated totals.
    _inv_metrics_sum: _InvMetricsSum = field(
        default_factory=lambda: _InvMetricsSum(prompt=0, completion=0, cached=0, cost=0.0)
    )

    def observe_user_step(self, prompt: str) -> None:
        """Append a user Step at invocation start (MAP-01, Pitfall 4).

        Constructs a minimal user Step — only step_id / timestamp / source /
        message / extra. NO agent-only fields (model_name, tool_calls,
        metrics, reasoning_content) so Step.validate_agent_only_fields
        passes.
        """
        try:
            self._close_open_step()
            user_step = Step(
                step_id=self.recorder._next_step_id(),
                timestamp=now_iso(),
                source="user",
                message=prompt,
                extra={
                    "daydream_phase": self.phase.value,
                    "daydream_run_flow": self.recorder.run_flow.value,
                },
            )
            self.steps.append(self.recorder.redactor.redact_step(user_step))
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
            print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")

    def mark_aborted(self, reason: str) -> None:
        """Record that this invocation was aborted (e.g. budget exceeded).

        The reason is stamped onto the closing Step's ``extra["stop_reason"]``
        when the open step is finalized (mirrors the ``extra["partial_step"]``
        mechanism), and the trajectory and its ancestors are marked partial.
        ATIF's Step model has no dedicated status field, so the ``extra`` dict
        is the established extension point.

        If the budget fires before any event is received, no step is open yet,
        so we open one here to ensure ``_close_open_step`` (called from
        ``finish()``) has a Step to stamp the reason onto.
        """
        self._stop_reason = reason
        self._ensure_open_step()
        recorder: TrajectoryRecorder | None = self.recorder
        while recorder is not None:
            recorder._aborted = True
            recorder = recorder.parent

    def mark_errored(self, subtype: str) -> None:
        """Record that this invocation ended in a fatal error.

        Mirrors :meth:`mark_aborted`: the ``subtype`` (e.g.
        ``"error_max_turns"``) is stamped onto the closing Step's
        ``extra["error"]`` / ``extra["error_subtype"]`` when the open step is
        finalized. ATIF's Step model has no dedicated status field, so the
        ``extra`` dict is the established extension point (D-19). Without this,
        a fatal failure (a backend raising mid-stream) is invisible in the
        archived trajectory.

        If the error fires before any event is received, no step is open yet,
        so we open one here to ensure ``_close_open_step`` (called from
        ``finish()`` on context-manager exit) has a Step to stamp the marker
        onto.
        """
        self._error_subtype = subtype
        self._ensure_open_step()

    def observe(self, event: "AgentEvent") -> None:
        """Dispatch an AgentEvent into the active Step buffer.

        Catch-and-degrade boundary (Architecture Q7): exceptions are caught
        here so trajectory recording NEVER crashes the user's review/fix
        run. The catch is local to this method — agent.py's event loop
        continues to surface its own errors.
        """
        try:
            self._dispatch(event)
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
            print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")

    def _reconcile_cost_delta(self, event: CostEvent) -> _CostDelta:
        """Compute the per-dimension take-max CostEvent residual (issue #747).

        The amount by which this CostEvent's total EXCEEDS the invocation's
        per-message MetricsEvent sum (``max(0, total - sum)``). Claude's
        authoritative session total exceeds the collapsed per-message single
        digits, so a positive delta is the true repair magnitude; codex/pi
        re-state the already summed totals, so delta 0 and the restatement
        never double-counts. Isolated here so the delta semantics are
        unit-testable independently of the observe() dispatch ladder.
        """
        return _CostDelta(
            prompt=max(0, (event.input_tokens or 0) - self._inv_metrics_sum["prompt"]),
            completion=max(
                0, (event.output_tokens or 0) - self._inv_metrics_sum["completion"]
            ),
            cached=max(0, (event.cached_tokens or 0) - self._inv_metrics_sum["cached"]),
            cost=max(0.0, (event.cost_usd or 0.0) - self._inv_metrics_sum["cost"]),
            reasoning=event.reasoning_tokens,
        )

    def _fold_cost_event(self, event: CostEvent) -> None:
        """Fold a ``CostEvent``'s residual onto a step and aggregate totals.

        End-of-call signal — fold per-step metrics onto the open Step so the
        renderer's per-step rollup sees real cost / tokens (Bug C: previously
        CostEvent only updated _final_totals).

        Per-dimension take-max delta (issue #747): the amount by which this
        CostEvent's total EXCEEDS the invocation's per-message sum. Claude's
        authoritative session total exceeds the collapsed per-message single
        digits -> positive delta repairs the under-count; codex/pi re-state
        totals equal to the per-message sum -> delta 0, so the restatement never
        double-counts. Never negative, never subtraction.
        """
        delta = self._reconcile_cost_delta(event)
        existing = (
            self._open_step_dict["_metrics"] if self._open_step_dict is not None else None
        )
        if existing is not None:
            # A MetricsEvent already populated this step. Fold the residual
            # delta onto it (previously only the recorder-level tally absorbed
            # it, so the Step's rollup dropped the magnitude and ``Sigma steps <
            # final``). Then backfill cost_usd / reasoning the per-message path
            # didn't surface. ``Sigma steps == final`` holds here too (issue
            # #747).
            if delta.nonzero:
                existing = _merge_metrics(
                    existing,
                    delta.residual_metrics(
                        cost_usd=event.cost_usd, include_reasoning=False
                    ),
                )
            updates: dict[str, Any] = {}
            if existing.cost_usd is None and event.cost_usd is not None:
                updates["cost_usd"] = event.cost_usd
            # #192: backfill reasoning_tokens via Metrics.extra when the
            # MetricsEvent path didn't carry it (mirrors cost_usd backfill).
            if (
                event.reasoning_tokens is not None
                and (existing.extra is None or "reasoning_tokens" not in existing.extra)
            ):
                merged_extra = dict(existing.extra or {})
                merged_extra["reasoning_tokens"] = event.reasoning_tokens
                updates["extra"] = merged_extra
            if updates:
                existing = existing.model_copy(update=updates)
            assert self._open_step_dict is not None
            self._open_step_dict["_metrics"] = existing
        elif delta.nonzero:
            # No metrics-bearing step is open and the residual is non-zero: mint
            # a fresh residual Step holding exactly the delta this CostEvent
            # adds beyond the invocation's per-message sum, so ``Sigma steps ==
            # recorder total`` holds (issue #747). reasoning_tokens (#192) is a
            # subset of completion_tokens and rides in Metrics.extra (D-03).
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_metrics"] = delta.residual_metrics(
                cost_usd=event.cost_usd, include_reasoning=True
            )
        # else: a delta-0 restatement (codex/pi) with no metrics-bearing step
        # open has nothing to fold — do NOT mint a phantom all-zero Metrics Step
        # that would inflate total_steps and per-step lists in archived
        # trajectories and rendered reports (issue #747).
        if event.model_name:
            if self._open_step_dict is not None:
                self._open_step_dict["_model_name"] = event.model_name
            self.recorder._upgrade_model_name(event.model_name)
        # Aggregate the delta into recorder-level totals: per-dimension take-max
        # at the per-invocation level, summed across invocations (a multi-phase
        # run shares one recorder, so each phase's session total should sum). A
        # CostEvent-only backend (no MetricsEvents at all) still totals by the
        # full delta.
        self.recorder._accumulate_metrics(
            prompt_tokens=delta.prompt,
            completion_tokens=delta.completion,
            cached_tokens=delta.cached,
            cost_usd=None if event.cost_usd is None else delta.cost,
        )

    def _dispatch(self, event: Any) -> None:
        # Function-local imports avoid load-order cycles with daydream.backends.
        from daydream.backends import (
            CostEvent,
            MetricsEvent,
            ResultEvent,
            TextEvent,
            ThinkingEvent,
            ToolResultEvent,
            ToolStartEvent,
            TurnEndEvent,
        )

        if isinstance(event, TextEvent):
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_text_chunks"].append(event.text)
        elif isinstance(event, ThinkingEvent):
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_thinking_chunks"].append(event.text)
        elif isinstance(event, ToolStartEvent):
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_tool_calls"].append(
                ToolCall(tool_call_id=event.id, function_name=event.name, arguments=event.input or {})
            )
            # Map tool_call_id -> THIS open step so paired ToolResultEvent lands
            # on the SAME step (CORE-06, Pitfall 3). The closed_index slot is
            # filled in by _close_open_step() if the host Step closes before
            # the matching ToolResultEvent arrives.
            self._in_flight_tools[event.id] = {
                "open_dict": self._open_step_dict,
                "closed_index": None,
            }
        elif isinstance(event, ToolResultEvent):
            host = self._in_flight_tools.pop(event.id, None)
            if host is None:
                # Dangling ToolResultEvent (Codex pending-id miss, Pitfall 3).
                # Mark via extra.unmatched_tool_results; do NOT emit a dangling
                # source_call_id reference (Trajectory validator hard-fail).
                self._ensure_open_step()
                assert self._open_step_dict is not None
                self._open_step_dict["_unmatched_tool_results"].append(event.id)
                return
            open_dict = host["open_dict"]
            if open_dict is not None:
                # Host Step is still open — append to its observation buffer.
                # Failure metadata (is_error/exit_code/status, issue #1126)
                # rides ObservationResult.extra alongside source_call_id and
                # content.
                open_dict["_observation_results"].append(
                    ObservationResult(
                        source_call_id=event.id,
                        content=event.output,
                        extra=_result_extra(event),
                    )
                )
            else:
                # Host Step was closed by an intervening TurnEndEvent. Patch the
                # closed Step in-place via model_copy so the observation stays
                # bound to its originating turn.
                self._amend_closed_step_observation(
                    closed_index=host["closed_index"],
                    result=ObservationResult(
                        source_call_id=event.id,
                        content=event.output,
                        extra=_result_extra(event),
                    ),
                )
        elif isinstance(event, MetricsEvent):
            # EVNT-02 attribute names verbatim. prompt_tokens is the total
            # input (backends fold cache read+creation into it); cached_tokens
            # is the cache-read hit subset (a subset of prompt_tokens, not
            # added).
            #
            # D-04 correlation fallback (Codex): Codex emits no per-message
            # id, so MetricsEvent.message_id is always '' on the Codex path.
            # In the common Codex case a TurnEndEvent closes the content Step
            # before turn.completed fires, so this MetricsEvent arrives with
            # no open Step and the ``target is None`` branch below opens a
            # fresh Step to hold the metrics. Correlation is therefore
            # TURN-granular for Codex — one MetricsEvent per turn.completed →
            # one metrics-bearing Step per turn — which is coarser than
            # Claude's per-message correlation via message_id. This is the
            # documented, tested fallback for the missing id surface (see
            # tests/contract/test_backend_codex_trajectory.py); it is not a
            # silent coarsening.
            target = self._open_step_dict
            if target is None and not self.steps:
                # No open Step and no prior agent Step to attach to: mint a
                # fresh one (a metrics-only backend with no Text/TurnEnd ever
                # reaching the recorder).
                self._ensure_open_step()
                target = self._open_step_dict
            # #192: reasoning_tokens is a SUBSET of completion_tokens (not
            # additive). Vendored Metrics has no dedicated field (D-03), so
            # carry it via the documented extension carrier ``extra``.
            #
            # Accumulate-or-assign: a Step spanning several turns carries the
            # sum of those turns' usage, not the last turn's snapshot — every
            # turn was billed. Keeps ``final == Σ steps`` (the recorder-level
            # tally below accumulates every event too).
            incoming = Metrics(
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
                cached_tokens=event.cached_tokens,
                cost_usd=event.cost_usd,
                extra=_reasoning_extra(event.reasoning_tokens),
            )
            if target is not None:
                # Open Step still in flight (Claude/Pi — usage arrives while the
                # turn Step is open). Fold the per-turn metrics onto it.
                prior = target["_metrics"]
                target["_metrics"] = (
                    incoming if prior is None else _merge_metrics(prior, incoming)
                )
                if event.model_name:
                    target["_model_name"] = event.model_name
            else:
                # Closed-Step fallback (Codex D-04): the turn's usage arrives on
                # ``turn.completed`` AFTER its content Step was closed by the
                # ``TurnEndEvent``. Fold onto the most recently closed agent Step
                # so each turn keeps ONE metrics-bearing Step — matching Claude,
                # which folds onto its still-open final turn Step — instead of
                # minting a phantom empty-message Step that splits Codex/Claude
                # Step parity (test_backend_step_parity.py, issue #747).
                self._fold_metrics_into_closed_last_step(event, incoming)
            if event.model_name:
                self.recorder._upgrade_model_name(event.model_name)
            # Aggregate into recorder-level totals for FinalMetrics (MAP-07),
            # and into the invocation-level sum the CostEvent delta reconciles
            # against (issue #747).
            if event.prompt_tokens is not None:
                self._inv_metrics_sum["prompt"] += event.prompt_tokens
            if event.completion_tokens is not None:
                self._inv_metrics_sum["completion"] += event.completion_tokens
            if event.cached_tokens is not None:
                self._inv_metrics_sum["cached"] += event.cached_tokens
            if event.cost_usd is not None:
                self._inv_metrics_sum["cost"] += event.cost_usd
            self.recorder._accumulate_metrics(
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
                cached_tokens=event.cached_tokens,
                cost_usd=event.cost_usd,
            )
        elif isinstance(event, CostEvent):
            # End-of-call signal — fold the CostEvent's per-dimension
            # take-max residual delta onto the open/metrics step (or mint a
            # residual step) and aggregate it into recorder totals. Isolated
            # in _fold_cost_event so the CostEvent merge/backfill wiring is
            # not inlined inside _dispatch (cuts the dispatch complexity
            # concentration; issue #747).
            self._fold_cost_event(event)
        elif isinstance(event, ResultEvent):
            if event.model_name:
                if self._open_step_dict is not None:
                    self._open_step_dict["_model_name"] = event.model_name
                else:
                    for index, step in enumerate(self.steps):
                        if (
                            step.source == "agent"
                            and (step.model_name or "") in _GENERIC_MODEL_LABELS
                        ):
                            self.steps[index] = self.recorder.redactor.redact_step(
                                step.model_copy(update={"model_name": event.model_name})
                            )
                self.recorder._upgrade_model_name(event.model_name)
            self._close_open_step()
        elif isinstance(event, TurnEndEvent):
            # Per-turn close: a TurnEndEvent arriving while no Step is open is
            # a no-op (never invent an empty Step just to close it).
            if self._open_step_dict is not None:
                self._close_open_step()

    def _ensure_open_step(self) -> None:
        """Open a new agent step if none currently in flight."""
        if self._open_step_dict is not None:
            return
        self._open_step_dict = {
            "_text_chunks": [],
            "_thinking_chunks": [],
            "_tool_calls": [],
            "_observation_results": [],
            "_metrics": None,
            "_model_name": self.recorder.agent_model_name,
            "_unmatched_tool_results": [],
        }

    def _materialize_agent_step(
        self, d: dict[str, Any], *, step_id: int, extra_overrides: dict[str, Any]
    ) -> Step:
        """Materialize the open-step dict *d* into a redacted agent Step."""
        message_text = "".join(d["_text_chunks"])
        reasoning = "\n".join(d["_thinking_chunks"]) if d["_thinking_chunks"] else None
        tool_calls = list(d["_tool_calls"]) or None
        observation = (
            Observation(results=list(d["_observation_results"]))
            if d["_observation_results"]
            else None
        )
        extra: dict[str, Any] = {
            "daydream_phase": self.phase.value,
            "daydream_run_flow": self.recorder.run_flow.value,
            **extra_overrides,
        }
        agent_step = Step(
            step_id=step_id,
            timestamp=now_iso(),
            source="agent",
            message=message_text,
            model_name=d["_model_name"],
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            observation=observation,
            metrics=d["_metrics"],
            llm_call_count=1,
            extra=extra,
        )
        return self.recorder.redactor.redact_step(agent_step)

    def _close_open_step(self) -> None:
        """Finalize the current open step into a Pydantic Step + redact + append.

        Called per ``TurnEndEvent`` (assistant-turn boundary), per
        ``ResultEvent`` (end-of-call), and once more from ``finish()`` for an
        idempotent final flush. After appending, any ``_in_flight_tools``
        entry whose host was the just-closed dict is amended to reference the
        closed Step by its index in ``self.steps`` so a ToolResultEvent
        arriving after the close still lands on the right turn.
        """
        if self._open_step_dict is None:
            return
        d = self._open_step_dict
        self._open_step_dict = None

        extra_overrides: dict[str, Any] = {}
        if d["_unmatched_tool_results"]:
            extra_overrides["unmatched_tool_results"] = list(d["_unmatched_tool_results"])
        if self._stop_reason is not None:
            extra_overrides["stop_reason"] = self._stop_reason
        if self._error_subtype is not None:
            extra_overrides["error"] = True
            extra_overrides["error_subtype"] = self._error_subtype

        self.steps.append(
            self._materialize_agent_step(
                d, step_id=self.recorder._next_step_id(), extra_overrides=extra_overrides
            )
        )
        closed_index = len(self.steps) - 1
        # Amend in-flight entries whose host Step just closed so a delayed
        # ToolResultEvent can still find its host via closed_index.
        for entry in self._in_flight_tools.values():
            if entry["open_dict"] is d:
                entry["open_dict"] = None
                entry["closed_index"] = closed_index

    def _amend_closed_step_observation(
        self, *, closed_index: int, result: ObservationResult
    ) -> None:
        """Attach *result* to a closed Step via ``model_copy``.

        Used when a ToolResultEvent arrives after a ``TurnEndEvent`` closed
        the host Step. The replacement is redacted again because the new
        ObservationResult content has not yet been run through the redactor.
        """
        existing = self.steps[closed_index]
        if existing.observation is None:
            new_observation = Observation(results=[result])
        else:
            new_observation = existing.observation.model_copy(
                update={"results": [*existing.observation.results, result]}
            )
        updated = existing.model_copy(update={"observation": new_observation})
        self.steps[closed_index] = self.recorder.redactor.redact_step(updated)

    def _fold_metrics_into_closed_last_step(
        self, event: Any, incoming: Metrics
    ) -> None:
        """Fold a terminal ``MetricsEvent`` onto the last closed agent Step.

        Codex emits each turn's usage on ``turn.completed``, which arrives AFTER
        that turn's content Step was closed by its ``TurnEndEvent`` (D-04).
        Rather than mint a phantom empty-message Step — which would make Codex's
        agent Step stream one longer than Claude's, where the same usage lands on
        the still-open final turn Step — fold the usage into the most recently
        closed agent Step so both backends emit identical Step shapes (the parity
        gate in ``tests/contract/test_backend_step_parity.py``). ``Σ steps ==
        final`` is preserved because the metrics still accumulate on a Step
        (issue #747).
        """
        for idx in range(len(self.steps) - 1, -1, -1):
            step = self.steps[idx]
            if step.source != "agent":
                continue
            metrics = (
                incoming
                if step.metrics is None
                else _merge_metrics(step.metrics, incoming)
            )
            updates: dict[str, Any] = {"metrics": metrics}
            if event.model_name:
                updates["model_name"] = event.model_name
            self.steps[idx] = self.recorder.redactor.redact_step(
                step.model_copy(update=updates)
            )
            return
        # No closed agent Step exists to fold onto (self.steps is non-empty but
        # every step is non-agent, e.g. only user/context steps recorded). Mint a
        # metrics-bearing agent Step so the recorder-level tally still sums to
        # final (Sigma steps == final) instead of silently dropping the metrics
        # while final_metrics keeps them (issue #747).
        self._ensure_open_step()
        assert self._open_step_dict is not None
        self._open_step_dict["_metrics"] = incoming
        if event.model_name:
            self._open_step_dict["_model_name"] = event.model_name

    def snapshot_steps(self, *, snapshot_step_id: int | None = None) -> list[Step]:
        """Return steps including a materialized copy of any open step (signal-safe, non-mutating).

        Args:
            snapshot_step_id: Pre-allocated step ID for the partial step. When multiple
                invocations are active, the caller allocates unique IDs to avoid duplicates.

        In-flight tool calls carry the same interrupted markers ``finish()``
        emits, so a partial flush and the final record agree about in-flight
        tool outcomes. Unlike ``finish()`` this never mutates: ``_in_flight_tools``
        stays populated (a later ``finish()`` still marks the same calls) and no
        Step is replaced in place.
        """
        in_flight = list(self._in_flight_tools.values())
        if self._open_step_dict is None:
            steps = list(self.steps)
        else:
            d = self._open_step_dict
            extra_overrides: dict[str, Any] = {"partial_step": True}
            if d["_unmatched_tool_results"]:
                extra_overrides["unmatched_tool_results"] = list(d["_unmatched_tool_results"])
            step_id = snapshot_step_id if snapshot_step_id is not None else self.recorder._step_id_counter + 1
            steps = [
                *self.steps,
                self._materialize_agent_step(d, step_id=step_id, extra_overrides=extra_overrides),
            ]
        # Markers land per host Step in the same LIFO order finish()'s popitem loop uses.
        for host in reversed(in_flight):
            if host["closed_index"] is not None:
                steps[host["closed_index"]] = self._with_observation_result(
                    steps[host["closed_index"]], self._interrupted_marker()
                )
            elif self._open_step_dict is not None and host["open_dict"] is self._open_step_dict:
                steps[-1] = self._with_observation_result(steps[-1], self._interrupted_marker())
        return steps

    @staticmethod
    def _interrupted_marker() -> ObservationResult:
        """Synthetic terminal outcome for a tool call still in flight.

        Deliberately carries NO ``source_call_id``: consumers derive completed
        tool calls from an observation result's string ``source_call_id``
        (``deep/coverage._completed_read_paths``, shared by the uncovered-file
        sweep, per-stack verdict evidence and diagram-grounding receipts), so
        stamping the in-flight tool's id here would make an interrupted read
        derive as completed and flip their fail-open invariant ("an interrupted
        read must NOT count as coverage") to fail-closed. Null ``source_call_id``
        is the ATIF v1.7 encoding for "not a standard tool-call result" and
        keeps the marker schema-valid. Content is fixed ASCII and ``extra``
        carries only the two fixed keys, so the marker is redaction-stable.
        """
        return ObservationResult(
            source_call_id=None,
            content=INCOMPLETE_CALL_CONTENT,
            extra={"is_error": True, "status": "interrupted"},
        )

    @staticmethod
    def _with_observation_result(step: Step, result: ObservationResult) -> Step:
        """Return a copy of *step* with *result* appended to its observation.

        Non-mutating twin of ``_amend_closed_step_observation`` for the
        signal-safe snapshot path. No re-redaction pass needed: the Step is
        already a redacted copy and the marker content is fixed ASCII.
        """
        if step.observation is None:
            observation = Observation(results=[result])
        else:
            observation = step.observation.model_copy(
                update={"results": [*step.observation.results, result]}
            )
        return step.model_copy(update={"observation": observation})

    def _emit_incomplete_call_markers(self) -> None:
        """Append a synthetic failure observation for every still-in-flight tool call.

        Called from ``finish()`` after ``_close_open_step()``, and every host
        entry's ``open_dict`` is nulled when its Step closes, so each marker
        lands on the closed Step via ``closed_index`` (the open-dict arm is
        unreachable by construction and omitted). Popping entries as we go
        makes this idempotent — a second ``finish()`` finds nothing. Each
        marker is ``_interrupted_marker()``: fixed ASCII content, the two fixed
        ``extra`` keys, and no ``source_call_id`` so no consumer derives an
        interrupted call as completed (see ``_interrupted_marker``).
        """
        while self._in_flight_tools:
            _, host = self._in_flight_tools.popitem()
            self._amend_closed_step_observation(
                closed_index=host["closed_index"],
                result=self._interrupted_marker(),
            )

    def finish(self) -> None:
        """Close any open step, mark in-flight tool calls interrupted, and flush.

        After the final step close, every tool call still in flight (no
        matching ``ToolResultEvent`` arrived) gets a synthetic
        ``ObservationResult`` marker appended to its host Step so no tool call
        dangles without a terminal outcome.
        """
        self._close_open_step()
        self._emit_incomplete_call_markers()
        self.recorder._extend_steps(self.steps)


@dataclass
class PhaseEvent:
    """Explicit phase-boundary event (``phase_start`` / ``phase_end``).

    Emitted by :meth:`TrajectoryRecorder.emit_phase_start` /
    :meth:`TrajectoryRecorder.emit_phase_end` and serialized into
    ``Trajectory.extra["phase_events"]`` so a per-phase wall-clock breakdown can
    be reconstructed without inferring invocation→phase membership from step
    timestamps (issue #203).

    Attributes:
        phase: The :class:`DaydreamPhase` this event brackets.
        event: ``"phase_start"`` or ``"phase_end"``.
        timestamp: ISO 8601 UTC timestamp (via :func:`now_iso`).
        metadata: Optional structured metadata (e.g. ``{"stage": "review"}``
            for the deep orchestrator's sub-stages).
    """

    phase: DaydreamPhase
    event: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable representation."""
        d: dict[str, Any] = {
            "phase": self.phase.value,
            "event": self.event,
            "timestamp": self.timestamp,
        }
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d


@asynccontextmanager
async def phase_scope(phase: DaydreamPhase, **metadata: Any) -> Any:
    """Async context manager that emits ``phase_start``/``phase_end`` events.

    Reads the active recorder via :func:`get_current_recorder`; a no-op when no
    recorder is active (e.g. direct phase invocation outside a run). Used by
    ``runner.py`` and ``deep/orchestrator.py`` to bracket phase boundaries so
    the trajectory JSON carries explicit timing events (issue #203).
    """
    recorder = get_current_recorder()
    if recorder is not None:
        recorder.emit_phase_start(phase, **metadata)
    try:
        yield
    finally:
        if recorder is not None:
            recorder.emit_phase_end(phase, **metadata)


@dataclass
class TrajectoryRecorder:
    """Owns the per-run ATIF Trajectory and writes it to disk on clean exit.

    A recorder is opened via ``async with`` from ``runner.py``. ``__aenter__``
    sets ``_RECORDER_VAR``; ``__aexit__`` writes the trajectory and clears the
    context variable. Redaction failures are represented explicitly rather than
    allowing sensitive values to pass through.

    Attributes:
        path: Output JSON path; default ``<target>/.daydream/runs/<session_id>/trajectory.json``.
        run_flow: Per-trajectory invariant (D-07) stamped on every Step.
        target_dir: Repo/target directory; recorded into Trajectory.extra.
        agent_model_name: Active model name; stamped into Agent and every
            agent Step's model_name.
        redactor: Redaction policy applied before trajectory data is written.
        session_id: UUID4 for this run, supplied by the caller (CORE-07).
        steps: Sequential Steps from every Invocation, step_id 1..N.
        pr_number: GitHub PR number if reviewing a PR. Stored in trajectory extra.
        pr_repo: GitHub repo (``owner/repo``) if reviewing a PR. Stored in trajectory extra.
        backend_name: Resolved backend kind (claude/codex/pi/osprey) for the run,
            resolved by ``_open_recorder`` via the ``per_stack_review`` phase for
            deep-flow runs (the phase that governs the deep flow's actual review
            fan-out) and via the ``review`` phase otherwise. Stored in
            trajectory extra as ``backend`` when non-empty.
        review_backend_name: Resolved backend kind for the review phase. Stored in
            trajectory extra as ``review_backend`` when set.
        fix_backend_name: Resolved backend kind for the fix phase. Stored in
            trajectory extra as ``fix_backend`` when set; left empty by
            ``_open_recorder`` for flows that never run fix (e.g. improve), so
            the key is omitted from their trajectories.
        test_backend_name: Resolved backend kind for the test phase. Stored in
            trajectory extra as ``test_backend`` when set; left empty by
            ``_open_recorder`` for flows that never run test (e.g. improve), so
            the key is omitted from their trajectories.
        _step_id_counter: Monotonic; never decreases (Pitfall 1).
        _final_totals: Running tally for FinalMetrics aggregation (MAP-07).
        _previous_token: ContextVar reset token; used by __aexit__ to restore.
    """

    path: Path
    run_flow: DaydreamRunFlow
    target_dir: Path
    agent_model_name: str
    session_id: str
    redactor: Redactor = field(default_factory=Redactor)
    steps: list[Step] = field(default_factory=list)
    parent: TrajectoryRecorder | None = None
    descriptor: str = ""
    explicit_path: bool = False
    pr_number: int | None = None
    pr_repo: str | None = None
    backend_name: str = ""
    review_backend_name: str = ""
    fix_backend_name: str = ""
    test_backend_name: str = ""
    _step_id_counter: int = 0
    _final_totals: dict[str, Any] = field(default_factory=lambda: _INITIAL_TOTALS.copy())
    _folded_fork_totals: bool = False
    _previous_token: Any = None
    _registered_siblings: list[tuple[Path, str]] = field(default_factory=list)
    # Active invocations whose in-flight steps haven't been flushed yet.
    # write_partial reads this so SIGINT mid-run_agent() captures partial
    # work rather than dropping it.
    _active_invocations: list[Invocation] = field(default_factory=list)
    # Explicit phase-boundary events (phase_start/phase_end) emitted by
    # emit_phase_start/emit_phase_end via phase_scope. Serialized into
    # Trajectory.extra["phase_events"] when non-empty (issue #203).
    _phase_events: list[PhaseEvent] = field(default_factory=list)
    # Per-Invocation timing summaries registered at _InvocationCM.__aexit__.
    # Serialized into Trajectory.extra["subtrajectories"] when non-empty.
    _subtrajectories: list[dict[str, Any]] = field(default_factory=list)
    # Resolved review-profile provenance (issue #885, R12): schema version,
    # name, source kind, and canonical digest of the profile this run executed
    # under, recorded via ``record_profile`` by the runner composition root.
    # Serialized into Trajectory.extra["profile_*"] when set (new runs).
    _profile: dict[str, Any] | None = None
    _aborted: bool = False
    on_write: Callable[[TrajectoryRecorder, str], None] | None = None

    async def __aenter__(self) -> "TrajectoryRecorder":
        self._previous_token = _RECORDER_VAR.set(self)
        _ACTIVE_RECORDERS.append(self)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, _exc_tb: Any) -> None:
        try:
            if exc_type is not None:
                self._aborted = True
            self._write()
        except Exception as exc:  # noqa: BLE001 - branch on explicit_path per D-06
            if self.explicit_path:
                # D-06: user asked for it, deliver or fail loud
                print_error(
                    _console,
                    "Trajectory write failed",
                    f"{type(exc).__name__}: {exc}",
                )
                raise SystemExit(2) from exc
            # Implicit/default path — degrade with warning per CORE-09 / D-11
            print_warning(
                _console,
                f"Trajectory write failed: {type(exc).__name__}: {exc}",
            )
        finally:
            if self._previous_token is not None:
                _RECORDER_VAR.reset(self._previous_token)
                self._previous_token = None
            try:
                _ACTIVE_RECORDERS.remove(self)
            except ValueError:
                pass  # already removed by reset_recorder_for_tests or never registered

    def invocation(self, *, phase: DaydreamPhase) -> "_InvocationCM":
        """Open an Invocation scope for one ``run_agent()`` call.

        Returns an async-context-manager that flushes its accumulated Steps
        to ``self.steps`` on exit. Phase 2 has no parent linkage — flat
        sequential append per D-08.
        """
        return _InvocationCM(self, phase)

    def current_phase(self) -> DaydreamPhase | None:
        """Return the firing :class:`DaydreamPhase`, or None if no invocation is active.

        The public read-seam for the phase of the innermost open Invocation,
        complementing :func:`get_current_recorder`. The replay harness reads this
        during ``execute()`` iteration to serve the right per-phase fixture: by
        the time a backend's first event is pulled, ``agent.py`` has already
        opened ``recorder.invocation(phase=...)`` around the stream, so the
        active phase is observable here.

        Returns:
            The ``.phase`` of the last (innermost) active Invocation, or
            ``None`` when ``self._active_invocations`` is empty — the documented,
            correct default for the direct-call no-op path (no active invocation),
            mirroring :func:`get_current_recorder`.
        """
        return self._active_invocations[-1].phase if self._active_invocations else None

    def _emit_phase_event(self, phase: DaydreamPhase, event: str, **metadata: Any) -> None:
        """Append a :class:`PhaseEvent` stamped with ``now_iso()``."""
        self._phase_events.append(
            PhaseEvent(phase=phase, event=event, timestamp=now_iso(), metadata=metadata)
        )

    def record_profile(
        self, *, schema_version: int, name: str, source_kind: str, digest: str
    ) -> None:
        """Record the resolved review-profile provenance for this run (R12).

        Called exactly once at the runner composition root (issue #885) with
        the run's resolved profile: schema version, human-readable name,
        source kind (explicit/env/repo/default), and the canonical digest.
        Serialized into ``Trajectory.extra`` as ``profile_schema_version`` /
        ``profile_name`` / ``profile_source_kind`` / ``profile_digest`` so a
        future optimizer can attribute results to the exact policy tested.
        Required on new runs; older trajectories simply omit the keys.

        Args:
            schema_version: Profile ``schema_version`` (1).
            name: Human-readable profile name.
            source_kind: One of ``explicit``/``env``/``repo``/``default``.
            digest: Canonical SHA-256 digest of the profile value.
        """
        self._profile = {
            "profile_schema_version": schema_version,
            "profile_name": name,
            "profile_source_kind": source_kind,
            "profile_digest": digest,
        }

    def emit_phase_start(self, phase: DaydreamPhase, **metadata: Any) -> None:
        """Record a ``phase_start`` boundary event (issue #203).

        Args:
            **metadata: Optional structured metadata (e.g. ``stage="review"``
                for the deep orchestrator's DEEP sub-stages).
        """
        self._emit_phase_event(phase, "phase_start", **metadata)

    def emit_phase_end(self, phase: DaydreamPhase, **metadata: Any) -> None:
        """Record a ``phase_end`` boundary event (issue #203).

        Args:
            **metadata: Optional structured metadata (mirrors emit_phase_start.
        """
        self._emit_phase_event(phase, "phase_end", **metadata)

    def emit_file_group_budget_exceeded(
        self, *, file: str, reason: str, items_processed: int, items_skipped: int
    ) -> None:
        """Record a ``file_group_budget_exceeded`` event for the FIX phase (#201).

        Emitted by ``phase_fix_parallel`` when a per-file-group aggregate budget
        fires, so future perf triage of a runaway file group (the #186 pattern)
        is mechanical: the trajectory names the file, the ceiling that tripped,
        and how many findings were processed vs. skipped. Serialized into
        ``Trajectory.extra["phase_events"]`` alongside the phase boundaries.

        Args:
            file: File-group key whose budget was exceeded (or ``"<no-file>"``).
            reason: Which ceiling tripped (e.g. ``"group_serial_item_limit"``).
            items_processed: Findings fixed before the budget fired.
            items_skipped: Remaining findings in the group left unfixed.
        """
        self._emit_phase_event(
            DaydreamPhase.FIX,
            "file_group_budget_exceeded",
            file=file,
            reason=reason,
            items_processed=items_processed,
            items_skipped=items_skipped,
        )

    def emit_supervisor_verdict(self, finding_id: int, action: str, reason: str) -> None:
        """Record a findings supervisor verdict in the deep phase."""
        self._emit_phase_event(
            DaydreamPhase.DEEP, "supervisor_verdict", finding_id=finding_id, action=action, reason=reason
        )

    def emit_tool_veto(
        self, tool_name: str, reason: str, *, phase: DaydreamPhase = DaydreamPhase.FIX
    ) -> None:
        """Record a tool-supervisor veto in the firing phase."""
        self._emit_phase_event(phase, "tool_veto", tool_name=tool_name, reason=reason)

    def emit_command_validation_summary(
        self,
        *,
        total_candidates: int,
        accepted: int,
        rejected: int,
        reasons: dict[str, int],
    ) -> None:
        """Record a redacted repository-command validation summary."""
        metadata: dict[str, Any] = {
            "counts": {
                "total_candidates": total_candidates,
                "accepted": accepted,
                "rejected": rejected,
            },
            "reasons": dict(sorted(reasons.items())),
        }
        self._emit_phase_event(
            DaydreamPhase.RECON,
            "command_validation",
            **metadata,
        )

    def _register_subtrajectory(self, inv: Invocation) -> None:
        """Register a per-Invocation timing summary (issue #203).

        Called from ``_InvocationCM.__aexit__`` after ``finish()`` so the
        Invocation's steps are finalized. The summary is surfaced via
        ``Trajectory.extra["subtrajectories"]`` when non-empty.
        """
        self._subtrajectories.append(
            {
                "phase": inv.phase.value,
                "started_at": inv.started_at,
                "ended_at": inv.ended_at,
                "step_ids": [s.step_id for s in inv.steps],
            }
        )

    def _register_fork_subtrajectory(
        self,
        *,
        phase: str,
        descriptor: str,
        started_at: str,
        ended_at: str,
        sibling_trajectory_ref: str,
    ) -> None:
        """Register a per-fork timing summary on the parent trajectory."""
        self._subtrajectories.append(
            {
                "phase": phase,
                "descriptor": descriptor,
                "started_at": started_at,
                "ended_at": ended_at,
                "sibling_trajectory_ref": sibling_trajectory_ref,
            }
        )

    def _next_step_id(self) -> int:
        self._step_id_counter += 1
        return self._step_id_counter

    def _extend_steps(self, steps: list[Step]) -> None:
        """Merge a finished Invocation's steps back in ``step_id`` order.

        ``step_id`` is allocated when a step opens but flushed here when its
        Invocation closes, so append-order only equals id-order while at most
        one Invocation is open. Concurrent siblings on one recorder (wonder
        alongside the per-stack fan-out, whose ``create_dispatch_step`` appends
        straight to ``self.steps``) break that: the run then dies at write time
        on ATIF's "sequential from 1" check. Sorting on insert keeps the
        documented ``steps: step_id 1..N`` invariant true by construction.
        """
        self.steps.extend(steps)
        self.steps.sort(key=lambda s: s.step_id)

    def _upgrade_model_name(self, candidate: str) -> None:
        """Promote *candidate* over a generic backend label.

        Runner stamps the recorder with a generic alias (``"claude"``,
        ``"codex"``, ``"osprey"``, ``"unknown"``) or empty string at init
        since the real SDK model id is only known after the first agent turn
        streams back. The first real model id observed from MetricsEvent /
        CostEvent / ResultEvent
        upgrades the recorder's ``agent_model_name`` so the rendered
        Trajectory.agent carries the real id rather than the alias. Any
        already-closed agent Steps carrying the same provisional label are
        upgraded too; this matters when a backend emits its identity only in a
        terminal ResultEvent after TurnEndEvent has closed the Step.
        """
        if candidate and (self.agent_model_name or "") in _GENERIC_MODEL_LABELS:
            self.agent_model_name = candidate
            for index, step in enumerate(self.steps):
                if (
                    step.source == "agent"
                    and (step.model_name or "") in _GENERIC_MODEL_LABELS
                ):
                    self.steps[index] = self.redactor.redact_step(
                        step.model_copy(update={"model_name": candidate})
                    )

    def _accumulate_metrics(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        if prompt_tokens is not None:
            self._final_totals["prompt"] += prompt_tokens
        if completion_tokens is not None:
            self._final_totals["completion"] += completion_tokens
        if cached_tokens is not None:
            self._final_totals["cached"] += cached_tokens
        if cost_usd is not None:
            self._final_totals["cost"] += cost_usd
            self._final_totals["any_cost_seen"] = True

    def compute_wall_clock_seconds(self) -> float | None:
        """Total wall-clock seconds spanned by recorded step timestamps.

        Derived from the earliest and latest ``Step.timestamp`` across the
        recorder's steps. Returns ``None`` when fewer than two timestamped
        steps exist (no measurable span).

        Independent of the eval pass: this mirrors the timestamp-span derivation
        in :func:`daydream.eval.analyzer.analyze_timing`, but reads in-memory
        steps so every archived run captures duration even when ``--no-eval``
        skips the deterministic evaluation pass. Fork-only steps live in sibling
        recorders and are not
        included here; the main flow's span bounds them because forks are
        dispatched and merged within it.

        Returns:
            Rounded duration in seconds, or ``None`` when unmeasurable —
            fewer than two timestamped steps, or an unparseable timestamp.
        """
        try:
            timestamps = [parse_iso_timestamp(s.timestamp) for s in self.steps if s.timestamp]
        except ValueError:
            return None
        if len(timestamps) < 2:
            return None
        return round((max(timestamps) - min(timestamps)).total_seconds(), 1)

    def compute_phase_timings(self) -> dict[str, Any] | None:
        """Per-phase wall-clock breakdown derived from ``phase_start``/``phase_end`` events.

        Pairs each ``phase_start`` with its matching ``phase_end`` (LIFO within a
        phase value) and sums the durations. Returns ``None`` when no phase
        events exist (backward compat for runs that predate issue #203).

        Each phase entry: ``{"wall_clock_seconds": float, "occurrences": int}``.
        Phases with the same :class:`DaydreamPhase` value (e.g. the deep
        orchestrator's ``review`` and ``arbiter`` stages both emit
        ``DaydreamPhase.DEEP``) fold into one bucket; the per-event ``metadata``
        in ``extra["phase_events"]`` carries the stage breakdown for finer
        analysis.

        Returns:
            Mapping of phase value → timing summary, or ``None`` when there are
            no phase events.
        """
        if not self._phase_events:
            return None
        by_phase: dict[str, dict[str, Any]] = {}
        pending_starts: dict[str, list[str]] = {}
        for ev in self._phase_events:
            key = ev.phase.value
            if ev.event == "phase_start":
                pending_starts.setdefault(key, []).append(ev.timestamp)
                by_phase.setdefault(key, {"wall_clock_seconds": 0.0, "occurrences": 0})
            elif ev.event == "phase_end":
                starts = pending_starts.get(key)
                if not starts:
                    continue  # orphaned end with no matching start
                bucket = by_phase[key]
                start_ts = starts.pop()
                try:
                    start = parse_iso_timestamp(start_ts)
                    end = parse_iso_timestamp(ev.timestamp)
                except ValueError:
                    continue  # unparseable timestamp; skip this pair
                duration = (end - start).total_seconds()
                bucket["wall_clock_seconds"] += max(0.0, duration)
                bucket["occurrences"] += 1
        return {
            key: {
                "wall_clock_seconds": round(val["wall_clock_seconds"], 3),
                "occurrences": val["occurrences"],
            }
            for key, val in by_phase.items()
            if val["occurrences"] > 0  # drop orphaned starts (start with no matching end)
        }

    def _sibling_path_for(self, descriptor: str) -> Path:
        """Return the sibling trajectory file path for *descriptor*.

        Layout: ``<target>/.daydream/runs/<session_id>/trajectories/<slug>.json``.
        Sibling files live under the same per-run directory as the parent
        trajectory, so every fork in the run dir belongs to this run by
        construction (no prefix filtering required).
        """
        slug = _safe_descriptor(descriptor)
        return (
            self.target_dir
            / _DAYDREAM_DIRNAME
            / _RUNS_SUBDIR
            / self.session_id
            / _TRAJECTORIES_SUBDIR
            / f"{slug}.json"
        )

    def fork(self, descriptor: str) -> "_ForkCM":
        """Create a child recorder for a parallel task group.

        Args:
            descriptor: Semantic label for the sibling (e.g. ``"fix-0"``).
        """
        return _ForkCM(parent=self, descriptor=descriptor)

    def _register_sibling(self, path: Path, descriptor: str) -> None:
        """Register a completed sibling trajectory (synchronous, no await)."""
        self._registered_siblings.append((path, descriptor))

    def create_dispatch_step(self, *, phase: DaydreamPhase) -> None:
        """Create an agent Step referencing all registered sibling trajectories.

        No-op when ``_registered_siblings`` is empty.
        """
        if not self._registered_siblings:
            return
        results: list[ObservationResult] = []
        for sibling_path, desc in self._registered_siblings:
            try:
                rel = str(sibling_path.relative_to(self.target_dir / ".daydream"))
            except ValueError:
                rel = sibling_path.name
            # trajectory_id is the sibling's canonical per-document id (mirrors
            # the fork's build_trajectory: session_id qualified by descriptor).
            # v1.7 makes it the resolution key for the ref; session_id stays as
            # informational run identity only (shared across siblings, not a
            # matching key), and trajectory_path remains the external file ref.
            results.append(
                ObservationResult(
                    content=f"Dispatched to {desc}",
                    subagent_trajectory_ref=[
                        SubagentTrajectoryRef(
                            trajectory_id=f"{self.session_id}:{desc}",
                            session_id=self.session_id,
                            trajectory_path=rel,
                        ),
                    ],
                )
            )
        count = len(self._registered_siblings)
        step = Step(
            step_id=self._next_step_id(),
            timestamp=now_iso(),
            source="agent",
            model_name=self.agent_model_name,
            message=f"Dispatching {count} parallel {phase.value} tasks",
            observation=Observation(results=results),
            # Deterministic (non-LLM) fan-out dispatch: no inference is made
            # here, so per the ATIF v1.7 no-LLM-orchestration rule this step
            # carries llm_call_count=0 and omits metrics / reasoning_content.
            llm_call_count=0,
            extra={
                "daydream_phase": phase.value,
                "daydream_run_flow": self.run_flow.value,
            },
        )
        self.steps.append(self.redactor.redact_step(step))
        self._registered_siblings.clear()

    def build_trajectory(self, steps: list[Step] | None = None) -> Trajectory:
        if steps is None:
            steps = self.steps
        version = daydream.__version__
        final_metrics_extra: dict[str, Any] | None = None
        if self._folded_fork_totals:
            # Token and cost totals include successful fork trajectories, but
            # total_steps remains scoped to this document's own step list.
            # Cached tokens remain a subset of prompt tokens, not an addition.
            final_metrics_extra = {
                "daydream_metric_scope": "whole_run_including_forks",
                "total_steps_scope": "local_trajectory",
            }

        final_metrics = FinalMetrics(
            total_prompt_tokens=self._final_totals["prompt"] or None,
            total_completion_tokens=self._final_totals["completion"] or None,
            total_cached_tokens=self._final_totals["cached"] or None,
            total_cost_usd=(
                self._final_totals["cost"] if self._final_totals["any_cost_seen"] else None
            ),
            total_steps=len(steps),
            extra=final_metrics_extra,
        )
        extra: dict[str, Any] = {"target_dir": str(self.target_dir)}
        if self.backend_name:
            # Backend identity mirrors archive/manifest.py's record: a
            # representative ``backend`` (resolved via the phase that governs the
            # deep flow's review fan-out) plus per-phase keys, each serialized
            # only when set — a flow that never runs a phase (improve never runs
            # fix/test) omits that phase's key entirely. Empty ``backend_name``
            # (direct construction outside the factory) serializes no backend keys.
            extra["backend"] = self.backend_name
            if self.review_backend_name:
                extra["review_backend"] = self.review_backend_name
            if self.fix_backend_name:
                extra["fix_backend"] = self.fix_backend_name
            if self.test_backend_name:
                extra["test_backend"] = self.test_backend_name
        if self.pr_number is not None:
            extra["pr_number"] = self.pr_number
        if self.pr_repo is not None:
            extra["pr_repo"] = self.pr_repo
        if self._profile is not None:
            extra.update(self._profile)
        if self._phase_events:
            extra["phase_events"] = [e.to_dict() for e in self._phase_events]
        if self._subtrajectories:
            extra["subtrajectories"] = [dict(s) for s in self._subtrajectories]
        # trajectory_id is the per-document identifier (distinct from the
        # run-scoped session_id): the root uses session_id directly; a fork
        # qualifies it with its descriptor so sibling documents stay unique
        # within the run (ATIF v1.7).
        trajectory_id = self.session_id if not self.descriptor else f"{self.session_id}:{self.descriptor}"
        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=self.session_id,
            trajectory_id=trajectory_id,
            agent=Agent(name="daydream", version=version, model_name=self.agent_model_name),
            steps=list(steps),
            final_metrics=final_metrics,
            extra=extra,
        )

    def _write(self) -> None:
        # Empty trajectory: skip — Pydantic Trajectory.steps has min_length=1.
        # Phase 4 may revisit if empty runs need a stub file on disk.
        if not self.steps:
            return
        trajectory = self.build_trajectory()
        trajectory_dict = trajectory.to_json_dict()
        if self._aborted:
            extra = trajectory_dict.setdefault("extra", {})
            extra["partial"] = True
        atomic_write_json(self.path, trajectory_dict)
        if self.on_write is not None:
            try:
                self.on_write(self, "complete")
            except Exception:  # noqa: BLE001 - archive failure must never affect the run
                pass

    def _snapshot_in_flight_steps(self) -> list[Step]:
        """Concatenate flushed steps with steps from any active invocations.

        Mid-``run_agent()``, an Invocation has accumulated steps that haven't
        been flushed back to ``self.steps`` (the flush happens in
        ``_InvocationCM.__aexit__`` via ``Invocation.finish``). For a partial
        flush we want those in-flight steps too; this helper concatenates
        them in ``step_id`` order without mutating either buffer — an active
        invocation's unflushed steps can predate an already-flushed one (see
        :meth:`_extend_steps`), and the partial file has to satisfy the same
        sequential-ids check as the final write.
        """
        if not self._active_invocations:
            return list(self.steps)
        snapshot = list(self.steps)
        next_id = self._step_id_counter + 1
        for inv in self._active_invocations:
            snapshot.extend(inv.snapshot_steps(snapshot_step_id=next_id))
            if inv._open_step_dict is not None:
                next_id += 1
        snapshot.sort(key=lambda s: s.step_id)
        return snapshot

    def write_partial(self) -> None:
        """SIGINT/SIGTERM flush path — write in-flight steps to ``<path>.partial``.

        Per D-07 the partial trajectory lives at a sibling path with the
        ``.partial`` suffix appended to the full filename (e.g.
        ``trajectory.json.partial``). The Trajectory's ``extra`` dict carries
        ``partial=true`` so consumers can detect incomplete runs without
        path-string parsing. Steps from any in-flight Invocation are
        included so SIGINT mid-``run_agent()`` does not lose work; empty
        trajectories are skipped (matches ``_write``).

        Idempotent: callable from a signal handler synchronously without
        awaiting ``__aexit__``; safe to invoke from outside the async context.
        Disk-write failures degrade with a warning per D-11 — partial flush
        must never crash shutdown.
        """
        snapshot_steps = self._snapshot_in_flight_steps()
        if not snapshot_steps:
            return
        try:
            trajectory = self.build_trajectory(steps=snapshot_steps)
            partial_path = self.path.with_suffix(self.path.suffix + ".partial")
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            json_dict = trajectory.to_json_dict()
            extra = json_dict.setdefault("extra", {})
            extra["partial"] = True
            partial_path.write_text(json.dumps(json_dict, indent=2), encoding="utf-8")
            if self.on_write is not None:
                try:
                    self.on_write(self, "partial")
                except Exception:  # noqa: BLE001 - archive failure must never crash shutdown
                    pass
            if self.parent is not None:
                self.parent.write_partial()
        except Exception as exc:  # noqa: BLE001 - partial flush must never crash shutdown
            print_warning(
                _console, f"Partial trajectory write failed: {type(exc).__name__}: {exc}"
            )


class _ForkCM:
    """Async context manager for forking a child recorder (D-01, D-02, D-03)."""

    def __init__(self, parent: TrajectoryRecorder, descriptor: str) -> None:
        self._parent = parent
        self._descriptor = descriptor
        self._child: TrajectoryRecorder | None = None
        self._entered_at: str | None = None
        self._exited_at: str | None = None

    async def __aenter__(self) -> TrajectoryRecorder:
        child = TrajectoryRecorder(
            path=self._parent._sibling_path_for(self._descriptor),
            run_flow=self._parent.run_flow,
            target_dir=self._parent.target_dir,
            agent_model_name=self._parent.agent_model_name,
            redactor=self._parent.redactor,
            session_id=self._parent.session_id,
            pr_number=self._parent.pr_number,
            pr_repo=self._parent.pr_repo,
            backend_name=self._parent.backend_name,
            review_backend_name=self._parent.review_backend_name,
            fix_backend_name=self._parent.fix_backend_name,
            test_backend_name=self._parent.test_backend_name,
        )
        child.parent = self._parent
        child.descriptor = self._descriptor
        child._previous_token = _RECORDER_VAR.set(child)
        _ACTIVE_RECORDERS.append(child)
        self._child = child
        self._entered_at = now_iso()
        return child

    async def __aexit__(self, exc_type: Any, exc_val: Any, _exc_tb: Any) -> None:
        child = self._child
        if child is None:
            return
        if exc_type is not None:
            child._aborted = True
        write_ok = False
        try:
            child._write()
            write_ok = bool(child.steps)
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run
            print_warning(_console, f"Sibling trajectory write failed: {type(exc).__name__}: {exc}")
        finally:
            if child._previous_token is not None:
                _RECORDER_VAR.reset(child._previous_token)
                child._previous_token = None
        try:
            _ACTIVE_RECORDERS.remove(child)
        except ValueError:
            pass
        self._exited_at = now_iso()
        if write_ok and child.parent is not None:
            # The root trajectory's final_metrics is whole-run truth: fold the
            # fork's totals in so manifest/eval consumers read one number
            # instead of re-summing sibling files. The fork file keeps its own
            # share. A failed child write folds nothing (the error already
            # degrades the record, D-11).
            child.parent._accumulate_metrics(
                prompt_tokens=child._final_totals["prompt"],
                completion_tokens=child._final_totals["completion"],
                cached_tokens=child._final_totals["cached"],
                cost_usd=child._final_totals["cost"] if child._final_totals["any_cost_seen"] else None,
            )
            child.parent._folded_fork_totals = True
            child.parent._register_sibling(child.path, self._descriptor)
            try:
                sibling_ref = str(child.path.relative_to(child.parent.target_dir / ".daydream"))
            except ValueError:
                sibling_ref = child.path.name
            phase = DaydreamPhase.FIX.value
            for step in child.steps:
                if step.extra is None:
                    continue
                candidate = step.extra.get("daydream_phase")
                if isinstance(candidate, str):
                    phase = candidate
                    break
            child.parent._register_fork_subtrajectory(
                phase=phase,
                descriptor=self._descriptor,
                started_at=self._entered_at or now_iso(),
                ended_at=self._exited_at or now_iso(),
                sibling_trajectory_ref=sibling_ref,
            )


class _InvocationCM:
    """Async context manager wrapping an Invocation (internal helper)."""

    def __init__(self, recorder: TrajectoryRecorder, phase: DaydreamPhase) -> None:
        self._recorder = recorder
        self._phase = phase
        self._invocation: Invocation | None = None

    async def __aenter__(self) -> Invocation:
        self._invocation = Invocation(recorder=self._recorder, phase=self._phase)
        self._invocation.started_at = now_iso()
        # Register with recorder so write_partial can capture in-flight steps
        # if SIGINT fires mid-invocation.
        self._recorder._active_invocations.append(self._invocation)
        return self._invocation

    async def __aexit__(self, exc_type: Any, exc_val: Any, _exc_tb: Any) -> None:
        if self._invocation is not None:
            # A fatal error propagating out of run_agent's event loop (e.g. a
            # backend raising MaxTurnsError on error_max_turns) would otherwise
            # vanish from the archive. Stamp it onto the closing Step BEFORE
            # finish() flushes — never swallow the exception, and never let a
            # recording failure mask it.
            if isinstance(exc_val, Exception):
                try:
                    subtype = getattr(exc_val, "subtype", None) or type(exc_val).__name__
                    self._invocation.mark_errored(str(subtype))
                except Exception:  # noqa: BLE001 - recording must never crash a run
                    pass
            try:
                self._invocation.finish()
                self._invocation.ended_at = now_iso()
                self._recorder._register_subtrajectory(self._invocation)
            finally:
                with suppress(ValueError):
                    self._recorder._active_invocations.remove(self._invocation)
                self._invocation = None
