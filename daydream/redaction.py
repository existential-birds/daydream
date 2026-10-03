"""Shared fail-closed redaction for logs, backend events, and stored artifacts.

Flat credential patterns, structured text, and native containers share this
implementation. This module does not depend on the trajectory recorder.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, Callable

from daydream.credential_patterns import (
    _QUERY_CREDENTIAL_PATTERN,
    _SCP_USERINFO_PATTERN,
    _TOKEN_ONLY_USERINFO_PATTERN,
    _URL_CREDENTIAL_PATTERN,
)

# Stages: auth headers/schemes, flat rules, then structured key/value pairs.
# Headers consume shaped credentials whole; URL and env rules precede API keys
# to avoid double marking and preserve names. PEM precedes env rules so no key
# body survives. Structured rules leave existing redaction markers intact.
_API_KEY_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{6,}|ghp_[A-Za-z0-9]{6,}|ghs_[A-Za-z0-9]{6,}|xoxb-[A-Za-z0-9\-]{6,}|AKIA[A-Z0-9]{16})\b"
)


_JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\b")


_USERNAME_PATH_PATTERN = re.compile(r"(/Users/|/home/|[A-Z]:\\Users\\)([^/\\\s]+)")


# Private PEM blocks collapse before flat API-key matching; public certificates
# stay untouched. Benchmark buffering imports this shared header pattern.
_PEM_HEADER = r"(?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY"


_PEM_KEY_PATTERN = re.compile(
    rf"-----BEGIN {_PEM_HEADER}-----" rf".*?-----END {_PEM_HEADER}-----",
    re.DOTALL,
)


#: Shared PEM marker used by benchmark buffering anchors.
_PEM_KEY_REDACTED_MARKER = "[REDACTED_PEM_KEY]"


#: Distinguish sensitive-key redaction from flat credential-pattern matches.
_REDACTED_CREDENTIAL = "[REDACTED_CREDENTIAL]"


#: An already-redacted value survives every later pass: structured rules replace a
#: value only when no marker is present, so re-redaction cannot swap one host
#: marker for another. Both spellings count — this module's bracketed form and the
#: angle-bracket form Improve's plan renderer emits.
_REDACTION_MARKER = re.compile(r"\[REDACTED|<redacted>", re.IGNORECASE)


# Match complete secret-name segments, preserving MONKEY_PATCH, KEYBOARD_LAYOUT,
# AUTHOR and TOKENIZED. Horizontal-only whitespace keeps an empty assignment
# from consuming the following line as its value.
_ENV_VAR_PATTERN = re.compile(
    r"\b((?:[A-Z][A-Z0-9]*_)*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|API_?KEY|APIKEY|AUTH)(?:_[A-Z0-9]+)*)[^\S\n\r]*=[^\S\n\r]*([^\s\n\r;]+)"  # noqa: E501 - secret-segment alternation
)


def _redact_url_userinfo(match: re.Match[str]) -> str:
    """Preserve user/password marker shape when the wider token rule overlaps."""
    userinfo = match.group(0)[len(match.group(1)) : -1]
    masked = "[REDACTED_USER]:[REDACTED_API_KEY]" if ":" in userinfo else "[REDACTED_USER]"
    return f"{match.group(1)}{masked}@"


_REDACTION_RULES: tuple[tuple[Any, str | Callable[[re.Match[str]], str]], ...] = (
    (_URL_CREDENTIAL_PATTERN, r"\1[REDACTED_USER]:[REDACTED_API_KEY]@"),
    (_TOKEN_ONLY_USERINFO_PATTERN, _redact_url_userinfo),
    (_SCP_USERINFO_PATTERN, r"[REDACTED_USER]:[REDACTED_API_KEY]@\2"),
    (_QUERY_CREDENTIAL_PATTERN, r"\1\2=[REDACTED_CREDENTIAL]"),
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
    """Return a fresh recursively redacted value; never mutate the input.

    String leaves under sensitive ancestor keys become credential markers; other
    strings use structured redaction. String keys use flat redaction. Other leaf
    types pass through. Failures produce a fixed redaction-failed marker."""
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


# Sensitive keys match normalized full keys, compound suffixes, or segments.
# Bare "key" and substring matches are excluded so keyStore, tokenizer and
# passwordless remain untouched.
_SENSITIVE_KEY_SUFFIXES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
        "secret_access_key",
        "set_cookie",
        "token",
    }
)


#: Chars matching the camelCase lookbehind ``[a-z0-9]`` (lowercase + digits).
_LOWER_OR_DIGIT_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")


#: Bound the strict suffix walk: longer normalized keys cannot equal a member.
_MAX_SENSITIVE_KEY_LEN = max(len(member) for member in _SENSITIVE_KEY_SUFFIXES)


# Normalize lowercase/digit-to-uppercase boundaries: apiKey → api_Key.
_CAMEL_CASE_BOUNDARY_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


# Normalize separator runs: client-secret → client_secret.
_NON_ALPHANUMERIC_KEY_PATTERN = re.compile(r"[^A-Za-z0-9]+")


def _normalize_sensitive_key(key: str) -> str:
    """Normalize camelCase and separator-delimited keys to lowercase snake_case."""
    return _NON_ALPHANUMERIC_KEY_PATTERN.sub("_", _CAMEL_CASE_BOUNDARY_PATTERN.sub("_", key)).lower().strip("_")


def _is_sensitive_key(key: str) -> bool:
    """Match sensitive normalized keys, suffixes, or complete segments.

    Substrings do not count: tokenizer, passwordless, monkeyPatch, keyStore, and
    max_tokens remain nonsensitive."""
    normalized = _normalize_sensitive_key(key)
    if normalized in _SENSITIVE_KEY_SUFFIXES:
        return True
    if any(normalized.endswith(f"_{member}") for member in _SENSITIVE_KEY_SUFFIXES):
        return True
    return any(segment in _SENSITIVE_KEY_SUFFIXES for segment in normalized.split("_"))


# Match headers case-insensitively at line start or embedded in output.
# Exclude word interiors and quoted keys; consume one token plus an optional
# auth scheme without crossing a line. The trailing boundary excludes prose
# such as "The authorization: feature is enabled now". Run before flat rules
# so shaped credentials are consumed whole without double marking.
_AUTHORIZATION_HEADER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_\"'])(Authorization|Proxy-Authorization|X-Api-Key|X-Auth-Token|Cookie|Set-Cookie):"
    r"[^\S\n\r]*(?:(Basic|Bearer|Token)[^\S\n\r]+)?([^\s,;\"']+)"
    r"(?=[^\S\n\r]*[,}\]\r\n]|$)",
    re.IGNORECASE,
)


# JSON, Python-repr, YAML and assignment pairs accept quoted keys and values,
# escaped quotes, auth-scheme pairs, or bare tokens. Bare values exclude comma,
# colon and equals separators so replacement preserves surrounding structure.
# Skip opening braces to reach nested keys; handle blocks in the later block
# pass. The lookahead protects redaction markers emitted by earlier stages.
_STRUCTURED_KEY_VALUE_PATTERN = re.compile(
    r"(['\"]?)([A-Za-z_][A-Za-z0-9_.\-]*)\1([^\S\n\r]*[:=][^\S\n\r]*)"
    r"(?:\"((?:\\.|[^\"\\])*)\"|'((?:\\.|[^'\\])*)'|(Basic|Bearer|Token)[^\S\n\r]+([^\s,;\"']+)|([^\s\"'\[\]{}():,=]++))"
    r"(?![^\S\n\r]*(?:\[REDACTED|<redacted>))",
)


def _redact_structured_key_value(match: re.Match[str], text: str) -> str:
    """Replace a sensitive pair’s value while preserving its key and quote wrapper.

    Bare colon values require a structural boundary; bare assignments always redact.
    Empty values and existing redaction markers remain unchanged."""
    key = match.group(2)
    if not _is_sensitive_key(key):
        return match.group(0)
    wrapped_key = f"{match.group(1)}{key}{match.group(1)}"
    sep = match.group(3)
    if match.group(6) is not None:
        # Bearer|Basic|Token <opaque>: keep the scheme, replace the token.
        token = match.group(7)
        if token is None or token == "" or _REDACTION_MARKER.search(token):
            return match.group(0)
        return f"{wrapped_key}{sep}{match.group(6)} {_REDACTED_CREDENTIAL}"
    value = match.group(4) or match.group(5) or match.group(8)
    if value is None or value == "" or _REDACTION_MARKER.search(value):
        return match.group(0)
    if match.group(8) is not None and not _bare_value_redactable(sep, text, match.end()):
        return match.group(0)
    quote = '"' if (match.group(4) is not None or match.group(8) is not None) else "'"
    return f"{wrapped_key}{sep}{quote}{_REDACTED_CREDENTIAL}{quote}"


def _bare_value_redactable(sep: str, text: str, end: int) -> bool:
    """Redact bare assignments; colon values require end-of-text/line or , } ].

    This preserves prose such as "the token: is now available"."""
    if "=" in sep:
        return True
    return not text[end:] or re.match(r"[^\S\n\r]*[,}\]\r\n]", text[end:]) is not None


def _redact_structured_key_values(text: str) -> str:
    """Redact sensitive free-text pairs, then consume multiline/structured blocks.

    Empty values and existing markers are skipped. Any failure returns a fixed
    redaction-failed marker, never the original text."""
    try:
        return _redact_structured_blocks(_redact_structured_pairs(text))
    except Exception:  # noqa: BLE001 - fail closed at every host boundary
        return "[REDACTION_FAILED]"


#: Maximal key run before a separator, matching [A-Za-z0-9_.\-].
_STRUCTURED_KEY_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
)


#: Valid key starts exclude digits: 1apiKey starts at "a".
_STRUCTURED_KEY_START_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
)


#: Key-quote wrapper characters (``['\"]?``).
_QUOTE_CHARS = frozenset("'\"")


def _match_start_before_separator(text: str, sep: int) -> int:
    """Find one leftmost key start by walking backward from the separator.

    Quoted keys anchor at their opening quote. For unquoted runs, skip leading
    characters that cannot start a key (e.g. digits in 123secret). Anchoring once
    per separator avoids quadratic retries across long key-shaped runs."""
    i = sep
    while i and text[i - 1].isspace() and text[i - 1] not in "\n\r":
        i -= 1
    if i and text[i - 1] in _QUOTE_CHARS:
        quote = text[i - 1]
        j = i - 1
        while j and text[j - 1] in _STRUCTURED_KEY_CHARS:
            j -= 1
        if j and text[j - 1] == quote:
            return j - 1
        # Impossible quote pairing: the char before the key run becomes a
        # boundary and the unquoted form starts at the run's first key char.
        return j
    run_end = i
    while i and text[i - 1] in _STRUCTURED_KEY_CHARS:
        i -= 1
    while i < run_end and text[i] not in _STRUCTURED_KEY_START_CHARS:
        i += 1
    return i


def _nearest_separator(text: str, pos: int) -> int:
    """Find the next : or =; return -1 without retrying separator-less key runs."""
    colon = text.find(":", pos)
    equals = text.find("=", pos)
    if colon < 0 and equals < 0:
        return -1
    if colon < 0:
        return equals
    if equals < 0:
        return colon
    return colon if colon < equals else equals


def _value_start(match: re.Match[str]) -> int:
    """Resume at a non-sensitive pair’s value, preserving the key/separator prefix.

    For quoted values, resume at the opening quote. This permits nested sensitive
    pairs and quoted-key reanchoring without rescanning a long value per character."""
    for group in (4, 5, 7, 8):
        index = match.start(group)
        if index >= 0:
            return index - 1 if group in (4, 5) else index
    return match.end()


def _strict_suffix_is_sensitive(text: str, s2: int, key_end: int) -> bool:
    """Test a strict suffix of a nonsensitive key in O(max sensitive-member length).

    A sensitive normalized suffix or later segment would already make the whole
    key sensitive. Only exact-match and first-segment rules remain. Inspect a
    bounded prefix using the same camelCase/separator normalization, keeping the
    scan linear across a long key’s candidates."""
    norm: list[str] = []
    first_seg: str | None = None
    pending_sep = False
    i = s2
    while i < key_end:
        c = text[i]
        if "A" <= c <= "Z" or c in _LOWER_OR_DIGIT_CHARS:
            # A camelCase boundary or a collapsed non-alphanumeric run closes
            # the current segment. The leading separator is stripped by the
            # normalization's edge rule (never emitted); the first separator
            # to survive records the FIRST segment, on which the segment rule
            # can fire. Later separators only shape the whole-suffix string.
            camel_boundary = (
                "A" <= c <= "Z"
                and i > s2
                and text[i - 1] in _LOWER_OR_DIGIT_CHARS
            )
            if camel_boundary or pending_sep:
                if norm and first_seg is None:
                    first_seg = "".join(norm)
                    if first_seg in _SENSITIVE_KEY_SUFFIXES:
                        return True
                if norm:
                    norm.append("_")
                pending_sep = False
            norm.append(c.lower() if "A" <= c <= "Z" else c)
            if len(norm) > _MAX_SENSITIVE_KEY_LEN:
                # The whole normalized suffix is now longer than every member,
                # so the exact-match rule is dead; the first segment was
                # either never closed (it is at least this long) or already
                # checked above.
                return False
        else:
            # A run of non-alphanumeric key chars (``_``, ``.``, ``-``)
            # collapses to a single ``_`` separator — emitted as pending so a
            # trailing run is dropped like the normalization's edge strip.
            pending_sep = True
        i += 1
    return "".join(norm) in _SENSITIVE_KEY_SUFFIXES


def _sensitive_suffix_matches(
    text: str,
    match: re.Match[str],
    pattern: re.Pattern[str],
) -> Iterator[re.Match[str]]:
    """Yield sensitive suffixes of a nonsensitive anchored key, leftmost first.

    Examples include defauthorization and fooapi_key. Test each candidate in
    bounded time and rematch only sensitive candidates, preserving linear scanning."""
    s2 = match.start(2) + 1
    key_end = match.end(2)
    while s2 < key_end:
        if text[s2] in _STRUCTURED_KEY_START_CHARS:
            if _strict_suffix_is_sensitive(text, s2, key_end):
                m2 = pattern.match(text, s2)
                if m2 is not None:
                    yield m2
            if text[s2] not in _LOWER_OR_DIGIT_CHARS and not (
                "A" <= text[s2] <= "Z"
            ):
                # Separator-run candidate: every position inside one
                # contiguous run of non-alphanumeric key chars normalizes to
                # the SAME suffix (leading separators collapse and the edge is
                # stripped), so the run needs at most one evaluation — jump
                # over the rest instead of walking it per position, which
                # re-enabled the O(n^2) hang on separator-heavy key runs
                # (issue #1236).
                while s2 + 1 < key_end and not (
                    "A" <= text[s2 + 1] <= "Z"
                    or text[s2 + 1] in _LOWER_OR_DIGIT_CHARS
                ):
                    s2 += 1
        s2 += 1


def _redact_structured_pairs(text: str) -> str:
    """Redact line-scoped pairs and resume inside each nonsensitive value.

    This catches nested pairs such as "config: token=opaque", which re.sub would
    skip. Separator anchoring avoids per-character retries on long keys or values."""
    out: list[str] = []
    pos = 0
    while True:
        sep = _nearest_separator(text, pos)
        if sep < 0:
            break
        start = _match_start_before_separator(text, sep)
        if start < pos:
            # The derived anchor lands behind the emission frontier — the
            # bytes [start, pos) were already emitted (or already consumed by
            # an earlier redaction), so this separator anchors no NEW match.
            out.append(text[pos : sep + 1])
            pos = sep + 1
            continue
        match = _STRUCTURED_KEY_VALUE_PATTERN.match(text, start)
        if match is None:
            # No pair can consume this separator: the value would fail
            # identically from every candidate start, so skip the separator
            # verbatim and re-anchor after it.
            out.append(text[pos : sep + 1])
            pos = sep + 1
            continue
        if _is_sensitive_key(match.group(2)):
            out.append(text[pos : match.start()])
            out.append(_redact_structured_key_value(match, text))
            pos = match.end()
        else:
            # A non-sensitive anchored key may still hold a sensitive suffix
            # (see _sensitive_suffix_matches). The old engine's one-character
            # advance re-matched every suffix at the SAME separator and
            # redacted the first sensitive one, so take the leftmost match and
            # stop.
            suffix = next(
                _sensitive_suffix_matches(text, match, _STRUCTURED_KEY_VALUE_PATTERN),
                None,
            )
            if suffix is not None:
                out.append(text[pos : suffix.start()])
                out.append(_redact_structured_key_value(suffix, text))
                pos = suffix.end()
            else:
                value_start = _value_start(match)
                out.append(text[pos:value_start])
                pos = value_start
    out.append(text[pos:])
    return "".join(out)


_BLOCK_VALUE_PATTERN = re.compile(
    r"(['\"]?)([A-Za-z_][A-Za-z0-9_.\-]*)\1([^\S\n\r]*[:=][^\S\n\r]*)"
    r"(?=[\[{](?!REDACTED)|\n)"
)


def _redact_structured_blocks(text: str) -> str:
    """Replace sensitive YAML/JSON blocks wholesale with a quoted credential marker.

    The separator-anchored scan resumes after nonsensitive or empty matches because
    the block value follows them. Exceptions fail closed."""
    try:
        out: list[str] = []
        pos = 0
        while True:
            sep = _nearest_separator(text, pos)
            if sep < 0:
                break
            start = _match_start_before_separator(text, sep)
            if start < pos:
                # Same behind-the-frontier guard as the pair scan: this
                # separator cannot anchor a NEW block match.
                out.append(text[pos : sep + 1])
                pos = sep + 1
                continue
            match = _BLOCK_VALUE_PATTERN.match(text, start)
            if match is None:
                out.append(text[pos : sep + 1])
                pos = sep + 1
                continue
            key = match.group(2)
            if not _is_sensitive_key(key):
                # Sensitive-suffix mirror of the pair scan, but a sensitive
                # suffix whose block is EMPTY is not a redaction either way —
                # the old scan advanced one character past it and kept
                # looking, so the scan continues to later suffixes.
                found: tuple[re.Match[str], int] | None = None
                for m2 in _sensitive_suffix_matches(text, match, _BLOCK_VALUE_PATTERN):
                    block_end = _block_value_end(text, m2.end())
                    if block_end != m2.end():
                        found = (m2, block_end)
                        break
                if found is not None:
                    m2, block_end = found
                    out.append(text[pos : m2.start()])
                    quote = m2.group(1) or ""
                    out.append(f'{quote}{m2.group(2)}{quote}{m2.group(3)}"{_REDACTED_CREDENTIAL}"')
                    pos = block_end
                    continue
                # No value is consumed by a block match; keep the matched
                # key/separator verbatim and re-anchor after it.
                out.append(text[pos : match.end()])
                pos = match.end()
                continue
            block_end = _block_value_end(text, match.end())
            if block_end == match.end():
                # no indented block follows (empty value): skip and re-scan
                out.append(text[pos : match.end()])
                pos = match.end()
                continue
            out.append(text[pos : match.start()])
            quote = match.group(1) or ""
            out.append(f'{quote}{key}{quote}{match.group(3)}"{_REDACTED_CREDENTIAL}"')
            pos = block_end
        out.append(text[pos:])
        return "".join(out)
    except Exception:  # noqa: BLE001 - fail closed at every host boundary
        return "[REDACTION_FAILED]"


def _block_value_end(text: str, start: int) -> int:
    """Find the matching quote-aware brace/bracket close, or text end if unbalanced.

    YAML blocks consume following indented lines until the key’s indentation;
    return start unchanged when no line is indented."""
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
    """Preserve the auth header and optional Basic/Bearer/Token scheme; redact its token."""
    name = match.group(1)
    if match.group(2) is not None:
        return f"{name}: {match.group(2)} {_REDACTED_CREDENTIAL}"
    return f"{name}: {_REDACTED_CREDENTIAL}"


def redact_structured_text(s: str) -> str:
    """Apply auth-header/scheme rules, flat patterns, then structured-pair/block rules.

    Earlier markers are preserved, preventing partial or double redaction. Any
    stage failure replaces the whole field; raw text never passes through."""
    try:
        s = _AUTHORIZATION_HEADER_PATTERN.sub(_redact_header_value, s)
        s = redact_text(s)
        return _redact_structured_key_values(s)
    except Exception:  # noqa: BLE001 - fail closed at every host boundary
        return "[REDACTION_FAILED]"
