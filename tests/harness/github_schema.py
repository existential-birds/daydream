"""Minimal checked-in GitHub GraphQL field-set subset + query field extractor.

The review-thread GraphQL queries in :mod:`daydream.benchmark.github_import`
may only request fields GitHub's schema actually defines. This module is the
enforcement point: :func:`unknown_query_fields` parses a GraphQL query's
selection set and returns the requested field names that are absent from the
checked-in subset below. It is shared by the contract test
(``tests/test_github_schema.py``) and the fake ``gh`` graphql handler
(``tests/harness/fake_gh.py``), so reintroducing an invented field fails CI.

The subset is intentionally minimal — it covers exactly the types the two
review-thread queries touch, per the issue-841 plan. A full GitHub
introspection snapshot is impractical to maintain; a small field-set map is
the sanctioned contract mechanism.
"""

from __future__ import annotations

import re

# Real GitHub GraphQL field names for the types the two review-thread queries
# touch. ``resolvedBy`` is in the subset but must NOT be requested (nothing
# reads it); it is present so a future *correct* addition is not falsely
# rejected.
SCHEMA_FIELDS: dict[str, set[str]] = {
    "PullRequestReviewThread": {
        "id",
        "isResolved",
        "isOutdated",
        "resolvedBy",
        "subjectType",
        "path",
        "line",
        "originalLine",
        "diffSide",
        "startDiffSide",
        "comments",
    },
    "Comment": {
        "id",
        "databaseId",
        "body",
        "author",
        "createdAt",
        "updatedAt",
        "url",
        "replyTo",
    },
    "Actor": {"login"},
}

# Query/connection machinery every query relies on, plus ``__typename`` which
# the GraphQL spec defines on every type.
_MACHINERY_FIELDS = {
    "repository",
    "pullRequest",
    "reviewThreads",
    "node",
    "nodes",
    "pageInfo",
    "hasNextPage",
    "endCursor",
}

_VALID_FIELDS = {
    field
    for fields in SCHEMA_FIELDS.values()
    for field in fields
} | _MACHINERY_FIELDS

_SKIP_CHARS = " \t\r\n,?$"


def _requested_fields(query: str) -> set[str]:
    """Recursive-descent over *query*'s selection set; returns requested names.

    - ``alias: realName`` validates ``realName`` (the alias is not a schema
      field, so only the real name is collected);
    - ``__typename`` is always valid and never collected;
    - ``(...)`` argument lists (including the query's variable declarations)
      and ``... on Type`` inline fragments are skipped.
    """
    out: set[str] = set()
    i, n = 0, len(query)
    while i < n:
        c = query[i]
        if c in "{}":
            i += 1
            continue
        if c == "(":
            depth = 1
            i += 1
            while i < n and depth:
                if query[i] == "(":
                    depth += 1
                elif query[i] == ")":
                    depth -= 1
                i += 1
            continue
        if c in _SKIP_CHARS or c == ".":  # '.' = '...' fragment spread
            i += 1
            continue
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", query[i:])
        if not m:
            i += 1
            continue
        tok = m.group(0)
        j = i + len(tok)
        if tok in ("query", "mutation"):
            # skip the operation name (if any) following the keyword
            i = j
            while i < n and query[i] in " \t\r\n":
                i += 1
            m2 = re.match(r"[A-Za-z_][A-Za-z0-9_]*", query[i:])
            if m2:
                i += len(m2.group(0))
            continue
        if tok == "on":
            # '... on Type' — skip 'on' and the type name
            i = j
            while i < n and query[i] in " \t\r\n":
                i += 1
            m2 = re.match(r"[A-Za-z_][A-Za-z0-9_]*", query[i:])
            if m2:
                i += len(m2.group(0))
            continue
        # strip an alias: "side: diffSide" validates "diffSide"
        k = j
        while k < n and query[k] in " \t\r\n":
            k += 1
        if k < n and query[k] == ":":
            k += 1
            while k < n and query[k] in " \t\r\n":
                k += 1
            m2 = re.match(r"[A-Za-z_][A-Za-z0-9_]*", query[k:])
            if m2:
                real = m2.group(0)
                if real != "__typename":
                    out.add(real)
                i = k + len(real)
                continue
        if tok != "__typename":
            out.add(tok)
        i = j
    return out


def unknown_query_fields(query: str) -> set[str]:
    """Return the requested field names *query* asks for that the schema subset
    does not define.

    ``__typename`` is always valid; an alias is validated by its real name.
    """
    return _requested_fields(query) - _VALID_FIELDS
