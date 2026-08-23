"""Minimal checked-in GitHub GraphQL field-set subset + query field extractor.

The review-thread GraphQL queries in :mod:`daydream.benchmark.github_import`
may only request fields GitHub's schema actually defines, *on the type they
are requested under*. This module is the enforcement point:
:func:`unknown_query_fields` parses a GraphQL query's selection set, binds
each requested name to the type it appears under, and returns the names that
are absent from that type's checked-in subset — a name valid on one type
requested on another is flagged exactly like an invented one. It is shared by
the contract test (``tests/test_github_schema.py``) and the fake ``gh``
graphql handler (``tests/harness/fake_gh.py``), so reintroducing an invented
field or misplacing a real one fails CI.

The subset is intentionally minimal — it covers exactly the fields the
review-thread queries that route through the fake ``gh`` touch (the import
queries in ``github_import`` and the prior-finding inventory query in
``reconcile``), per the issue-841 plan. A full GitHub introspection snapshot
is impractical to maintain; a small field-set map is the sanctioned contract
mechanism.
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
        "isMinimized",
        "createdAt",
        "updatedAt",
        "url",
        "replyTo",
        "viewerDidAuthor",
    },
    "Actor": {"login"},
}

# Field name -> GraphQL object type of the value it selects, for the few
# fields these queries use that carry a nested selection set. Together with
# SCHEMA_FIELDS this is what binds a requested name to exactly one type: a
# name is only acceptable on the type whose subset contains it.
_NESTED_SELECTION_TYPE: dict[str, str] = {
    # Query root
    "repository": "Repository",
    "node": "Node",  # interface; `... on Type` narrows the context
    # Repository
    "pullRequest": "PullRequest",
    # PullRequest
    "reviewThreads": "PullRequestReviewThreadConnection",
    # PullRequestReviewThread
    "comments": "PullRequestReviewCommentConnection",
    # Comment
    "author": "Actor",
    "replyTo": "Comment",
    # Connections
    "pageInfo": "PageInfo",
}

# The object type a connection's ``nodes`` selection resolves to, keyed by the
# enclosing connection type (context-dependent, so it cannot live in the
# single-name map above).
_CONNECTION_NODE_TYPE: dict[str, str] = {
    "PullRequestReviewThreadConnection": "PullRequestReviewThread",
    "PullRequestReviewCommentConnection": "Comment",
}

# Query/connection machinery every query relies on, plus ``__typename`` which
# the GraphQL spec defines on every type. Machinery routes to the next
# selection set, so it is acceptable in any type context; ``__typename`` is
# never collected at all.
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

_SKIP_CHARS = " \t\r\n,?$"


def _requested_fields(query: str) -> dict[str | None, set[str]]:
    """Recursive-descent over *query*'s selection sets; returns requested names
    grouped by the GraphQL type they were requested on.

    Each selection set is parsed in the type context of its enclosing parent
    field (``None`` when the subset does not model that type — the Query root,
    Repository, PullRequest, connections, PageInfo), so a name is never
    validated against a global union:

    - ``alias: realName`` validates ``realName`` (the alias is not a schema
      field, so only the real name is collected);
    - ``__typename`` is always valid and never collected;
    - ``(...)`` argument lists (including the query's variable declarations)
      are skipped;
    - ``... on Type`` inline fragments switch the type context; named fragment
      spreads are skipped;
    - ``nodes`` resolves to the enclosing connection's node type.
    """
    out: dict[str | None, set[str]] = {}
    i, n = 0, len(query)

    def skip_ws(i: int) -> int:
        while i < n and query[i] in _SKIP_CHARS:
            i += 1
        return i

    def read_name(i: int) -> tuple[str | None, int]:
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", query[i:])
        if not m:
            return None, i
        return m.group(0), i + len(m.group(0))

    def skip_args(i: int) -> int:
        if i < n and query[i] == "(":
            depth = 1
            i += 1
            while i < n and depth:
                if query[i] == "(":
                    depth += 1
                elif query[i] == ")":
                    depth -= 1
                i += 1
        return i

    def parse_selections(type_ctx: str | None) -> None:
        """Parse one selection set; returns at its closing ``}``."""
        nonlocal i
        while True:
            i = skip_ws(i)
            if i >= n:
                return
            c = query[i]
            if c == "}":
                i += 1
                return
            if c == "{":
                i += 1
                continue
            if c == ".":
                # '... on Type' switches the type context for its selection;
                # a named fragment spread is skipped.
                i = skip_ws(i + 3)
                tok, i = read_name(i)
                i = skip_ws(i)
                if tok == "on":
                    ftype, i = read_name(i)
                    i = skip_ws(i)
                    if i < n and query[i] == "{":
                        parse_selections(ftype)
                continue
            tok, i = read_name(i)
            if tok is None:
                i += 1
                continue
            if tok in ("query", "mutation"):
                # skip the operation name (if any) and variable declarations
                i = skip_ws(i)
                _, i = read_name(i)
                i = skip_ws(i)
                i = skip_args(i)
                i = skip_ws(i)
                if i < n and query[i] == "{":
                    parse_selections(None)  # root operation type is unmodeled
                continue
            # strip an alias: "side: diffSide" validates "diffSide"
            j = skip_ws(i)
            real = tok
            if j < n and query[j] == ":":
                j = skip_ws(j + 1)
                rname, j = read_name(j)
                if rname is not None:
                    real = rname
            if real != "__typename":
                out.setdefault(type_ctx, set()).add(real)
            j = skip_ws(j)
            j = skip_args(j)
            j = skip_ws(j)
            if j < n and query[j] == "{":
                if real == "nodes":
                    nested = _CONNECTION_NODE_TYPE.get(type_ctx or "")
                else:
                    nested = _NESTED_SELECTION_TYPE.get(real)
                i = j
                parse_selections(nested)
            else:
                i = j

    parse_selections(None)
    return out


def unknown_query_fields(query: str) -> set[str]:
    """Return the requested field names *query* asks for that its type context
    does not define: a name absent from the subset of the type it is requested
    under, or — under a type the subset does not model — any name that is not
    query machinery.

    ``__typename`` is always valid; an alias is validated by its real name.
    """
    unknown: set[str] = set()
    for type_ctx, fields in _requested_fields(query).items():
        valid: set[str] = _MACHINERY_FIELDS | (SCHEMA_FIELDS.get(type_ctx or "") or set())
        unknown |= fields - valid
    return unknown
