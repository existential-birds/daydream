"""Validate review-thread query fields against a checked-in GitHub GraphQL subset.

Fields are checked on their enclosing type, never a global union. Import and
reconciliation queries share this contract through the fake gh handler and
schema tests; the subset deliberately excludes unused GitHub API surfaces.
"""

from __future__ import annotations

import re

# resolvedBy is valid but unused; keeping it distinguishes unused from invented fields.
SCHEMA_FIELDS: dict[str, set[str]] = {"PullRequestReviewThread": {
        "id", "isResolved", "isOutdated", "resolvedBy", "subjectType", "path", "line", "originalLine",
        "originalStartLine", "diffSide", "startDiffSide", "comments",
    }, "Comment": {"id", "databaseId", "body", "author", "isMinimized", "createdAt", "updatedAt", "url", "replyTo",
        "viewerDidAuthor",
    }, "Actor": {"login"},
}

# Nested selections carry the selected value's type into field validation.
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
    "author": "Actor", "replyTo": "Comment",
    # Connections
    "pageInfo": "PageInfo",
}

# Unlike named fields, nodes resolves from its enclosing connection type.
_CONNECTION_NODE_TYPE: dict[str, str] = {
    "PullRequestReviewThreadConnection": "PullRequestReviewThread", "PullRequestReviewCommentConnection": "Comment",
}

# Routing fields are accepted in every context; __typename is never collected.
_MACHINERY_FIELDS = {
    "repository", "pullRequest", "reviewThreads", "node", "nodes", "pageInfo", "hasNextPage", "endCursor",
}

_SKIP_CHARS = " \t\r\n,?$"


def _requested_fields(query: str) -> dict[str | None, set[str]]:
    """Collect selected field names by enclosing GraphQL type.

    Resolve aliases and connection nodes, switch types for inline fragments,
    and skip arguments, named fragment spreads, and __typename. Unmodeled types
    retain their names so the caller permits only query machinery there.
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

    def record_field(tok: str, i: int, type_ctx: str | None) -> tuple[str, int]:
        """Record the real field name and return the cursor after aliases and arguments."""
        real = tok
        j = skip_ws(i)
        # strip an alias: "side: diffSide" validates "diffSide"
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
        return real, j

    # Push the enclosing type for each nested selection; closing braces restore it.
    pending_types: list[str | None] = []
    type_ctx: str | None = None
    while True:
        i = skip_ws(i)
        if i >= n:
            break
        c = query[i]
        if c == "}":
            i += 1
            if pending_types:
                type_ctx = pending_types.pop()
            continue
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
                    pending_types.append(type_ctx)
                    type_ctx = ftype
                    i += 1
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
                pending_types.append(type_ctx)
                type_ctx = None  # root operation type is unmodeled
                i += 1
            continue
        real, i = record_field(tok, i, type_ctx)
        if i < n and query[i] == "{":
            if real == "nodes":
                nested = _CONNECTION_NODE_TYPE.get(type_ctx or "")
            else:
                nested = _NESTED_SELECTION_TYPE.get(real)
            pending_types.append(type_ctx)
            type_ctx = nested
            i += 1
    return out


def unknown_query_fields(query: str) -> set[str]:
    """Return fields absent from their enclosing type's subset or shared query machinery."""
    unknown: set[str] = set()
    for type_ctx, fields in _requested_fields(query).items():
        valid: set[str] = _MACHINERY_FIELDS | (SCHEMA_FIELDS.get(type_ctx or "") or set())
        unknown |= fields - valid
    return unknown
