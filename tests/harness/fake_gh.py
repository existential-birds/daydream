"""Fake ``gh`` executable harness for real-path ``post-findings`` tests.

Installs an executable Python shim named ``gh`` into a tmp dir prepended to
``PATH`` so the real ``git_ops._run_gh`` subprocess seam, the ``gh_api``
tempfile-``--input`` path, and JSON response parsing all run for real. Only
the GitHub network boundary (the ``gh`` binary itself) is faked.

The shim records every invocation (argv + parsed ``--input`` payload) to a
JSONL log the :class:`FakeGh` helper parses, and replies from a canned
response map (``responses.json``) plus built-in behaviors:

- ``gh api graphql`` with a ``reviewThreads`` query returns the configured
  prior-thread inventory (empty by default);
- ``gh api graphql`` with a ``minimizeComment`` mutation returns a minimized
  success (the Task 0 spike's chosen stale-finding mechanism);
- ``GET repos/<o>/<r>/pulls/<n>/reviews`` returns the configured review list
  (``[]`` by default);
- ``POST repos/<o>/<r>/pulls/<n>/reviews`` returns a fake ``html_url``.
- ``gh pr view [<number>] --json ...`` emits the configured ``pr-view`` JSON
  and records the invocation; ``gh pr list`` projects that same response into
  the one-row list used by ``find_open_pr``;
- ``gh repo view --json nameWithOwner -q .nameWithOwner`` emits the configured
  ``repo-view`` slug (or ``acme/widgets`` when ``pr-view`` is configured).

A canned ``gh api`` response is looked up by ``"<METHOD> <endpoint>"``, falling
back to the endpoint with its query string stripped. Under ``--jq`` the reply is
emitted as NDJSON (one ``@json``-encoded element per line), matching real ``gh``
so :func:`daydream.git_ops.gh_api`'s paginate+jq contract runs for real.

Any other invocation exits non-zero so unexpected calls surface as failures.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daydream.pr_review import finding_marker


def _argv_opt(argv: list[str], name: str) -> str | None:
    """Return the value following ``name`` in *argv*, or None."""
    for i, tok in enumerate(argv):
        if tok == name and i + 1 < len(argv):
            return argv[i + 1]
    return None

_EMPTY_THREADS_RESPONSE: dict[str, Any] = {
    "data": {
        "repository": {
            "pullRequest": {
                "reviewThreads": {
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                    "nodes": [],
                }
            }
        }
    }
}

# Standalone stdlib-only shim; it must not import daydream (it runs as the
# ``gh`` subprocess spawned by git_ops._run_gh).
_SHIM_SOURCE = '''#!/usr/bin/env python3
"""Fake ``gh`` shim installed by tests/harness/fake_gh.py."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CALLS = HERE / "calls.jsonl"
RESPONSES = HERE / "responses.json"

_EMPTY_THREADS = {
    "data": {"repository": {"pullRequest": {"reviewThreads": {
        "pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": []}}}}
}


def _parse(argv):
    method = "GET"
    endpoint = None
    payload = None
    jq = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--method", "-X"):
            method = argv[i + 1].upper()
            i += 2
        elif tok == "--input":
            payload = json.loads(Path(argv[i + 1]).read_text(encoding="utf-8"))
            i += 2
        elif tok == "--jq":
            jq = argv[i + 1]
            i += 2
        elif tok in ("-H", "--header"):
            i += 2
        elif tok.startswith("-"):
            i += 1
        else:
            endpoint = tok
            i += 1
    return method, (endpoint or "").lstrip("/"), payload, jq


def _emit(value, jq):
    """Print a response the way gh does: NDJSON of @json-encoded values under --jq."""
    if jq is None:
        print(json.dumps(value))
        return
    # Production only ever passes the `(.[]) | @json` flattening filter, whose
    # output is one JSON-encoded element per line.
    for item in value if isinstance(value, list) else [value]:
        print(json.dumps(item))


def _opt(argv, name):
    """Return the value following ``name`` in argv, or None."""
    for i, tok in enumerate(argv):
        if tok == name and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _next_comment_seq():
    """Monotonic counter so each posted comment gets a distinct id."""
    seq_file = HERE / "comment_seq"
    n = int(seq_file.read_text()) + 1 if seq_file.exists() else 1
    seq_file.write_text(str(n))
    return n


def _record(kind, argv, stdin):
    with CALLS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, "argv": argv, "stdin": stdin}) + "\\n")


def _handle_set(kind, argv):
    """Handle ``secret set`` / ``variable set``. Value via stdin or --body."""
    name = argv[2] if len(argv) > 2 and not argv[2].startswith("-") else None
    body = _opt(argv, "--body")
    stdin = "" if body is not None else sys.stdin.read()
    _record(kind + " set", argv, stdin)
    return 0


def _handle_list(kind, argv):
    """Handle ``secret list`` / ``variable list`` with ``--json name``."""
    _record(kind + " list", argv, "")
    responses = json.loads(RESPONSES.read_text(encoding="utf-8")) if RESPONSES.exists() else {}
    names = responses.get(kind + "-list", [])
    print(json.dumps([{"name": n} for n in names]))
    return 0


def _handle_pr_create(argv):
    _record("pr create", argv, "")
    responses = json.loads(RESPONSES.read_text(encoding="utf-8")) if RESPONSES.exists() else {}
    url = responses.get("pr-create")
    if url is None:
        sys.stderr.write("fake gh: no pr-create response configured\\n")
        return 1
    print(url)
    return 0


def _handle_pr_view(argv):
    _record("pr view", argv, "")
    responses = json.loads(RESPONSES.read_text(encoding="utf-8")) if RESPONSES.exists() else {}
    value = responses.get("pr-view")
    if value is None:
        sys.stderr.write("fake gh: no pr-view response configured\\n")
        return 1
    print(json.dumps(value))
    return 0


def _handle_pr_list(argv):
    _record("pr list", argv, "")
    responses = json.loads(RESPONSES.read_text(encoding="utf-8")) if RESPONSES.exists() else {}
    value = responses.get("pr-list")
    if value is None:
        pr_view = responses.get("pr-view")
        if pr_view is None:
            sys.stderr.write("fake gh: no pr-list or pr-view response configured\\n")
            return 1
        value = [pr_view]
    print(json.dumps(value))
    return 0


def _handle_repo_view(argv):
    _record("repo view", argv, "")
    responses = json.loads(RESPONSES.read_text(encoding="utf-8")) if RESPONSES.exists() else {}
    value = responses.get("repo-view")
    if value is None:
        if "pr-view" not in responses:
            sys.stderr.write("fake gh: no repo-view response configured\\n")
            return 1
        value = "acme/widgets"
    if isinstance(value, dict):
        value = value.get("nameWithOwner")
    if not isinstance(value, str):
        sys.stderr.write("fake gh: invalid repo-view response configured\\n")
        return 1
    print(value)
    return 0


def main():
    argv = sys.argv[1:]
    if argv[:2] in (["secret", "set"], ["variable", "set"]):
        return _handle_set(argv[0], argv)
    if argv[:2] in (["secret", "list"], ["variable", "list"]):
        return _handle_list(argv[0], argv)
    if argv[:2] == ["pr", "view"]:
        return _handle_pr_view(argv)
    if argv[:2] == ["pr", "list"]:
        return _handle_pr_list(argv)
    if argv[:2] == ["pr", "create"]:
        return _handle_pr_create(argv)
    if argv[:2] == ["repo", "view"]:
        return _handle_repo_view(argv)
    if not argv or argv[0] != "api":
        sys.stderr.write("fake gh: unsupported invocation: %r\\n" % (argv,))
        return 1
    method, endpoint, payload, jq = _parse(argv[1:])
    with CALLS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"argv": argv, "method": method, "endpoint": endpoint, "payload": payload}
        ) + "\\n")
    responses = json.loads(RESPONSES.read_text(encoding="utf-8")) if RESPONSES.exists() else {}
    if endpoint == "graphql":
        query = (payload or {}).get("query", "")
        if "minimizeComment" in query:
            print(json.dumps({"data": {"minimizeComment": {
                "minimizedComment": {"isMinimized": True}}}}))
            return 0
        if "reviewThreads" in query:
            print(json.dumps(responses.get("graphql_threads", _EMPTY_THREADS)))
            return 0
        sys.stderr.write("fake gh: unrecognized graphql query\\n")
        return 1
    key = method + " " + endpoint
    if key in responses:
        _emit(responses[key], jq)
        return 0
    # Query strings select/paginate; the canned response is keyed by path alone.
    bare_key = method + " " + endpoint.split("?")[0]
    if bare_key in responses:
        _emit(responses[bare_key], jq)
        return 0
    if method == "GET" and re.fullmatch(r"repos/[^/]+/[^/]+/pulls/\\d+/reviews", endpoint):
        _emit([], jq)
        return 0
    if method == "POST" and re.fullmatch(r"repos/[^/]+/[^/]+/pulls/\\d+/reviews", endpoint):
        print(json.dumps({"html_url": "https://github.test/fake/pull/7#pullrequestreview-1"}))
        return 0
    if method == "POST" and re.fullmatch(r"repos/[^/]+/[^/]+/pulls/\\d+/comments", endpoint):
        # Real GitHub 422s a file-level comment whose path is not in the PR
        # diff; `diff-paths`, when configured, reproduces that rejection.
        allowed = responses.get("diff-paths")
        path = (payload or {}).get("path")
        if allowed is not None and path not in allowed:
            sys.stderr.write("fake gh: path %r not in PR diff (422)\\n" % (path,))
            return 1
        print(json.dumps({
            "id": 9000 + _next_comment_seq(),
            "html_url": "https://github.test/fake/pull/7#discussion_r1",
        }))
        return 0
    sys.stderr.write("fake gh: no canned response for %s\\n" % key)
    return 1


if __name__ == "__main__":
    sys.exit(main())
'''


@dataclass
class GhCall:
    """One recorded ``gh api`` invocation.

    Attributes:
        endpoint: Endpoint with any leading slash stripped (``graphql`` for
            GraphQL calls).
        payload: Parsed ``--input`` JSON payload, or None.
    """

    endpoint: str
    payload: Any


@dataclass
class GhCommandCall:
    """One recorded non-API ``gh`` invocation."""

    kind: str
    argv: list[str]


@dataclass
class GhSetCall:
    """One recorded ``gh secret set`` / ``gh variable set`` invocation.

    Attributes:
        name: The secret/variable name (first positional after ``set``).
        org: The ``--org`` scope value, or None.
        repo: The ``--repo`` scope value, or None.
        argv: The full argv (after ``gh``) — assert PEM material never appears.
        stdin: The value piped on stdin (for secrets set without ``--body``).
    """

    name: str | None
    org: str | None
    repo: str | None
    argv: list[str]
    stdin: str


class FakeGh:
    """Driver/inspector for the installed fake ``gh`` shim."""

    def __init__(self, bin_dir: Path) -> None:
        self.bin_dir = bin_dir
        self._calls_path = bin_dir / "calls.jsonl"
        self._responses_path = bin_dir / "responses.json"

    # --- inspection ---------------------------------------------------------

    def calls(self, method: str, endpoint: str | None = None) -> list[GhCall]:
        """Return recorded calls matching ``method`` (and ``endpoint`` if given)."""
        out: list[GhCall] = []
        if not self._calls_path.exists():
            return out
        wanted_endpoint = endpoint.lstrip("/") if endpoint is not None else None
        for line in self._calls_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if "method" not in record:  # non-api record (secret/variable/pr)
                continue
            if record["method"] != method.upper():
                continue
            if wanted_endpoint is not None and record["endpoint"] != wanted_endpoint:
                continue
            out.append(GhCall(endpoint=record["endpoint"], payload=record["payload"]))
        return out

    def command_calls(self, kind: str) -> list[GhCommandCall]:
        """Return recorded non-API calls matching *kind* (for example ``pr view``)."""
        out: list[GhCommandCall] = []
        if not self._calls_path.exists():
            return out
        for line in self._calls_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("kind") == kind:
                out.append(GhCommandCall(kind=kind, argv=record["argv"]))
        return out

    def pr_view_calls(self) -> list[GhCommandCall]:
        """Return recorded ``gh pr view`` invocations in order."""
        return self.command_calls("pr view")

    def _set_calls(self, kind: str) -> list[GhSetCall]:
        out: list[GhSetCall] = []
        if not self._calls_path.exists():
            return out
        for line in self._calls_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("kind") != kind:
                continue
            argv = record["argv"]
            name = argv[2] if len(argv) > 2 and not argv[2].startswith("-") else None
            out.append(
                GhSetCall(
                    name=name,
                    org=_argv_opt(argv, "--org"),
                    repo=_argv_opt(argv, "--repo"),
                    argv=argv,
                    stdin=record.get("stdin", ""),
                )
            )
        return out

    def secret_set_calls(self) -> list[GhSetCall]:
        """Return recorded ``gh secret set`` invocations in order."""
        return self._set_calls("secret set")

    def variable_set_calls(self) -> list[GhSetCall]:
        """Return recorded ``gh variable set`` invocations in order."""
        return self._set_calls("variable set")

    # --- canned-response configuration ---------------------------------------

    def set_response(self, method: str, endpoint: str | None = None, value: Any = None) -> None:
        """Configure a canned response.

        Two call shapes are supported:
        - ``set_response(method, endpoint, value)`` — keys the ``gh api``
          response under ``"<METHOD> <endpoint>"`` (existing behavior).
        - ``set_response("pr-create", value=...)`` (or ``"pr-view"`` /
          ``"repo-view"``) — keys a non-api response under the bare
          ``method`` token.
        """
        responses = self._read_responses()
        if endpoint is None:
            responses[method] = value
        else:
            responses[f"{method.upper()} {endpoint.lstrip('/')}"] = value
        self._responses_path.write_text(json.dumps(responses), encoding="utf-8")

    def serve_pr_view(self, response: dict[str, Any]) -> None:
        """Make ``gh pr view`` emit *response* and feed ``gh pr list``."""
        self.set_response("pr-view", value=response)

    def serve_repo_view(self, name_with_owner: str) -> None:
        """Make ``gh repo view`` emit *name_with_owner*."""
        self.set_response("repo-view", value=name_with_owner)

    def serve_secret_list(self, names: list[str]) -> None:
        """Make ``gh secret list --json name`` return *names*."""
        self.set_response("secret-list", value=names)

    def serve_variable_list(self, names: list[str]) -> None:
        """Make ``gh variable list --json name`` return *names*."""
        self.set_response("variable-list", value=names)

    def serve_installations(self, installations: list[dict[str, Any]]) -> None:
        """Make ``gh api /app/installations`` return *installations*.

        Each installation should carry an ``account.login`` so the verify
        doctor's App-installed check can confirm the target owner appears.
        """
        self.set_response("GET", "/app/installations", value=installations)

    def serve_prior_threads(self, *, fingerprints: list[str], thread_ids: list[str]) -> None:
        """Serve a prior-thread inventory: one unresolved inline thread per fingerprint."""
        nodes = [
            self._thread_node(thread_id, f"RC_{i}", 1000 + i, finding_marker(fingerprint))
            for i, (fingerprint, thread_id) in enumerate(
                zip(fingerprints, thread_ids, strict=True), start=1
            )
        ]
        self._write_threads(nodes)

    def serve_prior_threads_from(self, call: GhCall) -> None:
        """Make GitHub "remember" a recorded review POST as prior findings.

        Each inline comment in the posted payload becomes an unresolved review
        thread, and the review body becomes a REST review (body-only markers).
        """
        payload = call.payload or {}
        nodes = [
            self._thread_node(f"RT_{i}", f"RC_{i}", i, comment.get("body", ""))
            for i, comment in enumerate(payload.get("comments", []), start=1)
        ]
        self._write_threads(nodes)
        self.set_response(
            "GET", call.endpoint, [{"id": 1, "node_id": "PRR_1", "body": payload.get("body", "")}]
        )

    # --- internals ------------------------------------------------------------

    @staticmethod
    def _thread_node(
        thread_id: str, comment_node_id: str, database_id: int, body: str
    ) -> dict[str, Any]:
        return {
            "id": thread_id,
            "isResolved": False,
            "comments": {
                "nodes": [
                    {
                        "id": comment_node_id,
                        "databaseId": database_id,
                        "body": body,
                        "isMinimized": False,
                    }
                ]
            },
        }

    def _write_threads(self, nodes: list[dict[str, Any]]) -> None:
        response = json.loads(json.dumps(_EMPTY_THREADS_RESPONSE))
        response["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"] = nodes
        responses = self._read_responses()
        responses["graphql_threads"] = response
        self._responses_path.write_text(json.dumps(responses), encoding="utf-8")

    def _read_responses(self) -> dict[str, Any]:
        if self._responses_path.exists():
            return json.loads(self._responses_path.read_text(encoding="utf-8"))
        return {}


def install_fake_gh(bin_dir: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGh:
    """Write the ``gh`` shim into ``bin_dir`` and prepend it to ``PATH``."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "gh"
    # Pin the shebang to the interpreter running the suite. A bare
    # ``#!/usr/bin/env python3`` resolves ``python3`` off PATH to whatever shim
    # comes first (e.g. a pyenv shim), whose cold-start can intermittently
    # exceed git_ops's 60s ``gh`` timeout and flake every fake-gh test. The
    # current interpreter is always present and starts immediately.
    source = _SHIM_SOURCE.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1)
    shim.write_text(source, encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return FakeGh(bin_dir)
