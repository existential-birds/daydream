"""In-process fake ``gh`` harness for real-path tests.

:func:`install_fake_gh` patches ``subprocess.run`` as seen by
:mod:`daydream.git_ops` (the single point of contact for all ``gh`` calls)
with a router: an argv starting with ``gh`` is answered synchronously by
:func:`_handle_gh`; every other command (``git`` against real temp
worktrees, most importantly) runs for real. Everything of daydream's runs
unchanged — ``_run_gh``'s env/token merging and stdin piping, the ``gh_api``
tempfile-``--input`` path, JSON response parsing — only the OS process spawn
is replaced. No fork, no ``PATH`` shim, no wall-clock timeout: the fake
cannot stall, so these tests are deterministic under any host load.

The handler records every invocation (argv + parsed ``--input`` payload) to
a JSONL log the :class:`FakeGh` helper parses, and replies from a canned
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
import re
import subprocess
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

_GIT_CREDENTIAL_HELPER = (
    "protocol=https\n"
    "host=github.com\n"
    "username=x\n"
    "password=<fake>\n"
)

_LS_REMOTE_DEFAULT = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/heads/head\n"
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\trefs/heads/base\n"
)

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


# --- Request handler (the fake ``gh`` itself) --------------------------------
#
# Each handler returns ``(returncode, stdout, stderr)`` — the observable
# surface of a ``gh`` invocation. State (call log, canned responses, comment
# id counter) lives in files under ``state_dir`` so :class:`FakeGh` can
# inspect and configure it independently of the handler.


def _read_responses(state: Path) -> dict[str, Any]:
    path = state / "responses.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _record(state: Path, record: dict[str, Any]) -> None:
    with (state / "calls.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _emit(value: Any, jq: str | None) -> str:
    """Render a response the way ``gh`` prints it: NDJSON of ``@json``-encoded values under ``--jq``."""
    if jq is None:
        return json.dumps(value) + "\n"
    # Production only ever passes the `(.[]) | @json` flattening filter, whose
    # output is one JSON-encoded element per line.
    items = value if isinstance(value, list) else [value]
    return "".join(json.dumps(item) + "\n" for item in items)


def _next_comment_seq(state: Path) -> int:
    """Monotonic counter so each posted comment gets a distinct id."""
    seq_file = state / "comment_seq"
    n = int(seq_file.read_text()) + 1 if seq_file.exists() else 1
    seq_file.write_text(str(n))
    return n


def _parse_api(argv: list[str]) -> tuple[str, str, Any, str | None]:
    """Parse a ``gh api`` argv (after ``api``) into method/endpoint/payload/jq."""
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


def _handle_set(kind: str, argv: list[str], stdin_text: str, state: Path) -> tuple[int, str, str]:
    """Handle ``secret set`` / ``variable set``. Value via stdin or ``--body``."""
    body = _argv_opt(argv, "--body")
    stdin = "" if body is not None else stdin_text
    _record(state, {"kind": kind + " set", "argv": argv, "stdin": stdin})
    return 0, "", ""


def _handle_list(kind: str, argv: list[str], state: Path) -> tuple[int, str, str]:
    """Handle ``secret list`` / ``variable list`` with ``--json name``."""
    _record(state, {"kind": kind + " list", "argv": argv, "stdin": ""})
    names = _read_responses(state).get(kind + "-list", [])
    return 0, json.dumps([{"name": n} for n in names]) + "\n", ""


def _handle_pr_create(argv: list[str], state: Path) -> tuple[int, str, str]:
    _record(state, {"kind": "pr create", "argv": argv, "stdin": ""})
    url = _read_responses(state).get("pr-create")
    if url is None:
        return 1, "", "fake gh: no pr-create response configured\n"
    return 0, str(url) + "\n", ""


def _handle_pr_view(argv: list[str], state: Path) -> tuple[int, str, str]:
    _record(state, {"kind": "pr view", "argv": argv, "stdin": ""})
    value = _read_responses(state).get("pr-view")
    if value is None:
        return 1, "", "fake gh: no pr-view response configured\n"
    return 0, json.dumps(value) + "\n", ""


def _handle_pr_list(argv: list[str], state: Path) -> tuple[int, str, str]:
    _record(state, {"kind": "pr list", "argv": argv, "stdin": ""})
    responses = _read_responses(state)
    value = responses.get("pr-list")
    if value is None:
        pr_view = responses.get("pr-view")
        if pr_view is None:
            return 1, "", "fake gh: no pr-list or pr-view response configured\n"
        value = [pr_view]
    return 0, json.dumps(value) + "\n", ""


def _handle_repo_view(argv: list[str], state: Path) -> tuple[int, str, str]:
    _record(state, {"kind": "repo view", "argv": argv, "stdin": ""})
    responses = _read_responses(state)
    value = responses.get("repo-view")
    if value is None:
        if "pr-view" not in responses:
            return 1, "", "fake gh: no repo-view response configured\n"
        value = "acme/widgets"
    if isinstance(value, dict):
        value = value.get("nameWithOwner")
    if not isinstance(value, str):
        return 1, "", "fake gh: invalid repo-view response configured\n"
    return 0, value + "\n", ""


def _handle_api(argv: list[str], state: Path) -> tuple[int, str, str]:
    method, endpoint, payload, jq = _parse_api(argv[1:])
    _record(state, {"argv": argv, "method": method, "endpoint": endpoint, "payload": payload})
    responses = _read_responses(state)
    if endpoint == "graphql":
        query = (payload or {}).get("query", "")
        if "minimizeComment" in query:
            reply: dict[str, Any] = {"data": {"minimizeComment": {"minimizedComment": {"isMinimized": True}}}}
            return 0, json.dumps(reply) + "\n", ""
        if "reviewThreads" in query:
            return 0, json.dumps(responses.get("graphql_threads", _EMPTY_THREADS_RESPONSE)) + "\n", ""
        return 1, "", "fake gh: unrecognized graphql query\n"
    key = f"{method} {endpoint}"
    if key in responses:
        return 0, _emit(responses[key], jq), ""
    # Query strings select/paginate; the canned response is keyed by path alone.
    bare_key = f"{method} {endpoint.split('?')[0]}"
    if bare_key in responses:
        return 0, _emit(responses[bare_key], jq), ""
    if method == "GET" and re.fullmatch(r"repos/[^/]+/[^/]+/pulls/\d+/reviews", endpoint):
        return 0, _emit([], jq), ""
    if method == "POST" and re.fullmatch(r"repos/[^/]+/[^/]+/pulls/\d+/reviews", endpoint):
        return 0, json.dumps({"html_url": "https://github.test/fake/pull/7#pullrequestreview-1"}) + "\n", ""
    if method == "POST" and re.fullmatch(r"repos/[^/]+/[^/]+/pulls/\d+/comments", endpoint):
        # Real GitHub 422s a file-level comment whose path is not in the PR
        # diff; `diff-paths`, when configured, reproduces that rejection.
        allowed = responses.get("diff-paths")
        path = (payload or {}).get("path")
        if allowed is not None and path not in allowed:
            return 1, "", f"fake gh: path {path!r} not in PR diff (422)\n"
        reply = {
            "id": 9000 + _next_comment_seq(state),
            "html_url": "https://github.test/fake/pull/7#discussion_r1",
        }
        return 0, json.dumps(reply) + "\n", ""
    return 1, "", f"fake gh: no canned response for {key}\n"


def _handle_gh(argv: list[str], stdin_text: str, state: Path) -> tuple[int, str, str]:
    """Answer one ``gh`` invocation (argv after ``gh``). Returns (rc, stdout, stderr)."""
    if argv[:2] in (["secret", "set"], ["variable", "set"]):
        return _handle_set(argv[0], argv, stdin_text, state)
    if argv[:2] in (["secret", "list"], ["variable", "list"]):
        return _handle_list(argv[0], argv, state)
    if argv[:2] == ["pr", "view"]:
        return _handle_pr_view(argv, state)
    if argv[:2] == ["pr", "list"]:
        return _handle_pr_list(argv, state)
    if argv[:2] == ["pr", "create"]:
        return _handle_pr_create(argv, state)
    if argv[:2] == ["repo", "view"]:
        return _handle_repo_view(argv, state)
    if argv[:2] == ["auth", "status"]:
        _record(state, {"kind": "auth status", "argv": argv, "stdin": ""})
        return 0, "", ""
    if argv[:2] == ["auth", "git-credential"]:
        _record(state, {"kind": "auth git-credential", "argv": argv, "stdin": ""})
        return 0, _GIT_CREDENTIAL_HELPER, ""
    if not argv or argv[0] != "api":
        return 1, "", f"fake gh: unsupported invocation: {argv!r}\n"
    return _handle_api(argv, state)


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
    """Driver/inspector for the in-process fake ``gh``."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self._calls_path = state_dir / "calls.jsonl"
        self._responses_path = state_dir / "responses.json"

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

    def serve_prior_threads(
        self,
        *,
        fingerprints: list[str],
        thread_ids: list[str],
        authors: list[str] | None = None,
        viewer_did_author: bool | None = None,
    ) -> None:
        """Serve a prior-thread inventory: one unresolved inline thread per fingerprint.

        When *authors* is provided it must parallel *fingerprints* and is zipped
        into the comment nodes as ``author { login }``. When *viewer_did_author*
        is set the comment nodes carry ``viewerDidAuthor``. Either None leaves
        the key absent so absence remains testable.
        """
        if authors is not None and len(authors) != len(fingerprints):
            raise ValueError("authors must parallel fingerprints")
        nodes = [
            self._thread_node(
                thread_id,
                f"RC_{i}",
                1000 + i,
                finding_marker(fingerprint),
                author=authors[i - 1] if authors is not None else None,
                viewer_did_author=viewer_did_author,
            )
            for i, (fingerprint, thread_id) in enumerate(
                zip(fingerprints, thread_ids, strict=True), start=1
            )
        ]
        self._write_threads(nodes)

    def serve_prior_threads_from(
        self, call: GhCall, *,
        author: str | None = None,
        viewer_did_author: bool | None = None,
    ) -> None:
        """Make GitHub "remember" a recorded review POST as prior findings.

        Each inline comment in the posted payload becomes an unresolved review
        thread, and the review body becomes a REST review (body-only markers).
        When *author* is set, both the GraphQL thread comments and the REST
        review carry it (``author { login }`` / ``user { login }`` respectively)
        so the bot-author trust rule in ``fetch_prior_findings`` accepts them.
        """
        payload = call.payload or {}
        nodes = [
            self._thread_node(
                f"RT_{i}", f"RC_{i}", i, comment.get("body", ""),
                author=author, viewer_did_author=viewer_did_author,
            )
            for i, comment in enumerate(payload.get("comments", []), start=1)
        ]
        self._write_threads(nodes)
        review: dict[str, Any] = {"id": 1, "node_id": "PRR_1", "body": payload.get("body", "")}
        if author is not None:
            review["user"] = {"login": author}
        self.set_response("GET", call.endpoint, [review])

    # --- internals ------------------------------------------------------------

    @staticmethod
    def _thread_node(
        thread_id: str,
        comment_node_id: str,
        database_id: int,
        body: str,
        *,
        author: str | None = None,
        viewer_did_author: bool | None = None,
    ) -> dict[str, Any]:
        comment: dict[str, Any] = {
            "id": comment_node_id,
            "databaseId": database_id,
            "body": body,
            "isMinimized": False,
        }
        if author is not None:
            comment["author"] = {"login": author}
        if viewer_did_author is not None:
            comment["viewerDidAuthor"] = bool(viewer_did_author)
        return {
            "id": thread_id,
            "isResolved": False,
            "comments": {"nodes": [comment]},
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


def install_fake_gh(state_dir: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGh:
    """Route ``gh`` invocations to the in-process handler; run everything else for real."""
    state_dir.mkdir(parents=True, exist_ok=True)
    real_run = subprocess.run

    def router(args: Any, *pargs: Any, **kwargs: Any) -> Any:
        if isinstance(args, (list, tuple)) and args and args[0] == "gh":
            rc, out, err = _handle_gh(list(args[1:]), kwargs.get("input") or "", state_dir)
            return subprocess.CompletedProcess(list(args), rc, stdout=out, stderr=err)
        if (
            isinstance(args, (list, tuple))
            and args
            and args[0] == "git"
            and "ls-remote" in args
        ):
            refs = _read_responses(state_dir).get("git-ls-remote", _LS_REMOTE_DEFAULT)
            _record(
                state_dir,
                {"kind": "git ls-remote", "argv": list(args), "env": kwargs.get("env")},
            )
            return subprocess.CompletedProcess(list(args), 0, stdout=refs, stderr="")
        return real_run(args, *pargs, **kwargs)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", router)
    return FakeGh(state_dir)
