"""Shared fake gh handler for synchronous interception and real-process lifecycle tests.
Non-gh commands, including Git against temp worktrees, run normally. Record argv/input
payloads in JSONL; serve canned METHOD/endpoint responses before built-in behavior.
Query-string-free keys are a fallback, and --jq emits real-gh-style NDJSON. GraphQL
queries validate against the checked-in schema subset, including nested comment
pagination. Unsupported calls fail instead of inventing success.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

from daydream.reviews.identity import finding_marker
from tests.harness import github_schema


def _argv_opt(argv: list[str], name: str) -> str | None:
    for i, tok in enumerate(argv):
        if tok == name and i + 1 < len(argv):
            return argv[i + 1]
    return None

_GIT_CREDENTIAL_HELPER = ("protocol=https\n" "host=github.com\n" "username=x\n" "password=<fake>\n")

_LS_REMOTE_DEFAULT = (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\trefs/heads/head\n"
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\trefs/heads/base\n"
)

# Match gh 2.45: reject unsupported PR JSON fields before serving a response.
_GH_245_PR_JSON_FIELDS = frozenset({
        "number", "title", "body", "state", "headRefName", "baseRefName", "headRefOid", "url", "headRepository",
        "headRepositoryOwner",
    }
)

_EMPTY_THREADS_RESPONSE: dict[str, Any] = {"data": {"repository": {
            "pullRequest": {"reviewThreads": {"pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": []}}
        }
    }
}


# Handlers return (returncode, stdout, stderr). File-backed state lets FakeGh
# configure and inspect both intercepted calls and separate processes.


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


def _take_response(state: Path, key: str, value: Any) -> Any:
    """Return one canned response, advancing a configured sequence once."""
    if not isinstance(value, dict) or "__sequence__" not in value:
        return value
    sequence = value["__sequence__"]
    if not isinstance(sequence, list):
        return {"__error__": f"fake gh: invalid response sequence for {key}"}
    cursor_path = state / "response_cursors.json"
    cursors = (json.loads(cursor_path.read_text(encoding="utf-8")) if cursor_path.exists() else {})
    index = cursors.get(key, 0)
    if not isinstance(index, int) or index < 0 or index >= len(sequence):
        return {"__error__": f"fake gh: response sequence exhausted for {key}"}
    cursors[key] = index + 1
    cursor_path.write_text(json.dumps(cursors), encoding="utf-8")
    return sequence[index]


def _serve_api_response(state: Path, key: str, value: Any, jq: str | None,) -> tuple[int, str, str]:
    value = _take_response(state, key, value)
    if isinstance(value, dict) and isinstance(value.get("__error__"), str):
        return 1, "", value["__error__"] + "\n"
    if isinstance(value, dict) and isinstance(value.get("__stdout__"), str):
        return 0, value["__stdout__"], ""
    if isinstance(value, dict) and isinstance(value.get("__blocking__"), dict):
        pid_file = Path(str(value["__blocking__"].get("pid_file", "")))
        child_pid = os.fork()
        if child_pid == 0:
            while True:
                time.sleep(3600)
        pid_file.write_text(json.dumps({"direct": os.getpid(), "grandchild": child_pid}), encoding="utf-8",)
        while True:
            time.sleep(3600)
    if value is None:
        # Absence classification requires gh's exact "(HTTP 404)" stderr token.
        return 1, "", f"gh: Not Found (HTTP 404)\nfake gh: {key} (no such resource)\n"
    return 0, _emit(value, jq), ""


def _next_comment_seq(state: Path) -> int:
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


def _handle_pr(argv: list[str], state: Path) -> tuple[int, str, str]:
    action = argv[1]
    key = f"pr-{action}"
    _record(state, {"kind": f"pr {action}", "argv": argv, "stdin": ""})
    responses = _read_responses(state)
    value = responses.get(key)
    from_view = action == "list" and value is None
    if from_view:
        value = responses.get("pr-view")
        if value is None:
            return 1, "", "fake gh: no pr-list or pr-view response configured\n"
    if value is None:
        return 1, "", f"fake gh: no {key} response configured\n"
    if action == "create":
        return 0, str(value) + "\n", ""
    if isinstance(value, dict) and isinstance(value.get("__error__"), str):
        return 1, "", value["__error__"] + "\n"
    return 0, json.dumps([value] if from_view else value) + "\n", ""


def _handle_repo_view(argv: list[str], state: Path) -> tuple[int, str, str]:
    _record(state, {"kind": "repo view", "argv": argv, "stdin": ""})
    responses = _read_responses(state)
    # Positional OWNER/REPO requests return identity JSON; the local-repo
    # nameWithOwner query returns a bare slug.
    if len(argv) > 2 and not argv[2].startswith("-"):
        full = responses.get("repo-view-full")
        if full is None:
            return 1, "", "fake gh: no repo-view-full response configured\n"
        return 0, json.dumps(full) + "\n", ""
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
        variables = (payload or {}).get("variables") or {}
        if "minimizeComment" in query:
            reply: dict[str, Any] = {"data": {"minimizeComment": {"minimizedComment": {"isMinimized": True}}}}
            return 0, json.dumps(reply) + "\n", ""
        if "PullRequestReviewThread" in query or "reviewThreads" in query:
            unknown = github_schema.unknown_query_fields(query)
            if unknown:
                return (1, "", f"fake gh: graphql query requests fields not in GitHub schema: {sorted(unknown)}\n")
        if "PullRequestReviewThread" in query:
            # Per-thread ``node(id:)`` comments page catalog: serve one page per
            # incoming ``commentsAfter`` cursor, deterministic endCursor per page.
            thread_id = variables.get("threadId") if isinstance(variables, dict) else None
            pages = responses.get(f"graphql_thread_comments:{thread_id}")
            if not isinstance(pages, list) or not pages:
                return 1, "", f"fake gh: no thread-comment catalog for {thread_id}\n"
            after = variables.get("commentsAfter") if isinstance(variables, dict) else None
            if after is None:
                idx = 0
            else:
                idx = int(str(after).rsplit(":p", 1)[1]) + 1
            if idx >= len(pages):
                return 1, "", f"fake gh: thread-comment cursor {after!r} past the catalog end\n"
            return 0, json.dumps(pages[idx]) + "\n", ""
        if "reviewThreads" in query:
            pr_num = variables.get("number") if isinstance(variables, dict) else None
            key = f"graphql_threads:{pr_num}" if pr_num is not None else "graphql_threads"
            value = responses.get(key) or responses.get("graphql_threads") or _EMPTY_THREADS_RESPONSE
            return 0, json.dumps(value) + "\n", ""
        return 1, "", "fake gh: unrecognized graphql query\n"
    key = f"{method} {endpoint}"
    if key in responses:
        return _serve_api_response(state, key, responses[key], jq)
    # Query strings select/paginate; the canned response is keyed by path alone.
    bare_key = f"{method} {endpoint.split('?')[0]}"
    if bare_key in responses:
        return _serve_api_response(state, bare_key, responses[bare_key], jq)
    if method == "GET" and re.fullmatch(
        r"repos/[^/]+/[^/]+/(?:pulls/\d+/(?:reviews|files|comments)|issues/\d+/comments)", endpoint
    ):
        return 0, _emit([], jq), ""
    if method == "POST" and re.fullmatch(r"repos/[^/]+/[^/]+/pulls/\d+/reviews", endpoint):
        return 0, json.dumps({"html_url": "https://github.test/fake/pull/7#pullrequestreview-1"}) + "\n", ""
    if method == "POST" and re.fullmatch(r"repos/[^/]+/[^/]+/issues/\d+/comments", endpoint):
        # Issue-comment creation (issue #1113: the standalone diagram comment).
        # ``node_id`` is present because minimization is keyed on it.
        seq = _next_comment_seq(state)
        reply = {"id": 7000 + seq, "node_id": f"IC_fake{seq}",
            "html_url": f"https://github.test/fake/pull/7#issuecomment-{7000 + seq}",
        }
        return 0, json.dumps(reply) + "\n", ""
    if method == "POST" and re.fullmatch(r"repos/[^/]+/[^/]+/pulls/\d+/comments", endpoint):
        # Real GitHub 422s a file-level comment whose path is not in the PR
        # diff; `diff-paths`, when configured, reproduces that rejection.
        allowed = responses.get("diff-paths")
        path = (payload or {}).get("path")
        if allowed is not None and path not in allowed:
            return 1, "", f"fake gh: path {path!r} not in PR diff (422)\n"
        reply = {"id": 9000 + _next_comment_seq(state), "html_url": "https://github.test/fake/pull/7#discussion_r1"}
        return 0, json.dumps(reply) + "\n", ""
    return 1, "", f"fake gh: no canned response for {key}\n"


def _handle_gh(argv: list[str], stdin_text: str, state: Path) -> tuple[int, str, str]:
    """Answer one ``gh`` invocation (argv after ``gh``). Returns (rc, stdout, stderr)."""
    if argv[:2] in (["pr", "view"], ["pr", "list"]):
        requested = (_argv_opt(argv, "--json") or "").split(",")
        unsupported = sorted(set(requested) - _GH_245_PR_JSON_FIELDS)
        if unsupported:
            return 1, "", f'Unknown JSON field: "{unsupported[0]}"\n'
    if argv[:2] in (["secret", "set"], ["variable", "set"]):
        return _handle_set(argv[0], argv, stdin_text, state)
    if argv[:2] in (["secret", "list"], ["variable", "list"]):
        return _handle_list(argv[0], argv, state)
    if len(argv) > 1 and argv[0] == "pr" and argv[1] in ("view", "list", "create"):
        return _handle_pr(argv, state)
    if argv[:2] == ["repo", "view"]:
        return _handle_repo_view(argv, state)
    if argv[:2] == ["auth", "status"]:
        _record(state, {"kind": "auth status", "argv": argv, "stdin": ""})
        return 0, "", ""
    if argv[:2] == ["auth", "git-credential"]:
        # Validate Git's operation and stdin protocol before returning credentials.
        op = argv[2] if len(argv) > 2 else None
        _record(state, {"kind": "auth git-credential", "argv": argv, "stdin": stdin_text})
        if op not in ("get", "store", "erase") or (
            op in ("get", "store") and not ("protocol=" in stdin_text and "host=" in stdin_text)
        ):
            return 1, "", "fake gh: git-credential requires an operation and protocol/host on stdin\n"
        return 0, _GIT_CREDENTIAL_HELPER, ""
    if not argv or argv[0] != "api":
        return 1, "", f"fake gh: unsupported invocation: {argv!r}\n"
    return _handle_api(argv, state)


@dataclass
class GhCall:
    """API call with a slash-free endpoint and argv excluding ``gh``."""

    endpoint: str
    payload: Any
    argv: list[str] | None = None


@dataclass
class GhCommandCall:
    """One recorded non-API ``gh`` invocation."""

    kind: str
    argv: list[str]
    env: dict[str, Any] | None = None


@dataclass
class GhProcessCall:
    """One intercepted ``gh`` process with its exact checkout context."""

    cwd: Path
    argv: list[str]


@dataclass
class GhSetCall:
    """Secret/variable write recorded for credential-transport assertions."""

    name: str | None
    org: str | None
    repo: str | None
    argv: list[str]
    stdin: str


class FakeGh:
    """Driver/inspector for the shared in-process and PATH fake ``gh``."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self._calls_path = state_dir / "calls.jsonl"
        self._responses_path = state_dir / "responses.json"

    # --- inspection ---------------------------------------------------------

    def _records(self) -> Iterator[dict[str, Any]]:
        if not self._calls_path.exists():
            return
        for line in self._calls_path.read_text(encoding="utf-8").splitlines():
            yield json.loads(line)

    def calls(self, method: str, endpoint: str | None = None) -> list[GhCall]:
        """Return recorded calls matching ``method`` (and ``endpoint`` if given)."""
        out: list[GhCall] = []
        wanted_endpoint = endpoint.lstrip("/") if endpoint is not None else None
        for record in self._records():
            if "method" not in record:  # non-api record (secret/variable/pr)
                continue
            if record["method"] != method.upper():
                continue
            if wanted_endpoint is not None and record["endpoint"] != wanted_endpoint:
                continue
            out.append(GhCall(endpoint=record["endpoint"], payload=record["payload"], argv=record.get("argv"),))
        return out

    def command_calls(self, kind: str) -> list[GhCommandCall]:
        """Return recorded non-API calls matching *kind* (for example ``pr view``)."""
        out: list[GhCommandCall] = []
        for record in self._records():
            if record.get("kind") == kind:
                out.append(GhCommandCall(kind=kind, argv=record["argv"], env=record.get("env")))
        return out

    def process_calls(self) -> list[GhProcessCall]:
        """Return every intercepted ``gh`` process in invocation order."""
        out: list[GhProcessCall] = []
        for record in self._records():
            if record.get("kind") == "gh process":
                out.append(GhProcessCall(cwd=Path(record["cwd"]), argv=record["argv"],))
        return out

    def pr_view_calls(self) -> list[GhCommandCall]:
        return self.command_calls("pr view")

    def _set_calls(self, kind: str) -> list[GhSetCall]:
        out: list[GhSetCall] = []
        for record in self._records():
            if record.get("kind") != kind:
                continue
            argv = record["argv"]
            name = argv[2] if len(argv) > 2 and not argv[2].startswith("-") else None
            out.append(GhSetCall(name=name, org=_argv_opt(argv, "--org"), repo=_argv_opt(argv, "--repo"), argv=argv,
                    stdin=record.get("stdin", ""),
                )
            )
        return out

    def secret_set_calls(self) -> list[GhSetCall]:
        return self._set_calls("secret set")

    def variable_set_calls(self) -> list[GhSetCall]:
        return self._set_calls("variable set")

    # --- canned-response configuration ---------------------------------------

    def set_response(self, method: str, endpoint: str | None = None, value: Any = None) -> None:
        """Configure METHOD/endpoint API replies or a bare command key such as
        pr-create/pr-view/repo-view.
        """
        responses = self._read_responses()
        if endpoint is None:
            responses[method] = value
        else:
            responses[f"{method.upper()} {endpoint.lstrip('/')}"] = value
        self._responses_path.write_text(json.dumps(responses), encoding="utf-8")

    def set_response_sequence(self, key: str, responses: list[Any]) -> None:
        """Serve successive responses for one exact ``METHOD endpoint`` key."""
        self.set_response(key, value={"__sequence__": responses})
        cursor_path = self.state_dir / "response_cursors.json"
        if cursor_path.exists():
            cursors = json.loads(cursor_path.read_text(encoding="utf-8"))
            cursors.pop(key, None)
            cursor_path.write_text(json.dumps(cursors), encoding="utf-8")

    def serve_blocking_process(self, key: str, *, pid_file: Path) -> None:
        """Serve a process that blocks with a stdout-holding grandchild."""
        self.set_response(key, value={"__blocking__": {"pid_file": str(pid_file)}})

    def serve_pr_view(self, response: dict[str, Any]) -> None:
        """Make ``gh pr view`` emit *response* and feed ``gh pr list``."""
        self.set_response("pr-view", value=response)

    def serve_open_pr(self, target: Path) -> None:
        """Serve an open acme/widgets PR using the fixture repository's real HEAD."""
        from daydream import git_ops

        self.serve_pr_view({"number": 7, "state": "OPEN", "headRefName": "feature", "baseRefName": "main",
                "headRefOid": git_ops.head_sha(target),
                "headRepository": {"name": "widgets", "nameWithOwner": "acme/widgets"},
                "headRepositoryOwner": {"login": "acme"}, "url": "https://github.com/acme/widgets/pull/7", "body": "",
            }
        )

    def serve_secret_list(self, names: list[str]) -> None:
        self.set_response("secret-list", value=names)

    def serve_variable_list(self, names: list[str]) -> None:
        self.set_response("variable-list", value=names)

    def serve_installations(self, installations: list[dict[str, Any]]) -> None:
        """Serve App installations with ``account.login`` for owner verification."""
        self.set_response("GET", "/app/installations", value=installations)

    def serve_prior_issue_comments(self, comments: list[dict[str, Any]], *, repo: str = "acme/widgets", number: int = 7,
    ) -> None:
        """Configure REST issue comments for a PR. Diagram inventory requires node_id,
        marked body, and user.login; the canned response overrides the empty default.
        """
        self.set_response("GET", f"repos/{repo}/issues/{number}/comments", value=comments)

    def serve_prior_threads(self, *, fingerprints: list[str], thread_ids: list[str], authors: list[str] | None = None,
        viewer_did_author: bool | None = None,
    ) -> None:
        """Build one unresolved inline thread per fingerprint. Optional parallel
        authors/viewer flags preserve absent fields when unspecified so missing
        attribution remains testable.
        """
        if authors is not None and len(authors) != len(fingerprints):
            raise ValueError("authors must parallel fingerprints")
        nodes = [self._thread_node(thread_id, f"RC_{i}", 1000 + i, finding_marker(fingerprint),
                author=authors[i - 1] if authors is not None else None, viewer_did_author=viewer_did_author,
            )
            for i, (fingerprint, thread_id) in enumerate(zip(fingerprints, thread_ids, strict=True), start=1)
        ]
        self._write_threads(nodes)

    def serve_prior_threads_from(
        self, call: GhCall, *, author: str | None = None, viewer_did_author: bool | None = None,
    ) -> None:
        """Replay a recorded review POST as inline threads plus a body-only REST review.
        Optional author identity populates both protocol shapes for production trust
        checks.
        """
        payload = call.payload or {}
        nodes = [self._thread_node(
                f"RT_{i}", f"RC_{i}", i, comment.get("body", ""), author=author, viewer_did_author=viewer_did_author,
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
    def _thread_node(thread_id: str, comment_node_id: str, database_id: int, body: str, *, author: str | None = None,
        viewer_did_author: bool | None = None,
    ) -> dict[str, Any]:
        comment: dict[str, Any] = {"id": comment_node_id, "databaseId": database_id, "body": body, "isMinimized": False,
        }
        if author is not None:
            comment["author"] = {"login": author}
        if viewer_did_author is not None:
            comment["viewerDidAuthor"] = bool(viewer_did_author)
        return {"id": thread_id, "isResolved": False,
            "comments": {"nodes": [comment], "pageInfo": {"hasNextPage": False, "endCursor": None}},
        }

    def _write_threads(self, nodes: list[dict[str, Any]], number: int | None = None) -> None:
        """Serve thread nodes with nested pageInfo. Configured comment catalogs advertise
        further pages so production node-query pagination executes.
        """
        response = json.loads(json.dumps(_EMPTY_THREADS_RESPONSE))
        nodes = copy.deepcopy(nodes)
        responses = self._read_responses()
        for thread in nodes:
            comments = thread.get("comments")
            if not isinstance(comments, dict) or isinstance(comments.get("pageInfo"), dict):
                continue
            catalog = responses.get(f"graphql_thread_comments:{thread.get('id')}")
            if isinstance(catalog, list) and len(catalog) > 1:
                comments["pageInfo"] = {"hasNextPage": True, "endCursor": f"{thread['id']}:p0"}
            else:
                comments["pageInfo"] = {"hasNextPage": False, "endCursor": None}
        response["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"] = nodes
        key = f"graphql_threads:{number}" if number is not None else "graphql_threads"
        responses[key] = response
        self._responses_path.write_text(json.dumps(responses), encoding="utf-8")

    def _serve_thread_comments(self, thread_id: str, comment_nodes: list[dict[str, Any]], *, page_size: int) -> None:
        """Serve a deterministic nested-comment catalog through node queries, honoring
        requested page size and thread:pN cursors.
        """
        pages: list[dict[str, Any]] = []
        for start in range(0, len(comment_nodes), page_size):
            chunk = comment_nodes[start : start + page_size]
            page_index = start // page_size
            has_next = start + page_size < len(comment_nodes)
            pages.append({"data": {"node": {"comments": {"pageInfo": {"hasNextPage": has_next,
                                    "endCursor": f"{thread_id}:p{page_index}" if has_next else None,
                                }, "nodes": chunk,
                            }
                        }
                    }
                }
            )
        responses = self._read_responses()
        responses[f"graphql_thread_comments:{thread_id}"] = pages
        self._responses_path.write_text(json.dumps(responses), encoding="utf-8")

    def _read_responses(self) -> dict[str, Any]:
        return _read_responses(self.state_dir)


def _shim_main(state_dir: Path) -> int:
    """Run the fake handler behind the executable PATH boundary."""
    argv = sys.argv[1:]
    _record(state_dir, {"kind": "gh process", "cwd": str(Path.cwd().resolve()), "argv": ["gh", *argv]},)
    rc, stdout, stderr = _handle_gh(argv, sys.stdin.read(), state_dir)
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    return rc


def block_real_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail instead of executing the real ``gh`` CLI, keeping a suite hermetic."""
    real_run = subprocess.run

    def guarded_run(args: list[Any], *pargs: Any, **kwargs: Any) -> Any:
        if args and args[0] == "gh":
            raise AssertionError("test attempted to execute the real gh CLI")
        return real_run(args, *pargs, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)


def install_fake_gh(state_dir: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGh:
    """Route sync ``gh`` in process and async ``gh`` through a PATH shim."""
    state_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = state_dir / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "gh"
    source_root = Path(__file__).resolve().parents[2]
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"sys.path.insert(0, {str(source_root)!r})\n"
        "from tests.harness.fake_gh import _shim_main\n"
        f"raise SystemExit(_shim_main(pathlib.Path({str(state_dir)!r})))\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join((str(bin_dir), os.environ["PATH"])))
    real_run = subprocess.run

    def router(args: Any, *pargs: Any, **kwargs: Any) -> Any:
        if isinstance(args, (list, tuple)) and args and args[0] == "gh":
            cwd = Path(kwargs.get("cwd") or Path.cwd()).resolve()
            _record(state_dir, {"kind": "gh process", "cwd": str(cwd), "argv": list(args)},)
            rc, out, err = _handle_gh(list(args[1:]), kwargs.get("input") or "", state_dir)
            return subprocess.CompletedProcess(list(args), rc, stdout=out, stderr=err)
        if (isinstance(args, (list, tuple)) and args and args[0] == "git" and "ls-remote" in args):
            # Named remotes use real Git. URL requests must carry the
            # command-scoped credential helper before receiving fake refs.
            target = args[args.index("ls-remote") + 1] if args.index("ls-remote") + 1 < len(args) else ""
            if "://" not in target and "@" not in target:
                return real_run(args, *pargs, **kwargs)
            if not any("credential.helper=" in a for a in args):
                return subprocess.CompletedProcess(list(args), 1, stdout="",
                    stderr="fake gh: git ls-remote without a command-scoped credential helper\n",
                )
            refs = _read_responses(state_dir).get("git-ls-remote", _LS_REMOTE_DEFAULT)
            _record(state_dir, {"kind": "git ls-remote", "argv": list(args), "env": kwargs.get("env")},)
            return subprocess.CompletedProcess(list(args), 0, stdout=refs, stderr="")
        return real_run(args, *pargs, **kwargs)

    monkeypatch.setattr("daydream.git_ops.process.subprocess.run", router)
    return FakeGh(state_dir)
