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

Any other invocation exits non-zero so unexpected calls surface as failures.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daydream.pr_review import finding_marker

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
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("--method", "-X"):
            method = argv[i + 1].upper()
            i += 2
        elif tok == "--input":
            payload = json.loads(Path(argv[i + 1]).read_text(encoding="utf-8"))
            i += 2
        elif tok in ("--jq", "-H", "--header"):
            i += 2
        elif tok.startswith("-"):
            i += 1
        else:
            endpoint = tok
            i += 1
    return method, (endpoint or "").lstrip("/"), payload


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] != "api":
        sys.stderr.write("fake gh: unsupported invocation: %r\\n" % (argv,))
        return 1
    method, endpoint, payload = _parse(argv[1:])
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
        print(json.dumps(responses[key]))
        return 0
    if method == "GET" and re.fullmatch(r"repos/[^/]+/[^/]+/pulls/\\d+/reviews", endpoint):
        print(json.dumps([]))
        return 0
    if method == "POST" and re.fullmatch(r"repos/[^/]+/[^/]+/pulls/\\d+/reviews", endpoint):
        print(json.dumps({"html_url": "https://github.test/fake/pull/7#pullrequestreview-1"}))
        return 0
    sys.stderr.write("fake gh: no canned response for %s\\n" % key)
    return 1


if __name__ == "__main__":
    sys.exit(main())
'''


@dataclass
class GhCall:
    """One recorded ``gh`` invocation.

    Attributes:
        endpoint: Endpoint with any leading slash stripped (``graphql`` for
            GraphQL calls).
        payload: Parsed ``--input`` JSON payload, or None.
    """

    endpoint: str
    payload: Any


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
            if record["method"] != method.upper():
                continue
            if wanted_endpoint is not None and record["endpoint"] != wanted_endpoint:
                continue
            out.append(GhCall(endpoint=record["endpoint"], payload=record["payload"]))
        return out

    # --- canned-response configuration ---------------------------------------

    def set_response(self, method: str, endpoint: str, value: Any) -> None:
        """Configure the canned response for ``method endpoint``."""
        responses = self._read_responses()
        responses[f"{method.upper()} {endpoint.lstrip('/')}"] = value
        self._responses_path.write_text(json.dumps(responses), encoding="utf-8")

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
    shim.write_text(_SHIM_SOURCE, encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return FakeGh(bin_dir)
