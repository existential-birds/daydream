"""Hermetic tests for the P18 Task 6 readback verifier and replay tool.

Plan Task 6 steps 1-2 (P18-plan.md SHA-256
``1bb962866288395f6ddf0504fd8affd9764c019cbe024c8c61140f2f5b6a9738`` plus the
readback-deadline amendment). All tests run against a fake external HTTP
boundary on loopback, real loopback slow peers, local OTLP collectors and fake
vendor HTTP; no real vendor, model call, credential or repository content is
ever touched.

Coverage:
- HoneyHive ``POST /v1/events/search`` request shape, strict ``{events,count}``
  validation, pagination, duplicates, wrong-session rows, auth/redirect/HTTP
  errors, malformed and oversized responses;
- LangSmith project-session resolution (exact project-name lookup), tree
  discovery (``and(eq(metadata_key, 'daydream_run_id'),
  eq(metadata_value, …))`` in one explicit session with a bounded start
  time), single-root requirement, exact-ID freeze and re-read, two equal
  complete tree snapshots (``filter: eq(trace_id, …)``);
- separate real loopback peers delaying response headers or trickling JSON
  body bytes cannot extend a 0.12-second immutable verifier budget past 0.35
  seconds; each peer observes connection closure within one second; the
  verifier emits only its fixed redacted timeout disposition and its request
  log proves no later page or stability poll began;
- output redaction: never prompts/reasoning/tool/result content, headers,
  keys, full endpoints or exception response text;
- replay tool fail-closed gates (fixture/hash, dirty/private repo, real pi
  executable, wrong destinations, missing authorization) all run BEFORE any
  send, and the full hermetic replay writes the labeled receipt with
  model-call count 0 and operational cost 0.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import pytest

from tests.harness.git_helpers import commit, git, init_repo

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXTURES = ROOT / "tests" / "fixtures" / "observability_contract"
REPLAY_FIXTURE = ROOT / "tests" / "fixtures" / "pi_jsonl" / "long_generation_replay.jsonl"
VERIFIER_PATH = SCRIPTS / "verify_observability_readback.py"
REPLAY_PATH = SCRIPTS / "replay_observability_acceptance.py"

_SECRET_KEY = "sk-test-verifier-secret-9f3c"
_HH_URL = "http://127.0.0.1"  # replaced by the fake server's real base


# ---------------------------------------------------------------------------
# Loading the operator scripts (same pattern as conftest template assets)
# ---------------------------------------------------------------------------


def _load_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_verifier = _load_script(VERIFIER_PATH, "verify_observability_readback")
_replay = _load_script(REPLAY_PATH, "replay_observability_acceptance")


# ---------------------------------------------------------------------------
# Fake vendor HTTP server (scriptable JSON responses)
# ---------------------------------------------------------------------------


class FakeVendorServer:
    """Loopback HTTP server faking HoneyHive and/or LangSmith JSON endpoints.

    ``responders`` maps ``(method, path)`` to a callable returning
    ``(status, headers, body_bytes)``; every request is recorded.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responders: dict[tuple[str, str], Callable[[dict[str, Any]], tuple[int, dict[str, str], bytes]]] = {}
        self._lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            server: Any  # our ThreadingHTTPServer with attached state

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b""
                record = {
                    "method": self.command,
                    "path": self.path,
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": body,
                }
                with self.server._lock:
                    self.server.requests.append(record)
                # Route on the path only (query strings are transport detail);
                # responders still see the full raw path in the record.
                route_path = self.path.split("?", 1)[0]
                responder = self.server.responders.get((self.command, route_path))
                if responder is None:
                    status, headers, response = 404, {}, b'{"error":"no responder"}'
                else:
                    try:
                        status, headers, response = responder(record)
                    except Exception as exc:  # noqa: BLE001 - test harness
                        status, headers, response = 500, {}, str(exc).encode()
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def do_GET(self) -> None:  # noqa: N802
                self._handle()

            def do_POST(self) -> None:  # noqa: N802
                self._handle()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.requests = self.requests  # type: ignore[attr-defined]
        self._server.responders = self.responders  # type: ignore[attr-defined]
        self._server._lock = self._lock  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host = str(self._server.server_address[0])
        port = int(self._server.server_address[1])
        return f"http://{host}:{port}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def respond(
        self, method: str, path: str, responder: Callable[[dict[str, Any]], tuple[int, dict[str, str], bytes]]
    ) -> None:
        self.responders[(method, path)] = responder

    def json_responder(
        self, payload: Any, *, status: int = 200
    ) -> Callable[[dict[str, Any]], tuple[int, dict[str, str], bytes]]:
        body = json.dumps(payload).encode("utf-8")

        def responder(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
            return status, {"Content-Type": "application/json"}, body

        return responder

    def hang_responder(self) -> Callable[[dict[str, Any]], tuple[int, dict[str, str], bytes]]:
        def responder(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
            raise AssertionError("hang responder should never be called")

        return responder


@pytest.fixture
def fake_vendor() -> Iterator[FakeVendorServer]:
    server = FakeVendorServer()
    try:
        yield server
    finally:
        server.close()


# ---------------------------------------------------------------------------
# Receipt helpers
# ---------------------------------------------------------------------------


def _receipt(
    *,
    session_id: str = "7f3e9a2c-1111-4222-8333-444455556666",
    run_id: str = "bf50285e-72d2-4ce5-a9e6-bd913fecae15",
    kind: str = "representative_real_run",
    langsmith_project: str = "daydream-test",
    destinations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_version": "94f432d",
        "reviewed_commit": "023fc5b7ff8d44357aa41fccdf98db5a0455e396",
        "acceptance_kind": kind,
        "run_id": run_id,
        "session_id": session_id,
        "flow": "trace-field-mappings-test",
        "capture_mode": "full",
        "destinations": destinations or ["honeyhive", "langsmith"],
        "langsmith_project": langsmith_project,
        "started_at": "2026-09-07T00:00:00Z",
        "ended_at": "2026-09-07T00:01:00Z",
        "model_call_count": 0,
        "operational_cost_usd": 0,
    }


def _write_receipt(tmp_path: Path, **overrides: Any) -> Path:
    receipt = _receipt(**overrides)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _hh_event(event_id: str, session_id: str, *, event_type: str = "chain", **metadata: Any) -> dict[str, Any]:
    return {
        "id": event_id,
        "session_id": session_id,
        "event_name": "daydream.run",
        "event_type": event_type,
        "metadata": metadata,
        "inputs": {},
        "outputs": {},
        "config": {},
        "metrics": {},
        "feedback": [],
        "user_properties": {},
    }


def _ls_run(
    run_id: str,
    *,
    run_type: str = "chain",
    status: str = "success",
    trace_id: str = "trace-1",
    parent_run_id: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A vendor-actual run record: association metadata lives at ``extra.metadata``."""
    run: dict[str, Any] = {
        "id": run_id,
        "name": "daydream.run",
        "run_type": run_type,
        "status": status,
        "trace_id": trace_id,
        "start_time": "2026-09-07T00:00:00Z",
        "end_time": "2026-09-07T00:01:00Z",
        "parent_run_id": parent_run_id,
        "extra": {"metadata": dict(metadata or {})},
    }
    return run


def _ls_session(project: str) -> dict[str, Any]:
    """A minimal vendor session record for the project-name resolution lookup."""
    return {
        "id": "9e11a2de-1111-4222-8333-444455556666",
        "name": project,
        "start_time": "2026-09-07T00:00:00Z",
    }


def _run_verify(
    receipt_path: Path,
    result_path: Path,
    *,
    budget_s: float = 30.0,
    matrix_path: Path = FIXTURES / "readback-matrix.json",
) -> int:
    value = _verifier.run_verify(receipt_path, result_path, budget_s=budget_s, matrix_path=matrix_path)
    return int(value)


def _configure_verifier_env(monkeypatch: pytest.MonkeyPatch, *, base_url: str) -> None:
    monkeypatch.setenv("HH_API_URL", base_url)
    monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
    monkeypatch.setenv("LANGSMITH_ENDPOINT", base_url)
    monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET_KEY)


# ---------------------------------------------------------------------------
# HoneyHive verifier behavior
# ---------------------------------------------------------------------------


def test_honeyhive_search_request_shape_and_pass(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])
    session_id = json.loads(receipt.read_text())["session_id"]
    run_id = json.loads(receipt.read_text())["run_id"]
    captured: list[dict[str, Any]] = []

    def search(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        captured.append(dict(record))
        body = {
            "events": [
                _hh_event("e1", session_id, **{"daydream.run.id": run_id}),
                _hh_event("e2", session_id, event_type="model", **{"daydream.run.id": run_id}),
                _hh_event("e3", session_id, event_type="tool", **{"daydream.run.id": run_id}),
            ],
            "count": 3,
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(body).encode()

    fake_vendor.respond("POST", "/v1/events/search", search)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    exit_code = _run_verify(receipt, result_path)

    assert exit_code == 0
    # Two stable complete snapshots: page 1 is requested on both full reads.
    assert len(captured) == 2
    sent = captured[0]
    assert sent["method"] == "POST"
    assert sent["path"] == "/v1/events/search"
    assert sent["headers"]["authorization"] == f"Bearer {_SECRET_KEY}"
    payload = json.loads(sent["body"])
    assert payload == {
        "filters": [{"field": "session_id", "operator": "is", "value": session_id, "type": "string"}],
        "limit": 100,
        "page": 1,
    }
    result = json.loads(result_path.read_text())
    assert result["stored_contract_passed"] is True
    assert result["terminal"] == _verifier.DISPOSITION_PASS
    assert result["ui_inspected"] is False
    # Two stable complete snapshots are required: one full read + one repeat.
    assert len(captured) == 2


def test_honeyhive_paginates_until_count_satisfied(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])
    session_id = json.loads(receipt.read_text())["session_id"]
    # Page limit is 100; a full page means "more available", a short page ends
    # pagination. 250 rows = 3 pages (100 + 100 + 50) and must reconcile.
    pages: dict[int, list[str]] = {
        1: [f"e{i:03d}" for i in range(100)],
        2: [f"e{i:03d}" for i in range(100, 200)],
        3: [f"e{i:03d}" for i in range(200, 250)],
    }

    def search(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        page = json.loads(record["body"])["page"]
        events = [_hh_event(event_id, session_id) for event_id in pages.get(page, [])]
        # Representative-agent-tree minimums: one model and one tool event.
        if page == 3:
            events[-2] = _hh_event("e-tool-001", session_id, event_type="tool")
            events[-1] = _hh_event("e-model-001", session_id, event_type="model")
        return 200, {"Content-Type": "application/json"}, json.dumps({"events": events, "count": 250}).encode()

    fake_vendor.respond("POST", "/v1/events/search", search)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) == 0
    result = json.loads(result_path.read_text())
    assert result["stored_contract_passed"] is True
    # Two stable complete snapshots: each full read is 3 pages (100+100+50).
    assert len([r for r in fake_vendor.requests if r["path"] == "/v1/events/search"]) == 6


def test_honeyhive_rejects_duplicate_event_ids(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])
    session_id = json.loads(receipt.read_text())["session_id"]

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        events = [_hh_event("dup", session_id), _hh_event("dup", session_id)]
        return 200, {"Content-Type": "application/json"}, json.dumps({"events": events, "count": 2}).encode()

    fake_vendor.respond("POST", "/v1/events/search", search)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) != 0
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_DUPLICATE


def test_honeyhive_rejects_wrong_session_rows(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        events = [_hh_event("e1", "not-the-session")]
        return 200, {"Content-Type": "application/json"}, json.dumps({"events": events, "count": 1}).encode()

    fake_vendor.respond("POST", "/v1/events/search", search)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) != 0
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_WRONG_SESSION


def test_honeyhive_rejects_malformed_json(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 200, {"Content-Type": "application/json"}, b'{"events": ['

    fake_vendor.respond("POST", "/v1/events/search", search)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) != 0
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_MALFORMED


def test_honeyhive_rejects_oversized_response(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        big = {"events": [{"id": "e1", "session_id": "x", "blob": "z" * (5 * 1024 * 1024)}], "count": 1}
        return 200, {"Content-Type": "application/json"}, json.dumps(big).encode()

    fake_vendor.respond("POST", "/v1/events/search", search)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) != 0
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_OVERSIZED


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
def test_honeyhive_status_errors_are_bounded(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return status, {"Content-Type": "application/json"}, b'{"secret": "' + _SECRET_KEY.encode() + b'"}'

    fake_vendor.respond("POST", "/v1/events/search", search)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) != 0
    result = json.loads(result_path.read_text())
    if status == 404:
        assert result["terminal"] == _verifier.DISPOSITION_NOT_FOUND
    elif status in (401, 403):
        assert result["terminal"] == _verifier.DISPOSITION_AUTH
    else:
        assert result["terminal"] in (_verifier.DISPOSITION_HTTP_ERROR, _verifier.DISPOSITION_AUTH)
    # Bounded: the response body (with the secret) never reaches any output.
    assert _SECRET_KEY not in result_path.read_text()


def test_honeyhive_redirect_is_rejected_without_following(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 302, {"Location": "/elsewhere"}, b""

    fake_vendor.respond("POST", "/v1/events/search", search)
    fake_vendor.respond("POST", "/elsewhere", fake_vendor.json_responder({"events": [], "count": 0}))
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) != 0
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_REDIRECT
    assert not [r for r in fake_vendor.requests if r["path"] == "/elsewhere"]


def test_receipt_validation_fails_before_any_client(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"acceptance_kind": "sanitized_protocol_replay"}), encoding="utf-8")
    fake_vendor.respond("POST", "/v1/events/search", fake_vendor.hang_responder())
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    exit_code = _run_verify(receipt_path, result_path)
    assert exit_code == 1
    assert fake_vendor.requests == []
    assert json.loads(result_path.read_text())["terminal"] == _verifier.DISPOSITION_SHAPE


# ---------------------------------------------------------------------------
# LangSmith verifier behavior
# ---------------------------------------------------------------------------


def test_langsmith_discovery_exact_filter_freeze_and_exact_id_reads(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["langsmith"])
    data = json.loads(receipt.read_text())
    run_id, project = data["run_id"], data["langsmith_project"]
    session = _ls_session(project)
    session_id = session["id"]
    discovered: list[dict[str, Any]] = []
    # Every run in the tree carries the stored identity (vendor-actual:
    # association properties at extra.metadata), including the child.
    root = _ls_run(
        "root-1",
        run_type="chain",
        metadata={"daydream_run_id": run_id, "daydream.session.id": "daydream-session-1"},
    )
    child = _ls_run(
        "child-1",
        run_type="llm",
        parent_run_id="root-1",
        trace_id="trace-1",
        metadata={"daydream_run_id": run_id},
    )
    tool = _ls_run(
        "tool-1",
        run_type="tool",
        parent_run_id="root-1",
        trace_id="trace-1",
        metadata={"daydream_run_id": run_id},
    )

    def sessions(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        discovered.append(dict(record))
        return 200, {"Content-Type": "application/json"}, json.dumps([session]).encode()

    def query(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        payload = json.loads(record["body"])
        discovered.append(payload)
        if "id" in payload:
            runs = [root] if payload["id"] == ["root-1"] else []
        elif payload.get("filter", "").startswith("eq(trace_id"):
            runs = [root, child, tool]
        else:
            runs = [root, child, tool]
        return 200, {"Content-Type": "application/json"}, json.dumps({"runs": runs}).encode()

    fake_vendor.respond("GET", "/api/v1/sessions", sessions)
    fake_vendor.respond("POST", "/runs/query", query)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) == 0
    result = json.loads(result_path.read_text())
    assert result["stored_contract_passed"] is True
    resolution = discovered[0]
    assert resolution["path"].startswith("/api/v1/sessions")
    assert resolution["method"] == "GET"
    discovery = discovered[1]
    assert discovery["session"] == [session_id]
    assert discovery["filter"] == (
        f"and(eq(metadata_key, 'daydream_run_id'), eq(metadata_value, '{run_id}'))"
    )
    assert discovery.get("start_time") == data["started_at"]
    # Exact-ID reads: one for the frozen root id, two stable tree snapshots.
    ops = [d for d in discovered if "id" in d]
    assert ops and ops[0]["id"] == ["root-1"]
    trees = [d for d in discovered if d.get("filter", "").startswith("eq(trace_id")]
    assert len(trees) == 2
    ls_section = result["destinations"]["langsmith"]
    assert ls_section["root_id"] == "root-1" and ls_section["trace_id"] == "trace-1"


def test_langsmith_ambiguous_root_rejected(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["langsmith"])
    project = json.loads(receipt.read_text())["langsmith_project"]

    def sessions(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 200, {"Content-Type": "application/json"}, json.dumps([_ls_session(project)]).encode()

    def query(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        payload = json.loads(record["body"])
        if "id" in payload or payload.get("filter", "").startswith("eq(trace_id"):
            return 200, {"Content-Type": "application/json"}, b'{"runs": []}'
        runs = [_ls_run("root-a", trace_id="trace-a"), _ls_run("root-b", trace_id="trace-b")]
        return 200, {"Content-Type": "application/json"}, json.dumps({"runs": runs}).encode()

    fake_vendor.respond("GET", "/api/v1/sessions", sessions)
    fake_vendor.respond("POST", "/runs/query", query)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) != 0
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_AMBIGUOUS_ROOT


def test_langsmith_unstable_tree_fails_closed(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["langsmith"])
    data = json.loads(receipt.read_text())
    run_id, project = data["run_id"], data["langsmith_project"]
    root = _ls_run("root-1", metadata={"daydream_run_id": run_id})
    calls = {"tree": 0}

    def sessions(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 200, {"Content-Type": "application/json"}, json.dumps([_ls_session(project)]).encode()

    def query(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        payload = json.loads(record["body"])
        if "id" in payload:
            return 200, {"Content-Type": "application/json"}, json.dumps({"runs": [root]}).encode()
        if payload.get("filter", "").startswith("eq(trace_id"):
            # Every snapshot carries a different run id: no two complete
            # snapshots can ever agree, so stability must fail closed.
            calls["tree"] += 1
            runs = [_ls_run(f"root-{calls['tree']}", metadata={"daydream_run_id": run_id})]
            return 200, {"Content-Type": "application/json"}, json.dumps({"runs": runs}).encode()
        return 200, {"Content-Type": "application/json"}, json.dumps({"runs": [root]}).encode()

    fake_vendor.respond("GET", "/api/v1/sessions", sessions)
    fake_vendor.respond("POST", "/runs/query", query)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path, budget_s=5.0) != 0
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_UNSTABLE


# ---------------------------------------------------------------------------
# One immutable deadline: real loopback peers
# ---------------------------------------------------------------------------


class _LoopbackPeer:
    """A real socket peer that bounds the verifier's deadline behavior.

    Records ``(accepted_at, closed_at)`` per connection using monotonic
    seconds; the tests assert the peer observed closure within one second of
    accept ("peer observes connection closure within one second").
    """

    def __init__(self, *, trickle: bool = False) -> None:
        self.trickle = trickle
        self.connections: list[tuple[float, float]] = []
        self._lock = threading.Lock()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self.base_url = f"http://127.0.0.1:{self._listener.getsockname()[1]}"
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self) -> None:
        self._listener.settimeout(60)
        while True:
            try:
                conn, _addr = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        accepted_at = time.monotonic()
        conn.settimeout(30)

        # The client request arrives first; then we stall (never completing a
        # valid JSON body inside the 0.12s budget). The verifier must close
        # the connection itself when its immutable deadline fires.
        try:
            conn.recv(65536)
        except OSError:
            pass
        if self.trickle:
            try:
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Content-Length: 1000000\r\n\r\n{\"events\":"
                )
                for _ in range(50):
                    conn.sendall(b"[]")
                    time.sleep(0.05)
                conn.sendall(b",\"count\":0}")
            except OSError:
                pass
        # Delay headers indefinitely in the plain case; then observe client
        # closure (EOF/reset) — this is the peer-close proof. recv() returns
        # b"" as soon as the verifier closes the stream at its deadline.
        try:
            while conn.recv(4096):
                pass
        except OSError:
            pass
        closed_at = time.monotonic()
        with self._lock:
            self.connections.append((accepted_at, closed_at))
        try:
            conn.close()
        except OSError:
            pass

    def close(self) -> None:
        try:
            self._listener.close()
        except OSError:
            pass


@pytest.mark.parametrize("trickle", [False, True])
def test_immutable_budget_truncates_slow_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trickle: bool
) -> None:
    """0.12s budget: return before 0.35s, peer observes close within 1s, no later poll."""
    receipt = _write_receipt(tmp_path)
    peer = _LoopbackPeer(trickle=trickle)
    try:
        monkeypatch.setenv("HH_API_URL", peer.base_url)
        monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
        monkeypatch.setenv("DAYDREAM_TRACE_TO", "honeyhive")
        receipt_data = json.loads(receipt.read_text())
        receipt_data["destinations"] = ["honeyhive"]
        receipt.write_text(json.dumps(receipt_data), encoding="utf-8")
        result_path = tmp_path / "result.json"
        started = time.monotonic()
        exit_code = _run_verify(receipt, result_path, budget_s=0.12)
        elapsed = time.monotonic() - started
        assert elapsed < 0.35, f"verifier took {elapsed:.3f}s with a 0.12s budget"
        assert exit_code != 0
        result = json.loads(result_path.read_text())
        assert result["terminal"] == _verifier.DISPOSITION_TIMEOUT
        # The peer observed connection closure within one second of accept
        # (the verifier closes the stream when its immutable deadline fires).
        peer.close()
        deadline = time.monotonic() + 2.0
        while not peer.connections and time.monotonic() < deadline:
            time.sleep(0.02)
        assert peer.connections, "peer never accepted a connection"
        accepted_at, closed_at = peer.connections[0]
        assert closed_at - accepted_at < 1.0, (
            f"peer observed closure {closed_at - accepted_at:.3f}s after accept (must be < 1s)"
        )
        # Request log proves no later page or stability poll began.
        ops = [entry["op"] for entry in result["request_log"]]
        assert ops == ["search"], f"unexpected request log: {ops}"
        assert all(entry["op"] != "poll" for entry in result["request_log"])
        # Every logged request began strictly before the immutable deadline.
        for entry in result["request_log"]:
            assert entry["offset_ms"] < 120, f"request began after the 0.12s budget: {entry}"
    finally:
        peer.close()


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_verifier_output_never_leaks_keys_endpoints_or_bodies(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])
    session_id = json.loads(receipt.read_text())["session_id"]

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        private = {"prompt": "private prompt", "response": "private response", "daydream.run.id": "run-1"}
        events = [
            _hh_event("e1", session_id, event_type="model", **private),
            _hh_event("e2", session_id, event_type="tool", **{"daydream.run.id": "run-1"}),
            _hh_event("e3", session_id, **{"daydream.run.id": "run-1"}),
        ]
        return 200, {"Content-Type": "application/json"}, json.dumps({"events": events, "count": 3}).encode()

    fake_vendor.respond("POST", "/v1/events/search", search)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) == 0
    text = result_path.read_text()
    for forbidden in (_SECRET_KEY, "private prompt", "private response", fake_vendor.base_url):
        assert forbidden not in text
    result = json.loads(text)
    assert "matrix_rows" in result
    assert result["stored_contract_passed"] is True


# ---------------------------------------------------------------------------
# Replay tool: fail-closed gates run BEFORE any send
# ---------------------------------------------------------------------------


@pytest.fixture
def replay_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HH_API_URL", "http://127.0.0.1:9")  # unreachable: gates must fail before any send
    monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
    monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET_KEY)
    monkeypatch.setenv("LANGSMITH_PROJECT", "daydream-test")
    monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp,honeyhive,langsmith")
    monkeypatch.setenv("DAYDREAM_ACCEPTANCE_KIND", "sanitized_protocol_replay")


def _public_fixture_repo(
    tmp_path: Path, *, origin: str = "https://github.com/earendil-works/pi-coding-agent.git"
) -> Path:
    repo = tmp_path / "public-repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "README.md").write_text("# Public fixture repository\n", encoding="utf-8")
    init_repo(repo)
    git(repo, "add", "README.md")
    commit(repo, "initial fixture commit")
    git(repo, "remote", "add", "origin", origin)
    return repo


def _fake_pi_script(tmp_path: Path, *, marker: bool = True, fixture_path: Path | None = None) -> Path:
    script = tmp_path / "fake-pi"
    lines = ["#!/bin/sh"]
    if marker:
        lines.append("# daydream-sanitized-protocol-replay-fake-pi")
    lines.append(f'cat "{fixture_path or REPLAY_FIXTURE}"')
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def test_replay_gate_fixture_hash_mismatch_fails_before_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay_env: None  # noqa: PLR0913,V107 - pytest fixture arg (side-effect env setup)
) -> None:
    receipt_path = tmp_path / "receipt.json"
    bad_fixture = tmp_path / "bad.jsonl"
    bad_fixture.write_bytes(REPLAY_FIXTURE.read_bytes() + b'{"type":"extra"}\n')
    fk = _fake_pi_script(tmp_path)
    repo = _public_fixture_repo(tmp_path)
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _replay.run_replay(
            manifest_path=FIXTURES / "replay-manifest.json",
            fixture_path=bad_fixture,
            repo_path=repo,
            fake_pi=fk,
            receipt_path=receipt_path,
        )
    assert exit_code == 1
    assert not receipt_path.exists()
    assert "gate=identity" in buffer.getvalue()


def test_replay_gate_dirty_private_or_wrong_origin_repo_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay_env: None  # noqa: PLR0913,V107 - pytest fixture arg (side-effect env setup)
) -> None:
    receipt_path = tmp_path / "receipt.json"
    fk = _fake_pi_script(tmp_path)
    repo = _public_fixture_repo(tmp_path, origin="https://github.com/private-org/private-repo.git")
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _replay.run_replay(
            manifest_path=FIXTURES / "replay-manifest.json",
            fixture_path=REPLAY_FIXTURE,
            repo_path=repo,
            fake_pi=fk,
            receipt_path=receipt_path,
        )
    assert exit_code == 1
    assert not receipt_path.exists()
    assert "allowlist" in buffer.getvalue()

    # Dirty work tree fails before send too.
    repo = _public_fixture_repo(tmp_path / "dirty-repo")
    (repo / "README.md").write_text("# dirty\n", encoding="utf-8")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _replay.run_replay(
            manifest_path=FIXTURES / "replay-manifest.json",
            fixture_path=REPLAY_FIXTURE,
            repo_path=repo,
            fake_pi=fk,
            receipt_path=receipt_path,
        )
    assert exit_code == 1
    assert "dirty" in buffer.getvalue()


def test_replay_gate_real_pi_or_wrong_output_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay_env: None  # noqa: PLR0913,V107 - pytest fixture arg (side-effect env setup)
) -> None:
    receipt_path = tmp_path / "receipt.json"
    repo = _public_fixture_repo(tmp_path)
    # A real-pi-shaped executable: no marker and output that is not the fixture.
    real_like = tmp_path / "real-pi"
    real_like.write_text("#!/bin/sh\necho 'pi: this is real output'\n", encoding="utf-8")
    real_like.chmod(0o755)
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _replay.run_replay(
            manifest_path=FIXTURES / "replay-manifest.json",
            fixture_path=REPLAY_FIXTURE,
            repo_path=repo,
            fake_pi=real_like,
            receipt_path=receipt_path,
        )
    assert exit_code == 1
    assert not receipt_path.exists()
    assert "marker" in buffer.getvalue() or "replay" in buffer.getvalue()


def test_replay_gate_wrong_destinations_or_missing_auth_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HH_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
    monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET_KEY)
    monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp")  # wrong destinations
    monkeypatch.setenv("DAYDREAM_ACCEPTANCE_KIND", "sanitized_protocol_replay")
    receipt_path = tmp_path / "receipt.json"
    fk = _fake_pi_script(tmp_path)
    repo = _public_fixture_repo(tmp_path)
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _replay.run_replay(
            manifest_path=FIXTURES / "replay-manifest.json",
            fixture_path=REPLAY_FIXTURE,
            repo_path=repo,
            fake_pi=fk,
            receipt_path=receipt_path,
        )
    assert exit_code == 1
    assert "DAYDREAM_TRACE_TO" in buffer.getvalue()

    monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp,honeyhive,langsmith")
    monkeypatch.delenv("HH_API_KEY", raising=False)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _replay.run_replay(
            manifest_path=FIXTURES / "replay-manifest.json",
            fixture_path=REPLAY_FIXTURE,
            repo_path=repo,
            fake_pi=fk,
            receipt_path=receipt_path,
        )
    assert exit_code == 1
    assert "HH_API_KEY" in buffer.getvalue()


@pytest.mark.parametrize("marker", [False, True])
def test_replay_fake_pi_marker_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay_env: None, marker: bool  # noqa: PLR0913,V107 - pytest fixture arg (side-effect env setup)
) -> None:
    receipt_path = tmp_path / "receipt.json"
    repo = _public_fixture_repo(tmp_path)
    fk = _fake_pi_script(tmp_path, marker=marker)
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _replay.run_replay(
            manifest_path=FIXTURES / "replay-manifest.json",
            fixture_path=REPLAY_FIXTURE,
            repo_path=repo,
            fake_pi=fk,
            receipt_path=receipt_path,
        )
    if not marker:
        assert exit_code == 1
        assert not receipt_path.exists()
        assert "marker" in buffer.getvalue()
    else:
        assert exit_code == 0
        receipt = json.loads(receipt_path.read_text())
        assert receipt["acceptance_kind"] == "sanitized_protocol_replay"


# ---------------------------------------------------------------------------
# Gate integration (reviewer card t_d50a1bbe): replay→verify chain + HH stability
# ---------------------------------------------------------------------------


def test_replay_receipt_is_accepted_by_verifier_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The replay tool's receipt must validate under the verifier's schema.

    The two operator scripts are one pipeline: the replay tool writes the
    immutable receipt the verifier consumes. A receipt the verifier rejects
    (missing the canonical ``destinations`` list) breaks that pipeline before
    any network work.
    """
    fake_hh = FakeVendorServer()
    fake_ls = FakeVendorServer()
    try:
        for fake in (fake_hh, fake_ls):
            fake.respond(
                "POST",
                "/opentelemetry/v1/traces",
                lambda _r: (200, {"Content-Type": "application/x-protobuf"}, b""),
            )
            fake.respond("POST", "/v1/traces", lambda _r: (200, {"Content-Type": "application/x-protobuf"}, b""))
        monkeypatch.setenv("HH_API_URL", fake_hh.base_url)
        monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", fake_ls.base_url)
        monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET_KEY)
        monkeypatch.setenv("LANGSMITH_PROJECT", "daydream-test")
        monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp,honeyhive,langsmith")
        monkeypatch.setenv("DAYDREAM_ACCEPTANCE_KIND", "sanitized_protocol_replay")
        repo = _public_fixture_repo(tmp_path)
        fk = _fake_pi_script(tmp_path)
        receipt_path = tmp_path / "receipt.json"
        exit_code = _replay.run_replay(
            manifest_path=FIXTURES / "replay-manifest.json",
            fixture_path=REPLAY_FIXTURE,
            repo_path=repo,
            fake_pi=fk,
            receipt_path=receipt_path,
        )
        assert exit_code == 0, "replay must produce its receipt before validation"
        receipt = json.loads(receipt_path.read_text())
        # Must not raise: the replay receipt is the verifier's canonical input.
        _verifier.validate_receipt(receipt)
    finally:
        fake_hh.close()
        fake_ls.close()


def test_honeyhive_requires_two_stable_complete_snapshots(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HoneyHive must reach two equal complete exact-session snapshots.

    The card/plan gate is 'two stable complete post-shutdown snapshots' for
    HoneyHive exactly as for LangSmith. A vendor whose second full read
    disagrees must fail closed with READBACK_UNSTABLE.
    """
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])
    session_id = json.loads(receipt.read_text())["session_id"]
    calls = {"n": 0}
    lock = threading.Lock()

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        with lock:
            calls["n"] += 1
            which = 1 if calls["n"] == 1 else 2
        # Same count, different ids on the second full read: unstable vendor.
        ids = ["e1", "e2"] if which == 1 else ["e1", "e3"]
        events = [_hh_event(event_id, session_id) for event_id in ids]
        return 200, {"Content-Type": "application/json"}, json.dumps({"events": events, "count": 2}).encode()

    fake_vendor.respond("POST", "/v1/events/search", search)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) != 0
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_UNSTABLE


def test_honeyhive_two_stable_complete_snapshots_pass(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stable HoneyHive session passes only after two complete equal reads."""
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])
    data = json.loads(receipt.read_text())
    session_id, run_id = data["session_id"], data["run_id"]

    def search(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        page = json.loads(record["body"])["page"]
        if page == 1:
            events = [
                _hh_event("e1", session_id, **{"daydream.run.id": run_id}),
                _hh_event("e2", session_id, event_type="model", **{"daydream.run.id": run_id}),
                _hh_event("e3", session_id, event_type="tool", **{"daydream.run.id": run_id}),
            ]
        else:
            events = []
        return 200, {"Content-Type": "application/json"}, json.dumps({"events": events, "count": 3}).encode()

    fake_vendor.respond("POST", "/v1/events/search", search)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) == 0
    result = json.loads(result_path.read_text())
    assert result["stored_contract_passed"] is True
    # First complete read + second complete read of the same single page.
    assert len([r for r in fake_vendor.requests if r["path"] == "/v1/events/search"]) == 2


def test_replay_reconcile_accepts_second_generation_distinct_timing_and_missing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vendor-actual replay rows: only the pinned FIRST generation is exact.

    The sanitized fixture has two generations with distinct native/sealed
    timings; the manifest pins the first's exact historical interval. A second
    generation row must be self-consistent but never compared to the first's
    pins, and a billed owner whose vendor storage lacks the response-model key
    (HoneyHive stores the configured request model at config.model instead)
    must not fail on absence — only a present wrong value fails.
    """
    receipt = _write_receipt(tmp_path, destinations=[], kind="sanitized_protocol_replay")
    run_id = json.loads(receipt.read_text())["run_id"]
    # Vendor-actual: HH rows put identity+timing in metadata; the billed
    # attempt row carries usage but HH does not project gen_ai.response.model.
    first = _hh_event(
        "first",
        "29cb884b-712b-4de4-b478-3652932ff5dc",
        event_type="model",
        **{
            "daydream.run.id": run_id,
            "daydream.generation.native_started_at_unix_ms": 1788690314289,
            "daydream.generation.native_started_at_unix_ns": 1788690314289000000,
            "daydream.generation.sealed_end_unix_ns": 1788690709621000000,
            "daydream.generation.duration_ns": 395332000000,
        },
    )
    second = _hh_event(
        "second",
        "29cb884b-712b-4de4-b478-3652932ff5dc",
        event_type="model",
        **{
            "daydream.run.id": run_id,
            "daydream.generation.native_started_at_unix_ms": 1788690314500,
            "daydream.generation.native_started_at_unix_ns": 1788690314500000000,
            "daydream.generation.sealed_end_unix_ns": 1788690709624202000,
            "daydream.generation.duration_ns": 395124202000,
        },
    )
    billed = _hh_event(
        "billed",
        "29cb884b-712b-4de4-b478-3652932ff5dc",
        event_type="chain",
        **{"daydream.run.id": run_id, "gen_ai.usage.cost": 0.00402781},
    )
    data = {
        "honeyhive": {"rows": [billed, first, second]},
        # One identity-bearing LS run so the reconcile's per-destination row
        # requirement is met without turning this into an LS-shape test.
        "langsmith": {
            "run_id": run_id,
            "runs": [
                _ls_run(
                    "ls-run-1",
                    run_type="chain",
                    metadata={"daydream_run_id": run_id, "daydream.session.id": "dd-session-1"},
                )
            ]
        },
    }
    matrix = json.loads((FIXTURES / "readback-matrix.json").read_text())
    rows = _verifier.compare_stored(data, matrix, acceptance_kind="sanitized_protocol_replay")
    assert rows == [], f"expected reconcile pass, got: {rows}"

    # A PRESENT wrong response model on the billed owner still fails.
    bad = _hh_event(
        "billed-bad",
        "29cb884b-712b-4de4-b478-3652932ff5dc",
        event_type="chain",
        **{"daydream.run.id": run_id, "gen_ai.usage.cost": 0.00402781,
           "gen_ai.response.model": "some-other-model"},
    )
    data["honeyhive"] = {"rows": [bad, first, second]}
    rows = _verifier.compare_stored(data, matrix, acceptance_kind="sanitized_protocol_replay")
    assert any(r.get("field") == "honeyhive.rows[0].gen_ai.response.model" for r in rows)


def test_langsmith_discovery_empty_result_is_not_found_not_ambiguous(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero discovered runs is an honest absence (vendor ingest window),
    never reported as root ambiguity.
    """
    receipt = _write_receipt(tmp_path, destinations=["langsmith"])
    project = json.loads(receipt.read_text())["langsmith_project"]

    def sessions(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 200, {"Content-Type": "application/json"}, json.dumps([_ls_session(project)]).encode()

    def query(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 200, {"Content-Type": "application/json"}, b'{"runs": []}'

    fake_vendor.respond("GET", "/api/v1/sessions", sessions)
    fake_vendor.respond("POST", "/runs/query", query)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path) != 0
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_NOT_FOUND


def test_replay_then_verify_end_to_end_on_fake_vendors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full operator chain: replay tool writes the receipt the verifier accepts.

    The replay runs hermetically (fake vendors, loopback OTLP oracle, fake pi
    executable); the verifier then consumes the replay's receipt against the
    same fake vendor endpoints and must pass the stored contract for both
    destinations without any manual receipt editing.
    """
    fake_hh = FakeVendorServer()
    fake_ls = FakeVendorServer()
    try:
        for fake in (fake_hh, fake_ls):
            fake.respond(
                "POST",
                "/opentelemetry/v1/traces",
                lambda _r: (200, {"Content-Type": "application/x-protobuf"}, b""),
            )
            fake.respond("POST", "/otel/v1/traces", lambda _r: (200, {"Content-Type": "application/x-protobuf"}, b""))
            fake.respond("POST", "/v1/traces", lambda _r: (200, {"Content-Type": "application/x-protobuf"}, b""))
        monkeypatch.setenv("HH_API_URL", fake_hh.base_url)
        monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", fake_ls.base_url)
        monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET_KEY)
        monkeypatch.setenv("LANGSMITH_PROJECT", "daydream-test")
        monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp,honeyhive,langsmith")
        monkeypatch.setenv("DAYDREAM_ACCEPTANCE_KIND", "sanitized_protocol_replay")
        repo = _public_fixture_repo(tmp_path)
        fk = _fake_pi_script(tmp_path)
        receipt_path = tmp_path / "receipt.json"
        exit_code = _replay.run_replay(
            manifest_path=FIXTURES / "replay-manifest.json",
            fixture_path=REPLAY_FIXTURE,
            repo_path=repo,
            fake_pi=fk,
            receipt_path=receipt_path,
        )
        assert exit_code == 0, "replay must pass all gates before the verifier runs"
        receipt = json.loads(receipt_path.read_text())
        run_id = receipt["run_id"]
        session_id = receipt["session_id"]

        # Vendor readback stores derived from the actual receipt identity:
        # one HoneyHive session (stable across reads) and one LangSmith tree.
        hh_events = [_hh_event(f"ev-{i}", session_id) for i in range(2)]
        hh_payload = json.dumps({"events": hh_events, "count": len(hh_events)}).encode()
        fake_hh.respond(
            "POST",
            "/v1/events/search",
            lambda _r: (200, {"Content-Type": "application/json"}, hh_payload),
        )
        trace_id = "11111111-2222-4333-8444-555566667777"
        ls_root = _ls_run(
            "aaaaaaaa-1111-4222-8333-444455556666",
            run_type="chain",
            trace_id=trace_id,
            parent_run_id=None,
            metadata={"daydream_run_id": run_id},
        )
        ls_child = _ls_run(
            "bbbbbbbb-1111-4222-8333-444455556666",
            run_type="llm",
            trace_id=trace_id,
            parent_run_id=ls_root["id"],
            metadata={"daydream_run_id": run_id},
        )
        fake_ls.respond(
            "GET",
            "/api/v1/sessions",
            lambda _r: (
                200,
                {"Content-Type": "application/json"},
                json.dumps([_ls_session(str(receipt["langsmith_project"]))]).encode(),
            ),
        )
        fake_ls.respond(
            "POST",
            "/runs/query",
            lambda _r: (
                200,
                {"Content-Type": "application/json"},
                json.dumps({"runs": [json.loads(json.dumps(ls_root)), json.loads(json.dumps(ls_child))]}).encode(),
            ),
        )

        # The verifier must accept the replay receipt verbatim.
        result_path = tmp_path / "readback-result.json"
        assert _run_verify(receipt_path, result_path) == 0
        result = json.loads(result_path.read_text())
        assert result["stored_contract_passed"] is True
        assert set(result["destinations"]) == {"honeyhive", "langsmith"}
        assert result["destinations"]["honeyhive"]["disposition"] == _verifier.DISPOSITION_PASS
        assert result["destinations"]["langsmith"]["disposition"] == _verifier.DISPOSITION_PASS
        # The HoneyHive session was read twice completely (two stable snapshots).
        assert len([r for r in fake_hh.requests if r["path"] == "/v1/events/search"]) >= 2
    finally:
        fake_hh.close()
        fake_ls.close()



def test_replay_full_hermetic_run_writes_labeled_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The complete sanitized replay through real PiBackend/run_agent/trace_run.

    Vendor destinations are fake loopback OTLP/HTTP endpoints; the local
    generic OTLP destination is the tool's own loopback receiver; the pi
    subprocess boundary is the fake executable. Receipt must be labeled
    ``sanitized_protocol_replay`` with model calls 0 and operational cost 0.
    """

    fake_hh = FakeVendorServer()
    fake_ls = FakeVendorServer()
    try:
        # The owned vendors expect HTTP 200 + protobuf content type + empty
        # body as the canonical full-success ack (binding decision 8).
        for fake in (fake_hh, fake_ls):
            fake.respond(
                "POST",
                "/opentelemetry/v1/traces",
                lambda _r: (200, {"Content-Type": "application/x-protobuf"}, b""),
            )
            fake.respond(
                "POST",
                "/otel/v1/traces",
                lambda _r: (200, {"Content-Type": "application/x-protobuf"}, b""),
            )
            fake.respond("POST", "/v1/traces", lambda _r: (200, {"Content-Type": "application/x-protobuf"}, b""))
        monkeypatch.setenv("HH_API_URL", fake_hh.base_url)
        monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", fake_ls.base_url)
        monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET_KEY)
        monkeypatch.setenv("LANGSMITH_PROJECT", "daydream-replay-test")
        monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp,honeyhive,langsmith")
        monkeypatch.setenv("DAYDREAM_ACCEPTANCE_KIND", "sanitized_protocol_replay")
        repo = _public_fixture_repo(tmp_path)
        fk = _fake_pi_script(tmp_path)
        receipt_path = tmp_path / "receipt.json"
        exit_code = _replay.run_replay(
            manifest_path=FIXTURES / "replay-manifest.json",
            fixture_path=REPLAY_FIXTURE,
            repo_path=repo,
            fake_pi=fk,
            receipt_path=receipt_path,
        )
        assert exit_code == 0, "hermetic replay should pass all gates and the local wire check"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["acceptance_kind"] == "sanitized_protocol_replay"
        assert receipt["model_call_count"] == 0
        assert receipt["operational_cost_usd"] == 0
        assert receipt["fixture_sha256"] == hashlib.sha256(REPLAY_FIXTURE.read_bytes()).hexdigest()
        manifest_hash = hashlib.sha256((FIXTURES / "replay-manifest.json").read_bytes()).hexdigest()
        assert receipt["manifest_sha256"] == manifest_hash
        assert receipt["run_id"] and receipt["session_id"]
        assert receipt["local_otlp_root_span_id"]
        assert len(receipt["local_otlp_wire_sha256"]) >= 1
        assert receipt["reported_cost_usd"] == 0.00402781
        assert receipt["labels"]["reported_cost"].startswith("synthetic")
        # Both vendor destinations must have been reached by the real exporters.
        assert fake_hh.requests and fake_ls.requests
    finally:
        fake_hh.close()
        fake_ls.close()
