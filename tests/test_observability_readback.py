"""Hermetic replay/readback verification over local HTTP, OTLP, and slow peers.

HoneyHive covers strict search responses, pagination, duplicates, session filtering,
and malformed/auth/redirect failures. LangSmith covers exact project selection,
bounded tree discovery, one root, frozen ids, and two equal complete snapshots.
Slow peers verify immutable deadlines and observed connection closure. Output
must omit content and credentials. Replay gates run before sends; end-to-end
receipts remain labeled with zero model calls and zero operational cost."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import socket
import threading
import time
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, cast

import pytest

from tests.harness.git_helpers import commit, git, init_repo

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXTURES = ROOT / "tests" / "fixtures" / "observability_contract"
REPLAY_FIXTURE = ROOT / "tests" / "fixtures" / "pi_jsonl" / "long_generation_replay.jsonl"
VERIFIER_PATH = SCRIPTS / "verify_observability_readback.py"
REPLAY_PATH = SCRIPTS / "replay_observability_acceptance.py"

_SECRET_KEY = "sk-test-verifier-secret-9f3c"


# Loading the operator scripts (same pattern as conftest template assets)


def _load_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_verifier = _load_script(VERIFIER_PATH, "verify_observability_readback")
_replay = _load_script(REPLAY_PATH, "replay_observability_acceptance")


# Fake vendor HTTP server (scriptable JSON responses)


class FakeVendorServer:
    """Record loopback requests and dispatch (method,path) to status/headers/body responders."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responders: dict[tuple[str, str], Callable[[dict[str, Any]], tuple[int, dict[str, str], bytes]]] = {}
        self._lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            server: Any  # our ThreadingHTTPServer with attached state

            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b""
                record = {"method": self.command, "path": self.path,
                    "headers": {key.lower(): value for key, value in self.headers.items()}, "body": body,
                }
                with self.server._lock:
                    self.server.requests.append(record)
                # Route by path; the request record retains its query string.
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

    def respond(self, method: str, path: str, responder: Callable[[dict[str, Any]], tuple[int, dict[str, str], bytes]]
    ) -> None:
        self.responders[(method, path)] = responder

    def json_responder(self, payload: Any, *, status: int = 200
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


_OTLP_ACK: tuple[int, dict[str, str], bytes] = (200, {"Content-Type": "application/x-protobuf"}, b"")


@pytest.fixture
def fake_vendors(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[FakeVendorServer, FakeVendorServer]]:
    """Bind replay environment to paired fake vendors with canonical OTLP acknowledgment routes."""
    fake_hh = FakeVendorServer()
    fake_ls = FakeVendorServer()
    try:
        for fake in (fake_hh, fake_ls):
            for path in ("/opentelemetry/v1/traces", "/otel/v1/traces", "/v1/traces"):
                fake.respond("POST", path, lambda _r: _OTLP_ACK)
        monkeypatch.setenv("HH_API_URL", fake_hh.base_url)
        monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
        monkeypatch.setenv("LANGSMITH_ENDPOINT", fake_ls.base_url)
        monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET_KEY)
        monkeypatch.setenv("LANGSMITH_PROJECT", "daydream-test")
        monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp,honeyhive,langsmith")
        monkeypatch.setenv("DAYDREAM_ACCEPTANCE_KIND", "sanitized_protocol_replay")
        yield fake_hh, fake_ls
    finally:
        fake_hh.close()
        fake_ls.close()


# Receipt helpers


def _receipt(
    *, session_id: str = "7f3e9a2c-1111-4222-8333-444455556666", run_id: str = "bf50285e-72d2-4ce5-a9e6-bd913fecae15",
    kind: str = "representative_real_run", langsmith_project: str = "daydream-test",
    destinations: list[str] | None = None,
) -> dict[str, Any]:
    return {"schema_version": 1, "contract_version": "94f432d",
        "reviewed_commit": "023fc5b7ff8d44357aa41fccdf98db5a0455e396", "acceptance_kind": kind, "run_id": run_id,
        "session_id": session_id, "flow": "trace-field-mappings-test", "capture_mode": "full",
        "destinations": destinations or ["honeyhive", "langsmith"], "langsmith_project": langsmith_project,
        "started_at": "2026-09-07T00:00:00Z", "ended_at": "2026-09-07T00:01:00Z", "model_call_count": 0,
        "operational_cost_usd": 0,
    }


def _write_receipt(tmp_path: Path, **overrides: Any) -> Path:
    receipt = _receipt(**overrides)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _hh_event(event_id: str, session_id: str, *, event_type: str = "chain", **metadata: Any) -> dict[str, Any]:
    return {"id": event_id, "session_id": session_id, "event_name": "daydream.run", "event_type": event_type,
        "metadata": metadata, "inputs": {}, "outputs": {}, "config": {}, "metrics": {}, "feedback": [],
        "user_properties": {},
    }


def _ls_run(run_id: str, *, run_type: str = "chain", status: str = "success", trace_id: str = "trace-1",
    parent_run_id: Any = None, metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A vendor-actual run record: association metadata lives at ``extra.metadata``."""
    run: dict[str, Any] = {
        "id": run_id, "name": "daydream.run", "run_type": run_type, "status": status, "trace_id": trace_id,
        "start_time": "2026-09-07T00:00:00Z", "end_time": "2026-09-07T00:01:00Z", "parent_run_id": parent_run_id,
        "extra": {"metadata": dict(metadata or {})},
    }
    return run


def _ls_session(project: str) -> dict[str, Any]:
    """A minimal vendor session record for the project-name resolution lookup."""
    return {"id": "9e11a2de-1111-4222-8333-444455556666", "name": project, "start_time": "2026-09-07T00:00:00Z"}


def _sessions_ok(project: str) -> Callable[[Mapping[str, Any]], tuple[int, dict[str, str], bytes]]:
    """A LangSmith sessions responder resolving *project* to a single session."""

    def responder(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 200, {"Content-Type": "application/json"}, json.dumps([_ls_session(project)]).encode()

    return responder


def _empty_runs() -> tuple[int, dict[str, str], bytes]:
    """A LangSmith empty-runs response: an honest absence, not an error."""
    return 200, {"Content-Type": "application/json"}, b'{"runs": []}'


def _run_verify(receipt_path: Path, result_path: Path, *, budget_s: float = 30.0,
    matrix_path: Path = FIXTURES / "readback-matrix.json",
) -> int:
    value = _verifier.run_verify(receipt_path, result_path, budget_s=budget_s, matrix_path=matrix_path)
    return int(value)


def _configure_verifier_env(monkeypatch: pytest.MonkeyPatch, *, base_url: str) -> None:
    monkeypatch.setenv("HH_API_URL", base_url)
    monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
    monkeypatch.setenv("LANGSMITH_ENDPOINT", base_url)
    monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET_KEY)


def _verify_against(fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch, receipt: Path, tmp_path: Path, *,
    expect_zero: bool = True,
) -> Path:
    """Configure, run the verifier against ``fake_vendor``, and return the result path."""
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert (_run_verify(receipt, result_path) == 0) == expect_zero
    return result_path


# HoneyHive verifier behavior

def test_honeyhive_search_request_shape_and_pass(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])
    session_id = json.loads(receipt.read_text())["session_id"]
    run_id = json.loads(receipt.read_text())["run_id"]
    captured: list[dict[str, Any]] = []

    def search(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        captured.append(dict(record))
        body = {"events": [_hh_event("e1", session_id, **{"daydream.run.id": run_id}),
                _hh_event("e2", session_id, event_type="model", **{"daydream.run.id": run_id}),
                _hh_event("e3", session_id, event_type="tool", **{"daydream.run.id": run_id}),
            ], "count": 3,
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
        "filters": [{"field": "session_id", "operator": "is", "value": session_id, "type": "string"}], "limit": 100,
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
    # Full 100-row pages continue pagination; 250 rows require three pages.
    pages: dict[int, list[str]] = {1: [f"e{i:03d}" for i in range(100)], 2: [f"e{i:03d}" for i in range(100, 200)],
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
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path)
    result = json.loads(result_path.read_text())
    assert result["stored_contract_passed"] is True
    # Two stable complete snapshots: each full read is 3 pages (100+100+50).
    assert len([r for r in fake_vendor.requests if r["path"] == "/v1/events/search"]) == 6

@pytest.mark.parametrize(("body_builder", "disposition"),
    [(lambda sid: json.dumps({"events": [_hh_event("dup", sid), _hh_event("dup", sid)], "count": 2}).encode(),
            _verifier.DISPOSITION_DUPLICATE,
        ), (lambda sid: json.dumps({"events": [_hh_event("e1", "not-the-session")], "count": 1}).encode(),
            _verifier.DISPOSITION_WRONG_SESSION,
        ), (lambda sid: b'{"events": [', _verifier.DISPOSITION_MALFORMED),
        (lambda sid: json.dumps(
                {"events": [{"id": "e1", "session_id": "x", "blob": "z" * (5 * 1024 * 1024)}], "count": 1}
            ).encode(), _verifier.DISPOSITION_OVERSIZED,
        ),
    ],
)
def test_honeyhive_rejects_invalid_readbacks(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch,
    body_builder: Callable[[str], bytes], disposition: str,
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])
    session_id = json.loads(receipt.read_text())["session_id"]

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return 200, {"Content-Type": "application/json"}, body_builder(session_id)

    fake_vendor.respond("POST", "/v1/events/search", search)
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path, expect_zero=False)
    result = json.loads(result_path.read_text())
    assert result["terminal"] == disposition

@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
def test_honeyhive_status_errors_are_bounded(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return status, {"Content-Type": "application/json"}, b'{"secret": "' + _SECRET_KEY.encode() + b'"}'

    fake_vendor.respond("POST", "/v1/events/search", search)
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path, expect_zero=False)
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
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path, expect_zero=False)
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


# LangSmith verifier behavior

def test_langsmith_discovery_exact_filter_freeze_and_exact_id_reads(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["langsmith"])
    data = json.loads(receipt.read_text())
    run_id, project = data["run_id"], data["langsmith_project"]
    session = _ls_session(project)
    session_id = session["id"]
    discovered: list[dict[str, Any]] = []
    # Every run, including children, carries stored identity at extra.metadata.
    root = _ls_run(
        "root-1", run_type="chain", metadata={"daydream_run_id": run_id, "daydream.session.id": "daydream-session-1"},
    )
    child = _ls_run(
        "child-1", run_type="llm", parent_run_id="root-1", trace_id="trace-1", metadata={"daydream_run_id": run_id},
    )
    tool = _ls_run(
        "tool-1", run_type="tool", parent_run_id="root-1", trace_id="trace-1", metadata={"daydream_run_id": run_id},
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
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path)
    result = json.loads(result_path.read_text())
    assert result["stored_contract_passed"] is True
    resolution = discovered[0]
    assert resolution["path"].startswith("/api/v1/sessions")
    assert resolution["method"] == "GET"
    discovery = discovered[1]
    assert discovery["session"] == [session_id]
    assert discovery["filter"] == (f"and(eq(metadata_key, 'daydream_run_id'), eq(metadata_value, '{run_id}'))")
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

    def query(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        payload = json.loads(record["body"])
        if "id" in payload or payload.get("filter", "").startswith("eq(trace_id"):
            return _empty_runs()
        runs = [_ls_run("root-a", trace_id="trace-a"), _ls_run("root-b", trace_id="trace-b")]
        return 200, {"Content-Type": "application/json"}, json.dumps({"runs": runs}).encode()

    fake_vendor.respond("GET", "/api/v1/sessions", _sessions_ok(project))
    fake_vendor.respond("POST", "/runs/query", query)
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path, expect_zero=False)
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_AMBIGUOUS_ROOT

def test_langsmith_converged_third_snapshot_owns_returned_rows_count_and_hash(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["langsmith"])
    data = json.loads(receipt.read_text())
    run_id, project = data["run_id"], data["langsmith_project"]
    metadata = {"daydream_run_id": run_id}
    root = _ls_run("root-1", metadata=metadata)
    accepted = [
        root,
        _ls_run("child-1", run_type="llm", parent_run_id="root-1", metadata=metadata),
        _ls_run("tool-1", run_type="tool", parent_run_id="root-1", metadata=metadata),
    ]
    tree_reads = 0

    def query(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        nonlocal tree_reads
        payload = json.loads(record["body"])
        runs = accepted
        if "id" in payload:
            runs = [root]
        elif payload.get("filter", "").startswith("eq(trace_id"):
            tree_reads += 1
            # A/B/A: the second tree is incomplete, the third matches the first.
            runs = [root] if tree_reads == 2 else accepted
        return 200, {"Content-Type": "application/json"}, json.dumps({"runs": runs}).encode()

    fake_vendor.respond("GET", "/api/v1/sessions", _sessions_ok(project))
    fake_vendor.respond("POST", "/runs/query", query)
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path)
    result = json.loads(result_path.read_text())
    assert tree_reads == 3
    assert result["stored_contract_passed"] is True
    assert result["matrix_rows"] == []
    stored = result["destinations"]["langsmith"]
    assert stored["count"] == len(accepted)
    assert stored["snapshot_hash"] == _verifier.stable_hash([
        _verifier._run_identity(run) for run in accepted
    ])

def test_langsmith_unstable_tree_fails_closed(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["langsmith"])
    data = json.loads(receipt.read_text())
    run_id, project = data["run_id"], data["langsmith_project"]
    root = _ls_run("root-1", metadata={"daydream_run_id": run_id})
    calls = {"tree": 0}

    def query(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        payload = json.loads(record["body"])
        if "id" in payload:
            return 200, {"Content-Type": "application/json"}, json.dumps({"runs": [root]}).encode()
        if payload.get("filter", "").startswith("eq(trace_id"):
            # Changing IDs prevent two complete snapshots from agreeing.
            calls["tree"] += 1
            runs = [_ls_run(f"root-{calls['tree']}", metadata={"daydream_run_id": run_id})]
            return 200, {"Content-Type": "application/json"}, json.dumps({"runs": runs}).encode()
        return 200, {"Content-Type": "application/json"}, json.dumps({"runs": [root]}).encode()

    fake_vendor.respond("GET", "/api/v1/sessions", _sessions_ok(project))
    fake_vendor.respond("POST", "/runs/query", query)
    _configure_verifier_env(monkeypatch, base_url=fake_vendor.base_url)
    result_path = tmp_path / "result.json"
    assert _run_verify(receipt, result_path, budget_s=5.0) != 0
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_UNSTABLE

@pytest.mark.parametrize("destination", ["honeyhive", "langsmith"])
def test_third_snapshot_failure_retains_destination_disposition(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch, destination: str,
) -> None:
    receipt = _write_receipt(tmp_path, destinations=[destination])
    data = json.loads(receipt.read_text())
    root = _ls_run("root-1", metadata={"daydream_run_id": data["run_id"]})
    reads = 0

    def read(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        nonlocal reads
        payload = json.loads(record["body"])
        if destination == "langsmith" and not payload.get("filter", "").startswith("eq(trace_id"):
            return 200, {"Content-Type": "application/json"}, json.dumps({"runs": [root]}).encode()
        reads += 1
        if reads == 3:
            return 401, {"Content-Type": "application/json"}, b"{}"
        if destination == "honeyhive":
            result = {"events": [_hh_event(f"event-{reads}", data["session_id"])], "count": 1}
        else:
            result = {"runs": [_ls_run(f"root-{reads}", metadata={"daydream_run_id": data["run_id"]})]}
        return 200, {"Content-Type": "application/json"}, json.dumps(result).encode()

    if destination == "honeyhive":
        fake_vendor.respond("POST", "/v1/events/search", read)
    else:
        fake_vendor.respond("GET", "/api/v1/sessions", _sessions_ok(data["langsmith_project"]))
        fake_vendor.respond("POST", "/runs/query", read)
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path, expect_zero=False)
    result = json.loads(result_path.read_text())
    assert reads == 3
    expected = _verifier.DISPOSITION_AUTH if destination == "honeyhive" else _verifier.DISPOSITION_UNSTABLE
    assert result["terminal"] == expected


# One immutable deadline: real loopback peers


class _LoopbackPeer:
    """Record monotonic accept/close times so deadline tests can verify observed connection closure."""

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

        # Stall after receipt; the verifier must close the connection at its 0.12s deadline.
        try:
            conn.recv(65536)
        except OSError:
            pass
        if self.trickle:
            try:
                conn.sendall(
                    b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000000\r\n\r\n{"events":'
                )
                for _ in range(50):
                    conn.sendall(b"[]")
                    time.sleep(0.05)
                conn.sendall(b',"count":0}')
            except OSError:
                pass
        # Observe EOF/reset after delayed headers as proof the client closed at its deadline.
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
def test_immutable_budget_truncates_slow_peers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trickle: bool) -> None:
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
        # Bound peer-observed closure independently of verifier return.
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


# Redaction

def test_verifier_output_never_leaks_keys_endpoints_or_bodies(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])
    session_id = json.loads(receipt.read_text())["session_id"]

    def search(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        private = {"prompt": "private prompt", "response": "private response", "daydream.run.id": "run-1"}
        events = [_hh_event("e1", session_id, event_type="model", **private),
            _hh_event("e2", session_id, event_type="tool", **{"daydream.run.id": "run-1"}),
            _hh_event("e3", session_id, **{"daydream.run.id": "run-1"}),
        ]
        return 200, {"Content-Type": "application/json"}, json.dumps({"events": events, "count": 3}).encode()

    fake_vendor.respond("POST", "/v1/events/search", search)
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path)
    text = result_path.read_text()
    for forbidden in (_SECRET_KEY, "private prompt", "private response", fake_vendor.base_url):
        assert forbidden not in text
    result = json.loads(text)
    assert "matrix_rows" in result
    assert result["stored_contract_passed"] is True


# Replay tool: fail-closed gates run BEFORE any send


@pytest.fixture
def replay_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HH_API_URL", "http://127.0.0.1:9")  # unreachable: gates must fail before any send
    monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
    monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET_KEY)
    monkeypatch.setenv("LANGSMITH_PROJECT", "daydream-test")
    monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp,honeyhive,langsmith")
    monkeypatch.setenv("DAYDREAM_ACCEPTANCE_KIND", "sanitized_protocol_replay")


def _public_fixture_repo(tmp_path: Path, *, origin: str = "https://github.com/earendil-works/pi-coding-agent.git"
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


def _run_replay(repo: Path, fake_pi: Path, receipt_path: Path, *, fixture_path: Path = REPLAY_FIXTURE,) -> int:
    """Invoke the replay tool with the canonical fixture manifest."""
    return cast(int, _replay.run_replay(
        manifest_path=FIXTURES / "replay-manifest.json", fixture_path=fixture_path, repo_path=repo, fake_pi=fake_pi,
        receipt_path=receipt_path,
    ))


def _replay_receipt(tmp_path: Path, *, message: str) -> tuple[Path, dict[str, Any]]:
    """Run the hermetic replay; return ``(receipt_path, parsed_receipt)``."""
    repo = _public_fixture_repo(tmp_path)
    fk = _fake_pi_script(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    exit_code = _run_replay(repo, fk, receipt_path)
    assert exit_code == 0, message
    return receipt_path, json.loads(receipt_path.read_text())

def test_replay_gate_fixture_hash_mismatch_fails_before_send(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    replay_env: None,  # noqa: PLR0913,V107 - pytest fixture arg (side-effect env setup)
) -> None:
    receipt_path = tmp_path / "receipt.json"
    bad_fixture = tmp_path / "bad.jsonl"
    bad_fixture.write_bytes(REPLAY_FIXTURE.read_bytes() + b'{"type":"extra"}\n')
    fk = _fake_pi_script(tmp_path)
    repo = _public_fixture_repo(tmp_path)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _run_replay(repo, fk, receipt_path, fixture_path=bad_fixture)
    assert exit_code == 1
    assert not receipt_path.exists()
    assert "gate=identity" in buffer.getvalue()

def test_replay_gate_dirty_private_or_wrong_origin_repo_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    replay_env: None,  # noqa: PLR0913,V107 - pytest fixture arg (side-effect env setup)
) -> None:
    receipt_path = tmp_path / "receipt.json"
    fk = _fake_pi_script(tmp_path)
    repo = _public_fixture_repo(tmp_path, origin="https://github.com/private-org/private-repo.git")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _run_replay(repo, fk, receipt_path)
    assert exit_code == 1
    assert not receipt_path.exists()
    assert "allowlist" in buffer.getvalue()

    # Dirty work tree fails before send too.
    repo = _public_fixture_repo(tmp_path / "dirty-repo")
    (repo / "README.md").write_text("# dirty\n", encoding="utf-8")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _run_replay(repo, fk, receipt_path)
    assert exit_code == 1
    assert "dirty" in buffer.getvalue()

def test_replay_gate_real_pi_or_wrong_output_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    replay_env: None,  # noqa: PLR0913,V107 - pytest fixture arg (side-effect env setup)
) -> None:
    receipt_path = tmp_path / "receipt.json"
    repo = _public_fixture_repo(tmp_path)
    # A real-pi-shaped executable: no marker and output that is not the fixture.
    real_like = tmp_path / "real-pi"
    real_like.write_text("#!/bin/sh\necho 'pi: this is real output'\n", encoding="utf-8")
    real_like.chmod(0o755)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _run_replay(repo, real_like, receipt_path)
    assert exit_code == 1
    assert not receipt_path.exists()
    assert "marker" in buffer.getvalue() or "replay" in buffer.getvalue()

def test_replay_gate_wrong_destinations_or_missing_auth_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HH_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
    monkeypatch.setenv("LANGSMITH_API_KEY", _SECRET_KEY)
    monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp")  # wrong destinations
    monkeypatch.setenv("DAYDREAM_ACCEPTANCE_KIND", "sanitized_protocol_replay")
    receipt_path = tmp_path / "receipt.json"
    fk = _fake_pi_script(tmp_path)
    repo = _public_fixture_repo(tmp_path)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _run_replay(repo, fk, receipt_path)
    assert exit_code == 1
    assert "DAYDREAM_TRACE_TO" in buffer.getvalue()

    monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp,honeyhive,langsmith")
    monkeypatch.delenv("HH_API_KEY", raising=False)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _run_replay(repo, fk, receipt_path)
    assert exit_code == 1
    assert "HH_API_KEY" in buffer.getvalue()

@pytest.mark.parametrize("marker", [False, True])
def test_replay_fake_pi_marker_requirement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    replay_env: None,  # noqa - vulture: pytest fixture arg (side-effect env setup)
    marker: bool,  # noqa: PLR0913,V107 - pytest fixture arg (side-effect env setup)
) -> None:
    receipt_path = tmp_path / "receipt.json"
    repo = _public_fixture_repo(tmp_path)
    fk = _fake_pi_script(tmp_path, marker=marker)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = _run_replay(repo, fk, receipt_path)
    if not marker:
        assert exit_code == 1
        assert not receipt_path.exists()
        assert "marker" in buffer.getvalue()
    else:
        assert exit_code == 0
        receipt = json.loads(receipt_path.read_text())
        assert receipt["acceptance_kind"] == "sanitized_protocol_replay"


# Gate integration (reviewer card t_d50a1bbe): replay→verify chain + HH stability

def test_replay_receipt_is_accepted_by_verifier_validator(
    tmp_path: Path, fake_vendors: tuple[FakeVendorServer, FakeVendorServer]
) -> None:
    _receipt_path, receipt = _replay_receipt(tmp_path, message="replay must produce its receipt before validation")
    # Must not raise: the replay receipt is the verifier's canonical input.
    _verifier.validate_receipt(receipt)

def test_honeyhive_requires_two_stable_complete_snapshots(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two unequal complete exact-session reads must fail closed as READBACK_UNSTABLE."""
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
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path, expect_zero=False)
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_UNSTABLE

def test_honeyhive_two_stable_complete_snapshots_pass(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _write_receipt(tmp_path, destinations=["honeyhive"])
    data = json.loads(receipt.read_text())
    session_id, run_id = data["session_id"], data["run_id"]

    def search(record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        page = json.loads(record["body"])["page"]
        if page == 1:
            events = [_hh_event("e1", session_id, **{"daydream.run.id": run_id}),
                _hh_event("e2", session_id, event_type="model", **{"daydream.run.id": run_id}),
                _hh_event("e3", session_id, event_type="tool", **{"daydream.run.id": run_id}),
            ]
        else:
            events = []
        return 200, {"Content-Type": "application/json"}, json.dumps({"events": events, "count": 3}).encode()

    fake_vendor.respond("POST", "/v1/events/search", search)
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path)
    result = json.loads(result_path.read_text())
    assert result["stored_contract_passed"] is True
    # First complete read + second complete read of the same single page.
    assert len([r for r in fake_vendor.requests if r["path"] == "/v1/events/search"]) == 2

def test_replay_reconcile_accepts_second_generation_distinct_timing_and_missing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the first replay generation is bound to the manifest’s historical timing.

    Later generations remain self-consistent. Missing stored model evidence is allowed;
    a present wrong model must fail."""
    receipt = _write_receipt(tmp_path, destinations=[], kind="sanitized_protocol_replay")
    run_id = json.loads(receipt.read_text())["run_id"]
    # HH stores identity/timing in metadata and omits gen_ai.response.model on billed attempts.
    first = _hh_event("first", "29cb884b-712b-4de4-b478-3652932ff5dc", event_type="model",
        **{"daydream.run.id": run_id, "daydream.generation.native_started_at_unix_ms": 1788690314289,
            "daydream.generation.native_started_at_unix_ns": 1788690314289000000,
            "daydream.generation.sealed_end_unix_ns": 1788690709621000000,
            "daydream.generation.duration_ns": 395332000000,
        },
    )
    second = _hh_event("second", "29cb884b-712b-4de4-b478-3652932ff5dc", event_type="model",
        **{"daydream.run.id": run_id, "daydream.generation.native_started_at_unix_ms": 1788690314500,
            "daydream.generation.native_started_at_unix_ns": 1788690314500000000,
            "daydream.generation.sealed_end_unix_ns": 1788690709624202000,
            "daydream.generation.duration_ns": 395124202000,
        },
    )
    billed = _hh_event("billed", "29cb884b-712b-4de4-b478-3652932ff5dc", event_type="chain",
        **{"daydream.run.id": run_id, "gen_ai.usage.cost": 0.00402781},
    )
    data = {"honeyhive": {"rows": [billed, first, second]},
        # One identity-bearing LS row satisfies its destination requirement.
        "langsmith": {"run_id": run_id,
            "runs": [_ls_run("ls-run-1", run_type="chain",
                    metadata={"daydream_run_id": run_id, "daydream.session.id": "dd-session-1"},
                )
            ],
        },
    }
    matrix = json.loads((FIXTURES / "readback-matrix.json").read_text())
    rows = _verifier.compare_stored(data, matrix, acceptance_kind="sanitized_protocol_replay")
    assert rows == [], f"expected reconcile pass, got: {rows}"

    # A PRESENT wrong response model on the billed owner still fails.
    bad = _hh_event("billed-bad", "29cb884b-712b-4de4-b478-3652932ff5dc", event_type="chain",
        **{"daydream.run.id": run_id, "gen_ai.usage.cost": 0.00402781, "gen_ai.response.model": "some-other-model"},
    )
    data["honeyhive"] = {"rows": [bad, first, second]}
    rows = _verifier.compare_stored(data, matrix, acceptance_kind="sanitized_protocol_replay")
    assert any(r.get("field") == "honeyhive.rows[0].gen_ai.response.model" for r in rows)

def test_langsmith_discovery_empty_result_is_not_found_not_ambiguous(
    tmp_path: Path, fake_vendor: FakeVendorServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero discovered runs is an honest absence (vendor ingest window), never reported as root ambiguity."""
    receipt = _write_receipt(tmp_path, destinations=["langsmith"])
    project = json.loads(receipt.read_text())["langsmith_project"]

    def query(_record: Mapping[str, Any]) -> tuple[int, dict[str, str], bytes]:
        return _empty_runs()

    fake_vendor.respond("GET", "/api/v1/sessions", _sessions_ok(project))
    fake_vendor.respond("POST", "/runs/query", query)
    result_path = _verify_against(fake_vendor, monkeypatch, receipt, tmp_path, expect_zero=False)
    result = json.loads(result_path.read_text())
    assert result["terminal"] == _verifier.DISPOSITION_NOT_FOUND

def test_replay_then_verify_end_to_end_on_fake_vendors(
    tmp_path: Path, fake_vendors: tuple[FakeVendorServer, FakeVendorServer]
) -> None:
    fake_hh, fake_ls = fake_vendors
    receipt_path, receipt = _replay_receipt(tmp_path, message="replay must pass all gates before the verifier runs")
    run_id = receipt["run_id"]
    session_id = receipt["session_id"]

    # Derive both vendor stores from the actual receipt identity.
    hh_events = [_hh_event(f"ev-{i}", session_id) for i in range(2)]
    hh_payload = json.dumps({"events": hh_events, "count": len(hh_events)}).encode()
    fake_hh.respond("POST", "/v1/events/search", lambda _r: (200, {"Content-Type": "application/json"}, hh_payload),)
    trace_id = "11111111-2222-4333-8444-555566667777"
    ls_root = _ls_run("aaaaaaaa-1111-4222-8333-444455556666", run_type="chain", trace_id=trace_id, parent_run_id=None,
        metadata={"daydream_run_id": run_id},
    )
    ls_child = _ls_run(
        "bbbbbbbb-1111-4222-8333-444455556666", run_type="llm", trace_id=trace_id, parent_run_id=ls_root["id"],
        metadata={"daydream_run_id": run_id},
    )
    fake_ls.respond("GET", "/api/v1/sessions",
        lambda _r: (200, {"Content-Type": "application/json"},
            json.dumps([_ls_session(str(receipt["langsmith_project"]))]).encode(),
        ),
    )
    fake_ls.respond("POST", "/runs/query",
        lambda _r: (200, {"Content-Type": "application/json"},
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

def test_replay_full_hermetic_run_writes_labeled_receipt(
    tmp_path: Path, fake_vendors: tuple[FakeVendorServer, FakeVendorServer], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real PiBackend/run_agent/trace_run with a fake process emits a labeled zero-cost replay receipt."""
    monkeypatch.setenv("LANGSMITH_PROJECT", "daydream-replay-test")
    fake_hh, fake_ls = fake_vendors
    _receipt_path, receipt = _replay_receipt(
        tmp_path, message="hermetic replay should pass all gates and the local wire check"
    )
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
    # Replay must restore the host clock for subsequent worker tests.
    assert time.time_ns is _replay._stdlib_real_time_ns


def test_replay_receipt_retains_original_admitted_input_bytes(
    tmp_path: Path, fake_vendors: tuple[FakeVendorServer, FakeVendorServer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = tmp_path / "admitted.jsonl"
    fixture.write_bytes(REPLAY_FIXTURE.read_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes((FIXTURES / "replay-manifest.json").read_bytes())
    original_fixture = fixture.read_bytes()
    original_manifest = manifest.read_bytes()
    fake_pi = _fake_pi_script(tmp_path)
    probe = _replay._validate_fake_pi

    def replace_inputs_after_native_probe(*args: Any, **kwargs: Any) -> None:
        probe(*args, **kwargs)
        fixture.write_bytes(b"replacement fixture not admitted")
        manifest.write_bytes(b"replacement manifest not admitted")

    monkeypatch.setattr(_replay, "_validate_fake_pi", replace_inputs_after_native_probe)
    receipt_path = tmp_path / "receipt.json"
    repo = _public_fixture_repo(tmp_path)
    assert _replay.run_replay(
        manifest_path=manifest, fixture_path=fixture, repo_path=repo,
        fake_pi=fake_pi, receipt_path=receipt_path,
    ) == 0
    receipt = json.loads(receipt_path.read_text())
    assert receipt["fixture_sha256"] == hashlib.sha256(original_fixture).hexdigest()
    assert receipt["manifest_sha256"] == hashlib.sha256(original_manifest).hexdigest()
    assert fixture.read_bytes() == b"replacement fixture not admitted"
    assert manifest.read_bytes() == b"replacement manifest not admitted"
    assert receipt["model_call_count"] == receipt["operational_cost_usd"] == 0


@pytest.mark.parametrize("kind", ["bytes", "bool", "int", "wrong_string", "misbound_string"])
def test_replay_native_wire_marker_requires_exact_string_binding(kind: str) -> None:
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

    request = ExportTraceServiceRequest()
    resource = request.resource_spans.add()
    resource.scope_spans.add().spans.add(name="marker-admission", parent_span_id=b"parent00")
    marker = resource.resource.attributes.add(key="daydream.acceptance.kind")
    if kind == "bytes":
        marker.value.bytes_value = b"sanitized_protocol_replay"
    elif kind == "bool":
        marker.value.bool_value = True
    elif kind == "int":
        marker.value.int_value = 1
    else:
        marker.value.string_value = "wrong"
    if kind == "misbound_string":
        unrelated = resource.resource.attributes.add(key="unrelated")
        unrelated.value.string_value = "sanitized_protocol_replay"
    receiver = _replay._OtlpReceiver()
    try:
        receiver.batches.append(request.SerializeToString())
        spans, resources = receiver.decoded()
        with pytest.raises(_replay.ReplayValidationError, match="missing from exported resource"):
            _replay._require_local_wire_success(spans, resources)
    finally:
        receiver.stop()
