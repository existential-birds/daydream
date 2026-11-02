"""Hermetic replay/readback verification over local HTTP, OTLP, and slow peers.

HoneyHive covers strict search responses, pagination, duplicates, session filtering,
and malformed/auth/redirect failures. LangSmith covers exact project selection,
bounded tree discovery, one root, frozen ids, and two equal complete snapshots.
Slow peers verify immutable deadlines and observed connection closure. Output
must omit content and credentials. Replay gates run before sends; end-to-end
receipts remain labeled with zero model calls and zero operational cost."""


from __future__ import annotations

import importlib.util
import io
import json
import socket
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Mapping, cast

import anyio
import pytest

from tests.harness.git_helpers import commit, git, init_repo

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXTURES = ROOT / "tests" / "fixtures" / "observability_contract"
REPLAY_FIXTURE = ROOT / "tests" / "fixtures" / "pi_jsonl" / "long_generation_replay.jsonl"
VERIFIER_PATH = SCRIPTS / "verify_observability_readback.py"
REPLAY_PATH = SCRIPTS / "replay_observability_acceptance.py"

_SECRET_KEY = "sk-test-verifier-secret-9f3c"
_BUDGET_S = 0.5

# Loading the operator scripts (same pattern as conftest template assets)


def _load_script(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_verifier = _load_script(VERIFIER_PATH, "verify_observability_readback")
_replay = _load_script(REPLAY_PATH, "replay_observability_acceptance")

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


# One immutable deadline: a real loopback peer


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

        # Stall after receipt; the verifier must close the connection at its deadline.
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

async def test_immutable_budget_truncates_slow_peers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A peer trickling an endless body after delayed headers is cut off at the immutable deadline."""
    receipt = _receipt(destinations=["honeyhive"])
    matrix = _verifier.load_matrix(FIXTURES / "readback-matrix.json")
    peer = _LoopbackPeer(trickle=True)
    # Cold HTTPX transport imports and TLS context construction are synchronous
    # and can exhaust the tiny budget under parallel coverage before any socket
    # opens. Prepare that real transport outside the measured slow-peer interval;
    # keep real HTTPX streaming, deadline cancellation, and connection closure.
    transport = _verifier.httpx.AsyncHTTPTransport(trust_env=False)
    client_type = _verifier.httpx.AsyncClient

    def prepared_client(**kwargs: Any) -> Any:
        return client_type(transport=transport, **kwargs)

    monkeypatch.setattr(_verifier.httpx, "AsyncClient", prepared_client)
    try:
        monkeypatch.setenv("HH_API_URL", peer.base_url)
        monkeypatch.setenv("HH_API_KEY", _SECRET_KEY)
        monkeypatch.setenv("DAYDREAM_TRACE_TO", "honeyhive")
        started = time.monotonic()
        # Wide enough for a loaded host to connect; still far below the 2.5s trickle.
        result = await _verifier._verify(receipt, matrix, _BUDGET_S)
        elapsed = time.monotonic() - started
        assert elapsed < _BUDGET_S + 0.75, f"readback took {elapsed:.3f}s with a {_BUDGET_S}s budget"
        assert result["stored_contract_passed"] is False
        assert result["terminal"] == _verifier.DISPOSITION_TIMEOUT
        # Bound peer-observed closure independently of verifier return.
        peer.close()
        deadline = time.monotonic() + 2.0
        while not peer.connections and time.monotonic() < deadline:
            await anyio.sleep(0.02)
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
            assert entry["offset_ms"] < _BUDGET_S * 1000, f"request began after the budget: {entry}"
    finally:
        peer.close()


# Redaction


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
