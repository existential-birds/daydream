"""Native review admission authority and bounded callbacks at their HTTP owner."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest

from daydream import host_tools as policy
from daydream.host_tools import NATIVE_SUBMIT, HostToolBinding, HostToolRejected
from tests.harness.fake_clock import FakeClock

_SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"verdict": {"type": "string"}, "findings": {"type": "array"}},
    "required": ["verdict", "findings"]}
_RESULT = {"verdict": "complete", "findings": []}


@asynccontextmanager
async def _endpoint(binding: HostToolBinding) -> AsyncIterator[httpx.AsyncClient]:
    binding.configure_schema(_SCHEMA)
    async with binding.session() as server:
        async with httpx.AsyncClient(base_url=server.url, headers={
            "Authorization": server.packet()["authorization"],
        }, timeout=3) as client:
            yield client


@pytest.mark.parametrize("fault", ["missing", "extra", "number", "array", "null"])
async def test_http_terminal_schema_rejection_does_not_call_acceptance(fault: str) -> None:
    """Schema-invalid callback arguments cannot cross the synchronous acceptance boundary.

    Installed-Pi validation rejects earlier, so it cannot prove the callback owner
    independently rejects malformed input before any admitted-state mutation.
    """
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    arguments: Any = {"missing": {}, "extra": {**_RESULT, "other": True}, "number": 42,
                      "array": [], "null": None}[fault]
    async with _endpoint(binding) as client:
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        assert (await client.post("/admit", json={"id": "invalid", "name": NATIVE_SUBMIT,
            "arguments": arguments, "terminal": True})).json()["allowed"]
        rejected = (await client.post("/submit", json={"id": "invalid", "arguments": arguments})).json()
        assert rejected["id"] == "invalid" and rejected["accepted"] is False
        assert "schema" in rejected["error"].lower()
        assert binding.authority_open and installed == [] and not binding.accepted
        assert (await client.post("/admit", json={"id": "corrected", "name": NATIVE_SUBMIT,
            "arguments": _RESULT, "terminal": True})).json()["allowed"]
        response = await client.post("/submit", json={"id": "corrected", "arguments": _RESULT})
        assert response.json() == {"id": "corrected", "accepted": True, "result": _RESULT}
    assert installed == [_RESULT] and binding.calls == 2 and binding.rejections == 1 and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["json", "frame", "items", "depth", "unauthorized",
                                  "unicode-argument", "unicode-key", "unicode-id", "argument-bytes"])
async def test_http_admission_bounds_reject_before_acceptance(
    monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Raw frames cannot bypass finite bounds, credentials or strict UTF-8 charging.

    Normal native JSON and schema checks cannot exercise malformed HTTP bodies or
    invalid Unicode; the callback endpoint owns these pre-mutation bounds.
    """
    monkeypatch.setattr(policy, "_MAX_ITEMS", 128)
    monkeypatch.setattr(policy, "_MAX_DEPTH", 16)
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    async with _endpoint(binding) as client:
        if fault == "json":
            response = await client.post("/admit", content=b"{broken")
        elif fault == "frame":
            response = await client.post("/admit", content=b"x" * (policy._MAX_FRAME_BYTES + 1))
        elif fault == "items":
            response = await client.post("/admit", json={"id": "items", "name": NATIVE_SUBMIT,
                "arguments": {"values": list(range(130))}, "terminal": True})
        elif fault == "depth":
            response = await client.post("/admit", content=(b'{"id":"depth","name":"structured_output",'
                b'"arguments":{"v":' + b"[" * 18 + b"0" + b"]" * 18 + b"}}"))
        elif fault == "argument-bytes":
            response = await client.post("/admit", json={"id": "large", "name": NATIVE_SUBMIT,
                "arguments": {"verdict": "x" * policy._MAX_ARGUMENT_BYTES, "findings": []}, "terminal": True})
        elif fault == "unauthorized":
            response = await client.post("/submit", json={"id": "unauthorized", "arguments": _RESULT},
                                         headers={"Authorization": "Bearer wrong"})
        else:
            packet: dict[str, Any] = {"id": "unicode", "name": NATIVE_SUBMIT,
                                      "arguments": _RESULT, "terminal": True}
            if fault == "unicode-argument":
                packet["arguments"] = {"verdict": "\ud800", "findings": []}
            elif fault == "unicode-key":
                packet["arguments"] = {"\ud800": True}
            else:
                packet["id"] = "\ud800"
            response = await client.post("/admit", content=json.dumps(packet).encode(),
                                         headers={"Content-Type": "application/json"})
        assert response.status_code == {"json": 400, "frame": 413, "items": 200, "depth": 200,
            "unauthorized": 403, "unicode-argument": 200, "unicode-key": 200, "unicode-id": 200,
            "argument-bytes": 200}[fault]
        assert len(response.content) <= policy._MAX_FRAME_BYTES
        if fault in {"items", "depth", "argument-bytes"} or fault.startswith("unicode-"):
            assert response.json()["allowed"] is False
        assert installed == [] and not binding.accepted
        assert binding.authority_open is (fault not in {"json", "frame", "unicode-id"})
    assert binding.calls == int(fault in {"items", "depth", "unicode-argument", "unicode-key", "argument-bytes"})
    assert binding.cleanup_complete


async def test_http_readiness_requires_matching_terminal_definition_before_submission() -> None:
    """A reachable endpoint cannot stand in for the active model-facing native schema.

    The installed success journey cannot detect an endpoint that accepts a terminal
    call before the extension confirms its exact active definition.
    """
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    async with _endpoint(binding) as client:
        assert not (await client.get("/status")).json()["ready"]
        response = await client.post("/ready", json={"tool_name": NATIVE_SUBMIT,
            "schema": {"type": "object"}})
        assert response.status_code == 400 and not (await client.get("/status")).json()["ready"]
        assert (await client.post("/admit", json={"id": "not-ready", "name": NATIVE_SUBMIT,
            "arguments": _RESULT, "terminal": True})).json()["allowed"]
        response = await client.post("/submit", json={"id": "not-ready", "arguments": _RESULT})
        assert response.json()["id"] == "not-ready" and response.json()["accepted"] is False
        assert "readiness" in response.json()["error"]
        assert installed == []
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        assert (await client.get("/status")).json()["ready"]
        response = await client.post("/submit", json={"id": "not-ready", "arguments": _RESULT})
        assert response.json() == {"id": "not-ready", "accepted": True, "result": _RESULT}
    assert installed == [_RESULT] and binding.calls == 1 and binding.cleanup_complete


async def test_native_rejections_spend_original_allowance_and_retry_does_not_renew_it() -> None:
    """Invalid terminal calls and retries share one allowance and original deadline.

    Native event observers charge after starts; this owner proof catches early
    callback authorization that would let a corrected call outrun that observer.
    """
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    binding.configure_schema(_SCHEMA)
    binding.restrict(deadline=None, tool_call_budget=2)
    deadline = binding.deadline
    async with binding.session():
        with pytest.raises(HostToolRejected, match="alone"):
            binding.admit("mixed", NATIVE_SUBMIT, _RESULT, terminal=False)
        with pytest.raises(HostToolRejected, match="already charged"):
            binding.admit("mixed", NATIVE_SUBMIT, _RESULT, terminal=True)
        binding.admit("ordinary", "read", {"path": "/source", "offset": 1, "limit": 2})
        assert binding.calls == 2
    binding.restrict(deadline=time.monotonic() + 100, tool_call_budget=99)
    assert binding.deadline == deadline
    async with binding.session():
        with pytest.raises(HostToolRejected, match="Original cumulative tool allowance"):
            binding.admit("corrected", NATIVE_SUBMIT, _RESULT, terminal=True)
        with pytest.raises(HostToolRejected, match="no matching"):
            binding.submit("corrected", _RESULT)
    assert binding.calls == 3 and installed == [] and binding.cleanup_complete


@pytest.mark.parametrize("order", ["closure-first", "deadline-first", "acceptance-first"])
async def test_acceptance_and_authority_closure_have_one_synchronous_order(
    monkeypatch: pytest.MonkeyPatch, order: str,
) -> None:
    """Closure/expiry deny mutation, while accepted work survives closure and duplicate callbacks.

    Native process success cannot prove which operation won this boundary; the
    host callback owner must enforce it independently of later native settlement.
    """
    fake = FakeClock(monotonic_value=100).install(monkeypatch)
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=110, accept=accept)
    binding.configure_schema(_SCHEMA)
    async with binding.session():
        binding.admit("terminal", NATIVE_SUBMIT, _RESULT, terminal=True)
        if order == "acceptance-first":
            assert binding.submit("terminal", _RESULT) == _RESULT
            binding.close_authority()
            fake.advance(10)
            assert binding.submit("terminal", _RESULT) == _RESULT
            with pytest.raises(HostToolRejected, match="closed"):
                binding.submit("different-terminal", _RESULT)
        else:
            if order == "closure-first":
                binding.close_authority()
            else:
                fake.advance(10)
            with pytest.raises(HostToolRejected, match="closed|expired"):
                binding.submit("terminal", _RESULT)
    assert installed == ([_RESULT] if order == "acceptance-first" else [])
    assert binding.accepted is (order == "acceptance-first") and binding.calls == 1
    assert not binding.authority_open and not binding.native_settled and binding.cleanup_complete


async def test_oversized_canonical_result_cannot_be_truncated_into_native_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected oversized callback result preserves uncertain acceptance and forbids replay.

    Source payload bounds run before admission; they do not protect a faulty
    trusted callback or prove the transport never presents a truncated success.
    """
    monkeypatch.setattr(policy, "_MAX_RESULT_BYTES", 512)
    calls: list[Any] = []
    def accept(value: Any) -> dict[str, Any]:
        calls.append(value)
        return {"result": "界" * 300}
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    binding.configure_schema(_SCHEMA)
    async with binding.session():
        binding.admit("terminal", NATIVE_SUBMIT, _RESULT, terminal=True)
        with pytest.raises(HostToolRejected, match="invalid canonical result"):
            binding.submit("terminal", _RESULT)
        assert binding.possibly_accepted and not binding.accepted and not binding.authority_open
        with pytest.raises(HostToolRejected, match="closed"):
            binding.submit("terminal", _RESULT)
    with pytest.raises(ValueError, match="another native attempt"):
        async with binding.session():
            pytest.fail("uncertain acceptance must not reopen")
    assert calls == [_RESULT] and binding.cleanup_complete


async def test_endpoint_cleanup_joins_incomplete_callback_frames_before_return() -> None:
    """Shutdown revokes authority and joins a real stalled HTTP request before evidence can freeze.

    Complete-frame and installed-Pi settlement tests do not exercise a peer that
    retains an incomplete body while the owning invocation closes.
    """
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    binding.configure_schema(_SCHEMA)
    async with binding.session() as server:
        endpoint = urlsplit(server.url)
        reader, writer = await asyncio.open_connection(endpoint.hostname, endpoint.port)
        writer.write((f"POST /submit HTTP/1.1\r\nHost: {endpoint.netloc}\r\n"
            f"Authorization: {server.packet()['authorization']}\r\n"
            "Content-Type: application/json\r\nContent-Length: 100\r\n\r\n{\"id\":").encode())
        await writer.drain()
        async with httpx.AsyncClient(base_url=server.url,
            headers={"Authorization": server.packet()["authorization"]}) as client:
            assert (await client.get("/status")).status_code == 200
    assert await asyncio.wait_for(reader.read(), 3) == b""
    writer.close()
    await writer.wait_closed()
    assert installed == [] and not binding.authority_open and binding.cleanup_complete and server.closed
