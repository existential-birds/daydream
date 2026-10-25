"""Host authority and transport bounds at their real filesystem/process/HTTP owners."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from daydream import host_tools as policy
from daydream.host_tools import NATIVE_READ_FILE, HostToolBinding, HostToolRejected
from daydream.host_tools_server import PiHostToolServer
from tests.harness.host_tools import gated_reader


@asynccontextmanager
async def _endpoint(binding: HostToolBinding) -> AsyncIterator[tuple[PiHostToolServer, httpx.AsyncClient]]:
    async with binding.session() as server:
        async with httpx.AsyncClient(base_url=server.url, headers={
            "Authorization": server.packet()["authorization"], "Accept": "application/json, text/event-stream",
        }, timeout=3) as client:
            yield server, client


@pytest.mark.parametrize("fault", ["missing", "extra", "number", "array", "null", "params-array", "unicode-id"])
async def test_http_schema_rejections_precede_sdk_coercion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Raw schema/Unicode-ID errors stay exact and bounded; native validation hides SDK coercion/escaping."""
    path = tmp_path.resolve() / "allowed"
    path.write_text("secret bytes")
    monkeypatch.setattr(policy, "_MAX_FRAME_BYTES", 512)
    binding = HostToolBinding((path,), deadline=time.monotonic() + 10)
    arguments: Any = {"missing": {}, "extra": {"path": str(path), "other": True}, "number": {"path": 42},
                      "array": [], "null": None, "params-array": {"path": str(path)}, "unicode-id": {}}[fault]
    async with _endpoint(binding) as (_, client):
        params: Any = [] if fault == "params-array" else {"name": "read_file", "arguments": arguments}
        call_id = "界" * 100 if fault == "unicode-id" else "malformed"
        frame = json.dumps({"jsonrpc": "2.0", "id": call_id, "method": "tools/call", "params": params},
                           ensure_ascii=False).encode()
        assert len(frame) < policy._MAX_FRAME_BYTES
        response = await client.post("/mcp", content=frame, headers={"Content-Type": "application/json"})
        assert len(response.content) <= policy._MAX_FRAME_BYTES
        assert response.status_code == 200 and response.json()["id"] == call_id
        assert response.json()["result"] == {
            "isError": True, "content": [{"type": "text", "text": "Invalid host read schema"}]}
    assert binding.calls == 0 and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["json", "frame", "items", "depth", "unauthorized", "http-capacity",
                                  "unicode-path", "unicode-key", "unicode-id"])
async def test_http_admission_bounds_reject_before_tool_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Raw frames cannot bypass limits, credentials or UTF-8 charging; ordinary native calls miss these inputs."""
    monkeypatch.setattr(policy, "_MAX_HTTP_REQUESTS", 2)
    binding = HostToolBinding((tmp_path.resolve() / "allowed",), deadline=time.monotonic() + 10)
    async with _endpoint(binding) as (_, client):
        if fault == "json":
            response = await client.post("/admit", content=b"{broken")
        elif fault == "frame":
            response = await client.post("/admit", content=b"x" * (policy._MAX_FRAME_BYTES + 1))
        elif fault == "items":
            response = await client.post("/admit", json={"values": list(range(130))})
        elif fault == "depth":
            response = await client.post("/admit", content=b'{"v":' + b"[" * 18 + b"0" + b"]" * 18 + b"}")
        elif fault.startswith("unicode-"):
            packet: dict[str, Any] = {"id": "unicode", "name": NATIVE_READ_FILE, "arguments": {"path": "unused"}}
            if fault == "unicode-path":
                packet["arguments"]["path"] = "\ud800"
            elif fault == "unicode-key":
                packet["arguments"] = {"\ud800": True}
            elif fault == "unicode-id":
                packet["id"] = "\ud800"
            response = await client.post("/admit", content=json.dumps(packet).encode(),
                                         headers={"Content-Type": "application/json"})
        elif fault == "unauthorized":
            response = await client.get("/status", headers={"Authorization": "Bearer wrong"})
        else:
            assert (await client.get("/status")).status_code == 200
            assert (await client.get("/status")).status_code == 200
            response = await client.get("/status")
        assert response.status_code == {
            "json": 400, "frame": 413, "items": 200, "depth": 200, "unauthorized": 403, "http-capacity": 429,
            "unicode-path": 200, "unicode-key": 200, "unicode-id": 200,
        }[fault]
        assert len(response.content) <= policy._MAX_FRAME_BYTES
        if fault in {"items", "depth"} or fault.startswith("unicode-"):
            assert response.json()["allowed"] is False
        assert binding.authority_open is (fault in {"unauthorized", "http-capacity", "unicode-path", "unicode-key"})
    assert binding.calls == int(fault in {"unicode-path", "unicode-key"}) and binding.cleanup_complete


def test_host_grant_rejects_double_root_existing_path(tmp_path: Path) -> None:
    source = tmp_path.resolve() / "source"
    source.write_text("existing granted bytes")
    ambiguous = Path("/" + str(source))
    with pytest.raises(ValueError, match="exactly one /"):
        HostToolBinding((ambiguous,), deadline=time.monotonic() + 10)


@pytest.mark.parametrize("fault", ["traversal", "final-symlink", "parent-symlink", "fifo", "binary",
                                  "file-overflow", "serialized-overflow"])
async def test_host_read_preserves_exact_grant_and_rejects_non_authoritative_results(
    tmp_path: Path, fault: str,
) -> None:
    """Real reads reject escapes, links and inexact results; the valid native journey misses filesystem faults."""
    root = tmp_path.resolve()
    real = root / "real"
    real.mkdir()
    source = real / "source"
    source.write_text("unavailable bytes")
    granted = source
    requested = str(source)
    if fault == "traversal":
        requested = str(real / "..") + "/real/source"
    elif fault == "final-symlink":
        granted = root / "link"
        granted.symlink_to(source)
        requested = str(granted)
    elif fault == "parent-symlink":
        link = root / "linkdir"
        link.symlink_to(real, target_is_directory=True)
        granted = link / "source"
        requested = str(granted)
    elif fault == "fifo":
        source.unlink()
        os.mkfifo(source)
    elif fault == "binary":
        source.write_bytes(b"\xff")
    elif fault == "file-overflow":
        source.write_bytes(b"x" * 13000)
    elif fault == "serialized-overflow":
        source.write_text("\n" * 7000)
    binding = HostToolBinding((granted,), deadline=time.monotonic() + 10)
    async with binding.session():
        binding.admit("read", NATIVE_READ_FILE, {"path": requested})
        with pytest.raises(HostToolRejected) as failure:
            await binding.read_file({"path": requested})
        message = str(failure.value)
        assert "not permitted" in message if fault == "traversal" else (
            "not UTF-8" in message if fault == "binary" else
            "serialized result byte limit" in message if fault == "serialized-overflow" else "unavailable" in message)
        assert "unavailable bytes" not in message
    assert binding.calls == 1 and binding.rejections == 1 and binding.cleanup_complete


async def test_host_admissions_are_once_only_bounded_and_never_renewed(tmp_path: Path) -> None:
    """Duplicate/oversized IDs spend once without renewal; ordinary native calls do not probe direct admission."""
    path = tmp_path.resolve() / "source"
    path.write_text("uncharged bytes")
    binding = HostToolBinding((path,), deadline=time.monotonic() + 10)
    binding.restrict(deadline=None, tool_call_budget=2)
    deadline = binding.deadline
    async with _endpoint(binding) as (_, client):
        packet = {"id": "first", "name": NATIVE_READ_FILE, "arguments": {"path": str(path)}}
        assert (await client.post("/admit", json=packet)).json() == {"allowed": True}
        duplicate = (await client.post("/admit", json=packet)).json()
        assert duplicate["allowed"] is False and "already charged" in duplicate["error"]
        oversized = (await client.post("/admit", json={**packet, "id": "large",
            "arguments": {"path": "x" * 5000}})).json()
        assert oversized["allowed"] is False and "argument byte limit" in oversized["error"]
        binding.restrict(deadline=time.monotonic() + 100, tool_call_budget=99)
        assert binding.deadline == deadline
        spent = (await client.post("/admit", json={**packet, "id": "last"})).json()
        assert spent["allowed"] is False and "Original cumulative tool allowance" in spent["error"]
        assert binding.calls == 3
        for index in range(4, policy._MAX_CALLS + 1):
            assert (await client.post("/admit", json={**packet, "id": str(index)})).json()["allowed"] is False
        exhausted = (await client.post("/admit", json={**packet, "id": "overflow"})).json()
        assert exhausted["allowed"] is False and "capacity exhausted" in exhausted["error"]
        assert binding.calls == policy._MAX_CALLS
    assert binding.cleanup_complete


@pytest.mark.parametrize("reason", ["closed", "expired"])
async def test_revoked_or_expired_authority_cannot_start_admitted_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str,
) -> None:
    """Admission cannot retain expired/revoked authority; successful settlement only closes after work."""
    path = tmp_path.resolve() / "source"
    path.write_text("must not reach caller")
    binding = HostToolBinding((path,), deadline=time.monotonic() + 10)
    async with binding.session():
        binding.admit("late", NATIVE_READ_FILE, {"path": str(path)})
        if reason == "closed":
            binding.close_authority()
        else:
            monkeypatch.setattr("daydream.host_tools.clock.monotonic", lambda: binding.deadline)
        with pytest.raises(HostToolRejected, match="closed|expired"):
            await binding.read_file({"path": str(path)})
    assert not binding.authority_open and binding.cleanup_complete


@pytest.mark.parametrize("finish", ["close", "callback-deadline"])
async def test_active_read_is_reaped_and_concurrent_work_is_not_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish: str,
) -> None:
    """Owner close/deadline ends an actual blocked filesystem operation without queueing another."""
    path = tmp_path.resolve() / "source"
    path.write_text("must never be returned")
    monkeypatch.setattr(policy, "_CALLBACK_TIMEOUT_S", 0.3 if finish == "callback-deadline" else 5)
    binding = HostToolBinding((path,), deadline=time.monotonic() + 10)
    async with gated_reader(monkeypatch, tmp_path, path) as (entered, _, children, marker):
        async with binding.session():
            arguments = {"path": str(path)}
            binding.admit("active", NATIVE_READ_FILE, arguments)
            task = asyncio.create_task(binding.read_file(arguments))
            await asyncio.wait_for(entered.wait(), 3)
            binding.admit("parallel", NATIVE_READ_FILE, arguments)
            with pytest.raises(HostToolRejected, match="concurrency exhausted"):
                await binding.read_file(arguments)
            if finish == "callback-deadline":
                with pytest.raises(HostToolRejected, match="callback deadline"):
                    await task
        if finish == "close":
            with pytest.raises(HostToolRejected, match="reader failed"):
                await task
        with pytest.raises(HostToolRejected, match="closed"):
            await binding.read_file(arguments)
        assert len(children) == 1 and int(marker.read_text()) == children[0].pid
        with pytest.raises(ProcessLookupError):
            os.kill(children[0].pid, 0)
        assert not binding.authority_open and binding.cleanup_complete


async def test_cancelled_reader_startup_never_receives_command_and_is_joined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unfinished real spawn stays inert and is reaped when invocation setup is cancelled."""
    source = tmp_path.resolve() / "source"
    source.write_text("LATE_SPAWN_MUST_NEVER_READ_OR_RETURN_THIS\n")
    binding = HostToolBinding((source,), deadline=time.monotonic() + 10)
    closed = asyncio.Event()
    close_authority = binding.close_authority
    def observe_close() -> None:
        close_authority()
        closed.set()
    monkeypatch.setattr(binding, "close_authority", observe_close)
    async def invoke() -> None:
        async with binding.session():
            pytest.fail("Cancelled startup must not yield authority")
    async with gated_reader(monkeypatch, tmp_path, source, late_spawn=True) as (entered, release, children, marker):
        task = asyncio.create_task(invoke())
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        await asyncio.wait_for(closed.wait(), 3)
        assert not binding.authority_open
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert not marker.exists() and children[0].returncode is not None
        assert children[0].stdout is not None and await children[0].stdout.read() == b""
        with pytest.raises(ProcessLookupError):
            os.kill(children[0].pid, 0)
        assert binding.cleanup_complete
