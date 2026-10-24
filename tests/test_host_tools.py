"""Host authority and transport bounds at their real filesystem/process/HTTP owners."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from daydream.host_tools import NATIVE_READ_FILE, HostToolBinding, HostToolLimits, HostToolRejected
from daydream.host_tools_server import PiHostToolServer


@asynccontextmanager
async def _endpoint(binding: HostToolBinding) -> AsyncIterator[tuple[PiHostToolServer, httpx.AsyncClient]]:
    server = PiHostToolServer(binding)
    try:
        await server.start()
        async with httpx.AsyncClient(base_url=server.url, headers={
            "Authorization": server.packet()["authorization"], "Accept": "application/json, text/event-stream",
        }, timeout=3) as client:
            yield server, client
    finally:
        await server.close()


async def _native_call(client: httpx.AsyncClient, arguments: Any, *, call_id: str = "read") -> dict[str, Any]:
    admission = await client.post("/admit", json={"id": call_id, "name": NATIVE_READ_FILE,
                                               "arguments": arguments})
    assert admission.status_code == 200
    assert admission.json() == {"allowed": True}
    response = await client.post("/mcp", json={"jsonrpc": "2.0", "id": call_id, "method": "tools/call",
        "params": {"name": "read_file", "arguments": arguments}})
    assert response.status_code == 200
    result: dict[str, Any] = response.json()["result"]
    return result


@pytest.mark.parametrize("fault", ["missing", "extra", "number", "array", "null", "params-array"])
async def test_http_schema_rejections_precede_sdk_coercion(tmp_path: Path, fault: str) -> None:
    """The HTTP owner must reject malformed original shapes before FastMCP can coerce them.

    Native Pi validates its own tools first, so native journeys cannot exercise a malicious raw MCP caller.
    """
    path = tmp_path.resolve() / "allowed"
    path.write_text("secret bytes")
    binding = HostToolBinding((path,), deadline=time.monotonic() + 10)
    arguments: Any = {"missing": {}, "extra": {"path": str(path), "other": True}, "number": {"path": 42},
                      "array": [], "null": None, "params-array": {"path": str(path)}}[fault]
    async with _endpoint(binding) as (_, client):
        params: Any = [] if fault == "params-array" else {"name": "read_file", "arguments": arguments}
        response = await client.post("/mcp", json={"jsonrpc": "2.0", "id": "malformed", "method": "tools/call",
                                                 "params": params})
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["isError"] is True and "Invalid host read schema" in json.dumps(result)
    assert binding.calls == 0 and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["json", "frame", "items", "depth", "unauthorized", "http-capacity"])
async def test_http_admission_bounds_reject_before_tool_work(tmp_path: Path, fault: str) -> None:
    """A real HTTP client cannot bypass finite parsing/request bounds or transport credentials.

    The SDK and normal native frames never establish limits for malformed, oversized or excess transport work.
    """
    limits = HostToolLimits(max_frame_bytes=512, max_items=16, max_http_requests=2)
    binding = HostToolBinding((tmp_path.resolve() / "allowed",), deadline=time.monotonic() + 10, limits=limits)
    async with _endpoint(binding) as (_, client):
        if fault == "json":
            response = await client.post("/admit", content=b"{broken")
        elif fault == "frame":
            response = await client.post("/admit", content=b"x" * 513)
        elif fault == "items":
            response = await client.post("/admit", json={"values": list(range(20))})
        elif fault == "depth":
            response = await client.post("/admit", content=b'{"v":' + b"[" * 18 + b"0" + b"]" * 18 + b"}")
        elif fault == "unauthorized":
            response = await client.get("/status", headers={"Authorization": "Bearer wrong"})
        else:
            assert (await client.get("/status")).status_code == 200
            assert (await client.get("/status")).status_code == 200
            response = await client.get("/status")
        assert response.status_code == {"json": 400, "frame": 413, "items": 400, "depth": 400,
                                        "unauthorized": 403, "http-capacity": 429}[fault]
    assert binding.calls == 0 and binding.cleanup_complete


async def test_correlated_rejection_cannot_expand_response_past_frame_bound(tmp_path: Path) -> None:
    """A bounded UTF-8 request ID cannot expand into an oversized escaped error response.

    Normal ASCII native IDs miss JSON escaping expansion at the manual schema-rejection response boundary.
    """
    limits = HostToolLimits(max_frame_bytes=512)
    binding = HostToolBinding((tmp_path.resolve() / "source",), deadline=time.monotonic() + 10, limits=limits)
    frame = json.dumps({"jsonrpc": "2.0", "id": "界" * 100, "method": "tools/call",
                        "params": {"name": "read_file", "arguments": {}}}, ensure_ascii=False).encode()
    assert len(frame) < limits.max_frame_bytes
    async with _endpoint(binding) as (_, client):
        response = await client.post("/mcp", content=frame, headers={"Content-Type": "application/json"})
        assert len(response.content) <= limits.max_frame_bytes
        assert "source" not in response.text
    assert binding.calls == 0 and binding.cleanup_complete


@pytest.mark.parametrize("field", ["path", "key", "id"])
async def test_escaped_invalid_unicode_is_a_bounded_frame_rejection(tmp_path: Path, field: str) -> None:
    """ASCII JSON escapes must not smuggle unencodable strings into authority or result processing.

    Valid UTF-8 native calls and byte-overflow cases miss escaped unpaired-surrogate strings and object keys.
    """
    binding = HostToolBinding((tmp_path.resolve() / "source",), deadline=time.monotonic() + 10)
    packet: dict[str, Any] = {"id": "unicode", "name": NATIVE_READ_FILE, "arguments": {"path": "unused"}}
    if field == "id":
        packet["id"] = "\ud800"
    elif field == "key":
        packet["arguments"] = {"\ud800": True}
    else:
        packet["arguments"]["path"] = "\ud800"
    async with _endpoint(binding) as (_, client):
        response = await client.post("/admit", content=json.dumps(packet).encode(),
                                     headers={"Content-Type": "application/json"})
        assert response.status_code == 400
        assert len(response.content) <= binding.limits.max_frame_bytes
    assert binding.calls == 0 and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["traversal", "final-symlink", "parent-symlink", "fifo", "binary",
                                  "file-overflow", "serialized-overflow"])
async def test_host_read_preserves_exact_grant_and_rejects_non_authoritative_results(
    tmp_path: Path, fault: str,
) -> None:
    """Real child reads cannot follow links, escape exact grants or truncate authoritative bytes.

    A valid small native read does not cover adversarial filesystem identities or serialized-result expansion.
    """
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
        source.write_bytes(b"x" * 2049)
    elif fault == "serialized-overflow":
        source.write_text("\n" * 700)
    binding = HostToolBinding((granted,), deadline=time.monotonic() + 10,
                              limits=HostToolLimits(max_result_bytes=2048))
    async with _endpoint(binding) as (_, client):
        result = await _native_call(client, {"path": requested})
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert "not permitted" in text if fault == "traversal" else (
            "not UTF-8" in text if fault == "binary" else
            "serialized result byte limit" in text if fault == "serialized-overflow" else "unavailable" in text)
        assert "unavailable bytes" not in text
    assert binding.calls == 1 and binding.rejections == 1 and binding.cleanup_complete


async def test_host_admissions_are_once_only_bounded_and_never_renewed(tmp_path: Path) -> None:
    """Duplicate IDs, oversized arguments and spent allowance deny later effects without recharging IDs.

    Native normal calls supply distinct IDs and bounded arguments, missing the direct admission protocol risks.
    """
    path = tmp_path.resolve() / "source"
    path.write_text("uncharged bytes")
    binding = HostToolBinding((path,), deadline=time.monotonic() + 10,
        limits=HostToolLimits(max_calls=3, max_argument_bytes=256))
    binding.restrict(deadline=None, tool_call_budget=2)
    async with _endpoint(binding) as (_, client):
        packet = {"id": "first", "name": NATIVE_READ_FILE, "arguments": {"path": str(path)}}
        assert (await client.post("/admit", json=packet)).json() == {"allowed": True}
        duplicate = (await client.post("/admit", json=packet)).json()
        assert duplicate["allowed"] is False and "already charged" in duplicate["error"]
        oversized = (await client.post("/admit", json={**packet, "id": "large",
            "arguments": {"path": "x" * 300}})).json()
        assert oversized["allowed"] is False and "argument byte limit" in oversized["error"]
        spent = (await client.post("/admit", json={**packet, "id": "last"})).json()
        assert spent["allowed"] is False and "Original cumulative tool allowance" in spent["error"]
        assert binding.calls == 3
        binding.restrict(deadline=time.monotonic() + 100, tool_call_budget=99)
        exhausted = (await client.post("/admit", json={**packet, "id": "overflow"})).json()
        assert exhausted["allowed"] is False and "capacity exhausted" in exhausted["error"]
        assert binding.calls == 3
    assert binding.cleanup_complete


@pytest.mark.parametrize("reason", ["closed", "expired"])
async def test_revoked_or_expired_authority_cannot_start_admitted_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str,
) -> None:
    """Admission cannot keep a grant alive after closure or the original absolute deadline.

    Successful native settlement closes after work; it misses an admitted call arriving after revocation.
    """
    path = tmp_path.resolve() / "source"
    path.write_text("must not reach caller")
    binding = HostToolBinding((path,), deadline=time.monotonic() + 10)
    binding.claim()
    binding.admit("late", NATIVE_READ_FILE, {"path": str(path)})
    if reason == "closed":
        binding.close_authority()
    else:
        monkeypatch.setattr("daydream.host_tools.clock.monotonic", lambda: binding.deadline)
    with pytest.raises(HostToolRejected, match="closed|expired"):
        await binding.read_file({"path": str(path)})
    await binding.close()
    assert not binding.authority_open and binding.cleanup_complete


@pytest.mark.parametrize("finish", ["cancel", "callback-deadline"])
async def test_active_read_is_reaped_and_concurrent_work_is_not_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish: str,
) -> None:
    """Closing/timing out a blocked real read joins its child and denies queued or late work.

    CLI teardown tests do not own callback child processes, prospective read concurrency or late filesystem effects.
    """
    root = tmp_path.resolve()
    path = root / "source"
    path.write_text("must never be returned")
    fifo = root / "release"
    os.mkfifo(fifo)
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_reader(read_fd, entered.set)
    spawn = asyncio.create_subprocess_exec
    children: list[asyncio.subprocess.Process] = []

    async def blocked_reader(*args: Any, **options: Any) -> asyncio.subprocess.Process:
        if args[1:3] == ("-m", "daydream.host_tools_read"):
            script = "import os,sys; os.write(int(sys.argv[1]),b'R'); open(sys.argv[2],'rb').read(1)"
            child = await spawn(sys.executable, "-c", script, str(write_fd), str(fifo),
                                **{**options, "pass_fds": (write_fd,)})
            children.append(child)
            return child
        return await spawn(*args, **options)

    monkeypatch.setattr("daydream.host_tools.asyncio.create_subprocess_exec", blocked_reader)
    binding = HostToolBinding((path,), deadline=time.monotonic() + 10,
        limits=HostToolLimits(max_in_flight=1, callback_timeout_s=0.3 if finish == "callback-deadline" else 5))
    binding.claim()
    arguments = {"path": str(path)}
    binding.admit("active", NATIVE_READ_FILE, arguments)
    task = asyncio.create_task(binding.read_file(arguments))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        binding.admit("parallel", NATIVE_READ_FILE, arguments)
        with pytest.raises(HostToolRejected, match="concurrency exhausted"):
            await binding.read_file(arguments)
        if finish == "cancel":
            await binding.close()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(HostToolRejected, match="callback deadline"):
                await task
            await binding.close()
        with pytest.raises(HostToolRejected, match="closed"):
            await binding.read_file(arguments)
        assert len(children) == 1 and children[0].returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(children[0].pid, 0)
        assert not binding.authority_open and binding.cleanup_complete
    finally:
        await binding.close()
        await asyncio.gather(task, return_exceptions=True)
        loop.remove_reader(read_fd)
        os.close(read_fd)
        os.close(write_fd)


async def test_cancelled_late_spawn_never_receives_read_gate_and_is_joined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real helper returned after cancellation remains inert and is reaped before closure completes.

    The active-read case starts after spawn returns, missing cancellation between OS process creation and ownership.
    """
    source = tmp_path.resolve() / "source"
    source.write_text("LATE_SPAWN_MUST_NEVER_READ_OR_RETURN_THIS\n")
    spawned, release, authority_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = asyncio.create_subprocess_exec
    children: list[asyncio.subprocess.Process] = []

    async def late_spawn(*args: Any, **options: Any) -> asyncio.subprocess.Process:
        child = await original(*args, **options)
        children.append(child)
        spawned.set()
        await release.wait()
        return child

    monkeypatch.setattr("daydream.host_tools.asyncio.create_subprocess_exec", late_spawn)
    binding = HostToolBinding((source,), deadline=time.monotonic() + 10)
    binding.claim()
    binding.admit("late", NATIVE_READ_FILE, {"path": str(source)})
    close_authority = binding.close_authority

    def observe_close() -> None:
        close_authority()
        authority_closed.set()

    monkeypatch.setattr(binding, "close_authority", observe_close)
    task = asyncio.create_task(binding.read_file({"path": str(source)}))
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(spawned.wait(), 3)
        task.cancel()
        closing = asyncio.create_task(binding.close())
        await asyncio.wait_for(authority_closed.wait(), 3)
        assert not binding.authority_open
        release.set()
        await asyncio.wait_for(closing, 3)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(children) == 1 and children[0].returncode is not None
        assert children[0].stdout is not None and await children[0].stdout.read() == b""
        with pytest.raises(ProcessLookupError):
            os.kill(children[0].pid, 0)
        assert binding.cleanup_complete
    finally:
        release.set()
        await binding.close()
        await asyncio.gather(task, *([closing] if closing is not None else []), return_exceptions=True)
