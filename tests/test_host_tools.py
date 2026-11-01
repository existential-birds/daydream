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
from daydream.host_tools import (
    NATIVE_SUBMIT,
    HostTool,
    HostToolBinding,
    HostToolCallBudgetExceeded,
    HostToolFatal,
    HostToolRejected,
    NativeCapacityRequirement,
)
from daydream.review_source_refs import SOURCE_SELECTORS_SCHEMA
from tests.harness.fake_clock import FakeClock

_SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"verdict": {"type": "string"}, "findings": {"type": "array"}},
    "required": ["verdict", "findings"]}
_RESULT = {"verdict": "complete", "findings": []}


@asynccontextmanager
async def _endpoint(binding: HostToolBinding, schema: dict[str, Any] = _SCHEMA) -> AsyncIterator[httpx.AsyncClient]:
    binding.configure_schema(schema)
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
        rejected = (await client.post("/admit", json={"id": "invalid", "name": NATIVE_SUBMIT,
            "arguments": arguments, "terminal": True})).json()
        assert rejected["id"] == "invalid" and rejected["accepted"] is False
        assert rejected["outcome"] == "correctable" and rejected["code"] == "invalid_submission"
        assert "schema" in rejected["error"].lower()
        assert {"missing": '"verdict"', "extra": "undeclared", "number": "JSON object",
                "array": "JSON object", "null": "JSON object"}[fault] in rejected["error"]
        assert binding.authority_open and installed == [] and not binding.accepted
        assert (await client.post("/admit", json={"id": "corrected", "name": NATIVE_SUBMIT,
            "arguments": _RESULT, "terminal": True})).json()["allowed"]
        response = await client.post("/submit", json={"id": "corrected", "arguments": _RESULT})
        assert response.json() == {"id": "corrected", "accepted": True, "result": _RESULT}
    assert installed == [_RESULT] and binding.calls == 2 and binding.rejections == 1 and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["judgment", "range-shape", "range-bound", "selector-shape"])
async def test_http_schema_feedback_identifies_nested_fields_and_concrete_corrections(fault: str) -> None:
    """Nested schema failures explain the relevant judgment or selector correction.

    Root-object schema tests miss anyOf branch selection and nested source range
    constraints; a generic validator name leaves a valid model unable to correct
    those payloads and raw validator messages can disclose submitted contents.
    """
    schema = {"type": "object", "required": ["judgments"], "properties": {"judgments": {
        "type": "array", "items": {"type": "object", "required": ["judgment", "evidence_refs"],
            "properties": {"judgment": {"type": "string", "minLength": 1},
                           "evidence_refs": SOURCE_SELECTORS_SCHEMA}}}}}
    selector: dict[str, Any] = {"source_id": "source:" + "a" * 64, "basis": "supplied_diff", "source_range": None}
    judgment: dict[str, Any] = {"judgment": "checked", "evidence_refs": [selector.copy()]}
    if fault == "judgment":
        del judgment["judgment"]
    elif fault == "selector-shape":
        judgment["evidence_refs"] = [{"basis": "supplied_diff"}]
    else:
        judgment["evidence_refs"][0]["source_range"] = ("private-token" if fault == "range-shape" else
                                                        {"start": 0, "end": 1})
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return _RESULT
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    async with _endpoint(binding, schema) as client:
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": schema})).json()["ready"]
        response = (await client.post("/admit", json={"id": "invalid", "name": NATIVE_SUBMIT,
            "arguments": {"judgments": [judgment]}, "terminal": True})).json()
        assert response["allowed"] is False
        assert response["outcome"] == "correctable" and response["id"] == "invalid"
        assert 'judgments' in response["error"] and "private-token" not in response["error"]
        assert {"judgment": '"judgment"', "range-shape": "object", "range-bound": "at least 1",
                "selector-shape": '"source_id"'}[fault] in response["error"]
        if fault.startswith("range-"):
            assert 'source_range' in response["error"]
        valid = {"judgments": [{"judgment": "checked", "evidence_refs": [selector]}]}
        assert installed == [] and binding.authority_open
        assert (await client.post("/admit", json={"id": "corrected", "name": NATIVE_SUBMIT,
            "arguments": valid, "terminal": True})).json()["allowed"]
        assert (await client.post("/submit", json={"id": "corrected", "arguments": valid})).json()["accepted"]
    assert installed == [valid] and binding.calls == 2 and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["json", "frame", "items", "depth", "unauthorized",
                                  "unicode-argument", "unicode-key", "unicode-id", "argument-bytes"])
async def test_http_admission_bounds_reject_before_acceptance(
    monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Raw frames cannot bypass finite bounds, credentials or strict UTF-8 charging.

    Normal native JSON and schema checks cannot exercise malformed HTTP bodies or
    invalid Unicode; the callback endpoint owns these pre-mutation bounds.
    Item/depth overflow payloads remain schema-valid so a later schema rejection
    cannot mask removing the intended request bound.
    """
    monkeypatch.setattr(policy, "_MAX_ITEMS", 128)
    monkeypatch.setattr(policy, "_MAX_DEPTH", 16)
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    async with _endpoint(binding) as client:
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        if fault == "json":
            response = await client.post("/admit", content=b"{broken")
        elif fault == "frame":
            response = await client.post("/admit", content=b"x" * (policy._MAX_FRAME_BYTES + 1))
        elif fault == "items":
            response = await client.post("/admit", json={"id": "items", "name": NATIVE_SUBMIT,
                "arguments": {"verdict": "complete", "findings": list(range(130))}, "terminal": True})
        elif fault == "depth":
            response = await client.post("/admit", content=(b'{"id":"depth","name":"structured_output",'
                b'"arguments":{"verdict":"complete","findings":' + b"[" * 18 + b"0" + b"]" * 18
                + b'},"terminal":true}'))
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
        if fault != "unauthorized":
            assert response.json()["outcome"] == (
                "fatal" if fault in {"json", "frame", "unicode-id"} else "correctable")
        if fault in {"items", "depth"}:
            expected_error = (
                f"The entire host request exceeds {policy._MAX_ITEMS} items; submit fewer argument items"
                if fault == "items" else
                f"Host request nesting exceeds {policy._MAX_DEPTH} levels; flatten the submitted arguments")
            assert response.json() == {
                "id": fault, "outcome": "correctable", "code": "invalid_submission", "error": expected_error,
                "allowed": False, "accepted": False,
            }
            assert binding.calls == 1
        if fault == "argument-bytes":
            assert "entire" in response.json()["error"] and str(policy._MAX_ARGUMENT_BYTES) in response.json()["error"]
    assert binding.calls == int(fault in {"items", "depth", "unicode-argument", "unicode-key", "argument-bytes"})
    assert binding.cleanup_complete


@pytest.mark.parametrize("fault", ["schema", "admit", "submit"])
async def test_http_readiness_requires_matching_terminal_definition_before_submission(fault: str) -> None:
    """A reachable endpoint cannot stand in for the active model-facing native schema.

    The installed success journey cannot detect an endpoint that accepts a terminal
    call before the extension confirms its exact active definition. Previously a
    broken readiness contract invited payload correction and could reopen later.
    """
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    async with _endpoint(binding) as client:
        assert not (await client.get("/status")).json()["ready"]
        payload = ({"tool_name": NATIVE_SUBMIT, "schema": {"type": "object"}} if fault == "schema" else
                   {"id": "not-ready", "name": NATIVE_SUBMIT, "arguments": _RESULT, "terminal": True})
        response = (await client.post("/ready" if fault == "schema" else f"/{fault}", json=payload)).json()
        assert not binding.authority_open and installed == []
        assert response["outcome"] == "fatal" and response["code"] == "native_readiness"
        assert response["accepted"] is False and "cannot resolve" in response["error"]
        if fault != "schema":
            assert response["id"] == "not-ready"
        assert not binding.authority_open and installed == []
        corrected = (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()
        assert corrected["outcome"] == "fatal" and corrected["code"] == "native_readiness"
    with pytest.raises(HostToolFatal, match="cannot resolve"):
        async with binding.session():
            pytest.fail("fatal readiness must not reopen")
    assert installed == [] and binding.calls == 0 and binding.cleanup_complete


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
        binding.admit("ordinary", "read", {"path": "/source", "offset": 1, "limit": 2})
        assert binding.calls == 2
    binding.restrict(deadline=time.monotonic() + 100, tool_call_budget=99)
    assert binding.deadline == deadline
    async with binding.session():
        with pytest.raises(HostToolCallBudgetExceeded, match="Original cumulative tool allowance") as exhausted:
            binding.admit("corrected", NATIVE_SUBMIT, _RESULT, terminal=True)
        with pytest.raises(HostToolCallBudgetExceeded, match="Original cumulative tool allowance") as repeated:
            binding.submit("corrected", _RESULT)
        assert repeated.value is exhausted.value
    assert binding.calls == 3 and installed == [] and binding.cleanup_complete


async def test_http_operator_grants_are_frozen_and_retries_only_narrow_them() -> None:
    """Caller mutation or a retry must not authorize an ungranted native effect.

    Installed Pi excludes forbidden tools before native lookup, so it cannot prove
    the host endpoint independently denies their starts and preserves narrowed
    authority across attempt-local readiness sessions.
    """
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    grants = {"read", NATIVE_SUBMIT}
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept, allowed_tools=grants)
    grants.add("write")
    async with _endpoint(binding) as client:
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        rejected = (await client.post("/admit", json={"id": "write", "name": "write", "arguments": None})).json()
        assert rejected["allowed"] is False and rejected["code"] == "permission_denied"
        assert binding.authority_open and installed == []
    binding.restrict(deadline=None, tool_call_budget=None, allowed_tools=frozenset({NATIVE_SUBMIT}))
    binding.restrict(deadline=None, tool_call_budget=None, allowed_tools=frozenset({"read", "write", NATIVE_SUBMIT}))
    async with _endpoint(binding) as client:
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        rejected = (await client.post("/admit", json={"id": "read", "name": "read", "arguments": None})).json()
        assert rejected["allowed"] is False and rejected["code"] == "permission_denied"
        assert binding.authority_open and installed == []
        assert (await client.post("/admit", json={"id": "submit", "name": NATIVE_SUBMIT,
            "arguments": _RESULT, "terminal": True})).json()["allowed"]
        assert (await client.post("/submit", json={"id": "submit", "arguments": _RESULT})).json()["accepted"]
    assert installed == [_RESULT] and binding.calls == 3 and binding.rejections == 2 and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["schema", "semantic", "items", "unicode", "none"])
async def test_last_available_native_call_cannot_invite_an_impossible_correction(
    monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """The final call accepts valid output or terminates an unrepairable allowance.

    Existing cumulative-budget coverage only rejected the next start; it missed
    schema/native validation or domain rejection inviting another provider turn
    after the last authorized start had already consumed the remaining capacity.
    """
    monkeypatch.setattr(policy, "_MAX_ITEMS", 128)
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        if fault == "semantic":
            raise HostToolRejected('target "target.py": add the required judgment; '
                                   'Authorization: Bearer callback-secret-123; ' + 'é' * 1024)
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    binding.restrict(deadline=None, tool_call_budget=1)
    arguments = {"schema": {"verdict": "complete", "findings": False},
                 "items": {"verdict": "complete", "findings": list(range(130))},
                 "unicode": {"verdict": "\ud800", "findings": []}}.get(fault, _RESULT)
    async with _endpoint(binding) as client:
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        response = (await client.post("/admit", content=json.dumps({"id": "last", "name": NATIVE_SUBMIT,
            "arguments": arguments, "terminal": True}).encode(), headers={"Content-Type": "application/json"})).json()
        if fault in {"semantic", "none"}:
            assert response["allowed"] is True
            response = (await client.post("/submit", json={"id": "last", "arguments": arguments})).json()
        if fault == "none":
            assert response == {"id": "last", "accepted": True, "result": _RESULT}
            assert binding.accepted and installed == [_RESULT] and binding.fatal_error is None
        else:
            assert not binding.authority_open and response["allowed"] is False
            assert response["id"] == "last" and response["outcome"] == "fatal"
            assert response["code"] == "capacity_exhausted" and response["accepted"] is False
            assert "no calls remain" in response["error"] and not binding.authority_open
            assert {"schema": "findings", "semantic": "required judgment", "items": "host request",
                    "unicode": "UTF-8"}[fault] in response["error"]
            assert len(response["error"].encode()) <= 1024
            if fault == "semantic":
                assert "callback-secret" not in response["error"] and "REDACTED" in response["error"]
            assert installed == [] and not binding.accepted and not binding.possibly_accepted
            assert isinstance(binding.fatal_error, HostToolCallBudgetExceeded)
            assert isinstance(binding.fatal_error.__cause__, HostToolRejected)
            repeat = (await client.post("/admit", json={"id": "retry", "name": NATIVE_SUBMIT,
                "arguments": _RESULT, "terminal": True})).json()
            assert repeat["code"] == "capacity_exhausted"
    assert binding.calls == 1 and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["duplicate-start", "unmatched-submit", "changed-submit"])
async def test_broken_native_correlation_is_terminal_without_acceptance(fault: str) -> None:
    """Correlation faults cannot be repaired through another model submission.

    The ordinary correction journey only uses fresh, matching tool IDs; it does
    not catch reusing an ID or changing arguments after host authorization.
    """
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    async with _endpoint(binding) as client:
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        packet = {"id": "terminal", "name": NATIVE_SUBMIT, "arguments": _RESULT, "terminal": True}
        assert (await client.post("/admit", json=packet)).json()["allowed"]
        if fault == "duplicate-start":
            response = (await client.post("/admit", json=packet)).json()
        else:
            response = (await client.post("/submit", json={
                "id": "foreign" if fault == "unmatched-submit" else "terminal",
                "arguments": {**_RESULT, "verdict": "changed"} if fault == "changed-submit" else _RESULT,
            })).json()
        assert not binding.authority_open and installed == []
        assert response["outcome"] == "fatal" and response["code"] == "native_correlation"
        assert response["accepted"] is False and "cannot resolve" in response["error"]
        assert not binding.authority_open and not binding.possibly_accepted and installed == []
    assert binding.calls == 1 and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["known-input", "value-error", "mutated-value-error"])
async def test_callback_failure_guarantees_are_host_owned_and_preserve_causes(fault: str) -> None:
    """Only explicit callback guarantees can certify no acceptance occurred.

    Earlier HTTP cases fail before callback entry; they cannot catch arbitrary
    ValueError being mislabeled correctable after a callback has installed state,
    or a known input outage losing its cause and inviting selector corrections.
    """
    installed: list[Any] = []
    original = FileNotFoundError("private-captured-input") if fault == "known-input" else ValueError("private-token")
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        if fault == "known-input":
            raise HostToolFatal("Frozen input for target.py is unavailable; TOKEN=callback-secret-123",
                                code="input_unavailable") from original
        if fault == "mutated-value-error":
            installed.append(value)
        raise original
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    async with _endpoint(binding) as client:
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        assert (await client.post("/admit", json={"id": "terminal", "name": NATIVE_SUBMIT,
            "arguments": _RESULT, "terminal": True})).json()["allowed"]
        response = (await client.post("/submit", json={"id": "terminal", "arguments": _RESULT})).json()
        preacceptance = fault == "known-input"
        assert binding.possibly_accepted is not preacceptance and not binding.accepted and not binding.authority_open
        assert response["id"] == "terminal" and response["outcome"] == ("fatal" if preacceptance else "indeterminate")
        assert response["code"] == ("input_unavailable" if preacceptance else "callback_failed")
        if preacceptance:
            assert response["accepted"] is False and response["allowed"] is False
        else:
            assert "accepted" not in response and "allowed" not in response and "do not resubmit" in response["error"]
        assert "private-" not in response["error"] and "callback-secret" not in response["error"]
        assert len(response["error"].encode()) <= 1024
        assert binding.fatal_error is not None and binding.fatal_error.__cause__ is original
    with pytest.raises(HostToolFatal):
        async with binding.session():
            pytest.fail("terminal callback outcome must not reopen")
    assert installed == ([_RESULT] if fault == "mutated-value-error" else []) and binding.cleanup_complete


async def test_correctable_callback_keeps_its_code_and_accepts_only_the_correction() -> None:
    """Callback-authored correction details survive binding and transport intact.

    Schema errors originate in the binding, and terminal-callback tests never
    reach another accepted submission; neither catches reconstructing a callback
    rejection with a generic code or forbidding an ordinary semantic correction.
    """
    installed: list[Any] = []
    rejected = False
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        nonlocal rejected
        if not rejected:
            rejected = True
            raise HostToolRejected("target.py judgment: add the required explanation; "
                                   "Authorization: Bearer callback-secret-123", code="missing_judgment")
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    async with _endpoint(binding) as client:
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        for identifier in ["invalid", "corrected"]:
            assert (await client.post("/admit", json={"id": identifier, "name": NATIVE_SUBMIT,
                "arguments": _RESULT, "terminal": True})).json()["allowed"]
            response = (await client.post("/submit", json={"id": identifier, "arguments": _RESULT})).json()
            if identifier == "invalid":
                assert response["outcome"] == "correctable" and response["code"] == "missing_judgment"
                assert "target.py judgment" in response["error"] and response["accepted"] is False
                assert "callback-secret" not in response["error"] and "REDACTED" in response["error"]
                assert binding.authority_open and not binding.possibly_accepted and installed == []
            else:
                assert response == {"id": identifier, "accepted": True, "result": _RESULT}
    assert installed == [_RESULT] and binding.calls == 2 and binding.rejections == 1 and binding.cleanup_complete


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
    async with _endpoint(binding) as client:
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        assert (await client.get("/status")).json()["remaining_ms"] == 10_000
        fake.advance(1)
        assert (await client.get("/status")).json()["remaining_ms"] == 9_000
        binding.admit("terminal", NATIVE_SUBMIT, _RESULT, terminal=True)
        if order == "acceptance-first":
            assert binding.submit("terminal", _RESULT) == _RESULT
            binding.close_authority()
            fake.advance(10)
            assert binding.submit("terminal", _RESULT) == _RESULT
            assert (await client.post("/submit", json={"id": "terminal", "arguments": _RESULT})).json() == {
                "id": "terminal", "accepted": True, "result": _RESULT}
            with pytest.raises(HostToolFatal, match="closed|expired"):
                binding.submit("different-terminal", _RESULT)
        else:
            if order == "closure-first":
                binding.close_authority()
            else:
                fake.advance(10)
            with pytest.raises(HostToolFatal, match="closed|expired"):
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
        with pytest.raises(HostToolFatal, match="invalid canonical result"):
            binding.submit("terminal", _RESULT)
        assert binding.possibly_accepted and not binding.accepted and not binding.authority_open
        with pytest.raises(HostToolFatal, match="invalid canonical result"):
            binding.submit("terminal", _RESULT)
    with pytest.raises(HostToolFatal, match="invalid canonical result"):
        async with binding.session():
            pytest.fail("uncertain acceptance must not reopen")
    assert calls == [_RESULT] and binding.cleanup_complete


@pytest.mark.parametrize("fault", ["pending", "eof", "valid", "reply-error"])
async def test_endpoint_cleanup_joins_incomplete_callback_frames_before_return(
    monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Join pending frames and keep retired EOF/dispatch/reply faults out of a live retry."""
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=accept)
    binding.configure_schema(_SCHEMA)
    body = json.dumps({"id": "stale", "name": NATIVE_SUBMIT, "arguments": _RESULT, "terminal": True}).encode()
    reading_body = asyncio.Event()
    read_body = asyncio.StreamReader.readexactly
    async def observe_body(reader: asyncio.StreamReader, size: int) -> bytes:
        if size == len(body):
            reading_body.set()
        return await read_body(reader, size)
    monkeypatch.setattr(asyncio.StreamReader, "readexactly", observe_body)
    drain = asyncio.StreamWriter.drain
    async def fail_reply(writer: asyncio.StreamWriter) -> None:
        if fault == "reply-error" and writer in server._writers and not binding.authority_open:
            raise ConnectionResetError("Peer disconnected before the retired frame's error reply")
        await drain(writer)
    writer: asyncio.StreamWriter | None = None
    try:
        async with binding.session() as server:
            async with httpx.AsyncClient(base_url=server.url,
                headers={"Authorization": server.packet()["authorization"]}) as client:
                ready = await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})
                assert ready.json()["ready"]
            monkeypatch.setattr(asyncio.StreamWriter, "drain", fail_reply)
            endpoint = urlsplit(server.url)
            reader, writer = await asyncio.open_connection(endpoint.hostname, endpoint.port)
            writer.write((f"POST /admit HTTP/1.1\r\nHost: {endpoint.netloc}\r\n"
                f"Authorization: {server.packet()['authorization']}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body[:1])
            await writer.drain()
            await asyncio.wait_for(reading_body.wait(), 3)
            if fault != "pending":
                binding.close_attempt()
                with pytest.raises(ValueError, match="cannot start another native attempt"):
                    async with binding.session():
                        pytest.fail("a retired endpoint must join before retry")
                if fault == "valid":
                    writer.write(body[1:])
                    await writer.drain()
                else:
                    writer.write_eof()
                await asyncio.wait_for(asyncio.gather(reader.read(), *server._tasks, return_exceptions=True), 3)
                assert not binding.closed and binding.fatal_error is None
                assert binding.calls == 0 and installed == []
        assert await asyncio.wait_for(reader.read(), 3) == b""
    finally:
        monkeypatch.setattr(asyncio.StreamWriter, "drain", drain)
        if writer is not None:
            writer.close()
            await writer.wait_closed()
    assert installed == [] and not binding.authority_open and binding.cleanup_complete and server.closed
    async with _endpoint(binding) as client:
        await server.close()
        assert binding.authority_open
        assert (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA})).json()["ready"]
        assert (await client.post("/admit", json={"id": "retry", "name": NATIVE_SUBMIT,
            "arguments": _RESULT, "terminal": True})).json()["allowed"]
        assert (await client.post("/submit", json={"id": "retry", "arguments": _RESULT})).json()["accepted"]
    assert installed == [_RESULT] and binding.calls == 1 and binding.cleanup_complete


_LOOKUP_SCHEMA = {"type": "object", "additionalProperties": False,
                  "properties": {"target_id": {"type": "string"}}}


def _lookup_binding(callback: Any, *, mutating: bool = False, **kwargs: Any) -> HostToolBinding:
    return HostToolBinding(deadline=time.monotonic() + 10, accept=lambda value: value,
                           host_tools={"review_source": HostTool(_LOOKUP_SCHEMA, callback, mutating)}, **kwargs)


async def _ready(client: httpx.AsyncClient, **extra: Any) -> dict[str, Any]:
    response: dict[str, Any] = (await client.post("/ready", json={
        "tool_name": NATIVE_SUBMIT, "schema": _SCHEMA, "host_tools": {"review_source": _LOOKUP_SCHEMA}, **extra,
    })).json()
    return response


@pytest.mark.parametrize("fault", ["none", "unadmitted", "reused", "wrong-name"])
async def test_http_host_tool_call_serves_each_admitted_start_once(fault: str) -> None:
    """A /call reply is served only for one unused admitted start of that exact tool.

    Native journeys only exercise correlated calls, so they cannot catch an endpoint
    that serves an unadmitted identity, replays a consumed one, or lets a read
    admission authorize a different host tool.
    """
    served: list[Any] = []
    def lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        served.append(arguments)
        return {"target_id": arguments["target_id"], "content": "exact\n\"bytes\""}
    binding = _lookup_binding(lookup)
    async with _endpoint(binding) as client:
        ready = await _ready(client)
        assert ready["ready"] is True and ready["native_limits"]["reserve_tokens_source"] == "pi_1.4.0_default"
        admitted = "read" if fault == "wrong-name" else "review_source"
        if fault != "unadmitted":
            assert (await client.post("/admit", json={"id": "lookup", "name": admitted,
                                                     "arguments": None})).json()["allowed"]
        call = {"id": "lookup", "name": "review_source", "arguments": {"target_id": "t1"}}
        response = (await client.post("/call", json=call)).json()
        if fault in {"none", "reused"}:
            assert response["id"] == "lookup" and response["result"]["content"] == "exact\n\"bytes\""
            assert response["result"]["capacity"]["remaining_s"] > 0
            if fault == "none":
                return
            response = (await client.post("/call", json=call)).json()
        assert response["outcome"] == "fatal" and response["code"] == "native_correlation"
        assert not binding.authority_open
    assert served == ([{"target_id": "t1"}] if fault == "reused" else [])


async def test_http_host_tool_correctable_rejection_keeps_authority() -> None:
    """Invalid lookup arguments are correctable and leave authority for the corrected call.

    Terminal-submission correction tests use /submit; they cannot catch a /call
    path that turns an argument error into a fatal stop or serves invalid input.
    """
    served: list[Any] = []
    def lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        served.append(arguments)
        return {"ok": True}
    binding = _lookup_binding(lookup)
    async with _endpoint(binding) as client:
        assert (await _ready(client))["ready"]
        for identifier, arguments in [("bad", {"target_id": 7}), ("good", {"target_id": "t1"})]:
            assert (await client.post("/admit", json={"id": identifier, "name": "review_source",
                                                     "arguments": None})).json()["allowed"]
            response = (await client.post("/call", json={"id": identifier, "name": "review_source",
                                                         "arguments": arguments})).json()
            if identifier == "bad":
                assert response["outcome"] == "correctable" and response["allowed"] is False
                assert "review_source argument" in response["error"] and binding.authority_open
            else:
                assert response["result"]["ok"] is True
    assert served == [{"target_id": "t1"}] and binding.calls == 2 and binding.rejections == 1


@pytest.mark.parametrize("mutating", [False, True], ids=["read-only", "mutating"])
async def test_http_host_tool_unexpected_failure_is_indeterminate_only_when_mutating(mutating: bool) -> None:
    """An unexpected callback failure prohibits replay only for a mutating host tool.

    Terminal callback tests cover /submit only; they cannot catch /call presenting
    a possibly installed batch admission as a fatal preacceptance outcome.
    """
    def broken(_arguments: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("private-token")
    binding = _lookup_binding(broken, mutating=mutating)
    async with _endpoint(binding) as client:
        assert (await _ready(client))["ready"]
        assert (await client.post("/admit", json={"id": "x", "name": "review_source",
                                                 "arguments": None})).json()["allowed"]
        response = (await client.post("/call", json={"id": "x", "name": "review_source",
                                                     "arguments": {"target_id": "t1"}})).json()
    assert response["outcome"] == ("indeterminate" if mutating else "fatal")
    assert response["code"] == "callback_failed" and "private-token" not in response["error"]
    assert ("accepted" in response) is not mutating and not binding.authority_open


@pytest.mark.parametrize("case", ["insufficient", "unavailable", "host-schema"])
async def test_http_readiness_refuses_native_limits_or_host_schemas_before_provider_work(case: str) -> None:
    """Readiness refuses a model that cannot fit the invocation, or a drifted host tool.

    Installed native tests use one fixture model, so they cannot prove the endpoint
    refuses insufficient or undeclared limits before any admission, or that a
    host-tool schema differing from the bound one fails native readiness.
    """
    binding = _lookup_binding(lambda value: value,
                              native_requirement=NativeCapacityRequirement(largest_tool_result_bytes=90_000,
                                                                           submission_bytes=30_000))
    binding.prompt_bytes = 3_000
    limits = {"model": {"context_window": 32768, "max_tokens": 8192} if case == "insufficient" else {}}
    async with _endpoint(binding) as client:
        if case == "host-schema":
            response = (await client.post("/ready", json={"tool_name": NATIVE_SUBMIT, "schema": _SCHEMA,
                "host_tools": {"review_source": {"type": "object"}}, "native_limits": limits})).json()
            assert response["code"] == "native_readiness"
        else:
            response = await _ready(client, native_limits=limits)
            assert response["outcome"] == "fatal" and response["code"] == "native_capacity"
            assert ("needs contextWindow >= " in response["error"]) is (case == "insufficient")
            assert binding.native_limits is not None
            assert binding.native_limits["keep_recent_tokens_source"] == "pi_1.4.0_default"
            # 41,000 prompt/result/submission tokens plus Pi's default compaction allowances.
            assert binding.native_limits["required_context_tokens"] == 41_000 + 16_384 + 20_000
        admitted = (await client.post("/admit", json={"id": "late", "name": "read", "arguments": None})).json()
        assert admitted["outcome"] == "fatal" and not binding.authority_open
    assert binding.calls == 0
