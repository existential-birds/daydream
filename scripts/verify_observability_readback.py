#!/usr/bin/env python3
"""Verify stored HoneyHive/LangSmith native trees against an immutable receipt.

P18/#1156 Task 6 bounded native readback verifier. The transport contract is the
frozen plan (P18-plan.md SHA-256 ``1bb962866288395f6ddf0504fd8affd9764c019cbe024c8c61140f2f5b6a9738``
plus the readback-deadline amendment, P18-readback-deadline-plan-correction.md):

- JSON parsing only (this tool never reuses the OTLP protobuf machinery);
- one AnyIO run owning exactly one ``httpx.AsyncClient(trust_env=False,
  follow_redirects=False)`` closed on every exit path;
- one immutable monotonic deadline computed before the first request; every
  request, capped streamed JSON-body read, poll sleep and backoff uses only
  ``overall_remaining``; each request runs as ``async with client.stream(...)``
  under ``anyio.fail_after(overall_remaining)`` and therefore closes before
  another page/poll and on ordinary error or cancellation;
- deadline expiry emits ONE fixed redacted timeout disposition, starts no
  further request/poll, and never includes a URL, header, response body,
  exception text or credential;
- the immutable canonical input receipt is validated BEFORE the client is
  constructed; a separate result receipt is written atomically (input receipt
  is never modified);
- keys are read only from existing environment variables; base URLs are
  validated with the same production policy as the exporters;
- HoneyHive ``POST /v1/events/search`` is paginated with a strict
  ``{events, count}`` shape until ``count`` is satisfied or a page is
  exhausted; duplicate event IDs, wrong-session rows, malformed/changing
  shapes, page/item bounds and redirects fail closed;
- LangSmith ``POST /runs/query`` discovery filters the exact
  ``daydream.run.id`` equality in one explicit project with a bounded start
  time, requires one trace/root, freezes the vendor-returned root/trace IDs,
  then re-reads the exact IDs/tree; never derives IDs from HoneyHive or from
  the Daydream UUID;
- after shutdown the verifier polls until two complete exact-session snapshots
  agree or the shared immutable deadline expires;
- stored field presence/types/relationships are compared to the checked-in
  machine-readable matrix subset (``tests/fixtures/observability_contract/
  readback-matrix.json``);
- output contains only destination, IDs, counts, names, types, booleans,
  stable hashes and pass/fail matrix rows. It exits 0 only when the exact
  session/run has one root, expected descendants, complete parents, canonical
  types, expected generation/turn/tool counts, exact response identity/timing
  provenance (sanitized replay), complete usage invariants with no
  parent/child billing duplicate, resource identity and no forbidden/private
  or ambient-context fields. API storage proof is never UI proof.

Operator usage (keys come from the environment, never from the receipt):

    HH_API_URL=... HH_API_KEY=... LANGSMITH_API_KEY=... \
    python scripts/verify_observability_readback.py \
        --receipt /path/to/receipt.json --result /path/to/result.json

``--deadline`` defaults to 30 seconds and is the ONE immutable budget for all
requests, polls and backoff. ``--matrix`` defaults to the checked-in subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import anyio
import httpx

from daydream.json_utils import atomic_write_json

# ---------------------------------------------------------------------------
# Frozen fixed dispositions (never include URLs/headers/bodies/exception text)
# ---------------------------------------------------------------------------

DISPOSITION_PASS = "pass"
DISPOSITION_TIMEOUT = "READBACK_TIMEOUT"
DISPOSITION_AUTH = "READBACK_AUTH_FAILED"
DISPOSITION_HTTP_ERROR = "READBACK_HTTP_ERROR"
DISPOSITION_REDIRECT = "READBACK_REDIRECT_REJECTED"
DISPOSITION_NOT_FOUND = "READBACK_NOT_FOUND"
DISPOSITION_MALFORMED = "READBACK_MALFORMED_RESPONSE"
DISPOSITION_OVERSIZED = "READBACK_RESPONSE_OVERSIZED"
DISPOSITION_SHAPE = "READBACK_SHAPE_MISMATCH"
DISPOSITION_DUPLICATE = "READBACK_DUPLICATE_ID"
DISPOSITION_WRONG_SESSION = "READBACK_WRONG_SESSION"
DISPOSITION_BOUNDS = "READBACK_BOUNDS_EXCEEDED"
DISPOSITION_UNSTABLE = "READBACK_UNSTABLE_SNAPSHOT"
DISPOSITION_MISSING_FIELD = "READBACK_MISSING_REQUIRED_FIELD"
DISPOSITION_AMBIGUOUS_ROOT = "READBACK_AMBIGUOUS_ROOT"
DISPOSITION_ALL = (
    DISPOSITION_PASS,
    DISPOSITION_TIMEOUT,
    DISPOSITION_AUTH,
    DISPOSITION_HTTP_ERROR,
    DISPOSITION_REDIRECT,
    DISPOSITION_NOT_FOUND,
    DISPOSITION_MALFORMED,
    DISPOSITION_OVERSIZED,
    DISPOSITION_SHAPE,
    DISPOSITION_DUPLICATE,
    DISPOSITION_WRONG_SESSION,
    DISPOSITION_BOUNDS,
    DISPOSITION_UNSTABLE,
    DISPOSITION_MISSING_FIELD,
    DISPOSITION_AMBIGUOUS_ROOT,
)

#: Bounded decoded JSON body: 4 MiB + 1 byte (the frozen transport decode
#: bound). A larger response is rejected without buffering past the cap.
MAX_BODY_BYTES = 4 * 1024 * 1024 + 1

#: HoneyHive event identity field candidates (root event fields; an event
#: without either is malformed).
HH_EVENT_ID_KEYS = ("id", "event_id")

_LANGSMITH_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


class ReadbackError(ValueError):
    """A bounded, redacted verifier failure; never carries API details."""


# ---------------------------------------------------------------------------
# Input receipt validation (before any client construction)
# ---------------------------------------------------------------------------

_RECEIPT_REQUIRED = (
    "schema_version",
    "contract_version",
    "acceptance_kind",
    "run_id",
    "session_id",
    "flow",
    "capture_mode",
    "destinations",
    "langsmith_project",
    "started_at",
    "ended_at",
    "model_call_count",
    "operational_cost_usd",
)


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the immutable canonical input receipt; never construct a client first."""
    missing = [key for key in _RECEIPT_REQUIRED if key not in receipt]
    if missing:
        raise ReadbackError(f"Receipt missing required fields: {', '.join(sorted(missing))}")
    kind = receipt["acceptance_kind"]
    if kind not in ("representative_real_run", "sanitized_protocol_replay"):
        raise ReadbackError("Receipt acceptance_kind must be representative_real_run or sanitized_protocol_replay")
    if not isinstance(receipt["run_id"], str) or not isinstance(receipt["session_id"], str):
        raise ReadbackError("Receipt run_id/session_id must be strings")
    if not isinstance(receipt["destinations"], list) or not all(
        isinstance(name, str) for name in receipt["destinations"]
    ):
        raise ReadbackError("Receipt destinations must be a list of strings")
    for key in ("model_call_count", "operational_cost_usd"):
        if not isinstance(receipt[key], (int, float)) or isinstance(receipt[key], bool):
            raise ReadbackError(f"Receipt {key} must be a number")
    supported = {"honeyhive", "langsmith"}
    enabled = set(receipt["destinations"]) & supported
    if not enabled:
        raise ReadbackError("Receipt must enable honeyhive and/or langsmith for readback")


def validate_base_url(raw: str, setting: str) -> str:
    """Validate a readback base URL with the same production policy.

    Mirrors ``daydream.observability.exporters._validated_endpoint`` (HTTP(S)
    only, hostname present, no credentials/query/fragment/whitespace/control
    characters, parseable port). The verifier deliberately does not import the
    OTLP transport machinery; this is the identical URL policy.
    """
    try:
        parsed = urlsplit(raw)
        valid = (
            bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and not any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in raw)
            and parsed.scheme in ("http", "https")
        )
        parsed.port  # noqa: B018 - validates malformed/out-of-range ports
    except (ValueError, AttributeError):
        valid = False
    if not valid:
        raise ReadbackError(f"{setting} must be a valid HTTP(S) URL without credentials, query, fragment or whitespace")
    return raw.rstrip("/")


def _split_path(path: str) -> list[str]:
    current: list[str] = []
    parts: list[str] = []
    quoted = False
    for char in path:
        if char == "'":
            quoted = not quoted
        elif char == "." and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [part for part in parts if part]


def walk_metadata(value: Any, dotted: str) -> Any:
    """Walk a stored metadata/config object for a dotted Daydream key.

    Both flat metadata (``metadata["daydream.generation.sealed_end_unix_ns"]``)
    and nested objects (``metadata["daydream"]["generation"]["sealed_end_unix_ns"]``)
    are accepted; returns None when absent.
    """
    if isinstance(value, Mapping):
        if dotted in value:
            return value[dotted]
    parts = _split_path(dotted)
    node: Any = value
    for part in parts:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def stable_hash(payload: Any) -> str:
    """SHA-256 over canonical JSON of identity/count material only."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# ---------------------------------------------------------------------------
# Bounded HTTP layer (one AnyIO run, one owned client, one immutable deadline)
# ---------------------------------------------------------------------------


class ReadbackClient:
    """Own one AsyncClient and one immutable deadline; close on every exit."""

    def __init__(self, *, budget_s: float) -> None:
        if not isinstance(budget_s, (int, float)) or isinstance(budget_s, bool) or budget_s <= 0:
            raise ReadbackError("--deadline must be a positive number of seconds")
        self.budget_s = float(budget_s)
        self.request_log: list[dict[str, Any]] = []
        self._started: float | None = None

    def start(self) -> None:
        self._started = anyio.current_time()

    def remaining(self) -> float:
        assert self._started is not None
        return self._started + self.budget_s - anyio.current_time()

    def _record(self, destination: str, op: str, key: str) -> None:
        self.request_log.append(
            {
                "destination": destination,
                "op": op,
                "key": key,
                "offset_ms": int((anyio.current_time() - (self._started or anyio.current_time())) * 1000),
            }
        )

    async def post_json(
        self,
        client: httpx.AsyncClient,
        *,
        destination: str,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        op: str,
        key: str,
    ) -> tuple[str, int, dict[str, Any], bytes]:
        """One bounded POST under the shared immutable deadline.

        Returns (disposition, status, parsed_json, raw_body). A timeout returns
        only the fixed timeout disposition; no URL, body or exception text is
        ever propagated. Redirects (3xx) fail closed because the client never
        follows them.
        """
        left = self.remaining()
        if left <= 0:
            return (DISPOSITION_TIMEOUT, 0, {}, b"")
        self._record(destination, op, key)
        try:
            with anyio.fail_after(left):
                async with client.stream("POST", url, json=dict(payload), headers=dict(headers)) as response:
                    status = response.status_code
                    if status in (301, 302, 303, 307, 308):
                        return (DISPOSITION_REDIRECT, status, {}, b"")
                    if status in (401, 403):
                        return (DISPOSITION_AUTH, status, {}, b"")
                    if status == 404:
                        return (DISPOSITION_NOT_FOUND, status, {}, b"")
                    if status in (429, 502, 503, 504) or status >= 500:
                        return (DISPOSITION_HTTP_ERROR, status, {}, b"")
                    raw = bytearray()
                    oversized = False
                    async for chunk in response.aiter_bytes():
                        raw.extend(chunk)
                        if len(raw) > MAX_BODY_BYTES:
                            oversized = True
                            break  # bounded discard; the response context closes it
                    if oversized:
                        return (DISPOSITION_OVERSIZED, status, {}, b"")
        except (TimeoutError, anyio.ClosedResourceError, anyio.BrokenResourceError):
            return (DISPOSITION_TIMEOUT, 0, {}, b"")
        except httpx.HTTPError:
            # Connection refused/reset/DNS: bounded, redacted, no retry.
            return (DISPOSITION_HTTP_ERROR, 0, {}, b"")
        try:
            parsed = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return (DISPOSITION_MALFORMED, status, {}, bytes(raw))
        if not isinstance(parsed, dict):
            return (DISPOSITION_MALFORMED, status, {}, bytes(raw))
        return (DISPOSITION_PASS, status, parsed, bytes(raw))

    async def get_json(
        self,
        client: httpx.AsyncClient,
        *,
        destination: str,
        url: str,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
        op: str,
        key: str,
    ) -> tuple[str, int, Any, bytes]:
        """One bounded GET under the shared immutable deadline.

        Same redaction/failure contract as :meth:`post_json`; the parsed body
        may be a JSON array (e.g. the vendor session list) so ``Any`` is
        returned. A non-JSON or non-list/object body is MALFORMED.
        """
        left = self.remaining()
        if left <= 0:
            return (DISPOSITION_TIMEOUT, 0, {}, b"")
        self._record(destination, op, key)
        try:
            with anyio.fail_after(left):
                async with client.stream("GET", url, params=dict(params), headers=dict(headers)) as response:
                    status = response.status_code
                    if status in (301, 302, 303, 307, 308):
                        return (DISPOSITION_REDIRECT, status, {}, b"")
                    if status in (401, 403):
                        return (DISPOSITION_AUTH, status, {}, b"")
                    if status == 404:
                        return (DISPOSITION_NOT_FOUND, status, {}, b"")
                    if status in (429, 502, 503, 504) or status >= 500:
                        return (DISPOSITION_HTTP_ERROR, status, {}, b"")
                    raw = bytearray()
                    oversized = False
                    async for chunk in response.aiter_bytes():
                        raw.extend(chunk)
                        if len(raw) > MAX_BODY_BYTES:
                            oversized = True
                            break
                    if oversized:
                        return (DISPOSITION_OVERSIZED, status, {}, b"")
        except (TimeoutError, anyio.ClosedResourceError, anyio.BrokenResourceError):
            return (DISPOSITION_TIMEOUT, 0, {}, b"")
        except httpx.HTTPError:
            return (DISPOSITION_HTTP_ERROR, 0, {}, b"")
        try:
            parsed = json.loads(bytes(raw).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return (DISPOSITION_MALFORMED, status, {}, bytes(raw))
        if not isinstance(parsed, (dict, list)):
            return (DISPOSITION_MALFORMED, status, {}, bytes(raw))
        return (DISPOSITION_PASS, status, parsed, bytes(raw))


# ---------------------------------------------------------------------------
# HoneyHive exact-session readback
# ---------------------------------------------------------------------------


async def read_honeyhive(
    client_ctx: ReadbackClient,
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    session_id: str,
    expected_run_id: str,
) -> dict[str, Any]:
    """Exact-session paginated read of ``POST /v1/events/search``.

    Pages 1..1000 with limit 100; stops once ``count`` is satisfied or a page
    is exhausted; rejects duplicates, wrong-session rows, malformed/changing
    shapes and page/item bounds. Returns verified rows plus a stable snapshot
    hash when two consecutive complete snapshots agree.
    """
    limit = 100
    search_path = "/v1/events/search"
    url = f"{base_url}{search_path}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}

    async def _read_complete() -> dict[str, Any]:
        """One full paginated exact-session read with strict reconciliation.

        Stops once ``count`` is satisfied or a page is exhausted; rejects
        duplicates, wrong-session rows, malformed/changing shapes and
        page/item bounds. Returns ``{count, rows}`` on success or
        ``{error, detail}`` on any closed failure.
        """
        page = 1
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        while True:
            if page > 1000:
                return {"error": DISPOSITION_BOUNDS, "detail": "page bound exceeded"}
            payload = {
                "filters": [{"field": "session_id", "operator": "is", "value": session_id, "type": "string"}],
                "limit": limit,
                "page": page,
            }
            disposition, status, parsed, _raw = await client_ctx.post_json(
                client,
                destination="honeyhive",
                url=url,
                payload=payload,
                headers=headers,
                op="search",
                key=f"page:{page}",
            )
            if disposition != DISPOSITION_PASS:
                return {"error": disposition, "detail": f"search page {page}"}
            events = parsed.get("events")
            count = parsed.get("count")
            if not isinstance(events, list) or not isinstance(count, int) or isinstance(count, bool):
                return {"error": DISPOSITION_SHAPE, "detail": "expected {events: list, count: int}"}
            if len(events) > limit:
                return {"error": DISPOSITION_BOUNDS, "detail": "page item bound exceeded"}
            for event in events:
                if not isinstance(event, dict):
                    return {"error": DISPOSITION_MALFORMED, "detail": "non-object event"}
                event_session = event.get("session_id")
                if event_session != session_id:
                    return {"error": DISPOSITION_WRONG_SESSION, "detail": "event session mismatch"}
                identity = next((event.get(key) for key in HH_EVENT_ID_KEYS if isinstance(event.get(key), str)), None)
                if identity is None:
                    return {"error": DISPOSITION_MALFORMED, "detail": "event without string id"}
                if identity in seen_ids:
                    return {"error": DISPOSITION_DUPLICATE, "detail": f"duplicate event id {identity}"}
                seen_ids.add(identity)
                rows.append(event)
            if len(rows) >= count:
                if len(rows) != count:
                    return {"error": DISPOSITION_SHAPE, "detail": "rows/count reconciliation failed"}
                return {"count": count, "rows": rows}
            if len(events) < limit:
                return {"error": DISPOSITION_SHAPE, "detail": "rows/count reconciliation failed"}
            page += 1

    def _snapshot(read: dict[str, Any]) -> tuple[str, str]:
        """(ordered-ids hash, ordered id list) for one complete read."""
        ids = [event.get("id", event.get("event_id")) for event in read["rows"]]
        ordered = sorted(identity for identity in ids if isinstance(identity, str))
        return stable_hash({"session": session_id, "count": read["count"], "ids": ordered}), ",".join(ordered)

    # Two stable complete post-shutdown snapshots are required (the same
    # stability contract as the LangSmith exact-ID tree).
    first = await _read_complete()
    if "error" in first:
        return {"disposition": first["error"], "detail": first["detail"]}
    first_hash, first_ids = _snapshot(first)

    second = await _read_complete()
    if "error" in second:
        return {"disposition": second["error"], "detail": second["detail"]}
    second_hash, _second_ids = _snapshot(second)

    stable = second if second_hash == first_hash else None
    if stable is None:
        # One bounded stable re-check after a short sleep inside the deadline.
        if client_ctx.remaining() <= 0:
            return {"disposition": DISPOSITION_UNSTABLE, "detail": "session snapshots disagreed and deadline elapsed"}
        await anyio.sleep(min(0.5, client_ctx.remaining()))
        third = await _read_complete()
        if "error" in third:
            return {"disposition": third["error"], "detail": third["detail"]}
        third_hash, _third_ids = _snapshot(third)
        if third_hash != first_hash:
            return {"disposition": DISPOSITION_UNSTABLE, "detail": "two equal complete snapshots not reached"}
        stable = third

    return {
        "disposition": DISPOSITION_PASS,
        "destination": "honeyhive",
        "session_id": session_id,
        "count": stable["count"],
        "ids": [event.get("id", event.get("event_id")) for event in stable["rows"]],
        "snapshot_hash": first_hash,
        "ids_hash": first_ids,
        "rows": stable["rows"],
        "detail": f"expected run {expected_run_id}; two stable complete snapshots",
    }


# ---------------------------------------------------------------------------
# LangSmith exact-ID readback
# ---------------------------------------------------------------------------


async def read_langsmith(
    client_ctx: ReadbackClient,
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    project: str,
    run_id: str,
    started_at: str,
) -> dict[str, Any]:
    """Exact ``daydream_run_id`` discovery, freeze, then exact-ID tree reads.

    The vendor stores the OpenLLMetry association property
    ``traceloop.association.properties.daydream_run_id`` as run metadata at
    ``extra.metadata.daydream_run_id`` and filters metadata with the
    ``metadata_key``/``metadata_value`` equality grammar; a dotted-key
    metadata filter is rejected. ``/runs/query`` requires an explicit
    resolved session id (a project name alone is a 400), so the verifier
    first resolves the receipt's project name to its vendor session id via
    the bounded ``GET /api/v1/sessions?name=`` lookup, then discovers with
    ``and(eq(metadata_key, 'daydream_run_id'), eq(metadata_value, run_id))``
    inside one explicit session with a bounded start time. The verifier
    requires one trace/root, freezes the vendor-returned IDs for the exact
    re-reads (never derived from HoneyHive or from the Daydream UUID), and
    re-reads the exact tree with the vendor-accepted
    ``filter: eq(trace_id, ...)`` form (the vendor's maximum page limit is
    100). The exact-ID tree must reach two equal complete snapshots within
    the deadline.
    """
    query_path = "/runs/query"
    url = f"{base_url}{query_path}"
    headers = {"x-api-key": api_key, "Content-Type": "application/json", "Accept": "application/json"}

    # Bounded project-name -> vendor-session-id resolution. The lookup is by
    # exact name; zero or ambiguous matches fail closed.
    disposition_sessions, _status, parsed_sessions, _raw = await client_ctx.get_json(
        client,
        destination="langsmith",
        url=f"{base_url}/api/v1/sessions",
        params={"name": project, "limit": 5},
        headers=headers,
        op="resolve_project",
        key=project,
    )
    if disposition_sessions != DISPOSITION_PASS:
        return {"disposition": disposition_sessions, "detail": "project resolution"}
    sessions = parsed_sessions if isinstance(parsed_sessions, list) else parsed_sessions.get("sessions")
    if not isinstance(sessions, list):
        return {"disposition": DISPOSITION_SHAPE, "detail": "project resolution expected a session list"}
    named = [
        session
        for session in sessions
        if isinstance(session, dict) and isinstance(session.get("id"), str) and session.get("name") == project
    ]
    if len(named) != 1:
        return {"disposition": DISPOSITION_NOT_FOUND, "detail": "project name must resolve to exactly one session"}
    session_id = str(named[0]["id"])

    discovery_payload = {
        "session": [session_id],
        "filter": f"and(eq(metadata_key, 'daydream_run_id'), eq(metadata_value, '{run_id}'))",
        "limit": 100,
        "start_time": started_at,
    }
    disposition, status, parsed, _raw = await client_ctx.post_json(
        client,
        destination="langsmith",
        url=url,
        payload=discovery_payload,
        headers=headers,
        op="discovery",
        key=run_id,
    )
    if disposition != DISPOSITION_PASS:
        if disposition == DISPOSITION_NOT_FOUND:
            return {"disposition": DISPOSITION_NOT_FOUND, "detail": "no trace found"}
        return {"disposition": disposition, "detail": "discovery"}
    runs = parsed.get("runs")
    if not isinstance(runs, list):
        return {"disposition": DISPOSITION_SHAPE, "detail": "expected {runs: [...]}"}
    if not runs:
        # Zero runs is an honest absence (e.g. the sanitized replay's pinned
        # historical span times land outside LangSmith's ~24h OTLP ingest
        # window — vendor-reality note in readback-matrix.json), not a root
        # ambiguity: the exact run id discovered nothing at all.
        return {"disposition": DISPOSITION_NOT_FOUND, "detail": "no trace found for the exact run id"}
    roots = [run for run in runs if isinstance(run, dict) and not run.get("parent_run_id")]
    # The exact-session filtered discovery must identify exactly one trace.
    trace_ids = {run.get("trace_id") for run in runs if isinstance(run, dict) and isinstance(run.get("trace_id"), str)}
    if len(roots) != 1 or len(trace_ids) != 1:
        return {"disposition": DISPOSITION_AMBIGUOUS_ROOT, "detail": "discovery must return exactly one root/trace"}
    root_id = roots[0].get("id")
    trace_id = next(iter(trace_ids))
    if not isinstance(root_id, str) or not isinstance(trace_id, str):
        return {"disposition": DISPOSITION_SHAPE, "detail": "root id/trace id must be strings"}

    # Exact-ID read: documented semantics — providing the returned run ID
    # ignores all other filtering arguments and reads that exact record.
    disposition_root, _status, parsed_root, _raw = await client_ctx.post_json(
        client,
        destination="langsmith",
        url=url,
        payload={"session": [session_id], "id": [root_id]},
        headers=headers,
        op="exact_root",
        key=root_id,
    )
    if disposition_root != DISPOSITION_PASS:
        return {"disposition": disposition_root, "detail": "exact root-ID read"}
    root_runs = parsed_root.get("runs")
    if not isinstance(root_runs, list) or not any(
        isinstance(run, dict) and run.get("id") == root_id for run in root_runs
    ):
        return {"disposition": DISPOSITION_SHAPE, "detail": "exact root-ID read did not return the frozen root"}

    async def _tree_read() -> dict[str, Any]:
        disposition_tree, _status, parsed_tree, _raw_tree = await client_ctx.post_json(
            client,
            destination="langsmith",
            url=url,
            payload={"session": [session_id], "filter": f"eq(trace_id, '{trace_id}')", "limit": 100},
            headers=headers,
            op="tree",
            key=trace_id,
        )
        if disposition_tree != DISPOSITION_PASS:
            return {"disposition": disposition_tree, "detail": "exact-ID tree read"}
        tree_runs = parsed_tree.get("runs")
        if not isinstance(tree_runs, list):
            return {"disposition": DISPOSITION_SHAPE, "detail": "tree expected {runs: [...]}"}
        return {"disposition": DISPOSITION_PASS, "runs": tree_runs}

    first = await _tree_read()
    if first["disposition"] != DISPOSITION_PASS:
        return first
    second = await _tree_read()
    if second["disposition"] != DISPOSITION_PASS:
        return second
    first_runs = first["runs"]
    second_runs = second["runs"]
    first_hash = stable_hash([_run_identity(r) for r in first_runs])
    second_hash = stable_hash([_run_identity(r) for r in second_runs])
    if first_hash != second_hash:
        # One bounded stable re-check after a short sleep inside the deadline.
        if client_ctx.remaining() <= 0:
            return {"disposition": DISPOSITION_UNSTABLE, "detail": "tree snapshots disagreed and deadline elapsed"}
        await anyio.sleep(min(0.5, client_ctx.remaining()))
        third = await _tree_read()
        if third["disposition"] != DISPOSITION_PASS or stable_hash(
            [_run_identity(r) for r in third["runs"]]
        ) != first_hash:
            return {"disposition": DISPOSITION_UNSTABLE, "detail": "two equal complete snapshots not reached"}
    return {
        "disposition": DISPOSITION_PASS,
        "destination": "langsmith",
        "project": project,
        "run_id": run_id,
        "root_id": root_id,
        "trace_id": trace_id,
        "count": len(second_runs),
        "runs": second_runs,
        "snapshot_hash": first_hash,
        "detail": "exact-ID tree reached two equal snapshots",
    }


def _run_identity(run: Any) -> Any:
    if not isinstance(run, dict):
        return run
    return {
        "id": run.get("id"),
        "name": run.get("name"),
        "run_type": run.get("run_type"),
        "status": run.get("status"),
        "trace_id": run.get("trace_id"),
        "parent_run_id": run.get("parent_run_id"),
    }


# ---------------------------------------------------------------------------
# Matrix-driven stored-field comparison
# ---------------------------------------------------------------------------


def load_matrix(path: Path) -> dict[str, Any]:
    """Load the checked-in machine-readable matrix subset; malformed = fail closed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadbackError(f"Unable to load matrix subset: {exc}") from None
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ReadbackError("Matrix subset schema_version must be 1")
    for section in ("honeyhive", "langsmith"):
        if not isinstance(data.get(section), dict):
            raise ReadbackError(f"Matrix subset section {section} must be an object")
    return data


def _has_forbidden_key(value: Any, forbidden: list[str]) -> str | None:
    """Recursively check every mapping key for a forbidden private vendor field."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str):
                for candidate in forbidden:
                    if candidate == key or (candidate and key.startswith(candidate)):
                        return key
            found = _has_forbidden_key(child, forbidden)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _has_forbidden_key(child, forbidden)
            if found is not None:
                return found
    return None


def _type_ok(expected: str, value: Any) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "bool":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string_or_null":
        return value is None or isinstance(value, str)
    return True


def compare_stored(
    data: dict[str, Any],
    matrix: dict[str, Any],
    *,
    acceptance_kind: str,
) -> list[dict[str, Any]]:
    """Compare stored native rows against the matrix subset; all-pass returns [].

    Checks required keys/types, conditional keys (when present), forbidden
    vendor fields and exact expected-shape reconciliation for the sanitized
    protocol replay. Honest dispositions only: anything unverifiable is a
    `MISSING_FIELD`/`SHAPE` row, never an invented pass.
    """
    rows: list[dict[str, Any]] = []

    def _fail(code: str, field: str, detail: str) -> None:
        rows.append({"field": field, "disposition": code, "detail": detail})

    forbidden = matrix.get("forbidden_vendor_fields", [])
    hh_required: dict[str, Any] = matrix.get("honeyhive", {}).get("required_event_keys", {})
    hh_conditional: dict[str, Any] = matrix.get("honeyhive", {}).get("conditional_event_keys", {})
    ls_required: dict[str, Any] = matrix.get("langsmith", {}).get("required_run_keys", {})
    ls_conditional: dict[str, Any] = matrix.get("langsmith", {}).get("conditional_run_keys", {})
    ls_identity_key = matrix.get("langsmith", {}).get("identity_metadata_key", "daydream_run_id")
    expected_shape = matrix.get("expected_shape", {}).get(acceptance_kind)

    hh: dict[str, Any] = data.get("honeyhive") or {}
    ls: dict[str, Any] = data.get("langsmith") or {}

    if isinstance(hh.get("rows"), list):
        for event in hh["rows"]:
            for key, expected in hh_required.items():
                if key not in event:
                    _fail(DISPOSITION_MISSING_FIELD, f"honeyhive.event.{key}", "required key absent")
                elif not _type_ok(str(expected), event.get(key)):
                    _fail(DISPOSITION_SHAPE, f"honeyhive.event.{key}", f"expected {expected}")
            for key, expected in hh_conditional.items():
                # The vendor elides empty containers (e.g. metrics/feedback on
                # non-billed events) and synthesizes aggregate session events;
                # keys are type-checked when the vendor returns them.
                if key in event and not _type_ok(str(expected), event.get(key)):
                    _fail(DISPOSITION_SHAPE, f"honeyhive.event.{key}", f"expected {expected}")
            found = _has_forbidden_key(event, forbidden)
            if found is not None:
                _fail(DISPOSITION_SHAPE, f"honeyhive.forbidden.{found}", "forbidden private vendor field present")

    if isinstance(ls.get("runs"), list):
        for run in ls["runs"]:
            for key, expected in ls_required.items():
                if key not in run:
                    _fail(DISPOSITION_MISSING_FIELD, f"langsmith.run.{key}", "required key absent")
                elif not _type_ok(str(expected), run.get(key)):
                    _fail(DISPOSITION_SHAPE, f"langsmith.run.{key}", f"expected {expected}")
            for key, expected in ls_conditional.items():
                if key in run and not _type_ok(str(expected), run.get(key)):
                    _fail(DISPOSITION_SHAPE, f"langsmith.run.{key}", f"expected {expected}")
            # The exact-filtered discovery identity must survive into the tree.
            # The vendor stores the OpenLLMetry association property as run
            # metadata at ``extra.metadata``; top-level ``metadata`` is not
            # projected. A run record without the stored identity key is a
            # missing required vendor field (honest MISSING_FIELD row).
            extra = run.get("extra") if isinstance(run.get("extra"), dict) else {}
            stored_metadata = extra.get("metadata") if isinstance(extra.get("metadata"), dict) else {}
            if stored_metadata:
                run_identity = walk_metadata(stored_metadata, ls_identity_key)
                if run_identity is None:
                    missing = f"langsmith.run.extra.metadata.{ls_identity_key}"
                    _fail(DISPOSITION_MISSING_FIELD, missing, "required key absent")
                elif run_identity != ls.get("run_id"):
                    wrong = f"langsmith.run.extra.metadata.{ls_identity_key}"
                    _fail(DISPOSITION_WRONG_SESSION, wrong, "run identity mismatch")
            else:
                _fail(DISPOSITION_MISSING_FIELD, "langsmith.run.extra.metadata", "required key absent")
            found = _has_forbidden_key(run, forbidden)
            if found is not None:
                _fail(DISPOSITION_SHAPE, f"langsmith.forbidden.{found}", "forbidden private vendor field present")

    if expected_shape is not None and acceptance_kind == "sanitized_protocol_replay":
        _reconcile_replay(hh, ls, expected_shape, _fail)

    if expected_shape is not None and acceptance_kind == "representative_real_run":
        # Structural minimums on the stored trees: one root/one session is
        # already enforced by the exact-session/exact-trace reads, so the
        # remaining expectations are the descendant minimums. A tree without
        # a generation child or a tool sibling is not a complete agent tree.
        generations_min = int(expected_shape.get("generations_min", 0))
        tools_min = int(expected_shape.get("tools_min", 0))
        if isinstance(ls.get("runs"), list):
            llm = sum(1 for run in ls["runs"] if isinstance(run, dict) and run.get("run_type") == "llm")
            tools = sum(1 for run in ls["runs"] if isinstance(run, dict) and run.get("run_type") == "tool")
            if llm < generations_min:
                _fail(DISPOSITION_SHAPE, "langsmith.tree.generations", f"expected at least {generations_min}")
            if tools < tools_min:
                _fail(DISPOSITION_SHAPE, "langsmith.tree.tools", f"expected at least {tools_min}")
        if isinstance(hh.get("rows"), list):
            models = sum(1 for event in hh["rows"] if isinstance(event, dict) and event.get("event_type") == "model")
            tools = sum(1 for event in hh["rows"] if isinstance(event, dict) and event.get("event_type") == "tool")
            if models < generations_min:
                _fail(DISPOSITION_SHAPE, "honeyhive.session.model_events", f"expected at least {generations_min}")
            if tools < tools_min:
                _fail(DISPOSITION_SHAPE, "honeyhive.session.tool_events", f"expected at least {tools_min}")

    return rows


def _reconcile_replay(
    hh: dict[str, Any],
    ls: dict[str, Any],
    expected: dict[str, Any],
    fail: Any,
) -> None:
    """Exact historical-equivalent reconciliation of the sanitized replay.

    The generation identity/timing/usage evidence is searched flat and nested
    in stored metadata/config (the canonical destination mapping is
    documented per-field in docs/observability-fields.md; the verifier uses
    the same key spellings). The billing duplicate invariant (usage on exactly
    one billable owner) is enforced for both destinations, and the fixture's
    native session/response identities never substitute for Daydream
    run/session identities.
    """
    session = expected.get("session_id")  # native pi session identity (fixture)
    model = expected.get("model_name")
    provider = expected.get("provider_name")
    response_ids = expected.get("response_ids", [])
    start_ms = expected.get("native_started_at_unix_ms")
    sealed_ns = expected.get("sealed_end_unix_ns")
    duration_ns = expected.get("duration_ns")
    input_tokens = expected.get("normalized_input_tokens")
    output_tokens = expected.get("output_tokens")
    reasoning_tokens = expected.get("reasoning_tokens")
    cost = expected.get("reported_cost_usd")

    usage_keys = (
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.reasoning.output_tokens",
        "gen_ai.usage.cost",
    )
    billed_by_destination: dict[str, int] = {}

    for destination, container, rows_key in (("honeyhive", hh, "rows"), ("langsmith", ls, "runs")):
        rows = container.get(rows_key) if isinstance(container, dict) else None
        if not isinstance(rows, list):
            fail(DISPOSITION_MISSING_FIELD, f"{destination}.rows", "no stored rows to reconcile")
            continue
        for row_index, row in enumerate(rows):
            metadata = {}
            if isinstance(row, dict):
                # The vendor stores the OpenLLMetry association properties at
                # ``extra.metadata``; top-level ``metadata``/``config`` are
                # the legacy canonical mappings the verifier still accepts on
                # equal footing (only for lookup — storage shape is checked by
                # compare_stored). The first-seen key wins, so the vendor-actual
                # location takes precedence.
                for candidate_name in ("extra.metadata", "metadata", "config"):
                    candidate: Any = row
                    ok = True
                    for part in candidate_name.split("."):
                        if not isinstance(candidate, dict):
                            ok = False
                            break
                        candidate = candidate.get(part)
                    if ok and isinstance(candidate, dict):
                        for key, value in candidate.items():
                            metadata.setdefault(str(key), value)
            # Separation invariant (plan §250): a native pi session identity
            # never substitutes for the Daydream session identity.
            daydream_session = walk_metadata(metadata, "daydream.session.id")
            if session is not None and daydream_session == session:
                fail(
                    DISPOSITION_WRONG_SESSION,
                    f"{destination}.rows[{row_index}].daydream.session.id",
                    "native pi session must never substitute for the Daydream session id",
                )
            usage_seen = 0
            for key in usage_keys:
                value = walk_metadata(metadata, key)
                if value is None:
                    continue
                usage_seen += 1
                if key == "gen_ai.usage.input_tokens" and input_tokens is not None and value != input_tokens:
                    fail(DISPOSITION_SHAPE, f"{destination}.rows[{row_index}].{key}", "input token mismatch")
                if key == "gen_ai.usage.output_tokens" and output_tokens is not None and value != output_tokens:
                    fail(DISPOSITION_SHAPE, f"{destination}.rows[{row_index}].{key}", "output token mismatch")
                if (
                    key == "gen_ai.usage.reasoning.output_tokens"
                    and reasoning_tokens is not None
                    and value != reasoning_tokens
                ):
                    fail(DISPOSITION_SHAPE, f"{destination}.rows[{row_index}].{key}", "reasoning token mismatch")
                if key == "gen_ai.usage.cost" and cost is not None:
                    if not isinstance(value, (int, float)) or abs(float(value) - float(cost)) > 1e-9:
                        fail(DISPOSITION_SHAPE, f"{destination}.rows[{row_index}].{key}", "cost mismatch")
            if usage_seen:
                billed_by_destination[destination] = billed_by_destination.get(destination, 0) + 1
                # The billable owner also carries the exact response identity.
                # Absent keys are tolerated (the vendor does not always project
                # the response-model key — e.g. HoneyHive stores the configured
                # request model at config.model), identical to how the provider
                # key below is treated; a PRESENT wrong value still fails.
                stored_model = walk_metadata(metadata, "gen_ai.response.model")
                if model is not None and stored_model not in (None, model):
                    fail(
                        DISPOSITION_SHAPE,
                        f"{destination}.rows[{row_index}].gen_ai.response.model",
                        "billed owner model mismatch",
                    )
                if provider is not None and walk_metadata(metadata, "gen_ai.provider.name") not in (provider, None):
                    fail(
                        DISPOSITION_SHAPE,
                        f"{destination}.rows[{row_index}].gen_ai.provider.name",
                        "billed owner provider mismatch",
                    )
                stored_response = walk_metadata(metadata, "gen_ai.response.id")
                if stored_response is not None and response_ids and stored_response not in response_ids:
                    fail(
                        DISPOSITION_SHAPE,
                        f"{destination}.rows[{row_index}].gen_ai.response.id",
                        "stored response id is not one of the pinned replay responses",
                    )
            # Generation timing provenance (replay identity). The manifest pins
            # the FIRST generation's exact historical interval; later
            # generations in the same fixture have their own distinct
            # native/sealed times and must not be compared to the first's pins.
            stored_native_ms = walk_metadata(metadata, "daydream.generation.native_started_at_unix_ms")
            is_pinned_generation = (
                stored_native_ms is not None and start_ms is not None and stored_native_ms == start_ms
            )
            sealed = walk_metadata(metadata, "daydream.generation.sealed_end_unix_ns")
            elapsed = walk_metadata(metadata, "daydream.generation.duration_ns")
            if is_pinned_generation:
                # The pinned first generation: exact historical-equivalent
                # equality for native start, sealed end and duration.
                if stored_native_ms is not None and start_ms is not None and stored_native_ms != start_ms:
                    fail(
                        DISPOSITION_SHAPE,
                        f"{destination}.rows[{row_index}].daydream.generation.native_started_at_unix_ms",
                        "native start mismatch",
                    )
                if sealed is not None and sealed_ns is not None and sealed != sealed_ns:
                    fail(
                        DISPOSITION_SHAPE,
                        f"{destination}.rows[{row_index}].daydream.generation.sealed_end_unix_ns",
                        "sealed end mismatch",
                    )
                if elapsed is not None and duration_ns is not None and elapsed != duration_ns:
                    fail(
                        DISPOSITION_SHAPE,
                        f"{destination}.rows[{row_index}].daydream.generation.duration_ns",
                        "duration mismatch",
                    )
            elif stored_native_ms is not None:
                # A later generation in the same fixture: never compared to the
                # first generation's pins; only itself-consistency between its
                # own stored native start, sealed end and duration is checked.
                if isinstance(sealed, int) and isinstance(stored_native_ms, int):
                    own_duration = sealed - stored_native_ms * 1_000_000
                    if elapsed is not None and elapsed != own_duration:
                        fail(
                            DISPOSITION_SHAPE,
                            f"{destination}.rows[{row_index}].daydream.generation.duration_ns",
                            "non-pinned generation duration inconsistent with its own timing",
                        )
                elif elapsed is not None and sealed is not None:
                    fail(
                        DISPOSITION_SHAPE,
                        f"{destination}.rows[{row_index}].daydream.generation.sealed_end_unix_ns",
                        "non-pinned generation timing is incomplete (sealed/native not both ints)",
                    )
        # No parent/child billing duplicate: usage/cost on exactly ONE
        # billable owner per destination.
        if billed_by_destination.get(destination, 0) > 1:
            fail(DISPOSITION_SHAPE, f"{destination}.billing_owner", "usage/cost present on more than one owner")
        if not isinstance(rows, list) or not rows:
            fail(DISPOSITION_MISSING_FIELD, f"{destination}.rows", "no rows for replay reconciliation")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _verify(
    receipt: Mapping[str, Any],
    matrix: dict[str, Any],
    budget_s: float,
) -> dict[str, Any]:
    ctx = ReadbackClient(budget_s=budget_s)
    ctx.start()
    result: dict[str, Any] = {
        "schema_version": 1,
        "dispositions": [],
        "request_log": ctx.request_log,
        "honeyhive": None,
        "langsmith": None,
    }
    destinations = set(receipt["destinations"])
    enabled: list[str] = []

    async with httpx.AsyncClient(
        trust_env=False,
        follow_redirects=False,
        timeout=httpx.Timeout(None),  # the immutable fail_after budget is the only timeout
    ) as client:
        if "honeyhive" in destinations:
            enabled.append("honeyhive")
            base = validate_base_url(os.environ.get("HH_API_URL", ""), "HH_API_URL")
            key = os.environ.get("HH_API_KEY", "")
            if not key:
                return {
                    **result,
                    "terminal": DISPOSITION_AUTH,
                    "detail": "HH_API_KEY is required for honeyhive readback",
                }
            result["honeyhive"] = await read_honeyhive(
                ctx,
                client,
                base_url=base,
                api_key=key,
                session_id=str(receipt["session_id"]),
                expected_run_id=str(receipt["run_id"]),
            )
        if "langsmith" in destinations:
            enabled.append("langsmith")
            base = validate_base_url(
                os.environ.get("LANGSMITH_ENDPOINT", _LANGSMITH_DEFAULT_ENDPOINT), "LANGSMITH_ENDPOINT"
            )
            key = os.environ.get("LANGSMITH_API_KEY", "")
            if not key:
                return {
                    **result,
                    "terminal": DISPOSITION_AUTH,
                    "detail": "LANGSMITH_API_KEY is required for langsmith readback",
                }
            result["langsmith"] = await read_langsmith(
                ctx,
                client,
                base_url=base,
                api_key=key,
                project=str(receipt["langsmith_project"]),
                run_id=str(receipt["run_id"]),
                started_at=str(receipt["started_at"]),
            )
    result["enabled_destinations"] = enabled
    rows = compare_stored(result, matrix, acceptance_kind=str(receipt["acceptance_kind"]))
    result["matrix_rows"] = rows
    result["dispositions"] = [row["disposition"] for row in rows]
    destination_failure: str | None = None
    for dest in enabled:
        section = result.get(dest)
        if isinstance(section, dict) and section.get("disposition") != DISPOSITION_PASS:
            destination_failure = str(section.get("disposition"))
            break
    if destination_failure is not None:
        result["terminal"] = destination_failure
    elif rows:
        result["terminal"] = rows[0]["disposition"]
    else:
        result["terminal"] = DISPOSITION_PASS
    result["ui_inspected"] = False
    result["stored_contract_passed"] = result["terminal"] == DISPOSITION_PASS and all(
        (result[dest] or {}).get("disposition") == DISPOSITION_PASS for dest in enabled
    )
    return result


def run_verify(receipt_path: Path, result_path: Path, *, budget_s: float, matrix_path: Path) -> int:
    """Entry point shared by the CLI and hermetic tests; returns the exit code."""
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ReadbackError("Receipt must be a JSON object")
        validate_receipt(receipt)
        matrix = load_matrix(matrix_path)
    except (OSError, json.JSONDecodeError, ReadbackError) as exc:
        # Receipt validation precedes client construction; the diagnostic is
        # fixed/redacted and no network work has happened.
        result = {
            "schema_version": 1,
            "terminal": DISPOSITION_SHAPE,
            "detailed": False,
            "detail": str(exc),
            "request_log": [],
        }
        atomic_write_json(result_path, result)
        print(f"verdict=fail disposition={DISPOSITION_SHAPE}")
        return 1
    result = anyio.run(_verify, receipt, matrix, budget_s)
    destination_summaries: dict[str, dict[str, Any]] = {}
    for destination in ("honeyhive", "langsmith"):
        section = result.get(destination)
        if not isinstance(section, dict):
            continue
        summary: dict[str, Any] = {"destination": destination, "disposition": section.get("disposition")}
        for key in ("session_id", "count", "root_id", "trace_id", "project", "run_id", "snapshot_hash", "detail"):
            # ``detail`` carries only the verifier's fixed redacted phrases.
            if section.get(key) is not None:
                summary[key] = section[key]
        destination_summaries[destination] = summary
    atomic_write_json(
        result_path,
        {
            key: value
            for key, value in result.items()
            if key in ("schema_version", "terminal", "dispositions", "request_log", "enabled_destinations",
                       "matrix_rows", "ui_inspected", "stored_contract_passed", "detail")
        }
        | {"destinations": destination_summaries},
    )
    passed = result.get("stored_contract_passed") is True
    terminal = str(result.get("terminal", DISPOSITION_SHAPE))
    print(f"verdict={'pass' if passed else 'fail'} disposition={terminal}")
    if not passed:
        for destination in ("honeyhive", "langsmith"):
            section = result.get(destination)
            if isinstance(section, dict) and section.get("disposition") != DISPOSITION_PASS:
                print(f"destination={destination} disposition={section.get('disposition')}")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify stored native traces against an immutable receipt.")
    parser.add_argument("--receipt", required=True, type=Path, help="immutable canonical input receipt JSON")
    parser.add_argument("--result", required=True, type=Path, help="atomic result receipt JSON output path")
    parser.add_argument("--deadline", type=float, default=30.0, help="one immutable budget in seconds (default 30)")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/observability_contract/readback-matrix.json",
        help="checked-in machine-readable matrix subset",
    )
    args = parser.parse_args(argv)
    try:
        return run_verify(args.receipt, args.result, budget_s=args.deadline, matrix_path=args.matrix)
    except ReadbackError as exc:
        print(f"verdict=fail disposition=READBACK_CONFIGURATION detail={exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
