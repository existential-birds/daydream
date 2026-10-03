#!/usr/bin/env python3
"""Verify stored HoneyHive/LangSmith trees against an immutable canonical receipt.

Validate the input receipt and base URLs before creating one trust_env=False,
follow_redirects=False AsyncClient. JSON only; never initialize OTLP machinery.
Use one immutable monotonic deadline for requests, capped streaming reads,
polling, and backoff. Close each stream before continuing and on cancellation;
expiry emits one fixed redacted timeout result and starts no further request.

HoneyHive uses bounded exact-session pagination with strict shape, count, and
identity checks. LangSmith resolves one explicit project, discovers by exact
run-id metadata and bounded start time, freezes vendor root/trace ids, then
reads that tree. Never derive vendor IDs from Daydream or another destination.
Require two equal complete snapshots within the shared deadline.

Compare stored types, parent/root relationships, generation/turn/tool counts,
identity/timing provenance, resource identity, and unique billing ownership
against the checked-in readback-matrix.json and sanitized replay. Refuse
forbidden/private/ambient fields. Output only IDs, counts, names, types,
booleans, stable hashes, and matrix results; storage proof is not UI proof.

Keys come from environment variables. --receipt is never modified; --result
is written atomically. --deadline defaults to 30 seconds, and --matrix to the
checked-in observability contract subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, Mapping

import anyio
import httpx

from daydream.json_utils import atomic_write_json
from daydream.observability.config import ObservabilityError, validate_endpoint

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
    """Use the exporter endpoint contract without loading OTLP transport machinery."""
    try:
        return validate_endpoint(raw, setting)
    except ObservabilityError as exc:
        raise ReadbackError(str(exc)) from None


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
    """Find a dotted Daydream key in flat or nested metadata/config; return None when absent."""
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

    async def request_json(
        self,
        client: httpx.AsyncClient,
        method: Literal["GET", "POST"],
        *,
        destination: str,
        url: str,
        headers: Mapping[str, str],
        op: str,
        key: str,
        payload: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> tuple[str, int, Any, bytes]:
        """One bounded request under the shared immutable deadline."""
        request_kwargs: dict[str, Any] = (
            {"json": dict(payload or {})} if method == "POST" else {"params": dict(params or {})}
        )
        left = self.remaining()
        if left <= 0:
            return (DISPOSITION_TIMEOUT, 0, {}, b"")
        self._record(destination, op, key)
        try:
            with anyio.fail_after(left):
                async with client.stream(method, url, **request_kwargs, headers=dict(headers)) as response:
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
        if not isinstance(parsed, dict if method == "POST" else (dict, list)):
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
    """Read exact-session POST /v1/events/search pages (100 rows, at most 1000 pages).

    Stop at declared count or exhaustion; reject duplicate/wrong-session rows and
    shape/count/bound violations. Return rows and a stable hash after equal snapshots.
    """
    limit = 100
    search_path = "/v1/events/search"
    url = f"{base_url}{search_path}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}

    async def _read_complete() -> dict[str, Any]:
        """Read one bounded complete session, stopping at count or page exhaustion.

        Reject duplicates, wrong sessions, or shape/count changes; return {count, rows}
        or a closed {error, detail} failure.
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
            disposition, status, parsed, _raw = await client_ctx.request_json(
                client,
                "POST",
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

    def _snapshot(read: dict[str, Any]) -> str:
        """Fingerprint the admitted identities of one complete read."""
        ids = [event.get("id", event.get("event_id")) for event in read["rows"]]
        ordered = sorted(identity for identity in ids if isinstance(identity, str))
        return stable_hash({"session": session_id, "count": read["count"], "ids": ordered})

    # Two stable complete post-shutdown snapshots are required (the same
    # stability contract as the LangSmith exact-ID tree).
    first = await _read_complete()
    if "error" in first:
        return {"disposition": first["error"], "detail": first["detail"]}
    first_hash = _snapshot(first)

    second = await _read_complete()
    if "error" in second:
        return {"disposition": second["error"], "detail": second["detail"]}
    stable = second if _snapshot(second) == first_hash else None
    if stable is None:
        # One bounded stable re-check after a short sleep inside the deadline.
        if client_ctx.remaining() <= 0:
            return {"disposition": DISPOSITION_UNSTABLE, "detail": "session snapshots disagreed and deadline elapsed"}
        await anyio.sleep(min(0.5, client_ctx.remaining()))
        third = await _read_complete()
        if "error" in third:
            return {"disposition": third["error"], "detail": third["detail"]}
        if _snapshot(third) != first_hash:
            return {"disposition": DISPOSITION_UNSTABLE, "detail": "two equal complete snapshots not reached"}
        stable = third

    ids = [event.get("id", event.get("event_id")) for event in stable["rows"]]
    return {
        "disposition": DISPOSITION_PASS,
        "destination": "honeyhive",
        "session_id": session_id,
        "count": stable["count"],
        "ids": ids,
        "snapshot_hash": first_hash,
        "ids_hash": ",".join(sorted(identity for identity in ids if isinstance(identity, str))),
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
    """Discover by exact Daydream run metadata, then freeze and reread native root/trace IDs.

    Resolve the receipt's project name with GET /api/v1/sessions?name=; /runs/query
    requires its session id. Match metadata_key=daydream_run_id and metadata_value,
    not a dotted-key filter. Bound start time, require one root/trace, then use the
    vendor's eq(trace_id, ...) filter with page limit 100 until two complete trees
    agree within the deadline.
    """
    query_path = "/runs/query"
    url = f"{base_url}{query_path}"
    headers = {"x-api-key": api_key, "Content-Type": "application/json", "Accept": "application/json"}

    # Bounded project-name -> vendor-session-id resolution. The lookup is by
    # exact name; zero or ambiguous matches fail closed.
    disposition_sessions, _status, parsed_sessions, _raw = await client_ctx.request_json(
        client,
        "GET",
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
    disposition, status, parsed, _raw = await client_ctx.request_json(
        client,
        "POST",
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
        # Zero exact-ID matches are honest absence, including historical spans outside vendor ingest windows.
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
    disposition_root, _status, parsed_root, _raw = await client_ctx.request_json(
        client,
        "POST",
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
        disposition_tree, _status, parsed_tree, _raw_tree = await client_ctx.request_json(
            client,
            "POST",
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
    stable = await _tree_read()
    if stable["disposition"] != DISPOSITION_PASS:
        return stable
    first_hash = stable_hash([_run_identity(r) for r in first["runs"]])
    if stable_hash([_run_identity(r) for r in stable["runs"]]) != first_hash:
        # One bounded stable re-check after a short sleep inside the deadline.
        if client_ctx.remaining() <= 0:
            return {"disposition": DISPOSITION_UNSTABLE, "detail": "tree snapshots disagreed and deadline elapsed"}
        await anyio.sleep(min(0.5, client_ctx.remaining()))
        stable = await _tree_read()
        if (
            stable["disposition"] != DISPOSITION_PASS
            or stable_hash([_run_identity(r) for r in stable["runs"]]) != first_hash
        ):
            return {"disposition": DISPOSITION_UNSTABLE, "detail": "two equal complete snapshots not reached"}
    return {
        "disposition": DISPOSITION_PASS,
        "destination": "langsmith",
        "project": project,
        "run_id": run_id,
        "root_id": root_id,
        "trace_id": trace_id,
        "count": len(stable["runs"]),
        "runs": stable["runs"],
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
    """Return failed matrix rows, or [] when all stored type/shape/privacy checks pass.

    Unverifiable fields yield MISSING_FIELD/SHAPE, never inferred success.
    """
    rows: list[dict[str, Any]] = []

    def _fail(code: str, field: str, detail: str) -> None:
        rows.append({"field": field, "disposition": code, "detail": detail})

    forbidden = matrix.get("forbidden_vendor_fields", [])
    expected_shape = matrix.get("expected_shape", {}).get(acceptance_kind)
    hh: dict[str, Any] = data.get("honeyhive") or {}
    ls: dict[str, Any] = data.get("langsmith") or {}

    for destination, container, rows_key, noun in (
        ("honeyhive", hh, "rows", "event"),
        ("langsmith", ls, "runs", "run"),
    ):
        stored = container.get(rows_key)
        if not isinstance(stored, list):
            continue
        section = matrix.get(destination, {})
        required = section.get(f"required_{noun}_keys", {})
        conditional = section.get(f"conditional_{noun}_keys", {})
        for row in stored:
            for key, expected in required.items():
                field = f"{destination}.{noun}.{key}"
                if key not in row:
                    _fail(DISPOSITION_MISSING_FIELD, field, "required key absent")
                elif not _type_ok(str(expected), row.get(key)):
                    _fail(DISPOSITION_SHAPE, field, f"expected {expected}")
            # Vendors may elide empty containers; only returned values have a type contract.
            for key, expected in conditional.items():
                if key in row and not _type_ok(str(expected), row.get(key)):
                    _fail(DISPOSITION_SHAPE, f"{destination}.{noun}.{key}", f"expected {expected}")
            if destination == "langsmith":
                # Discovery identity must survive at the vendor's actual storage path.
                identity_key = section.get("identity_metadata_key", "daydream_run_id")
                extra = row.get("extra") if isinstance(row.get("extra"), dict) else {}
                metadata = extra.get("metadata") if isinstance(extra.get("metadata"), dict) else {}
                field = "langsmith.run.extra.metadata"
                if metadata:
                    identity = walk_metadata(metadata, identity_key)
                    field = f"{field}.{identity_key}"
                    if identity is None:
                        _fail(DISPOSITION_MISSING_FIELD, field, "required key absent")
                    elif identity != ls.get("run_id"):
                        _fail(DISPOSITION_WRONG_SESSION, field, "run identity mismatch")
                else:
                    _fail(DISPOSITION_MISSING_FIELD, field, "required key absent")
            found = _has_forbidden_key(row, forbidden)
            if found is not None:
                _fail(DISPOSITION_SHAPE, f"{destination}.forbidden.{found}", "forbidden private vendor field present")

    if expected_shape is not None and acceptance_kind == "sanitized_protocol_replay":
        _reconcile_replay(hh, ls, expected_shape, _fail)

    if expected_shape is not None and acceptance_kind == "representative_real_run":
        # Exact reads enforce one root/session; require generation/tool descendants for a complete agent tree.
        minimums = (int(expected_shape.get("generations_min", 0)), int(expected_shape.get("tools_min", 0)))
        for container, rows_key, type_key, prefix, checks in (
            (ls, "runs", "run_type", "langsmith.tree", (("llm", "generations"), ("tool", "tools"))),
            (hh, "rows", "event_type", "honeyhive.session", (("model", "model_events"), ("tool", "tool_events"))),
        ):
            if not isinstance(container.get(rows_key), list):
                continue
            for (kind, field), minimum in zip(checks, minimums, strict=True):
                count = sum(1 for row in container[rows_key] if isinstance(row, dict) and row.get(type_key) == kind)
                if count < minimum:
                    _fail(DISPOSITION_SHAPE, f"{prefix}.{field}", f"expected at least {minimum}")

    return rows


def _reconcile_replay(
    hh: dict[str, Any],
    ls: dict[str, Any],
    expected: dict[str, Any],
    fail: Any,
) -> None:
    """Reconcile sanitized replay identities, timing, and usage from flat/nested metadata.

    Require exactly one billable owner. Native response/session IDs never substitute
    for Daydream run/session identity; mappings live in observability-fields.md.
    """
    session = expected.get("session_id")  # native pi session identity (fixture)
    model = expected.get("model_name")
    provider = expected.get("provider_name")
    response_ids = expected.get("response_ids", [])
    start_ms = expected.get("native_started_at_unix_ms")
    sealed_ns = expected.get("sealed_end_unix_ns")
    duration_ns = expected.get("duration_ns")
    usage_expectations = (
        ("gen_ai.usage.input_tokens", expected.get("normalized_input_tokens"), "input token mismatch"),
        ("gen_ai.usage.output_tokens", expected.get("output_tokens"), "output token mismatch"),
        ("gen_ai.usage.reasoning.output_tokens", expected.get("reasoning_tokens"), "reasoning token mismatch"),
        ("gen_ai.usage.cost", expected.get("reported_cost_usd"), "cost mismatch"),
    )
    billed_by_destination: dict[str, int] = {}

    for destination, container, rows_key in (("honeyhive", hh, "rows"), ("langsmith", ls, "runs")):
        rows = container.get(rows_key) if isinstance(container, dict) else None
        if not isinstance(rows, list):
            fail(DISPOSITION_MISSING_FIELD, f"{destination}.rows", "no stored rows to reconcile")
            continue
        for row_index, row in enumerate(rows):
            metadata: dict[str, Any] = {}
            if isinstance(row, dict):
                # First-seen keys retain extra.metadata > metadata > config precedence.
                extra = row.get("extra")
                candidates = (
                    extra.get("metadata") if isinstance(extra, dict) else None,
                    row.get("metadata"),
                    row.get("config"),
                )
                for candidate in candidates:
                    if isinstance(candidate, dict):
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
            for key, expected_value, mismatch in usage_expectations:
                value = walk_metadata(metadata, key)
                if value is None:
                    continue
                usage_seen += 1
                if expected_value is None:
                    continue
                if key == "gen_ai.usage.cost":
                    differs = not isinstance(value, (int, float)) or abs(float(value) - float(expected_value)) > 1e-9
                else:
                    differs = value != expected_value
                if differs:
                    fail(DISPOSITION_SHAPE, f"{destination}.rows[{row_index}].{key}", mismatch)
            if usage_seen:
                billed_by_destination[destination] = billed_by_destination.get(destination, 0) + 1
                # Absent response-model/provider keys are tolerated; present wrong identities fail.
                for key, identity, detail in (
                    ("gen_ai.response.model", model, "billed owner model mismatch"),
                    ("gen_ai.provider.name", provider, "billed owner provider mismatch"),
                ):
                    if identity is not None and walk_metadata(metadata, key) not in (None, identity):
                        fail(
                            DISPOSITION_SHAPE,
                            f"{destination}.rows[{row_index}].{key}",
                            detail,
                        )
                stored_response = walk_metadata(metadata, "gen_ai.response.id")
                if stored_response is not None and response_ids and stored_response not in response_ids:
                    fail(
                        DISPOSITION_SHAPE,
                        f"{destination}.rows[{row_index}].gen_ai.response.id",
                        "stored response id is not one of the pinned replay responses",
                    )
            # Only the first generation uses pinned historical times; later generations validate their own intervals.
            stored_native_ms = walk_metadata(metadata, "daydream.generation.native_started_at_unix_ms")
            is_pinned_generation = (
                stored_native_ms is not None and start_ms is not None and stored_native_ms == start_ms
            )
            sealed = walk_metadata(metadata, "daydream.generation.sealed_end_unix_ns")
            elapsed = walk_metadata(metadata, "daydream.generation.duration_ns")
            if is_pinned_generation:
                # The pinned first generation: exact historical-equivalent
                # equality for sealed end and duration.
                for key, stored, pinned, detail in (
                    ("sealed_end_unix_ns", sealed, sealed_ns, "sealed end mismatch"),
                    ("duration_ns", elapsed, duration_ns, "duration mismatch"),
                ):
                    if stored is not None and pinned is not None and stored != pinned:
                        fail(
                            DISPOSITION_SHAPE,
                            f"{destination}.rows[{row_index}].daydream.generation.{key}",
                            detail,
                        )
            elif stored_native_ms is not None:
                # Later generations must satisfy their own native-start/sealed-end/duration relationship.
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
            if key
            in (
                "schema_version",
                "terminal",
                "dispositions",
                "request_log",
                "enabled_destinations",
                "matrix_rows",
                "ui_inspected",
                "stored_contract_passed",
                "detail",
            )
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
