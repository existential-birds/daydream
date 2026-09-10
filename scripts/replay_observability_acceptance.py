#!/usr/bin/env python3
"""Sanitized Pi protocol replay through the REAL Daydream trace path.

P18/#1156 Task 6 replay tool. This is a separate operator boundary, never a
test monkeypatch or a model call. It:

- accepts ONLY the checked-in manifest-pinned sanitized fixture (SHA-256 and
  byte length) and a clean public disposable repository whose origin remote
  appears in the manifest's public-repo allowlist (identity);
- requires a caller-provided external fake ``pi`` executable at the subprocess
  boundary: the file must be executable, must NOT be the real ``pi`` resolved
  from PATH, and must replay the pinned fixture byte-for-byte to stdout when
  invoked with ``--mode json`` (proves it is fixture-derived, not the real
  CLI);
- validates the exact destination set (local generic OTLP + the two explicitly
  enabled vendor destinations) and every authorization value BEFORE
  constructing exporters or sending anything;
- sets the admitted resource marker ``daydream.acceptance.kind=
  sanitized_protocol_replay`` (required via ``DAYDREAM_ACCEPTANCE_KIND``);
- invokes the ACTUAL ``PiBackend``/``run_agent``/``trace_run`` path once with
  only the subprocess boundary faked, pointing the local generic OTLP at its
  OWN bounded loopback receiver (the local-wire oracle; loopback-only, no
  external access), requires local wire success, shuts down normally, and
  writes an immutable canonical receipt labeled ``sanitized_protocol_replay``
  with model-call count ``0``, operational cost ``$0`` and hashes/IDs only;
- reproduces the manifest-pinned host receipt clock for ``message_end`` so the
  exact 395.332-second historical interval and native-ms conversion are
  reconciled deterministically (the pin is applied ONLY for this admitted
  replay kind and restored to the host clock immediately after the traced
  run, exactly as plan Task 7 requires; the reported $0.00402781 is
  labeled synthetic historical-equivalent telemetry, never actual billing);
- never includes private repository prompts: the invocation prompt is a fixed
  public placeholder and the receipt contains no prompts, responses, tool
  content, credentials, endpoints or exceptions.

Hermetic tests drive this tool against local collectors and fake vendor HTTP
only. Fail-closed gates all run BEFORE any send; a failure writes no receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import anyio
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from daydream.agent import run_agent
from daydream.backends.pi import PiBackend
from daydream.extensions import Registry
from daydream.extensions.builtins import register_builtins
from daydream.json_utils import atomic_write_json
from daydream.observability.config import ObservabilityConfig
from daydream.observability.runtime import associate_run_trajectory, trace_run
from daydream.trajectory import DaydreamPhase

# Pristine host clock captured at import, before any replay pin: the restore
# hook always reinstates THIS reference, so repeated in-process replays can
# never leave the stdlib clock skewed.
_stdlib_real_time_ns = time.time_ns

_REPLAY_PROMPT = "Sanitized protocol replay: report the README's stated purpose."


class ReplayValidationError(ValueError):
    """A fixed, redacted replay gate failure; never echoes fixture content."""


def _restore_env_or_pop(name: str, previous: str) -> None:
    """Restore a previously-absent env var to absent (never an empty string)."""
    if previous:
        os.environ[name] = previous
    else:
        os.environ.pop(name, None)


# ---------------------------------------------------------------------------
# Fail-closed validation (all gates run BEFORE any send)
# ---------------------------------------------------------------------------


def _load_manifest(path: Path, *, expected_kind: str = "sanitized_protocol_replay") -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayValidationError(f"Unreadable replay manifest: {exc}") from None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ReplayValidationError("Replay manifest schema_version must be 1")
    if manifest.get("kind") != expected_kind:
        raise ReplayValidationError("Replay manifest kind must be sanitized_protocol_replay")
    fixture = manifest.get("fixture")
    if (
        not isinstance(fixture, dict)
        or not isinstance(fixture.get("sha256"), str)
        or not isinstance(fixture.get("bytes"), int)
    ):
        raise ReplayValidationError("Replay manifest fixture pin (sha256/bytes) is required")
    if not isinstance(manifest.get("public_repo_allowlist"), list):
        raise ReplayValidationError("Replay manifest public_repo_allowlist is required")
    return manifest


def _validate_fixture(path: Path, fixture_pin: dict[str, Any]) -> None:
    """Byte/hash pin plus structural identity; unexpected content fails closed."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReplayValidationError(f"Cannot read the sanitized fixture: {exc}") from None
    if len(raw) != int(fixture_pin["bytes"]):
        raise ReplayValidationError("Sanitized fixture byte length does not match the manifest pin")
    if hashlib.sha256(raw).hexdigest() != fixture_pin["sha256"]:
        raise ReplayValidationError("Sanitized fixture SHA-256 does not match the manifest pin")
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ReplayValidationError("Sanitized fixture is empty")
    for line in lines:
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayValidationError(f"Sanitized fixture contains malformed JSONL: {exc}") from None
    identity = fixture_pin.get("identity")
    if isinstance(identity, dict) and isinstance(identity.get("session_id"), str):
        first = json.loads(lines[0])
        if first.get("sessionId") != identity["session_id"]:
            raise ReplayValidationError("Sanitized fixture session identity does not match the manifest pin")


def _git(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=30, check=False)
    return result.stdout.strip()


def _validate_repo(path: Path, allowlist: list[str]) -> None:
    """The repo must be a git work tree, clean, and PUBLIC (origin on the allowlist)."""
    if not path.is_dir():
        raise ReplayValidationError("Repository path is not a directory")
    inside = _git(["rev-parse", "--is-inside-work-tree"], cwd=path)
    if inside != "true":
        raise ReplayValidationError("Repository path is not inside a git work tree")
    status = _git(["status", "--porcelain"], cwd=path)
    if status:
        raise ReplayValidationError("Repository work tree is dirty; the replay requires a clean public fixture repo")
    origin = _git(["remote", "get-url", "origin"], cwd=path)
    if not origin or origin not in allowlist:
        raise ReplayValidationError("Repository origin is not on the manifest public-repo allowlist")
    parsed = urlsplit(origin)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        raise ReplayValidationError("Repository origin must be an HTTP(S) public URL")


def _validate_fake_pi(path: Path, marker: str, *, fixture_raw: bytes, timeout_s: float) -> None:
    """The fake pi must be executable, NOT the real pi, and replay the fixture.

    Identity is proven behaviorally: invoking the file with ``--mode json``
    must emit the pinned fixture bytes (normalized) to stdout within the
    timeout. The real pi cannot produce the fixture stream, so this rejects it
    before any exporter is constructed.
    """
    if not path.is_file():
        raise ReplayValidationError("Fake pi path is not a regular file")
    if not os.access(path, os.X_OK):
        raise ReplayValidationError("Fake pi path is not executable")
    real_pi = shutil.which("pi")
    if real_pi is not None:
        try:
            if path.resolve() == Path(real_pi).resolve():
                raise ReplayValidationError("Fake pi path resolves to the REAL pi executable")
        except OSError:
            pass  # unreadable real path: the behavior check below still fails closed
    raw = path.read_bytes()
    if marker and marker.encode("utf-8") not in raw:
        raise ReplayValidationError("Fake pi executable does not carry the manifest identity marker")
    expected_lines = [line for line in fixture_raw.splitlines() if line.strip()]
    expected = b"\n".join(expected_lines) + b"\n"
    try:
        probe = subprocess.run(
            [str(path), "--mode", "json"],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReplayValidationError(f"Fake pi probe timed out after {timeout_s:g}s") from exc
    if probe.returncode != 0:
        raise ReplayValidationError("Fake pi probe did not exit successfully")
    actual_lines = [line for line in probe.stdout.splitlines() if line.strip()]
    if b"\n".join(actual_lines) != expected.rstrip(b"\n"):
        raise ReplayValidationError("Fake pi stdout does not replay the pinned sanitized fixture")


def _validate_destinations(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Destinations bound to the manifest: required set exactly, forbidden absent.

    The manifest's ``destinations`` block is load-bearing: editing it fails
    this gate instead of silently drifting from what the replay enforces.
    """
    destinations_block = manifest.get("destinations", {})
    required = destinations_block.get("required")
    forbidden = destinations_block.get("forbidden", [])
    if sorted(required) != ["honeyhive", "langsmith", "otlp"]:
        raise ReplayValidationError("Manifest destinations.required must be exactly otlp,honeyhive,langsmith")
    raw = os.environ.get("DAYDREAM_TRACE_TO", "")
    destinations = tuple(name.strip() for name in raw.split(",") if name.strip())
    if set(destinations) != set(required):
        raise ReplayValidationError(
            "DAYDREAM_TRACE_TO must be exactly 'otlp,honeyhive,langsmith' for the sanitized replay"
        )
    if set(destinations) & set(forbidden):
        raise ReplayValidationError("Manifest forbidden destinations must not be enabled")
    return destinations


def _validate_authorization() -> None:
    if not os.environ.get("HH_API_KEY") or not os.environ.get("HH_API_URL"):
        raise ReplayValidationError("HH_API_KEY and HH_API_URL are required for the honeyhive destination")
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise ReplayValidationError("LANGSMITH_API_KEY is required for the langsmith destination")


def _validate_acceptance_marker(manifest: dict[str, Any]) -> None:
    """Acceptance contract bound to the manifest, not tool-local literals."""
    acceptance = manifest.get("acceptance", {})
    expected_kind = acceptance.get("kind")
    marker_key = acceptance.get("resource_marker_key")
    if expected_kind != "sanitized_protocol_replay" or marker_key != "daydream.acceptance.kind":
        raise ReplayValidationError(
            "Manifest acceptance.kind/resource_marker_key must pin "
            "sanitized_protocol_replay on daydream.acceptance.kind"
        )
    if os.environ.get("DAYDREAM_ACCEPTANCE_KIND") != expected_kind:
        raise ReplayValidationError("DAYDREAM_ACCEPTANCE_KIND must be sanitized_protocol_replay")


# ---------------------------------------------------------------------------
# Local generic OTLP wire oracle (loopback-only, owned by this tool)
# ---------------------------------------------------------------------------


class _OtlpReceiver:
    """Bounded loopback OTLP/HTTP receiver; the local-wire oracle."""

    def __init__(self) -> None:
        self.batches: list[bytes] = []
        self._lock = threading.Lock()
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                with receiver._lock:
                    receiver.batches.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> str:
        host = str(self._server.server_address[0])
        port = int(self._server.server_address[1])
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def decoded(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (spans, resources) decoded from the captured protobuf batches."""
        with self._lock:
            raw_batches = list(self.batches)
        spans: list[dict[str, Any]] = []
        resources: list[dict[str, Any]] = []
        for batch in raw_batches:
            request = ExportTraceServiceRequest()
            request.ParseFromString(batch)
            decoded = MessageToDict(request)
            for resource in decoded.get("resourceSpans", []):
                resources.append(resource.get("resource", {}))
                for scope in resource.get("scopeSpans", []):
                    spans.extend(scope.get("spans", []))
        return spans, resources


# ---------------------------------------------------------------------------
# Hook: deterministic host receipt clock (manifest-pinned replay only)
# ---------------------------------------------------------------------------


def _pin_replay_clock(pinned_first_end_ns: int) -> Callable[[], None]:
    """Pin the host-observed ``message_end`` receipt and return a restore hook.

    Mirrors the test-side pin (``_pin_first_message_end_receipt``) but as an
    operator-level deterministic clock: the FIRST host receipt read lands one
    ns before the pinned end; the second (the first ``message_end`` receipt)
    lands exactly on the pinned ns; later receipts advance by real elapsed ns
    (monotonic, never backward). This is what makes the exact 395.332-second
    historical interval reproducible through the REAL PiBackend.

    The pin deliberately replaces the process-global stdlib ``time.time_ns``
    inside the REAL ``backends.pi`` module. The caller MUST invoke the
    returned callable (in a ``finally``) so the host clock is restored even
    on failure or cancellation — an unguarded global mutation would skew
    every later in-process consumer, e.g. a pytest worker's remaining tests.
    """
    # The stdlib ``time`` module is the same object ``pi.py`` imports; pin the
    # host receipt clock there so the REAL backends.pi reads land deterministically.
    import time as _stdlib_time

    import daydream.backends.pi as pi_module

    clock_state: dict[str, int] = {"reads": 0, "first_end_real": 0}
    real_time_ns = _stdlib_time.time_ns

    def pinned_time_ns() -> int:
        now = real_time_ns()
        reads = clock_state["reads"] + 1
        clock_state["reads"] = reads
        if reads < 2:
            return pinned_first_end_ns - 1
        if reads == 2:
            clock_state["first_end_real"] = now
            return pinned_first_end_ns
        return pinned_first_end_ns + (now - clock_state["first_end_real"])

    if not isinstance(pinned_first_end_ns, int) or pinned_first_end_ns <= 0:
        raise ReplayValidationError("Manifest pinned first message_end receipt must be a positive int")

    def _restore_pinned_clock() -> None:
        _stdlib_time.time_ns = _stdlib_real_time_ns

    # Deliberate operator pin of the host receipt clock inside the REAL
    # backends.pi module, exactly as the task-7 replay requires it; the
    # caller restores via the returned hook in a finally.
    setattr(_stdlib_time, "time_ns", pinned_time_ns)
    _ = pi_module  # backends.pi reads time.time_ns via the shared stdlib module
    return _restore_pinned_clock


# ---------------------------------------------------------------------------
# The actual traced run (real PiBackend/run_agent/trace_run)
# ---------------------------------------------------------------------------


def _any_value(value: Any) -> Any:
    """Normalize an OTLP proto-json AnyValue (int64 arrives as a string)."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "intValue" in value:
        raw = value["intValue"]
        try:
            return int(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            return raw
    if "doubleValue" in value:
        return value["doubleValue"]
    return None


def _span_attrs(span: dict[str, Any]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for item in span.get("attributes", []):
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not isinstance(key, str):
            continue
        attrs[key] = _any_value(item.get("value"))
    return attrs


async def _run_traced(
    manifest: dict[str, Any], repo: Path, fake_pi: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[bytes]]:
    """One real traced invocation with only the subprocess boundary faked.

    Returns (decoded spans, decoded resources, raw wire batches). The
    receiver is owned, bounded and loopback-only; the exporters run for real
    against the vendor endpoints.
    """
    receiver = _OtlpReceiver()
    fake_bin = Path(tempfile.mkdtemp(prefix="daydream-replay-pi-"))
    shutil.copy2(fake_pi, fake_bin / "pi")
    (fake_bin / "pi").chmod(0o755)
    old_path = os.environ.get("PATH", "")
    old_otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    old_otel_protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "")
    old_resource = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = receiver.endpoint
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
    marker = "daydream.acceptance.kind=sanitized_protocol_replay"
    os.environ["OTEL_RESOURCE_ATTRIBUTES"] = f"{old_resource},{marker}" if old_resource else marker
    registry = Registry()
    register_builtins(registry)
    config = ObservabilityConfig(
        destinations=("otlp", "honeyhive", "langsmith"),
        capture_content=True,  # the manifest fixture is deliberately public and sanitized
    )
    session_id = str(uuid.uuid4())
    try:
        async with trace_run(config, registry, flow="sanitized_protocol_replay") as root:
            # The run root is owned by trace_run; the trajectory session id
            # association is the same seam the runner uses (runner.py:612).
            associate_run_trajectory(session_id)
            del root
            backend = PiBackend(cwd=repo)
            await run_agent(
                backend,
                repo,
                _REPLAY_PROMPT,
                phase=DaydreamPhase.REVIEW,
                read_only=True,
                persist_session=False,
            )
    finally:
        os.environ["PATH"] = old_path
        # Restore exactly: vars that were absent stay absent (an empty string
        # is an invalid protocol/endpoint and would break later otlp users in
        # the same process).
        _restore_env_or_pop("OTEL_EXPORTER_OTLP_ENDPOINT", old_otel_endpoint)
        _restore_env_or_pop("OTEL_EXPORTER_OTLP_PROTOCOL", old_otel_protocol)
        _restore_env_or_pop("OTEL_RESOURCE_ATTRIBUTES", old_resource)
        receiver.stop()
        shutil.rmtree(fake_bin, ignore_errors=True)
    spans, resources = receiver.decoded()
    return spans, resources, list(receiver.batches)


def _require_local_wire_success(spans: list[dict[str, Any]], resources: list[dict[str, Any]]) -> dict[str, Any]:
    """Local generic OTLP wire success + exact wire reconciliation facts."""
    if not spans:
        raise ReplayValidationError("Local OTLP wire captured no spans: local wire success required")
    # The admitted resource marker must be on the exported resource.
    resource_attrs = [
        item.get("key") for resource in resources for item in resource.get("attributes", []) if isinstance(item, dict)
    ]
    if "daydream.acceptance.kind" not in resource_attrs or "sanitized_protocol_replay" not in str(resources):
        raise ReplayValidationError("daydream.acceptance.kind=sanitized_protocol_replay missing from exported resource")
    roots = [span for span in spans if not span.get("parentSpanId")]
    if len(roots) != 1:
        raise ReplayValidationError("Local OTLP wire must contain exactly one root span")
    root_attrs = _span_attrs(roots[0])
    run_id = root_attrs.get("daydream.run.id")
    session_id = root_attrs.get("daydream.session.id")
    if not isinstance(run_id, str) or not isinstance(session_id, str):
        raise ReplayValidationError("Local OTLP root must carry daydream.run.id and daydream.session.id")
    generations = [span for span in spans if _span_attrs(span).get("daydream.span.kind") == "generation"]
    if len(generations) < 2:
        raise ReplayValidationError("Local OTLP wire must contain the replay's two generation spans")
    first = _span_attrs(generations[0])
    native_ms = first.get("daydream.generation.native_started_at_unix_ms")
    native_ns = first.get("daydream.generation.native_started_at_unix_ns")
    sealed_ns = first.get("daydream.generation.sealed_end_unix_ns")
    duration_ns = first.get("daydream.generation.duration_ns")
    if (
        not isinstance(native_ms, int)
        or not isinstance(native_ns, int)
        or not isinstance(sealed_ns, int)
        or not isinstance(duration_ns, int)
    ):
        raise ReplayValidationError("Local OTLP generation timing provenance is incomplete")
    if native_ns != native_ms * 1_000_000:
        raise ReplayValidationError("Local OTLP native-ms to ns conversion is not exact")
    if duration_ns != sealed_ns - native_ns or duration_ns != 395332000000:
        raise ReplayValidationError("Local OTLP generation duration must reconcile to 395.332 seconds")
    if sealed_ns != 1788690709621000000:
        raise ReplayValidationError("Local OTLP sealed end must equal the pinned historical receipt")
    # No model-child invocation prompt: the invocation prompt is structural on
    # the attempt, never copied onto generation children (binding decision 3).
    for span in generations:
        if "gen_ai.input.messages" in _span_attrs(span):
            raise ReplayValidationError("Generation child carries an invocation prompt; must stay structural-only")
    # Exactly one billing owner across the wire (binding decision 5).
    cost_owners = [span for span in spans if "gen_ai.usage.cost" in _span_attrs(span)]
    if len(cost_owners) != 1:
        raise ReplayValidationError("Local OTLP wire must carry usage/cost on exactly one billing owner")
    owner = _span_attrs(cost_owners[0]).get("daydream.billing.owner")
    if owner not in ("structural_attempt", "generation_children"):
        raise ReplayValidationError("Local OTLP billing owner is not a closed resolved state")
    return {
        "run_id": run_id,
        "session_id": session_id,
        "root_span_id": roots[0].get("spanId"),
        "generation_count": len(generations),
        "billing_owner": str(owner),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_replay(
    *,
    manifest_path: Path,
    fixture_path: Path,
    repo_path: Path,
    fake_pi: Path,
    receipt_path: Path,
    probe_timeout_s: float = 10.0,
) -> int:
    """Validate every gate BEFORE any send, then run the actual trace once."""
    try:
        manifest = _load_manifest(manifest_path)
    except ReplayValidationError as exc:
        print(f"verdict=fail gate=manifest detail={exc}")
        return 1
    try:
        _validate_authorization()
        _validate_acceptance_marker(manifest)
        _validate_destinations(manifest)
    except ReplayValidationError as exc:
        print(f"verdict=fail gate=authorization detail={exc}")
        return 1
    try:
        _validate_fixture(fixture_path, manifest["fixture"])
        _validate_repo(repo_path, manifest["public_repo_allowlist"])
        _validate_fake_pi(
            fake_pi,
            str(manifest.get("fake_pi_identity_marker", "")),
            fixture_raw=fixture_path.read_bytes(),
            timeout_s=probe_timeout_s,
        )
    except ReplayValidationError as exc:
        print(f"verdict=fail gate=identity detail={exc}")
        return 1
    identity = manifest.get("fixture", {}).get("identity", {})
    pinned_end = identity.get("first_message_end_receipt_unix_ns")
    if not isinstance(pinned_end, int):
        print("verdict=fail gate=clock detail=Manifest must pin the first message_end receipt")
        return 1
    try:
        restore_clock = _pin_replay_clock(pinned_end)
    except ReplayValidationError as exc:
        print(f"verdict=fail gate=clock detail={exc}")
        return 1
    try:
        try:
            spans, resources, wire_batches = anyio.run(_run_traced, manifest, repo_path, fake_pi)
            facts = _require_local_wire_success(spans, resources)
        finally:
            # The pinned stdlib clock is restored on every path — success,
            # failure, or cancellation — so the host clock is never left
            # skewed for later in-process consumers.
            restore_clock()
    except Exception as exc:  # noqa: BLE001 - operator-facing boundary
        print(f"verdict=fail gate=wire detail={type(exc).__name__}")
        return 1
    wire_hashes = [hashlib.sha256(batch).hexdigest() for batch in wire_batches]
    started_iso = datetime.fromtimestamp(time.time_ns() / 1_000_000_000, tz=timezone.utc).isoformat()
    ended_iso = datetime.fromtimestamp(time.time_ns() / 1_000_000_000, tz=timezone.utc).isoformat()
    receipt = {
        "schema_version": 1,
        "contract_version": "94f432d",
        "acceptance_kind": "sanitized_protocol_replay",
        "reviewed_commit": _daydream_head(),
        "run_id": facts["run_id"],
        "session_id": facts["session_id"],
        "flow": "sanitized_protocol_replay",
        "capture_mode": "full",
        # Canonical verifier receipt key: the full destination set this replay
        # enabled (local otlp plus the two vendors). The verifier intersects
        # with its supported set for readback.
        "destinations": ["otlp", "honeyhive", "langsmith"],
        "langsmith_project": os.environ.get("LANGSMITH_PROJECT", "daydream"),
        "started_at": started_iso,
        "ended_at": ended_iso,
        "model_call_count": 0,
        "operational_cost_usd": 0,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "local_otlp_wire_sha256": wire_hashes,
        "local_otlp_root_span_id": facts["root_span_id"],
        "generation_count": facts["generation_count"],
        "billing_owner": facts["billing_owner"],
        "reported_cost_usd": identity.get("reported_cost_usd"),
        "delivery": {
            "otlp": "accepted (local loopback oracle)",
            "honeyhive": "export flushed; native verification is the readback verifier's step",
            "langsmith": "export flushed; native verification is the readback verifier's step",
        },
        "labels": {
            "reported_cost": (
                "synthetic historical-equivalent telemetry from the manifest-pinned fixture, "
                "not actual provider billing"
            ),
            "model_calls": 0,
            "operational_cost": 0,
        },
    }
    atomic_write_json(receipt_path, receipt)
    print(f"verdict=pass kind=sanitized_protocol_replay run_id={facts['run_id']} session_id={facts['session_id']}")
    return 0


def _daydream_head() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sanitized Pi protocol replay through the real trace path.")
    parser.add_argument("--manifest", type=Path, default=None, help="checked-in replay manifest JSON")
    parser.add_argument("--fixture", type=Path, required=True, help="path to the sanitized replay fixture")
    parser.add_argument("--repo", type=Path, required=True, help="clean public disposable repository (git work tree)")
    parser.add_argument("--fake-pi", type=Path, required=True, help="external fake pi executable (subprocess boundary)")
    parser.add_argument("--receipt", type=Path, required=True, help="immutable canonical receipt output path")
    parser.add_argument("--probe-timeout", type=float, default=10.0, help="fake-pi identity probe timeout (seconds)")
    args = parser.parse_args(argv)
    manifest = (
        args.manifest
        or Path(__file__).resolve().parents[1] / "tests/fixtures/observability_contract/replay-manifest.json"
    )
    return run_replay(
        manifest_path=manifest,
        fixture_path=args.fixture,
        repo_path=args.repo,
        fake_pi=args.fake_pi,
        receipt_path=args.receipt,
        probe_timeout_s=args.probe_timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
