"""Durable local OpenTelemetry evidence: one append-only OTLP/JSON log journal per document.

Each line is a complete OTLP/JSON ``ExportLogsServiceRequest`` holding exactly one
``LogRecord`` from the ``daydream.evidence`` scope, so standard OTel tooling (for
example a collector's file receiver) can replay the file. Records carry
``daydream.evidence.seq`` (0, 1, 2, ...) and, when tracing is active, the
current span's trace and span ids.

A record is accepted once its whole line has been written to the private file with
one ``os.write`` loop; there is no in-memory queue to overflow. The writer keeps a
running SHA-256 and size, so a reference to the bytes written so far needs no
re-read. Closing a document appends one ``daydream.evidence.seal`` record whose body
names the preceding record count and digest, then fsyncs. Only a verified seal makes
a document complete: a failed write, a truncated last line or a missing seal is read
back as truthful incomplete evidence, never as complete.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_core import to_jsonable_python

EVIDENCE_SCOPE = "daydream.evidence"
SEAL_EVENT = "daydream.evidence.seal"
SEQ_ATTRIBUTE = "daydream.evidence.seq"
_INT64 = 1 << 63


class EvidenceEncodingError(ValueError):
    """A value has no lossless OTLP AnyValue representation."""


def any_value(value: Any) -> dict[str, Any]:
    """Encode one JSON-shaped value as an OTLP/JSON ``AnyValue`` without loss.

    ``None`` is the empty AnyValue. Integers outside int64 are refused rather than
    rounded. Other objects first pass through their JSON-compatible projection."""
    if value is None:
        return {}
    if value is True or value is False:
        return {"boolValue": value}
    if isinstance(value, int):
        if not -_INT64 <= value < _INT64:
            raise EvidenceEncodingError("integer outside int64")
        return {"intValue": str(value)}
    if isinstance(value, float):
        if math.isfinite(value):
            return {"doubleValue": value}
        return {"doubleValue": "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [any_value(item) for item in value]}}
    if isinstance(value, Mapping):
        return {"kvlistValue": {"values": [
            {"key": key if isinstance(key, str) else str(key), "value": any_value(item)}
            for key, item in value.items()
        ]}}
    try:
        projected = to_jsonable_python(value)
    except Exception as exc:  # noqa: BLE001 - an unencodable value is an encoding failure
        raise EvidenceEncodingError(type(value).__name__) from exc
    if projected is value:
        raise EvidenceEncodingError(type(value).__name__)
    return any_value(projected)


_DOUBLES = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


def from_any_value(value: Mapping[str, Any]) -> Any:
    """Decode one OTLP/JSON ``AnyValue`` produced by :func:`any_value`."""
    if not value:
        return None
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        number = value["doubleValue"]
        return _DOUBLES[number] if isinstance(number, str) else float(number)
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [from_any_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {item["key"]: from_any_value(item.get("value", {})) for item in value["kvlistValue"].get("values", [])}
    raise ValueError("unsupported AnyValue")


def _key_values(attributes: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"key": key, "value": any_value(value)} for key, value in attributes.items() if value is not None]


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _current_trace_ids() -> tuple[str, str] | None:
    """Correlate a record with the active span when one is recording; tracing stays optional."""
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            return format(context.trace_id, "032x"), format(context.span_id, "016x")
    except Exception:  # noqa: BLE001 - correlation never blocks evidence
        pass
    return None


class JournalWriter:
    """Append-only writer for one evidence document.

    The file is created privately (mode 0600) when the writer opens; an existing
    file at the same path is replaced, as a rewritten document was before. The
    first write or encoding failure stops the writer and is retained in
    ``failure``: later records are refused and no seal can be written."""

    def __init__(self, path: Path, *, resource: Mapping[str, Any], version: str) -> None:
        self.path = path
        self.failure: str | None = None
        self.sealed = False
        self._seq = 0
        self._size = 0
        self._hash = hashlib.sha256()
        self._lock = threading.Lock()
        self._fd: int | None = None
        self._prefix = ('{"resourceLogs":[{"resource":{"attributes":' + _dumps(_key_values(resource))
                        + '},"scopeLogs":[{"scope":{"name":"' + EVIDENCE_SCOPE + '","version":'
                        + _dumps(version) + '},"logRecords":[').encode()
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
                               | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except OSError as exc:
            self._fail(exc)

    @property
    def records(self) -> int:
        return self._seq

    def reference(self) -> tuple[int, str]:
        """Size and SHA-256 of every byte accepted so far, without I/O."""
        with self._lock:
            return self._size, self._hash.hexdigest()

    def append(self, event_name: str, body: Any, attributes: Mapping[str, Any] | None = None, *,
               blocking: bool = True) -> bool:
        """Accept one record, or return ``False`` when it was refused.

        ``blocking=False`` is for the signal handler: when a record is already being
        written on this thread, the call returns ``False`` instead of interleaving."""
        if not self._lock.acquire(blocking=blocking):
            return False
        try:
            return self._append_locked(event_name, body, attributes or {})
        finally:
            self._lock.release()

    def _append_locked(self, event_name: str, body: Any, attributes: Mapping[str, Any]) -> bool:
        if self._fd is None or self.failure is not None or self.sealed:
            return False
        try:
            now = str(time.time_ns())
            record: dict[str, Any] = {
                "timeUnixNano": now, "observedTimeUnixNano": now, "severityNumber": 9, "severityText": "INFO",
                "eventName": event_name, "body": any_value(body),
                "attributes": _key_values({SEQ_ATTRIBUTE: self._seq, **attributes}),
            }
            ids = _current_trace_ids()
            if ids is not None:
                record["traceId"], record["spanId"] = ids
            line = self._prefix + _dumps(record).encode() + b"]}]}]}\n"
        except Exception as exc:  # noqa: BLE001 - truthful incomplete capture, never a raised run failure
            self._fail(exc)
            return False
        try:
            view = memoryview(line)
            while view:
                written = os.write(self._fd, view)
                view = view[written:]
        except OSError as exc:
            self._fail(exc)
            return False
        self._hash.update(line)
        self._size += len(line)
        self._seq += 1
        return True

    def sync(self) -> None:
        """Flush accepted records to stable storage; a failure marks the document incomplete."""
        with self._lock:
            if self._fd is not None and self.failure is None:
                try:
                    os.fsync(self._fd)
                except OSError as exc:
                    self._fail(exc)

    def seal(self, body: Mapping[str, Any]) -> bool:
        """Append the seal naming the preceding count and digest, fsync and close."""
        with self._lock:
            if self.sealed:
                return True
            sealed_body = {**body, "record_count": self._seq, "sha256": self._hash.hexdigest()}
            unsealed = self._seq, self._size, self._hash.copy()
            if not self._append_locked(SEAL_EVENT, sealed_body, {}):
                return False
            try:
                assert self._fd is not None
                os.fsync(self._fd)
            except OSError as exc:
                # A seal that may not be durable never counts: references keep the unsealed prefix.
                self._seq, self._size, self._hash = unsealed
                self._fail(exc)
                return False
            self.sealed = True
            self._close()
            return True

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def _fail(self, exc: BaseException) -> None:
        if self.failure is None:
            self.failure = type(exc).__name__
            from daydream import ui

            ui.print_warning(ui.create_console(), (
                f"Evidence write failed ({self.failure}); the review continues and "
                f"{self.path.name} is recorded as incomplete"
            ))
        self._close()


@dataclass(frozen=True)
class JournalRecord:
    """One decoded evidence record."""

    seq: int
    event_name: str
    time_unix_nano: int
    body: Any
    attributes: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None


@dataclass(frozen=True)
class JournalContents:
    """The decoded records of one document and whether its seal verified."""

    records: tuple[JournalRecord, ...]
    sealed: bool
    problem: str | None = None

    @property
    def complete(self) -> bool:
        return self.sealed and self.problem is None

    @property
    def seal(self) -> JournalRecord | None:
        return self.records[-1] if self.sealed else None


def iter_lines(raw: bytes) -> Iterator[bytes]:
    """Yield complete lines; a final line without its newline is truncated evidence."""
    start = 0
    while start < len(raw):
        end = raw.find(b"\n", start)
        if end < 0:
            raise ValueError("journal ends with a truncated record")
        yield raw[start:end + 1]
        start = end + 1


def decode_line(line: bytes) -> JournalRecord:
    """Decode one OTLP/JSON line holding exactly one evidence record."""
    payload = json.loads(line)
    (resource_logs,) = payload["resourceLogs"]
    (scope_logs,) = resource_logs["scopeLogs"]
    if scope_logs["scope"]["name"] != EVIDENCE_SCOPE:
        raise ValueError("record is outside the evidence scope")
    (record,) = scope_logs["logRecords"]
    attributes = {item["key"]: from_any_value(item.get("value", {})) for item in record.get("attributes", [])}
    seq = attributes.pop(SEQ_ATTRIBUTE)
    if type(seq) is not int:
        raise ValueError("record sequence is malformed")
    return JournalRecord(
        seq=seq, event_name=record["eventName"], time_unix_nano=int(record["timeUnixNano"]),
        body=from_any_value(record.get("body", {})), attributes=attributes,
        trace_id=record.get("traceId") or None, span_id=record.get("spanId") or None,
    )


def read_journal(raw: bytes) -> JournalContents:
    """Decode and verify one document's bytes.

    Records before the first problem are kept. A seal verifies only as the last
    record whose count and digest match the bytes before it."""
    records: list[JournalRecord] = []
    digest = hashlib.sha256()
    try:
        for line in iter_lines(raw):
            record = decode_line(line)
            if record.seq != len(records):
                raise ValueError("journal record sequence is broken")
            if records and records[-1].event_name == SEAL_EVENT:
                raise ValueError("journal continues after its seal")
            if record.event_name == SEAL_EVENT:
                body = record.body if isinstance(record.body, dict) else {}
                if body.get("record_count") != record.seq or body.get("sha256") != digest.hexdigest():
                    records.append(record)
                    return JournalContents(tuple(records), False, "journal seal does not match its records")
            records.append(record)
            digest.update(line)
    except (ValueError, KeyError, TypeError) as exc:
        return JournalContents(tuple(records), False, str(exc) if isinstance(exc, ValueError) else "malformed record")
    sealed = bool(records) and records[-1].event_name == SEAL_EVENT
    return JournalContents(tuple(records), sealed, None if sealed else "journal is not sealed")
