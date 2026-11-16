"""Test fixture: encode a document view as real evidence lines, standard library only.

The encoding matches Daydream's journal (one OTLP/JSON logs request per line, a
sequence attribute and a verified seal), so tests exercise evidence.py as it reads
actual captures.
"""

import hashlib
import json
import math

_REDUCED = ("phase_events", "subtrajectories", "run_ended_at", "snapshot_at", "partial", "active_dispatches")


def any_value(value):
    if value is None:
        return {}
    if value is True or value is False:
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        # Non-finite doubles are strings, as the journal writes them under allow_nan=False.
        if math.isfinite(value):
            return {"doubleValue": value}
        return {"doubleValue": "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [any_value(item) for item in value]}}
    return {"kvlistValue": {"values": [{"key": key, "value": any_value(item)} for key, item in value.items()]}}


def _line(seq, event, body):
    record = {"timeUnixNano": "1", "observedTimeUnixNano": "1", "severityNumber": 9, "eventName": event,
              "body": any_value(body),
              "attributes": [{"key": "daydream.evidence.seq", "value": {"intValue": str(seq)}}]}
    request = {"resourceLogs": [{"resource": {"attributes": []}, "scopeLogs": [
        {"scope": {"name": "daydream.evidence", "version": "test"}, "logRecords": [record]}]}]}
    return (json.dumps(request, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def evidence_bytes(view, session_id, *, sealed=True):
    """Encode one view: open record, steps, phase events, summaries and a verified seal."""
    extra = dict(view.get("extra") or {})
    events = [("daydream.evidence.open", {
        "session_id": view.get("session_id", session_id), "trajectory_id": view["trajectory_id"],
        "descriptor": "", "parent_trajectory_id": None, "run_flow": "deep", "agent_version": "test",
        "agent_model_name": (view.get("agent") or {}).get("model_name", "codex"),
        "extra": {key: value for key, value in extra.items() if key not in _REDUCED}})]
    events += [("daydream.transcript.step", step) for step in view.get("steps", [])]
    events += [("daydream.phase_event", event) for event in extra.get("phase_events", [])]
    events += [("daydream.subtrajectory", summary) for summary in extra.get("subtrajectories", [])]
    raw = b"".join(_line(seq, event, body) for seq, (event, body) in enumerate(events))
    if sealed:
        raw += _line(len(events), "daydream.evidence.seal", {
            "run_ended_at": extra.get("run_ended_at", "2026-10-17T00:00:00Z"), "aborted": False,
            "capture_complete": True, "failed_documents": [],
            "final_metrics": view.get("final_metrics") or {"total_steps": len(view.get("steps", []))},
            "record_count": len(events), "sha256": hashlib.sha256(raw).hexdigest()})
    return raw


def v3_document(view, session_id):
    """One ``daydream.run.v3`` embedded document: exact lines plus their digest."""
    raw = evidence_bytes(view, session_id)
    return {"trajectory_id": view["trajectory_id"], "session_id": view.get("session_id", session_id),
            "sha256": hashlib.sha256(raw).hexdigest(), "otlp_jsonl": raw.decode("ascii")}


def v3_record(record):
    """Encode a test record whose documents are written as views into a real v3 record."""
    section = record.get("trajectories") or {}
    value = section.get("value") if isinstance(section, dict) else None
    if not isinstance(value, dict) or "documents" not in value:
        return record
    documents = [v3_document(view, record["run_id"]) for view in value["documents"]]
    return {**record, "trajectories": {**section, "value": {**value, "documents": documents}}}
