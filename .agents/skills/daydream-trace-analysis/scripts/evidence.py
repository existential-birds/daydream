"""Standard-library reading of Daydream's local OpenTelemetry evidence documents.

A document is OTLP/JSON log lines: each line is one ``ExportLogsServiceRequest``
holding exactly one ``LogRecord`` of the ``daydream.evidence`` scope. This module
verifies the sequence and seal and reduces the records to the document view that
Daydream's own reducer (``daydream/trajectory/view.py``) produces. No Daydream imports.
"""

import hashlib
import json
import math

SCOPE = "daydream.evidence"
SEAL = "daydream.evidence.seal"
GENERIC_MODELS = frozenset({"claude", "codex", "osprey", "unknown", ""})
_DOUBLES = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


def any_value(value):
    """Decode one OTLP/JSON AnyValue."""
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
        return [any_value(item) for item in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {item["key"]: any_value(item.get("value", {})) for item in value["kvlistValue"].get("values", [])}
    raise ValueError("unsupported AnyValue")


def is_evidence_line(value):
    """Whether a decoded JSON line is an OTLP/JSON logs request."""
    return isinstance(value, dict) and "resourceLogs" in value


def records(raw):
    """Decode and verify lines; return (records, sealed, problem). Records before a problem are kept."""
    decoded, digest, start = [], hashlib.sha256(), 0
    try:
        while start < len(raw):
            end = raw.find(b"\n", start)
            if end < 0:
                raise ValueError("document ends with a truncated record")
            line = raw[start:end + 1]
            start = end + 1
            (resource,) = json.loads(line)["resourceLogs"]
            (scope,) = resource["scopeLogs"]
            if scope["scope"]["name"] != SCOPE:
                raise ValueError("record is outside the evidence scope")
            (record,) = scope["logRecords"]
            attributes = {item["key"]: any_value(item.get("value", {})) for item in record.get("attributes", [])}
            if attributes.pop("daydream.evidence.seq", None) != len(decoded):
                raise ValueError("record sequence is broken")
            if decoded and decoded[-1][0] == SEAL:
                raise ValueError("document continues after its seal")
            body = any_value(record.get("body", {}))
            if record["eventName"] == SEAL and (body.get("record_count") != len(decoded)
                                                or body.get("sha256") != digest.hexdigest()):
                decoded.append((SEAL, body))
                return decoded, False, "seal does not match its records"
            decoded.append((record["eventName"], body))
            digest.update(line)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return decoded, False, str(exc)
    sealed = bool(decoded) and decoded[-1][0] == SEAL
    return decoded, sealed, None if sealed else "document is not sealed"


def _upgrade(step, model):
    if step.get("source") == "agent" and (step.get("model_name") or "") in GENERIC_MODELS:
        return {**step, "model_name": model}
    return step


def reduce(raw):
    """Reduce one document's bytes to the document view (``extra.partial`` when incomplete)."""
    decoded, sealed, problem = records(raw)
    if not decoded or decoded[0][0] != "daydream.evidence.open":
        raise ValueError("evidence document has no open record")
    opened = decoded[0][1]
    model = opened.get("agent_model_name") or ""
    steps, phase_events, summaries, profile, snapshot, finished = {}, [], [], None, None, set()
    for name, body in decoded[1:]:
        if name == "daydream.transcript.step":
            steps[body["step_id"]] = body
        elif name == "daydream.agent.model":
            model = body["model_name"]
            steps = {key: _upgrade(step, model) for key, step in steps.items()}
        elif name == "daydream.phase_event":
            phase_events.append(body)
        elif name == "daydream.subtrajectory":
            summaries.append(body)
            if snapshot is not None and isinstance(body.get("invocation_id"), str):
                finished.add(body["invocation_id"])
        elif name == "daydream.profile":
            profile = body
        elif name == "daydream.evidence.snapshot":
            snapshot, finished = body, set()
    seal = decoded[-1][1] if sealed else None
    extra = dict(opened.get("extra") or {})
    complete = sealed and problem is None
    partial = seal is None or not complete or bool(seal.get("aborted")) or seal.get("capture_complete") is False
    active = []
    if seal is not None:
        extra["run_ended_at"] = seal["run_ended_at"]
        if seal.get("capture_complete") is False:
            extra["snapshot_at"] = seal["run_ended_at"]
        final_metrics = dict(seal["final_metrics"])
    elif snapshot is not None:
        extra["snapshot_at"] = snapshot["snapshot_at"]
        for invocation in snapshot.get("active_invocations", []):
            if invocation["summary"].get("invocation_id") in finished:
                continue
            for step in invocation["steps"]:
                steps.setdefault(step["step_id"], step)
            active.append(invocation["summary"])
        active.extend(snapshot.get("active_forks", []))
        final_metrics = {**snapshot["final_metrics"], "total_steps": len(steps)}
        if snapshot.get("active_dispatches"):
            extra["active_dispatches"] = snapshot["active_dispatches"]
    else:
        final_metrics = {"total_steps": len(steps)}
    if profile is not None:
        extra.update(profile)
    if phase_events:
        extra["phase_events"] = phase_events
    if summaries or active:
        extra["subtrajectories"] = [*summaries, *active]
    if partial:
        extra["partial"] = True
    view = {"session_id": opened["session_id"], "trajectory_id": opened["trajectory_id"],
            "parent_trajectory_id": opened.get("parent_trajectory_id"), "run_flow": opened.get("run_flow"),
            "agent": {"name": "daydream", "version": opened["agent_version"], "model_name": model},
            "steps": [steps[key] for key in sorted(steps)], "final_metrics": final_metrics, "extra": extra}
    return view, problem


def embedded(document):
    """Verify one record document against its digest and identity; return (view, problem)."""
    if not isinstance(document, dict) or set(document) != {"trajectory_id", "session_id", "sha256", "otlp_jsonl"}:
        raise ValueError("evidence document fields are malformed")
    raw = document["otlp_jsonl"].encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != document["sha256"]:
        raise ValueError(f"evidence document {document['trajectory_id']!r} differs from its sha256")
    view, problem = reduce(raw)
    if view["trajectory_id"] != document["trajectory_id"] or view["session_id"] != document["session_id"]:
        raise ValueError("evidence document identity differs from its records")
    return view, problem


def with_views(record):
    """Copy a record, replacing each document's raw lines by its verified ``view``.

    Pointers into embedded evidence address this copy:
    ``/trajectories/value/documents/<i>/view/steps/<n>/...``. Raw ``otlp_jsonl`` text is
    dropped from the copy so searches see each value once; ``sha256`` stays.
    """
    section = record.get("trajectories")
    if not isinstance(section, dict) or section.get("status") != "available":
        return record
    value = section.get("value") or {}
    documents = []
    for document in value.get("documents", []):
        view, problem = embedded(document)
        documents.append({"trajectory_id": document["trajectory_id"], "session_id": document["session_id"],
                          "sha256": document["sha256"], "evidence_problem": problem, "view": view})
    return {**record, "trajectories": {**section, "value": {**value, "documents": documents}}}
