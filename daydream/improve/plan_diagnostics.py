"""Sanitized plan-attempt metadata and durable diagnostic envelopes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from daydream.improve.assemble import AdmittedPlan
from daydream.improve.redaction import redact_model_value
from daydream.trajectory import redact_text

PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION = 1
_SAFE_ERROR_DETAIL = re.compile(r"^[A-Za-z0-9_.;=-]{1,80}$")
_SAFE_METADATA_LABEL = re.compile(r"^[A-Za-z0-9._:/-]{1,160}$")


def _safe_metadata_label(value: Any, *, fallback: str) -> str:
    text = redact_text(str(value or "").strip())
    if not _SAFE_METADATA_LABEL.fullmatch(text):
        return fallback
    return text


def _received_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, AdmittedPlan):
        value = value.authored
    received_type = (
        "null"
        if value is None
        else "object"
        if isinstance(value, dict)
        else "array"
        if isinstance(value, list)
        else type(value).__name__
    )
    metadata: dict[str, Any] = {
        "type": received_type,
        "object_count": 0,
        "array_count": 0,
        "string_count": 0,
        "string_length": 0,
        "top_level_count": (
            len(value) if isinstance(value, (dict, list)) else None
        ),
    }

    def count_shape(item: Any) -> None:
        if isinstance(item, dict):
            metadata["object_count"] += 1
            for child in item.values():
                count_shape(child)
        elif isinstance(item, list):
            metadata["array_count"] += 1
            for child in item:
                count_shape(child)
        elif isinstance(item, str):
            metadata["string_count"] += 1
            metadata["string_length"] += len(item)

    count_shape(value)
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        metadata["sha256"] = None
        metadata["serialized_length"] = None
    else:
        metadata["sha256"] = hashlib.sha256(serialized).hexdigest()
        metadata["serialized_length"] = len(serialized)
    return metadata


def _validation_error(code_with_pointer: str) -> dict[str, str]:
    code, separator, remainder = code_with_pointer.partition("@")
    embedded_pointer, _, detail = remainder.partition("#")
    # Assembly issues always carry their own pointer; the host codes raised
    # around them are plan-wide.
    pointer = (
        embedded_pointer
        if separator and embedded_pointer.startswith("/")
        else "/"
    )
    if detail and _SAFE_ERROR_DETAIL.fullmatch(detail):
        return {"code": code, "pointer": pointer, "detail": detail}
    return {"code": code, "pointer": pointer}


# Codes emitted by assemble._collect_issues (seam 2, model authoring defects).
_AUTHORING_CODES = frozenset(
    {
        "AUTHOR_SCHEMA_INVALID",
        "MALFORMED_APPENDED_ARGS",
        "MALFORMED_PATH",
        "PATH_OUTSIDE_REPOSITORY",
        "EMPTY_SCOPE",
        "EXISTING_PATH_MISSING",
        "EXISTING_PATH_NOT_QUOTED",
        "NEW_PATH_ALREADY_EXISTS",
        "EXCERPT_ANCHOR_INVALID",
        "EXCERPT_PATH_MISSING",
        "RECON_COMMAND_UNKNOWN",
        "CREATE_PATH_NOT_NEW",
        "CHANGE_PATH_NOT_EXISTING",
        "TEST_EXEMPLAR_INVALID",
        "STOP_PATH_UNKNOWN",
    }
)


def _validation_stage(errors: Sequence[str]) -> str:
    codes = [error.partition("@")[0] for error in errors]
    if any(code == "RENDER_FAILED" for code in codes):
        return "render"
    if any(code in _AUTHORING_CODES for code in codes):
        return "authoring"
    return "semantic"


def _attempt_diagnostic(
    *,
    finding: dict[str, Any],
    attempt: dict[str, Any] | None,
    received: Any,
    disposition: str,
    stage: str,
    errors: Sequence[str] = (),
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attempt = attempt or {}
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "finding": {
            "fingerprint": str(finding.get("fingerprint") or ""),
            "title": redact_text(
                str(finding.get("title") or "Selected finding")
            ),
        },
        "planner": {
            "descriptor": _safe_metadata_label(
                attempt.get("descriptor"),
                fallback="plan-writer",
            ),
            "backend": _safe_metadata_label(
                attempt.get("backend"),
                fallback="unknown-backend",
            ),
            "model": _safe_metadata_label(
                attempt.get("model"),
                fallback="unknown-model",
            ),
        },
        "disposition": disposition,
        "stage": stage,
        "errors": [_validation_error(error) for error in errors],
        "received": _received_metadata(received),
        "artifact": artifact,
    }


def record_plan_write_diagnostics(
    path: Path,
    attempts: Sequence[dict[str, Any]],
    *,
    artifact_provenance: dict[str, str] | None = None,
) -> None:
    """Append sanitized plan-attempt metadata without retaining model content."""
    existing_attempts: list[dict[str, Any]] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("schema_version")
            == PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION
            and isinstance(existing.get("attempts"), list)
            and (
                artifact_provenance is None
                or existing.get("artifact_provenance")
                == artifact_provenance
            )
        ):
            existing_attempts = [
                redact_model_value(item)
                for item in existing["attempts"]
                if isinstance(item, dict)
            ]
    payload = {
        "schema_version": PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION,
        "artifact_type": "daydream.plan-write-diagnostics",
        **(
            {"artifact_provenance": dict(artifact_provenance)}
            if artifact_provenance is not None
            else {}
        ),
        "attempts": [
            redact_model_value(item)
            for item in [*existing_attempts, *attempts]
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
