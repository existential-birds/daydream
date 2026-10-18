"""Acquire bounded existing report evidence without running checks or models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.deep.artifacts import per_stack_report_context_path
from daydream.json_utils import atomic_write_json
from daydream.redaction import redact_structured_text, redact_value
from daydream.review_rationale import valid_scope_review_context

AUTHOR_CONTEXT = "review-author-context.json"
REPORT_CONTEXT = "review-report-context.json"
_SCOPE_BUDGET_BYTES = 128 * 1024
_VERIFICATION_BUDGET_BYTES = 256 * 1024


def _exported_size(value: Any) -> int:
    """Conservatively include pretty JSON and enclosing findings-artifact indentation."""
    encoded = json.dumps(value, indent=2, ensure_ascii=False)
    return len(encoded.encode('utf-8')) + 8 * (encoded.count('\n') + 1)


def retain_author_context(deep: Path, body: str | None, head_sha: str | None) -> None:
    """Keep only the PR body already fetched by intent, with its observed head."""
    atomic_write_json(deep / AUTHOR_CONTEXT, {
        "source": "pull_request_body", "head_sha": head_sha,
        "body": redact_structured_text(body)[:4096], "truncated": len(body) > 4096,
    } if body else None)


def collect_report_context(deep: Path, coverage: dict[str, Any]) -> dict[str, Any]:
    """Read only current planned scopes and the existing verification acquisition."""
    from daydream.dataset_capture import collect_verification

    failures: list[str] = []

    def read_json(path: Path, *, redact: bool = True) -> Any:
        try:
            if path.stat().st_size > 65536:
                failures.append(path.name)
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            return redact_value(payload) if redact else payload
        except (OSError, UnicodeError, ValueError, RecursionError):
            failures.append(path.name)
            return None

    scopes = []
    scope_bytes = 0
    planned = coverage["planned_scopes"]
    for scope in planned[:32]:
        scope_id = scope["scope_id"]
        payload = read_json(deep / f"stack-{scope_id}-records.json")
        metadata = None
        origin = None
        revision = None
        if isinstance(payload, dict):
            from daydream.phases.review import valid_record_artifact
            if valid_record_artifact(payload, scope_id=scope_id, analyzed_revision=coverage["analyzed_revision"]):
                origin = payload['originating_run_id']
                revision = payload['analyzed_revision']
                sidecar_path = per_stack_report_context_path(deep, scope_id)
                sidecar = read_json(sidecar_path, redact=False) if sidecar_path.is_file() else None
                if isinstance(sidecar, dict) and valid_scope_review_context(sidecar, scope_id=scope_id,
                                              analyzed_revision=revision, originating_run_id=origin):
                    metadata = sidecar['review_context']
        entry = {"scope_id": scope_id, "originating_run_id": origin,
                 "analyzed_revision": revision, "review_context": metadata}
        size = _exported_size(entry)
        if scope_bytes + size > _SCOPE_BUDGET_BYTES:
            break
        scopes.append(entry)
        scope_bytes += size
    failures.clear()
    verification = collect_verification(deep, read_json, [])
    verification_failures = list(failures)
    verification_bytes = 0
    retained_verification = {}
    omitted = []
    priority = {'test-verdict.json': 0, 'remote-ci-verdict.json': 1}
    for name in sorted(verification, key=lambda name: (priority.get(name, 2), name)):
        payload = verification[name]
        size = _exported_size({name: payload})
        if verification_bytes + size > _VERIFICATION_BUDGET_BYTES:
            omitted.append(name)
            continue
        retained_verification[name] = payload
        verification_bytes += size
    author = read_json(deep / AUTHOR_CONTEXT) if (deep / AUTHOR_CONTEXT).is_file() else None
    return {"scopes": scopes, "scopes_omitted": max(0, len(planned) - len(scopes)),
            "verification": {"status": "failed" if verification_failures or omitted else
                             "available" if retained_verification else "unproduced",
                             "value": retained_verification or None,
                             **({'omitted_artifacts': omitted} if omitted else {})},
            "author_reported": author}
