"""Acquire bounded existing report evidence without running checks or models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.json_utils import atomic_write_json, validates_schema
from daydream.redaction import redact_structured_text, redact_value
from daydream.review_rationale import REPORT_INPUT_MAX_BYTES

AUTHOR_CONTEXT = "review-author-context.json"
REPORT_CONTEXT = "review-report-context.json"
_VERIFICATION_BUDGET_BYTES = 256 * 1024


def _exported_size(value: Any) -> int:
    """Conservatively include pretty JSON and enclosing findings-artifact indentation."""
    encoded = json.dumps(value, indent=2, ensure_ascii=False)
    return len(encoded.encode('utf-8')) + 8 * (encoded.count('\n') + 1)


def retain_author_context(deep: Path, body: str | None, head_sha: str | None) -> None:
    """Keep only the PR body the review already fetched, with its observed head."""
    atomic_write_json(deep / AUTHOR_CONTEXT, {
        "source": "pull_request_body", "head_sha": head_sha,
        "body": redact_structured_text(body)[:4096], "truncated": len(body) > 4096,
    } if body else None)


def collect_report_context(
    deep: Path, coverage: dict[str, Any], *, author_unavailable: bool = False,
) -> dict[str, Any]:
    """Name the planned scopes and read the existing verification acquisition."""
    from daydream.dataset_capture import collect_verification

    failures: list[str] = []

    def read_json(path: Path, *, redact: bool = True) -> Any:
        try:
            if path.stat().st_size > REPORT_INPUT_MAX_BYTES:
                failures.append(path.name)
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            return redact_value(payload) if redact else payload
        except (OSError, UnicodeError, ValueError, RecursionError):
            failures.append(path.name)
            return None

    planned = coverage["planned_scopes"]
    # No reviewer writes per-scope rationale; each entry keeps its stored shape with empty values.
    scopes = [{"scope_id": scope["scope_id"], "originating_run_id": None, "analyzed_revision": None,
               "review_context": None} for scope in planned[:32]]
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
    author = read_json(deep / AUTHOR_CONTEXT) if not author_unavailable and (deep / AUTHOR_CONTEXT).is_file() else None
    if author is not None:
        from daydream.review_reporting import REVIEW_CONTEXT_SCHEMA
        try:
            if not validates_schema(author, REVIEW_CONTEXT_SCHEMA['properties']['author_reported']):
                author = None
        except RecursionError:
            author = None
    return {"scopes": scopes, "scopes_omitted": max(0, len(planned) - len(scopes)),
            "verification": {"status": "failed" if verification_failures or omitted else
                             "available" if retained_verification else "unproduced",
                             "value": retained_verification or None,
                             **({'omitted_artifacts': omitted} if omitted else {})},
            "author_reported": author}
