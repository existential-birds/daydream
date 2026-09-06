#!/usr/bin/env bash
# Capture an unpublished public Codex generic-tool candidate without touching
# the committed fixture. After capture, an operator must privately corroborate
# this exact run before proposing a separately reviewed fixture update. This
# script never reads, copies, or publishes private rollout bytes or claims.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANDIDATE_ROOT="${DAYDREAM_CODEX_CAPTURE_ROOT:-$REPO_ROOT/.daydream/capture-candidates/codex-generic-tool-failure}"
PROBE='This is a transport-contract probe. You MUST invoke the functions.exec tool exactly once with this exact JavaScript body: const r = await tools.wait({cell_id:"definitely-missing-cell",yield_time_ms:250}); text(r); Do not call any other tool. Do not claim success unless you receive the tool output. Afterward report whether the nested wait call was visible to you and include its error category without inventing it.'

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex CLI not found on PATH." >&2
  exit 1
fi
mkdir -p "$CANDIDATE_ROOT"
CAPTURE_TMP="$(mktemp -d "$CANDIDATE_ROOT/.capture.XXXXXX")"
trap 'rm -rf -- "$CAPTURE_TMP"' EXIT
PUBLISH_TMP="$CAPTURE_TMP/candidate"
RAW_OUT="$CAPTURE_TMP/raw.jsonl"
SANITIZED_OUT="$PUBLISH_TMP/public.jsonl"
META_TMP="$PUBLISH_TMP/public.meta.json"
STDERR_OUT="$CAPTURE_TMP/stderr.log"
PROBE_REPO="$CAPTURE_TMP/repo"
VERSION="$(codex --version)"
CAPTURED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CAPTURE_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$PROBE_REPO" "$PUBLISH_TMP"
git -C "$PROBE_REPO" init -q
git -C "$PROBE_REPO" -c user.name=daydream-capture -c user.email=capture.invalid \
  commit --allow-empty -q -m "capture fixture"

# Public stdout and stderr are separated. Any capture/validation failure
# removes only the unpublished staging directory.
printf '%s\n' "$PROBE" \
  | codex exec --json --sandbox read-only --enable code_mode --cd "$PROBE_REPO" \
    >"$RAW_OUT" 2>"$STDERR_OUT"

uv run --project "$REPO_ROOT" python - \
  "$RAW_OUT" "$SANITIZED_OUT" "$META_TMP" "$VERSION" "$CAPTURED_AT" \
  "$PROBE" <<'PY'
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

raw_path, fixture_path, meta_path = map(Path, sys.argv[1:4])
version, captured_at, prompt = sys.argv[4:7]
raw = raw_path.read_bytes()
lines = raw.splitlines()
if not lines or not raw.endswith(b"\n"):
    raise SystemExit("capture is empty or lacks a trailing newline; no candidate published")

records = []
for number, line in enumerate(lines, start=1):
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"public stdout line {number} is not JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise SystemExit(f"public stdout line {number} is not an object")
    records.append(record)

generic_item_types = {
    "custom_tool_call",
    "custom_tool_call_output",
    "function_call",
    "function_call_output",
    "generic_tool_call",
    "generic_tool_result",
}
public_item_types = [
    record.get("item", {}).get("type")
    for record in records
    if isinstance(record.get("item"), dict)
]
if any(item_type in generic_item_types for item_type in public_item_types):
    raise SystemExit(
        "Codex now exposes a public generic call/result pair; update the parser contract before recapturing"
    )
if "error" not in public_item_types:
    raise SystemExit("capture lacks the public error-item coverage sentinel")
if sum(record.get("type") == "turn.completed" for record in records) != 1:
    raise SystemExit("capture must contain exactly one completed turn")
if any(record.get("type") == "turn.failed" for record in records):
    raise SystemExit("capture contains turn.failed")

raw_text = raw.decode("utf-8")
home_pattern = re.compile(r"/(?:Users|home)/[^/\s]+/\.codex/config\.toml")
sanitized_text, replacements = home_pattern.subn(
    "/Users/[REDACTED_USER]/.codex/config.toml",
    raw_text,
)
sanitized = sanitized_text.encode("utf-8")
fixture_path.write_bytes(sanitized)

ordered_shape = []
for record in records:
    event_type = record.get("type")
    item = record.get("item")
    if event_type == "item.completed" and isinstance(item, dict):
        ordered_shape.append(f"item.completed/{item.get('type')}")
    else:
        ordered_shape.append(event_type)

meta = {
    "capture_sha256_raw": hashlib.sha256(raw).hexdigest(),
    "capture_working_directory": "disposable_git_repo",
    "captured_at_utc": captured_at,
    "cli_version": version.removeprefix("codex-cli ").strip(),
    "exit_code": 0,
    "fixture_sanitized": replacements > 0,
    "fixture_sha256_sanitized": hashlib.sha256(sanitized).hexdigest(),
    "ordered_public_shape": ordered_shape,
    "publication_status": "candidate_unpublished",
    "probe_prompt": prompt,
    "public_flags": ["--json", "--sandbox", "read-only", "--enable", "code_mode"],
    "public_generic_function_items_exposed": False,
    "review_requirements": [
        "privately_correlate_this_exact_candidate_run_before_fixture_update"
    ],
    "sanitizations": ([{
        "field": "item.message",
        "occurrences": replacements,
        "reason": "personal_home_path",
        "replacement": "/Users/[REDACTED_USER]/.codex/config.toml",
    }] if replacements else []),
}
forbidden_metadata_keys = {
    "arguments",
    "call_id",
    "credentials",
    "environment_context",
    "home_path",
    "output_text",
}
if forbidden_metadata_keys.intersection(meta):
    raise SystemExit("generated provenance contains a forbidden private metadata key")
meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

# One same-filesystem directory rename publishes the candidate's public JSONL
# and matching metadata as a unit. No committed fixture path is referenced.
CANDIDATE_DIR="$CANDIDATE_ROOT/$CAPTURE_STAMP-$(basename "$CAPTURE_TMP")"
mv -- "$PUBLISH_TMP" "$CANDIDATE_DIR"
echo "Captured unpublished candidate: $CANDIDATE_DIR"
echo "Privately corroborate this exact run, then submit any fixture update for separate review."
