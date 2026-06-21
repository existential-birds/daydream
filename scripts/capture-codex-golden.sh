#!/usr/bin/env bash
#
# Capture (or refresh) the REAL codex CLI golden fixture.
#
# Runs `codex exec --experimental-json --sandbox read-only` against the tiny
# in-repo sample repo and writes the JSONL stdout to
# tests/fixtures/codex_jsonl/real/golden.jsonl.
#
# WHEN TO REFRESH:
#   - After a codex CLI version bump (codex --version).
#   - After any change to the JSONL parser in daydream/backends/codex.py.
#   - When you intentionally want to update the structural golden to match a
#     new CLI/model output shape.
#
# REQUIREMENTS:
#   - `codex` on $PATH (codex --version) with valid auth (ChatGPT or API).
#   - The default model must be one the authenticated account can use.
#
# The committed golden is REAL output, not synthesized. Do not hand-edit it;
# re-run this script instead.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE_REPO="$REPO_ROOT/tests/fixtures/real_cli_sample_repo"
GOLDEN_OUT="$REPO_ROOT/tests/fixtures/codex_jsonl/real/golden.jsonl"

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex CLI not found on PATH. Install/configure codex first." >&2
  exit 1
fi

echo "→ codex version: $(codex --version)"
echo "→ sample repo:   $SAMPLE_REPO"
echo "→ golden output: $GOLDEN_OUT"
echo "→ prompt: read README.md then hello.py, then describe both"

# Read prompt from stdin, run read-only so nothing mutates the sample repo.
# --cd pins the codex working directory to SAMPLE_REPO so the golden is always
# captured against the in-repo sample repo, never the caller's cwd.
echo "Read README.md, then read hello.py, then describe both files in one sentence each." \
  | codex exec --experimental-json --sandbox read-only --cd "$SAMPLE_REPO" > "$GOLDEN_OUT"

LINES=$(wc -l < "$GOLDEN_OUT" | tr -d ' ')
echo "✓ captured $LINES JSONL lines"
echo "  commit the refreshed golden with the codex version in the message."
