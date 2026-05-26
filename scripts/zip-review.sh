#!/usr/bin/env bash
set -euo pipefail

# Zips the most recent daydream review run (trajectory + review-output.md) for sharing on Slack.
#
# Usage:
#   zip-review.sh <daydream-dir> [branch-name]
#
# Examples:
#   zip-review.sh /Users/ka/ro/mono-daydream-review/.daydream gsadalgekar/fix-cancel-future-shipments-crm
#   zip-review.sh /Users/ka/.daydream/archive
#   zip-review.sh .daydream  # relative path works too

DAYDREAM_DIR="${1:?Usage: zip-review.sh <daydream-dir> [branch-name]}"
BRANCH_NAME="${2:-}"

# Resolve to absolute path
DAYDREAM_DIR="$(cd "$DAYDREAM_DIR" && pwd)"

# Find the runs directory
RUNS_DIR=""
for candidate in "$DAYDREAM_DIR/runs" "$DAYDREAM_DIR"; do
  if [ -d "$candidate" ] && ls "$candidate"/*/trajectory.json &>/dev/null; then
    RUNS_DIR="$candidate"
    break
  fi
done

if [ -z "$RUNS_DIR" ]; then
  echo "Error: no runs with trajectory.json found under $DAYDREAM_DIR" >&2
  exit 1
fi

# Get most recent run (by modification time)
LATEST_RUN="$(ls -dt "$RUNS_DIR"/*/ | head -1)"
LATEST_RUN="${LATEST_RUN%/}"
RUN_ID="$(basename "$LATEST_RUN")"

echo "Latest run: $RUN_ID ($(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$LATEST_RUN"))"

# Derive a zip filename from branch name or run ID
if [ -z "$BRANCH_NAME" ]; then
  # Try to get branch from manifest.json
  if [ -f "$LATEST_RUN/manifest.json" ]; then
    BRANCH_NAME="$(python3 -c "import json; print(json.load(open('$LATEST_RUN/manifest.json'))['git']['branch'])" 2>/dev/null || true)"
  fi
  # Try git in the target dir
  if [ -z "$BRANCH_NAME" ]; then
    TARGET_DIR="$(python3 -c "
import json, sys
for f in ['$LATEST_RUN/trajectory.json', '$LATEST_RUN/manifest.json']:
    try:
        d = json.load(open(f))
        td = d.get('extra', {}).get('target_dir') or d.get('git', {}).get('branch')
        if td: print(td); sys.exit()
    except: pass
" 2>/dev/null || true)"
    if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
      BRANCH_NAME="$(git -C "$TARGET_DIR" branch --show-current 2>/dev/null || true)"
    fi
  fi
fi

if [ -n "$BRANCH_NAME" ]; then
  # Sanitize branch name for filename: take last segment after /, replace non-alnum with -
  ZIP_STEM="$(echo "$BRANCH_NAME" | sed 's|.*/||' | sed 's/[^a-zA-Z0-9._-]/-/g')"
else
  ZIP_STEM="$RUN_ID"
fi

ZIP_FILE="${ZIP_STEM}.zip"
echo "Output: $ZIP_FILE"

# Collect files to zip
FILES_TO_ZIP=()

# Always include the run directory (trajectory + sub-trajectories)
FILES_TO_ZIP+=("$LATEST_RUN")

# Find review-output.md — check run dir first, then deep/ at daydream level
REVIEW_OUTPUT=""
for candidate in \
  "$LATEST_RUN/review-output.md" \
  "$LATEST_RUN/deep/review-output.md" \
  "$DAYDREAM_DIR/deep/review-output.md"; do
  if [ -f "$candidate" ]; then
    REVIEW_OUTPUT="$candidate"
    break
  fi
done

if [ -n "$REVIEW_OUTPUT" ]; then
  # Only add if it's not already inside the run dir
  case "$REVIEW_OUTPUT" in
    "$LATEST_RUN"/*) ;;  # already included
    *) FILES_TO_ZIP+=("$REVIEW_OUTPUT") ;;
  esac
  echo "Included review-output.md: $REVIEW_OUTPUT"
else
  echo "Warning: no review-output.md found" >&2
fi

# Create zip
zip -r "$ZIP_FILE" "${FILES_TO_ZIP[@]}"
echo ""
echo "Done: $(du -h "$ZIP_FILE" | cut -f1) $ZIP_FILE"
