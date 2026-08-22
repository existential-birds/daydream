#!/bin/sh
set -e
# Harbor solver: copy the provenance-free oracle candidate artifact to the
# location the verifier reads (/logs/artifacts/review.json).
SOLUTION_DIR="$(dirname "$0")"
mkdir -p /logs/artifacts
cp "$SOLUTION_DIR/golden-review.json" /logs/artifacts/review.json