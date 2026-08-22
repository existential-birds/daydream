#!/bin/sh
set -e
# Harbor verifier self-test: confirm the solver produced a candidate artifact,
# the candidate artifact loads, the judge entry runs to completion, and the
# reward files were written. Exits non-zero on any verifier error.
if [ ! -f /logs/artifacts/review.json ]; then
  echo "verifier: missing /logs/artifacts/review.json" >&2
  exit 1
fi

python3 -c "
import json
from pathlib import Path
from verifier_core import validate_candidate_artifact
raw = json.loads(Path('/logs/artifacts/review.json').read_text())
validate_candidate_artifact(raw)
print('candidate artifact valid')
"

python3 score_review.py

if [ ! -f /logs/verifier/reward.json ]; then
  echo "verifier: reward.json was not written" >&2
  exit 1
fi

echo "verifier: reward.json written"