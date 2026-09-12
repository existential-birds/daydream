#!/usr/bin/env bash
# #1093 naming gate: fails nonzero when a project-owned versioned identifier
# reappears anywhere in tracked files. External tool-protocol version strings
# (Anthropic hook API, vendored ATIF, HoneyHive/OTel API versions, ...) name
# external contracts, not project corpus generations, and are allowlisted
# inline below — each entry carries a one-line comment naming that contract.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Versioned identifiers the project owned and renamed in #1093. A match here
# is a regression: the greenfield rule is exactly one neutral name per
# identifier and no supported code paths referencing removed loaders.
VERSIONED_RE='corpus_v2|corpus-v2|stacks_v2|rubric_v2|daydream_review_v1|daydream-review-v1|build-v2|run_build_corpus_v2|curation-manifest-v1|member-v1|calibration-artifact-v1|AUDIT_ROOT_ISOLATION_V1|claude-pretooluse-v1|derive_curation_id_v2|TRAINING_SCHEMA_V1_PATH|TRAINING_SCHEMA_V2_PATH|TRAINING_SCHEMA_V2_VERSION'

# Paths excluded from the scan. Each entry names why the path is out of scope.
EXCLUDED_PATHS=(
  "scripts/check-naming.sh"           # the gate's own pattern list necessarily names the versioned identifiers
  ".beagle/"                          # planning notes describing the renames themselves
  "CHANGELOG.md"                      # historical release notes; entries name paths as they were at release time
  "tests/test_neutral_names.py"       # deletion-pin test: asserts the versioned identifiers are gone
  "tests/test_corpus_projection.py"   # deletion-pin test: asserts the legacy build-v2 subverb is gone
  "rl/daydream_review/tests/test_env_identity.py"  # deletion-pin test: asserts the v1 package name is gone
)

# Allowlisted line patterns. Each entry names the external contract it covers —
# keep these narrow so a project-owned versioned name can never hide behind one.
ALLOWLIST_RE=(
  'verifiers\.v1'                     # verifiers package: external RL env API protocol name
  'ATIF'                              # Harbor ATIF trajectory format (vendored, externally versioned)
  '/inference/v1/'                    # external inference-API endpoint path
  'honeyhive\.ai/v2'                  # HoneyHive SaaS API version in the URL
  'opentelemetry.*v1'                 # OpenTelemetry protobuf/API v1 versions
)

allowlisted() {
  local line="$1" pattern
  for pattern in "${ALLOWLIST_RE[@]}"; do
    if [[ "$line" =~ $pattern ]]; then
      return 0
    fi
  done
  return 1
}

violations=0
while IFS= read -r file; do
  skip=0
  for excluded in "${EXCLUDED_PATHS[@]}"; do
    if [[ "$file" == "$excluded"* ]]; then
      skip=1
      break
    fi
  done
  [[ "$skip" -eq 1 ]] && continue
  while IFS= read -r lineno; do
    line=$(sed -n "${lineno}p" "$file")
    if allowlisted "$line"; then
      continue
    fi
    if [[ "$violations" -eq 0 ]]; then
      echo "project-owned versioned names found (external contracts are allowlisted):"
      echo "offending lines:"
    fi
    echo "  $file:$lineno: $line"
    violations=$((violations + 1))
  done < <(grep -En "$VERSIONED_RE" -- "$file" | cut -d: -f1 || true)
done < <(git ls-files)

if [[ "$violations" -gt 0 ]]; then
  echo "$violations offending line(s)."
  exit 1
fi