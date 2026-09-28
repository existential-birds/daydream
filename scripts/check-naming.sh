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

# Allowlisted line patterns, combined into one ERE alternation so a single
# ``grep -vE`` drops allowlisted hits before the report loop. Each alternative
# names the external contract it covers — keep these narrow so a project-owned
# versioned name can never hide behind one:
#   verifiers\.v1     verifiers package: external RL env API protocol name
#   ATIF              Harbor ATIF trajectory format (vendored, externally versioned)
#   /inference/v1/    external inference-API endpoint path
#   honeyhive\.ai/v2  HoneyHive SaaS API version in the URL
#   opentelemetry.*v1 OpenTelemetry protobuf/API v1 versions
ALLOWLIST_RE='verifiers\.v1|ATIF|/inference/v1/|honeyhive\.ai/v2|opentelemetry.*v1'

# One git grep over the tracked tree, excluding the out-of-scope paths above and
# dropping allowlisted lines before formatting the report. ``|| true`` keeps the
# no-match exit (1) from tripping ``set -e``.
hits="$(git grep -nE "$VERSIONED_RE" -- . "${EXCLUDED_PATHS[@]/#/:(exclude)}" | grep -vE "$ALLOWLIST_RE" || true)"
if [[ -n "$hits" ]]; then
  echo "project-owned versioned names found (external contracts are allowlisted):"
  echo "offending lines:"
  # git grep prints ``file:line:text``; restore the gate's ``  file:line: text`` spacing.
  sed -E 's/^([^:]*:[0-9]+:)/  \1 /' <<<"$hits"
  echo "$(wc -l <<<"$hits") offending line(s)."
  exit 1
fi
