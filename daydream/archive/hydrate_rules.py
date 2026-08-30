"""Version constants and stable reason-code registry for hub hydration (#982).

Expanded by later hydrate tasks; Task 2 lands only the constants the frozen
curation-manifest-v1 schema (the #983 contract) references.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SANITIZER_VERSION = "1"
HYDRATION_INDEX_SCHEMA_VERSION = "1"
ADMISSION_POLICY_VERSION = "1"

# Stable quarantine/exclusion reason codes (fixed registry at v1).
REASON_CODE_FIXTURE_PYTEST_PATH = "fixture_pytest_path"
REASON_CODE_FIXTURE_TMP_ARTIFACT = "fixture_tmp_artifact"
REASON_CODE_NON_PRODUCTION_BUNDLE = "non_production_bundle"
REASON_CODE_PIPELINE_STATUS_EVIDENCE_ABSENT = "pipeline_status_evidence_absent"
REASON_CODE_SECRETS_SCAN_DIRTY = "secrets_scan_dirty"

# ---------------------------------------------------------------------------
# Curation-id derivation (Task 3)
# ---------------------------------------------------------------------------

CURATION_ID_RE = re.compile(r"cur-[0-9a-f]{16}")


def derive_curation_id(
    source_commit: str,
    sanitizer_version: str,
    index_schema_version: str,
    admission_policy_version: str,
) -> str:
    """Content-addressed curation id: same inputs -> same curated/ prefix.

    sha256 over the canonical tab-joined string with a single trailing
    newline, hex-truncated to 16 chars, prefixed ``cur-`` (same canonical-
    string hashing discipline as sanitize._derivative_digest).
    """
    canonical = f"cur-v1\t{source_commit}\t{sanitizer_version}\t{index_schema_version}\t{admission_policy_version}\n"
    return "cur-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


# Frozen exclusion registry at v1. Markers -> stable codes; order never matters
# because codes are returned as a de-duplicated, sorted list.
_PYTEST_TMP_MARKERS = ("/tmp/pytest", "pytest-of-", "/tmp/tmp", "PYTEST_CURRENT_TEST")
_NON_PRODUCTION_MARKERS = ("\"is_fixture\": true", "\"example_only\": true")

EXCLUSION_CODES = (
    "fixture_pytest_path",
    "fixture_tmp_artifact",
    "non_production_bundle",
)


def fixture_exclusion_codes(bundle_dir: Path) -> list[str]:
    """Detect pytest/tmp-style fixture provenance from manifest content.

    Reads only ``manifest.json`` inside ``bundle_dir``; JSON parse errors
    propagate (the orchestrator maps them to a ``bundle_unreadable``
    quarantine code). Returns stable codes from ``EXCLUSION_CODES``.
    """
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    text = json.dumps(manifest)
    codes: list[str] = []
    if any(marker in text for marker in _PYTEST_TMP_MARKERS):
        codes.append("fixture_pytest_path")
    if any(marker in text for marker in ("/tmp/tmp", "PYTEST_CURRENT_TEST")):
        codes.append("fixture_tmp_artifact")
    if any(marker in text for marker in _NON_PRODUCTION_MARKERS):
        codes.append("non_production_bundle")
    return codes


def legacy_pipeline_status(
    pipeline_status: str | None,
    deep_artifacts: dict[str, object] | None,
) -> str | tuple[str, str]:
    """Revalidate a legacy ``pipeline_status`` field; never silently success.

    Non-``unknown`` values pass through unchanged. ``unknown``/missing values
    are revalidated when the bundle carries deep-artifact evidence; without
    evidence the bundle is excluded with the stable code
    ``pipeline_status_evidence_absent`` (spec M9).
    """
    if pipeline_status and pipeline_status != "unknown":
        return pipeline_status
    if not deep_artifacts:
        return ("excluded", REASON_CODE_PIPELINE_STATUS_EVIDENCE_ABSENT)
    from daydream.archive.pipeline import derive_pipeline_status  # lazy: avoid import cycle

    status = deep_artifacts.get("status")
    if isinstance(status, str) and status and status != "unknown":
        return status
    fix_failures = deep_artifacts.get("fix_failures")
    phase_states = deep_artifacts.get("phase_states")
    archive_status = deep_artifacts.get("archive_status")
    if isinstance(fix_failures, dict) or isinstance(phase_states, dict):
        return derive_pipeline_status(
            archive_status if isinstance(archive_status, str) else "complete",
            fix_failures if isinstance(fix_failures, dict) else None,
            phase_states if isinstance(phase_states, dict) else {},
        )
    return ("excluded", REASON_CODE_PIPELINE_STATUS_EVIDENCE_ABSENT)
