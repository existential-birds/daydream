"""Version constants and stable reason-code registry for hub hydration (#982).

Expanded by later hydrate tasks; Task 2 lands only the constants the frozen
curation-manifest-v1 schema (the #983 contract) references.
"""
from __future__ import annotations

SANITIZER_VERSION = "1"
HYDRATION_INDEX_SCHEMA_VERSION = "1"
ADMISSION_POLICY_VERSION = "1"

# Stable quarantine/exclusion reason codes (fixed registry at v1).
REASON_CODE_FIXTURE_PYTEST_PATH = "fixture_pytest_path"
REASON_CODE_FIXTURE_TMP_ARTIFACT = "fixture_tmp_artifact"
REASON_CODE_NON_PRODUCTION_BUNDLE = "non_production_bundle"
REASON_CODE_PIPELINE_STATUS_EVIDENCE_ABSENT = "pipeline_status_evidence_absent"
REASON_CODE_SECRETS_SCAN_DIRTY = "secrets_scan_dirty"
