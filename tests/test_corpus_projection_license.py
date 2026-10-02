import hashlib
import json
from pathlib import Path

from daydream.archive import hydrate_rules
from daydream.training.corpus_projection.license import (
    LicensePolicy,
    RepoDecision,
    load_license_policy,
    resolve_repo_decision,
)


def _expected_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy() -> LicensePolicy:
    return LicensePolicy(policy_version="1", spdx_decisions={"MIT": "accepted", "GPL-3.0-only": "rejected"})


_MIT_EVIDENCE = {"spdx_id": "MIT", "source": "manifest"}
_GPL_EVIDENCE = {"spdx_id": "GPL-3.0-only", "source": "manifest"}


def _resolve(
    slug: str,
    evidence: dict[str, str] | None = _MIT_EVIDENCE,
    *,
    allow: frozenset[str] = frozenset(),
) -> RepoDecision:
    return resolve_repo_decision(
        repo_slug=slug, evidence=evidence, policy=_policy(), allow_copyleft=allow
    )


def test_reason_codes_are_frozen_strings() -> None:
    assert hydrate_rules.REASON_CODE_C5_EXCLUDED_REPO == "c5_excluded_repo"
    assert hydrate_rules.REASON_CODE_C8_COPYLEFT_UNOPTED == "c8_copyleft_unopted"
    assert hydrate_rules.REASON_CODE_LICENSE_EVIDENCE_MISSING == "license_evidence_missing"
    assert hydrate_rules.REASON_CODE_REPO_IDENTITY_MISSING == "repo_identity_missing"

def test_load_license_policy_pins_version_and_digest(tmp_path: Path) -> None:
    policy_path = tmp_path / "license-policy.json"
    policy_path.write_text(json.dumps({
        "policy_version": "1", "spdx_decisions": {"MIT": "accepted", "GPL-3.0-only": "rejected"},
    }))
    policy, digest = load_license_policy(policy_path)
    assert isinstance(policy, LicensePolicy)
    assert policy.policy_version == "1"
    assert digest == _expected_digest(policy_path)  # sha256 hex of file bytes

def test_resolve_repo_decision_fail_closed_on_missing_evidence() -> None:
    decision = _resolve("owner/repo", None)
    assert decision.status == "rejected"
    assert decision.reason_code == hydrate_rules.REASON_CODE_LICENSE_EVIDENCE_MISSING

def test_resolve_repo_decision_c5_is_hard() -> None:
    # C5 rejection wins even with evidence saying accepted and even when the
    # slug appears in allow_copyleft (no override exists for C5 — spec M3).
    decision = _resolve("getsentry/sentry", allow=frozenset({"getsentry/sentry"}))
    assert decision.status == "rejected"
    assert decision.reason_code == hydrate_rules.REASON_CODE_C5_EXCLUDED_REPO

def test_resolve_repo_decision_c5_catches_non_canonical_spellings() -> None:
    # A manifest may stamp the clone URL, a '.git'-suffixed slug, or padded
    # whitespace; the C5 gate compares the canonical owner/repo identity, so
    # no producer spelling bypasses the exclusion (spec M3, fail-closed).
    for slug in ("https://github.com/getsentry/sentry", "getsentry/sentry.git", "  getsentry/sentry  ",
        "https://github.com/getsentry/sentry.git", "git@github.com:getsentry/sentry.git", "git@host:getsentry/sentry",
    ):
        decision = _resolve(slug)
        assert decision.status == "rejected"
        assert decision.reason_code == hydrate_rules.REASON_CODE_C5_EXCLUDED_REPO
        assert decision.repo_slug == "getsentry/sentry"

def test_resolve_repo_decision_stamps_canonical_identity() -> None:
    decision = _resolve(" https://github.com/OWNER/Repo.git ")
    assert decision.status == "admitted"
    assert decision.repo_slug == "OWNER/Repo"  # never the raw URL spelling

def test_resolve_repo_decision_non_canonical_shape_is_identity_missing() -> None:
    # A slug that does not reduce to owner/repo is not a repo identity;
    # fail-closed as repo_identity_missing, never admitted under the raw shape.
    decision = _resolve("owner/repo/extra")
    assert decision.status == "rejected"
    assert decision.reason_code == hydrate_rules.REASON_CODE_REPO_IDENTITY_MISSING

def test_resolve_repo_decision_c8_exact_slug_opt_in_only() -> None:
    opted = _resolve(
        "owner/gpl-repo", _GPL_EVIDENCE, allow=frozenset({"OWNER/GPL-REPO"})
    )
    assert opted.status == "admitted"
    other = _resolve(
        "owner/similar-gpl-repo", _GPL_EVIDENCE, allow=frozenset({"owner/gpl-repo"})
    )
    assert other.status == "rejected"
    assert other.reason_code == hydrate_rules.REASON_CODE_C8_COPYLEFT_UNOPTED
