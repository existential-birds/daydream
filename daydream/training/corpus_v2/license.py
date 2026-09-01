"""License-policy artifact and deterministic per-repo admission decision (#1080).

The policy is a digest-pinned versioned file; the decision is a pure function
of its inputs so every record's license decision is reproducible and immutable
once recorded. Fail-closed: missing evidence, unknown SPDX ids, and missing
repo identity all resolve to rejection with a stable reason code — never a
silently substituted fallback.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

from daydream.archive.hydrate_rules import (
    REASON_CODE_C5_EXCLUDED_REPO,
    REASON_CODE_C8_COPYLEFT_UNOPTED,
    REASON_CODE_LICENSE_EVIDENCE_MISSING,
    REASON_CODE_REPO_IDENTITY_MISSING,
)
from daydream.training.exclusion import load_exclusion_list


class LicensePolicy(BaseModel):
    """Digest-pinned license decision policy (frozen)."""

    model_config = {"frozen": True}

    policy_version: str
    spdx_decisions: dict[str, str]

    @field_validator("policy_version")
    @classmethod
    def _policy_version_present(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("policy_version must be a non-empty string")
        return v

    @field_validator("spdx_decisions")
    @classmethod
    def _decisions_closed_vocab(cls, v: dict[str, str]) -> dict[str, str]:
        for spdx_id, decision in v.items():
            if decision not in {"accepted", "rejected"}:
                raise ValueError(
                    f"spdx_decisions[{spdx_id!r}] must be 'accepted' or 'rejected', got {decision!r}"
                )
        return v


@dataclass(frozen=True)
class RepoDecision:
    """Immutable per-repo license decision recorded at admission."""

    repo_slug: str
    status: str  # "admitted" | "rejected"
    reason_code: str | None
    spdx_id: str | None
    policy_version: str
    evidence_ref: str


def load_license_policy(path: str | Path) -> tuple[LicensePolicy, str]:
    """Load the license policy file; return it with the sha256 hex digest of
    the raw file bytes (the lineage pin). Fail-closed on malformed policies."""
    policy_path = Path(path)
    raw = policy_path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"license policy {policy_path} must be a JSON object")
    if "policy_version" not in data:
        raise ValueError(f"license policy {policy_path} is missing 'policy_version'")
    try:
        policy = LicensePolicy.model_validate(data)
    except Exception as exc:
        raise ValueError(f"invalid license policy {policy_path}: {exc}") from exc
    return policy, hashlib.sha256(raw).hexdigest()


def normalize_repo_slug(raw: str) -> str:
    """Normalize a producer-authored repo slug to canonical ``owner/repo``.

    Manifest ``repo_slug`` fields are producer-authored data; the C5 exclusion
    and C8 copyleft lists are canonical ``owner/repo``. A stamped clone URL
    ('https://github.com/owner/repo'), a '.git'-suffixed slug, or padded
    whitespace must reduce to the canonical spelling so the fail-closed gates
    cannot be bypassed by spelling. Returns ``''`` when the input does not
    reduce to an ``owner/repo`` shape (never a false C5/C8 match, and treated
    as missing identity by the caller).
    """
    slug = raw.strip()
    if "://" in slug:
        remainder = slug.split("://", 1)[1]
        slug = remainder.split("/", 1)[1] if "/" in remainder else ""
    if slug.endswith(".git"):
        slug = slug[: -len(".git")]
    owner, sep, repo = slug.strip().partition("/")
    if not sep or not owner.strip() or not repo.strip() or "/" in repo:
        return ""
    return f"{owner.strip()}/{repo.strip()}"


def resolve_repo_decision(
    repo_slug: str,
    evidence: dict[str, Any] | None,
    policy: LicensePolicy,
    allow_copyleft: frozenset[str] | set[str],
) -> RepoDecision:
    """Resolve the immutable license decision for one repo. Pure function of
    its inputs (determinism constraint) — the only I/O is the C5 exclusion
    list, which is static repo data.

    The decision identity is the canonical ``owner/repo`` spelling of the
    producer-authored slug (see :func:`normalize_repo_slug`), so a manifest
    that stamps the clone URL, a '.git' suffix, or padded whitespace cannot
    bypass C5/C8 by spelling, and the stamped identity is never a raw URL.

    Precedence:
    1. C5 exclusion list -> ``c5_excluded_repo`` unconditionally (no override).
    2. Missing/blank/non-canonical repo slug -> ``repo_identity_missing``.
    3. Missing/unknown evidence -> ``license_evidence_missing``.
    4. SPDX rejected and slug not opted in -> ``c8_copyleft_unopted``.
    5. Otherwise admitted.
    """
    evidence_ref = str(evidence.get("source", "")) if isinstance(evidence, dict) else ""

    canonical_slug = normalize_repo_slug(repo_slug)
    folded_slug = canonical_slug.casefold()
    if folded_slug in {slug.casefold() for slug in load_exclusion_list()}:
        return RepoDecision(
            repo_slug=canonical_slug,
            status="rejected",
            reason_code=REASON_CODE_C5_EXCLUDED_REPO,
            spdx_id=None,
            policy_version=policy.policy_version,
            evidence_ref=evidence_ref,
        )

    if not isinstance(repo_slug, str) or not canonical_slug:
        return RepoDecision(
            repo_slug=canonical_slug,
            status="rejected",
            reason_code=REASON_CODE_REPO_IDENTITY_MISSING,
            spdx_id=None,
            policy_version=policy.policy_version,
            evidence_ref=evidence_ref,
        )

    spdx_id: str | None = None
    if isinstance(evidence, dict):
        raw_spdx = evidence.get("spdx_id")
        if isinstance(raw_spdx, str) and raw_spdx.strip():
            spdx_id = raw_spdx.strip()
    if spdx_id is None or spdx_id not in policy.spdx_decisions:
        return RepoDecision(
            repo_slug=canonical_slug,
            status="rejected",
            reason_code=REASON_CODE_LICENSE_EVIDENCE_MISSING,
            spdx_id=spdx_id,
            policy_version=policy.policy_version,
            evidence_ref=evidence_ref,
        )

    decision = policy.spdx_decisions[spdx_id]
    allowed = {slug.casefold() for slug in allow_copyleft}
    if decision == "rejected" and folded_slug not in allowed:
        return RepoDecision(
            repo_slug=canonical_slug,
            status="rejected",
            reason_code=REASON_CODE_C8_COPYLEFT_UNOPTED,
            spdx_id=spdx_id,
            policy_version=policy.policy_version,
            evidence_ref=evidence_ref,
        )

    return RepoDecision(
        repo_slug=canonical_slug,
        status="admitted",
        reason_code=None,
        spdx_id=spdx_id,
        policy_version=policy.policy_version,
        evidence_ref=evidence_ref,
    )
