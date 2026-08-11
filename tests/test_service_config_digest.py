"""Hermetic tests for protected policy-source resolution and the config digest.

Covers plan Step 5's protected-policy source rule: merge-authorizing policy is
resolved from a base/default-branch snapshot or an explicitly protected
per-service source, never the PR head, and an ambient/unprotected file cannot
weaken it. Also covers the canonical effective-config digest that binds every
round and the published Check to one immutable policy.
"""

from __future__ import annotations

import pytest

from daydream.service.config_digest import (
    ProtectedPolicyError,
    policy_digest,
    resolve_policy_source,
)
from tests.harness.service_fakes import BASE_SHA


def _policy_config(**overrides) -> dict:
    review_policy = {
        "backend": "pi",
        "provider": "nous",
        "model": "deepseek/deepseek-v4-flash-0731",
        "required_rounds": 2,
        "complete_lens": ["python", "security"],
        "executor": "local-fake",
    }
    if "required_rounds" in overrides:
        review_policy["required_rounds"] = overrides.pop("required_rounds")
    base = {
        "review_policy": review_policy,
        "publication": {
            "publisher": "github-checks",
            "check_name": "daydream/review",
        },
    }
    base.update(overrides)
    return base


# --- canonical digest --------------------------------------------------------


def test_policy_digest_is_stable_regardless_of_key_order() -> None:
    a = policy_digest(_policy_config())
    # reordered keys + reordered list -> same effective digest
    reordered = _policy_config()
    reordered["publication"] = {"check_name": "daydream/review", "publisher": "github-checks"}
    reordered["review_policy"]["complete_lens"] = ["security", "python"]
    reordered["review_policy"] = {
        "executor": "local-fake",
        "model": "deepseek/deepseek-v4-flash-0731",
        "provider": "nous",
        "backend": "pi",
        "required_rounds": 2,
        "complete_lens": ["security", "python"],
    }
    assert policy_digest(reordered) == a


def test_policy_digest_is_order_insensitive_for_nested_dicts() -> None:
    d1 = policy_digest({"publication": {"publisher": "x", "check_name": "y"}, "required_rounds": 2})
    d2 = policy_digest({"required_rounds": 2, "publication": {"check_name": "y", "publisher": "x"}})
    assert d1 == d2
    assert len(d1) == 64  # hex sha256


def test_policy_digest_changes_when_authorizing_field_changes() -> None:
    assert policy_digest(_policy_config(required_rounds=2)) != policy_digest(
        _policy_config(required_rounds=3)
    )


def test_policy_digest_changes_when_check_name_changes() -> None:
    p = _policy_config()
    p["publication"]["check_name"] = "daydream/other"
    assert policy_digest(_policy_config()) != policy_digest(p)


def test_policy_digest_ignores_non_authorizing_fields() -> None:
    """Cosmetic/agent-tuning keys do not change the merge-authorizing digest."""
    with_extra = _policy_config()
    with_extra["model"] = "different-model"
    with_extra["reasoning_effort"] = "high"
    with_extra["approve_on_clean"] = True
    # Top-level model override (agent tuning) must NOT affect authorization.
    assert policy_digest(_policy_config()) == policy_digest(with_extra)


def test_policy_digest_is_sha256_hex() -> None:
    digest = policy_digest(_policy_config())
    assert len(digest) == 64
    int(digest, 16)  # hex-only


# --- protected source resolution ---------------------------------------------


def test_resolve_from_base_branch_snapshot() -> None:
    source, effective = resolve_policy_source(
        base_config=_policy_config(),
        base_sha=BASE_SHA,
    )
    assert source.kind == "base"
    assert source.ref == "refs/heads/main"
    assert source.sha == BASE_SHA
    assert effective == _policy_config()


def test_explicit_protected_source_wins_over_base() -> None:
    protected = _policy_config(required_rounds=3)
    source, effective = resolve_policy_source(
        base_config=_policy_config(required_rounds=2),
        base_sha=BASE_SHA,
        protected_source_ref="refs/heads/protected",
        protected_source_sha="9" * 40,
        protected_source_config=protected,
    )
    assert source.kind == "protected"
    assert source.ref == "refs/heads/protected"
    assert effective["review_policy"]["required_rounds"] == 3


def test_no_protected_source_raises() -> None:
    with pytest.raises(ProtectedPolicyError):
        resolve_policy_source(base_config=None, base_sha=BASE_SHA)


def test_empty_explicit_protected_source_raises() -> None:
    with pytest.raises(ProtectedPolicyError):
        resolve_policy_source(
            base_config=_policy_config(),
            base_sha=BASE_SHA,
            protected_source_ref="refs/heads/protected",
            protected_source_config={},
        )


def test_ambient_file_cannot_lower_protected_policy() -> None:
    """A PR-controlled ambient file cannot lower rounds or rename the required Check."""
    ambient = {
        "review_policy": {
            "required_rounds": 1,  # weaker
            "backend": "pi",
            "provider": "nous",
            "model": "x",
            "complete_lens": ["python"],
            "executor": "local-fake",
        },
        "publication": {"publisher": "github-checks", "check_name": "weak/review"},
        "some_new_knob": True,
    }
    source, effective = resolve_policy_source(
        base_config=_policy_config(),
        base_sha=BASE_SHA,
        ambient_config=ambient,
    )
    # protected base wins on every authorizing field
    assert effective["review_policy"]["required_rounds"] == 2
    assert effective["publication"]["check_name"] == "daydream/review"
    # a genuinely new, non-conflicting knob may be added by ambient (additive only)
    assert effective.get("some_new_knob") is True


def test_ambient_cannot_weaken_explicit_protected_source() -> None:
    ambient = {"review_policy": {"required_rounds": 1, "complete_lens": []}}
    source, effective = resolve_policy_source(
        base_config=_policy_config(),
        base_sha=BASE_SHA,
        protected_source_ref="refs/heads/protected",
        protected_source_sha="9" * 40,
        protected_source_config=_policy_config(required_rounds=3),
        ambient_config=ambient,
    )
    assert source.kind == "protected"
    assert effective["review_policy"]["required_rounds"] == 3


def test_ambient_without_base_but_with_protected_source_uses_protected() -> None:
    source, effective = resolve_policy_source(
        base_config=None,
        base_sha=BASE_SHA,
        protected_source_ref="refs/heads/ops",
        protected_source_sha="8" * 40,
        protected_source_config=_policy_config(required_rounds=4),
    )
    assert source.kind == "protected"
    assert effective["review_policy"]["required_rounds"] == 4


def test_incomplete_protected_source_raises() -> None:
    """A partial pinned source must not silently fall back to the base snapshot."""
    with pytest.raises(ProtectedPolicyError):
        resolve_policy_source(
            base_config=_policy_config(),
            base_sha=BASE_SHA,
            protected_source_config=_policy_config(required_rounds=3),  # no ref/sha
        )
    with pytest.raises(ProtectedPolicyError):
        resolve_policy_source(
            base_config=_policy_config(),
            base_sha=BASE_SHA,
            protected_source_ref="refs/heads/protected",
            protected_source_config=_policy_config(required_rounds=3),
        )  # no sha: must not fabricate base_sha
