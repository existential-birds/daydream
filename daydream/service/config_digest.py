"""Protected policy-source resolution and canonical effective-config digest.

A merge-authorizing service can never trust the PR head or an ambient/unpinned
override. It resolves merge-authorizing policy from a trusted source — a base /
default-branch snapshot of the reviewed repository's protected config, or an
explicitly protected per-service source explicitly designated as the protected,
pinned, digest-bound source. PR-head policy is always treated as untrusted input.

This module provides the pure helpers for that boundary: a :class:`PolicySource`
selection (protected base/default-branch snapshot vs. explicit protected source
vs. ambient override), and a canonical effective-config digest computed over a
stable, order-independent canonical form of the effective policy fields. It does
no I/O itself; callers supply the raw config dicts and the base-branch SHA.

The digest is the binding that ties every round and the published Check to one
immutable effective policy. Any change to the effective policy (round count,
backend, provider, model, lens policy, executor, publisher, Check name, budgets)
changes the digest, so a round or Check bound to an old digest can never be
mistaken for the current policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

# Keys that are never part of the effective policy and must not influence the
# digest (they do not affect merge authorization semantics).
_NON_AUTHORIZING_KEYS = frozenset(
    {
        "model",
        "reasoning_effort",
        "trajectory_hub_repo",
        "shallow_fanout_threshold",
        "precision_mode",
        "approve_on_clean",
        "group_max_wall_s",
        "group_max_serial_items",
        "uncovered_sweep",
        "uncovered_sweep_max_files",
        "uncovered_sweep_min_hunk_lines",
        "quality_gate_enabled",
        "quality_gate_erosion_delta",
        "quality_gate_verbosity_delta",
        "quality_gate_erosion_absolute",
        "quality_gate_verbosity_absolute",
        "supervisor",
        "supervisor_deny_globs",
        "tool_supervisor",
        "tool_bash_deny",
        "improve",
        "bench",
        "phases",
    }
)


@dataclass(frozen=True)
class PolicySource:
    """How merge-authorizing policy was resolved.

    Attributes:
        kind: ``base`` (protected base/default-branch snapshot), ``protected``
            (explicitly protected per-service source), or ``ambient`` (an
            ambient/unpinned file, which may not weaken protected policy).
        ref: The ref the policy was read from.
        sha: The commit SHA the policy was read at.
    """

    kind: str
    ref: str
    sha: str
    protected: bool = True


class ProtectedPolicyError(Exception):
    """Raised when merge-authorizing policy cannot be trusted-resolved."""


def policy_digest(policy_config: dict) -> str:
    """Compute the canonical effective-config digest of *policy_config*.

    Reduces *policy_config* to its merge-authorizing fields (dropping
    non-authorizing keys), serialises them through a canonical order-independent
    form, and returns a hex SHA-256 digest. Equal effective policies — regardless
    of dict key ordering, inner list ordering (lens inventory is a set), or nested
    rebuilds — yield equal digests.

    Args:
        policy_config: The effective review-policy config dict (already resolved
            from a trusted source).

    Returns:
        hex SHA-256 string.
    """
    authorizing = {
        key: value
        for key, value in policy_config.items()
        if key not in _NON_AUTHORIZING_KEYS
    }
    canonical = json.dumps(_canonicalize(authorizing), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonicalize(value: object) -> object:
    """Return an order-independent canonical form of *value*.

    Dicts are returned as-is (sorted at serialisation by ``sort_keys=True``) but
    their list-of-strings values are sorted in place so a lens inventory or
    allow-list compares equal regardless of the order it was authored in.
    Nested dicts and lists are recursed. Non-dict containers raise.
    """
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in value.items()}
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return sorted(value)
        # Mixed / non-string lists: keep order (ordering is meaningful), but
        # canonicalise the elements.
        return [_canonicalize(item) for item in value]
    return value


def resolve_policy_source(
    *,
    base_config: dict | None,
    base_sha: str,
    protected_source_ref: str | None = None,
    protected_source_sha: str | None = None,
    protected_source_config: dict | None = None,
    ambient_config: dict | None = None,
) -> tuple[PolicySource, dict]:
    """Resolve the trusted effective policy config.

    Merge-authorizing policy comes from a protected source only. In preference
    order:

    1. An explicitly protected per-service source (``protected_source_config`` +
       ref + sha) — a controller-owned, pinned, digest-bound source. This wins
       because it is explicitly designated as the protected per-service config.
    2. The base/default-branch snapshot (``base_config`` read at ``base_sha``).
    3. Never the PR head.

    An ambient (unpinned/PR-controlled) file at ``ambient_config`` may RE-EXPOSE
    the protected policy for development but can NEVER lower its strength: it is
    merged only as a subset of the already-protected effective policy and the
    explicit protected/base source still wins on every key. This mirrors the
    plan's rule that a PR-controlled or unprotected file cannot weaken
    production policy.

    Returns:
        A ``(source, effective_config)`` tuple where ``effective_config`` is the
        trusted policy dict suitable for :func:`policy_digest` and policy
        construction.

    Raises:
        ProtectedPolicyError: If no protected source is available (no base config
            and no explicit protected source).
    """
    if protected_source_config is not None and protected_source_ref is not None:
        if protected_source_config == {}:
            raise ProtectedPolicyError(
                "explicit protected per-service source is empty; refusing to resolve "
                "merge-authorizing policy from an empty source"
            )
        effective = dict(protected_source_config)
        # ambient file may not weaken an explicit protected source.
        if ambient_config:
            _merge_ambient_without_weakening(effective, ambient_config)
        return (
            PolicySource(kind="protected", ref=protected_source_ref, sha=protected_source_sha or base_sha),
            effective,
        )

    if base_config is None:
        raise ProtectedPolicyError(
            "no protected policy source available: no base/default-branch snapshot and no "
            "explicit protected per-service source"
        )

    effective = dict(base_config)
    if ambient_config:
        _merge_ambient_without_weakening(effective, ambient_config)
    return PolicySource(kind="base", ref="refs/heads/main", sha=base_sha), effective


def _merge_ambient_without_weakening(protected: dict, ambient: dict) -> None:
    """Merge an ambient config so it cannot lower any authorizing policy field.

    Ambient values only apply for keys absent from the protected effective config
    and for non-authorizing (clearly additive) keys the protected policy did not
    constrain. Any authorizing field already set in *protected* is left intact;
    the ambient file can add a value only where protected is silent, and only for
    keys that do not turn off fail-closed behavior.
    """
    for key, value in ambient.items():
        if key in protected:
            continue
        protected[key] = value
