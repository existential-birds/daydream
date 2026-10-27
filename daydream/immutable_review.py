"""Versioned trusted identity for offline exact and synthetic committed reviews.

This host-only envelope is evidence, never publication authority or model policy.
The local revision and source references always identify the materialized objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jsonschema

from daydream.json_utils import canonical_json
from daydream.output_schema import strict_object

_SHA = {"type": "string", "pattern": "^[0-9a-f]{40}$|^[0-9a-f]{64}$"}
_DIGEST = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
IMMUTABLE_REVIEW_INPUT_SCHEMA = strict_object({
    "schema_version": {"const": 1},
    "provenance": strict_object({
        "repository": {"type": "string", "pattern": "^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", "maxLength": 256},
        "pr_number": {"type": "integer", "minimum": 1},
        **{key: _SHA for key in ("head_sha", "requested_base_sha", "comparison_base_sha")},
        "title": {"type": "string", "maxLength": 512},
        "body": {"type": "string", "maxLength": 8192},
        "context": {"type": "string", "maxLength": 65536},
    }),
    "materialization": strict_object({
        "kind": {"enum": ["exact", "synthetic"]},
        **{key: _SHA for key in (
            "head_sha", "requested_base_sha", "comparison_base_sha", "head_tree_sha", "comparison_tree_sha",
        )},
        "canonical_diff_sha256": _DIGEST,
        "review_diff_key": _DIGEST,
        "bundle_sha256": {"anyOf": [_DIGEST, {"type": "null"}]},
    }),
})


def validate_immutable_input(value: dict[str, Any]) -> None:
    """Reject unknown fields/versions and contradictory original/local identities."""
    try:
        jsonschema.validate(value, IMMUTABLE_REVIEW_INPUT_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"invalid immutable review input at {exc.json_path}") from exc
    original, local = value["provenance"], value["materialization"]
    if type(value["schema_version"]) is not int:
        raise ValueError("immutable input version must be an exact integer")
    if any(part in {".", ".."} for part in original["repository"].split("/")):
        raise ValueError("immutable repository identity is malformed")
    if type(original["pr_number"]) is not int:
        raise ValueError("immutable PR number must be an exact positive integer")
    if any(len(original[key].encode()) > limit for key, limit in (("title", 512), ("body", 8192), ("context", 65536))):
        raise ValueError("immutable historical context exceeds its UTF-8 bound")
    if local["kind"] == "exact":
        if any(original[key] != local[key] for key in ("head_sha", "requested_base_sha", "comparison_base_sha")):
            raise ValueError("exact immutable objects differ from captured original identity")
    elif local["requested_base_sha"] != local["comparison_base_sha"] or local["bundle_sha256"] is None:
        raise ValueError("synthetic immutable input requires a two-commit base and frozen bundle evidence")


@dataclass(frozen=True, init=False)
class ImmutableReviewInput:
    """Freeze a validated envelope as canonical bytes, returning independent copies."""

    json_text: str

    def __init__(self, value: dict[str, Any]) -> None:
        validate_immutable_input(value)
        object.__setattr__(self, "json_text", canonical_json(value))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ImmutableReviewInput:
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(self.json_text)
        return value
