"""Versioned trusted identity for offline exact and synthetic committed reviews.

This host-only envelope is evidence, never publication authority or model policy.
The local revision and source references always identify the materialized objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import jsonschema

from daydream.json_utils import canonical_json, validate_json_schema
from daydream.output_schema import strict_object

_SHA = {"type": "string", "pattern": "^[0-9a-f]{40}$|^[0-9a-f]{64}$"}
_DIGEST = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
# Captured author text is bounded before any of it reaches a model assignment: a pull request's title and
# body by GitHub's own limits in characters (code points, as the schema counts them), so every pull request
# GitHub accepts parses, and other captured context by UTF-8 bytes.
HISTORICAL_TEXT_MAX_CHARS = {"title": 256, "body": 65536}
HISTORICAL_CONTEXT_MAX_BYTES = 65536
IMMUTABLE_REVIEW_INPUT_SCHEMA = strict_object({
    "schema_version": {"const": 1},
    "provenance": strict_object({
        "repository": {"type": "string", "pattern": "^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", "maxLength": 256},
        "pr_number": {"type": "integer", "minimum": 1},
        **{key: _SHA for key in ("head_sha", "requested_base_sha", "comparison_base_sha")},
        **{key: {"type": "string", "maxLength": limit} for key, limit in HISTORICAL_TEXT_MAX_CHARS.items()},
        "context": {"type": "string", "maxLength": HISTORICAL_CONTEXT_MAX_BYTES},
    }),
    "materialization": strict_object({
        "kind": {"enum": ["exact", "synthetic"]},
        **{key: _SHA for key in (
            "head_sha", "requested_base_sha", "comparison_base_sha", "head_tree_sha", "comparison_tree_sha",
        )},
        "review_diff_key": _DIGEST,
        "bundle_sha256": {"anyOf": [_DIGEST, {"type": "null"}]},
    }),
})


def validate_immutable_input(value: dict[str, Any]) -> None:
    """Reject unknown fields/versions and contradictory original/local identities."""
    try:
        validate_json_schema(value, IMMUTABLE_REVIEW_INPUT_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"invalid immutable review input at {exc.json_path}") from exc
    original, local = value["provenance"], value["materialization"]
    if type(value["schema_version"]) is not int:
        raise ValueError("immutable input version must be an exact integer")
    if any(part in {".", ".."} for part in original["repository"].split("/")):
        raise ValueError("immutable repository identity is malformed")
    if type(original["pr_number"]) is not int:
        raise ValueError("immutable PR number must be an exact positive integer")
    # Encoding also refuses author text that is not well-formed Unicode; only context has a byte bound.
    encoded = {key: original[key].encode() for key in ("title", "body", "context")}
    if len(encoded["context"]) > HISTORICAL_CONTEXT_MAX_BYTES:
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
