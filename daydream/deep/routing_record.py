"""The per-run latency-routing record (issue #732).

One artifact, one writer, one meaning. ``latency-routing.json`` is *evidence*,
not an input: no step reads it to decide behaviour, so a missing or corrupt
record can never fail a run. Each step that owns a routing decision appends its
slice through :func:`write_routing_record`, which read-modify-writes top-level
keys so a later write never clobbers an earlier one and a ``--start-at`` resume
does not erase a prior run's record.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from daydream.deep.artifacts import DeepArtifact
from daydream.json_utils import read_json_object


def read_routing_record(deep_dir_path: Path) -> dict[str, Any]:
    """Return the routing record, or ``{}`` when absent, non-object, or malformed."""
    return read_json_object(DeepArtifact.LATENCY_ROUTING.at(deep_dir_path))


def write_routing_record(deep_dir_path: Path, updates: Mapping[str, Any]) -> Path:
    """Merge ``updates`` into the record and return the written path.

    Top-level keys in ``updates`` replace the existing value unless both are
    mappings, in which case their contents are merged one level deep (so a step
    can add ``arbiter`` without dropping the ``wonder`` slice already written).
    """
    path = DeepArtifact.LATENCY_ROUTING.at(deep_dir_path)
    merged = dict(read_routing_record(deep_dir_path))
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            nested = dict(current)
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = ["read_routing_record", "write_routing_record"]
