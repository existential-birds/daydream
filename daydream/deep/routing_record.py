"""The per-run latency-routing record (issue #732).

One artifact, one writer, one meaning. ``latency-routing.json`` is *evidence*,
not an input: no step reads it to decide behaviour, so a missing or corrupt
record can never fail a run. Each step that owns a routing decision appends its
slice through :func:`write_routing_record`, which read-modify-writes top-level
keys so a later phase write preserves earlier slices. The preamble replaces
snapshot-owned risk and diff-population evidence together on a valid resume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream.deep.artifacts import DeepArtifact
from daydream.json_utils import read_json_object


def read_routing_record(deep_dir_path: Path) -> dict[str, Any]:
    """Return the routing record, or ``{}`` when absent, non-object, or malformed."""
    return read_json_object(DeepArtifact.LATENCY_ROUTING.at(deep_dir_path))


__all__ = ["read_routing_record",]
