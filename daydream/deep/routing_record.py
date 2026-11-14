"""The per-run latency-routing record (issue #732).

Earlier builds wrote ``latency-routing.json`` as evidence of their routing
decisions; this build writes none. The reader stays for the archive evaluation
and for embedders: a missing or corrupt record reads as empty and never fails a run.
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
