"""The routing record: one artifact, one writer, one meaning (issue #732)."""
from __future__ import annotations

import json
from pathlib import Path

from daydream.deep.routing_record import read_routing_record, write_routing_record


def test_routing_record_merges_instead_of_clobbering(tmp_path: Path) -> None:
    dd = tmp_path / "deep"
    dd.mkdir()
    write_routing_record(dd, {"profile": {"selected": "balanced"}, "risk": {"floors": ["security_surface"]}})
    path = write_routing_record(dd, {"wonder": {"outcome": "run", "effort": "medium", "reason": "profile floor"}})

    record = read_routing_record(dd)
    assert path.name == "latency-routing.json"
    assert json.loads(path.read_text(encoding="utf-8"))["profile"]["selected"] == "balanced"
    assert record["profile"]["selected"] == "balanced"
    assert record["risk"]["floors"] == ["security_surface"]
    assert record["wonder"]["reason"] == "profile floor"


def test_absent_record_reads_as_an_empty_mapping(tmp_path: Path) -> None:
    assert read_routing_record(tmp_path) == {}
