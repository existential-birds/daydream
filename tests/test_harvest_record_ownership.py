"""Harvest refuses mismatched immutable inputs before acquisition or mutation."""

from dataclasses import replace
from pathlib import Path

import pytest

from daydream.training.harvest import HarvestConfig, make_harvest_services, run_harvest
from tests.harness.adjudication import record_store, snapshot_id


@pytest.mark.anyio
@pytest.mark.parametrize("mismatch", ["store", "snapshot", "dry_run"])
async def test_entrypoint_rejects_mismatched_services_before_side_effects(tmp_path: Path, mismatch: str) -> None:
    first = record_store(tmp_path / "first")
    second = record_store(tmp_path / "second")
    config = HarvestConfig(first.root, snapshot_id(first), cache_dir=tmp_path / "cache")
    services = make_harvest_services(config)
    if mismatch == "store":
        incoming = replace(config, store_dir=second.root, snapshot_id=snapshot_id(second))
    elif mismatch == "snapshot":
        pin = first.select_snapshot(observed_before="2026-10-04T10:00:00Z")
        incoming = replace(config, snapshot_id=pin["snapshot_id"])
    else:
        incoming = replace(config, dry_run=True)
    before = first.read_records(), second.read_records()
    with pytest.raises(ValueError, match="ownership"):
        await run_harvest(incoming, services=services)
    assert (first.read_records(), second.read_records()) == before
    assert not (tmp_path / "cache").exists()
