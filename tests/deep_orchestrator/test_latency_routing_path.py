"""The routing record: one artifact, one writer, one meaning (issue #732)."""
from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest

from daydream.deep.routing_record import read_routing_record, write_routing_record
from daydream.runner import run
from tests.harness.review_profile import independent_alternatives_profile
from tests.harness.stub_backend import install_stub_backend, silence
from tests.test_deep_orchestrator import MakeConfig, Mute


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


async def test_skipped_wonder_records_profile_signals_and_reason(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    mute_side_effects: Mute,
) -> None:
    """MH7: the routing record states why wonder did not run."""
    silence(monkeypatch)
    install_stub_backend(monkeypatch, multi_stack_target)
    mute_side_effects()
    config = make_config(
        multi_stack_target,
        latency_profile="fast",
        review_profile=independent_alternatives_profile(),
    )
    with anyio.fail_after(90):
        assert await run(config) in (0, 1)

    record = read_routing_record(multi_stack_target / ".daydream" / "deep")
    assert record["wonder"]["outcome"] == "skip"
    assert record["profile"]["selected"] == "fast"
    assert record["risk"]["floors"] == []
    assert "fast" in record["wonder"]["reason"]
