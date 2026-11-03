"""Supervision And Parallel Fix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.config_file import load_file_config
from daydream.runner import run
from tests.deep_orchestrator.support import (
    _scan_phase_events,
)
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import (
    MakeConfig,
    Mute,
    _add_to_reviewed_diff,
    _force_interactive,
    _install_stub_backend,
    _merge_item,
    _pin_findings_pr,
    _silence,
)


def _prepare_fix_stub(target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute) -> StubBackend:
    """Drive fix-cycle tests through the same interactive real-path setup."""
    _force_interactive(monkeypatch)
    return _supervision_stub(target, monkeypatch, mute_side_effects)


@pytest.fixture
def stub(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute) -> StubBackend:
    """Interactive stub backend for this module's real-path fix-cycle tests."""
    return _prepare_fix_stub(multi_stack_target, monkeypatch, mute_side_effects)


def _supervision_stub(
    target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Mute | None = None, *, pin_pr: bool = False,
    **install_kwargs: Any,
) -> StubBackend:
    """Silence UI, optionally mute/pin findings-out, then install the stub backend."""
    _silence(monkeypatch)
    if mute_side_effects is not None:
        mute_side_effects()
    if pin_pr:
        _pin_findings_pr(monkeypatch, target)
    return _install_stub_backend(monkeypatch, target, **install_kwargs)


def _findings_out_config(
    make_config: MakeConfig, target: Path, out: Path, trajectory: Path | None = None,
    *, monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Bind the supervision configuration and PR to a clean committed snapshot."""
    _add_to_reviewed_diff(target, [".daydream.toml"])
    _pin_findings_pr(monkeypatch, target)
    kwargs: dict[str, Any] = {"pr_number": 7, "findings_out": str(out), "file_config": load_file_config(target)}
    if trajectory is not None:
        kwargs["trajectory_path"] = trajectory
    return make_config(target, **kwargs)


def _forbid_pr_post(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a findings-out PR poster that fails if it is ever invoked."""
    async def _post_forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("findings-out must not post to the PR")
    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _post_forbidden)

@pytest.mark.parametrize("policy,items,verdicts,action,expected", [
    pytest.param('supervisor = "rules"\nsupervisor_deny_globs = ["vendor/**"]\n',
                 [("vendor/generated.py", "high", "drop this finding"), ("src/app.py", "low", "keep this finding")],
                 {}, "drop", [("keep this finding", "low")], id="rules-drop-denied-file"),
])
async def test_supervision_transforms_canonical_items_before_findings_export(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    mute_side_effects: Mute, policy: str, items: list[tuple[str, str, str]], verdicts: dict[int, dict[str, Any]],
    action: str, expected: list[tuple[str, str]],
) -> None:
    stub = _supervision_stub(multi_stack_target, monkeypatch, mute_side_effects, pin_pr=True)
    stub.merge_items = [_merge_item(index, file, severity, desc=description)
                        for index, (file, severity, description) in enumerate(items, 1)]
    stub.supervise_verdicts = verdicts
    (multi_stack_target / ".daydream.toml").write_text(policy)
    out, trajectory = multi_stack_target / "findings.json", tmp_path / "trajectory.json"
    _forbid_pr_post(monkeypatch)
    assert await run(_findings_out_config(
        make_config, multi_stack_target, out, trajectory, monkeypatch=monkeypatch,
    )) == 0
    payload = json.loads((multi_stack_target / ".daydream" / "deep" / "merged-items.json").read_text())
    descriptions = {description for _, _, description in items}
    assert [(item["description"], item["severity"]) for item in payload["items"]
            if item["description"] in descriptions] == expected
    findings = json.loads(out.read_text())["findings"]
    assert [(finding["title"], finding["severity"]) for finding in findings
            if finding["title"] in descriptions] == expected
    if not expected:
        assert findings == []
    events = _scan_phase_events(multi_stack_target / ".daydream", trajectory, "supervisor_verdict")
    assert any(event.get("metadata", {}).get("finding_id") == 1
               and event["metadata"].get("action") == action for event in events)
    starts = _scan_phase_events(multi_stack_target / ".daydream", trajectory, "phase_start")
    assert any(event.get("metadata", {}).get("stage") == "supervise" for event in starts) is ('"llm"' in policy)
