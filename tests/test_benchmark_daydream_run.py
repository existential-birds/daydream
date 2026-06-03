"""Tests for the non-interactive daydream review subprocess wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from daydream.benchmark.daydream_run import DaydreamArtifactError, run_daydream_review


def test_runs_daydream_noninteractive_with_pinned_base_and_trajectory(tmp_path, monkeypatch):
    checkout = tmp_path / "co"
    (checkout / ".daydream" / "deep").mkdir(parents=True)
    cap = {}

    def fake_run(cmd, **kw):
        cap["cmd"] = cmd
        (checkout / ".daydream" / "deep" / "merged-items.json").write_text('{"items": []}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.run", fake_run)
    out = run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json")
    cmd = cap["cmd"]
    assert "--non-interactive" in cmd
    assert cmd[cmd.index("--base") + 1] == "d" * 40
    assert cmd[cmd.index("--trajectory") + 1] == str(tmp_path / "t.json")
    assert str(checkout) in cmd
    assert out == checkout / ".daydream" / "deep" / "merged-items.json" and out.exists()


def test_raises_when_artifact_missing(tmp_path, monkeypatch):
    checkout = tmp_path / "co"
    checkout.mkdir()
    monkeypatch.setattr(
        "daydream.benchmark.daydream_run.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    with pytest.raises(DaydreamArtifactError):
        run_daydream_review(checkout, base_sha="x", trajectory_path=tmp_path / "t.json")
