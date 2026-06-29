"""Tests for the non-interactive daydream review subprocess wrapper."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from daydream.benchmark.daydream_run import DaydreamArtifactError, DaydreamRunError, run_daydream_review


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


def test_forwards_backend_model_argv_and_provider_env(tmp_path, monkeypatch):
    checkout = tmp_path / "co"
    (checkout / ".daydream" / "deep").mkdir(parents=True)
    cap = {}

    def fake_run(cmd, **kw):
        cap["cmd"] = cmd
        cap["env"] = kw.get("env")
        (checkout / ".daydream" / "deep" / "merged-items.json").write_text('{"items": []}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.run", fake_run)
    run_daydream_review(
        checkout,
        base_sha="d" * 40,
        trajectory_path=tmp_path / "t.json",
        backend="pi",
        model="glm-5.2",
        provider="openrouter",
    )
    cmd = cap["cmd"]
    assert cmd[cmd.index("--backend") + 1] == "pi"
    assert cmd[cmd.index("--model") + 1] == "glm-5.2"
    assert "--provider" not in cmd  # provider never argv
    assert cap["env"]["PI_PROVIDER"] == "openrouter"  # provider via env


def test_no_overrides_matches_today(tmp_path, monkeypatch):
    checkout = tmp_path / "co"
    (checkout / ".daydream" / "deep").mkdir(parents=True)
    cap = {}

    def fake_run(cmd, **kw):
        cap["cmd"] = cmd
        cap["env"] = kw.get("env")
        (checkout / ".daydream" / "deep" / "merged-items.json").write_text('{"items": []}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.run", fake_run)
    run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json")
    cmd = cap["cmd"]
    assert "--backend" not in cmd
    assert "--model" not in cmd
    assert "PI_PROVIDER" not in cap["env"]


def test_clears_inherited_provider_when_no_override(tmp_path, monkeypatch):
    """A shell-exported PI_PROVIDER must not leak into a run with no --reviewer-provider:
    the ``provider`` argument is the single source of truth, so the inherited value is dropped."""
    checkout = tmp_path / "co"
    (checkout / ".daydream" / "deep").mkdir(parents=True)
    cap = {}

    def fake_run(cmd, **kw):
        cap["env"] = kw.get("env")
        (checkout / ".daydream" / "deep" / "merged-items.json").write_text('{"items": []}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("PI_PROVIDER", "leaked-from-shell")
    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.run", fake_run)
    run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json")
    assert "PI_PROVIDER" not in cap["env"]


def test_strips_github_app_creds_from_review_env(tmp_path, monkeypatch):
    """Regression: App creds in the operator's shell must not reach the review
    subprocess — they make daydream attempt App auth for the upstream owner and
    exit 1 before any review runs."""
    checkout = tmp_path / "co"
    (checkout / ".daydream" / "deep").mkdir(parents=True)
    cap = {}

    def fake_run(cmd, **kw):
        cap["env"] = kw.get("env")
        (checkout / ".daydream" / "deep" / "merged-items.json").write_text('{"items": []}')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("DAYDREAM_APP_ID", "4014446")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----")
    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.run", fake_run)
    run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json")
    assert "DAYDREAM_APP_ID" not in cap["env"]
    assert "DAYDREAM_APP_PRIVATE_KEY" not in cap["env"]


def test_streams_lines_and_keeps_tail_on_failure(tmp_path, monkeypatch):
    lines: list[str] = []

    class FakeProc:
        def __init__(self):
            self.stdout = iter(["a\n", "boom\n"])
            self.returncode = 1

        def wait(self, timeout=None):
            return 1

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.Popen", lambda *a, **k: FakeProc())
    checkout = tmp_path / "co"
    checkout.mkdir()
    with pytest.raises(DaydreamRunError) as e:
        run_daydream_review(
            checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json", on_line=lines.append
        )
    assert lines == ["a\n", "boom\n"]  # streamed live
    assert "boom" in str(e.value)  # tail retained in the error


def test_streamed_timeout_kills_quiet_child_holding_stdout_open(tmp_path, monkeypatch):
    """A child that keeps stdout open but emits nothing must be killed by the
    wall-clock deadline rather than blocking the read loop forever."""

    class BlockingStdout:
        """Iterator whose ``__next__`` blocks until the proc is killed."""

        def __init__(self):
            self.released = threading.Event()

        def __iter__(self):
            return self

        def __next__(self):
            self.released.wait()  # unblocks only on kill(); never yields a line
            raise StopIteration

    class FakeProc:
        def __init__(self):
            self.stdout = BlockingStdout()
            self.killed = False

        def kill(self):
            self.killed = True
            self.stdout.released.set()

        def wait(self, timeout=None):
            return 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    proc = FakeProc()
    monkeypatch.setattr("daydream.benchmark.daydream_run._DAYDREAM_TIMEOUT", 0.2)
    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.Popen", lambda *a, **k: proc)
    checkout = tmp_path / "co"
    checkout.mkdir()

    start = time.monotonic()
    with pytest.raises(DaydreamRunError, match="timed out after 0.2s"):
        run_daydream_review(
            checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json", on_line=lambda _: None
        )
    elapsed = time.monotonic() - start
    assert proc.killed  # the deadline killed the child
    assert elapsed < 5  # bounded by the (patched) timeout, not hung


def test_raises_when_artifact_missing(tmp_path, monkeypatch):
    checkout = tmp_path / "co"
    checkout.mkdir()
    monkeypatch.setattr(
        "daydream.benchmark.daydream_run.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    with pytest.raises(DaydreamArtifactError):
        run_daydream_review(checkout, base_sha="x", trajectory_path=tmp_path / "t.json")
