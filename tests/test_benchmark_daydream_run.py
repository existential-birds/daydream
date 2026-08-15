"""Tests for the non-interactive daydream review subprocess wrapper."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from daydream.benchmark.daydream_run import (
    DaydreamArtifactError,
    DaydreamRunError,
    _is_transient,
    _review_complete,
    run_daydream_review,
)

# A real `PiError: terminated` Backend-Execution-Error panel as captured in
# daydream's stdout when the z.ai/GLM provider drops the streaming connection.
PIERROR_TERMINATED_STDOUT = """\
Reviewing changed files...

╔═ ⚠️  Backend Execution Error ═════════════════════════════════════════════
║ PiError: terminated
╚═══════════════════════════════════════════════════════════════════════════

╔═ ⚠️  Fatal Error ═════════════════════════════════════════════════════════
║ terminated
╚═══════════════════════════════════════════════════════════════════════════
"""

ECONNRESET_STDOUT = """\
╔═ ⚠️  Backend Execution Error ═════════════════════════════════════════════
║ PiError: read ECONNRESET
╚═══════════════════════════════════════════════════════════════════════════
"""

SOCKET_HANG_UP_STDOUT = """\
╔═ ⚠️  Fatal Error ═════════════════════════════════════════════════════════
║ socket hang up
╚═══════════════════════════════════════════════════════════════════════════
"""

OVERLOADED_429_STDOUT = """\
╔═ ⚠️  Backend Execution Error ═════════════════════════════════════════════
║ 429 service may be temporarily overloaded, please try again later
╚═══════════════════════════════════════════════════════════════════════════
"""

# A clean, successful review with NO error. Deliberately mentions "error" in
# benign prose ("error handling", "ECONNRESET") to guard against over-broad
# matching that would retry a perfectly good review forever.
CLEAN_SUCCESS_STDOUT = """\
Review complete. 2 findings written to findings.json.

  - [medium] The new fetch() call has no error handling; a dropped
    connection (e.g. ECONNRESET) would crash the worker. Wrap it in
    try/except and log the failure.
  - [low] The "terminated" status string is hardcoded; prefer an enum.

Done.
"""

# A complete review: merged-items artifact with an items list + a trajectory
# whose final_metrics is populated. This is exactly what's on disk after a
# tail-end stream-drop (the review finished; only the closing socket died).
COMPLETE_ITEMS = {"schema_version": 1, "items": [{"id": "a"}, {"id": "b"}]}
COMPLETE_TRAJ = {
    "final_metrics": {
        "total_prompt_tokens": 100,
        "total_completion_tokens": 50,
        "total_cached_tokens": 0,
        "total_cost_usd": 0.0,
    }
}


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


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
    sentinel = "ghp_" + "x" * 16

    class FakeProc:
        def __init__(self):
            self.stdout = iter(["a\n", f"token={sentinel}\n"])
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
        run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json", on_line=lines.append)
    assert lines == ["a\n", "token=[REDACTED_API_KEY]\n"]  # each callback line redacted
    assert sentinel not in str(e.value)                     # retained tail is redacted
    assert "[REDACTED_API_KEY]" in str(e.value)


@pytest.mark.parametrize(
    ("begin_line", "end_line", "body"),
    [
        (
            "-----BEGIN RSA PRIVATE KEY-----\n",
            "-----END RSA PRIVATE KEY-----\n",
            "PRIVATEKEYMATERIAL" * 20,
        ),
        ("-----BEGIN PRIVATE KEY-----\n", "-----END PRIVATE KEY-----\n", "PKCS8KEYMATERIAL" * 20),
        (
            "-----BEGIN ENCRYPTED PRIVATE KEY-----\n",
            "-----END ENCRYPTED PRIVATE KEY-----\n",
            "ENCRYPTEDKEYMATERIAL" * 20,
        ),
        (
            "-----BEGIN OPENSSH PRIVATE KEY-----\n",
            "-----END OPENSSH PRIVATE KEY-----\n",
            "OPENSSHKEYMATERIAL" * 20,
        ),
        ("-----BEGIN EC PRIVATE KEY-----\n", "-----END EC PRIVATE KEY-----\n", "ECKEYMATERIAL" * 20),
        ("-----BEGIN DSA PRIVATE KEY-----\n", "-----END DSA PRIVATE KEY-----\n", "DSAKEYMATERIAL" * 20),
    ],
    ids=["pkcs1", "pkcs8", "encrypted", "openssh", "ec", "dsa"],
)
def test_streamed_failure_redacts_pem_block_spanning_lines(
    tmp_path, monkeypatch, begin_line: str, end_line: str, body: str
):
    """A PEM block split across streamed lines (BEGIN/body/END on separate
    lines) must still be redacted whole before reaching on_line and the failure
    tail — per-line redaction would never present BEGIN and END together, so
    the base64 body would leak raw. Parametrized over every variant the
    buffering anchors recognize, so EC/DSA/ENCRYPTED/PKCS8 blocks get the same
    streamed coverage as RSA/OPENSSH."""
    lines: list[str] = []

    class FakeProc:
        def __init__(self):
            self.stdout = iter([
                begin_line,
                body + "\n",
                end_line,
            ])
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
        run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json", on_line=lines.append)
    assert body not in "".join(lines)   # live echo carries no raw PEM body
    assert "[REDACTED_PEM_KEY]" in "".join(lines)
    assert body not in str(e.value)     # retained tail carries no raw PEM body
    assert "[REDACTED_PEM_KEY]" in str(e.value)


def test_streamed_failure_redacts_pem_block_truncated_at_eof(tmp_path, monkeypatch):
    """A stream that ends mid-PEM-block (BEGIN/body seen, END never arrives)
    must still redact the held fragment before it reaches on_line and the
    failure tail — per-line redaction could never match _PEM_KEY_PATTERN's
    BEGIN+END anchors, so the buffered fragment must be redacted whole."""
    lines: list[str] = []
    body = "OPENSSHKEYMATERIAL" * 20

    class FakeProc:
        def __init__(self):
            self.stdout = iter([
                "-----BEGIN OPENSSH PRIVATE KEY-----\n",
                body + "\n",
            ])
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
        run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json", on_line=lines.append)
    assert body not in "".join(lines)   # live echo carries no raw PEM body
    assert "[REDACTED_PEM_KEY]" in "".join(lines)
    assert body not in str(e.value)     # retained tail carries no raw PEM body
    assert "[REDACTED_PEM_KEY]" in str(e.value)


def test_captured_failure_redacts_before_tail_truncation(tmp_path, monkeypatch):
    """A PEM block whose tail would be sliced off mid-block must still be fully
    redacted before the _STDERR_TAIL cut — redact-after-slice would leave the
    raw END line (BEGIN already cut away) leaking."""
    monkeypatch.setattr("daydream.benchmark.daydream_run._STDERR_TAIL", 80)
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        + ("PRIVATEKEYMATERIAL" * 40)
        + "\n-----END RSA PRIVATE KEY-----\n"
    )
    combined = "z" * 50 + pem  # the last-80 tail contains the END but not the BEGIN
    monkeypatch.setattr(
        "daydream.benchmark.daydream_run.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=combined, stderr=""),
    )
    checkout = tmp_path / "co"
    checkout.mkdir()
    with pytest.raises(DaydreamRunError) as e:
        run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json")
    tail = str(e.value)
    assert "PRIVATEKEYMATERIAL" not in tail
    assert "-----END RSA PRIVATE KEY-----" not in tail
    assert "[REDACTED_PEM_KEY]" in tail


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
        run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json", on_line=lambda _: None)
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


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        pytest.param(PIERROR_TERMINATED_STDOUT, True, id="pierror-terminated"),
        pytest.param(ECONNRESET_STDOUT, True, id="econnreset"),
        pytest.param(SOCKET_HANG_UP_STDOUT, True, id="socket-hang-up"),
        pytest.param(OVERLOADED_429_STDOUT, True, id="429-overload"),
        pytest.param(CLEAN_SUCCESS_STDOUT, False, id="clean-success"),
        pytest.param("", False, id="empty"),
    ],
)
def test_transient_output_classifier(output, expected):
    """Distinguish retryable transport output from successful subprocess output."""
    assert _is_transient(output) is expected


def test_review_complete_when_artifact_and_metrics_present(tmp_path):
    artifact, traj = tmp_path / "merged-items.json", tmp_path / "traj.json"
    _write(artifact, COMPLETE_ITEMS)
    _write(traj, COMPLETE_TRAJ)
    assert _review_complete(artifact, traj)


def test_review_incomplete_when_artifact_missing(tmp_path):
    artifact, traj = tmp_path / "merged-items.json", tmp_path / "traj.json"
    _write(traj, COMPLETE_TRAJ)
    assert not _review_complete(artifact, traj)


def test_review_incomplete_when_items_key_absent(tmp_path):
    artifact, traj = tmp_path / "merged-items.json", tmp_path / "traj.json"
    _write(artifact, {"schema_version": 1})  # no "items" array
    _write(traj, COMPLETE_TRAJ)
    assert not _review_complete(artifact, traj)


def test_review_incomplete_when_trajectory_lacks_final_metrics(tmp_path):
    artifact, traj = tmp_path / "merged-items.json", tmp_path / "traj.json"
    _write(artifact, COMPLETE_ITEMS)
    _write(traj, {"steps": []})  # trajectory written but no final_metrics
    assert not _review_complete(artifact, traj)


def test_tail_drop_with_complete_artifacts_returns_artifact_without_retry(tmp_path, monkeypatch):
    """daydream 'completes' (artifact + metrics on disk) but the closing stream
    dies — exits non-zero with a PiError: terminated panel. That is a success."""
    checkout = tmp_path / "co"
    traj = tmp_path / "t.json"
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        _write(checkout / ".daydream" / "deep" / "merged-items.json", COMPLETE_ITEMS)
        _write(traj, COMPLETE_TRAJ)
        return SimpleNamespace(returncode=1, stdout=PIERROR_TERMINATED_STDOUT, stderr="")

    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.run", fake_run)
    out = run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=traj)

    assert calls["n"] == 1  # must NOT re-run a completed review
    assert out == checkout / ".daydream" / "deep" / "merged-items.json"
    assert json.loads(out.read_text())["items"] == COMPLETE_ITEMS["items"]


def test_stale_artifacts_from_prior_run_do_not_rescue_a_failed_attempt(tmp_path, monkeypatch):
    """The checkout is reused and .daydream/ is gitignored, so a previous run can
    leave complete artifacts behind. A subprocess that crashes before writing
    anything must be reported as a failure, not salvaged by those stale files."""
    checkout = tmp_path / "co"
    traj = tmp_path / "t.json"
    artifact = checkout / ".daydream" / "deep" / "merged-items.json"
    _write(artifact, COMPLETE_ITEMS)
    _write(traj, COMPLETE_TRAJ)
    seen: list[bool] = []

    def fake_run(cmd, **kw):
        seen.append(artifact.exists())
        return SimpleNamespace(returncode=1, stdout="daydream: unknown flag --base\n", stderr="")

    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.run", fake_run)
    with pytest.raises(DaydreamRunError, match="exit 1"):
        run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=traj)

    assert seen == [False]  # stale outputs cleared before the attempt ran
    assert not artifact.exists()
    assert not traj.exists()


def test_stale_artifacts_do_not_preempt_transient_retries(tmp_path, monkeypatch):
    """A transient failure must still consume its retries even when a prior run
    left complete artifacts on disk; the salvage path must not short-circuit it."""
    checkout = tmp_path / "co"
    traj = tmp_path / "t.json"
    artifact = checkout / ".daydream" / "deep" / "merged-items.json"
    _write(artifact, COMPLETE_ITEMS)
    _write(traj, COMPLETE_TRAJ)
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        return SimpleNamespace(returncode=1, stdout=OVERLOADED_429_STDOUT, stderr="")

    monkeypatch.setattr("daydream.benchmark.daydream_run._RETRY_BACKOFF_S", 0)
    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.run", fake_run)
    with pytest.raises(DaydreamRunError, match="exit 1"):
        run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=traj)

    assert calls["n"] == 3  # 1 initial + 2 retries, not rescued on attempt 1


def test_transient_failure_without_artifacts_retries_then_raises(tmp_path, monkeypatch):
    """No artifact ever written -> the loop must exhaust its retries and raise."""
    checkout = tmp_path / "co"
    checkout.mkdir()
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        return SimpleNamespace(returncode=1, stdout=PIERROR_TERMINATED_STDOUT, stderr="")

    monkeypatch.setattr("daydream.benchmark.daydream_run._RETRY_BACKOFF_S", 0)
    monkeypatch.setattr("daydream.benchmark.daydream_run.subprocess.run", fake_run)
    with pytest.raises(DaydreamRunError, match="exit 1"):
        run_daydream_review(checkout, base_sha="d" * 40, trajectory_path=tmp_path / "t.json")

    assert calls["n"] == 3  # 1 initial + 2 retries
