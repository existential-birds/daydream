import stat

import pytest

from daydream.benchmark.schema import BenchmarkManifest
from daydream.benchmark.storage import load_yaml_strict
from daydream.benchmark.workspace import InitError, init_workspace


def test_init_creates_private_layout_and_modes(tmp_path):
    root = tmp_path / "review-bench"
    init_workspace(
        root,
        "OWNER/REPO",
        ["api.anthropic.com"],
        ["api.anthropic.com"],
    )
    assert root.exists()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    for sub in ("imports", "cases", "snapshots", "transactions", "runtime", "cache", "harbor"):
        d = root / sub
        assert d.is_dir() and stat.S_IMODE(d.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "benchmark.yaml").stat().st_mode) == 0o600


def test_init_gitignore_ignores_everything_except_itself(tmp_path):
    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["h1.example.com"], ["h2.example.com"])
    gi = (root / ".gitignore").read_text()
    assert "*" in gi and ".gitignore" in gi
    assert stat.S_IMODE((root / ".gitignore").stat().st_mode) == 0o600


def test_init_manifest_is_valid_and_immutable_fields(tmp_path):
    root = tmp_path / "ws"
    m = init_workspace(root, "OWNER/REPO", ["API.Anthropic.COM"], ["api.anthropic.com"])
    assert isinstance(m, BenchmarkManifest)
    raw = load_yaml_strict(root / "benchmark.yaml")
    loaded = BenchmarkManifest.model_validate(raw)
    assert loaded.source.hostname == "github.com"
    assert loaded.source.repository == "OWNER/REPO"
    assert loaded.source.repository_id is None and loaded.source.visibility == "unresolved"
    # reviewer host normalized to lowercase
    assert loaded.privacy.reviewer_allowed_hosts == ["api.anthropic.com"]
    assert loaded.privacy.reviewer_allowed_hosts and loaded.privacy.judge_allowed_hosts


def test_init_refuses_existing_nonempty_dir(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "existing.txt").write_text("x")
    with pytest.raises(InitError):
        init_workspace(root, "O/R", ["h1.example.com"], ["h2.example.com"])


def test_init_refuses_empty_reviewer_hosts(tmp_path):
    with pytest.raises((InitError, ValueError)):
        init_workspace(tmp_path / "ws", "O/R", [], ["h2.example.com"])


def test_status_fresh_workspace_is_empty_and_unresolved(tmp_path):
    from daydream.benchmark.workspace import init_workspace, workspace_status

    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["h1.example.com"], ["h2.example.com"])
    st = workspace_status(root)
    assert st.workspace_state == "empty"
    assert st.repository_identity_resolved is False
    assert st.ledger is not None and st.ledger.pull_requests == []


def test_status_surfaces_unresolved_identity(tmp_path):
    from daydream.benchmark.workspace import init_workspace, workspace_status

    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["h1.example.com"], ["h2.example.com"])
    st = workspace_status(root)
    assert st.source.hostname == "github.com"
    assert st.source.repository_id is None and st.source.visibility == "unresolved"
