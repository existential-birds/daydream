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
    with pytest.raises(InitError):
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


def test_validate_fresh_workspace_returns_2(tmp_path):
    from daydream.benchmark.workspace import init_workspace, validate_workspace

    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["h1.example.com"], ["h2.example.com"])
    code, label = validate_workspace(root)
    assert code == 2  # structurally valid but incomplete (unresolved repo identity)
    assert "incomplete" in label


def test_validate_corrupt_manifest_returns_1(tmp_path):
    from daydream.benchmark.workspace import init_workspace, validate_workspace

    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["h1.example.com"], ["h2.example.com"])
    (root / "benchmark.yaml").write_text("schema_version: 1\nbogus_key: true\n")
    code, label = validate_workspace(root)
    assert code == 1  # schema corruption
    assert "corrupt" in label.lower() or "invalid" in label.lower()


def test_validate_missing_manifest_returns_1(tmp_path):
    from daydream.benchmark.workspace import init_workspace, validate_workspace

    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["h1.example.com"], ["h2.example.com"])
    (root / "benchmark.yaml").unlink()
    code, _ = validate_workspace(root)
    assert code == 1


def _write_curated_workspace(tmp_path, curation_state, *, resolved=True):
    """Build a workspace whose single indexed case carries ``curation_state``.

    Reuses ``init_workspace`` for the base layout then resolves the source
    identity and adds one ready/unready case so ``validate_workspace`` /
    ``workspace_status`` exercise the case-driven readiness path (regression
    for the finding that ``cases=[]`` made exit 0 unreachable).
    """
    import yaml

    from daydream.benchmark.workspace import init_workspace

    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["h1.example.com"], ["h2.example.com"])
    raw = yaml.safe_load((root / "benchmark.yaml").read_text())
    if resolved:
        raw["source"]["repository_id"] = 12345
        raw["source"]["visibility"] = "private"
    case_id = "pr-000101-0123456789ab"
    case_file = f"cases/{case_id}.yaml"
    raw["cases"] = [{"case_id": case_id, "pr_number": 101, "case_file": case_file}]
    (root / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    (root / "cases").mkdir(parents=True, exist_ok=True)
    (root / case_file).write_text(
        yaml.safe_dump(
            {"schema_version": 1, "case_id": case_id, "curation": {"state": curation_state}},
            sort_keys=False,
        )
    )
    return root


def test_validate_ready_workspace_returns_0(tmp_path):
    # A resolved, fully-curated workspace must be able to reach the documented
    # exit 0 ("ready") — it was previously unreachable because derive_workspace_state
    # was fed cases=[].
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    code, label = validate_workspace(root)
    assert code == 0
    assert label == "ready"


def test_validate_curating_workspace_returns_2(tmp_path):
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "draft")
    code, label = validate_workspace(root)
    assert code == 2
    assert "incomplete" in label


def test_validate_unresolved_but_ready_case_still_returns_2(tmp_path):
    # Readiness of the cases does not trump an unresolved repository identity.
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready", resolved=False)
    code, label = validate_workspace(root)
    assert code == 2
    assert "incomplete" in label


def test_status_derives_ready_from_curated_cases(tmp_path):
    from daydream.benchmark.workspace import workspace_status

    root = _write_curated_workspace(tmp_path, "ready")
    st = workspace_status(root)
    assert st.workspace_state == "ready"
    assert st.repository_identity_resolved is True


def _seed_frozen_case(ws):
    """Seed one ``ready`` snapshot case + its bundle + the indexed ledger.

    Builds on ``_write_curated_workspace``'s ready case shape, adding the
    frozen ``snapshot.ready`` block (bundle_file + bundle_sha256) and writing a
    real ``snapshots/<case>.bundle`` whose sha256 matches. Resolves the source
    identity so ``validate_workspace`` reaches exit 0.
    """
    import hashlib

    import yaml

    raw = yaml.safe_load((ws / "benchmark.yaml").read_text())
    raw["source"]["repository_id"] = 12345
    raw["source"]["visibility"] = "private"
    head_sha = "0123456789ab" + "0" * 28
    case_id = f"pr-000101-{head_sha[:12]}"
    case_file = f"cases/{case_id}.yaml"
    raw["cases"] = [{"case_id": case_id, "pr_number": 101, "case_file": case_file}]
    (ws / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    bundle_rel = f"snapshots/{case_id}.bundle"
    bundle_bytes = b"frozen-snapshot-bundle-bytes"
    bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()
    (ws / "cases").mkdir(parents=True, exist_ok=True)
    (ws / "snapshots").mkdir(parents=True, exist_ok=True)
    (ws / bundle_rel).write_bytes(bundle_bytes)
    (ws / case_file).write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "case_id": case_id,
                "curation": {"state": "ready"},
                "snapshot": {
                    "status": "ready",
                    "policy": "final_pr_head",
                    "requested_head": "final",
                    "original_base_sha": "b" * 40,
                    "original_head_sha": head_sha,
                    "base_tree_sha": "1" * 40,
                    "head_tree_sha": "2" * 40,
                    "diff_sha256": "3" * 64,
                    "bundle_file": bundle_rel,
                    "bundle_sha256": bundle_sha,
                    "error": None,
                },
            },
            sort_keys=False,
        )
    )
    return ws


def test_ready_bundle_checksum_mismatch_is_validate_corruption(tmp_path):
    from daydream.benchmark.storage import load_yaml_strict
    from daydream.benchmark.workspace import init_workspace, validate_workspace

    ws = tmp_path / "ws"
    init_workspace(ws, "o/r", ["api.anthropic.com"], ["api.anthropic.com"])
    _seed_frozen_case(ws)   # one ready case: bundle YAML + snapshots/<case>.bundle
    code, _ = validate_workspace(ws)
    assert code == 0
    # Corrupt the bundle bytes (keeps the case document and ledger intact).
    case = load_yaml_strict(next((ws / "cases").glob("*.yaml")))
    (ws / case["snapshot"]["bundle_file"]).write_bytes(b"tampered")
    code2, label = validate_workspace(ws)
    assert code2 == 1 and "corrupt" in label
    # Case state on disk is unchanged.
    case_after = load_yaml_strict(next((ws / "cases").glob("*.yaml")))
    assert case_after["snapshot"]["status"] == "ready"


def test_status_reports_snapshot_state_per_case(tmp_path, capsys):
    """``status`` surfaces each case's snapshot state + frozen head prefix."""
    from daydream.benchmark.cli import _handle_benchmark_command
    from daydream.benchmark.workspace import init_workspace, workspace_status

    ws = tmp_path / "ws"
    init_workspace(ws, "o/r", ["api.anthropic.com"], ["api.anthropic.com"])
    _seed_frozen_case(ws)   # one ready case (from Task 11's helper)
    st = workspace_status(ws)
    assert st.workspace_state in ("ready", "curating")
    rc = _handle_benchmark_command(["status", str(ws)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ready" in out and "pr-000101-0123456789ab" in out and "0123456789ab" in out
