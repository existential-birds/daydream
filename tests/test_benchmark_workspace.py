import stat

import pytest

from daydream.benchmark.schema import BenchmarkManifest, CaseDocument, ImportDocument
from daydream.benchmark.storage import load_json_strict, load_yaml_strict
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


def _write_case_docs(root, curation_state):
    """Write a fully-valid ledger + import + bundle + case doc into ``root``.

    The workspace at ``root`` must already be ``init_workspace``-created. The
    artifacts written here are all schema-valid: a resolved manifest, a
    ``fetched`` ledger entry referencing a real import file + sha256, a real
    bundle whose sha256 matches the case doc's ``snapshot.bundle_sha256``, and
    a ``CaseDocument`` whose ``pull_request``/``snapshot``/``source``/
    ``curation`` model-validate without any ``_schema_ready`` strip.
    """
    import hashlib

    import yaml

    from daydream.benchmark.schema import (
        CaseDocument,
        CaseSource,
        ImportDocument,
        PullRequestEntry,
        PullRequestMeta,
        SnapshotReady,
        derive_finding_id,
    )

    raw = yaml.safe_load((root / "benchmark.yaml").read_text())
    raw["source"]["repository_id"] = "R_kgDOABC123"
    raw["source"]["visibility"] = "private"

    head_sha = "0123456789ab" + "0" * 28
    base_sha = "b" * 40
    case_id = f"pr-000101-{head_sha[:12]}"
    case_file = f"cases/{case_id}.yaml"
    import_file = "imports/pr-000101.json"
    bundle_rel = f"snapshots/{case_id}.bundle"

    pr_meta = PullRequestMeta(
        number=101,
        url="https://github.com/O/R/pull/101",
        title="Fix cache",
        state="open",
        base={"ref": "main", "sha": base_sha},
        head={"ref": "feature/cache", "sha": head_sha},
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        author={"login": "alice", "type": "User"},
        body="Fix the cache invalidation on write.",
    )
    import_doc = ImportDocument(
        schema_version=1,
        repository={"id": "R_kgDOABC123", "name_with_owner": "O/R", "visibility": "private"},
        pull_request=pr_meta,
        evidence=[],
        fetch={
            "fetched_at": "2026-01-01T00:00:00Z",
            "etag": None,
            "payload_sha256": "0" * 64,
        },
    )
    import_bytes = import_doc.model_dump_json(indent=2).encode("utf-8")
    import_sha256 = hashlib.sha256(import_bytes).hexdigest()

    bundle_bytes = b"frozen-snapshot-bundle-bytes"
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()

    curation: dict
    if curation_state == "ready":
        finding = {
            "finding_id": "f" * 64,
            "title": "Cache is never invalidated",
            "body": "The cache key is stable across writes, so stale data is served.",
            "severity": "high",
            "location": {"path": "feature.py", "start_line": 1, "end_line": 1},
            "provenance": {"kind": "authored", "source_ids": []},
        }
        finding["finding_id"] = derive_finding_id(finding, case_id=case_id)
        curation = {
            "state": "ready",
            "snapshot_attested": True,
            "clean_attested": False,
            "gold_status": "findings",
            "findings": [finding],
            "exclusions": [],
            "case_exclusion": None,
        }
    elif curation_state == "clean":
        curation = {
            "state": "ready",
            "snapshot_attested": True,
            "clean_attested": True,
            "gold_status": "clean",
            "findings": [],
            "exclusions": [],
            "case_exclusion": None,
        }
    else:  # draft
        curation = {
            "state": "draft",
            "snapshot_attested": False,
            "clean_attested": False,
            "gold_status": None,
            "findings": [],
            "exclusions": [],
            "case_exclusion": None,
        }

    case_doc = CaseDocument(
        schema_version=2,
        case_id=case_id,
        pull_request=pr_meta,
        snapshot=SnapshotReady(
            status="ready",
            policy="final_pr_head",
            requested_head="final",
            original_base_sha=base_sha,
            original_head_sha=head_sha,
            base_tree_sha="1" * 40,
            head_tree_sha="2" * 40,
            diff_sha256="3" * 64,
            bundle_file=bundle_rel,
            bundle_sha256=bundle_sha256,
            error=None,
        ),
        source=CaseSource(import_file=import_file, import_sha256=import_sha256),
        curation=curation,
        candidates=[],
    )

    raw["pull_requests"] = [
        PullRequestEntry(
            number=101,
            import_state="fetched",
            import_file=import_file,
            import_sha256=import_sha256,
            case_ids=[case_id],
        ).model_dump(mode="json")
    ]
    raw["cases"] = [
        {"case_id": case_id, "pr_number": 101, "case_file": case_file}
    ]
    (root / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))

    (root / "imports").mkdir(parents=True, exist_ok=True)
    (root / import_file).write_bytes(import_bytes)
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    (root / bundle_rel).write_bytes(bundle_bytes)
    (root / "cases").mkdir(parents=True, exist_ok=True)
    (root / case_file).write_text(
        yaml.safe_dump(case_doc.model_dump(mode="json"), sort_keys=False)
    )
    return root


def _write_curated_workspace(tmp_path, curation_state, *, resolved=True):
    """Build a fully-valid workspace whose single indexed case is curated.

    Reuses ``init_workspace`` for the base layout, resolves the source
    identity (unless ``resolved=False``), and adds a real fetched ledger
    entry + import + bundle + schema-valid case doc so ``validate_workspace``
    / ``workspace_status`` exercise the case-driven readiness path on
    documents that model-validate directly.
    """
    import yaml

    from daydream.benchmark.workspace import init_workspace

    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["h1.example.com"], ["h2.example.com"])
    _write_case_docs(root, curation_state)
    if not resolved:
        raw = yaml.safe_load((root / "benchmark.yaml").read_text())
        raw["source"]["repository_id"] = None
        raw["source"]["visibility"] = "unresolved"
        (root / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
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

    Builds on ``_write_curated_workspace``'s fully-valid ready case shape,
    writing the frozen ``snapshot.ready`` block (bundle_file + bundle_sha256)
    and a real ``snapshots/<case>.bundle`` whose sha256 matches, plus the
    fetched ledger entry + import file. Resolves the source identity so
    ``validate_workspace`` reaches exit 0.
    """
    _write_case_docs(ws, "ready")
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


def test_curated_fixture_writes_schema_valid_case(tmp_path):
    """The curated-workspace fixture itself writes only schema-valid documents."""
    root = _write_curated_workspace(tmp_path, "ready")   # rewritten in this task; same module
    raw = load_yaml_strict(next((root / "cases").glob("*.yaml")))
    CaseDocument.model_validate(raw)                      # must validate WITHOUT _schema_ready (no gold_mode)
    m = BenchmarkManifest.model_validate(load_yaml_strict(root / "benchmark.yaml"))
    pr = m.pull_requests[0]
    assert pr.import_state == "fetched" and pr.import_file and pr.import_sha256
    ImportDocument.model_validate(load_json_strict(root / pr.import_file))   # import round-trips
