import json
import os
import stat
import subprocess

import pytest

from daydream.benchmark.schema import BenchmarkManifest, CaseDocument, ImportDocument
from daydream.benchmark.storage import load_json_strict, load_yaml_strict
from daydream.benchmark.workspace import InitError, init_workspace

_SEED_ENV = {
    "GIT_AUTHOR_NAME": "Tester",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "Tester",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


def _git(repo, *args, env=None, check=True):
    proc_env = {**os.environ, **env} if env is not None else os.environ.copy()
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, env=proc_env, check=check
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _commit(repo, message):
    _git(repo, "commit", "-m", message, env=_SEED_ENV)
    return _git(repo, "rev-parse", "HEAD")


def _write_seed(repo, name, content):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _git(repo, "add", name)


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
    for sub in ("imports", "cases", "snapshots", "transactions", "runtime", "cache"):
        d = root / sub
        assert d.is_dir() and stat.S_IMODE(d.stat().st_mode) == 0o700
    assert not (root / "harbor").exists()   # compiled dataset only after a build
    assert stat.S_IMODE((root / "benchmark.yaml").stat().st_mode) == 0o600


def test_init_does_not_create_harbor(tmp_path):
    root = tmp_path / "review-bench"
    init_workspace(
        root,
        "O/R",
        ["api.anthropic.com"],
        ["api.anthropic.com"],
    )
    assert not (root / "harbor").exists()   # compiled dataset only after a build
    for sub in ("imports", "cases", "snapshots", "transactions", "runtime", "cache"):
        assert (root / sub).is_dir()


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


def _seed_local_origin(root):
    """Real local bare origin: main base1->base2->base3 + feature head off base2.

    The feature head adds ``feature.py`` — the ready fixture's finding
    location — so the frozen head tree contains the location the curated
    finding references. Returns ``(origin_url, base_sha, head_sha)``.
    """
    import shutil as _sh

    seed = root.parent
    repo = seed / "local_wt"
    if repo.exists():
        _sh.rmtree(repo)
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _write_seed(repo, "readme.txt", "base1\n")
    _commit(repo, "base1")
    _write_seed(repo, "base.py", "BASE = 2\n")
    base_sha = _commit(repo, "base2")
    _write_seed(repo, "beyond.py", "BEYOND = 3\n")
    _commit(repo, "base3")
    _git(repo, "checkout", "--detach", base_sha)
    (repo / "base.py").write_text("BASE = 20\n")
    _git(repo, "add", "base.py")
    _write_seed(repo, "feature.py", "LINE 1\n")
    head_sha = _commit(repo, "feature")
    bare = seed / "origin_local.git"
    if bare.exists():
        _sh.rmtree(bare)
    bare.mkdir()
    _git(bare, "init", "--bare")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "origin", "main:main")
    _git(repo, "push", "origin", f"{head_sha}:refs/pull/101/head")
    return str(bare), base_sha, head_sha


def _write_case_docs(root, curation_state):
    """Write a fully-valid ledger + import + bundle + case doc into ``root``.

    The workspace at ``root`` must already be ``init_workspace``-created. The
    artifacts written here are all schema-valid: a resolved manifest, a
    ``fetched`` ledger entry referencing a real import file + sha256, a REAL
    deterministic git bundle (built from a seeded local origin whose frozen
    head tree contains ``feature.py``) whose sha256 matches the case doc's
    ``snapshot.bundle_sha256`` and whose tree IDs + canonical diff digest are
    recorded from that origin, and a ``CaseDocument`` whose ``pull_request``/
    ``snapshot``/``source``/``curation`` model-validate without any
    ``_schema_ready`` strip.
    """
    import hashlib

    import yaml

    from daydream.benchmark import snapshot as sn
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
    repo_slug = raw["source"]["repository"]

    # The PR-meta SHAs recorded on the doc stay the fixture's fixed
    # schema-valid values (provenance metadata, never cross-checked by
    # validate); the bundle bytes + tree IDs + digests come from the real
    # origin so the authoritative offline-clone fidelity check passes.
    head_sha = "0123456789ab" + "0" * 28
    base_sha = "b" * 40
    case_id = f"pr-000101-{head_sha[:12]}"
    case_file = f"cases/{case_id}.yaml"
    import_file = "imports/pr-000101.json"
    bundle_rel = f"snapshots/{case_id}.bundle"

    origin_url, real_base_sha, real_head_sha = _seed_local_origin(root)
    sn.ensure_mirror(root, repo_slug, origin_url)
    sn.fetch_pr_refs(root, repo_slug, 101, base_tip=real_base_sha,
                     explicit_shas=[real_head_sha], origin_url=origin_url)
    m = sn.mirror(root)
    bundle_path = root / bundle_rel
    sn.build_bundle(m, real_base_sha, real_head_sha, bundle_path)
    base_tree_sha = sn.rev_parse(m, f"{real_base_sha}^{{tree}}")
    head_tree_sha = sn.rev_parse(m, f"{real_head_sha}^{{tree}}")
    diff_sha256 = sn.canonical_diff_sha256(m, real_base_sha, real_head_sha)
    bundle_sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    pr_meta = PullRequestMeta(
        number=101,
        url=f"https://github.com/{repo_slug}/pull/101",
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
        repository={
            "id": "R_kgDOABC123",
            "name_with_owner": repo_slug,
            "visibility": "private",
        },
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
            "task_spec_sha256": "d" * 64,
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
            "task_spec_sha256": "d" * 64,
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
            requested_base_sha=base_sha,
            original_head_sha=head_sha,
            base_tree_sha=base_tree_sha,
            head_tree_sha=head_tree_sha,
            diff_sha256=diff_sha256,
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
    import shutil

    import yaml

    from daydream.benchmark.workspace import init_workspace

    root = tmp_path / "ws"
    if root.exists():
        shutil.rmtree(root)   # the fixture is re-invocable (b-valid case)
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


def test_validate_restamped_tampered_bundle_fails(tmp_path):
    """Acceptance (b): a checksum-restamped tampered bundle fails validate.

    The recorded ``bundle_sha256`` is re-stamped to match the tampered bytes,
    so the checksum gate alone would accept it; the authoritative offline-clone
    fidelity check must flag the corruption."""
    import hashlib

    import yaml

    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    bundle = next((root / "snapshots").glob("*.bundle"))
    tampered = bundle.read_bytes() + b"INJECTED"          # content changed
    bundle.write_bytes(tampered)
    case_yaml = next((root / "cases").glob("*.yaml"))
    raw = yaml.safe_load(case_yaml.read_text())
    raw["snapshot"]["bundle_sha256"] = hashlib.sha256(tampered).hexdigest()   # restamp
    case_yaml.write_text(yaml.safe_dump(raw, sort_keys=False))
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()        # checksum alone no longer suffices

    # A genuine real-bundle ready workspace still passes as ready.
    root2 = _write_curated_workspace(tmp_path, "ready")
    code2, label2 = validate_workspace(root2)
    assert code2 == 0 and label2 == "ready"


def test_validate_missing_cache_dir_maps_to_corrupt(tmp_path):
    """A ready workspace whose ``cache/`` scratch dir is absent maps to exit 1.

    ``validate_offline_clone``'s mkdtemp raises FileNotFoundError when
    ``root/cache`` is gone; it must surface as corruption (exit 1 + label) per
    the no-raw-traceback contract — like the sibling freeze path's ``OSError``
    catch — never a bare traceback. ``workspace_status`` raises
    :class:`WorkspaceCorrupt` for the same state.
    """
    import shutil

    from daydream.benchmark.storage import WorkspaceCorrupt
    from daydream.benchmark.workspace import validate_workspace, workspace_status

    root = _write_curated_workspace(tmp_path, "ready")
    assert validate_workspace(root) == (0, "ready")
    shutil.rmtree(root / "cache")
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()
    with pytest.raises(WorkspaceCorrupt):
        workspace_status(root)


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


def _write_minimal_invalid_workspace(tmp_path, curation_state="ready"):
    """A workspace whose case doc is the OLD minimal shape (Task 2 removed)."""
    import yaml

    root = _write_curated_workspace(tmp_path, curation_state)
    case_file = next((root / "cases").glob("*.yaml"))
    case_file.write_text(
        yaml.safe_dump(
            {"schema_version": 1, "case_id": "pr-000101-0123456789ab",
             "curation": {"state": curation_state}},
            sort_keys=False,
        )
    )
    return root


def test_validate_minimal_invalid_ready_returns_1(tmp_path):
    from daydream.benchmark.workspace import validate_workspace

    root = _write_minimal_invalid_workspace(tmp_path, "ready")
    code, label = validate_workspace(root)
    assert code == 1
    assert "corrupt" in label.lower()


def test_status_rejects_minimal_invalid_case(tmp_path):
    from daydream.benchmark.storage import WorkspaceCorrupt
    from daydream.benchmark.workspace import workspace_status

    root = _write_minimal_invalid_workspace(tmp_path, "ready")
    with pytest.raises(WorkspaceCorrupt):
        workspace_status(root)


def _restamp_import_sha(tmp_path, imp_bytes: bytes) -> None:
    """Overwrite the import file and re-stamp the ledger sha to match.

    Leaves the import structurally invalid but byte-exact per the ledger, so
    only the model gate (not the checksum gate) can catch it.
    """
    import hashlib

    import yaml

    root = tmp_path / "ws"
    imp = next((root / "imports").glob("pr-*.json"))
    imp.write_bytes(imp_bytes)
    sha = hashlib.sha256(imp_bytes).hexdigest()
    raw = yaml.safe_load((root / "benchmark.yaml").read_text())
    raw["pull_requests"][0]["import_sha256"] = sha
    (root / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))


def test_checksum_restamped_corrupt_import_is_corruption(tmp_path):
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    imp = next((root / "imports").glob("pr-*.json"))
    raw = load_json_strict(imp)
    # Structurally invalid, but its sha is re-stamped to match the ledger.
    raw["pull_request"]["number"] = 999   # wrong PR id; wrong shape vs intent
    _restamp_import_sha(tmp_path, json.dumps(raw).encode())
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()


def test_import_missing_on_disk_is_corruption(tmp_path):
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    (next((root / "imports").glob("pr-*.json"))).unlink()
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()


def _mutate_manifest_case(tmp_path, pr_number=None, case_file=None):
    """Rewrite the manifest cases[] entry (pr_number and/or case_file)."""
    import yaml

    root = tmp_path / "ws"
    raw = yaml.safe_load((root / "benchmark.yaml").read_text())
    row = raw["cases"][0]
    if pr_number is not None:
        row["pr_number"] = pr_number
    if case_file is not None:
        row["case_file"] = case_file
    (root / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))


def _drop_ledger_entry(tmp_path):
    """Remove the pull_requests[] entry for PR 101."""
    import yaml

    root = tmp_path / "ws"
    raw = yaml.safe_load((root / "benchmark.yaml").read_text())
    raw["pull_requests"] = [
        pr for pr in raw["pull_requests"] if pr.get("number") != 101
    ]
    (root / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))


def test_case_pr_number_mismatch_manifest_is_corruption(tmp_path):
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    # manifest cases[] pr_number disagrees with the case doc's pull_request.number
    _mutate_manifest_case(tmp_path, pr_number=999)
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()


def test_case_file_not_exact_index_path_is_corruption(tmp_path):
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    # manifest case_file is not exactly cases/<case_id>.yaml
    _mutate_manifest_case(tmp_path, case_file="cases/other.yaml")
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()


def test_case_pr_absent_from_ledger_is_corruption(tmp_path):
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    _drop_ledger_entry(tmp_path)   # remove the pull_requests[] entry for PR 101
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()


def test_orphan_import_is_corruption(tmp_path):
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    (root / "imports" / "pr-000999.json").write_text('{"unindexed": true}')   # unindexed import
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()


def test_orphan_bundle_is_corruption(tmp_path):
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    (root / "snapshots" / "pr-000999-abcdef012345.bundle").write_bytes(b"orphan")
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()


def test_referenced_bundle_missing_is_corruption(tmp_path):
    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    (next((root / "snapshots").glob("*.bundle"))).unlink()
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()


def test_duplicate_inode_indexed_files_is_corruption(tmp_path):
    import hashlib
    import os

    import yaml

    from daydream.benchmark.workspace import validate_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    # hard-link the import to the bundle path (one inode, two indexed names).
    # the import bytes are a valid ImportDocument whose ledger sha is unchanged;
    # only the bundle sha is re-stamped to the import bytes, so every checksum /
    # model gate stays green and only the duplicate-inode cross-check can flag it
    imp = next((root / "imports").glob("pr-*.json"))
    bundle = next((root / "snapshots").glob("*.bundle"))
    imp_bytes = imp.read_bytes()
    bundle.unlink()
    os.link(imp, bundle)   # duplicate-inode surprise
    case_raw = yaml.safe_load(next((root / "cases").glob("*.yaml")).read_text())
    case_raw["snapshot"]["bundle_sha256"] = hashlib.sha256(imp_bytes).hexdigest()
    next((root / "cases").glob("*.yaml")).write_text(yaml.safe_dump(case_raw, sort_keys=False))
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()   # gated on Task 0 spike 4 verdict


def test_status_surfaces_failed_refresh_with_good_linkage(tmp_path, fake_gh):
    """A PR whose latest refresh failed but whose last import is intact: the
    status surface reports the attempt failure distinctly and does NOT classify
    the workspace as collecting.  Task 6 (issue #813)."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.workspace import workspace_status
    from tests.test_benchmark_import_prs import _curate_case, _seed_preflight

    ws = tmp_path / "ws"
    _seed_preflight(ws, fake_gh)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], origin_url=None) == 0
    _curate_case(ws, "pr-000101-aaaaaaaaaaaa.yaml")
    # now make the REFRESH fetch fail (PR already fetched: last import is intact)
    fake_gh.set_response(
        "GET", "repos/o/r/pulls/101", {"__error__": "API rate limit exceeded Retry-After: 1"}
    )
    rc = gi.run_import_prs(ws, pr_numbers=[101], heads=["final"], refresh=True, origin_url=None)
    assert rc != 0
    st = workspace_status(ws)
    pr = st.ledger.pull_requests[0]
    assert pr.latest_error is not None and pr.latest_error["code"] == "rate_limit"
    assert pr.import_state == "fetched"                # intact linkage, not collecting
    assert st.workspace_state != "collecting"          # evidence is not missing
