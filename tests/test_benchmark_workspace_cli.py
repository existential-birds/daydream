import hashlib
import subprocess


def _write_curated_workspace_with_sensitive_evidence(tmp_path):
    """The Task-2 curated fixture + evidence/finding bodies carrying a secret.

    Plants ``SUPER_SECRET_EVIDENCE`` in the import's ``evidence[].body`` and in
    the case doc's finding body, then corrupts the workspace (import file
    rewritten so the ledger ``import_sha256`` no longer matches) so ``validate``
    reports a failure whose diagnostics must never disclose the sentinel.
    """
    import json

    from test_benchmark_workspace import _write_curated_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    imp = next((root / "imports").glob("pr-*.json"))
    doc = json.loads(imp.read_text())
    doc["evidence"] = [
        {
            "source_id": "github:inline_comment:4242",
            "kind": "inline_comment",
            "database_id": 4242,
            "node_id": "DIFF_4242",
            "author": {"login": "alice", "type": "User"},
            "body": "SUPER_SECRET_EVIDENCE",
            "body_sha256": hashlib.sha256(b"SUPER_SECRET_EVIDENCE").hexdigest(),
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "submitted_at": None,
            "commit_id": None,
            "original_commit_id": None,
            "path": "feature.py",
            "line": 2,
            "original_line": 2,
            "review_id": None,
            "thread_id": None,
            "reply_to_id": None,
            "subject_type": "line",
            "side": "RIGHT",
            "start_side": None,
            "resolved": False,
            "outdated": False,
            "dismissed": False,
            "state": None,
            "is_bot": False,
            "url": "https://github.com/o/r/pull/101#discussion_r4242",
        }
    ]
    imp.write_bytes(json.dumps(doc).encode())   # ledger sha no longer matches -> corrupt
    case = next((root / "cases").glob("*.yaml"))
    text = case.read_text()
    assert "The cache key is stable across writes" in text
    case.write_text(
        text.replace(
            "The cache key is stable across writes, so stale data is served.",
            "SUPER_SECRET_EVIDENCE",
        )
    )
    return root


def _tree_sha(root) -> str:
    """Deterministic sha256 over a workspace tree's file bytes (read-only check)."""
    import hashlib as _h

    digest = _h.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            digest.update(p.read_bytes())
    return digest.hexdigest()


def test_benchmark_help_lists_subcommands():
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        ["daydream", "benchmark", "--help"], capture_output=True, text=True  # noqa: S607 - trusted command
    )
    assert r.returncode == 0 and "init" in r.stdout and "status" in r.stdout and "validate" in r.stdout


def test_benchmark_init_status_validate_roundtrip(tmp_path):
    ws = tmp_path / "ws"
    r = subprocess.run(  # noqa: S603
        [
            "daydream", "benchmark", "init", str(ws),
            "--repo", "OWNER/REPO",
            "--reviewer-host", "api.anthropic.com",
            "--judge-host", "api.anthropic.com",
        ],
        capture_output=True, text=True,  # noqa: S607
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "confidential" in r.stdout  # prints privacy classification
    assert "api.anthropic.com" in r.stdout  # prints egress boundary
    assert (ws / "benchmark.yaml").exists()

    r2 = subprocess.run(  # noqa: S603
        ["daydream", "benchmark", "status", str(ws)],  # noqa: S607
        capture_output=True, text=True,
    )
    assert r2.returncode == 0 and "empty" in r2.stdout and "unresolved" in r2.stdout

    r3 = subprocess.run(  # noqa: S603
        ["daydream", "benchmark", "validate", str(ws)],  # noqa: S607
        capture_output=True, text=True,
    )
    assert r3.returncode == 2  # fresh workspace: structurally valid but incomplete


def test_legacy_bench_still_works_alongside_benchmark():
    # The old `bench` verb must remain registered and dispatch to its own help,
    # proving coexistence (issue 15 owns removal).
    r = subprocess.run(  # noqa: S603
        ["daydream", "bench", "--help"], capture_output=True, text=True  # noqa: S607
    )
    assert r.returncode == 0 and "--benchmark-repo" in r.stdout


def test_validate_diagnostics_never_disclose_evidence_bodies(tmp_path, capsys):
    from daydream.benchmark.cli import _handle_benchmark_command

    ws = _write_curated_workspace_with_sensitive_evidence(tmp_path)
    before = _tree_sha(ws)
    rc = _handle_benchmark_command(["validate", str(ws)])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "SUPER_SECRET_EVIDENCE" not in out       # evidence bodies never printed
    assert "corrupt" in out.lower() or "ready" in out.lower()   # only labels/short strings
    assert rc == 1
    assert _tree_sha(ws) == before                  # validate is read-only: nothing written


def test_validate_exit_code_contract_preserved(tmp_path):
    from test_benchmark_workspace import _write_curated_workspace, _write_minimal_invalid_workspace

    from daydream.benchmark.workspace import validate_workspace

    assert validate_workspace(_write_curated_workspace(tmp_path / "r1", "ready"))[0] == 0
    assert validate_workspace(_write_curated_workspace(tmp_path / "r2", "draft"))[0] == 2
    assert validate_workspace(_write_minimal_invalid_workspace(tmp_path / "r3"))[0] == 1
    assert validate_workspace(_write_curated_workspace(tmp_path / "r4", "ready", resolved=False))[0] == 2


def test_private_benchmark_docs_document_the_new_verb():
    # The new verb must be documented so users can discover init/status/validate.
    from pathlib import Path

    text = Path("docs/benchmark.md").read_text(encoding="utf-8")
    assert "daydream benchmark init" in text
    assert "--reviewer-host" in text and "--judge-host" in text
    assert "daydream benchmark validate" in text
    assert "exit" in text.lower()  # 0/2/1 codes surfaced
