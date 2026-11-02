"""Deterministic leak-resistant content compiler (issue #778)."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from daydream.benchmark import curation as cu, github_import as gi, storage
from daydream.benchmark.cli import _handle_benchmark_command
from daydream.benchmark.harbor import build
from daydream.benchmark.harbor.build import CompileError
from daydream.benchmark.storage import load_yaml_strict
from daydream.benchmark.workspace import init_workspace
from daydream.pr_review import FINDING_MARKER_RE, finding_marker
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import seed_pr_origin

REPO = Path(__file__).resolve().parents[1]

def _pr_header(number: int = 101, *, base_sha: str = "b" * 40, head_sha: str = "a" * 40) -> dict[str, Any]:
    """A canned GitHub PR-header response for *number*."""
    return {"number": number, "url": f"https://github.com/o/r/pull/{number}", "title": "Fix cache", "state": "open",
        "base": {"ref": "main", "sha": base_sha}, "head": {"ref": "feature/cache", "sha": head_sha}, "merged_at": None,
        "closed_at": None, "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
        "user": {"login": "alice", "type": "User"},
    }


def _seed_preflight(fake_gh: FakeGh, *, number: int = 101) -> None:
    """Seed canned identity + preflight/REST responses for one PR."""
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response("repo-view-full", value={"id": "R_kgDOABC123", "nameWithOwner": "o/r",
               "url": "https://github.com/o/r", "visibility": "PRIVATE", "defaultBranchRef": {"name": "main"}},
    )
    fake_gh.set_response("GET", f"repos/o/r/pulls/{number}/reviews", [])
    fake_gh.set_response("GET", f"repos/o/r/pulls/{number}/comments", [])
    fake_gh.set_response("GET", f"repos/o/r/issues/{number}/comments", [])


def _seed_local_origin(tmp_path: Path, fake_gh: FakeGh, *, number: int = 101, lines: int = 3) -> tuple[str, str, str]:
    """Build a real local bare origin whose base/head are the PR's SHAs.

    The feature head adds ``feature.py`` with exactly *lines* lines. Returns
    ``(origin_url, base_sha, head_sha)``. Callers seed identity (
    ``_seed_preflight``) first; this only adds the canned PR header.
    """
    origin_url, base_sha, head_sha = seed_pr_origin(
        tmp_path, repo_name=f"local_wt_{number}", bare_name=f"origin_{number}.git",
        feature_body="".join(f"LINE {i}\n" for i in range(1, lines + 1)),
        feature_message=f"feature{number}", number=number,
    )
    header = _pr_header(number, base_sha=base_sha, head_sha=head_sha)
    fake_gh.set_response("GET", f"repos/o/r/pulls/{number}", header)
    return origin_url, base_sha, head_sha


def _seed_candidate(fake_gh: FakeGh, *, number: int = 101, head_sha: str, body: str = "please fix",) -> None:
    """Seed one REST inline comment so the case has one exact-acceptable candidate."""
    comment = {"id": number, "node_id": f"DIFF_{number}", "user": {"login": "alice", "type": "User"}, "body": body,
        "commit_id": head_sha, "original_commit_id": head_sha, "path": "feature.py", "line": 2, "original_line": 2,
        "subject_type": "line", "side": "RIGHT", "in_reply_to_id": None, "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z", "html_url": f"https://github.com/o/r/pull/{number}#discussion_r{number}",
    }
    fake_gh.set_response("GET", f"repos/o/r/pulls/{number}/comments", [comment])

_SEED_SEQ = {"n": 0}


def _mark_ready(ws: Path, case_id: str, head_sha: str) -> None:
    """Mark *case_id* ready with the freshly-rendered task-spec digest."""

    task_spec_sha256 = hashlib.sha256(
        build.render_task_spec(load_yaml_strict(ws / "cases" / f"{case_id}.yaml"), instruction=build.ASSIGNMENT_TEXT)
    ).hexdigest()
    cu.mark_ready(ws, case_id, head_sha=head_sha, task_spec_sha256=task_spec_sha256)

PK_BODY = b"PK\x05\x06" + b"\x00" * 18


def _stub_wheel(directory: Path, content: bytes = PK_BODY) -> tuple[Path, str]:
    """Write a minimal wheel for the installed ``daydream`` version into *directory*.

    Returns ``(wheel_path, version)`` so callers asserting on the version do not
    re-read package metadata.
    """
    version = importlib.metadata.version("daydream")
    wheel = directory / f"daydream-{version}-py3-none-any.whl"
    wheel.write_bytes(content)
    return wheel, version


def _import_case(tmp_path: Path, fake_gh: FakeGh, *, number: int, lines: int = 3,
    ws: Path | None = None, with_candidate: bool = True,
) -> tuple[Path, str, str]:
    """Import one PR into a fresh (or given) workspace; returns (ws, case_id, head_sha)."""

    if ws is None:
        _SEED_SEQ["n"] += 1
        ws = tmp_path / f"ws-{_SEED_SEQ['n']}"
        init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])
    _seed_preflight(fake_gh, number=number)
    origin_url, _, head_sha = _seed_local_origin(tmp_path, fake_gh, number=number, lines=lines)
    if with_candidate:
        _seed_candidate(fake_gh, number=number, head_sha=head_sha)
    assert gi.run_import_prs(ws, pr_numbers=[number], heads=[], origin_url=origin_url) == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    case_id: str = next(c["case_id"] for c in raw["cases"] if c["pr_number"] == number)
    return ws, case_id, head_sha


def _seed_ready_workspace(tmp_path: Path, fake_gh: FakeGh, *, lines: int = 3) -> tuple[Path, str, str]:
    """Seed a genuine frozen ``ready`` workspace for one imported PR.

    Returns ``(ws, case_id, head_sha)``.
    """

    ws, case_id, head_sha = _import_case(tmp_path, fake_gh, number=101, lines=lines)
    candidate = next(c for c in cu.get_case(ws, case_id)["candidates"] if c["exact_acceptable"])
    cu.accept_candidate(ws, case_id, candidate["source_id"])
    _mark_ready(ws, case_id, head_sha)
    return ws, case_id, head_sha


def _truncation_marker_digest(text: str) -> str:
    """The ``full_body_sha256`` value attested by the truncation marker."""
    inner = text.split("<historical_pr_context>", 1)[1].split("</historical_pr_context>", 1)[0]
    marker = next(line for line in inner.splitlines() if line.startswith("[truncated"))
    return marker.split("full_body_sha256=", 1)[1].rstrip("]")


def _harbor_tree_bytes(ws: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    base = ws / "harbor"
    for p in sorted(base.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(base))] = p.read_bytes()
    return out


def test_bounded_pr_context_truncates_on_utf8_boundary_and_marks() -> None:
    emoji = "😀"  # 4 UTF-8 bytes
    body = "a" * 1000 + emoji * 50 + "Z" * 500            # ends on a 4-byte char
    # With no persisted body_sha256 key the marker falls back to the digest of the
    # stored normalized body (never the escaped title: prefix).
    # fixed 15-byte prefix ("title: T\nbody: ") puts the first emoji at bytes
    # 1015..1018; max_bytes=1021 slices 2 bytes into the second emoji, so
    # the truncator must back off byte-by-byte to 1018 (the whole first
    # emoji), exercising the UnicodeDecodeError path -- max_bytes=200 would
    # cut inside the ASCII a*1000 run and never reach the multibyte block.
    ctx = build.bounded_pr_context({"title": "T", "body": body}, max_bytes=1021)
    assert ctx.endswith("</historical_pr_context>")
    assert "[truncated; full_body_sha256=" in ctx
    inner = ctx.split("<historical_pr_context>", 1)[1].split("</historical_pr_context>", 1)[0]
    body_line = next(line for line in inner.splitlines() if line.startswith("body: ")).removeprefix("body: ")
    body_line.encode("utf-8")                            # decodes the whole: boundary is valid
    assert body_line.endswith(emoji)                      # kept the whole emoji, never split one
    assert len(body_line.encode("utf-8")) <= 1021
    digest = _truncation_marker_digest(ctx)
    assert digest == hashlib.sha256(body.encode("utf-8")).hexdigest()   # stored normalized-body digest

def test_build_gold_and_oracle_reject_partial_location_and_over_cap() -> None:
    key = build.derive_task_key("pr-000101-1a2b3c4d5e6f")
    with pytest.raises(CompileError):
        build.build_gold_list([{"finding_id": "a" * 64, "title": "T", "body": "B", "severity": None,
            "location": {"path": "src/a.py", "start_line": None, "end_line": None},
            "provenance": {"kind": "authored", "source_ids": []},
        }], key=key)
    for oracle, count in ((False, 51), (True, 101)):
        findings = [{"finding_id": f"{i:064x}", "title": f"T{i}", "body": "B", "severity": "low",
            "location": {"path": "src/a.py", "start_line": 1, "end_line": 1},
            "provenance": {"kind": "authored", "source_ids": []},
        } for i in range(count)]
        with pytest.raises(build.CompileError):
            if oracle:
                build.build_oracle_artifact(key, findings)
            else:
                build.build_gold_list(findings, key=key)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def test_finding_marker_import_curate_compile_preserves_raw_source(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = finding_marker("f" * 64)
    raw_body = f"\n{marker}\n## Cache race\nProtect the shared cache.\n{marker}\n"
    ws = tmp_path / "ws-marker-flow"
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])
    _seed_preflight(fake_gh)
    origin_url, _base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh)
    _seed_candidate(fake_gh, head_sha=head_sha, body=raw_body)
    config_index = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    monkeypatch.setenv(f"GIT_CONFIG_KEY_{config_index}", f"url.{origin_url}.insteadOf")
    monkeypatch.setenv(f"GIT_CONFIG_VALUE_{config_index}", "https://github.com/o/r.git")
    monkeypatch.setenv("GIT_CONFIG_COUNT", str(config_index + 1))

    assert _handle_benchmark_command(["import-prs", str(ws), "--pr", "101"]) == 0
    manifest = storage.load_yaml_strict(ws / "benchmark.yaml")
    ledger_entry = manifest["pull_requests"][0]
    case_id = ledger_entry["case_ids"][0]
    case = cu.get_case(ws, case_id)
    import_path = ws / ledger_entry["import_file"]
    import_doc = storage.load_json_strict(import_path)
    evidence = import_doc["evidence"][0]
    assert evidence["body"] == raw_body
    assert evidence["body_sha256"] == hashlib.sha256(raw_body.encode()).hexdigest()
    payload = {key: import_doc[key] for key in ("schema_version", "repository", "pull_request", "evidence")}
    assert import_doc["fetch"]["payload_sha256"] == gi._payload_sha256(payload)
    assert ledger_entry["import_sha256"] == storage.sha256_file(import_path)
    candidate = case["candidates"][0]
    assert candidate["source_id"] == evidence["source_id"]
    assert candidate["title"] == "Cache race"
    assert candidate["exact_acceptable"] is True
    assert candidate["location"] == {"path": "feature.py", "start_line": 2, "end_line": 2}
    assert not FINDING_MARKER_RE.search(candidate["body"])

    edited_body = candidate["body"] + "\nCurator clarification."
    fragment = tmp_path / "gold.yaml"
    fragment.write_text(yaml.safe_dump({"findings": [{
            "title": candidate["title"], "body": edited_body, "severity": None,
            "location": candidate["location"], "source_ids": [candidate["source_id"]],
        }], "exclusions": [], "case_exclusion": None, "clean": False,
    }, sort_keys=False))
    assert _handle_benchmark_command(["curate", str(ws), "--case", case_id, "--apply-gold", str(fragment)]) == 0
    curated = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    finding = curated["curation"]["findings"][0]
    assert finding["provenance"]["kind"] == "edited"
    assert finding["provenance"]["source_ids"] == [evidence["source_id"]]
    assert finding["body"] == edited_body
    _mark_ready(ws, case_id, head_sha)
    wheel, _ = _stub_wheel(tmp_path)
    assert _handle_benchmark_command(["build-harbor", str(ws), "--daydream-wheel", str(wheel)]) == 0

    compiled = ws / "harbor" / build.derive_task_key(case_id)
    for relative in ("tests/golden-review.json", "solution/golden-review.json"):
        artifact_path = compiled / relative
        serialized = artifact_path.read_text()
        assert "daydream-finding" not in serialized
        assert not FINDING_MARKER_RE.search(serialized)
        artifact = _load_json(artifact_path)
        emitted = artifact if isinstance(artifact, list) else artifact["findings"]
        assert len(emitted) == 1
        assert emitted[0]["title"] == "Cache race"
        assert emitted[0]["body"] == edited_body

def test_leakage_scan_covers_task_toml_and_job_configs() -> None:
    cases = {"case-abcdef123456/task.toml": (
            'schema_version = "1.4"\n# leak: ghp_abcdefghijklmnopqrstuv\n'
        ),
        "harbor-job.yaml": "jobs_dir: jobs\n# leak: https://user:pass@github.com/o/r\n",
    }
    with pytest.raises(build.CompileError) as rejected:
        build.leakage_scan(cases, repository_slug="o/r")
    assert "case-abcdef123456/task.toml" in str(rejected.value)
    assert "harbor-job.yaml" in str(rejected.value)

def test_compile_records_template_version_and_rejects_stale_task_spec(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)
    lock = build.compile_workspace(ws)
    key = build.derive_task_key(case_id)
    metadata = json.loads((ws / "harbor" / key / "tests" / "verifier-metadata.json").read_bytes())
    assert lock["template_version"] == build.TEMPLATE_VERSION
    assert metadata["template_version"] == lock["template_version"]

    before = _harbor_tree_bytes(ws)
    case_path = ws / "cases" / f"{case_id}.yaml"
    raw = storage.load_yaml_strict(case_path)
    raw["pull_request"] = dict(raw["pull_request"])
    raw["pull_request"]["title"] = "Changed after approval"
    raw["pull_request"]["title_sha256"] = hashlib.sha256(b"Changed after approval").hexdigest()
    # Retain the previously approved task-spec digest; the changed rendered
    # spec must not replace the compiled tree without fresh approval.
    storage.atomic_write_yaml(case_path, raw)

    with pytest.raises(build.CompileError, match="task spec digest"):
        build.compile_workspace(ws)
    assert _harbor_tree_bytes(ws) == before

    raw["curation"].update(state="draft", snapshot_attested=False)
    storage.atomic_write_yaml(case_path, raw)
    _mark_ready(ws, case_id, raw["snapshot"]["original_head_sha"])
    renewed = build.compile_workspace(ws)
    assert renewed["authoring_input_digest"] != lock["authoring_input_digest"]
    assert renewed["cases"][key]["task_spec_sha256"] != lock["cases"][key]["task_spec_sha256"]
    assert (ws / "harbor" / key / "Task.md").read_bytes() != before[f"{key}/Task.md"]

    # A corrupt manifest or a missing frozen bundle fails the compile closed.
    manifest = ws / "benchmark.yaml"
    manifest_bytes = manifest.read_bytes()
    manifest.write_text("schema_version: [\n")
    with pytest.raises(CompileError):
        build.compile_workspace(ws)
    manifest.write_bytes(manifest_bytes)
    raw = storage.load_yaml_strict(case_path)
    bundle = storage.resolve_authoring_path(ws, raw["snapshot"]["bundle_file"])
    bundle.rename(bundle.with_suffix(".moved"))
    with pytest.raises(CompileError, match="missing bundle"):
        build.compile_workspace(ws)
    bundle.with_suffix(".moved").rename(bundle)

    raw["curation"].update(state="stale", snapshot_attested=False)
    storage.atomic_write_yaml(case_path, raw)
    with pytest.raises(CompileError) as rejected:
        build.compile_workspace(ws)
    assert case_id in str(rejected.value)
