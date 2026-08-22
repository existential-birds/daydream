"""Deterministic leak-resistant content compiler (issue #778)."""
import hashlib
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# deterministic seed identity + fake_gh fixtures (mirrors
# tests/test_benchmark_curation.py::_seed_ready_case and tests/conftest.py)

_SEED_ENV = {
    "GIT_AUTHOR_NAME": "Tester",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
    "GIT_COMMITTER_NAME": "Tester",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
}


def _pr_header(number: int = 101, *, base_sha: str = "b" * 40, head_sha: str = "a" * 40) -> dict:
    """A canned GitHub PR-header response for *number*."""
    return {
        "number": number,
        "url": f"https://github.com/o/r/pull/{number}",
        "title": "Fix cache",
        "state": "open",
        "base": {"ref": "main", "sha": base_sha},
        "head": {"ref": "feature/cache", "sha": head_sha},
        "merged_at": None,
        "closed_at": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "user": {"login": "alice", "type": "User"},
    }


def _seed_git(repo, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        env={**os.environ, **_SEED_ENV}, check=check,
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _seed_write(repo, name: str, content: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _seed_git(repo, "add", name)


def _seed_commit(repo, message: str) -> str:
    _seed_git(repo, "commit", "-m", message)
    return _seed_git(repo, "rev-parse", "HEAD")


def _seed_preflight(fake_gh, *, number: int = 101) -> None:
    """Seed canned identity + preflight/REST responses for one PR."""
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response(
        "repo-view-full",
        value={"id": 5, "nameWithOwner": "o/r",
               "url": "https://github.com/o/r", "visibility": "PRIVATE",
               "defaultBranchRef": {"name": "main"}},
    )
    fake_gh.set_response("GET", f"repos/o/r/pulls/{number}/reviews", [])
    fake_gh.set_response("GET", f"repos/o/r/pulls/{number}/comments", [])
    fake_gh.set_response("GET", f"repos/o/r/issues/{number}/comments", [])


def _seed_local_origin(tmp_path: Path, fake_gh, *, number: int = 101, lines: int = 3) -> tuple[str, str, str]:
    """Build a real local bare origin whose base/head are the PR's SHAs.

    The feature head adds ``feature.py`` with exactly *lines* lines. Returns
    ``(origin_url, base_sha, head_sha)``.
    """
    import shutil as _sh

    repo = tmp_path / f"local_wt_{number}"
    if repo.exists():
        _sh.rmtree(repo)
    repo.mkdir()
    _seed_git(repo, "init", "-b", "main")
    _seed_write(repo, "readme.txt", "base1\n")
    _seed_commit(repo, "base1")
    _seed_write(repo, "base.py", "BASE = 2\n")
    base_sha = _seed_commit(repo, "base2")
    _seed_write(repo, "beyond.py", "BEYOND = 3\n")
    _seed_commit(repo, "base3")
    _seed_git(repo, "checkout", "--detach", base_sha)
    (repo / "base.py").write_text("BASE = 20\n")
    _seed_git(repo, "add", "base.py")
    _seed_write(repo, "feature.py", "".join(f"LINE {i}\n" for i in range(1, lines + 1)))
    head_sha = _seed_commit(repo, f"feature{number}")
    bare = tmp_path / f"origin_{number}.git"
    if bare.exists():
        _sh.rmtree(bare)
    bare.mkdir()
    _seed_git(bare, "init", "--bare")
    _seed_git(repo, "remote", "add", "origin", str(bare))
    _seed_git(repo, "push", "origin", "main:main")
    _seed_git(repo, "push", "origin", f"{head_sha}:refs/pull/{number}/head", check=False)
    fake_gh.set_response("GET", "user", {"login": "octocat", "type": "User"})
    fake_gh.set_response(
        "repo-view-full",
        value={"id": 5, "nameWithOwner": "o/r",
               "url": "https://github.com/o/r", "visibility": "PRIVATE",
               "defaultBranchRef": {"name": "main"}},
    )
    header = _pr_header(number, base_sha=base_sha, head_sha=head_sha)
    fake_gh.set_response("GET", f"repos/o/r/pulls/{number}", header)
    return str(bare), base_sha, head_sha


def _seed_candidate(fake_gh, *, number: int = 101, head_sha: str) -> None:
    """Seed one REST inline comment so the case has one exact-acceptable candidate."""
    comment = {
        "id": number,
        "node_id": f"DIFF_{number}",
        "user": {"login": "alice", "type": "User"},
        "body": "please fix",
        "commit_id": head_sha,
        "original_commit_id": head_sha,
        "path": "feature.py",
        "line": 2,
        "subject_type": "line",
        "side": "RIGHT",
        "in_reply_to_id": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "html_url": f"https://github.com/o/r/pull/{number}#discussion_r{number}",
    }
    fake_gh.set_response("GET", f"repos/o/r/pulls/{number}/comments", [comment])


_SEED_SEQ = {"n": 0}


def _seed_ready_workspace(tmp_path: Path, fake_gh, *, lines: int = 3) -> tuple[Path, str, str]:
    """Seed a genuine frozen ``ready`` workspace for one imported PR.

    Builds a real bare origin, runs the real import (freezing a ready snapshot
    + bundle), accepts the first exact-acceptable candidate, and final-attests
    the case ready. Returns ``(ws, case_id, head_sha)``.
    """
    from daydream.benchmark import curation as cu
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict
    from daydream.benchmark.workspace import init_workspace

    _SEED_SEQ["n"] += 1
    ws = tmp_path / f"ws-{_SEED_SEQ['n']}"
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])
    _seed_preflight(fake_gh, number=101)
    origin_url, base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh, number=101, lines=lines)
    _seed_candidate(fake_gh, number=101, head_sha=head_sha)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    case_id = raw["cases"][0]["case_id"]
    candidate = next(
        c for c in cu.get_case(ws, case_id)["candidates"]
        if c["exact_acceptable"]
    )
    cu.accept_candidate(ws, case_id, candidate["source_id"])
    cu.mark_ready(ws, case_id, head_sha=head_sha)
    return ws, case_id, head_sha


def _seed_clean_workspace(tmp_path: Path, fake_gh) -> tuple[Path, str, str]:
    """Seed a reviewed-clean workspace: import with no comments, then attest clean."""
    from daydream.benchmark import curation as cu
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict
    from daydream.benchmark.workspace import init_workspace

    _SEED_SEQ["n"] += 1
    ws = tmp_path / f"ws-{_SEED_SEQ['n']}"
    init_workspace(ws, "o/r", ["h1.example.com"], ["h2.example.com"])
    _seed_preflight(fake_gh, number=101)
    origin_url, base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh, number=101, lines=3)
    assert gi.run_import_prs(ws, pr_numbers=[101], heads=[], origin_url=origin_url) == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    case_id = raw["cases"][0]["case_id"]
    cu.attest_clean(ws, case_id)
    return ws, case_id, head_sha


def _seed_second_ready_case(ws: Path, tmp_path: Path, fake_gh, *, lines: int = 3) -> str:
    """Import a second PR (102, a different head) into *ws* and mark it ready.

    Returns the second case id.
    """
    from daydream.benchmark import curation as cu
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.storage import load_yaml_strict

    _seed_preflight(fake_gh, number=102)
    origin_url, base_sha, head_sha = _seed_local_origin(tmp_path, fake_gh, number=102, lines=lines)
    _seed_candidate(fake_gh, number=102, head_sha=head_sha)
    assert gi.run_import_prs(ws, pr_numbers=[102], heads=[], origin_url=origin_url) == 0
    raw = load_yaml_strict(ws / "benchmark.yaml")
    case_id = next(c["case_id"] for c in raw["cases"] if c["pr_number"] == 102)
    candidate = next(
        c for c in cu.get_case(ws, case_id)["candidates"]
        if c["exact_acceptable"]
    )
    cu.accept_candidate(ws, case_id, candidate["source_id"])
    cu.mark_ready(ws, case_id, head_sha=head_sha)
    return case_id


def _inject_body(ws: Path, case_id: str, body: str) -> None:
    """Seed a body into the case document's pull_request block (schema-legal raw dict)."""
    from daydream.benchmark import storage
    path = ws / "cases" / f"{case_id}.yaml"
    raw = storage.load_yaml_strict(path)
    raw["pull_request"] = dict(raw["pull_request"])
    raw["pull_request"]["body"] = body
    storage.atomic_write_yaml(path, raw)


def _compile(ws: Path):
    from daydream.benchmark.harbor import build
    return build.compile_workspace(ws)


def _harbor_file_sha(ws: Path, rel: str) -> str:
    import hashlib as _hashlib
    data = (ws / "harbor" / rel).read_bytes()
    return _hashlib.sha256(data).hexdigest()


def _tree_bytes(ws: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for p in sorted(ws.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(ws))] = p.read_bytes()
    return out


def _harbor_tree_bytes(ws: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    base = ws / "harbor"
    for p in sorted(base.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(base))] = p.read_bytes()
    return out


def _seed_bare_bundle(tmp_path: Path) -> tuple[Path, bytes]:
    """Build a real base/head repo + bare mirror + build_bundle."""
    from daydream.benchmark import snapshot
    src = tmp_path / "src"
    src.mkdir()
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
           "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}

    def g(*a: str) -> str:
        return subprocess.run(["git", "-C", str(src), *a], check=True,
                              env=env, capture_output=True).stdout.decode().strip()

    g("init", "-q")
    g("config", "user.email", "t@t")
    g("config", "user.name", "t")
    (src / "f.py").write_text("x=1\n")
    g("add", ".")
    g("commit", "-qm", "base")
    base = g("rev-parse", "HEAD")
    (src / "f.py").write_text("x=2\n")
    g("add", ".")
    g("commit", "-qm", "head")
    head = g("rev-parse", "HEAD")
    m = snapshot.ensure_mirror(tmp_path, "o/r")
    # push the base/head commits (objects + refs) into the mirror so build_bundle can resolve trees
    subprocess.run(["git", "-C", str(src), "push", str(m),
                    f"{base}:refs/heads/base", f"{head}:refs/heads/head"],
                   check=True, env=env, capture_output=True)
    bundle = tmp_path / "b.bundle"
    snapshot.build_bundle(m, base, head, bundle)
    return m, bundle.read_bytes()


def test_spike_persisted_pull_request_field_set():
    """The import persists pull_request as a raw dict; the compiler must tolerate a missing body."""
    from daydream.benchmark.schema import ImportDocument
    assert ImportDocument.model_fields["pull_request"].annotation is dict
    # Field set is pinned by the constructor at github_import.py:1080-1090: it carries
    # number/url/title/state/base/head/created_at/updated_at/author and NO body.
    from daydream.benchmark import github_import as gi
    assert hasattr(gi, "fetch_and_normalize")  # module import guard (no network call here)


def test_spike_bundle_heads_is_exactly_base_head(tmp_path):
    from daydream.benchmark import snapshot
    m, bundle_bytes = _seed_bare_bundle(tmp_path)
    (tmp_path / "b.bundle").write_bytes(bundle_bytes)
    heads = snapshot.bundle_heads(tmp_path / "b.bundle")
    assert heads == {"refs/heads/base", "refs/heads/head"}


def test_derive_task_key_is_opaque_and_deterministic():
    from daydream.benchmark.harbor import build
    case_id = "pr-000101-1a2b3c4d5e6f"
    k = build.derive_task_key(case_id)
    assert k.startswith("case-") and len(k) == len("case-") + 12
    assert k == build.derive_task_key(case_id)          # deterministic
    assert k != build.derive_task_key("pr-000101-1a2b3c4d5e60")  # distinct case -> distinct key
    assert "pr-" not in k and case_id not in k          # reveals no authoring case id
    assert all(c in "0123456789abcdef" for c in k[len("case-"):])  # hex suffix


def test_bounded_pr_context_short_no_truncation():
    from daydream.benchmark.harbor import build
    ctx = build.bounded_pr_context({"title": "Fix cache", "body": "narrowly scoped"})
    assert ctx == (
        "<historical_pr_context>\ntitle: Fix cache\nbody: narrowly scoped\n"
        "</historical_pr_context>"
    )
    assert "[truncated" not in ctx


def test_bounded_pr_context_truncates_on_utf8_boundary_and_marks():
    from daydream.benchmark.harbor import build
    emoji = "😀"  # 4 UTF-8 bytes
    body = "a" * 1000 + emoji * 50 + "Z" * 500            # ends on a 4-byte char
    full = f"title: T\nbody: {body}"
    # fixed 15-byte prefix ("title: T\nbody: ") puts the first emoji at bytes
    # 1015..1018; max_bytes=1021 slices 2 bytes into the second emoji, so
    # _truncate_utf8 must back off byte-by-byte to 1018 (the whole first
    # emoji), exercising the UnicodeDecodeError path -- max_bytes=200 would
    # cut inside the ASCII a*1000 run and never reach the multibyte block.
    ctx = build.bounded_pr_context({"title": "T", "body": body}, max_bytes=1021)
    assert ctx.endswith("</historical_pr_context>")
    assert "[truncated; full_body_sha256=" in ctx
    # the truncated body must end on a whole UTF-8 char (no replacement chars / no split bytes)
    inner = ctx.split("<historical_pr_context>", 1)[1].split("</historical_pr_context>", 1)[0]
    body_line = next(
        line for line in inner.splitlines() if line.startswith("body: ")
    ).removeprefix("body: ")
    body_line.encode("utf-8")                            # decodes the whole: boundary is valid
    assert body_line.endswith(emoji)                      # kept the whole emoji, never split one
    assert len(body_line.encode("utf-8")) <= 1021
    marker = next(line for line in inner.splitlines() if line.startswith("[truncated"))
    digest = marker.split("full_body_sha256=", 1)[1].rstrip("]")
    assert digest == hashlib.sha256(full.encode("utf-8")).hexdigest()   # full-text digest


def test_bounded_pr_context_missing_body_is_empty():
    from daydream.benchmark.harbor import build
    ctx = build.bounded_pr_context({"title": "Fix cache"})          # no body key
    assert "body: \n" in ctx and "[truncated" not in ctx


def test_build_gold_list_is_provenance_free_and_location_required():
    from daydream.benchmark.harbor import build
    findings = [
        {"finding_id": "c" * 64, "title": "Cache", "body": "collides", "severity": "high",
         "location": {"path": "src/cache.py", "start_line": 42, "end_line": 42},
         "provenance": {"kind": "historical", "source_ids": ["github:review:1"]}},
        {"finding_id": "a" * 64, "title": "Escape", "body": "unvalidated", "severity": "medium",
         "location": {"path": "src/render.py", "start_line": 10, "end_line": 14},
         "provenance": {"kind": "authored", "source_ids": []}},
    ]
    gold = build.build_gold_list(findings)
    assert [f["finding_id"] for f in gold] == ["a" * 64, "c" * 64]        # ordered by finding_id
    assert all(set(f) == {"finding_id", "title", "body", "severity", "path", "start_line", "end_line"}
               for f in gold)                                             # no provenance/source/gold keys
    assert gold[0]["path"] == "src/render.py" and gold[0]["start_line"] == 10


def test_build_gold_list_clean_is_empty():
    from daydream.benchmark.harbor import build
    assert build.build_gold_list([]) == []


def test_build_gold_list_rejects_locationless_finding():
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor.build import CompileError
    try:
        build.build_gold_list([{
            "finding_id": "a" * 64, "title": "T", "body": "B",
            "severity": None, "location": None,
            "provenance": {"kind": "authored", "source_ids": []},
        }])
        assert False, "expected CompileError for a location-less finding"
    except CompileError as exc:
        assert "location" in str(exc)


def test_build_oracle_artifact_passes_validation_and_derives_candidate_ids():
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor import verifier_core as vc
    findings = [
        {"finding_id": "b" * 64, "title": "Cache", "body": "collides", "severity": "high",
         "location": {"path": "src/cache.py", "start_line": 42, "end_line": 42},
         "provenance": {"kind": "historical", "source_ids": ["github:review:1"]}},
        {"finding_id": "a" * 64, "title": "Escape", "body": "unvalidated", "severity": None,
         "location": {"path": "src/render.py", "start_line": 10, "end_line": 14},
         "provenance": {"kind": "authored", "source_ids": []}},
    ]
    key = build.derive_task_key("pr-000101-1a2b3c4d5e6f")
    art = build.build_oracle_artifact(key, findings)
    assert art["schema_version"] == 1 and art["case_id"] == key
    assert art["base_ref"] == "base" and art["head_ref"] == "head"
    # findings are ordered by finding_id ascending; ordinal = position in that order
    flat = [
        {"title": f["title"], "body": f["body"], "severity": f["severity"],
         "path": f["location"]["path"], "start_line": f["location"]["start_line"],
         "end_line": f["location"]["end_line"]}
        for f in sorted(findings, key=lambda f: f["finding_id"])
    ]
    expected_ids = []
    groups: dict[tuple, int] = {}
    for f in flat:
        canon = (f["title"], f["body"], f["severity"] or "", f["path"], f["start_line"], f["end_line"])
        ordinal = groups.get(canon, 0)
        groups[canon] = ordinal + 1
        expected_ids.append(vc.derive_candidate_id(key, f, ordinal))
    assert [f["candidate_id"] for f in art["findings"]] == expected_ids
    # round-trips through the verifier's own validation
    assert vc.validate_candidate_artifact(art)


def test_build_oracle_artifact_clean_has_empty_findings():
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor import verifier_core as vc
    key = build.derive_task_key("pr-000101-1a2b3c4d5e6f")
    art = build.build_oracle_artifact(key, [])
    assert art["findings"] == []
    assert vc.validate_candidate_artifact(art) == []


def test_copy_assets_places_templates_and_keeps_verifier_core_byte_identical(tmp_path):
    from daydream.benchmark.harbor import build
    dst = tmp_path / "case"
    build._copy_assets(dst)
    expected = {
        "tests/score_review.py", "tests/verifier_core.py", "tests/judge_prompt.md",
        "tests/test.sh", "tests/Dockerfile", "solution/solve.sh",
    }
    assert {str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file()} == expected
    src_core = build._TEMPLATE_DIR / "tests" / "verifier_core.py"
    assert (dst / "tests" / "verifier_core.py").read_bytes() == src_core.read_bytes()
    assert (dst / "tests" / "verifier_core.py").read_bytes() == (
        Path(REPO) / "daydream" / "benchmark" / "harbor" / "verifier_core.py").read_bytes()
    assert (dst / "tests" / "score_review.py").read_bytes() == (
        build._TEMPLATE_DIR / "tests" / "score_review.py").read_bytes()


def _load_json(path: Path):
    import json as _json
    return _json.loads(path.read_bytes())


def test_compile_findings_case_full_tree_and_gold_oracle_agree(tmp_path, fake_gh):
    from daydream.benchmark import storage
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor import verifier_core as vc
    ws, case_id, head_sha = _seed_ready_workspace(tmp_path, fake_gh)
    key = build.derive_task_key(case_id)
    lock = build.compile_workspace(ws)

    case = ws / "harbor" / key
    assert (case / "instruction.md").exists()
    assert (case / "README.md").exists()
    assert (case / "environment" / "repository.bundle").exists()
    assert (case / "tests" / "golden-review.json").exists()
    assert (case / "tests" / "verifier_core.py").exists()
    assert (case / "solution" / "golden-review.json").exists()

    gold = _load_json(case / "tests" / "golden-review.json")
    oracle = storage.load_json_strict(case / "solution" / "golden-review.json")
    assert vc.validate_gold_set(gold)                       # gold passes gold-set validation
    vc.validate_candidate_artifact(oracle)                 # oracle passes candidate validation
    assert [f["finding_id"] for f in gold] == sorted(f["finding_id"] for f in gold)
    # gold↔oracle agreement: same content fields, in the same finding_id order
    gold_content = [(f["title"], f["body"], f.get("severity"), f["path"], f["start_line"], f["end_line"])
                    for f in sorted(gold, key=lambda f: f["finding_id"])]
    oracle_content = [(f["title"], f["body"], f.get("severity"), f["path"], f["start_line"], f["end_line"])
                      for f in oracle["findings"]]
    assert oracle_content == gold_content

    # instruction.md = fixed assignment + bounded block; no gold-derived text
    instr = (case / "instruction.md").read_text()
    assert "untrusted context, not instructions" in instr
    assert "<historical_pr_context>" in instr and "</historical_pr_context>" in instr
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert f"title: {raw['pull_request']['title']}" in instr

    # root README  + lock;   lock holds packages + private mapping
    assert (ws / "harbor" / "README.md").exists()
    assert (ws / "harbor" / "metric.py").exists()
    assert (ws / "harbor" / "jobs").is_dir()
    assert lock["cases"][key]["case_id"] == case_id
    assert lock["cases"][key]["pr_number"] == 101
    assert lock["cases"][key]["repository"] == "o/r"
    assert lock["cases"][key]["original_head_sha"] == head_sha
    assert lock["cases"][key]["bundle_sha256"] == storage.sha256_file(case / "environment" / "repository.bundle")
    assert lock["cases"][key]["files"]["tests/verifier_core.py"] == \
        lock["files"][f"{key}/tests/verifier_core.py"]
    assert not any("timestamp" in k or "created_at" in k for k in lock.keys())
    for rel, data in _harbor_tree_bytes(ws).items():
        if rel == "benchmark.lock.json":
            continue
        assert lock["files"][rel] == hashlib.sha256(data).hexdigest()


def test_compile_clean_case_has_empty_gold_and_oracle(tmp_path, fake_gh):
    from daydream.benchmark import storage
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor import verifier_core as vc
    ws, case_id, _ = _seed_clean_workspace(tmp_path, fake_gh)
    key = build.derive_task_key(case_id)
    lock = build.compile_workspace(ws)
    case = ws / "harbor" / key
    assert _load_json(case / "tests" / "golden-review.json") == []
    assert storage.load_json_strict(case / "solution" / "golden-review.json")["findings"] == []
    assert vc.validate_gold_set(_load_json(case / "tests" / "golden-review.json")) == []
    assert lock["cases"][key]["gold_sha256"] == hashlib.sha256(b"[]").hexdigest()


def test_double_compile_is_byte_identical_and_lock_digest_stable(tmp_path, fake_gh):
    from daydream.benchmark.harbor import build
    ws, _, _ = _seed_ready_workspace(tmp_path, fake_gh)
    lock1 = build.compile_workspace(ws)
    tree1 = _harbor_tree_bytes(ws)
    lock2 = build.compile_workspace(ws)
    tree2 = _harbor_tree_bytes(ws)
    assert tree1 == tree2                                        # byte-identical compiled tree
    assert lock1 == lock2                                        # identical lock digest/content
    # no timestamps anywhere in any compiled file or lock
    lock_text = (ws / "harbor" / "benchmark.lock.json").read_text()
    assert "created_at" not in lock_text and "timestamp" not in lock_text


def test_compiled_case_dirs_are_canonically_sorted_by_opaque_key(tmp_path, fake_gh):
    from daydream.benchmark import storage
    from daydream.benchmark.harbor import build
    ws, _, _ = _seed_ready_workspace(tmp_path, fake_gh)
    _seed_second_ready_case(ws, tmp_path, fake_gh)
    manifest = storage.load_yaml_strict(ws / "benchmark.yaml")
    case_ids = [c["case_id"] for c in manifest["cases"]]
    assert len(case_ids) == 2

    lock_a = build.compile_workspace(ws)
    tree_a = _harbor_tree_bytes(ws)

    # reverse the manifest cases[] order - output must not change (canonical ordering)
    manifest["cases"] = manifest["cases"][::-1]
    storage.atomic_write_yaml(ws / "benchmark.yaml", manifest)
    build.compile_workspace(ws)
    tree_b = _harbor_tree_bytes(ws)
    assert tree_a == tree_b

    dirs = sorted(p.name for p in (ws / "harbor").iterdir() if p.is_dir() and p.name.startswith("case-"))
    assert dirs == sorted(build.derive_task_key(c) for c in case_ids)
    assert list(lock_a["cases"].keys()) == sorted(lock_a["cases"].keys())


def test_staging_failure_preserves_prior_tree(tmp_path, fake_gh, monkeypatch):
    from daydream.benchmark import storage
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor.build import CompileError
    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)
    build.compile_workspace(ws)                                # successful baseline
    before = _harbor_tree_bytes(ws)
    # force a mid-compile failure: drop the snapshot bundle so the case can no longer compile
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    (ws / raw["snapshot"]["bundle_file"]).unlink()
    try:
        build.compile_workspace(ws)
        assert False, "expected CompileError for a missing bundle"
    except CompileError:
        pass
    assert _harbor_tree_bytes(ws) == before                    # prior tree fully intact
    assert not (ws / "cache" / "harbor-build-stage").exists()  # no stage residue at the output


def test_leakage_scan_rejects_forbidden_tokens_and_names_file_and_token():
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor.build import CompileError
    cases = {
        "README.md": "A benchmark of historical code reviews.\n",
        "case-abcdef123456/instruction.md": "assignment\ntitle: Fix cache\nbody: ok\n</historical_pr_context>",
        "case-abcdef123456/README.md": (
            "This case references the gold_status and provenance of pr-000101.\n"
            "see https://github.com/o/r/pull/101 and sha "
            "1a2b3c4d5e6f7890abcdef1234567890abcdef12\n"
            "token=ghp_ABCDEFGHIJKLMNOPQRSTUVWX and https://user:pass@host/x\n"
            "source github:review:42"
        ),
    }
    try:
        build.leakage_scan(cases, repository_slug="o/r")
        assert False, "expected CompileError"
    except CompileError as exc:
        msg = str(exc)
        assert "case-abcdef123456/README.md" in msg           # names the file
        assert "pr-000101" in msg or "gold_status" in msg     # names a forbidden token


def test_leakage_scan_permits_bounded_block_raw_text():
    from daydream.benchmark.harbor import build
    instr = (
        "assignment text\n"
        "<historical_pr_context>\n"
        "title: Handle pull/999 regressions\n"
        "body: references o/r and sha 1a2b3c4d5e6f7890abcdef1234567890abcdef12\n"
        "</historical_pr_context>\n"
    )
    build.leakage_scan({"case-x/instruction.md": instr}, repository_slug="o/r")   # no raise


def test_leakage_scan_rejects_clean_readme():
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor.build import CompileError
    # clean marker leaks into a README
    try:
        build.leakage_scan({"README.md": "gold_status clean_attested snapshot_attested\n"},
                           repository_slug="o/r")
        assert False, "expected CompileError"
    except CompileError as exc:
        assert "clean_attested" in str(exc)


def test_validate_bundle_inventory_accepts_valid_base_head_bundle(tmp_path):
    from daydream.benchmark.harbor import build
    m, bundle_bytes = _seed_bare_bundle(tmp_path)
    bp = tmp_path / "b.bundle"
    bp.write_bytes(bundle_bytes)
    build.validate_bundle_inventory(bp)


def test_validate_bundle_inventory_rejects_extra_ref(tmp_path):
    import subprocess as _subprocess

    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor.build import CompileError
    m, _ = _seed_bare_bundle(tmp_path)
    bp = tmp_path / "bad.bundle"
    # add an extra ref to the mirror, then rebuild the bundle including it
    _subprocess.run(["git", "-C", str(m), "update-ref", "refs/heads/extra", "refs/heads/base"], check=True)
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
           "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
    _subprocess.run(["git", "-C", str(m), "bundle", "create", str(bp),
                     "refs/heads/base", "refs/heads/head", "refs/heads/extra"],
                    check=True, env=env, capture_output=True)
    try:
        build.validate_bundle_inventory(bp)
        assert False, "expected to fail for an extra ref"
    except CompileError as exc:
        assert "ref" in str(exc)


def test_compiled_tree_contains_no_raw_authoring_files(tmp_path, fake_gh):
    from daydream.benchmark.harbor import build
    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)
    build.compile_workspace(ws)
    rels = {str(p.relative_to(ws / "harbor")) for p in (ws / "harbor").rglob("*") if p.is_file()}
    forbidden_substrs = ("imports/", "cases/", "benchmark.yaml", "provenance", "exclusions")
    assert not any(any(f in r for f in forbidden_substrs) for r in rels)
    # every compiled path lives under a case dir, root control files, or the metric
    assert all(r.startswith("case-") or r in {"README.md", "benchmark.lock.json", "metric.py"} for r in rels)


def test_compile_rejects_when_a_case_is_not_compilable(tmp_path, fake_gh):
    from daydream.benchmark import storage
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor.build import CompileError
    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)   # mark_ready done
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    raw["curation"]["state"] = "stale"
    raw["curation"]["snapshot_attested"] = False
    storage.atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw)
    try:
        build.compile_workspace(ws)
        assert False, "expected CompileError for a non-ready case"
    except CompileError as exc:
        assert case_id in str(exc)
