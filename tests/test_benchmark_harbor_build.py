"""Deterministic leak-resistant content compiler (issue #778)."""
from pathlib import Path
import hashlib

REPO = Path(__file__).resolve().parents[1]


def _seed_bare_bundle(tmp_path: Path) -> tuple[Path, bytes]:
    """Build a real base/head repo + bare mirror + build_bundle; return (mirror, bundle_bytes)."""
    import os
    import subprocess
    from daydream.benchmark import snapshot
    src = tmp_path / "src"; src.mkdir()
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
           "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}

    def g(*a: str) -> str:
        return subprocess.run(["git", "-C", str(src), *a], check=True,
                              env=env, capture_output=True).stdout.decode().strip()

    g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    (src / "f.py").write_text("x=1\n"); g("add", "."); g("commit", "-qm", "base"); base = g("rev-parse", "HEAD")
    (src / "f.py").write_text("x=2\n"); g("add", "."); g("commit", "-qm", "head"); head = g("rev-parse", "HEAD")
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
    ctx = build.bounded_pr_context({"title": "T", "body": body}, max_bytes=200)
    assert ctx.endswith("</historical_pr_context>")
    assert "[truncated; full_body_sha256=" in ctx
    # the truncated body must end on a whole UTF-8 char (no replacement chars / no split bytes)
    inner = ctx.split("<historical_pr_context>", 1)[1].split("</historical_pr_context>", 1)[0]
    body_line = inner.splitlines()[-1].removeprefix("body: ")
    body_line.encode("utf-8")                            # decodes cleanly: boundary is valid
    assert "[truncated; full_body_sha256=" in body_line
    digest = body_line.split("full_body_sha256=", 1)[1].rstrip("]")
    assert digest == hashlib.sha256(full.encode("utf-8")).hexdigest()   # full-text digest
    assert len(body_line.encode("utf-8")) <= 200


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
        build.build_gold_list([{"finding_id": "a" * 64, "title": "T", "body": "B",
                                "severity": None, "location": None, "provenance": {"kind": "authored", "source_ids": []}}])
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