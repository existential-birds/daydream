"""Golden corpus-shape parity tests.

Pin the fidelity contract between daydream's injected benchmark entries and the
schema produced by the upstream code-review-benchmark harvester. These are
assertion-only tests: a failure indicates an earlier task's production contract
(mapping.py or benchmark_data.py) is wrong, not the test.
"""

from daydream.benchmark.benchmark_data import inject_daydream_review
from daydream.benchmark.mapping import merged_items_to_review_comments

URL = "https://github.com/owner/repo/pull/1"

# A step1-harvested review_comment has these keys (code-review-benchmark
# step1_download_prs.py:134-138). Our injected comments carry every harvested
# key and add the structured ``confidence``/``severity`` fields (issue #231).
HARVEST_KEYS = {"path", "line", "body", "created_at"}
INJECTED_KEYS = HARVEST_KEYS | {"confidence", "severity"}


def test_injected_comment_keys_match_harvested_schema():
    doc = {"items": [{"file": "a.py", "line": 3, "description": "x", "severity": "high",
                      "confidence": "HIGH", "rationale": "y"}]}
    comments = merged_items_to_review_comments(doc, created_at="2026-06-03T00:00:00Z")
    assert comments and all(set(c) == INJECTED_KEYS for c in comments)
    assert all(HARVEST_KEYS <= set(c) for c in comments)   # superset of the harvested schema
    c = comments[0]
    assert isinstance(c["path"], str) and isinstance(c["body"], str) and isinstance(c["created_at"], str)
    assert c["line"] is None or isinstance(c["line"], int)


def test_injected_review_entry_matches_harvested_review_shape():
    data = {URL: {"reviews": []}}
    inject_daydream_review(data, URL, [{"path": "a.py", "line": 1, "body": "b", "created_at": "T"}],
                           force=False, tool="daydream-glm")
    entry = data[URL]["reviews"][0]
    assert set(entry) == {"tool", "repo_name", "pr_url", "review_comments"}   # step1 review-entry keys


def test_corpus_shape_is_backend_independent():
    items = {
        "items": [
            {"file": "a.py", "line": 1, "description": "d", "severity": "high", "confidence": "HIGH", "rationale": "r"}
        ]
    }
    cmts = merged_items_to_review_comments(items, created_at="T")
    a = {URL: {"reviews": []}}
    b = {URL: {"reviews": []}}
    inject_daydream_review(a, URL, cmts, force=False, tool="daydream-claude")
    inject_daydream_review(b, URL, cmts, force=False, tool="daydream-glm")
    ra, rb = a[URL]["reviews"][0], b[URL]["reviews"][0]
    assert {k: ra[k] for k in ("pr_url","review_comments")} == {k: rb[k] for k in ("pr_url","review_comments")}
    assert set(ra) == set(rb)                            # identical structure across backends
