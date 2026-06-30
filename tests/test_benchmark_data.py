import pytest

from daydream.benchmark.benchmark_data import has_daydream_review, inject_daydream_review

URL = "https://github.com/grafana/grafana/pull/90939"
CMTS = [{"path": "f.go", "line": 1, "body": "finding", "created_at": "2026-06-03T00:00:00Z"}]


def _data():
    return {
        URL: {
            "golden_comments": [{"comment": "g", "severity": "Medium"}],
            "reviews": [{"tool": "coderabbit", "repo_name": "r", "pr_url": "u", "review_comments": []}],
        }
    }


def test_inject_appends_one_daydream_review_without_touching_others():
    d = _data()
    assert inject_daydream_review(d, URL, CMTS, force=False) is True
    reviews = d[URL]["reviews"]
    assert [r["tool"] for r in reviews] == ["coderabbit", "daydream"]
    assert reviews[1]["review_comments"] == CMTS
    assert d[URL]["golden_comments"] == [{"comment": "g", "severity": "Medium"}]


def test_inject_is_idempotent_unless_forced():
    d = _data()
    inject_daydream_review(d, URL, CMTS, force=False)
    assert inject_daydream_review(d, URL, CMTS, force=False) is False
    assert sum(r["tool"] == "daydream" for r in d[URL]["reviews"]) == 1
    assert inject_daydream_review(d, URL, [], force=True) is True
    dd = [r for r in d[URL]["reviews"] if r["tool"] == "daydream"]
    assert len(dd) == 1 and dd[0]["review_comments"] == []


def test_inject_uses_custom_tool_label_without_clobbering_default():
    data = {URL: {"reviews": [{"tool": "daydream", "review_comments": []}]}}
    assert inject_daydream_review(data, URL, CMTS, force=False, tool="daydream-glm") is True
    tools = [r["tool"] for r in data[URL]["reviews"]]
    assert tools == ["daydream", "daydream-glm"]                 # distinct entry, no overwrite
    assert has_daydream_review(data[URL], tool="daydream-glm") is True
    assert has_daydream_review(data[URL], tool="daydream-codex") is False


def test_missing_golden_url_raises():
    with pytest.raises(KeyError):
        inject_daydream_review(_data(), "https://github.com/x/y/pull/1", CMTS, force=False)
