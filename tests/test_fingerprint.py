from daydream.pr_review import (
    ParsedIssue,
    compute_fingerprint,
    extract_anchors,
    parsed_issues_from_items,
)


def test_fingerprint_is_stable_sha256():
    fp1 = compute_fingerprint("src/auth.py", "Missing null check on `user_email`", "NPE rationale")
    fp2 = compute_fingerprint("src/auth.py", "Missing null check on `user_email`", "NPE rationale")
    assert fp1 == fp2
    assert len(fp1) == 64


def test_fingerprint_differs_on_file():
    assert compute_fingerprint("a.py", "bug `tok`", "") != compute_fingerprint("b.py", "bug `tok`", "")


def test_fingerprint_differs_on_description():
    a = compute_fingerprint("a.py", "bug `tok`", "first rationale")
    b = compute_fingerprint("a.py", "bug `tok`", "second rationale")
    assert a != b


def test_fingerprint_ignores_line_number():
    base = {
        "file": "a.py",
        "description": "bug `tok`",
        "rationale": "why",
        "severity": "high",
        "confidence": "HIGH",
    }
    issues = parsed_issues_from_items([{**base, "line": 10}, {**base, "line": 900}])
    assert issues[0].fingerprint == issues[1].fingerprint


def test_fingerprint_differs_on_title_word_order():
    """Description word order is preserved so differently-worded findings don't collide."""
    a = compute_fingerprint("a.py", "`alpha` and `bravo`", "")
    b = compute_fingerprint("a.py", "`bravo` and `alpha`", "")
    assert a != b


def test_fingerprint_ignores_severity_and_confidence():
    base = {
        "file": "src/auth.py",
        "line": 10,
        "description": "Token `token_env` leaks",
        "rationale": "rationale explains leaks",
    }
    run1 = parsed_issues_from_items([{**base, "severity": "high", "confidence": "HIGH"}])
    run2 = parsed_issues_from_items([{**base, "severity": "medium", "confidence": "MEDIUM"}])
    assert run1[0].fingerprint == run2[0].fingerprint


def test_fingerprint_anchors_exclude_rendering_markup():
    """Anchors come from raw description + rationale, never the rendered body."""
    anchors = extract_anchors("Token `token_env` leaks\nrationale explains leaks")
    assert "Severity" not in anchors
    assert "Confidence" not in anchors
    assert "token_env" in anchors


def test_parsed_issues_carry_fingerprint():
    items = [
        {
            "file": "src/auth.py",
            "line": 10,
            "description": "Missing check on `token`",
            "rationale": "NPE",
            "severity": "high",
            "confidence": "HIGH",
        },
        {
            "file": "src/db.py",
            "line": 20,
            "description": "SQL injection on `query`",
            "rationale": "unescaped",
            "severity": "high",
            "confidence": "HIGH",
        },
    ]
    issues = parsed_issues_from_items(items)
    assert len(issues) == 2
    for issue, item in zip(issues, items):
        assert issue.fingerprint is not None
        assert len(issue.fingerprint) == 64
        # The stamped value equals a direct recompute from the raw fields.
        assert issue.fingerprint == compute_fingerprint(
            item["file"], item["description"], item["rationale"]
        )
    assert issues[0].fingerprint != issues[1].fingerprint


def test_parsed_issue_fingerprint_defaults_none():
    """Other construction sites (parse_report etc.) are unaffected."""
    assert ParsedIssue(path="a.py", line=1, title="t", body="b").fingerprint is None
