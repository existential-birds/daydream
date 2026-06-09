from daydream.pr_review import ParsedIssue, compute_fingerprint


def test_fingerprint_is_stable_sha256():
    issue = ParsedIssue(path="src/auth.py", line=42, title="Missing null check on `user_email`", body="b")
    fp1 = compute_fingerprint(issue)
    fp2 = compute_fingerprint(issue)
    assert fp1 == fp2
    assert len(fp1) == 64


def test_fingerprint_differs_on_file():
    a = ParsedIssue(path="a.py", line=1, title="bug `tok`", body="")
    b = ParsedIssue(path="b.py", line=1, title="bug `tok`", body="")
    assert compute_fingerprint(a) != compute_fingerprint(b)


def test_fingerprint_differs_on_description():
    a = ParsedIssue(path="a.py", line=1, title="bug `alpha`", body="")
    b = ParsedIssue(path="a.py", line=1, title="bug `bravo`", body="")
    assert compute_fingerprint(a) != compute_fingerprint(b)


def test_fingerprint_ignores_line_number():
    a = ParsedIssue(path="a.py", line=10, title="bug `tok`", body="")
    b = ParsedIssue(path="a.py", line=900, title="bug `tok`", body="")
    assert compute_fingerprint(a) == compute_fingerprint(b)


def test_fingerprint_stable_across_anchor_order():
    a = ParsedIssue(path="a.py", line=1, title="`alpha` and `bravo`", body="")
    b = ParsedIssue(path="a.py", line=1, title="`bravo` and `alpha`", body="")
    assert compute_fingerprint(a) == compute_fingerprint(b)
