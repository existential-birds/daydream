from daydream.deep.verify_selection import changed_text_at

_DIFF = """diff --git a/auth.py b/auth.py
--- a/auth.py
+++ b/auth.py
@@ -1,3 +1,4 @@
 def login(user):
+    if not user.token: raise PermissionError
     return user
"""


def test_cited_changed_lines_are_the_added_text_of_the_covering_hunk() -> None:
    assert "permissionerror" in changed_text_at(_DIFF, "auth.py", 2).lower()
    assert changed_text_at(_DIFF, "auth.py", 3).strip() == ""  # unchanged line
    assert changed_text_at(_DIFF, "other.py", 2) == ""  # file not in the diff
    assert changed_text_at("", "auth.py", 1) == ""  # no diff at all
