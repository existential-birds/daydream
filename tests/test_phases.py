# tests/test_phases.py
"""Tests for phase functions with backend abstraction."""

import pytest

from daydream.backends import (
    ContinuationToken,
    ResultEvent,
    TextEvent,
)
from daydream.config import REVIEW_OUTPUT_FILE


@pytest.mark.asyncio
async def test_phase_test_and_heal_fix_uses_fresh_context(tmp_path, monkeypatch):
    """Test that fix-and-retry starts fresh (no continuation) with enriched prompt."""
    from daydream.phases import phase_test_and_heal

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_menu", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_error", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    call_count = 0
    token = ContinuationToken(backend="codex", data={"thread_id": "th_test"})
    captured_prompts: list[str] = []
    captured_continuations: list[ContinuationToken | None] = []

    class FreshContextBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            nonlocal call_count
            call_count += 1
            captured_prompts.append(prompt)
            captured_continuations.append(continuation)
            if call_count == 1:
                # First call: tests fail
                yield TextEvent(text="1 failed, 0 passed")
                yield ResultEvent(structured_output=None, continuation=token)
            elif call_count == 2:
                # Fix call: should start fresh (no continuation)
                yield TextEvent(text="Fixed")
                yield ResultEvent(structured_output=None, continuation=None)
            else:
                # Retry: tests pass, should also start fresh
                yield TextEvent(text="All 1 tests passed")
                yield ResultEvent(structured_output=None, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    # Simulate: fail -> choice "2" (fix and retry) -> pass
    choices = iter(["2"])
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: next(choices, "3"))

    feedback_items = [
        {"id": 1, "description": "Bug in handler", "file": "src/handler.py", "line": 10},
        {"id": 2, "description": "Missing import", "file": "src/utils.py", "line": 1},
    ]

    backend = FreshContextBackend()
    success, retries = await phase_test_and_heal(backend, tmp_path, feedback_items=feedback_items)

    assert success is True
    assert retries == 1
    assert call_count == 3

    # Fix call (call_count == 2) should have no continuation (fresh start)
    assert captured_continuations[1] is None, "Fix call should start fresh with no continuation"

    # Retry call (call_count == 3) should also have no continuation
    assert captured_continuations[2] is None, "Retry after fix should start fresh"

    # Fix prompt should contain test failure output and file paths
    fix_prompt = captured_prompts[1]
    assert "1 failed, 0 passed" in fix_prompt
    assert "src/handler.py" in fix_prompt
    assert "src/utils.py" in fix_prompt
    assert "Analyze the failures and fix them" in fix_prompt


@pytest.mark.asyncio
async def test_phase_parse_feedback_empty_response_returns_empty_list(tmp_path, monkeypatch):
    """When the agent returns empty text (schema miss), treat as no issues."""
    from daydream.phases import phase_parse_feedback

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    # Write a review file so the prompt references a real path
    (tmp_path / REVIEW_OUTPUT_FILE).write_text("## Verdict\n\nReady: Yes\n")

    class EmptyResponseBackend:
        """Simulates a schema miss: no structured output, no text."""

        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            # Only yield a result with no structured output (schema miss)
            yield ResultEvent(structured_output=None, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    result = await phase_parse_feedback(EmptyResponseBackend(), tmp_path)
    assert result == []


@pytest.mark.asyncio
async def test_phase_parse_feedback_json_fallback(tmp_path, monkeypatch):
    """When structured output fails but raw text is valid JSON, parse it."""
    from daydream.phases import phase_parse_feedback

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    (tmp_path / REVIEW_OUTPUT_FILE).write_text("## Issues\n\n1. [foo.py:10] Bug\n")

    class JsonTextBackend:
        """Simulates a schema miss where the model outputs JSON as plain text."""

        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            yield TextEvent(text='{"issues": [{"id": 1, "description": "Bug", "file": "foo.py", "line": 10}]}')
            yield ResultEvent(structured_output=None, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    result = await phase_parse_feedback(JsonTextBackend(), tmp_path)
    assert len(result) == 1
    assert result[0]["file"] == "foo.py"


class TestBuildFixPrompt:
    """Tests for _build_fix_prompt helper."""

    def test_short_output_included_fully(self):
        from daydream.phases import _build_fix_prompt

        output = "FAILED test_foo.py::test_bar - AssertionError"
        result = _build_fix_prompt(output)

        assert "Here is the test output:" in result
        assert "tail" not in result
        assert output in result
        assert "Analyze the failures and fix them" in result

    def test_long_output_truncated(self):
        from daydream.phases import TEST_OUTPUT_TAIL_LINES, _build_fix_prompt

        lines = [f"line {i}" for i in range(200)]
        output = "\n".join(lines)
        result = _build_fix_prompt(output)

        assert "tail of the test output" in result
        # Should contain the last 100 lines
        assert "line 199" in result
        assert "line 100" in result
        # Should NOT contain early lines
        assert "line 0\n" not in result
        assert f"line {200 - TEST_OUTPUT_TAIL_LINES - 1}\n" not in result

    def test_feedback_items_adds_file_list(self):
        from daydream.phases import _build_fix_prompt

        items = [
            {"id": 1, "description": "Bug", "file": "src/foo.py", "line": 10},
            {"id": 2, "description": "Typo", "file": "src/bar.py", "line": 5},
            {"id": 3, "description": "Dup", "file": "src/foo.py", "line": 20},
        ]
        result = _build_fix_prompt("test failed", items)

        assert "- src/bar.py" in result
        assert "- src/foo.py" in result
        assert "Focus on the files listed above" in result
        # Deduplication: foo.py should appear only once
        assert result.count("- src/foo.py") == 1

    def test_none_feedback_items_omits_file_section(self):
        from daydream.phases import _build_fix_prompt

        result = _build_fix_prompt("test failed", None)

        assert "Files modified" not in result
        assert "Focus on the files" not in result
        assert "Analyze the failures and fix them" in result

    def test_empty_feedback_items_omits_file_section(self):
        from daydream.phases import _build_fix_prompt

        result = _build_fix_prompt("test failed", [])

        assert "Files modified" not in result
        assert "Focus on the files" not in result


import subprocess


def test_git_diff_returns_diff(tmp_path):
    """Test _git_diff returns diff output against default branch."""
    from daydream.phases import _git_diff

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=tmp_path, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True,
                    env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                         "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"})
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=tmp_path, capture_output=True)
    (tmp_path / "file.txt").write_text("world")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "change"], cwd=tmp_path, capture_output=True,
                    env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                         "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"})

    diff = _git_diff(tmp_path)
    assert "hello" in diff or "world" in diff


def test_git_log_returns_log(tmp_path):
    """Test _git_log returns commit log."""
    from daydream.phases import _git_log

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=tmp_path, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_path, capture_output=True,
                    env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                         "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"})
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=tmp_path, capture_output=True)
    (tmp_path / "new.txt").write_text("new")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add new file"], cwd=tmp_path, capture_output=True,
                    env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                         "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"})

    log = _git_log(tmp_path)
    assert "add new file" in log


def test_git_branch_returns_branch(tmp_path):
    """Test _git_branch returns current branch name."""
    from daydream.phases import _git_branch

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "my-feature"], cwd=tmp_path, capture_output=True)
    (tmp_path / "file.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True,
                    env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                         "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"})

    branch = _git_branch(tmp_path)
    assert branch == "my-feature"


def test_git_diff_empty_when_no_changes(tmp_path):
    """Test _git_diff returns empty string when branch has no diff."""
    from daydream.phases import _git_diff

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=tmp_path, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True,
                    env={**__import__("os").environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
                         "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"})

    diff = _git_diff(tmp_path)
    assert diff == ""


@pytest.mark.asyncio
async def test_phase_understand_intent_confirmed_first_try(tmp_path, monkeypatch):
    """User confirms the agent's understanding on the first attempt."""
    from daydream.phases import phase_understand_intent

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    class IntentBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            yield TextEvent(text="This PR adds a login page with email/password authentication.")
            yield ResultEvent(structured_output=None, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    # User confirms with "y"
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    result = await phase_understand_intent(
        IntentBackend(), tmp_path,
        diff="diff --git a/login.py ...",
        log="abc1234 add login page",
        branch="feat/login",
    )

    assert "login" in result.lower()


@pytest.mark.asyncio
async def test_phase_understand_intent_correction_then_confirm(tmp_path, monkeypatch):
    """User corrects the agent's understanding, then confirms on second attempt."""
    from daydream.phases import phase_understand_intent

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    call_count = 0

    class IntentBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield TextEvent(text="This PR adds a signup page.")
                yield ResultEvent(structured_output=None, continuation=None)
            else:
                yield TextEvent(text="This PR adds a login page with OAuth support.")
                yield ResultEvent(structured_output=None, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    # First: correction, second: confirm
    responses = iter(["No, it's a login page with OAuth, not signup", "y"])
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: next(responses))

    result = await phase_understand_intent(
        IntentBackend(), tmp_path,
        diff="diff --git ...",
        log="abc1234 add login",
        branch="feat/login",
    )

    assert call_count == 2
    assert "login" in result.lower()


def test_parse_issue_selection_all():
    from daydream.phases import _parse_issue_selection
    issues = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert _parse_issue_selection("all", issues) == [1, 2, 3]


def test_parse_issue_selection_none():
    from daydream.phases import _parse_issue_selection
    issues = [{"id": 1}, {"id": 2}]
    assert _parse_issue_selection("none", issues) == []
    assert _parse_issue_selection("", issues) == []


def test_parse_issue_selection_specific():
    from daydream.phases import _parse_issue_selection
    issues = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]
    assert _parse_issue_selection("1,3,5", issues) == [1, 3, 5]


def test_parse_issue_selection_with_spaces():
    from daydream.phases import _parse_issue_selection
    issues = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert _parse_issue_selection("1, 3", issues) == [1, 3]


def test_parse_issue_selection_invalid_ignored():
    from daydream.phases import _parse_issue_selection
    issues = [{"id": 1}, {"id": 2}]
    # "99" doesn't exist, silently ignored
    assert _parse_issue_selection("1,99", issues) == [1]


def test_parse_issue_selection_single():
    from daydream.phases import _parse_issue_selection
    issues = [{"id": 1}, {"id": 2}]
    assert _parse_issue_selection("2", issues) == [2]


@pytest.mark.asyncio
async def test_phase_alternative_review_returns_issues(tmp_path, monkeypatch):
    """Agent returns numbered issues via structured output."""
    from daydream.phases import phase_alternative_review

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_issues_table", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    structured_issues = {
        "issues": [
            {
                "id": 1,
                "title": "Use dependency injection",
                "description": "Hard-coded dependencies make testing difficult",
                "recommendation": "Use constructor injection",
                "severity": "high",
                "files": ["src/service.py"],
            },
            {
                "id": 2,
                "title": "Missing error handling",
                "description": "No error handling for API calls",
                "recommendation": "Add try/except with retries",
                "severity": "medium",
                "files": ["src/api.py"],
            },
        ]
    }

    class ReviewBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            yield TextEvent(text="Found 2 issues.")
            yield ResultEvent(structured_output=structured_issues, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    issues = await phase_alternative_review(
        ReviewBackend(), tmp_path,
        diff="diff --git ...",
        intent_summary="Adds a user authentication service.",
    )

    assert len(issues) == 2
    assert issues[0]["title"] == "Use dependency injection"
    assert issues[1]["severity"] == "medium"


@pytest.mark.asyncio
async def test_phase_alternative_review_no_issues(tmp_path, monkeypatch):
    """Agent finds no issues — returns empty list."""
    from daydream.phases import phase_alternative_review

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_issues_table", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    class NoIssuesBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            yield TextEvent(text="Implementation looks good.")
            yield ResultEvent(structured_output={"issues": []}, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    issues = await phase_alternative_review(
        NoIssuesBackend(), tmp_path,
        diff="diff --git ...",
        intent_summary="Adds a login page.",
    )

    assert issues == []
