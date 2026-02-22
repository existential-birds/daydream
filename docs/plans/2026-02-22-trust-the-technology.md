# `--trust-the-technology` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `--trust-the-technology` flag that runs a three-phase conversational review: understand intent, evaluate alternatives, generate a plan document.

**Architecture:** Three new phase functions in `phases.py`, a `run_trust()` orchestrator in `runner.py`, CLI flag in `cli.py`, and new UI elements (phase subtitles, issue table rendering). All phases use the `Backend` protocol — no skill invocations needed.

**Tech Stack:** Python 3.12, argparse, Rich (tables/panels), existing Backend abstraction, pytest + pytest-asyncio

---

### Task 1: Add `trust_the_technology` to RunConfig and CLI

**Files:**
- Modify: `daydream/runner.py:52-91` (RunConfig dataclass)
- Modify: `daydream/cli.py:72-287` (_parse_args function)
- Test: `tests/test_cli.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_cli.py

def test_trust_the_technology_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--trust-the-technology"])
    config = _parse_args()
    assert config.trust_the_technology is True


def test_ttt_short_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt"])
    config = _parse_args()
    assert config.trust_the_technology is True


def test_ttt_excludes_skill_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--python"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_ttt_excludes_review_only(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--review-only"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_ttt_excludes_loop(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--loop"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_ttt_excludes_pr(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--pr", "1", "--bot", "x"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_ttt_compatible_with_backend(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--backend", "codex"])
    config = _parse_args()
    assert config.trust_the_technology is True
    assert config.backend == "codex"


def test_ttt_compatible_with_model(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--model", "sonnet"])
    config = _parse_args()
    assert config.trust_the_technology is True
    assert config.model == "sonnet"


def test_ttt_default_is_false(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.trust_the_technology is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v -k "ttt or trust"`
Expected: FAIL — `RunConfig` has no `trust_the_technology` attribute, `--ttt` flag not recognized

**Step 3: Add RunConfig field**

In `daydream/runner.py`, add to the `RunConfig` dataclass after `max_iterations`:

```python
    trust_the_technology: bool = False
```

**Step 4: Add CLI flag and validation**

In `daydream/cli.py` `_parse_args()`:

Add the flag after the `--max-iterations` argument (before `args = parser.parse_args()`):

```python
    parser.add_argument(
        "--trust-the-technology", "--ttt",
        action="store_true",
        default=False,
        dest="trust_the_technology",
        help="Technology-agnostic review: understand intent, evaluate alternatives, generate plan",
    )
```

Add validation after the `--loop` validation block (after `if args.loop:` block):

```python
    # Validate --trust-the-technology mutual exclusions
    if args.trust_the_technology:
        if args.skill:
            parser.error("--trust-the-technology and skill flags are mutually exclusive")
        if args.review_only:
            parser.error("--trust-the-technology and --review-only are mutually exclusive")
        if args.loop:
            parser.error("--trust-the-technology and --loop are mutually exclusive")
        if args.pr is not None:
            parser.error("--trust-the-technology and --pr are mutually exclusive")
```

Add `trust_the_technology=args.trust_the_technology` to the `RunConfig(...)` constructor call.

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v -k "ttt or trust"`
Expected: ALL PASS

**Step 6: Run full test suite**

Run: `make check`
Expected: PASS (lint + typecheck + tests)

**Step 7: Commit**

```bash
git add daydream/cli.py daydream/runner.py tests/test_cli.py
git commit -m "feat: add --trust-the-technology / --ttt CLI flag"
```

---

### Task 2: Add git helper functions

**Files:**
- Modify: `daydream/phases.py` (add git helpers near `_detect_default_branch`)
- Test: `tests/test_phases.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_phases.py

import subprocess

def test_git_diff_returns_diff(tmp_path):
    """Test _git_diff returns diff output against default branch."""
    from daydream.phases import _git_diff

    # Set up a git repo with a commit and a branch change
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_phases.py -v -k "git_diff or git_log or git_branch"`
Expected: FAIL — `_git_diff`, `_git_log`, `_git_branch` don't exist

**Step 3: Implement git helpers**

Add to `daydream/phases.py` after the `_detect_default_branch` function:

```python
def _git_diff(cwd: Path) -> str:
    """Get the diff of current branch against the default branch.

    Returns:
        The diff output, or empty string if detection fails or no diff.

    """
    base_branch = _detect_default_branch(cwd)
    if not base_branch:
        return ""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "diff", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
            shell=False,
        )
        return result.stdout if result.returncode == 0 else ""
    except (subprocess.SubprocessError, OSError):
        return ""


def _git_log(cwd: Path) -> str:
    """Get the commit log of the current branch since diverging from default branch.

    Returns:
        The log output, or empty string if detection fails.

    """
    base_branch = _detect_default_branch(cwd)
    if not base_branch:
        return ""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "log", f"{base_branch}..HEAD", "--oneline"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
            shell=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.SubprocessError, OSError):
        return ""


def _git_branch(cwd: Path) -> str:
    """Get the current branch name.

    Returns:
        The branch name, or empty string if detection fails.

    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
            shell=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (subprocess.SubprocessError, OSError):
        return ""
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_phases.py -v -k "git_diff or git_log or git_branch"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add daydream/phases.py tests/test_phases.py
git commit -m "feat: add git helper functions for trust-the-technology"
```

---

### Task 3: Add WONDER and ENVISION phase subtitles to UI

**Files:**
- Modify: `daydream/ui.py:129-160` (PHASE_SUBTITLES dict)
- Test: `tests/test_phases.py` (or a quick inline check)

**Step 1: Write the failing test**

```python
# Append to tests/test_cli.py (or a new test file — but test_cli.py already imports from daydream)

def test_phase_subtitles_include_wonder_and_envision():
    from daydream.ui import PHASE_SUBTITLES
    assert "WONDER" in PHASE_SUBTITLES
    assert "ENVISION" in PHASE_SUBTITLES
    assert len(PHASE_SUBTITLES["WONDER"]) >= 2
    assert len(PHASE_SUBTITLES["ENVISION"]) >= 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_phase_subtitles_include_wonder_and_envision -v`
Expected: FAIL — "WONDER" and "ENVISION" not in PHASE_SUBTITLES

**Step 3: Add the subtitles**

In `daydream/ui.py`, add to `PHASE_SUBTITLES` dict before the closing `}`:

```python
    "WONDER": [
        "imagining what could be",
        "seeing with different eyes",
        "questioning the obvious",
        "the road not taken",
    ],
    "ENVISION": [
        "shaping the path forward",
        "drawing the map",
        "from thought to intention",
        "the blueprint emerges",
    ],
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_phase_subtitles_include_wonder_and_envision -v`
Expected: PASS

**Step 5: Commit**

```bash
git add daydream/ui.py tests/test_cli.py
git commit -m "feat: add WONDER and ENVISION phase subtitles"
```

---

### Task 4: Add issue display helpers to UI

**Files:**
- Modify: `daydream/ui.py` (add `print_issues_table` and `print_issue_detail` functions)
- Test: `tests/test_cli.py` (or `tests/test_ui.py` if one exists)

**Step 1: Write the failing test**

```python
# Append to tests/test_cli.py

def test_print_issues_table_renders(capsys):
    from io import StringIO
    from rich.console import Console
    from daydream.ui import NEON_THEME, print_issues_table

    test_console = Console(file=StringIO(), theme=NEON_THEME, force_terminal=True)
    issues = [
        {"id": 1, "title": "Bad pattern", "severity": "high", "description": "Uses antipattern",
         "recommendation": "Refactor", "files": ["src/main.py"]},
        {"id": 2, "title": "Missing test", "severity": "low", "description": "No test coverage",
         "recommendation": "Add tests", "files": ["src/utils.py"]},
    ]
    print_issues_table(test_console, issues)
    output = test_console.file.getvalue()
    assert "Bad pattern" in output
    assert "Missing test" in output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_print_issues_table_renders -v`
Expected: FAIL — `print_issues_table` doesn't exist

**Step 3: Implement issue display**

Add to `daydream/ui.py` near the end (before `ShutdownPanel` class or after `print_summary`):

```python
def print_issues_table(console: Console, issues: list[dict]) -> None:
    """Display issues as a numbered Rich table.

    Args:
        console: Rich Console instance.
        issues: List of issue dicts with id, title, severity, description,
                recommendation, files keys.

    """
    table = Table(
        box=box.SIMPLE_HEAVY,
        border_style=STYLE_PURPLE,
        show_header=True,
        header_style=STYLE_BOLD_CYAN,
    )
    table.add_column("#", style=STYLE_YELLOW, width=4)
    table.add_column("Severity", style=STYLE_ORANGE, width=8)
    table.add_column("Issue", style=STYLE_FG)

    severity_style = {"high": STYLE_RED, "medium": STYLE_YELLOW, "low": STYLE_GREEN}

    for issue in issues:
        sev = issue.get("severity", "medium")
        table.add_row(
            str(issue.get("id", "?")),
            Text(sev, style=severity_style.get(sev, STYLE_FG)),
            issue.get("title", issue.get("description", "No title")),
        )

    console.print()
    console.print(table)

    # Print full details for each issue
    for issue in issues:
        console.print()
        issue_id = issue.get("id", "?")
        title = issue.get("title", "No title")
        console.print(Text(f"  #{issue_id}: {title}", style=STYLE_BOLD_PINK))
        if "description" in issue:
            console.print(Text(f"  {issue['description']}", style=STYLE_FG))
        if "recommendation" in issue:
            console.print(Text(f"  Recommendation: {issue['recommendation']}", style=STYLE_CYAN))
        if "files" in issue and issue["files"]:
            files_str = ", ".join(issue["files"])
            console.print(Text(f"  Files: {files_str}", style=STYLE_DIM))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_print_issues_table_renders -v`
Expected: PASS

**Step 5: Commit**

```bash
git add daydream/ui.py tests/test_cli.py
git commit -m "feat: add issue table display for trust-the-technology"
```

---

### Task 5: Implement `phase_understand_intent`

**Files:**
- Modify: `daydream/phases.py` (add phase function)
- Test: `tests/test_phases.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_phases.py

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_phases.py -v -k "understand_intent"`
Expected: FAIL — `phase_understand_intent` doesn't exist

**Step 3: Implement `phase_understand_intent`**

Add to `daydream/phases.py`:

```python
async def phase_understand_intent(
    backend: Backend,
    cwd: Path,
    diff: str,
    log: str,
    branch: str,
) -> str:
    """Phase: Understand the intent of the PR through conversational confirmation.

    The agent examines the diff, commit log, and branch name to understand
    what the PR is trying to accomplish. The user confirms or corrects until
    the understanding is accurate.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for exploration.
        diff: Git diff output (main...HEAD).
        log: Git log output (main..HEAD --oneline).
        branch: Current branch name.

    Returns:
        The confirmed intent summary string.

    """
    print_phase_hero(console, "LISTEN", phase_subtitle("LISTEN"))

    prompt = f"""You have full access to explore the codebase. Examine the diff below and the codebase to understand the intent of these changes. Present your understanding concisely — what problem is being solved and how.

Branch: {branch}

Commit log:
{log}

Diff:
{diff}
"""

    while True:
        console.print()
        print_info(console, "Agent is analyzing the changes...")

        output, _ = await run_agent(backend, cwd, prompt)
        intent_text = output if isinstance(output, str) else str(output)

        console.print()
        response = prompt_user(
            console,
            "Is this understanding correct? [y/provide correction]",
            "y",
        )

        if response.lower() in ("y", "yes"):
            return intent_text

        # User provided a correction — build new prompt with context
        prompt = f"""You previously described the intent of these changes as:

{intent_text}

The user corrected your understanding: {response}

Re-examine the codebase and diff, and present an updated understanding of the intent.

Branch: {branch}

Commit log:
{log}

Diff:
{diff}
"""
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_phases.py -v -k "understand_intent"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add daydream/phases.py tests/test_phases.py
git commit -m "feat: add phase_understand_intent for trust-the-technology"
```

---

### Task 6: Add alternative review schema and implement `phase_alternative_review`

**Files:**
- Modify: `daydream/phases.py` (add schema + phase function)
- Test: `tests/test_phases.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_phases.py

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_phases.py -v -k "alternative_review"`
Expected: FAIL — `phase_alternative_review` doesn't exist

**Step 3: Implement schema and phase function**

Add to `daydream/phases.py`:

```python
ALTERNATIVE_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "files": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "title", "description", "recommendation", "severity", "files"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["issues"],
    "additionalProperties": False,
}
```

And the import at the top of `phases.py` — add `print_issues_table` to the UI imports:

```python
from daydream.ui import (
    ...
    print_issues_table,
)
```

Then the phase function:

```python
async def phase_alternative_review(
    backend: Backend,
    cwd: Path,
    diff: str,
    intent_summary: str,
) -> list[dict[str, Any]]:
    """Phase: Evaluate whether there's a better way to implement the PR.

    A fresh agent receives the confirmed intent summary and explores the
    codebase to identify issues — both architectural alternatives and
    incremental improvements.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for exploration.
        diff: Git diff output.
        intent_summary: Confirmed intent summary from phase_understand_intent.

    Returns:
        List of issue dicts, each with id, title, description, recommendation,
        severity, and files keys.

    """
    print_phase_hero(console, "WONDER", phase_subtitle("WONDER"))

    prompt = f"""The intent of this PR has been confirmed as:

{intent_summary}

Given this intent, explore the codebase and evaluate the implementation in the diff below. Would you have done this differently?

Return a numbered list of issues covering both architectural alternatives and incremental improvements. For each issue, include: a sequential id number, a brief title, a description of what's wrong or could be better, your recommended alternative, a severity level (high/medium/low), and the relevant file paths.

If the implementation is solid and you wouldn't change anything, return an empty issues list.

Diff:
{diff}
"""

    console.print()
    print_info(console, "Agent is evaluating the implementation...")

    result, _ = await run_agent(backend, cwd, prompt, output_schema=ALTERNATIVE_REVIEW_SCHEMA)

    if isinstance(result, dict) and "issues" in result:
        issues = result["issues"]
    else:
        _log_debug(f"[TTT_REVIEW] unexpected result type: {type(result).__name__}: {result!r:.500}\n")
        issues = []

    if issues:
        print_info(console, f"Found {len(issues)} issues")
        print_issues_table(console, issues)
    else:
        print_info(console, "No issues found — the implementation looks good")

    return issues
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_phases.py -v -k "alternative_review"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add daydream/phases.py tests/test_phases.py
git commit -m "feat: add phase_alternative_review for trust-the-technology"
```

---

### Task 7: Add issue selection parser helper

**Files:**
- Modify: `daydream/phases.py` (add `_parse_issue_selection`)
- Test: `tests/test_phases.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_phases.py

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_phases.py -v -k "parse_issue_selection"`
Expected: FAIL — `_parse_issue_selection` doesn't exist

**Step 3: Implement the parser**

Add to `daydream/phases.py`:

```python
def _parse_issue_selection(user_input: str, issues: list[dict[str, Any]]) -> list[int]:
    """Parse user's issue selection into a list of issue IDs.

    Args:
        user_input: User input string ("all", "none", "", or comma-separated IDs).
        issues: Full list of issue dicts with "id" keys.

    Returns:
        List of selected issue IDs. Empty list means skip.

    """
    cleaned = user_input.strip().lower()

    if cleaned in ("none", ""):
        return []

    if cleaned == "all":
        return [issue["id"] for issue in issues]

    valid_ids = {issue["id"] for issue in issues}
    selected = []
    for part in cleaned.split(","):
        part = part.strip()
        if part.isdigit():
            issue_id = int(part)
            if issue_id in valid_ids:
                selected.append(issue_id)

    return selected
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_phases.py -v -k "parse_issue_selection"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add daydream/phases.py tests/test_phases.py
git commit -m "feat: add issue selection parser for trust-the-technology"
```

---

### Task 8: Implement `phase_generate_plan`

**Files:**
- Modify: `daydream/phases.py` (add schema + phase function + markdown writer)
- Test: `tests/test_phases.py`

**Step 1: Write the failing tests**

```python
# Append to tests/test_phases.py

@pytest.mark.asyncio
async def test_phase_generate_plan_writes_markdown(tmp_path, monkeypatch):
    """Selected issues produce a markdown plan file in .daydream/."""
    from daydream.phases import phase_generate_plan

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    structured_plan = {
        "plan": {
            "summary": "Refactor auth to use dependency injection",
            "issues": [
                {
                    "id": 1,
                    "title": "Use dependency injection",
                    "changes": [
                        {"file": "src/service.py", "description": "Extract interface", "action": "modify"},
                    ],
                },
            ],
        }
    }

    class PlanBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            yield TextEvent(text="Here's the plan.")
            yield ResultEvent(structured_output=structured_plan, continuation=None)

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    issues = [
        {"id": 1, "title": "Use dependency injection", "description": "Hard-coded deps",
         "recommendation": "Use constructor injection", "severity": "high", "files": ["src/service.py"]},
        {"id": 2, "title": "Missing tests", "description": "No coverage",
         "recommendation": "Add tests", "severity": "low", "files": ["src/test.py"]},
    ]

    # User selects issue 1 only
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "1")

    plan_path = await phase_generate_plan(
        PlanBackend(), tmp_path,
        diff="diff --git ...",
        intent_summary="Adds authentication service",
        issues=issues,
    )

    assert plan_path is not None
    assert plan_path.exists()
    assert (tmp_path / ".daydream").is_dir()
    content = plan_path.read_text()
    assert "Implementation Plan" in content
    assert "dependency injection" in content.lower()


@pytest.mark.asyncio
async def test_phase_generate_plan_skip_on_none(tmp_path, monkeypatch):
    """User enters 'none' — no plan file generated."""
    from daydream.phases import phase_generate_plan

    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_dim", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    class NeverCalledBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            raise AssertionError("Should not be called when user selects 'none'")
            yield  # make it an async generator

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    issues = [{"id": 1, "title": "Issue", "description": "Desc",
               "recommendation": "Fix", "severity": "low", "files": []}]

    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "none")

    plan_path = await phase_generate_plan(
        NeverCalledBackend(), tmp_path,
        diff="diff ...",
        intent_summary="Test intent",
        issues=issues,
    )

    assert plan_path is None
    assert not (tmp_path / ".daydream").exists()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_phases.py -v -k "generate_plan"`
Expected: FAIL — `phase_generate_plan` doesn't exist

**Step 3: Implement schema, phase function, and markdown writer**

Add the plan schema to `daydream/phases.py`:

```python
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "title": {"type": "string"},
                            "changes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file": {"type": "string"},
                                        "description": {"type": "string"},
                                        "action": {"type": "string", "enum": ["modify", "create", "delete"]},
                                    },
                                    "required": ["file", "description", "action"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["id", "title", "changes"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["summary", "issues"],
            "additionalProperties": False,
        },
    },
    "required": ["plan"],
    "additionalProperties": False,
}
```

Add the markdown writer helper:

```python
def _write_plan_markdown(
    plan_path: Path,
    plan_data: dict[str, Any],
    intent_summary: str,
    branch: str,
    original_issues: list[dict[str, Any]],
) -> None:
    """Write the plan data as a markdown file.

    Args:
        plan_path: Path to write the markdown file.
        plan_data: Structured plan output from the agent.
        intent_summary: Confirmed intent summary.
        branch: Current branch name.
        original_issues: Full issue list (for severity/recommendation metadata).

    """
    from datetime import datetime

    issue_map = {i["id"]: i for i in original_issues}
    plan = plan_data.get("plan", plan_data)  # Handle both wrapped and unwrapped

    lines = [
        "# Implementation Plan",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Branch:** {branch}",
        "",
        "## Intent",
        intent_summary,
        "",
        "## Plan Summary",
        plan.get("summary", "No summary provided."),
        "",
    ]

    for plan_issue in plan.get("issues", []):
        issue_id = plan_issue.get("id", "?")
        title = plan_issue.get("title", "Untitled")
        original = issue_map.get(issue_id, {})

        lines.append(f"## Issue {issue_id}: {title}")
        lines.append(f"**Severity:** {original.get('severity', 'unknown')}")
        if original.get("description"):
            lines.append(f"**Problem:** {original['description']}")
        if original.get("recommendation"):
            lines.append(f"**Recommendation:** {original['recommendation']}")
        lines.append("")

        changes = plan_issue.get("changes", [])
        if changes:
            lines.append("### Changes")
            for change in changes:
                action = change.get("action", "modify")
                file_path = change.get("file", "unknown")
                desc = change.get("description", "")
                lines.append(f"- **{action}** `{file_path}` — {desc}")
            lines.append("")

    plan_path.write_text("\n".join(lines))
```

Then the phase function:

```python
async def phase_generate_plan(
    backend: Backend,
    cwd: Path,
    diff: str,
    intent_summary: str,
    issues: list[dict[str, Any]],
) -> Path | None:
    """Phase: Generate an implementation plan for selected issues.

    Prompts the user to select which issues to address, then launches an
    agent to create a detailed plan. Writes the plan as markdown to
    .daydream/plan-{timestamp}.md.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory (plan written relative to this).
        diff: Git diff output.
        intent_summary: Confirmed intent summary.
        issues: Full list of issues from phase_alternative_review.

    Returns:
        Path to the generated plan file, or None if user skipped.

    """
    print_phase_hero(console, "ENVISION", phase_subtitle("ENVISION"))

    console.print()
    response = prompt_user(
        console,
        "Create an implementation plan? Enter issue numbers (e.g., 1,3,5) or 'all', or 'none' to skip",
        "all",
    )

    selected_ids = _parse_issue_selection(response, issues)
    if not selected_ids:
        print_dim(console, "Skipping plan generation")
        return None

    selected_issues = [i for i in issues if i["id"] in selected_ids]
    issues_text = "\n".join(
        f"- #{i['id']} [{i.get('severity', '?')}] {i.get('title', 'No title')}: "
        f"{i.get('description', '')} → {i.get('recommendation', '')}"
        for i in selected_issues
    )

    prompt = f"""The intent of this PR is:
{intent_summary}

Create a detailed implementation plan for fixing these issues:
{issues_text}

For each issue, specify what files to change, what the change should be, and why. Make this actionable enough to hand to another developer or agent.

Diff for context:
{diff}
"""

    console.print()
    print_info(console, f"Generating plan for {len(selected_issues)} issue(s)...")

    result, _ = await run_agent(backend, cwd, prompt, output_schema=PLAN_SCHEMA)

    if not isinstance(result, dict):
        _log_debug(f"[TTT_PLAN] unexpected result type: {type(result).__name__}\n")
        print_warning(console, "Failed to generate structured plan")
        return None

    # Ensure .daydream/ directory exists
    daydream_dir = cwd / ".daydream"
    daydream_dir.mkdir(exist_ok=True)

    # Write plan file
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    plan_path = daydream_dir / f"plan-{timestamp}.md"

    _write_plan_markdown(plan_path, result, intent_summary, _git_branch(cwd), selected_issues)

    print_success(console, f"Plan written to {plan_path}")
    return plan_path
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_phases.py -v -k "generate_plan"`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add daydream/phases.py tests/test_phases.py
git commit -m "feat: add phase_generate_plan for trust-the-technology"
```

---

### Task 9: Implement `run_trust` orchestrator in runner.py

**Files:**
- Modify: `daydream/runner.py` (add `run_trust` function + wire into `run()`)
- Test: `tests/test_integration.py`

**Step 1: Write the failing test**

```python
# Append to tests/test_integration.py

@pytest.mark.asyncio
async def test_run_trust_full_flow(tmp_path, monkeypatch):
    """Integration test: full --trust-the-technology flow through all three phases."""
    import subprocess
    import os

    # Set up a git repo with a branch
    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=tmp_path, capture_output=True)
    (tmp_path / "app.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, env=env)
    subprocess.run(["git", "checkout", "-b", "feat/test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "app.py").write_text("print('world')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "change"], cwd=tmp_path, capture_output=True, env=env)

    call_count = 0

    class TrustMockBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Phase 1: understand intent
                yield TextEvent(text="This PR changes the hello message to world.")
                yield ResultEvent(structured_output=None, continuation=None)
            elif call_count == 2:
                # Phase 2: alternative review
                yield TextEvent(text="Found 1 issue.")
                yield ResultEvent(
                    structured_output={
                        "issues": [{
                            "id": 1, "title": "Use constants", "description": "Hardcoded string",
                            "recommendation": "Use a constant", "severity": "low", "files": ["app.py"],
                        }]
                    },
                    continuation=None,
                )
            elif call_count == 3:
                # Phase 3: generate plan
                yield TextEvent(text="Plan generated.")
                yield ResultEvent(
                    structured_output={
                        "plan": {
                            "summary": "Extract string to constant",
                            "issues": [{
                                "id": 1, "title": "Use constants",
                                "changes": [{"file": "app.py", "description": "Extract to constant", "action": "modify"}],
                            }],
                        }
                    },
                    continuation=None,
                )

        async def cancel(self):
            pass

        def format_skill_invocation(self, skill_key, args=""):
            return f"/{skill_key}"

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: TrustMockBackend())

    # Mock UI functions
    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_dim", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_issues_table", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())

    # Mock runner UI
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.console", type("C", (), {"print": lambda *a, **kw: None})())

    # Phase 1: user confirms intent; Phase 3: user selects "all" issues
    prompt_calls = iter(["y", "all"])
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: next(prompt_calls))

    config = RunConfig(
        target=str(tmp_path),
        trust_the_technology=True,
    )

    exit_code = await run(config)

    assert exit_code == 0
    assert call_count == 3
    # Plan file should exist in .daydream/
    daydream_dir = tmp_path / ".daydream"
    assert daydream_dir.exists()
    plan_files = list(daydream_dir.glob("plan-*.md"))
    assert len(plan_files) == 1
    content = plan_files[0].read_text()
    assert "Implementation Plan" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration.py::test_run_trust_full_flow -v`
Expected: FAIL — `run_trust` not defined, `trust_the_technology` not on RunConfig (if task 1 not yet done) or not wired into `run()`

**Step 3: Implement `run_trust` and wire it into `run()`**

Add imports to `daydream/runner.py`:

```python
from daydream.phases import (
    ...
    phase_alternative_review,
    phase_generate_plan,
    phase_understand_intent,
    _git_branch,
    _git_diff,
    _git_log,
)
```

Add the `run_trust` function after `run_pr_feedback`:

```python
async def run_trust(config: RunConfig, target_dir: Path) -> int:
    """Execute the trust-the-technology flow: understand, evaluate, plan.

    Args:
        config: Run configuration.
        target_dir: Resolved target directory path.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    backend = _resolve_backend(config, "review")

    # Gather git context
    diff = _git_diff(target_dir)
    log = _git_log(target_dir)
    branch = _git_branch(target_dir)

    if not diff:
        print_warning(console, "No diff found — nothing to review")
        return 0

    console.print()
    print_info(console, f"Target directory: {target_dir}")
    print_info(console, f"Branch: {branch}")
    print_info(console, f"Model: {config.model or '<backend-default>'}")
    console.print()

    # Phase 1: Understand intent
    intent_summary = await phase_understand_intent(backend, target_dir, diff, log, branch)

    # Phase 2: Alternative review
    issues = await phase_alternative_review(backend, target_dir, diff, intent_summary)

    if not issues:
        print_success(console, "No issues found — the implementation looks good!")
        return 0

    # Phase 3: Generate plan
    plan_path = await phase_generate_plan(backend, target_dir, diff, intent_summary, issues)

    if plan_path:
        print_success(console, f"Plan written to {plan_path}")

    return 0
```

Wire into `run()` — add this block right after the PR feedback branch (after `if config.pr_number is not None: return await run_pr_feedback(config, target_dir)`, around line 366):

```python
        # Trust-the-technology mode: separate flow
        if config.trust_the_technology:
            return await run_trust(config, target_dir)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_integration.py::test_run_trust_full_flow -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `make check`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add daydream/runner.py daydream/phases.py tests/test_integration.py
git commit -m "feat: add run_trust orchestrator and wire into runner"
```

---

### Task 10: Skip skill selection when --ttt is active

**Files:**
- Modify: `daydream/runner.py` (adjust skill selection logic)
- Test: `tests/test_integration.py`

The current `run()` function prompts for skill selection unless `start_at == "test"`. When `trust_the_technology` is True, the flow returns early via `run_trust()` before reaching the skill selection code. However, we should verify this works correctly and add a guard.

**Step 1: Verify the flow**

The `run_trust()` call is placed before the skill selection logic. Since it returns early, no change is needed. The integration test from Task 9 already covers this (no skill is set in the config, and it doesn't prompt).

**Step 2: Write an explicit guard test**

```python
# Append to tests/test_integration.py

@pytest.mark.asyncio
async def test_run_trust_does_not_prompt_for_skill(tmp_path, monkeypatch):
    """--ttt mode should never prompt for skill selection."""
    import subprocess
    import os

    env = {**os.environ, "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "checkout", "-b", "main"], cwd=tmp_path, capture_output=True)
    (tmp_path / "f.txt").write_text("a")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, env=env)
    subprocess.run(["git", "checkout", "-b", "feat"], cwd=tmp_path, capture_output=True)
    (tmp_path / "f.txt").write_text("b")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "change"], cwd=tmp_path, capture_output=True, env=env)

    class MinimalBackend:
        async def execute(self, cwd, prompt, output_schema=None, continuation=None):
            yield TextEvent(text="Intent: changes f.txt.")
            yield ResultEvent(structured_output={"issues": []}, continuation=None)
        async def cancel(self): pass
        def format_skill_invocation(self, k, a=""): return f"/{k}"

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: MinimalBackend())
    monkeypatch.setattr("daydream.phases.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.print_issues_table", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.phases.console", type("C", (), {"print": lambda *a, **kw: None})())
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_success", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.console", type("C", (), {"print": lambda *a, **kw: None})())

    # confirm intent
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    # This should NOT be called — if skill prompt_user is called, fail
    original_runner_prompt = None
    def runner_prompt_trap(*args, **kwargs):
        raise AssertionError("Should not prompt for skill selection in --ttt mode")
    monkeypatch.setattr("daydream.runner.prompt_user", runner_prompt_trap)

    config = RunConfig(target=str(tmp_path), trust_the_technology=True)
    exit_code = await run(config)
    assert exit_code == 0
```

**Step 3: Run test**

Run: `pytest tests/test_integration.py::test_run_trust_does_not_prompt_for_skill -v`
Expected: PASS (if task 9 is done correctly, the early return handles this)

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: verify --ttt skips skill selection prompt"
```

---

### Task 11: Final validation — full test suite and lint

**Files:** None (validation only)

**Step 1: Run the full CI check**

Run: `make check`
Expected: ALL PASS (lint + typecheck + tests)

**Step 2: Run just the new tests**

Run: `pytest tests/ -v -k "ttt or trust or understand_intent or alternative_review or generate_plan or parse_issue_selection or git_diff or git_log or git_branch or wonder or envision"`
Expected: ALL PASS

**Step 3: Manual smoke test (optional)**

Run from a branch with changes:
```bash
cd /path/to/some/project
daydream --ttt .
```
Verify:
- LISTEN phase hero appears
- Agent presents intent understanding
- User can confirm or correct
- WONDER phase hero appears
- Issues are displayed as a table
- ENVISION phase hero appears
- User can select issues
- Plan file written to `.daydream/plan-*.md`

**Step 4: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix: address lint/type issues from trust-the-technology implementation"
```
