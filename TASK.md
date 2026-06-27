# TASK: Fix #167 — harvest: git log --since timeout silently degrades fix-applied signal

## Context

You are working in a git worktree of the `daydream` repo. The issue is #167.

**Problem:** During `daydream corpus harvest`, `log_shas_since` runs `git log --since='30 days ago' head..base` with a 10s timeout. Against large monorepo clones, this times out and silently returns `[]`, degrading the fix-applied signal to `unknown`.

**Root cause:** The `--since` filter is redundant (the `head..base` range already bounds the walk), the 10s timeout is too short for large repos, and the silent `[]` return hides the degradation.

## Implementation Steps

### Step 1: Fix `log_shas_since` in `daydream/git_ops.py`

Find the function `log_shas_since` (around line 888). Currently:

```python
def log_shas_since(repo: Path, head: str, base: str, *, since_days: int) -> list[str]:
    """..."""
    try:
        proc = _run_git(
            repo,
            ["log", f"--since={since_days} days ago", "--pretty=%H", f"{head}..{base}"],
            timeout=10,
        )
    except GitError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
```

Change to:
- Remove the `since_days` parameter entirely
- Remove the `--since={since_days} days ago` argument from the git log command
- Raise timeout from `10` to `30`
- In the `except GitError` block, add `_logger.warning(...)` before `return []`
- In the `proc.returncode != 0` block, add `_logger.warning(...)` before `return []`
- Update the docstring to reflect the new signature and behavior

The `_logger` variable is already defined at module level: `_logger = logging.getLogger(__name__)` (line 49).

New implementation:
```python
def log_shas_since(repo: Path, head: str, base: str) -> list[str]:
    """Return full SHAs of commits on ``head..base``.

    The ``head..base`` range already bounds the walk; no ``--since`` date
    filter is needed (it was redundant and caused timeouts on large
    monorepos — see #167).

    Soft-failure semantics: returns ``[]`` on git error, but logs a
    warning so a degraded fix-applied verdict is not silent.

    Args:
        repo: Repository working directory.
        head: The divergence point; commits reachable from *head* are excluded.
        base: The tip ref; only commits reachable from *base* are included.

    Returns:
        List of 40-character SHA strings in ``git log`` output order
        (newest first). Empty list on any soft failure.
    """
    try:
        proc = _run_git(
            repo,
            ["log", "--pretty=%H", f"{head}..{base}"],
            timeout=30,
        )
    except GitError:
        _logger.warning(
            "log_shas_since: git log %s..%s failed after retries; "
            "returning empty window (fix-applied verdict may degrade to unknown)",
            head,
            base,
        )
        return []
    if proc.returncode != 0:
        _logger.warning(
            "log_shas_since: git log %s..%s exited non-zero (%d); "
            "returning empty window (fix-applied verdict may degrade to unknown)",
            head,
            base,
            proc.returncode,
        )
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
```

### Step 2: Update `_commits_in_window` in `daydream/training/harvest.py`

Find `_commits_in_window` (around line 302). Currently:

```python
def _commits_in_window(repo: Path, head: str, base: str, days: int) -> list[str]:
    """Return commits on ``base`` since ``head``'s ancestor, within ``days``.

    Used by the fix-applied cascade to bound the upstream review window.
    """
    return git_ops.log_shas_since(repo, head, base, since_days=days)
```

Change to:
```python
def _commits_in_window(repo: Path, head: str, base: str) -> list[str]:
    """Return commits on ``base`` since ``head``'s ancestor.

    Used by the fix-applied cascade to bound the upstream review window.
    The ``head..base`` range already bounds the walk; no date filter is
    needed (see #167).
    """
    return git_ops.log_shas_since(repo, head, base)
```

### Step 3: Update `fix_applied_signal` in `daydream/training/labeler_signals.py`

Find `fix_applied_signal` (around line 217). Three changes:

**3a.** In the function signature, change the type annotation:
```python
# Before:
commits_in_window_fetcher: Callable[[Path, str, str, int], list[str]],
# After:
commits_in_window_fetcher: Callable[[Path, str, str], list[str]],
```

**3b.** In the function body, change the call (around line 277):
```python
# Before:
window = commits_in_window_fetcher(repo_clone, head_sha, base_branch, window_days)
# After:
window = commits_in_window_fetcher(repo_clone, head_sha, base_branch)
```

**3c.** Update the docstring:
- In the cascade description (step 1), change to mention that the `head..base` range bounds the walk and no date filter is needed (see #167)
- In the `commits_in_window_fetcher` parameter doc, change "Called as `(repo_clone, head_sha, base_branch, window_days)`" to "Called as `(repo_clone, head_sha, base_branch)`"

Keep the `window_days` parameter on `fix_applied_signal` itself (it's part of the public API and used by callers), but it no longer feeds into `log_shas_since`.

### Step 4: Update tests in `tests/test_training_labeler_signals.py`

There are 4 lambda expressions that pass `commits_in_window_fetcher`. Each currently has 4 parameters `(repo, base, head, days)`. Change all 4 to 3 parameters `(repo, base, head)`:

- Line ~80: `lambda repo, base, head, days: ["commit1"]` → `lambda repo, base, head: ["commit1"]`
- Line ~103: `lambda repo, base, head, days: []` → `lambda repo, base, head: []`
- Line ~123: `lambda repo, base, head, days: ["c1"]` → `lambda repo, base, head: ["c1"]`
- Line ~146: `lambda repo, base, head, days: ["c1"]` → `lambda repo, base, head: ["c1"]`

### Step 5: Add real-path tests in `tests/test_git_ops.py`

Add two new test functions at the end of the file, following the existing test patterns (using `_make_repo_with_main`, `_commit`, `_git` helpers from `conftest`):

**Test 1: `test_log_shas_since_returns_commits_in_range`**
```python
def test_log_shas_since_returns_commits_in_range(tmp_path: Path) -> None:
    """log_shas_since returns SHAs for commits in head..base range."""
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _commit(repo, "topic-1")
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "b.txt")
    _commit(repo, "topic-2")
    shas = git_ops.log_shas_since(repo, "main", "topic")
    assert len(shas) == 2
    # newest first (git log order)
    topic2_sha = _git(repo, "rev-parse", "topic").strip()
    assert shas[0] in topic2_sha or len(shas[0]) == 40
```

**Test 2: `test_log_shas_since_warns_on_git_error`**
```python
def test_log_shas_since_warns_on_git_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """log_shas_since returns [] and logs a warning on git error."""
    repo = _make_repo_with_main(tmp_path)
    import logging
    with caplog.at_level(logging.WARNING, logger="daydream.git_ops"):
        result = git_ops.log_shas_since(repo, "main", "nonexistent-ref")
    assert result == []
    assert any("log_shas_since" in record.message for record in caplog.records)
```

## Verification

After making all changes, run:
```bash
make check
```
This runs ruff + mypy + pytest. All must pass (exit 0).

Also run individually if needed:
```bash
make lint       # ruff
make typecheck  # mypy
make test       # pytest
```

## Constraints

- Do NOT use `--no-verify` or skip any checks
- Do NOT modify any files not listed above
- The `_logger` variable in `git_ops.py` is already defined — do not re-import logging
- Follow existing code style (look at surrounding code for patterns)
- The repo uses `from __future__ import annotations` — type hints are strings
