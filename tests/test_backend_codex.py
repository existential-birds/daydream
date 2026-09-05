# tests/test_backend_codex.py
"""Tests for CodexBackend with canned JSONL fixtures."""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream import git_ops
from daydream.backends import (
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.backends.codex import (
    _CODEX_STDOUT_LIMIT_BYTES,
    CodexBackend,
    CodexError,
    _unwrap_shell_command,
)
from daydream.pricing import compute_cost, load_user_prices, resolve_prices
from tests.harness.codex_replay import make_mock_process, make_mock_process_from_fixture
from tests.harness.git_helpers import git as _git

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "codex_jsonl"


async def _run_fixture(backend: Any, prompt: Any, fixture: Any, **kwargs: Any) -> Any:
    """Drive ``execute`` over a canned fixture and collect the event list."""
    mock_proc = make_mock_process_from_fixture(fixture)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        return [event async for event in backend.execute(Path("/tmp"), prompt, **kwargs)]


@pytest.mark.asyncio
async def test_simple_text_events() -> None:
    backend = CodexBackend(model="gpt-5.3-codex")
    events = await _run_fixture(backend, "Say hello", "simple_text.jsonl")

    text_events = [e for e in events if isinstance(e, TextEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    result_events = [e for e in events if isinstance(e, ResultEvent)]

    assert len(text_events) == 1
    assert text_events[0].text == "Hello from Codex"
    assert len(cost_events) == 1
    # #194: gpt-5.3-codex is in MODEL_PRICES → cost is now synthesized at the
    # backend layer (D-16 reversed), matching compute_cost for these tokens.
    expected_cost = compute_cost(
        model="gpt-5.3-codex",
        input_tokens=100,
        cached_input_tokens=0,
        output_tokens=50,
        prices=resolve_prices(load_user_prices()),
    )
    assert cost_events[0].cost_usd is not None
    assert cost_events[0].cost_usd == pytest.approx(expected_cost)
    assert cost_events[0].input_tokens == 100
    assert cost_events[0].output_tokens == 50
    assert len(result_events) == 1
    assert result_events[0].continuation is not None
    assert result_events[0].continuation.backend == "codex"
    assert result_events[0].continuation.data["thread_id"] == "th_abc123"


@pytest.mark.asyncio
async def test_tool_use_events() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Run ls", "tool_use.jsonl")

    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    texts = [e for e in events if isinstance(e, TextEvent)]

    assert len(thinking) == 1
    assert thinking[0].text == "Let me run a command"

    assert any(ts.name == "shell" and ts.input == {"command": "ls -la"} for ts in tool_starts)
    assert any(tr.output == "file.py\ntest.py" and not tr.is_error for tr in tool_results)

    # file_change → synthetic ToolStart("patch") + ToolResult
    assert any(ts.name == "patch" for ts in tool_starts)
    assert any("main.py" in tr.output for tr in tool_results)

    assert any(t.text == "Done!" for t in texts)


@pytest.mark.asyncio
async def test_file_change_legacy_scalar_payload_unchanged() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Run ls", "tool_use.jsonl")
    starts = [e for e in events if isinstance(e, ToolStartEvent) and e.name == "patch"]
    assert len(starts) == 1
    assert starts[0].input == {"file": "main.py", "action": "modified"}


@pytest.mark.asyncio
async def test_file_change_changes_map_single_path() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Edit", "file_change_add.jsonl")
    starts = [e for e in events if isinstance(e, ToolStartEvent) and e.name == "patch"]
    assert len(starts) == 1
    assert starts[0].input["changes"] == [{"path": "spike-repo/a.py", "kind": "add"}]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1 and not results[0].is_error


@pytest.mark.asyncio
async def test_file_change_changes_map_multi_path() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Edit", "file_change_multi.jsonl")
    starts = [e for e in events if isinstance(e, ToolStartEvent) and e.name == "patch"]
    assert len(starts) == 1  # exactly ONE pair for the item — no id collisions
    assert sorted(
        (c["path"], c["kind"]) for c in starts[0].input["changes"]
    ) == [
        ("spike-repo/a.py", "add"),
        ("spike-repo/b.py", "update"),
        ("spike-repo/c.py", "delete"),
        ("spike-repo/d.py", "move"),
    ]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1 and not results[0].is_error


@pytest.mark.asyncio
async def test_file_change_changes_map_declined_is_error() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Edit", "file_change_declined.jsonl")
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].status == "declined"


@pytest.mark.asyncio
async def test_file_change_changes_map_failed_is_error_with_stderr() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Edit", "file_change_failed.jsonl")
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].status == "failed"
    assert "failed to apply hunk" in results[0].output


@pytest.mark.asyncio
async def test_file_change_pathless_payload_diagnostic() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Edit", "file_change_pathless.jsonl")
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert "unparseable" in results[0].output
    assert "file_change" in results[0].output  # echoes available fields
    assert "unknown" not in results[0].output


@pytest.mark.parametrize(
    ("fixture", "expected_output", "expected_text_count"),
    [
        (
            "structured_output.jsonl",
            {"issues": [{"id": 1, "description": "Fix type hints", "file": "app.py", "line": 5}]},
            None,
        ),
        (
            # Text delivered via item.updated deltas (item.completed has empty content).
            "streamed_structured_output.jsonl",
            {"issues": [{"id": 1, "description": "Missing type hint", "file": "app.py", "line": 10}]},
            1,
        ),
        (
            # agent_message with output_text content blocks (schema-constrained).
            "output_text_blocks.jsonl",
            {"issues": [{"id": 1, "description": "Bad import", "file": "main.py", "line": 3}]},
            None,
        ),
        (
            # Structured output returned in turn.completed result field.
            "turn_completed_result.jsonl",
            {"issues": [{"id": 1, "description": "Unused variable", "file": "utils.py", "line": 22}]},
            None,
        ),
    ],
    ids=["item-completed", "streamed-item-updated", "output-text-blocks", "turn-completed-result"],
)
@pytest.mark.asyncio
async def test_structured_output(fixture: Any, expected_output: Any, expected_text_count: Any) -> None:
    """Structured output is extracted across the Codex delivery shapes."""
    backend = CodexBackend(model="fixture-model")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    events = await _run_fixture(backend, "Parse", fixture, output_schema=schema)

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == expected_output
    if expected_text_count is not None:
        assert len([e for e in events if isinstance(e, TextEvent)]) == expected_text_count


@pytest.mark.asyncio
async def test_turn_failed_raises() -> None:
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("turn_failed.jsonl")

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="Model returned an error") as exc_info:
            async for _ in backend.execute(Path("/tmp"), "Fail"):
                pass

    # Structured turn.failed is NOT reclassified — category stays None.
    assert exc_info.value.category is None


@pytest.mark.asyncio
async def test_nonzero_exit_raises_with_captured_output() -> None:
    """Non-zero exit surfaces codex's diagnostic output as a PROCESS_EXIT CodexError."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process(
        ["Error: authentication required. Run `codex login` to authenticate."]
    )
    mock_proc.returncode = 1

    events = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="return code 1") as exc_info:
            async for event in backend.execute(Path("/tmp"), "Fail"):
                events.append(event)

    # No events yielded — the run must not appear successful.
    assert events == []
    msg = str(exc_info.value)
    assert "authentication required" in msg
    assert exc_info.value.category == "PROCESS_EXIT"


@pytest.mark.asyncio
async def test_nonzero_exit_with_no_output_still_informative() -> None:
    """If codex crashes with zero output, the error says so explicitly."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process([])
    mock_proc.returncode = 1

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="return code 1") as exc_info:
            async for _ in backend.execute(Path("/tmp"), "Fail"):
                pass

    assert "no non-JSON output captured" in str(exc_info.value)
    assert exc_info.value.category == "PROCESS_EXIT"


@pytest.mark.asyncio
async def test_continuation_token_resumes() -> None:
    """Test that continuation token is passed as 'resume' argument."""
    from daydream.backends import ContinuationToken

    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    token = ContinuationToken(backend="codex", data={"thread_id": "th_prev"})

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "Continue", continuation=token):
            pass

        call_args = mock_exec.call_args
        flat_args = list(call_args.args) if call_args.args else []
        assert "resume" in flat_args
        assert "th_prev" in flat_args


@pytest.mark.asyncio
async def test_codex_read_only_uses_read_only_sandbox(
    tmp_path: Path, linked_worktree: tuple[Path, Path],
) -> None:
    """read_only=True at a worktree runs in a disposable standalone clone:
    read-only sandbox, isolated cwd != source, matching HEAD + staged patch,
    feature-only files present, no origin, rebound prompt, cleaned up. The
    copy loop mirrors worktree content — unstaged edits to tracked files and
    untracked files — not just HEAD + staged index."""
    _main, source = linked_worktree
    parser = source / "services" / "taste" / "parser.go"
    parser.write_text("package taste\n\n// caller staged\nfunc CallerStaged() {}\n")
    _git(source, "add", "services/taste/parser.go")
    # Unstaged edit to a tracked file + untracked file: both must be carried
    # into the clone by the shutil.copy2 mirror loop.
    lexer = source / "services" / "taste" / "lexer.go"
    lexer.write_text("package taste\n\n// caller unstaged\nfunc CallerUnstaged() {}\n")
    notes = source / "notes.md"
    notes.write_text("untracked caller note\n")
    source_head = git_ops.head_sha(source)
    source_patch = git_ops.staged_patch(source)

    captured: dict[str, Any] = {}
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        flat = list(args)
        cd = flat[flat.index("--cd") + 1]
        isolated = Path(cd)
        captured["isolated"] = isolated
        captured["head"] = git_ops.head_sha(isolated)
        captured["patch"] = git_ops.staged_patch(isolated)
        captured["has_parser"] = (isolated / "services" / "taste" / "parser.go").exists()
        captured["lexer"] = (isolated / "services" / "taste" / "lexer.go").read_text()
        captured["has_notes"] = (isolated / "notes.md").exists()
        captured["notes"] = (isolated / "notes.md").read_text() if captured["has_notes"] else None
        captured["remote"] = git_ops.remote_url(isolated)
        captured["branches"] = git_ops.list_local_branches(isolated)
        captured["source_branches"] = git_ops.list_local_branches(source)
        captured["args"] = flat
        return mock_proc

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        async for _ in CodexBackend(model="fixture-model").execute(
            source, f"Audit repository at {source}", read_only=True,
        ):
            pass

    flat = captured["args"]
    assert flat[flat.index("--sandbox") + 1] == "read-only"
    isolated = captured["isolated"]
    assert isolated != source
    assert captured["head"] == source_head
    assert captured["patch"] == source_patch
    assert captured["has_parser"] is True
    # Mirror loop carries unstaged edits and untracked files into the clone.
    assert captured["lexer"] == lexer.read_text()
    assert captured["has_notes"] is True
    assert captured["notes"] == notes.read_text()
    assert captured["remote"] is None
    # Issue #1121: every source local branch resolves in the clone by name,
    # to the exact OID it had on the source at snapshot time.
    assert captured["branches"] == captured["source_branches"]
    assert "main" in captured["branches"]
    assert "feature" in captured["branches"]
    # Prompt rebound: isolated path present, source path absent in stdin bytes.
    written = mock_proc.stdin.write.call_args.args[0]
    assert isinstance(written, bytes)
    assert str(isolated).encode() in written
    assert str(source).encode() not in written
    # Temp dir removed after execute.
    assert not isolated.exists()


@pytest.mark.asyncio
async def test_codex_read_only_snapshot_all_branches_diff_and_source_immutable(
    tmp_path: Path, linked_worktree: tuple[Path, Path],
) -> None:
    """Issue #1121: a source with >=3 branches (incl. a slash name) snapshots
    ALL of them into the clone by OID; git diff <base>...HEAD works; the
    clone has no remote and no source path in its config; ref mutation in
    the clone cannot alter the source's HEAD, refs, or index."""
    _main, source = linked_worktree
    # Third branch with a slash-containing name, branched off main.
    _git(source, "branch", "release/9.9", "main")
    source_branches = git_ops.list_local_branches(source)
    assert set(source_branches) == {"main", "feature", "release/9.9"}
    head_before = git_ops.head_sha(source)

    captured: dict[str, Any] = {}
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        isolated = Path(list(args)[list(args).index("--cd") + 1])
        captured["branches"] = git_ops.list_local_branches(isolated)
        # git diff main...HEAD must succeed inside the clone (the exact
        # command from the archived failure: 'ambiguous argument main...HEAD').
        captured["diff_rc"] = subprocess.run(
            ["git", "diff", "main...HEAD", "--stat"], cwd=isolated,
            capture_output=True, text=True,
        ).returncode
        captured["diff_feature_rc"] = subprocess.run(
            ["git", "diff", "release/9.9...HEAD", "--stat"], cwd=isolated,
            capture_output=True, text=True,
        ).returncode
        captured["head"] = git_ops.head_sha(isolated)
        captured["config"] = subprocess.run(
            ["git", "config", "--local", "--list"], cwd=isolated,
            capture_output=True, text=True,
        ).stdout
        # Source-immutability sentinel: mutate a ref inside the clone, then
        # verify the source's refs and HEAD are untouched.
        git_ops.update_ref(isolated, "refs/heads/main", git_ops.head_sha(isolated))
        return mock_proc

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        async for _ in CodexBackend(model="fixture-model").execute(
            source, "Audit repository", read_only=True,
        ):
            pass

    assert captured["branches"] == source_branches  # all names, exact OIDs
    assert captured["head"] == head_before  # still detached at source HEAD
    assert captured["diff_rc"] == 0
    assert captured["diff_feature_rc"] == 0
    assert "remote" not in captured["config"]
    assert str(source) not in captured["config"]
    # Source untouched by the in-clone ref mutation.
    assert git_ops.head_sha(source) == head_before
    assert git_ops.list_local_branches(source) == source_branches


@pytest.mark.asyncio
async def test_codex_read_only_isolation_failure_is_fail_closed(
    tmp_path: Path,
    linked_worktree: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolation preparation failure raises CodexError and leaves source Git state untouched."""
    _main, source = linked_worktree
    (source / "services" / "taste" / "parser.go").write_text(
        "package taste\n\n// staged\nfunc S() {}\n"
    )
    _git(source, "add", "services/taste/parser.go")
    before_head = git_ops.head_sha(source)
    before_patch = git_ops.staged_patch(source)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise git_ops.GitError("isolation probe failure")

    monkeypatch.setattr("daydream.backends.codex.git_ops.clone", boom)
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="failed to create disposable read-only checkout"):
            async for _ in CodexBackend(model="fixture-model").execute(source, "Audit", read_only=True):
                pass

    assert git_ops.head_sha(source) == before_head
    assert git_ops.staged_patch(source) == before_patch


@pytest.mark.asyncio
async def test_codex_read_only_snapshot_failure_is_fail_closed(
    tmp_path: Path, linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GitError during branch snapshotting aborts preparation: CodexError
    raised, no codex process launched, source git state untouched."""
    _main, source = linked_worktree
    before_head = git_ops.head_sha(source)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise git_ops.GitError("snapshot ref failure")

    monkeypatch.setattr("daydream.backends.codex.git_ops.update_refs", boom)
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as exec_mock:
        with pytest.raises(CodexError, match="failed to create disposable read-only checkout"):
            async for _ in CodexBackend(model="fixture-model").execute(
                source, "Audit repository", read_only=True,
            ):
                pass
    exec_mock.assert_not_called()
    assert git_ops.head_sha(source) == before_head


def test_rebind_source_paths_preserves_sibling_paths() -> None:
    """The prompt rebind is anchored at path boundaries (issues #1/#9): sibling
    paths that merely share the source prefix survive, while sub-paths, the exact
    path, and ``//``-doubled renderings all map onto the isolated checkout."""
    from daydream.backends.codex import _rebind_source_paths

    source = Path("/home/exedev/work")
    execution = Path("/tmp/daydream-codex-read-only-abc/repo")

    # Sibling paths sharing only the prefix are untouched — nothing is rebound.
    siblings = f"never {source}-2 or {source}2 or {source}space or {source}.py"
    out = _rebind_source_paths(siblings, source, execution)
    assert f"{source}-2" in out
    assert f"{source}2" in out
    assert f"{source}space" in out
    assert f"{source}.py" in out
    assert str(execution) not in out

    # The exact path and sub-paths are rebound; no standalone source path remains.
    exact = f"Audit {source} and now {source}/sub/a then {source}/sub/b finally {source}"
    out2 = _rebind_source_paths(exact, source, execution)
    assert str(execution / "sub" / "a") in out2
    assert str(execution / "sub" / "b") in out2
    assert out2.startswith(f"Audit {execution}")
    assert out2.endswith(f"finally {execution}")
    assert str(source) not in out2

    # //-doubled renderings are caught too.
    doubled = f"Audit {source}//inner//file and {source}//two"
    out3 = _rebind_source_paths(doubled, source, execution)
    assert str(source) not in out3
    assert str(execution) in out3
    assert "/work//inner" not in out3


def test_isolated_child_env_strips_redirect_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """_isolated_child_env returns None when no isolation and strips the
    repo-redirect env vars (PWD/$GIT_*) when running in the disposable clone."""
    from daydream.backends.codex import _isolated_child_env

    monkeypatch.setenv("PWD", "/home/exedev/work")
    monkeypatch.setenv("OLDPWD", "/home/exedev")
    monkeypatch.setenv("GIT_DIR", "/home/exedev/work/.git")
    monkeypatch.setenv("HOME", "/home/exedev")
    monkeypatch.setenv("PATH", "/usr/bin")

    assert _isolated_child_env(Path("/home/exedev/work"), Path("/home/exedev/work")) is None

    env = _isolated_child_env(Path("/home/exedev/work"), Path("/tmp/clone/repo"))
    assert env is not None
    assert "PWD" not in env
    assert "OLDPWD" not in env
    assert "GIT_DIR" not in env
    # Non-redirect vars are preserved (isolation is path-hiding only).
    assert env["HOME"] == "/home/exedev"
    assert env["PATH"] == "/usr/bin"


@pytest.mark.asyncio
async def test_codex_read_only_resume_is_refused(
    tmp_path: Path, linked_worktree: tuple[Path, Path],
) -> None:
    """A read-only session passed a codex resume token fails closed: the resumed
    thread's stored cwd is the per-call clone, deleted when the turn ends."""
    from daydream.backends import ContinuationToken

    _main, source = linked_worktree
    backend = CodexBackend(model="fixture-model")
    token = ContinuationToken(backend="codex", data={"thread_id": "th_prev"})
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="cannot be resumed"):
            async for _ in backend.execute(source, "Continue", continuation=token, read_only=True):
                pass


@pytest.mark.asyncio
async def test_codex_read_only_parallel_calls_share_one_clone(
    linked_worktree: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent read-only calls on one backend build a single disposable clone
    (parallel fan-out must not clone the monorepo once per call) and remove it
    only after the last holder's generator exits."""
    from daydream.backends.codex import CodexBackend, _prepare_read_only_checkout

    _main, source = linked_worktree
    build_calls: list[Path] = []
    real_prep = _prepare_read_only_checkout

    def counting_prep(src: Path, destination: Path) -> Path:
        build_calls.append(destination)
        return real_prep(src, destination)

    monkeypatch.setattr("daydream.backends.codex._prepare_read_only_checkout", counting_prep)
    seen: list[str] = []
    entered = asyncio.Event()

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        flat = list(args)
        seen.append(flat[flat.index("--cd") + 1])
        if not entered.is_set():
            entered.set()
            # Keep both subprocesses alive simultaneously so the second call
            # reaches the checkout lock while the first still holds its ref.
            await asyncio.sleep(0.2)
        return make_mock_process_from_fixture("simple_text.jsonl")

    backend = CodexBackend(model="fixture-model")

    async def drive() -> None:
        async for _ in backend.execute(source, f"Audit {source}", read_only=True):
            pass

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        await asyncio.gather(drive(), drive())

    assert len(build_calls) == 1, "the two concurrent calls must share one clone build"
    assert len(seen) == 2
    assert seen[0] == seen[1], "both calls ran inside the same shared clone"
    # The shared clone is removed once the last holder exits.
    assert not Path(seen[0]).exists()


@pytest.mark.asyncio
async def test_codex_read_only_mirrors_symlinks_and_unstaged_deletions(
    linked_worktree: tuple[Path, Path],
) -> None:
    """The mirror loop keeps symlinks as links (file, dir, and dangling targets)
    and mirrors unstaged deletions, so the audit model sees the true worktree."""
    _main, source = linked_worktree
    taste = source / "services" / "taste"

    target = taste / "real.go"
    target.write_text("package taste\n")
    (taste / "link.go").symlink_to("real.go")
    _git(source, "add", "services/taste/real.go", "services/taste/link.go")
    _git(source, "commit", "-m", "add symlink")

    # Unstaged deletion of a tracked file.
    (taste / "lexer.go").unlink()
    # Untracked dangling symlink and an untracked symlink-to-directory.
    (taste / "deadlink").symlink_to("no-such-target")
    subdir = taste / "subdir"
    subdir.mkdir()
    (subdir / "inner.txt").write_text("i")
    (taste / "dir-link").symlink_to(subdir)

    captured: dict[str, Any] = {}
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        flat = list(args)
        isolated = Path(flat[flat.index("--cd") + 1])
        captured["isolated"] = isolated
        captured["link"] = (isolated / "services/taste/link.go").is_symlink()
        captured["link_target"] = str((isolated / "services/taste/link.go").readlink())
        captured["deadlink"] = (isolated / "services/taste/deadlink").is_symlink()
        captured["dir_link"] = (isolated / "services/taste/dir-link").is_symlink()
        captured["doomed_present"] = (isolated / "services/taste/lexer.go").exists()
        return mock_proc

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        async for _ in CodexBackend(model="fixture-model").execute(source, "Audit", read_only=True):
            pass

    assert captured["link"] is True
    assert captured["link_target"] == "real.go"
    assert captured["deadlink"] is True
    assert captured["dir_link"] is True
    assert captured["doomed_present"] is False
    assert not captured["isolated"].exists()


@pytest.mark.asyncio
async def test_codex_default_uses_full_access_sandbox() -> None:
    """read_only=False (default) keeps the existing danger-full-access sandbox."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p"):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert flat_args[flat_args.index("--sandbox") + 1] == "danger-full-access"
        assert "read-only" not in flat_args
        assert flat_args[flat_args.index("--cd") + 1] == "/tmp"


@pytest.mark.asyncio
async def test_codex_default_full_access_at_worktree_skips_isolation(
    linked_worktree: tuple[Path, Path],
) -> None:
    """read_only=False (default) at a Git worktree root keeps the caller's cwd.

    The disposable-clone isolation guard requires ``read_only=True``; with the
    default sandbox, ``--cd`` must be the source path itself — not a clone —
    so a regression that drops ``read_only`` from the guard is caught even
    though the non-worktree test above cannot observe it.
    """
    _main, source = linked_worktree
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(source, "p"):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert flat_args[flat_args.index("--sandbox") + 1] == "danger-full-access"
        assert "read-only" not in flat_args
        assert flat_args[flat_args.index("--cd") + 1] == str(source)


@pytest.mark.asyncio
async def test_codex_reasoning_effort_appends_config_override() -> None:
    """reasoning_effort forwards as -c model_reasoning_effort=<value>."""
    backend = CodexBackend(model="fixture-model", reasoning_effort="high")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p"):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert flat_args[flat_args.index("-c") + 1] == 'model_reasoning_effort="high"'


@pytest.mark.asyncio
async def test_codex_no_reasoning_effort_omits_config_override() -> None:
    """reasoning_effort=None (default) never adds a -c flag."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p"):
            pass

        flat_args = list(mock_exec.call_args.args)
        assert "-c" not in flat_args


@pytest.mark.asyncio
async def test_spawn_uses_start_new_session() -> None:
    """CLI spawns create a new session so the process group is killable."""
    backend = CodexBackend(model="gpt-5.1-codex")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ) as mock_exec:
        events = []
        async for event in backend.execute(Path("/tmp"), "hello"):
            events.append(event)
    assert mock_exec.call_args.kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_execute_finally_reaps_process_and_closes_pipes_after_exit() -> None:
    """Even when the CLI already exited, the finally reaps it and closes stdin.

    A grandchild holding the pipe write end means the stream never reaches EOF,
    so the fds are only released by an explicit teardown.
    """
    from tests.harness.fake_cli_process import FakeCliProcess, FakeCliSpawner

    backend = CodexBackend(model="gpt-5.1-codex")
    captured = FakeCliSpawner()

    async def fake_exec(*args: Any, **kwargs: Any) -> FakeCliProcess:
        proc = FakeCliProcess(
            [
                '{"type":"item.completed","item":{"type":"agent_message","text":"hello"}}',
                '{"type":"turn.completed","usage":{}}',
            ],
            exit_code=0,
        )
        captured.procs.append(proc)
        return proc

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        events = [event async for event in backend.execute(Path("/tmp"), "hello")]

    assert events  # the turn ran to completion
    proc = captured.procs[0]
    assert proc.returncode == 0
    assert proc.reaped, "the finally must reap the child even after clean exit"
    assert proc.stdin.closed, "the finally must close the stdin pipe"
    assert proc._transport.closed, "the finally must release the pipe fds even after clean exit"
    assert backend._transports == [], "the finally must drop the transport from the backend list"


@pytest.mark.asyncio
async def test_codex_stdout_limit_allows_large_jsonl_events() -> None:
    backend = CodexBackend(model="fixture-model")
    large_text = "x" * (70 * 1024)
    large_line = (
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": large_text}}) + "\n"
    ).encode()
    lines = [
        large_line,
        b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
    ]
    captured_kwargs: dict[str, object] = {}

    class _LimitAwareStdout:
        def __init__(self, limit: int) -> None:
            self._limit = limit
            self._lines = iter(lines)

        async def readline(self) -> bytes:
            try:
                line = next(self._lines)
            except StopIteration:
                return b""
            if len(line) > self._limit:
                raise ValueError("Separator is found, but chunk is longer than limit")
            return line

    async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
        captured_kwargs.update(kwargs)
        raw_limit = kwargs.get("limit", 64 * 1024)
        limit = raw_limit if isinstance(raw_limit, int) else 64 * 1024
        process = MagicMock()
        process.stdout = _LimitAwareStdout(limit)
        process.stdin = MagicMock()
        process.stdin.write = MagicMock()
        process.stdin.close = MagicMock()
        process.wait = AsyncMock(return_value=0)
        process.returncode = 0
        process.terminate = MagicMock()
        process.kill = MagicMock()
        return process

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        events = [event async for event in backend.execute(Path("/tmp"), "large event")]

    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert text_events[0].text == large_text
    assert captured_kwargs["limit"] == _CODEX_STDOUT_LIMIT_BYTES
    assert _CODEX_STDOUT_LIMIT_BYTES > len(large_line)


@pytest.mark.asyncio
async def test_toplevel_text_field() -> None:
    """Real Codex format: text directly on item, not in content blocks."""
    backend = CodexBackend(model="fixture-model")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    events = await _run_fixture(backend, "Parse", "toplevel_text.jsonl", output_schema=schema)

    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    assert len(thinking) == 1
    assert "read the review" in thinking[0].text

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [
            {
                "id": 1,
                "description": "Missing yield for non-result events",
                "file": "agents/architect.py",
                "line": 134,
            }
        ]
    }

    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert len(text_events) == 1


@pytest.mark.asyncio
async def test_turn_completed_cached_input_tokens() -> None:
    """Codex emits cached_input_tokens on turn.completed.usage; surface it on
    MetricsEvent and CostEvent so cache-hit ratios work for the Codex backend
    (refs #65, K4 — fix for the historical hardcoded cached_tokens=None)."""
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Cached", "turn_completed_cached_tokens.jsonl")

    metrics_events = [e for e in events if isinstance(e, MetricsEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]

    assert len(metrics_events) == 1
    assert metrics_events[0].prompt_tokens == 300
    assert metrics_events[0].completion_tokens == 150
    assert metrics_events[0].cached_tokens == 200
    # fixture-model is unknown to the price table → cost_usd stays None (#156
    # observable-marker preserved after #194 reversed D-16).
    assert metrics_events[0].cost_usd is None

    assert len(cost_events) == 1
    assert cost_events[0].input_tokens == 300
    assert cost_events[0].output_tokens == 150
    assert cost_events[0].cached_tokens == 200


@pytest.mark.asyncio
async def test_codex_synthesizes_cost_for_known_model() -> None:
    """#194: a known-priced model synthesizes cost at the backend layer.

    Drives a turn.completed with gpt-5.5 (in MODEL_PRICES) and known token
    counts. Both MetricsEvent and CostEvent must carry a non-None cost_usd
    matching compute_cost for the uncached-input/cached/output split. Mirrors
    how Claude (SDK total_cost_usd) and Pi (usage.cost.total) populate cost
    at the event layer. Reverses D-16.
    """
    backend = CodexBackend(model="gpt-5.5")
    # input_tokens is the TOTAL (cached is a subset per D-15); 15000 total
    # with 5000 cached → 10000 uncached. output 2000.
    lines = [
        '{"type":"thread.started","thread_id":"th_synth"}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
        '{"type":"turn.completed","usage":{"input_tokens":15000,'
        '"cached_input_tokens":5000,"output_tokens":2000}}',
    ]

    mock_proc = make_mock_process(lines)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Synth"):
            events.append(event)

    expected = compute_cost(
        model="gpt-5.5",
        input_tokens=10_000,  # uncached = 15000 - 5000
        cached_input_tokens=5_000,
        output_tokens=2_000,
        prices=resolve_prices(load_user_prices()),
    )
    assert expected is not None

    metrics_events = [e for e in events if isinstance(e, MetricsEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    assert len(metrics_events) == 1
    assert len(cost_events) == 1
    mev = metrics_events[0]
    cev = cost_events[0]
    assert mev.cost_usd is not None and mev.cost_usd > 0
    assert cev.cost_usd is not None and cev.cost_usd > 0
    assert mev.cost_usd == pytest.approx(expected)
    assert cev.cost_usd == pytest.approx(expected)
    # Token wiring unchanged: cached surfaced, input/output renamed.
    assert mev.prompt_tokens == 15_000
    assert mev.completion_tokens == 2_000
    assert mev.cached_tokens == 5_000


@pytest.mark.asyncio
async def test_codex_cost_none_for_unknown_model() -> None:
    """#156 preserved after #194: a model unknown to the price table yields cost_usd=None.

    Drives a turn.completed with ``definitely-not-a-real-model`` (absent from
    MODEL_PRICES and any user override). compute_cost returns None, so both
    MetricsEvent and CostEvent keep cost_usd=None — the observable marker that
    downstream renderers use to show "cost unavailable".
    """
    backend = CodexBackend(model="definitely-not-a-real-model")
    lines = [
        '{"type":"thread.started","thread_id":"th_unknown"}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
        '{"type":"turn.completed","usage":{"input_tokens":1000,'
        '"cached_input_tokens":200,"output_tokens":50}}',
    ]

    mock_proc = make_mock_process(lines)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        events = []
        async for event in backend.execute(Path("/tmp"), "Unknown"):
            events.append(event)

    metrics_events = [e for e in events if isinstance(e, MetricsEvent)]
    cost_events = [e for e in events if isinstance(e, CostEvent)]
    assert len(metrics_events) == 1
    assert len(cost_events) == 1
    assert metrics_events[0].cost_usd is None
    assert cost_events[0].cost_usd is None


@pytest.mark.asyncio
async def test_codex_backend_emits_turn_end_after_each_agent_message() -> None:
    """One TurnEndEvent per item.completed of type agent_message."""
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Two turns", "two_agent_turns.jsonl")

    texts = [e for e in events if isinstance(e, TextEvent)]
    turn_ends = [e for e in events if isinstance(e, TurnEndEvent)]
    assert len(texts) == 2
    assert len(turn_ends) == 2
    assert all(e.message_id == "" for e in turn_ends)


@pytest.mark.asyncio
async def test_concurrent_execute_calls_do_not_share_stdout_reader() -> None:
    """Overlapping runs on one backend must keep reading their own process."""
    backend = CodexBackend(model="fixture-model")

    class _ImmediateStdout:
        def __init__(self, lines: list[str]) -> None:
            self._lines = iter(lines)

        async def readline(self) -> bytes:
            try:
                return (next(self._lines) + "\n").encode()
            except StopIteration:
                return b""

    class _BlockingStdout:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self._waiting = False

        async def readline(self) -> bytes:
            if self._waiting:
                raise RuntimeError("readuntil() called while another coroutine is already waiting for incoming data")
            self._waiting = True
            self.entered.set()
            try:
                await self.release.wait()
                return b""
            finally:
                self._waiting = False

    def _proc(stdout: object) -> MagicMock:
        process = MagicMock()
        process.stdout = stdout
        process.stdin = MagicMock()
        process.stdin.write = MagicMock()
        process.stdin.close = MagicMock()
        process.wait = AsyncMock(return_value=0)
        process.returncode = 0
        process.terminate = MagicMock()
        process.kill = MagicMock()
        return process

    first_proc = _proc(
        _ImmediateStdout(
            [
                '{"type":"item.completed","item":{"type":"agent_message","text":"first"}}',
                '{"type":"turn.completed","usage":{}}',
            ]
        )
    )
    second_stdout = _BlockingStdout()
    second_proc = _proc(second_stdout)
    procs = iter([first_proc, second_proc])

    async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
        return next(procs)

    async def consume_second() -> list[object]:
        return [event async for event in backend.execute(Path("/tmp"), "second")]

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        first_iter = backend.execute(Path("/tmp"), "first")
        first_event = await anext(first_iter)
        assert isinstance(first_event, TextEvent)

        second_task = asyncio.create_task(consume_second())
        await second_stdout.entered.wait()

        try:
            turn_end = await anext(first_iter)
            assert isinstance(turn_end, TurnEndEvent)
            next_first_event = await anext(first_iter)
            assert isinstance(next_first_event, CostEvent)
        finally:
            second_stdout.release.set()
            await second_task


class TestUnwrapShellCommand:
    """Tests for _unwrap_shell_command helper."""

    def test_zsh_wrapper_with_cd(self) -> None:
        cmd = '/bin/zsh -lc "cd /home/user/project && make test"'
        assert _unwrap_shell_command(cmd) == "make test"

    def test_bash_wrapper_with_cd(self) -> None:
        cmd = '/bin/bash -lc "cd /tmp/work && pytest -x"'
        assert _unwrap_shell_command(cmd) == "pytest -x"

    def test_sh_wrapper_with_cd(self) -> None:
        cmd = '/bin/sh -lc "cd /app && echo hello"'
        assert _unwrap_shell_command(cmd) == "echo hello"

    def test_wrapper_without_cd(self) -> None:
        cmd = '/bin/zsh -lc "ls -la"'
        assert _unwrap_shell_command(cmd) == "ls -la"

    def test_plain_command_passthrough(self) -> None:
        assert _unwrap_shell_command("ls -la") == "ls -la"

    def test_empty_command(self) -> None:
        assert _unwrap_shell_command("") == ""

    def test_single_quotes(self) -> None:
        cmd = "/bin/zsh -lc 'cd /project && git status'"
        assert _unwrap_shell_command(cmd) == "git status"

    def test_unquoted_simple(self) -> None:
        """Real Codex format: no quotes around simple commands."""
        assert _unwrap_shell_command("/bin/zsh -lc ls") == "ls"

    def test_single_quoted_git_diff(self) -> None:
        """Real Codex format: single-quoted multi-word command."""
        cmd = "/bin/zsh -lc 'git diff main...HEAD'"
        assert _unwrap_shell_command(cmd) == "git diff main...HEAD"

    def test_double_quoted_sed(self) -> None:
        """Real Codex format: double-quoted command with inner single quotes."""
        cmd = """/bin/zsh -lc "sed -n '1,260p' amelia/agents/architect.py\""""
        assert _unwrap_shell_command(cmd) == "sed -n '1,260p' amelia/agents/architect.py"


@pytest.mark.asyncio
async def test_execute_raises_on_agents() -> None:
    """CodexBackend refuses the unsupported agents argument."""
    backend = CodexBackend(model="fixture-model")
    mock_agent = {"description": "test", "prompt": "test"}

    with pytest.raises(NotImplementedError, match="Codex backend does not support exploration"):
        async for _ in backend.execute(Path("/tmp"), "Test", agents={"explorer": mock_agent}):
            pass


# ---------------------------------------------------------------------------
# Parser hardening: deterministic tool-id correlation and observable
# parse-failure paths.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_well_formed_multi_tool_no_orphans(caplog: pytest.LogCaptureFixture) -> None:
    """No-id item.started/completed pairs correlate via FIFO with zero orphans.

    Drives ``item_started_no_id.jsonl``: two ``command_execution`` items and one
    ``mcp_tool_call`` item, all lacking ``id`` and arriving in start order.
    Every ToolResultEvent must pair with a ToolStartEvent (id-set equality) and
    no parser warning may fire.
    """
    backend = CodexBackend(model="fixture-model")
    with caplog.at_level(logging.WARNING, logger="daydream.backends.codex"):
        events = await _run_fixture(backend, "Run tools", "item_started_no_id.jsonl")

    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_starts) == 3
    assert len(tool_results) == 3
    start_ids = {e.id for e in tool_starts}
    result_ids = {e.id for e in tool_results}
    assert start_ids == result_ids, (
        f"every tool result must pair with a tool start; starts={start_ids} results={result_ids}"
    )

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warnings, [r.getMessage() for r in warnings]


@pytest.mark.asyncio
async def test_orphaned_tool_result_is_observable(caplog: pytest.LogCaptureFixture) -> None:
    """Orphaned tool result (no matching item.started) emits an OBSERVABLE warning.

    Drives ``orphaned_tool_result.jsonl``: a ``command_execution`` item.completed
    with no preceding item.started. Pre-fix this silently bucketed the result
    into ``unmatched_tool_results`` via a fresh random UUID. Post-fix the parser
    must (a) log a WARNING identifiable from caplog and (b) emit the
    ToolResultEvent with a deterministic ``codex-unmatched-<seq>`` id so the
    trajectory recorder still buckets it without a silent drop.
    """
    backend = CodexBackend(model="fixture-model")
    with caplog.at_level(logging.WARNING, logger="daydream.backends.codex"):
        events = await _run_fixture(backend, "Orphan", "orphaned_tool_result.jsonl")

    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].id == "codex-unmatched-0", (
        f"orphan should receive a deterministic sequence id; got {tool_results[0].id!r}"
    )

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("unmatched tool result" in w for w in warnings), (
        f"expected an 'unmatched tool result' WARNING; got {warnings}"
    )


@pytest.mark.asyncio
async def test_malformed_structured_output_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Malformed structured-output agent_text emits an OBSERVABLE warning.

    Drives ``malformed_structured_output.jsonl``: agent_text is the literal
    string "this is not valid JSON" while ``output_schema`` is set. Pre-fix the
    ``except json.JSONDecodeError: pass`` swallowed this and silently degraded
    to ``structured_result=None``. Post-fix the parser must log a WARNING
    identifiable from caplog; the ResultEvent continues to carry
    ``structured_output=None`` (the schema parse genuinely failed).
    """
    backend = CodexBackend(model="fixture-model")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}

    with caplog.at_level(logging.WARNING, logger="daydream.backends.codex"):
        events = await _run_fixture(backend, "Parse", "malformed_structured_output.jsonl", output_schema=schema)

    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output is None

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("structured output parse failed" in w for w in warnings), (
        f"expected a 'structured output parse failed' WARNING; got {warnings}"
    )


@pytest.mark.parametrize(
    ("fixture", "expected_substring"),
    [
        # Top-level ``text`` wins (real Codex CLI format).
        ("toplevel_text.jsonl", "Missing yield"),
        # ``content[].type == "text"`` block extraction.
        ("simple_text.jsonl", "Hello from Codex"),
        # ``content[].type == "output_text"`` block extraction.
        ("output_text_blocks.jsonl", "Bad import"),
    ],
)
def test_text_extraction_precedence(fixture: str, expected_substring: str) -> None:
    """Per-shape text extraction across the three Codex item shapes.

    Each fixture exercises exactly one shape; ``_extract_text`` must return the
    expected substring. The dedicated ``test_text_extraction_top_level_wins``
    below pins the precedence between top-level ``text`` and ``content[]`` blocks
    when both are present.
    """
    fixture_path = FIXTURES_DIR / fixture
    lines = fixture_path.read_text().strip().split("\n")
    items = [json.loads(line) for line in lines if "item.completed" in line]
    candidates = [i["item"] for i in items if i["item"].get("type") == "agent_message"]
    assert candidates, f"no agent_message item.completed in {fixture}"
    extracted = CodexBackend._extract_text(candidates[0])
    assert expected_substring in extracted, (
        f"fixture={fixture} extracted={extracted!r} expected substring={expected_substring!r}"
    )


def test_text_extraction_top_level_wins_over_content_blocks() -> None:
    """When BOTH top-level text and content[] blocks are set, top-level wins.

    Pinning the precedence: the per-shape fixtures above each carry only one
    shape, so this test fixes the relative ordering between the two paths.
    """
    item = {
        "text": "TOP-LEVEL",
        "content": [{"type": "text", "text": "BLOCK"}, {"type": "output_text", "text": "OUTPUT"}],
    }
    assert CodexBackend._extract_text(item) == "TOP-LEVEL"


# DAYDREAM_FANOUT_CONCURRENCY (#164) — codex shares the knob with claude, so a
# training run that swaps `--backend` does not silently change how many turns it
# asks the endpoint for.


@pytest.mark.parametrize(
    ("raw", "ceiling", "expected"),
    [(None, 10, 8), ("3", 10, 3), ("8", 2, 2), ("0", 10, 8), ("-1", 10, 8), ("notanint", 10, 8)],
)
def test_codex_fanout_concurrency_honours_the_shared_env_override(
    monkeypatch: pytest.MonkeyPatch,
    raw: str | None,
    ceiling: int,
    expected: int,
) -> None:
    from daydream.backends import effective_fanout_concurrency
    from daydream.backends.codex import CodexBackend

    if raw is None:
        monkeypatch.delenv("DAYDREAM_FANOUT_CONCURRENCY", raising=False)
    else:
        monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", raw)

    assert effective_fanout_concurrency(ceiling, CodexBackend("gpt-test")) == expected


@pytest.mark.asyncio
async def test_codex_preserves_exit_code_and_status_on_results() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Run failing command", "command_failures_issue1126.jsonl")
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    failed = results[0]
    assert failed.is_error is True
    assert failed.exit_code == 128
    assert failed.status == "completed"
    ok = results[1]
    assert ok.is_error is False
    assert ok.exit_code == 0
    assert ok.status == "completed"
