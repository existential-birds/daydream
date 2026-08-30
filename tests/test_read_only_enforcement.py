"""Cross-backend gate: each backend enforces its read-only profile under read_only.

Claude refuses mutation at the tool layer (PreToolUse guard). Codex combines
its ``--sandbox read-only`` sandbox with a disposable standalone clone whenever
the cwd is a Git worktree root: a ``git commit`` inside that cwd can advance
only the disposable clone's refs and index — the caller's HEAD, staged index,
refs, and remotes are physically unreachable and deleted with the clone at
exit. The audit worktree remains, and Codex now additionally gets this hard
disposable-clone boundary. The real committing-subprocess regression proves
the caller's Git state survives a read-only Codex commit byte-for-byte; the
Claude guard tests pin the tool-layer denial surface.
"""

import os
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import pytest

from tests.harness.git_helpers import git as _harness_git

_COMMITTING_CODEX = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os, subprocess, sys
    marker = os.environ["COMMIT_MARKER"]
    # CodexBackend passes the workdir as --cd (the real `codex` binary chdirs
    # internally); honor it so the fake runs inside the isolated checkout.
    argv = sys.argv[1:]
    cwd = argv[argv.index("--cd") + 1]
    os.chdir(cwd)
    # git_ops.clone does not inherit the source repo's local user.name/
    # user.email, so pin a hermetic identity for the commit instead of
    # depending on ambient global git config (absent on CI).
    env = dict(os.environ)
    env.setdefault("GIT_AUTHOR_NAME", "daydream-test")
    env.setdefault("GIT_AUTHOR_EMAIL", "daydream-test@example.com")
    env.setdefault("GIT_COMMITTER_NAME", "daydream-test")
    env.setdefault("GIT_COMMITTER_EMAIL", "daydream-test@example.com")
    before = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "commit", "--allow-empty", "-m", "isolated audit commit"], env=env, check=True)
    after = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    remotes = subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip()
    with open(marker, "w") as fh:
        fh.write(f"cwd={cwd}\\nbefore={before}\\nafter={after}\\nremotes={remotes}\\n")
    sys.stdout.write('{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\\n')
    """
)


def _install_committing_codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: Path,
) -> None:
    """Put a real fake `codex` on $PATH that commits inside its cwd and records
    cwd/before/after/remotes to *marker*. Shebang pinned to sys.executable."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    script = bin_dir / "codex"
    script.write_text(
        _COMMITTING_CODEX.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1),
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("COMMIT_MARKER", str(marker))


def test_claude_read_only_profile_refuses_mutation() -> None:
    """Claude's observable refusal is the Bash-guard decision."""
    from daydream.backends.claude import _is_read_only_command

    # Mutating commands denied; read-only inspection allowed.
    assert _is_read_only_command("git commit -m x") is False
    assert _is_read_only_command("git reset --hard HEAD") is False
    assert _is_read_only_command("rm -rf build") is False
    # Newline/carriage-return command separators are bash chaining that shlex
    # elides as whitespace; they must be rejected on the raw string.
    assert _is_read_only_command("ls \nrm -rf /") is False
    assert _is_read_only_command("cat foo\rrm x") is False
    assert _is_read_only_command("git log") is True
    assert _is_read_only_command("git blame -L 1,1 f.py") is True


@pytest.mark.asyncio
async def test_claude_read_only_guard_blocks_write_tool() -> None:
    """Under read_only, the guard denies the Write tool outright (not just Bash)."""
    from daydream.backends.claude import _read_only_guard

    decision = cast(dict[str, Any], await _read_only_guard(
        {"tool_name": "Write", "tool_input": {"file_path": "x", "content": "y"}}, None, {},
    ))
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_codex_read_only_commit_cannot_change_source_head_or_index(
    tmp_path: Path,
    linked_worktree: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real fake Codex commits in its isolated cwd; the caller's linked-worktree
    HEAD and cached index stay byte-for-byte identical."""
    from daydream.agent import run_agent
    from daydream.backends.codex import CodexBackend
    from daydream.trajectory import DaydreamPhase

    _main, source = linked_worktree
    parser = source / "services" / "taste" / "parser.go"
    parser.write_text("package taste\n\n// caller staged sentinel\nfunc S() {}\n")
    _harness_git(source, "add", "services/taste/parser.go")
    before_head = _harness_git(source, "rev-parse", "HEAD")
    before_index = _harness_git(source, "diff", "--cached", "--binary")

    marker = tmp_path / "marker.txt"
    _install_committing_codex(tmp_path, monkeypatch, marker)

    await run_agent(
        CodexBackend(model="test-model"),
        source,
        f"Audit {source}",
        phase=DaydreamPhase.AUDIT,
        read_only=True,
        persist_session=False,
    )

    record = dict(
        line.split("=", 1) for line in marker.read_text().splitlines() if "=" in line
    )
    assert Path(record["cwd"]) != source
    assert record["before"] != record["after"]  # the fake really committed
    assert record["remotes"].strip() == ""  # no origin/source remote
    # Caller Git state unchanged.
    assert _harness_git(source, "rev-parse", "HEAD") == before_head
    assert _harness_git(source, "diff", "--cached", "--binary") == before_index
    assert not Path(record["cwd"]).exists()  # temp dir removed after run_agent
