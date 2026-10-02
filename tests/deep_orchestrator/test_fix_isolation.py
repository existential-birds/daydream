"""Real-run shell writes must obey each fixer's exact assigned footprint."""

from __future__ import annotations

import re
import stat
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream import git_ops
from daydream.backends import AgentEvent, MaxTurnsError, ResultEvent, ToolResultEvent, ToolStartEvent
from daydream.config_file import DaydreamFileConfig
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.fix_isolation import FixIsolationRound
from daydream.runner import run
from daydream.workspace import WorkContext
from tests.harness.git_helpers import git
from tests.harness.remote_ci import NoCIRemote
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import MakeConfig, Mute, _add_bare_remote, _merge_item, _silence


class _ShellEscapeBackend(StubBackend):
    """Use real shell-equivalent Python writes, including absolute parent paths."""

    def __init__(self, target: Path, outcome: str) -> None:
        super().__init__(target)
        self.outcome = outcome
        self.fix_edit_line = "\n# successful sibling\n"
        self.merge_items = [_merge_item(1, "App.tsx", "high"), _merge_item(2, "api.py", "high")]
        self.parse_by_stack = {"structure": {
            "file": "App.tsx", "severity": "low", "confidence": "MEDIUM", "line": 1, "description": "docs",
        }}

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        match = re.search(r"^File: (.+)$", prompt, re.M) or re.search(r"^Fix these \d+ issues in (.+):$", prompt, re.M)
        if prompt.lower().startswith(("fix this issue", "fix these")) and match and Path(match[1]).name == "App.tsx":
            # The script changes a reviewed sibling, a reviewed file without a
            # finding, and private owner work. Repeat against the original tree
            # to reproduce fixers that explicitly cd back to the source checkout.
            script = """
from pathlib import Path
import subprocess, sys
for root in (Path.cwd(), Path(sys.argv[1])):
    for name in ('api.py', 'README.md', 'owner-draft.txt'):
        (root / name).write_text('unauthorized shell overwrite\\n')
    (root / 'orphan.py').write_text('unauthorized new source\\n')
    subprocess.run(['git', 'add', 'api.py'], cwd=root, check=True)
    (root / 'README.md').chmod(0o600)
(Path.cwd() / 'App.tsx').write_text('// assigned fix\\n')
"""
            yield ToolStartEvent(id="shell-escape", name="Bash", input={"command": "python restore.py"})
            await anyio.run_process([sys.executable, "-c", script, str(self._target)], cwd=cwd)
            yield ToolResultEvent(id="shell-escape", output="restored files", is_error=False)
            if self.outcome == "failure":
                raise MaxTurnsError("failed after writing outside the assigned file")
            if self.outcome == "timeout":
                await anyio.sleep(10)
            yield ResultEvent(structured_output=None, continuation=None)
            return
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event

@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
async def test_run_contains_shell_writes_and_keeps_successful_sibling(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, mute_side_effects: Mute,
    outcome: str,
) -> None:
    target = multi_stack_target
    _silence(monkeypatch)
    mute_side_effects()
    before_readme = (target / "README.md").read_text() + "\nowner's tracked edit\n"
    (target / "README.md").write_text(before_readme)
    (target / "owner-draft.txt").write_text("owner's private draft\n")
    before_app = (target / "App.tsx").read_text()
    before_index = git_ops.snapshot_index(target)
    backend = _ShellEscapeBackend(target, outcome)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    with anyio.fail_after(30):
        exit_code = await run(make_config(target, assume="yes", output_mode="loop",
            file_config=DaydreamFileConfig(group_max_wall_s=2.0, quality_gate_enabled=False),
        ))

    assert (target / "README.md").read_text() == before_readme
    assert stat.S_IMODE((target / "README.md").stat().st_mode) == 0o644
    assert (target / "owner-draft.txt").read_text() == "owner's private draft\n"
    assert not (target / "orphan.py").exists()
    assert "# successful sibling" in (target / "api.py").read_text()
    assert "unauthorized shell overwrite" not in (target / "api.py").read_text()
    assert git_ops.snapshot_index(target) == before_index
    assert (target / "App.tsx").read_text() == ("// assigned fix\n" if outcome == "success" else before_app)
    assert exit_code == (1 if outcome == "failure" else 0)

async def test_run_publishes_only_assigned_shell_fixes_with_real_checks_and_hooks(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig, no_ci_remote: NoCIRemote,
) -> None:
    target = multi_stack_target
    bare = _add_bare_remote(target)
    no_ci_remote.connect(target, bare)
    _silence(monkeypatch)
    backend = _ShellEscapeBackend(target, "success")
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    (target / "owner-draft.txt").write_text("owner draft\n")
    before_readme = (target / "README.md").read_bytes()
    hook_marker = target.parent / "scope-hook-ran"
    hook = target / ".git/hooks/pre-commit"
    hook.write_text(f"#!/bin/sh\nprintf ran > '{hook_marker}'\n")
    hook.chmod(0o755)
    check = (f'{sys.executable} -c "from pathlib import Path; '
        "assert 'successful sibling' in Path('api.py').read_text(); "
        "assert 'unauthorized' not in Path('README.md').read_text(); "
        "assert not Path('orphan.py').exists()\""
    )
    with anyio.fail_after(120):
        exit_code = await run(make_config(target, assume="yes", output_mode="loop", test_command=check,
            file_config=DaydreamFileConfig(quality_gate_enabled=False),
            pr_number=no_ci_remote.pr_number, pr_repo=no_ci_remote.base_repository,
        ))
    assert exit_code == 0
    assert hook_marker.read_text() == "ran"
    assert (target / "README.md").read_bytes() == before_readme
    assert (target / "owner-draft.txt").read_text() == "owner draft\n"
    assert set(git(target, "show", "--name-only", "--format=", "HEAD").splitlines()) == {"App.tsx", "api.py"}
    assert git(target, "rev-parse", "HEAD") == git(bare, "rev-parse", "refs/heads/feature")

async def test_run_preserves_preexisting_staged_and_unstaged_owner_edits(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    target = multi_stack_target
    _silence(monkeypatch)
    (target / "README.md").write_text("owner staged edit\n")
    git(target, "add", "README.md")
    (target / "README.md").write_text("owner staged plus unstaged edit\n")
    (target / "owner-draft.txt").write_text("private owner draft\n")
    before_index = git_ops.snapshot_index(target)
    backend = _ShellEscapeBackend(target, "success")
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    exit_code = await run(make_config(target, assume="yes", output_mode="loop"))
    assert exit_code == 1
    assert git_ops.snapshot_index(target) == before_index
    assert (target / "README.md").read_text() == "owner staged plus unstaged edit\n"
    assert (target / "owner-draft.txt").read_text() == "private owner draft\n"
    assert not (target / "orphan.py").exists()

@pytest.mark.parametrize("index_kind", ["staged", "intent-to-add", "assume-unchanged"])
def test_round_recovers_parent_types_modes_ignored_owner_files_and_exact_index(
    multi_stack_target: Path, tmp_path: Path, index_kind: str,
) -> None:
    """Supplement the runner cases with exact index flags and path types."""
    target = multi_stack_target
    (target / "pkg").mkdir()
    (target / "pkg/owner.py").write_text("owner source\n")
    (target / ".gitignore").write_text("owner.env\nhidden.py\n")
    git(target, "add", "pkg/owner.py", ".gitignore")
    git(target, "commit", "-m", "test: owner sources")
    (target / "owner.env").write_bytes(b"private ignored owner bytes\x00")
    outside = tmp_path / "external.txt"
    outside.write_text("external owner content\n")
    (target / "owner-link").symlink_to(outside)
    original_readme = (target / "README.md").read_bytes()
    original_api = (target / "api.py").read_bytes()
    if index_kind == "staged":
        (target / "README.md").write_text("owner staged edit\n")
        git(target, "add", "README.md")
        (target / "README.md").write_bytes(original_readme)
    elif index_kind == "intent-to-add":
        (target / "intent.py").write_text("owner intent to add\n")
        git(target, "add", "-N", "intent.py")
    else:
        git(target, "update-index", "--assume-unchanged", "api.py")
    index_before = git_ops.snapshot_raw_index(target)
    item = {**_merge_item(1, "App.tsx", "high"), "item_uid": "item:1"}
    footprint = AuthorizedFixFootprint.build(target, {"api.py", "App.tsx", "README.md"}, [item])
    work = WorkContext(target, target, "main", git(target, "rev-parse", "main"), "feature",
        git(target, "rev-parse", "HEAD"), False, "scope-regression",
    )
    isolation = FixIsolationRound(work, footprint)
    try:
        sibling, baseline = isolation.create_group()
        (sibling.repo / "App.tsx").write_text("// retained assigned fix\n")
        isolation.retain_group(sibling.repo, frozenset({"App.tsx"}), baseline)
        (target / "README.md").unlink()
        (target / "README.md").symlink_to(outside)
        (target / "pkg/owner.py").unlink()
        (target / "pkg").rmdir()
        (target / "pkg").symlink_to(tmp_path)
        (target / "api.py").write_text("hidden unauthorized write\n")
        (target / "api.py").chmod(0o600)
        (target / "owner.env").write_text("overwritten ignored owner file\n")
        (target / "hidden.py").write_text("new ignored source\n")
        git(target, "add", "App.tsx")
        isolation.restore_parent()
        isolation.publish()
        assert git_ops.snapshot_raw_index(target) == index_before
        assert (target / "README.md").read_bytes() == original_readme
        assert (target / "api.py").read_bytes() == original_api
        assert stat.S_IMODE((target / "api.py").stat().st_mode) == 0o644
        assert (target / "pkg/owner.py").read_text() == "owner source\n"
        assert (target / "owner.env").read_bytes() == b"private ignored owner bytes\x00"
        assert not (target / "hidden.py").exists()
        assert (target / "owner-link").is_symlink()
        assert outside.read_text() == "external owner content\n"
        assert (target / "App.tsx").read_text() == "// retained assigned fix\n"
    finally:
        isolation.close()

def test_round_accepts_owner_deletion_of_a_tracked_directory(multi_stack_target: Path) -> None:
    target = multi_stack_target
    (target / "pkg").mkdir()
    (target / "pkg/removed.py").write_text("owner deleted source\n")
    git(target, "add", "pkg/removed.py")
    git(target, "commit", "-m", "test: deleted directory baseline")
    (target / "pkg/removed.py").unlink()
    (target / "pkg").rmdir()
    item = {**_merge_item(1, "App.tsx", "high"), "item_uid": "item:1"}
    footprint = AuthorizedFixFootprint.build(target, {"App.tsx"}, [item])
    work = WorkContext(target, target, "main", git(target, "rev-parse", "main"), "feature",
        git(target, "rev-parse", "HEAD"), False, "deleted-directory",
    )
    isolation = FixIsolationRound(work, footprint)
    try:
        isolation.restore_parent()
        assert not (target / "pkg").exists()
    finally:
        isolation.close()
