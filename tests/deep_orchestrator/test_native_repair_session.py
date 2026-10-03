"""Real registered repair lifetimes across retries, host mutation and publication."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops
from daydream.config_file import DaydreamFileConfig
from daydream.runner import run
from tests.deep_orchestrator.test_fix_isolation import _ShellEscapeBackend
from tests.harness.git_helpers import git
from tests.test_deep_orchestrator import MakeConfig, _add_bare_remote, _silence


class RetryingRepairBackend(_ShellEscapeBackend):
    """External model fixture: first fix remains unresolved, later attempts resolve."""

    def __init__(self, target: Path) -> None:
        super().__init__(target, "success")
        self.verifications = 0

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> Any:
        if "post-fix fix-verifier agent" in prompt.lower():
            self.verifications += 1
            self.fix_verify_resolve_after_round = 2 if self.verifications == 1 else 1
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event


async def test_registered_repair_retry_host_mutation_heal_and_hook_refusal(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    repo = multi_stack_target
    bare = _add_bare_remote(repo)
    git(repo, "push", "origin", "feature")
    initial_remote = git(bare, "rev-parse", "refs/heads/feature")
    private = repo / "owner-draft.txt"
    private.write_bytes(b"private owner bytes\x00")
    original_readme = (repo / "README.md").read_bytes()
    counter = tmp_path / "native-host-count"
    host = tmp_path / "native-test.py"
    host.write_text(
        "import pathlib, sys\n"
        "counter = pathlib.Path(sys.argv[1])\n"
        "n = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(n))\n"
        "if n == 1:\n"
        "    p = pathlib.Path('api.py')\n"
        "    p.write_text(p.read_text() + '# native host mutation\\n')\n"
        "sys.exit(1 if n == 1 else 0)\n"
    )
    hook_marker = tmp_path / "native-hook-ran"
    hook = repo / ".git/hooks/post-commit"
    hook.write_text(
        f"#!/bin/sh\nprintf ran > '{hook_marker}'\n"
        "printf '\n# native post-commit mutation\n' >> api.py\n"
    )
    hook.chmod(0o755)
    backend = RetryingRepairBackend(repo)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_a, **_k: backend)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    _silence(monkeypatch)

    code = await run(make_config(
        repo, assume="yes", output_mode="loop", test_command=f"{sys.executable} {host} {counter}",
        file_config=DaydreamFileConfig(quality_gate_enabled=False),
    ))

    assert code == 1
    assert backend.verifications >= 3
    assert counter.read_text() == "3"
    assert any(call["prompt"].lower().startswith("the tests failed") for call in backend.calls)
    assert "# native host mutation" in (repo / "api.py").read_text()
    assert hook_marker.read_text() == "ran"
    assert "# native post-commit mutation" in (repo / "api.py").read_text()
    assert private.read_bytes() == b"private owner bytes\x00"
    assert (repo / "README.md").read_bytes() == original_readme
    assert not (repo / "orphan.py").exists()
    assert git_ops.snapshot_index(repo).paths == ()
    assert git(bare, "rev-parse", "refs/heads/feature") == initial_remote
    audit = json.loads((repo / ".daydream/deep/test-verdict.json").read_text())
    assert audit["passed"] is True and len(audit["attempts"]) == 3


@pytest.mark.parametrize("mutation", ["source", "index", "private"])
async def test_registered_extension_mutation_cannot_publish_the_previous_candidate(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig, ext_dir: Any, mutation: str,
) -> None:
    repo = multi_stack_target
    bare = _add_bare_remote(repo)
    git(repo, "push", "origin", "feature")
    initial_head = git_ops.head_sha(repo)
    private = repo / "owner-draft.txt"
    private.write_text("private owner bytes\n")
    backend = _ShellEscapeBackend(repo, "success")
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_a, **_k: backend)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    _silence(monkeypatch)
    action = {
        "source": "(ctx.work.repo / 'api.py').write_text('extension target changed\\n')",
        "index": "git_ops.stage_paths(ctx.work.repo, [Path('api.py')])",
        "private": "(ctx.work.repo / 'owner-draft.txt').write_text('extension owner changed\\n')",
    }[mutation]
    ext_dir.write_module(
        "from pathlib import Path\n"
        "from daydream import git_ops\n"
        "from daydream.extensions.api import FlowStep\n"
        "async def mutate(ctx):\n"
        "    assert ctx.data['items']\n"
        f"    {action}\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='extension-mutation', run=mutate))\n"
        "    registry.insert_before('deep', anchor='commit', step='extension-mutation')\n"
    )
    assert await run(make_config(
        repo, assume="yes", output_mode="loop", test_command="true",
        file_config=DaydreamFileConfig(quality_gate_enabled=False),
    )) == 1
    assert git_ops.head_sha(repo) == initial_head
    assert git(bare, "rev-parse", "refs/heads/feature") == initial_head
    assert not (repo / "orphan.py").exists()
    if mutation != "private":
        assert private.read_text() == "private owner bytes\n"
