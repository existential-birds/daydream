"""Related Regression Commit."""

from __future__ import annotations

import json
import re
import shlex
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ResultEvent, TextEvent
from tests.harness.git_helpers import bare_remote as _bare_remote
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo
from tests.harness.remote_ci import NoCIRemote
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import (
    MakeConfig,
    _silence,
)


@pytest.mark.parametrize("remaining_verdict", ["unresolved", "wrong_target", "regressed"])
async def test_exhausted_fix_rounds_validate_and_publish_partial_fixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig, remaining_verdict: str,
    no_ci_remote: NoCIRemote,
) -> None:
    """Remaining findings cannot strand a tested fix; new regressions still block."""
    from daydream.runner import run

    repo = tmp_path / "partial-fixes"
    _init_repo(repo)
    (repo / "api.py").write_text("A = 1\n")
    _git(repo, "add", ".")
    _commit(repo, "base")
    _git(repo, "checkout", "-b", "feature")
    (repo / "api.py").write_text("A = 2\n")
    _git(repo, "add", "api.py")
    _commit(repo, "feature")
    initial_head = _git(repo, "rev-parse", "HEAD")
    remote = _bare_remote(tmp_path / "origin.git")
    _git(repo, "remote", "add", "origin", str(remote))
    if remaining_verdict != "regressed":
        no_ci_remote.connect(repo, remote)

    hook_log = tmp_path / "hooks.log"
    for hook in ("pre-commit", "pre-push"):
        path = repo / ".git" / "hooks" / hook
        prefix = path.read_text() if path.exists() else "#!/bin/sh\n"
        path.write_text(prefix + f"echo {hook} >> {shlex.quote(str(hook_log))}\n")
        path.chmod(0o755)
    test_script = tmp_path / "validate.py"
    test_log = tmp_path / "tests.log"
    test_script.write_text(
        "from pathlib import Path\n"
        "assert '# repaired' in Path('api.py').read_text()\n"
        f"Path({str(test_log)!r}).write_text('passed')\n"
    )
    class PartialBackend(StubBackend):
        async def execute(
            self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,
        ) -> AsyncIterator[AgentEvent]:
            if "post-fix fix-verifier agent" in prompt.lower():
                ids = [int(value) for value in re.findall(r"(?m)^(\d+)\. \[", prompt)]
                yield ResultEvent(structured_output={"verdicts": [
                    {"issue_id": i, "verdict": remaining_verdict if i == 1 else "resolved",
                     "reason": "remaining" if i == 1 else "repaired", "path": "api.py"}
                    for i in ids
                ]}, continuation=None)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    backend = PartialBackend(repo)
    backend.fix_edit_line = "# repaired\n"
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_a, **_k: backend)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    _silence(monkeypatch)

    rc = await run(make_config(
        repo, assume="yes", output_mode="loop",
        test_command=shlex.join([sys.executable, str(test_script)]),
        pr_number=no_ci_remote.pr_number, pr_repo=no_ci_remote.base_repository,
    ))

    if remaining_verdict == "regressed":
        assert rc == 1
        assert _git(repo, "rev-parse", "HEAD") == initial_head
        assert not test_log.exists()
        assert not hook_log.exists()
        return
    assert rc == 0
    assert test_log.read_text() == "passed"
    assert hook_log.read_text().splitlines() == ["pre-commit", "pre-push"]
    assert _git(repo, "rev-parse", "HEAD") != initial_head
    assert _git(remote, "rev-parse", "refs/heads/feature") == _git(repo, "rev-parse", "HEAD")
    message = _git(repo, "log", "-1", "--format=%B")
    assert "Structural maintainability concern" in message
    assert "Sample issue" not in message
    deep = repo / ".daydream" / "deep"
    outcomes = json.loads((deep / "fix-outcomes.json").read_text())["outcomes"]
    assert outcomes["item:1"]["verdict"] == remaining_verdict
    assert outcomes["item:2"]["verdict"] == "resolved"
    assert json.loads((deep / "push-verdict.json").read_text())["status"] == "succeeded"


@pytest.mark.parametrize("new_file_permissions", [0o600, 0o644])
async def test_related_regression_real_runner_stabilizes_and_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
    new_file_permissions: int,
) -> None:
    """Real host-test/Git path retains A/B/T and restores C + user scratch."""
    from daydream.runner import run

    repo = tmp_path / "footprint-run"
    _init_repo(repo)
    for name, value in (("api.py", "A = 1\n"), ("sibling.py", "B = 1\n"), ("other.py", "C = 1\n")):
        (repo / name).write_text(value)
    _git(repo, "add", ".")
    _commit(repo, "base")
    _git(repo, "checkout", "-b", "feature")
    (repo / "api.py").write_text("A = 10\n")
    _git(repo, "add", "api.py")
    _commit(repo, "feature")
    remote = _bare_remote(tmp_path / "origin.git")
    _git(repo, "remote", "add", "origin", str(remote))

    scratch_one = repo / "scratch-one.bin"
    scratch_two = repo / "scratch-two.txt"
    scratch_one.write_bytes(b"\x00owner-one")
    scratch_one.chmod(0o600)
    scratch_two.write_text("owner-two\n")
    deep = repo / ".daydream" / "deep"
    item = {
        "id": 1,
        "file": "api.py",
        "line": 1,
        "severity": "high",
        "description": "keep sibling fix and regression test",
        "evidence": "api.py:1",
        "recommendation": "repair both values",
        "lens": "python",
        "related_files": ["sibling.py", "tests/test_a.py"],
    }
    counter = tmp_path / "host-test-count"

    class FootprintBackend(StubBackend):
        def __init__(self, target: Path) -> None:
            super().__init__(target)
            self.fix_round = 0
            self.verify_round = 0

        async def execute(
            self,
            cwd: Path,
            prompt: str,
            *args: Any,
            **kwargs: Any,
        ) -> AsyncIterator[AgentEvent]:
            lowered = prompt.lower()
            if lowered.startswith(("fix this issue", "fix these")):
                self.fix_round += 1
                if self.fix_round == 1:
                    (cwd / "api.py").write_text("A = 2\n")
                    (cwd / "sibling.py").write_text("B = 5\n")
                    (cwd / "tests").mkdir(exist_ok=True)
                    (cwd / "tests/test_a.py").write_text(
                        "import pathlib, sys\n"
                        "counter = pathlib.Path(sys.argv[1])\n"
                        "counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')\n"
                        "root = pathlib.Path(__file__).parents[1]\n"
                        "assert (root / 'api.py').read_text() == 'A = 3\\n'\n"
                        "assert (root / 'sibling.py').read_text() == 'B = 7\\n'\n"
                    )
                    (cwd / "tests/test_a.py").chmod(new_file_permissions)
                    (cwd / "other.py").write_text("C = 9\n")
                    scratch_one.write_bytes(b"damaged")
                    scratch_two.unlink()
                else:
                    assert (cwd / "sibling.py").read_text() == "B = 5\n"
                    assert (cwd / "tests/test_a.py").is_file()
                    assert (cwd / "other.py").read_text() == "C = 1\n"
                    (cwd / "api.py").write_text("A = 3\n")
                yield TextEvent(text="fixed")
                yield ResultEvent(structured_output=None, continuation=None)
                return
            if "post-fix fix-verifier agent" in lowered:
                self.verify_round += 1
                assert kwargs.get("read_only") is True
                assert (cwd / "tests/test_a.py").is_file()
                assert (cwd / "other.py").read_text() == "C = 1\n"
                first = (
                    {"issue_id": 1, "verdict": "wrong_target", "path": "other.py", "reason": "retry original"}
                    if self.verify_round == 1
                    else {"issue_id": 1, "verdict": "resolved", "reason": "complete"}
                )
                ids = [int(value) for value in re.findall(r"(?m)^(\d+)\. \[", prompt)]
                verdicts = [first] + [
                    {"issue_id": issue_id, "verdict": "resolved", "reason": "complete"}
                    for issue_id in ids
                    if issue_id != 1
                ]
                yield TextEvent(text="")
                yield ResultEvent(structured_output={"verdicts": verdicts}, continuation=None)
                return
            if lowered.startswith("the tests failed"):
                (cwd / "sibling.py").write_text("B = 7\n")
                (cwd / "other.py").write_text("C = 8\n")
                yield TextEvent(text="healed")
                yield ResultEvent(structured_output=None, continuation=None)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    backend = FootprintBackend(repo)
    backend.parse_by_stack = {
        "python": {
            "severity": "high",
            "confidence": "HIGH",
            "issue": item,
        }
    }
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_a, **_k: backend)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    monkeypatch.setattr("daydream.run_context._prompt_user", lambda *_a, **_k: "2")
    _silence(monkeypatch, prompts=False)

    rc = await run(
        make_config(
            repo,
            assume="yes",
            archive=True,
            test_command=f"python tests/test_a.py {counter}",
        )
    )
    # The local bare remote deliberately has no GitHub identity. The retained
    # tree still stabilizes, commits, and pushes, but the new remote-CI phase
    # must fail closed with an explicit handoff rather than claim completion.
    assert rc == 1
    assert counter.read_text() == "3"
    assert (repo / "api.py").read_text() == "A = 3\n"
    assert (repo / "sibling.py").read_text() == "B = 7\n"
    assert (repo / "tests/test_a.py").is_file()
    assert (repo / "tests/test_a.py").stat().st_mode & 0o777 == new_file_permissions
    assert (repo / "other.py").read_text() == "C = 1\n"
    assert scratch_one.read_bytes() == b"\x00owner-one"
    assert scratch_one.stat().st_mode & 0o777 == 0o600
    assert scratch_two.read_text() == "owner-two\n"
    assert set(_git(repo, "show", "--pretty=", "--name-only", "HEAD").splitlines()) == {
        "api.py",
        "sibling.py",
        "tests/test_a.py",
    }
    assert _git(remote, "rev-parse", "refs/heads/feature") == _git(repo, "rev-parse", "HEAD")
    unavailable = json.loads((deep / "remote-ci-verdict.json").read_text())
    assert unavailable["status"] == "unavailable"
    assert unavailable["target"] is None
    assert (deep / "remote-ci-handoff.json").is_file()

    audit = json.loads((deep / "fix-footprint.json").read_text())
    assert any(event["action"] == "rejected_retarget" for event in audit["events"])
    authorization = {(event["path"], event["origin"]) for event in audit["events"] if event["action"] == "authorize"}
    assert {
        ("api.py", "reviewed"),
        ("api.py", "primary"),
        ("sibling.py", "related"),
        ("tests/test_a.py", "related"),
    } <= authorization
    restored = {event["path"] for event in audit["events"] if event["action"] == "restore"}
    assert {"other.py", "scratch-one.bin", "scratch-two.txt"} <= restored
    assert not {"sibling.py", "tests/test_a.py"} & restored
    assert {event["path"] for event in audit["events"] if event["action"] == "stage"} == {
        "api.py",
        "sibling.py",
        "tests/test_a.py",
    }
    verdict = json.loads((deep / "test-verdict.json").read_text())
    assert len(verdict["attempts"]) == 3
    assert verdict["session_id"] == audit["session_id"]
    outcomes = json.loads((deep / "fix-outcomes.json").read_text())
    capture = json.loads((deep / "recommended-capture.json").read_text())
    assert outcomes["session_id"] == capture["session_id"] == audit["session_id"]
    assert outcomes["evidence_key"] == capture["evidence_key"] == audit["evidence_key"]
    assert verdict["attempts"][-1]["input_tree_key"] == capture["tree_key"]
    assert verdict["attempts"][-1]["output_tree_key"] == capture["tree_key"]
    from daydream.archive import get_archive_dir

    archived = get_archive_dir() / "runs" / audit["session_id"]
    assert json.loads((archived / "deep/fix-footprint.json").read_text()) == audit
    for artifact in (
        "fix-outcomes.json",
        "test-verdict.json",
        "recommended-capture.json",
    ):
        assert (archived / "deep" / artifact).read_bytes() == (deep / artifact).read_bytes()
    assert (archived / "recommended.patch").read_bytes() == (repo / ".daydream/recommended.patch").read_bytes()
