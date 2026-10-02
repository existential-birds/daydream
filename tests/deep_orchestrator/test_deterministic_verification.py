"""Host checks substantiate post-fix deterministic regression claims."""

from __future__ import annotations

import json
import re
import shlex
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ResultEvent
from daydream.check_claims import CheckClaimUnavailable, substantiate_check_claims
from daydream.runner import run
from daydream.test_execution import resolve_test_recipe
from tests.harness.git_helpers import bare_remote, git, seed_feature_branch
from tests.harness.remote_ci import NoCIRemote
from tests.harness.stub_backend import StubBackend, silence
from tests.test_deep_orchestrator import MakeConfig


@pytest.mark.parametrize("scenario", [
    "false_lint", "real_lint", "unavailable", "semantic", "structured", "mixed", "mutating_check",
])
async def test_runner_substantiates_blocking_check_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    no_ci_remote: NoCIRemote, scenario: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """Replay the false Ruff claim through real Git, host checks, and publication."""
    repo = tmp_path / "check-claims"
    # E rules alone do not enable preview-only rules. Real invalid syntax is a
    # separate case proving that actual lint failures still block publication.
    seed_feature_branch(repo,
        base={
            "api.py": "VALUE = 1\n",
            "pyproject.toml": '[tool.ruff.lint]\nselect = ["E", "F", "I", "W"]\n',
            "Makefile": f"lint:\n\t{shlex.quote(sys.executable)} -m ruff check api.py\n",
        },
        feature={"api.py": "VALUE = 2\n\n\n\n\nOTHER = 3\n"},
    )
    initial_head = git(repo, "rev-parse", "HEAD")
    remote = bare_remote(tmp_path / "origin.git")
    git(repo, "remote", "add", "origin", str(remote))
    if scenario in {"false_lint", "structured"}:
        no_ci_remote.connect(repo, remote)
    test_log = tmp_path / "test.log"
    test_script = tmp_path / "validate.py"
    test_script.write_text(f"from pathlib import Path\nPath({str(test_log)!r}).write_text('passed')\n")
    test_command = shlex.join([sys.executable, str(test_script)])
    if scenario == "mutating_check":
        (repo / "Makefile").write_text(
            "lint:\n\t" + shlex.join([sys.executable, "-c", "open('api.py', 'w').write('VALUE = 99\\n')"]) + "\n"
        )

    class ClaimBackend(StubBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,) -> AsyncIterator[AgentEvent]:
            if "post-fix fix-verifier agent" in prompt.lower():
                ids = [int(value) for value in re.findall(r"(?m)^(\d+)\. \[", prompt)]
                reason = "New runs of three-plus blank lines violate Ruff, so make lint/check fails."
                command = None
                if scenario == "unavailable":
                    reason = "The mandatory make typecheck gate fails."
                elif scenario == "semantic":
                    reason = "The retained change returns an incorrect value for a new input."
                elif scenario == "structured":
                    reason = "A deterministic check is red."
                    command = test_command
                elif scenario == "mixed":
                    reason = "Ruff reports lint errors and the retained function returns an incorrect value."
                yield ResultEvent(structured_output={"verdicts": [
                    {"issue_id": issue_id, "verdict": "regressed" if issue_id == 1 else "resolved",
                     "reason": reason if issue_id == 1 else "repaired", "path": "api.py",
                     "check_command": command if issue_id == 1 else None,
                     "check_only": scenario not in {"semantic", "mixed"}}
                    for issue_id in ids
                ]}, continuation=None)
                return
            if prompt.lower().startswith(("fix this issue", "fix these")):
                path = cwd / "api.py"
                path.write_text(path.read_text() + (
                    "invalid syntax\n" if scenario == "real_lint" else "# repaired\n"
                ))
                yield ResultEvent(structured_output=None, continuation=None)
                return
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    backend = ClaimBackend(repo)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_a, **_k: backend)
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    silence(monkeypatch)
    rc = await run(make_config(repo, assume="yes", output_mode="loop", test_command=test_command,
        pr_number=no_ci_remote.pr_number, pr_repo=no_ci_remote.base_repository,
    ))
    if scenario in {"false_lint", "structured"}:
        assert rc == 0
        assert test_log.read_text() == "passed"
        assert git(repo, "rev-parse", "HEAD") != initial_head
        assert git(remote, "rev-parse", "refs/heads/feature") == git(repo, "rev-parse", "HEAD")
        outcomes = json.loads((repo / ".daydream/deep/fix-outcomes.json").read_text())["outcomes"]
        claim = outcomes["item:1"]
        assert claim["verdict"] == "unresolved"
        assert claim["check_evidence"]["status"] == "passed"
        assert claim["check_evidence"]["exit_status"] == 0
        assert "Unsupported regression claim" in claim["reason"]
    else:
        assert rc == 1
        assert git(repo, "rev-parse", "HEAD") == initial_head
        assert not test_log.exists()
        if scenario == "real_lint":
            outcomes = json.loads((repo / ".daydream/deep/fix-outcomes.json").read_text())["outcomes"]
            assert outcomes["item:1"]["verdict"] == "regressed"
            assert outcomes["item:1"]["check_evidence"]["status"] == "failed"
        elif scenario in {"unavailable", "mutating_check"}:
            failures = capsys.readouterr().out
            assert "unavailable" in failures.lower()
            assert "make typecheck" in failures if scenario == "unavailable" else "mutated" in failures
            assert not (repo / ".daydream/deep/fix-outcomes.json").exists()

@pytest.mark.parametrize("command_kind", ["missing_executable", "timeout"])
async def test_check_evidence_unavailable_fails_closed(tmp_path: Path, command_kind: str) -> None:
    """Real subprocess failures cannot become confirmed model regressions."""
    from types import SimpleNamespace
    script = tmp_path / "check.py"
    script.write_text("import time\ntime.sleep(20)\n")
    seed_feature_branch(tmp_path, base={"api.py": "VALUE = 1\n"}, feature={"api.py": "VALUE = 2\n"})
    argv = (["daydream-nonexistent-verifier-check"] if command_kind == "missing_executable"
            else [sys.executable, str(script)])
    config = SimpleNamespace(test_command=shlex.join(argv))
    recipe = resolve_test_recipe(config, SimpleNamespace(test_command=None), repo_root=tmp_path)
    verdicts = [{"issue_id": 1, "verdict": "regressed", "path": "api.py",
                 "reason": "The repository check fails", "check_command": shlex.join(argv)}]
    with pytest.raises(CheckClaimUnavailable, match="unavailable"):
        await substantiate_check_claims(tmp_path, verdicts, recipe=recipe, wall_budget_s=0.01)

async def test_verifier_does_not_execute_model_shell_commands(tmp_path: Path) -> None:
    """Check commands need repository authorization, even if they would pass."""
    verdicts = [{"issue_id": 1, "verdict": "regressed", "reason": "check fails", "check_command": "touch unauthorized"}]
    with pytest.raises(CheckClaimUnavailable, match="not repository-declared"):
        await substantiate_check_claims(tmp_path, verdicts, recipe=None)
    assert not (tmp_path / "unauthorized").exists()

@pytest.mark.parametrize("reason", ["mypy reports errors", "build fails"])
async def test_unrelated_test_recipe_cannot_refute_other_checks(tmp_path: Path, reason: str) -> None:
    from types import SimpleNamespace
    recipe = resolve_test_recipe(SimpleNamespace(test_command=f"{sys.executable} -c pass"),
        SimpleNamespace(test_command=None), repo_root=tmp_path,
    )
    with pytest.raises(CheckClaimUnavailable, match="no repository-declared command"):
        await substantiate_check_claims(tmp_path, [{"verdict": "regressed", "reason": reason}], recipe=recipe)

@pytest.mark.parametrize("explicit_command", [None, "make lint"])
async def test_all_explicit_check_claims_need_evidence(tmp_path: Path, explicit_command: str | None) -> None:
    seed_feature_branch(
        tmp_path, base={"api.py": "VALUE = 1\n", "Makefile": "lint:\n\ttrue\ntypecheck:\n\tfalse\n"},
        feature={"api.py": "VALUE = 2\n"},
    )
    result = await substantiate_check_claims(tmp_path, [{
        "verdict": "regressed", "reason": "make lint passes but make typecheck fails", "check_only": True,
        "check_command": explicit_command,
    }], recipe=None)
    assert result[0]["verdict"] == "regressed"
    assert result[0]["check_evidence"]["status"] == "failed"
    assert [check["status"] for check in result[0]["check_evidence"]["checks"]] == ["passed", "failed"]

async def test_legacy_green_check_needs_regression_classification(tmp_path: Path) -> None:
    seed_feature_branch(
        tmp_path, base={"api.py": "VALUE = 1\n", "Makefile": "lint:\n\ttrue\n"},
        feature={"api.py": "VALUE = 2\n"},
    )
    with pytest.raises(CheckClaimUnavailable, match="classification is unavailable"):
        await substantiate_check_claims(tmp_path, [{
            "verdict": "regressed", "reason": "Ruff rejects these blank lines so make lint fails",
        }], recipe=None)

@pytest.mark.parametrize("mutate", [False, True])
async def test_check_identity_preserves_symlinks_and_detects_clean_file_chmod(tmp_path: Path, mutate: bool) -> None:
    from types import SimpleNamespace

    seed_feature_branch(
        tmp_path, base={"api.py": "VALUE = 1\n", "clean.py": "VALUE = 7\n"},
        feature={"api.py": "VALUE = 2\n"},
    )
    (tmp_path / "owner-link").symlink_to("owner-notes")
    code = "import os; os.chmod('clean.py', 0o600)" if mutate else "pass"
    command = shlex.join([sys.executable, "-c", code])
    recipe = resolve_test_recipe(
        SimpleNamespace(test_command=command), SimpleNamespace(test_command=None), repo_root=tmp_path,
    )
    verdicts = [{"verdict": "regressed", "reason": "A deterministic repository gate is red",
                 "check_only": True, "check_command": command}]
    if mutate:
        with pytest.raises(CheckClaimUnavailable, match="mutated"):
            await substantiate_check_claims(tmp_path, verdicts, recipe=recipe)
    else:
        outcomes = await substantiate_check_claims(tmp_path, verdicts, recipe=recipe)
        assert outcomes[0]["verdict"] == "unresolved"
    assert (tmp_path / "owner-link").is_symlink()

async def test_explicit_check_classification_requires_host_evidence(tmp_path: Path) -> None:
    with pytest.raises(CheckClaimUnavailable, match="no repository-declared validation command"):
        await substantiate_check_claims(tmp_path, [{
            "verdict": "regressed", "reason": "this will break", "check_only": True,
        }], recipe=None)

async def test_host_checker_cannot_modify_ignored_owner_files(tmp_path: Path) -> None:
    from types import SimpleNamespace
    seed_feature_branch(
        tmp_path, base={"api.py": "VALUE = 1\n", ".gitignore": "owner.env\n"},
        feature={"api.py": "VALUE = 2\n"},
    )
    (tmp_path / "owner.env").write_text("owner's private state\n")
    command = shlex.join([sys.executable, "-c", "open('owner.env', 'w').write('damaged')"])
    recipe = resolve_test_recipe(
        SimpleNamespace(test_command=command), SimpleNamespace(test_command=None), repo_root=tmp_path,
    )
    with pytest.raises(CheckClaimUnavailable, match="mutated"):
        await substantiate_check_claims(tmp_path, [{
            "verdict": "regressed", "reason": "repository gate fails", "check_only": True, "check_command": command,
        }], recipe=recipe)
