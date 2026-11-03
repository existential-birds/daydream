"""Host checks substantiate post-fix deterministic regression claims."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

from daydream.check_claims import CheckClaimUnavailable, substantiate_check_claims
from daydream.test_execution import resolve_test_recipe
from tests.harness.git_helpers import seed_feature_branch


@pytest.mark.parametrize("command_kind", ["timeout"])
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
    """Check commands need repository authorization, even if they would pass.

    Malformed commands and escaping package directories are unavailable evidence;
    semantic and non-regressed verdicts pass through without any check.
    """
    from dataclasses import replace
    from types import SimpleNamespace

    verdicts = [{"issue_id": 1, "verdict": "regressed", "reason": "check fails", "check_command": "touch unauthorized"}]
    with pytest.raises(CheckClaimUnavailable, match="not repository-declared"):
        await substantiate_check_claims(tmp_path, verdicts, recipe=None)
    assert not (tmp_path / "unauthorized").exists()
    with pytest.raises(CheckClaimUnavailable, match="malformed check_command"):
        await substantiate_check_claims(
            tmp_path, [{"verdict": "regressed", "check_command": "make 'lint"}], recipe=None)
    recipe = resolve_test_recipe(SimpleNamespace(test_command=f"{sys.executable} -c pass"),
                                 SimpleNamespace(test_command=None), repo_root=tmp_path)
    escaping = replace(recipe, package=replace(recipe.package, cwd_relative="../outside"))
    with pytest.raises(CheckClaimUnavailable, match="escapes repository"):
        await substantiate_check_claims(tmp_path, [{"verdict": "regressed", "check_only": True}], recipe=escaping)
    with pytest.raises(CheckClaimUnavailable, match="make typecheck is not repository-declared"):
        await substantiate_check_claims(tmp_path, [{"verdict": "regressed", "reason": "make typecheck fails"}],
                                        recipe=None)
    passthrough = [{"verdict": "regressed", "reason": "returns the wrong value"}, {"verdict": "resolved"}]
    assert await substantiate_check_claims(tmp_path, passthrough, recipe=None) == passthrough

@pytest.mark.parametrize("reason", ["mypy reports errors", ])
async def test_unrelated_test_recipe_cannot_refute_other_checks(tmp_path: Path, reason: str) -> None:
    from types import SimpleNamespace
    recipe = resolve_test_recipe(SimpleNamespace(test_command=f"{sys.executable} -c pass"),
        SimpleNamespace(test_command=None), repo_root=tmp_path,
    )
    with pytest.raises(CheckClaimUnavailable, match="no repository-declared command"):
        await substantiate_check_claims(tmp_path, [{"verdict": "regressed", "reason": reason}], recipe=recipe)

@pytest.mark.parametrize("explicit_command", ["make lint"])
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

async def test_check_identity_preserves_symlinks_and_detects_clean_file_chmod(tmp_path: Path) -> None:
    """A green check refutes the claim; a check that mutates a clean file is unavailable evidence."""
    from types import SimpleNamespace

    seed_feature_branch(
        tmp_path, base={"api.py": "VALUE = 1\n", "clean.py": "VALUE = 7\n"},
        feature={"api.py": "VALUE = 2\n"},
    )
    (tmp_path / "owner-link").symlink_to("owner-notes")
    for mutate in (False, True):
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
