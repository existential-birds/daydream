"""Embedded review dispatch boundaries through the production runner and real Git."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, runner
from daydream.artifact_visibility import private_root_locations
from daydream.backends import BackendExecutionInput, create_backend as real_create_backend
from daydream.config_file import DaydreamFileConfig
from daydream.github_app import GitHubExecutionInput
from daydream.review_result import FinalizedReview
from daydream.run_config import FrozenAuthorDescription, RunConfig
from tests.harness import diagram_repos as dr
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.stub_backend import StubBackend


@pytest.fixture(autouse=True)
def _no_open_pr(fake_gh: FakeGh) -> None:
    fake_gh.set_response("pr-list", value=[])


def config_for(target: Path, **overrides: Any) -> RunConfig:
    fields: dict[str, Any] = {
        "target": str(target), "base": "main", "output_mode": "review", "non_interactive": True,
        "cleanup": False, "archive": False, "run_eval": False,
    }
    fields.update(overrides)
    return RunConfig(**fields)


@pytest.mark.parametrize("phase", ["diagram", "arbiter"])
async def test_embedded_review_preserves_backend_dispatch_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], phase: str,
) -> None:
    """Optional planning preserves lazy dispatch and primary failure evidence."""
    if phase == "diagram":
        target = tmp_path / "docs-only"
        seed_feature_branch(target, base={"README.md": "# Before\n"}, feature={"README.md": "# After\n"})
    else:
        target = dr.build_cross_module_repo(tmp_path)
    backend = StubBackend(target)
    monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", "4")
    environment = {
        "PATH": os.environ["PATH"], "HOME": str(tmp_path / "embedded-home"),
        "GH_CONFIG_DIR": str(tmp_path / "embedded-gh"), "DAYDREAM_FANOUT_CONCURRENCY": "4",
    }
    backend_input = BackendExecutionInput.from_environment(environment, backend="codex")
    execution = runner.RunnerExecutionInput(
        backend_input, GitHubExecutionInput(git_ops.StaticGitHubAuth(environment)),
    )
    created: list[str] = []

    efforts: list[str | None] = []

    def create_backend(name: str, *args: Any, **kwargs: Any) -> Any:
        created.append(name)
        efforts.append(kwargs.get("reasoning_effort"))
        if name == "codex":
            assert kwargs["execution_input"] is backend_input
            return backend
        # Keep unsupported/unused backend construction real: the regression
        # fails at its actual execution-input guard rather than a test sentinel.
        return real_create_backend(name, *args, **kwargs)

    monkeypatch.setattr(runner, "create_backend", create_backend)
    # Settings only the replaced reviewer read are named once and change nothing.
    config = config_for(target, diagram="auto", file_config=DaydreamFileConfig(
        backend="codex", latency_profile="fast", precision_mode=True, verify_all=False,
        phases={phase: {"backend": "osprey"}},
    ))
    roots = private_root_locations(base=tmp_path / "private")
    delivered: list[FinalizedReview] = []
    exit_code = await runner.run(config, execution=execution, private_roots=roots,
                                 review_result_handler=delivered.append)
    if phase == "arbiter":
        # The fresh challenge needs native host tools: the review is refused before any backend,
        # workspace or artifact exists, and nothing is delivered.
        assert exit_code == 1
        assert "'osprey' does not serve them" in capsys.readouterr().out
        # So is embedder-supplied author text over its bound, or frozen for another head than the approved one.
        head = git(target, "rev-parse", "HEAD")
        supported = replace(config, file_config=DaydreamFileConfig(backend="codex"), approved_head_sha=head)
        for description, refusal in ((FrozenAuthorDescription(head, "!" * 65_531), "exceeds its encoded UTF-8 bound"),
                                     (FrozenAuthorDescription("0" * 40, "intent"), "does not match analyzed HEAD")):
            assert await runner.run(replace(supported, frozen_author_description=description), execution=execution,
                                    private_roots=roots, review_result_handler=delivered.append) == 1
            assert refusal in " ".join(capsys.readouterr().out.split())
        assert created == [] and delivered == [] and backend.calls == []
        assert not (target / ".daydream").exists() and not (tmp_path / "private").exists()
    else:
        assert exit_code == 0
        assert set(created) == {"codex"}
        printed = " ".join(capsys.readouterr().out.replace("│", " ").split())
        assert printed.count("are ignored; remove them: latency_profile, precision_mode, verify_all") == 1
        # The run says what it is about to do, and a review announces no fix gate.
        assert "Review: 1 changed file(s) in 1 discovery assignment(s), then a fresh challenge" in printed
        assert "fix gate" not in printed
        # And it says how the review ended.
        assert printed.count("Review complete: 1 confirmed finding.") == 1
        # Discovery and its challenge keep their own default efforts; no latency route lowers the challenge.
        assert efforts == ["medium", "high"]
        artifact = dr.load_diagram_artifact(target)
        assert all(result["status"] == "skipped" for result in artifact["results"].values())
        assert backend.diagram_turns == {}
        # The embedded callback receives the one sealed result and the projected canonical items.
        (review,) = delivered
        assert review.terminal_result["analysis_state"] == "complete"
        assert review.terminal_result["analyzed_revision"]["head_sha"] == git(target, "rev-parse", "HEAD")
        assert [scope["scope_id"] for scope in review.terminal_result["planned_scopes"]] == ["committed:00000"]
        (item,) = review.items
        assert {key: item[key] for key in ("id", "file", "line", "description")} == {
            "id": 1, "file": "README.md", "line": 1, "description": "Sample issue"}
        assert item["item_uid"] and item["rationale"].startswith("The changed behavior is incorrect.")
        assert not {"severity", "confidence", "lens"} & item.keys()
        assert json.loads((target / ".daydream/deep/merged-items.json").read_text())["items"] == review.items
