"""The actual frozen Harbor bundle enters the shared offline native review API."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.harbor import build, entrypoint, package as pkg, verifier_core
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git
from tests.harness.harbor_review import answer_review, assignment_from_request, forbid_github
from tests.harness.native_pi import _NativeProvider
from tests.test_benchmark_harbor_build import _seed_ready_workspace, _stub_wheel
from tests.test_pi_native_output import native_provider as native_provider


def _compiled_case(tmp_path: Path, fake_gh: FakeGh) -> tuple[Path, Path, dict[str, Any]]:
    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)
    wheel, _ = _stub_wheel(tmp_path)
    pkg.build_harbor(ws, wheel=wheel)
    case = ws / "harbor" / build.derive_task_key(case_id)
    immutable = json.loads((case / "environment/immutable-request.json").read_text())
    repo = tmp_path / "runtime-repo"
    git(tmp_path, "clone", "--no-checkout", str(case / "environment/repository.bundle"), str(repo))
    git(repo, "branch", "base", "origin/base")
    git(repo, "checkout", "-b", "head", "origin/head")
    git(repo, "remote", "remove", "origin")
    return case, repo, immutable


def _environment(case: Path, repo: Path, root: Path, config: Path) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"], "LANG": "C.UTF-8", "DAYDREAM_REVIEW_BACKEND": "pi",
        "DAYDREAM_REVIEW_MODEL": "output-model", "DAYDREAM_REVIEW_API_KEY": "synthetic-loopback-only",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api/v1", "DAYDREAM_REVIEW_REPO_DIR": str(repo),
        "DAYDREAM_REVIEW_CASE_ID": case.name, "DAYDREAM_REVIEW_ARTIFACT_PATH": str(root / "review.json"),
        "DAYDREAM_REVIEW_TRAJECTORY_PATH": str(root / "logs/trajectory.json"),
        "DAYDREAM_REVIEW_IMMUTABLE_INPUT": str(case / "environment/immutable-request.json"),
        "DAYDREAM_REVIEW_BUNDLE": str(case / "environment/repository.bundle"),
        "DAYDREAM_REVIEW_AGENT_DIR": str(config),
    }


@pytest.mark.parametrize("outcome", [
    "finding", "clean", "partial", "noninline", "bundle-tamper", "model-spelling",
    "model-missing", "model-empty", "model-blank",
    "output-fault", "candidate-fault", "trajectory-fault", "long-title", "blank-title", "oversized-body",
])
async def test_compiled_harbor_review_is_offline_source_bound_and_failure_atomic(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider,
    outcome: str, dataset_hub: Any, operator_dataset_sentinel: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The old fake-runner journey reviewed another repo and allowed /user/uploads.

    This owns the consumer boundary: actual compiled bundle, real native host tools,
    trusted provenance confinement, truthful partial evidence and filesystem installs.
    """
    case, repo, immutable = _compiled_case(tmp_path, fake_gh)
    models = json.loads((native_provider.config / "models.json").read_text())
    models["providers"] = {"openrouter": models["providers"]["output-fixture"]}
    (native_provider.config / "models.json").write_text(json.dumps(models))
    environment = _environment(case, repo, tmp_path, native_provider.config)
    profile = tmp_path / "candidate.toml"
    profile.write_text('schema_version = 1\nname = "operator-harbor"\n[strategies."discovery.generic_fallback"]\n'
                       'content = "HARBOR_OPERATOR_DISCOVERY"\nsource = "copied: isolation"\n'
                       '[strategies.arbitration]\ncontent = "HARBOR_OPERATOR_CHALLENGE"\n'
                       'source = "copied: isolation"\n')
    environment["DAYDREAM_REVIEW_PROFILE_CANDIDATE"] = str(profile)
    if outcome == "model-spelling":
        environment["DAYDREAM_REVIEW_MODEL"] = " output-model "
    elif outcome == "model-missing":
        environment.pop("DAYDREAM_REVIEW_MODEL")
    elif outcome in {"model-empty", "model-blank"}:
        environment["DAYDREAM_REVIEW_MODEL"] = "" if outcome == "model-empty" else " \t "
    forbidden_credentials = {
        "GH_TOKEN": "forbidden-gh", "DAYDREAM_JUDGE_API_KEY": "forbidden-judge", "HF_TOKEN": "forbidden-hub",
    }
    environment.update(forbidden_credentials)
    original_environment = dict(environment)
    parent_before = dict(os.environ)
    baseline_runs = {path.name: path.read_bytes() for path in (operator_dataset_sentinel / "runs").glob("*.jsonl")}

    def choose(request: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        response = answer_review(request, empty=outcome == "clean", partial=outcome == "partial")
        for _, tool, arguments in response:
            if tool == "structured_output" and arguments.get("claims"):
                if outcome == "noninline":
                    arguments["claims"][0]["payload"]["line"] = None
                elif outcome in {"long-title", "blank-title"}:
                    arguments["claims"][0]["title"] = "T" * 4096 if outcome == "long-title" else " \t\n"
                elif outcome == "oversized-body":
                    arguments["claims"][0]["title"] = "T" * 4096
                    arguments["claims"][0]["payload"].update({
                        key: character * 4096
                        for key, character in zip(("claim", "trigger", "consequence", "grounds"), "界果因証")
                    })
        return response

    native_provider.choose = choose
    native_credential_checks: list[dict[str, bool]] = []
    if outcome == "finding":
        real_spawn = asyncio.create_subprocess_exec

        async def observe_native_environment(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
            if args and Path(str(args[0])).name == "pi" and args[1:3] == ("--mode", "json"):
                child = kwargs.get("env") or {}
                native_credential_checks.append({
                    **{name: name not in child and secret not in child.values()
                       for name, secret in forbidden_credentials.items()},
                    "selected_reviewer": child.get("OPENROUTER_API_KEY") == "synthetic-loopback-only",
                })
            return await real_spawn(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", observe_native_environment)
    artifact = tmp_path / "review.json"
    artifact.write_bytes(b"prior candidate")
    if outcome == "bundle-tamper":
        with (case / "environment/repository.bundle").open("ab") as stream:
            stream.write(b"tampered")
    real_replace = os.replace

    def fail_install(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        destination = Path(dst)
        if (outcome == "candidate-fault" and destination == artifact
            or outcome == "trajectory-fault" and destination == tmp_path / "logs/trajectory.json"
            or outcome == "output-fault" and destination.name == "committed-review.json"
        ):
            raise OSError("injected filesystem installation failure")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_install)
    with forbid_github() as attempts:
        rc = await entrypoint.main(environment)
    assert rc == (0 if outcome in {"finding", "clean", "long-title", "blank-title"} else 1)
    assert attempts == []
    assert environment == original_environment and dict(os.environ) == parent_before
    assert git(repo, "remote") == ""
    assert not dataset_hub.commits
    retained_runs = {path.name: path.read_bytes() for path in (operator_dataset_sentinel / "runs").glob("*.jsonl")}
    assert retained_runs == baseline_runs
    assert not (repo / ".daydream").exists() and not (repo / ".review-output.md").exists()
    if outcome in {"bundle-tamper", "model-spelling", "model-missing", "model-empty", "model-blank"}:
        assert native_provider.requests == []
        assert artifact.read_bytes() == b"prior candidate"
        if outcome in {"model-missing", "model-empty", "model-blank"}:
            assert not (tmp_path / "logs").exists()
            assert "DAYDREAM_REVIEW_MODEL" in capsys.readouterr().err
        return
    assignments = [assignment_from_request(request) for request in native_provider.requests]
    if outcome == "finding":
        assert native_credential_checks
        assert all(all(check.values()) for check in native_credential_checks), native_credential_checks
        assert native_provider.headers
        assert all(not {"http-referer", "x-openrouter-title", "x-openrouter-categories"} & headers.keys()
                   for headers in native_provider.headers)
    assert all(assignment["profile"]["profile_name"] == "operator-harbor" for assignment in assignments)
    for assignment in assignments:
        strategy = "HARBOR_OPERATOR_" + assignment["kind"].upper()
        assert assignment["profile"]["strategies"][assignment["kind"]] == strategy
    assigned = json.dumps(assignments)
    prompts = json.dumps(native_provider.requests)
    assert immutable["provenance"]["head_sha"] not in prompts and '"pr_number"' not in assigned
    assert '"repository": "o/r"' not in assigned and '"pr_repo": "o/r"' not in assigned
    assert "Fix cache" in prompts
    assert all(secret not in prompts for secret in forbidden_credentials.values())
    if outcome in {"finding", "clean", "partial", "noninline", "candidate-fault", "trajectory-fault",
                   "long-title", "blank-title", "oversized-body"}:
        result_path, = (tmp_path / "logs").glob("committed-*/committed-review.json")
        result = json.loads(result_path.read_text())
        assert result["immutable_input"] == immutable
        assert result["analyzed_revision"]["head_sha"] == immutable["materialization"]["head_sha"]
        trajectory_path, = (tmp_path / "logs").glob("committed-*/trajectory.json")
        trajectory = json.loads(trajectory_path.read_text())
        assert trajectory["extra"]["pr_number"] == 101 and trajectory["extra"]["pr_repo"] == "o/r"
        assert trajectory["extra"]["immutable_input"] == immutable
        assert len(result["claims"]) == (0 if outcome == "clean" else 1)
        if outcome == "partial":
            assert result["terminal_result"]["analysis_state"] == "incomplete"
            assert result["claims"][0]["disposition"] == "unresolved"
        elif outcome == "noninline":
            assert result["terminal_result"]["analysis_state"] == "complete"
            assert result["findings"][0]["payload"]["line"] is None
        elif outcome == "oversized-body":
            assert "cannot project committed finding: body exceeds 8 KiB" in capsys.readouterr().err
            assert result["terminal_result"]["analysis_state"] == "complete"
            assert result["findings"][0]["title"] == "T" * 4096
            for key, character in zip(("claim", "trigger", "consequence", "grounds"), "界果因証"):
                assert result["findings"][0]["payload"][key] == character * 4096
    if rc:
        assert artifact.read_bytes() == b"prior candidate"
    else:
        published = json.loads(artifact.read_text())
        parsed = verifier_core.validate_candidate_artifact(published)
        assert len(parsed) == (0 if outcome == "clean" else 1)
        assert (tmp_path / "logs/trajectory.json").is_file()
        if parsed:
            assert parsed[0].path == "feature.py" and parsed[0].start_line == 2
            if outcome == "long-title":
                assert len(parsed[0].title) == 500 and "T" * 4096 in parsed[0].body
                assert result["findings"][0]["title"] == "T" * 4096
            elif outcome == "blank-title":
                assert parsed[0].title == result["findings"][0]["payload"]["claim"]
                assert result["findings"][0]["title"] == " \t\n"


async def test_concurrent_harbor_reviews_keep_execution_and_artifacts_private(
    tmp_path: Path, fake_gh: FakeGh, native_provider: _NativeProvider,
) -> None:
    """Distinct operator contexts and sessions must survive concurrent native calls."""
    case, first, _ = _compiled_case(tmp_path, fake_gh)
    second = tmp_path / "second-repo"
    shutil.copytree(first, second, symlinks=True)
    model_config = json.loads((native_provider.config / "models.json").read_text())
    model_config["providers"] = {"openrouter": model_config["providers"]["output-fixture"]}
    environments: list[dict[str, str]] = []
    for name, repo in (("first", first), ("second", second)):
        root = tmp_path / name
        root.mkdir()
        config = root / "native"
        config.mkdir()
        (config / "models.json").write_text(json.dumps(model_config))
        profile = root / "profile.toml"
        profile.write_text(f'schema_version = 1\nname = "{name}-candidate"\n')
        environment = _environment(case, repo, root, config)
        environment["DAYDREAM_REVIEW_PROFILE_CANDIDATE"] = str(profile)
        environments.append(environment)
    native_provider.choose = answer_review
    before = dict(os.environ)
    assert await asyncio.gather(*(entrypoint.main(environment) for environment in environments)) == [0, 0]
    assert dict(os.environ) == before
    sessions: set[str] = set()
    for name in ("first", "second"):
        result_path, = (tmp_path / name / "logs").glob("committed-*/committed-review.json")
        value = json.loads(result_path.read_text())
        sessions.add(value["run_id"])
        assert value["settings"]["profile_name"] == f"{name}-candidate"
        assert (tmp_path / name / "review.json").is_file()
    assert len(sessions) == 2
    assert {assignment_from_request(request)["profile"]["profile_name"] for request in native_provider.requests} == {
        "first-candidate", "second-candidate"}
