"""The actual frozen Harbor bundle enters the shared offline native review API."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.harbor import build, candidate, entrypoint, package as pkg
from daydream.committed_review_model import CommittedReviewResult
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


def _assert_compiled_case_is_file_level_isolated(case: Path, repo: Path, wheel_name: str) -> None:
    """Prove at file level what the retired Docker image checks observed at runtime.

    /workspace/repo is exactly the bundle's head tree, so the bundle refs and trees
    must hold only source; each image sees only its own exact build context.
    """
    heads = git(repo, "bundle", "list-heads", str(case / "environment/repository.bundle")).splitlines()
    assert sorted(line.split()[1] for line in heads) == ["refs/heads/base", "refs/heads/head"]
    trees = {ref: set(git(repo, "ls-tree", "-r", "--name-only", ref).splitlines()) for ref in ("base", "head")}
    assert trees == {"base": {"base.py", "readme.txt"}, "head": {"base.py", "feature.py", "readme.txt"}}
    assert {path.name for path in (case / "environment").iterdir()} == {
        "Dockerfile", "immutable-request.json", "repository.bundle", "runtime-requirements.lock", wheel_name}
    verifier_files = {"golden-review.json", "judge_prompt.md", "score_review.py", "test.sh",
                      "verifier-metadata.json", "verifier_core.py"}
    assert {path.name for path in (case / "tests").iterdir()} == {"Dockerfile", *verifier_files}
    environment = (case / "environment/Dockerfile").read_text()
    for forbidden in ("Task.md", "instruction.md", "solution/", "tests/score_review", "tests/test.sh", "COPY ..",
                      "DAYDREAM_REVIEW"):
        assert forbidden not in environment
    verifier = (case / "tests/Dockerfile").read_text()
    for forbidden in ("repository.bundle", "instruction.md", "/workspace", "DAYDREAM_REVIEW", "ENV ", "COPY .."):
        assert forbidden not in verifier
    copied, = [line.split()[1:-1] for line in verifier.splitlines() if line.startswith("COPY ")]
    assert set(copied) == verifier_files
    # Entrypoint defaults must name the paths this image and verifier actually use.
    parsed = entrypoint.parse_reviewer_environment({
        "PATH": os.environ["PATH"], "DAYDREAM_REVIEW_BACKEND": "pi", "DAYDREAM_REVIEW_MODEL": "m",
        "DAYDREAM_REVIEW_API_KEY": "k", "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api/v1",
        "DAYDREAM_REVIEW_CASE_ID": case.name,
    })
    assert parsed.repo_dir == Path("/workspace/repo")
    assert f"/tmp/repository.bundle {parsed.repo_dir}" in environment and f"WORKDIR {parsed.repo_dir}" in environment
    assert f"COPY immutable-request.json {parsed.immutable_path}" in environment
    assert f"mv /tmp/repository.bundle {parsed.bundle_path}" in environment
    assert str(parsed.artifact_path) in (case / "tests/test.sh").read_text()
    assert parsed.trajectory_path == Path("/logs/agent/trajectory.json")


def _assert_committed_projection_bounds(result: dict[str, Any], immutable: dict[str, Any], case_id: str) -> None:
    """Project the sealed result: bound titles keep full text in the body; unprojectable findings fail typed."""
    def project(mutate: Any = None, **facts: Any) -> dict[str, Any]:
        value = json.loads(json.dumps(result))
        if mutate is not None:
            # Findings are exactly the confirmed claims; change both representations together.
            mutate(value["findings"][0])
            mutate(value["claims"][0])
        sealed = CommittedReviewResult(json_text=json.dumps(value), output_installed=True,
                                       capture_state=facts.get("capture_state", "complete"))
        return candidate.build_committed_candidate(sealed, immutable_input=immutable, case_id=case_id)

    finding, = project()["findings"]
    assert finding["path"] == "feature.py" and finding["start_line"] == 2
    long_title, = project(lambda item: item.update(title="T" * 4096))["findings"]
    assert len(long_title["title"]) == 500 and "T" * 4096 in long_title["body"]
    blank_title, = project(lambda item: item.update(title=" \t\n"))["findings"]
    assert blank_title["title"] == result["findings"][0]["payload"]["claim"].strip().splitlines()[0][:500]
    failures: list[tuple[Any, dict[str, Any], str]] = [
        (None, {"capture_state": "partial"}, "incomplete"),
        (lambda item: item["payload"].update(line=None), {}, "unsupported_location"),
        (lambda item: item["payload"].update(claim="界" * 4096, grounds="証" * 4096), {}, "invalid_finding"),
    ]
    for mutate, facts, kind in failures:
        with pytest.raises(candidate.CandidateError) as rejected:
            project(mutate, **facts)
        assert rejected.value.kind == kind


async def test_compiled_harbor_review_is_offline_source_bound_and_failure_atomic(
    tmp_path: Path, fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider,
    dataset_hub: Any, operator_dataset_sentinel: Path,
) -> None:
    """The actual compiled bundle enters the native review offline; failed installs and tampering publish nothing.

    This owns the consumer boundary: actual compiled bundle, real native host tools,
    trusted provenance confinement, truthful partial evidence and filesystem installs.
    """
    case, repo, immutable = _compiled_case(tmp_path, fake_gh)
    _assert_compiled_case_is_file_level_isolated(
        case, repo, f"daydream-{importlib.metadata.version('daydream')}-py3-none-any.whl")
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
    forbidden_credentials = {
        "GH_TOKEN": "forbidden-gh", "DAYDREAM_JUDGE_API_KEY": "forbidden-judge", "HF_TOKEN": "forbidden-hub",
    }
    environment.update(forbidden_credentials)
    original_environment = dict(environment)
    parent_before = dict(os.environ)
    baseline_runs = {path.name: path.read_bytes() for path in (operator_dataset_sentinel / "runs").glob("*.jsonl")}
    native_provider.choose = answer_review
    artifact = tmp_path / "review.json"
    artifact.write_bytes(b"prior candidate")
    real_replace = os.replace

    def fail_install(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(dst) == artifact:
            raise OSError("injected filesystem installation failure")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_install)
    with forbid_github() as attempts:
        assert await entrypoint.main(environment) == 1
    assert attempts == []
    assert environment == original_environment and dict(os.environ) == parent_before
    assert git(repo, "remote") == ""
    assert not dataset_hub.commits
    retained_runs = {path.name: path.read_bytes() for path in (operator_dataset_sentinel / "runs").glob("*.jsonl")}
    assert retained_runs == baseline_runs
    assert not (repo / ".daydream").exists() and not (repo / ".review-output.md").exists()
    assignments = [assignment_from_request(request) for request in native_provider.requests]
    for assignment in assignments:
        assert assignment["profile"]["profile_name"] == "operator-harbor"
        strategy = "HARBOR_OPERATOR_" + assignment["kind"].upper()
        assert assignment["profile"]["strategies"][assignment["kind"]] == strategy
    assigned = json.dumps(assignments)
    prompts = json.dumps(native_provider.requests)
    assert immutable["provenance"]["head_sha"] not in prompts and '"pr_number"' not in assigned
    assert '"repository": "o/r"' not in assigned and '"pr_repo": "o/r"' not in assigned
    assert "Fix cache" in prompts
    assert all(secret not in prompts for secret in forbidden_credentials.values())
    result_path, = (tmp_path / "logs").glob("committed-*/committed-review.json")
    result = json.loads(result_path.read_text())
    assert result["immutable_input"] == immutable
    assert result["analyzed_revision"]["head_sha"] == immutable["materialization"]["head_sha"]
    assert len(result["claims"]) == 1
    trajectory_path, = (tmp_path / "logs").glob("committed-*/trajectory.json")
    trajectory = json.loads(trajectory_path.read_text())
    assert trajectory["extra"]["pr_number"] == 101 and trajectory["extra"]["pr_repo"] == "o/r"
    assert trajectory["extra"]["immutable_input"] == immutable
    assert artifact.read_bytes() == b"prior candidate"
    _assert_committed_projection_bounds(result, immutable, case.name)

    # A tampered bundle is refused before any inference and preserves the prior candidate.
    requests = len(native_provider.requests)
    with (case / "environment/repository.bundle").open("ab") as stream:
        stream.write(b"tampered")
    with forbid_github() as attempts:
        assert await entrypoint.main(environment) == 1
    assert attempts == [] and len(native_provider.requests) == requests
    assert artifact.read_bytes() == b"prior candidate"
