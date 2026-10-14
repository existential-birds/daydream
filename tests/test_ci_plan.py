"""CI policy at its CLI boundary: real commits/merges, external GitHub stubbed."""

from __future__ import annotations

import io
import json
import runpy
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_plan.py"
_REPO = "acme/widgets"
_RULES = f"repos/{_REPO}/rules/branches/main"
_RULESET = f"repos/{_REPO}/rulesets/42"
_RUNS = f"repos/{_REPO}/actions/workflows/ci.yml/runs"
_ARTIFACTS = f"repos/{_REPO}/actions/runs/101/artifacts"
_ARCHIVE = f"repos/{_REPO}/actions/artifacts/202/zip"
_ACTIVE_RULES = [{
    "type": "merge_queue", "ruleset_id": 42, "ruleset_source": _REPO, "ruleset_source_type": "Repository",
}, {"type": "required_status_checks", "parameters": {
    "required_status_checks": [{"context": "check", "integration_id": 15368}],
}}]
_QUEUE_RULESET = {
    "id": 42, "source": _REPO, "source_type": "Repository", "target": "branch", "enforcement": "active",
    "rules": [{"type": "merge_queue"}], "bypass_actors": [],
}
_QUEUE_REF = "refs/heads/gh-readonly-queue/main/pr-7-queue"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit(repo: Path, path: str) -> str:
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(file.read_text() + "change\n" if file.exists() else "change\n")
    _git(repo, "add", path)
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "fixture change")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def ci_cli(git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., tuple[str, str]]:
    namespace = runpy.run_path(str(_SCRIPT))
    responses: dict[str, Any] = {}
    event_path = tmp_path / "event.json"
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("GITHUB_REPOSITORY", _REPO)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_RUN_ID", "101")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    def api(endpoint: str, *, binary: bool = False) -> Any:
        url = urlsplit(endpoint)
        if url.path == _RUNS:
            # Exercise the actual query too: successful queue evidence only.
            assert parse_qs(url.query)["event"] == ["merge_group"]
            assert parse_qs(url.query)["status"] == ["success"]
        if url.path.startswith(f"repos/{_REPO}/rulesets/"):
            assert parse_qs(url.query)["includes_parents"] == ["true"]
        value = responses[url.path]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setitem(namespace["main"].__globals__, "api", api)

    def invoke(
        event: dict[str, Any], *, event_name: str = "pull_request", replies: dict[str, Any] | None = None,
        record: Path | None = None, github_sha: str | None = None,
    ) -> tuple[str, str]:
        event_path.write_text(json.dumps(event))
        responses.clear()
        responses.update(replies or {})
        monkeypatch.setenv("GITHUB_EVENT_NAME", event_name)
        monkeypatch.setenv("GITHUB_SHA", github_sha or _git(git_repo, "rev-parse", "HEAD"))
        monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "record", "--proof-dir", str(record)] if record else
                            [str(_SCRIPT), "plan"])
        output.write_text("")
        summary.write_text("")
        namespace["main"]()
        return output.read_text(), summary.read_text()

    return invoke


def _pr(base: str, head: str) -> dict[str, Any]:
    return {"number": 7, "pull_request": {
        "base": {"sha": base, "ref": "main", "repo": {"full_name": _REPO}},
        "head": {"sha": head, "ref": "feature"},
    }}


def _group(base: str, head: str) -> dict[str, Any]:
    return {"action": "checks_requested", "repository": {"full_name": _REPO}, "merge_group": {
        "base_ref": "refs/heads/main", "base_sha": base, "head_ref": _QUEUE_REF, "head_sha": head,
    }}


@pytest.mark.parametrize("path", [
    "docs/guide.md", "README.md", "rl/daydream_review/uv.lock", "daydream/prompts/review.md",
    "daydream/templates/workflows/README.md", "uv.lock", "tests/test_contract.py", ".github/workflows/ci.yml",
])
@pytest.mark.parametrize("queue", [False, True])
def test_ci_cli_pr_checks_require_verified_active_queue(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], path: str, queue: bool,
) -> None:
    base = _git(git_repo, "rev-parse", "HEAD")
    head = _commit(git_repo, path)
    output, _ = ci_cli(_pr(base, head), replies={
        _RULES: _ACTIVE_RULES if queue else [], _RULESET: _QUEUE_RULESET,
    })
    assert f"mode={'basic' if queue else 'full'}\n" in output
    assert f"full={'false' if queue else 'true'}\n" in output


@pytest.mark.parametrize("fault", [
    "api", "malformed", "missing-check", "wrong-app", "wrong-repository", "wrong-branch",
])
def test_ci_cli_queue_uncertainty_keeps_full_pr_gate(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], fault: str,
) -> None:
    sha = _git(git_repo, "rev-parse", "HEAD")
    event = _pr(sha, sha)
    rules: Any = json.loads(json.dumps(_ACTIVE_RULES))
    if fault == "api":
        rules = subprocess.CalledProcessError(1, "gh")
    elif fault == "malformed":
        rules = {"type": "merge_queue"}
    elif fault == "missing-check":
        rules = [{"type": "merge_queue"}]
    elif fault == "wrong-app":
        rules[1]["parameters"]["required_status_checks"][0]["integration_id"] = 999
    elif fault == "wrong-repository":
        event["pull_request"]["base"]["repo"]["full_name"] = "other/repo"
    else:
        event["pull_request"]["base"]["ref"] = "release"
    output, _ = ci_cli(event, replies={_RULES: rules, _RULESET: _QUEUE_RULESET})
    assert "mode=full\n" in output and "full=true\n" in output


@pytest.mark.parametrize("source_type,source", [
    ("Repository", _REPO), ("Organization", "acme"), ("Enterprise", "acme-enterprise"),
])
def test_ci_cli_verifies_queue_ruleset_at_its_origin(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], source_type: str, source: str,
) -> None:
    rules = json.loads(json.dumps(_ACTIVE_RULES))
    rules[0].update(ruleset_source=source, ruleset_source_type=source_type)
    ruleset = {**_QUEUE_RULESET, "source": source, "source_type": source_type}
    sha = _git(git_repo, "rev-parse", "HEAD")
    output, _ = ci_cli(_pr(sha, sha), replies={_RULES: rules, _RULESET: ruleset})
    assert "mode=basic\n" in output and "full=false\n" in output


@pytest.mark.parametrize("fault", [
    "always", "pull_request", "exempt", "omitted", "null", "object", "string", "inaccessible",
    "malformed", "wrong-id", "wrong-source", "wrong-source-type", "wrong-target", "inactive",
    "missing-queue", "missing-origin", "invalid-id", "other-queue-bypass", "float-id",
    "null-source", "empty-source", "null-source-type", "unknown-source-type",
])
def test_ci_cli_queue_bypass_or_unverified_ruleset_keeps_full_pr_gate(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], fault: str,
) -> None:
    rules = json.loads(json.dumps(_ACTIVE_RULES))
    ruleset: Any = json.loads(json.dumps(_QUEUE_RULESET))
    replies: dict[str, Any] = {_RULES: rules, _RULESET: ruleset}
    if fault in {"always", "pull_request", "exempt"}:
        ruleset["bypass_actors"] = [{"actor_id": 7, "actor_type": "Team", "bypass_mode": fault}]
    elif fault == "omitted":
        del ruleset["bypass_actors"]
    elif fault in {"null", "object", "string"}:
        ruleset["bypass_actors"] = {"null": None, "object": {}, "string": ""}[fault]
    elif fault == "inaccessible":
        replies[_RULESET] = subprocess.CalledProcessError(1, "gh")
    elif fault == "malformed":
        replies[_RULESET] = []
    elif fault == "wrong-id":
        ruleset["id"] = 99
    elif fault == "wrong-source":
        ruleset["source"] = "other/repo"
    elif fault == "wrong-source-type":
        ruleset["source_type"] = "Organization"
    elif fault == "wrong-target":
        ruleset["target"] = "tag"
    elif fault == "inactive":
        ruleset["enforcement"] = "evaluate"
    elif fault == "missing-queue":
        ruleset["rules"] = []
    elif fault == "missing-origin":
        del rules[0]["ruleset_source"]
    elif fault == "invalid-id":
        rules[0]["ruleset_id"] = True
    elif fault == "float-id":
        ruleset["id"] = 42.0
    elif fault in {"null-source", "empty-source"}:
        rules[0]["ruleset_source"] = ruleset["source"] = None if fault == "null-source" else ""
    elif fault in {"null-source-type", "unknown-source-type"}:
        rules[0]["ruleset_source_type"] = ruleset["source_type"] = (
            None if fault == "null-source-type" else "Unknown"
        )
    else:
        rules.append({**rules[0], "ruleset_id": 99})
        replies[f"repos/{_REPO}/rulesets/99"] = {
            **ruleset, "id": 99, "bypass_actors": [{"actor_id": 7, "actor_type": "Team", "bypass_mode": "always"}],
        }
    sha = _git(git_repo, "rev-parse", "HEAD")
    output, _ = ci_cli(_pr(sha, sha), replies=replies)
    assert "mode=full\n" in output and "full=true\n" in output


def _successful_merge_proof(
    repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "feature")
    _commit(repo, "daydream/app.py")
    _git(repo, "checkout", "main")
    _git(repo, "-c", "commit.gpgsign=false", "merge", "--no-ff", "feature", "-m", "tested queue merge")
    merge = _git(repo, "rev-parse", "HEAD")
    proof_dir = tmp_path / "proof"
    ci_cli(_group(base, merge), event_name="merge_group", record=proof_dir)
    proof = json.loads((proof_dir / "proof.json").read_text())
    assert proof["commit"] == proof["head_sha"] == merge
    assert proof["tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
    assert proof["base_tree"] == _git(repo, "rev-parse", f"{base}^{{tree}}")
    assert proof["event"] == "merge_group" and proof["gate"] == "full" and proof["schema"] == 2
    # Real local transport proves historical checkout object retrieval.
    _git(repo, "remote", "add", "origin", str(repo))
    replies: dict[str, Any] = {
        _RUNS: {"workflow_runs": [{
            "id": 101, "event": "merge_group", "conclusion": "success", "path": ".github/workflows/ci.yml",
            "repository": {"full_name": _REPO}, "head_sha": merge,
            "head_branch": _QUEUE_REF.removeprefix("refs/heads/"),
            "html_url": "https://github.com/acme/widgets/actions/runs/101",
        }]},
        _ARTIFACTS: {"artifacts": [{"id": 202, "name": "ci-proof", "expired": False, "size_in_bytes": 800}]},
    }
    return base, proof, replies


def _archive(proof: dict[str, Any]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("proof.json", json.dumps(proof))
    return stream.getvalue()


def _push_replies(repo: Path, replies: dict[str, Any]) -> None:
    sha = _git(repo, "rev-parse", "HEAD")
    replies[f"repos/{_REPO}/commits/{sha}/pulls"] = [{
        "number": 7, "merged_at": "2026-10-14T12:00:00Z", "merge_commit_sha": sha,
        "base": {"ref": "main", "repo": {"full_name": _REPO}},
    }]


def _push(repo: Path, before: str) -> dict[str, Any]:
    return {"ref": "refs/heads/main", "before": before, "after": _git(repo, "rev-parse", "HEAD")}


@pytest.mark.parametrize("fault", [
    "none", "tree", "head", "repository", "run", "commit", "base-tree", "base-ancestry", "changed-base",
    "cross-branch", "run-repo", "run-branch", "wrong-workflow", "legacy-pr", "basic-proof", "expired",
    "failed", "cancelled", "bad-zip", "api-error", "missing", "oversized",
])
def test_ci_cli_reuses_only_verified_exact_merge_group_trees(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path, fault: str,
) -> None:
    base, proof, replies = _successful_merge_proof(git_repo, ci_cli, tmp_path)
    run = replies[_RUNS]["workflow_runs"][0]
    artifact = replies[_ARTIFACTS]["artifacts"][0]
    if fault in {"tree", "head", "commit", "base-tree"}:
        proof[{"tree": "tree", "head": "head_sha", "commit": "commit", "base-tree": "base_tree"}[fault]] = "f" * 40
    elif fault == "run":
        proof["run_id"] = 999
    elif fault == "repository":
        proof["repository"] = "other/repo"
    elif fault == "expired":
        artifact["expired"] = True
    elif fault in {"failed", "cancelled"}:
        run["conclusion"] = "failure" if fault == "failed" else "cancelled"
    elif fault == "api-error":
        replies[_RUNS] = subprocess.CalledProcessError(1, "gh")
    elif fault == "base-ancestry":
        # A real resolvable base/tree that isn't in the tested history reaches
        # the ancestry guard, rather than failing SHA parsing or tree lookup.
        tested = _git(git_repo, "rev-parse", "HEAD")
        _git(git_repo, "checkout", "--orphan", "unrelated")
        proof["base_sha"] = _commit(git_repo, "unrelated.txt")
        proof["base_tree"] = _git(git_repo, "rev-parse", "HEAD^{tree}")
        _git(git_repo, "checkout", "main")
        assert _git(git_repo, "rev-parse", "HEAD") == tested
    elif fault == "changed-base":
        _commit(git_repo, "daydream/base_change.py")
    elif fault == "cross-branch":
        proof["base_ref"] = "refs/heads/master"
        proof["head_ref"] = _QUEUE_REF.replace("/main/", "/master/")
        run["head_branch"] = proof["head_ref"].removeprefix("refs/heads/")
    elif fault == "run-repo":
        run["repository"]["full_name"] = "other/repo"
    elif fault == "run-branch":
        run["head_branch"] = "feature"
    elif fault == "wrong-workflow":
        run["path"] = ".github/workflows/other.yml"
    elif fault == "legacy-pr":
        proof["event"] = "pull_request"
    elif fault == "basic-proof":
        proof["gate"] = "basic"
    elif fault == "missing":
        replies[_ARTIFACTS]["artifacts"] = []
    elif fault == "oversized":
        artifact["size_in_bytes"] = 4097
    replies[_ARCHIVE] = b"invalid" if fault == "bad-zip" else _archive(proof)
    _push_replies(git_repo, replies)
    output, summary = ci_cli(_push(git_repo, base), event_name="push", replies=replies)
    assert f"mode={'reuse' if fault == 'none' else 'full'}\n" in output
    if fault == "none":
        assert "actions/runs/101" in summary
    if fault == "base-ancestry":
        assert "CalledProcessError" in summary


def test_ci_cli_reuses_queue_tree_after_squash_commit_identity_changes(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path,
) -> None:
    base, proof, replies = _successful_merge_proof(git_repo, ci_cli, tmp_path)
    _git(git_repo, "reset", "--soft", base)
    _git(git_repo, "-c", "commit.gpgsign=false", "commit", "-m", "queue squash result")
    assert _git(git_repo, "rev-parse", "HEAD") != proof["commit"]
    assert _git(git_repo, "rev-parse", "HEAD^{tree}") == proof["tree"]
    replies[_ARCHIVE] = _archive(proof)
    _push_replies(git_repo, replies)
    output, _ = ci_cli(_push(git_repo, base), event_name="push", replies=replies)
    assert "mode=reuse\n" in output


@pytest.mark.parametrize("event_name", ["workflow_dispatch", "merge_group", "push"])
def test_ci_cli_direct_push_queue_and_manual_dispatch_run_full_gate(
    ci_cli: Callable[..., tuple[str, str]], git_repo: Path, event_name: str,
) -> None:
    sha = _git(git_repo, "rev-parse", "HEAD")
    replies: dict[str, Any] = {f"repos/{_REPO}/commits/{sha}/pulls": []}
    output, _ = ci_cli(_push(git_repo, sha), event_name=event_name, replies=replies)
    assert "mode=full\n" in output and "full=true\n" in output


@pytest.mark.parametrize("fault", ["event", "action", "repo", "head", "checkout", "dirty", "base", "ref"])
def test_ci_cli_record_rejects_non_queue_or_unverified_checkouts(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path, fault: str,
) -> None:
    base, proof, _ = _successful_merge_proof(git_repo, ci_cli, tmp_path)
    event = _group(base, proof["commit"])
    name = "merge_group"
    github_sha = proof["commit"]
    if fault == "event":
        name = "pull_request"
    elif fault == "action":
        event["action"] = "destroyed"
    elif fault == "repo":
        event["repository"]["full_name"] = "other/repo"
    elif fault == "head":
        event["merge_group"]["head_sha"] = "f" * 40
    elif fault == "checkout":
        github_sha = "f" * 40
    elif fault == "dirty":
        (git_repo / "daydream/app.py").write_text("unverified edit\n")
    elif fault == "base":
        event["merge_group"]["base_sha"] = proof["commit"]
    elif fault == "ref":
        event["merge_group"]["head_ref"] = "refs/heads/feature"
    dest = tmp_path / "rejected-proof"
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        ci_cli(event, event_name=name, github_sha=github_sha, record=dest)
    assert not (dest / "proof.json").exists()


@pytest.mark.parametrize("fault", [
    "unmerged", "merge-sha", "base-repository", "base-branch", "push-after", "checkout",
    "unrelated-before", "unresolvable-before",
])
def test_ci_cli_push_authority_and_old_tip_cannot_borrow_valid_queue_proof(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path, fault: str,
) -> None:
    base, proof, replies = _successful_merge_proof(git_repo, ci_cli, tmp_path)
    replies[_ARCHIVE] = _archive(proof)
    _push_replies(git_repo, replies)
    event = _push(git_repo, base)
    sha = event["after"]
    github_sha = sha
    associated = replies[f"repos/{_REPO}/commits/{sha}/pulls"][0]
    if fault == "unmerged":
        associated["merged_at"] = None
    elif fault == "merge-sha":
        associated["merge_commit_sha"] = "f" * 40
    elif fault == "base-repository":
        associated["base"]["repo"]["full_name"] = "other/repo"
    elif fault == "base-branch":
        associated["base"]["ref"] = "master"
    elif fault == "push-after":
        event["after"] = "f" * 40
    elif fault == "checkout":
        github_sha = event["after"] = "f" * 40
        # Every downstream API/proof guard remains satisfiable if HEAD identity
        # verification is removed. Missing API fixtures must not mask it.
        associated["merge_commit_sha"] = github_sha
        replies[f"repos/{_REPO}/commits/{github_sha}/pulls"] = [associated]
    elif fault == "unrelated-before":
        _git(git_repo, "checkout", "--orphan", "unrelated-before")
        event["before"] = _commit(git_repo, "unrelated.txt")
        _git(git_repo, "checkout", "main")
        assert _git(git_repo, "rev-parse", "HEAD") == proof["commit"]
    else:
        event["before"] = "f" * 40
    output, summary = ci_cli(event, event_name="push", github_sha=github_sha, replies=replies)
    assert "mode=full\n" in output and "full=true\n" in output
    if fault in {"unrelated-before", "unresolvable-before"}:
        assert "CalledProcessError" in summary


def test_ci_cli_valid_merge_group_requests_complete_gate(
    ci_cli: Callable[..., tuple[str, str]], git_repo: Path,
) -> None:
    base = _git(git_repo, "rev-parse", "HEAD")
    head = _commit(git_repo, "daydream/app.py")
    output, _ = ci_cli(_group(base, head), event_name="merge_group", replies={_RULES: _ACTIVE_RULES})
    assert "mode=full\n" in output and "full=true\n" in output
