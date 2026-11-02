"""CI policy at its CLI boundary: real commits/merges, external GitHub stubbed."""

from __future__ import annotations

import copy
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
_ACTIVE_RULES: list[dict[str, Any]] = [{
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


def test_ci_cli_pr_gate_is_basic_only_under_a_verified_bypass_free_queue(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]],
) -> None:
    base = _git(git_repo, "rev-parse", "HEAD")
    head = _commit(git_repo, "tests/test_contract.py")
    # A ruleset is verified at its own origin, whichever level declares it.
    for source_type, source in (("Repository", _REPO), ("Organization", "acme"), ("Enterprise", "acme-enterprise")):
        verified: list[dict[str, Any]] = copy.deepcopy(_ACTIVE_RULES)
        verified[0].update(ruleset_source=source, ruleset_source_type=source_type)
        origin = {**_QUEUE_RULESET, "source": source, "source_type": source_type}
        output, _ = ci_cli(_pr(base, head), replies={_RULES: verified, _RULESET: origin})
        assert "mode=basic\n" in output and "full=false\n" in output, source_type

    def rules_fault(fault: str, rules: Any, event: dict[str, Any]) -> Any:
        if fault == "no-queue":
            return []
        if fault == "api":
            return subprocess.CalledProcessError(1, "gh")
        if fault == "malformed":
            return {"type": "merge_queue"}
        if fault == "missing-check":
            return [{"type": "merge_queue"}]
        if fault == "wrong-app":
            rules[1]["parameters"]["required_status_checks"][0]["integration_id"] = 999
        elif fault == "wrong-repository":
            event["pull_request"]["base"]["repo"]["full_name"] = "other/repo"
        elif fault == "wrong-branch":
            event["pull_request"]["base"]["ref"] = "release"
        elif fault == "missing-origin":
            del rules[0]["ruleset_source"]
        elif fault == "invalid-id":
            rules[0]["ruleset_id"] = True
        return rules

    def ruleset_fault(fault: str, ruleset: dict[str, Any], replies: dict[str, Any]) -> None:
        if fault == "always":
            ruleset["bypass_actors"] = [{"actor_id": 7, "actor_type": "Team", "bypass_mode": fault}]
        elif fault == "omitted":
            del ruleset["bypass_actors"]
        elif fault == "null":
            ruleset["bypass_actors"] = None
        elif fault == "inaccessible":
            replies[_RULESET] = subprocess.CalledProcessError(1, "gh")
        elif fault == "malformed-ruleset":
            replies[_RULESET] = []
        elif fault in {"wrong-id", "float-id"}:
            ruleset["id"] = 99 if fault == "wrong-id" else 42.0
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

    # Every unverifiable queue or bypassable ruleset keeps the complete PR gate.
    for fault in (
        "no-queue", "api", "malformed", "missing-check", "wrong-app", "wrong-repository", "wrong-branch",
        "missing-origin", "invalid-id", "always", "omitted", "null", "inaccessible", "malformed-ruleset",
        "wrong-id", "float-id", "wrong-source", "wrong-source-type", "wrong-target", "inactive", "missing-queue",
        "null-source", "empty-source", "null-source-type", "unknown-source-type", "other-queue-bypass",
    ):
        event = _pr(base, head)
        rules: list[dict[str, Any]] = copy.deepcopy(_ACTIVE_RULES)
        ruleset: dict[str, Any] = copy.deepcopy(_QUEUE_RULESET)
        replies: dict[str, Any] = {_RULESET: ruleset}
        replies[_RULES] = rules_fault(fault, rules, event)
        ruleset_fault(fault, ruleset, replies)
        if fault in {"null-source", "empty-source"}:
            rules[0]["ruleset_source"] = ruleset["source"] = None if fault == "null-source" else ""
        elif fault in {"null-source-type", "unknown-source-type"}:
            rules[0]["ruleset_source_type"] = ruleset["source_type"] = (
                None if fault == "null-source-type" else "Unknown"
            )
        elif fault == "other-queue-bypass":
            rules.append({**rules[0], "ruleset_id": 99})
            replies[f"repos/{_REPO}/rulesets/99"] = {
                **ruleset, "id": 99, "bypass_actors": [{"actor_id": 7, "actor_type": "Team", "bypass_mode": "always"}],
            }
        output, _ = ci_cli(event, replies=replies)
        assert "mode=full\n" in output and "full=true\n" in output, fault


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


def _push(repo: Path, before: str, replies: dict[str, Any]) -> dict[str, Any]:
    sha = _git(repo, "rev-parse", "HEAD")
    replies[f"repos/{_REPO}/commits/{sha}/pulls"] = [{
        "number": 7, "merged_at": "2026-10-14T12:00:00Z", "merge_commit_sha": sha,
        "base": {"ref": "main", "repo": {"full_name": _REPO}},
    }]
    return {"ref": "refs/heads/main", "before": before, "after": sha}


def test_ci_cli_reuses_last_verified_queue_tree_from_run_inventory(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path,
) -> None:
    base, old_proof, replies = _successful_merge_proof(git_repo, ci_cli, tmp_path)
    tested = _commit(git_repo, "daydream/base_change.py")
    output, _ = ci_cli(_group(base, tested), event_name="merge_group")
    assert "mode=full\n" in output and "full=true\n" in output
    proof_dir = tmp_path / "proof"
    ci_cli(_group(base, tested), event_name="merge_group", record=proof_dir)
    proof = json.loads((proof_dir / "proof.json").read_text())
    _git(git_repo, "reset", "--soft", base)
    _git(git_repo, "-c", "commit.gpgsign=false", "commit", "-m", "queue squash result")
    assert _git(git_repo, "rev-parse", "HEAD") != proof["commit"]
    assert _git(git_repo, "rev-parse", "HEAD^{tree}") == proof["tree"]
    candidates: list[tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]] = [
        ({"head_sha": old_proof["commit"]}, {}, old_proof),
        ({}, {}, {"head_sha": "f" * 40}),
        ({}, {}, {"repository": "other/repo"}),
        ({}, {}, {"run_id": 999}),
        ({}, {}, {"commit": "f" * 40}),
        ({}, {}, {"base_tree": "f" * 40}),
        ({"head_branch": _QUEUE_REF.replace("/main/", "/master/").removeprefix("refs/heads/")}, {},
         {"base_ref": "refs/heads/master", "head_ref": _QUEUE_REF.replace("/main/", "/master/")}),
        ({"repository": {"full_name": "other/repo"}}, {}, {}),
        ({"head_branch": "feature"}, {}, {}),
        ({"path": ".github/workflows/other.yml"}, {}, {}),
        ({}, {}, {"event": "pull_request"}),
        ({}, {}, {"gate": "basic"}),
        ({}, {"expired": True}, {}),
        ({"conclusion": "failure"}, {}, {}),
        ({"conclusion": "cancelled"}, {}, {}),
        ({}, None, {}),
        ({}, {"size_in_bytes": 4097}, {}),
        ({}, {}, {}),
    ]
    run_template = replies[_RUNS]["workflow_runs"].pop()
    artifact_template = replies[_ARTIFACTS]["artifacts"][0]
    for run_id, (run_fields, artifact_fields, proof_fields) in enumerate(candidates, 101):
        run = {**run_template, "id": run_id, "head_sha": tested, **run_fields}
        run["html_url"] = f"https://github.com/{_REPO}/actions/runs/{run_id}"
        replies[_RUNS]["workflow_runs"].append(run)
        artifact = {**artifact_template, "id": run_id, **(artifact_fields or {})}
        replies[f"repos/{_REPO}/actions/runs/{run_id}/artifacts"] = {
            "artifacts": [artifact] if artifact_fields is not None else [],
        }
        replies[f"repos/{_REPO}/actions/artifacts/{run_id}/zip"] = _archive({
            **proof, "run_id": run_id, **proof_fields,
        })
    verified_run = replies[_RUNS]["workflow_runs"].pop()
    event = _push(git_repo, base, replies)
    output, summary = ci_cli(event, event_name="push", replies=replies)
    assert "full=true\n" in output and "CI policy: **full**. Run the complete root gate." in summary
    replies[_RUNS]["workflow_runs"].append(verified_run)
    output, summary = ci_cli(event, event_name="push", replies=replies)
    assert "mode=reuse\n" in output and "full=false\n" in output
    assert "https://github.com/acme/widgets/actions/runs/118" in summary


def test_ci_cli_record_rejects_non_queue_or_unverified_checkouts(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path,
) -> None:
    base, proof, _ = _successful_merge_proof(git_repo, ci_cli, tmp_path)
    # "dirty" mutates the checkout, so it runs last.
    for fault in ("event", "action", "repo", "head", "checkout", "base", "ref", "dirty"):
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
        else:
            event["merge_group"]["head_ref"] = "refs/heads/feature"
        dest = tmp_path / f"rejected-{fault}"
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            ci_cli(event, event_name=name, github_sha=github_sha, record=dest)
        assert not (dest / "proof.json").exists(), fault


def test_ci_cli_push_without_proven_authority_or_readable_proof_runs_full_gate(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path,
) -> None:
    sha = _git(git_repo, "rev-parse", "HEAD")
    # Direct pushes and manual dispatches have no queue authority at all.
    for event_name in ("workflow_dispatch", "push"):
        output, _ = ci_cli({"ref": "refs/heads/main", "before": sha, "after": sha}, event_name=event_name,
                           replies={f"repos/{_REPO}/commits/{sha}/pulls": []})
        assert "mode=full\n" in output and "full=true\n" in output, event_name

    base, proof, template = _successful_merge_proof(git_repo, ci_cli, tmp_path)
    template[_ARCHIVE] = _archive(proof)
    tip = _git(git_repo, "rev-parse", "HEAD")
    for fault in (
        "api-error", "bad-zip", "base-ancestry", "unmerged", "merge-sha", "base-repository", "base-branch",
        "push-after", "checkout", "unrelated-before", "unresolvable-before",
    ):
        replies = copy.deepcopy(template)
        event = _push(git_repo, base, replies)
        github_sha = event["after"]
        associated = replies[f"repos/{_REPO}/commits/{github_sha}/pulls"][0]
        if fault == "api-error":
            replies[_RUNS] = subprocess.CalledProcessError(1, "gh")
        elif fault == "bad-zip":
            replies[_ARCHIVE] = b"invalid"
        elif fault == "base-ancestry":
            _git(git_repo, "checkout", "--orphan", "unrelated")
            foreign = {**proof, "base_sha": _commit(git_repo, "unrelated.txt"),
                       "base_tree": _git(git_repo, "rev-parse", "HEAD^{tree}")}
            _git(git_repo, "checkout", "main")
            replies[_ARCHIVE] = _archive(foreign)
        elif fault == "unmerged":
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
        else:
            event["before"] = "f" * 40
        assert _git(git_repo, "rev-parse", "HEAD") == tip
        output, summary = ci_cli(event, event_name="push", github_sha=github_sha, replies=replies)
        assert "mode=full\n" in output and "full=true\n" in output, fault
        if fault == "bad-zip":
            assert "BadZipFile" in summary
        elif fault in {"api-error", "base-ancestry", "unrelated-before", "unresolvable-before"}:
            assert "CalledProcessError" in summary, fault
