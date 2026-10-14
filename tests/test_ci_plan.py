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

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_plan.py"
_REPO = "acme/widgets"


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
    monkeypatch.chdir(git_repo)
    monkeypatch.setenv("GITHUB_REPOSITORY", _REPO)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_RUN_ID", "101")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))

    def api(endpoint: str, *, binary: bool = False) -> Any:
        value = responses[endpoint.split("?")[0]]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setitem(namespace["main"].__globals__, "api", api)

    def invoke(
        event: dict[str, Any], *, event_name: str = "pull_request", replies: dict[str, Any] | None = None,
        record: Path | None = None,
    ) -> tuple[str, str]:
        event_path.write_text(json.dumps(event))
        responses.clear()
        responses.update(replies or {})
        monkeypatch.setenv("GITHUB_EVENT_NAME", event_name)
        monkeypatch.setenv("GITHUB_SHA", _git(git_repo, "rev-parse", "HEAD"))
        monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "record", "--proof-dir", str(record)] if record else
                            [str(_SCRIPT), "plan"])
        output.write_text("")
        namespace["main"]()
        return output.read_text(), (tmp_path / "summary").read_text() if (tmp_path / "summary").exists() else ""

    return invoke


def _pr(base: str, head: str) -> dict[str, Any]:
    return {"number": 7, "pull_request": {"base": {"sha": base}, "head": {"sha": head, "ref": "feature"}}}


@pytest.mark.parametrize("path,mode", [
    ("docs/guide.md", "lightweight"), ("README.md", "lightweight"),
    ("rl/daydream_review/uv.lock", "lightweight"), ("docs/example.py", "full"),
    ("daydream/prompts/review.md", "full"), ("daydream/templates/workflows/README.md", "full"),
    ("uv.lock", "full"), ("tests/test_contract.py", "full"), (".github/workflows/ci.yml", "full"),
])
def test_ci_cli_routes_changed_content(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], path: str, mode: str,
) -> None:
    base = _git(git_repo, "rev-parse", "HEAD")
    head = _commit(git_repo, path)
    output, _ = ci_cli(_pr(base, head))
    assert f"mode={mode}\n" in output
    assert f"full={'true' if mode == 'full' else 'false'}\n" in output


def _successful_merge_proof(
    repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path,
) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "feature")
    head = _commit(repo, "daydream/app.py")
    _git(repo, "checkout", "main")
    _git(repo, "-c", "commit.gpgsign=false", "merge", "--no-ff", "feature", "-m", "tested synthetic merge")
    merge = _git(repo, "rev-parse", "HEAD")
    proof_dir = tmp_path / "proof"
    ci_cli(_pr(base, head), record=proof_dir)
    proof = json.loads((proof_dir / "proof.json").read_text())
    assert proof["commit"] == merge and proof["head_sha"] == head
    assert proof["tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
    # Real local transport proves historical checkout object retrieval.
    _git(repo, "remote", "add", "origin", str(repo))
    replies: dict[str, Any] = {
        f"repos/{_REPO}/actions/workflows/ci.yml/runs": {"workflow_runs": [{
            "id": 101, "event": "pull_request", "conclusion": "success", "path": ".github/workflows/ci.yml",
            "head_sha": head, "html_url": "https://github.com/acme/widgets/actions/runs/101",
        }]},
        f"repos/{_REPO}/actions/runs/101/artifacts": {"artifacts": [{
            "id": 202, "name": "ci-proof", "expired": False, "size_in_bytes": 500,
        }]},
    }
    return base, head, proof, replies


def _archive(proof: dict[str, Any]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("proof.json", json.dumps(proof))
    return stream.getvalue()


@pytest.mark.parametrize("change,mode", [("docs/guide.md", "lightweight"), ("daydream/app.py", "full")])
def test_ci_cli_rechecks_code_but_reuses_pr_code_for_documentation_updates(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path, change: str, mode: str,
) -> None:
    base, _, proof, replies = _successful_merge_proof(git_repo, ci_cli, tmp_path)
    replies[f"repos/{_REPO}/actions/artifacts/202/zip"] = _archive(proof)
    _git(git_repo, "checkout", "feature")
    head = _commit(git_repo, change)
    _git(git_repo, "checkout", "main")
    _git(git_repo, "reset", "--soft", base)
    # Restore the fixture's base tree without any hook bypass.
    _git(git_repo, "restore", "--source", base, "--staged", "--worktree", ".")
    _git(git_repo, "-c", "commit.gpgsign=false", "merge", "--no-ff", "feature", "-m", "next merge")
    output, summary = ci_cli(_pr(base, head), replies=replies)
    assert f"mode={mode}\n" in output
    if mode == "lightweight":
        assert "actions/runs/101" in summary


@pytest.mark.parametrize("fault", [
    "none", "tree", "head", "pr", "repository", "run", "commit", "expired", "failed", "bad-zip", "api-error",
])
def test_ci_cli_reuses_only_verified_exact_merged_trees(
    git_repo: Path, ci_cli: Callable[..., tuple[str, str]], tmp_path: Path, fault: str,
) -> None:
    _, head, proof, replies = _successful_merge_proof(git_repo, ci_cli, tmp_path)
    if fault in {"tree", "head", "commit"}:
        proof[{"tree": "tree", "head": "head_sha", "commit": "commit"}[fault]] = "f" * 40
    elif fault in {"pr", "run"}:
        proof["pr_number" if fault == "pr" else "run_id"] = 999
    elif fault == "repository":
        proof["repository"] = "other/repo"
    elif fault == "expired":
        replies[f"repos/{_REPO}/actions/runs/101/artifacts"]["artifacts"][0]["expired"] = True
    elif fault == "failed":
        replies[f"repos/{_REPO}/actions/workflows/ci.yml/runs"]["workflow_runs"][0]["conclusion"] = "failure"
    elif fault == "api-error":
        replies[f"repos/{_REPO}/actions/workflows/ci.yml/runs"] = subprocess.CalledProcessError(1, "gh")
    replies[f"repos/{_REPO}/actions/artifacts/202/zip"] = b"invalid" if fault == "bad-zip" else _archive(proof)
    sha = _git(git_repo, "rev-parse", "HEAD")
    replies[f"repos/{_REPO}/commits/{sha}/pulls"] = [{
        "number": 7, "merged_at": "2026-10-14T12:00:00Z", "merge_commit_sha": sha,
        "base": {"ref": "main"}, "head": {"ref": "feature", "sha": head},
    }]
    output, _ = ci_cli({"ref": "refs/heads/main"}, event_name="push", replies=replies)
    assert f"mode={'reuse' if fault == 'none' else 'full'}\n" in output


def test_ci_cli_direct_push_and_manual_dispatch_run_full_gate(
    ci_cli: Callable[..., tuple[str, str]], git_repo: Path,
) -> None:
    sha = _git(git_repo, "rev-parse", "HEAD")
    events: list[tuple[str, dict[str, Any]]] = [
        ("workflow_dispatch", {}), ("push", {f"repos/{_REPO}/commits/{sha}/pulls": []}),
    ]
    for name, replies in events:
        output, _ = ci_cli({"ref": "refs/heads/main"}, event_name=name, replies=replies)
        assert "mode=full\n" in output and "full=true\n" in output


def test_ci_cli_rename_out_of_product_input_does_not_hide_deletion(
    ci_cli: Callable[..., tuple[str, str]], git_repo: Path,
) -> None:
    base = _commit(git_repo, "daydream/prompts/review.md")
    (git_repo / "docs").mkdir()
    _git(git_repo, "mv", "daydream/prompts/review.md", "docs/review.md")
    _git(git_repo, "-c", "commit.gpgsign=false", "commit", "-m", "rename")
    output, _ = ci_cli(_pr(base, _git(git_repo, "rev-parse", "HEAD")))
    assert "mode=full\n" in output
