#!/usr/bin/env python3
"""Choose the repository CI gate; unverifiable reuse always runs the full gate.

Uses only the runner's Python standard library, git, and authenticated gh CLI.
Successful PR jobs publish a small proof of the actual checkout, including the
synthetic merge commit (the Actions run's head_sha names the PR head instead).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

_SHA = re.compile(r"[0-9a-f]{40}\Z")
_DOCS = {"README.md", "CONTRIBUTING.md", "CHANGELOG.md", ".github/workflows/README.md"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, timeout=30).strip()


def api(endpoint: str, *, binary: bool = False) -> Any:
    result = subprocess.check_output(["gh", "api", endpoint], timeout=15)
    return result if binary else json.loads(result)


def lightweight_diff(before: str, after: str) -> bool:
    paths = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", before, after], timeout=30
    ).decode().split("\0")
    # Use a narrow documentation allowlist. Markdown under prompts/templates
    # is executable product input and must still run the full gate.
    return all(
        not path or path in _DOCS or (path.startswith("docs/") and path.endswith(".md")) or path.startswith("rl/")
        for path in paths
    )


def proofs(repo: str, branch: str, pr_number: int, head_sha: str | None = None) -> Iterator[tuple[dict[str, Any], str]]:
    query = {"event": "pull_request", "status": "success", "branch": branch, "per_page": "20"}
    if head_sha:
        query["head_sha"] = head_sha
    runs = api(f"repos/{repo}/actions/workflows/ci.yml/runs?{urlencode(query)}")["workflow_runs"]
    for run in runs:
        if (
            run["event"] != "pull_request"
            or run["conclusion"] != "success"
            or run["path"] != ".github/workflows/ci.yml"
        ):
            continue
        artifacts = api(f"repos/{repo}/actions/runs/{run['id']}/artifacts")["artifacts"]
        artifact = next((a for a in artifacts if a["name"] == "ci-proof" and not a["expired"]), None)
        if artifact is None or artifact["size_in_bytes"] > 4096:
            continue
        archive = api(f"repos/{repo}/actions/artifacts/{artifact['id']}/zip", binary=True)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            if z.namelist() != ["proof.json"] or z.getinfo("proof.json").file_size > 4096:
                continue
            proof = json.loads(z.read("proof.json"))
        if (
            proof.get("repository") == repo
            and proof.get("pr_number") == pr_number
            and proof.get("run_id") == run["id"]
            and proof.get("head_sha") == run["head_sha"]
            and (head_sha is None or proof["head_sha"] == head_sha)
            and _SHA.fullmatch(proof.get("tree", ""))
            and _SHA.fullmatch(proof.get("commit", ""))
        ):
            yield proof, run["html_url"]


def verify_commit(proof: dict[str, Any]) -> bool:
    git("fetch", "--no-tags", "origin", proof["commit"])
    return (
        git("rev-parse", f"{proof['commit']}^{{tree}}") == proof["tree"]
        and git("rev-parse", f"{proof['commit']}^2") == str(proof["head_sha"])
    )


def plan(event_name: str, event: dict[str, Any], repo: str) -> tuple[str, str]:
    current = git("rev-parse", "HEAD^{tree}")
    if event_name == "pull_request":
        pr = event["pull_request"]
        if lightweight_diff(pr["base"]["sha"], "HEAD"):
            return "lightweight", "Only documentation or dormant RL files changed."
        for proof, url in proofs(repo, pr["head"]["ref"], event["number"]):
            # Historical synthetic merge commits can disappear from refs. An
            # unavailable object/fetch fails closed rather than guessing a base.
            if not verify_commit(proof):
                continue
            if lightweight_diff(proof["commit"], "HEAD"):
                return "lightweight", f"Only documentation/RL differs from successful PR CI: {url}"
            # The latest valid successful checkout changed code: run the gate
            # rather than downloading every older artifact for this branch.
            break
    elif event_name == "push":
        sha = os.environ["GITHUB_SHA"]
        # Never reuse a direct push, a multi-commit branch tip unrelated to a
        # merged PR, or an earlier base's PR result. Match both PR identity and
        # the entire tested merge tree, including workflow/dependency files.
        for pr in api(f"repos/{repo}/commits/{sha}/pulls"):
            if (
                not pr["merged_at"]
                or pr["merge_commit_sha"] != sha
                or event["ref"] != f"refs/heads/{pr['base']['ref']}"
            ):
                continue
            for proof, url in proofs(repo, pr["head"]["ref"], pr["number"], pr["head"]["sha"]):
                if proof["tree"] == current and verify_commit(proof):
                    return "reuse", f"Exact merged tree already passed PR CI: {url}"
    return "full", "Run the complete root gate."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "record"])
    parser.add_argument("--proof-dir", type=Path)
    args = parser.parse_args()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    repo = os.environ["GITHUB_REPOSITORY"]
    if args.command == "record":
        if args.proof_dir is None:
            parser.error("record requires --proof-dir")
        pr = event["pull_request"]
        commit = git("rev-parse", "HEAD")
        if commit != os.environ["GITHUB_SHA"]:
            raise ValueError("The checkout changed identity during CI")
        git("diff", "--exit-code", "HEAD", "--")
        proof = {
            "repository": repo,
            "pr_number": event["number"],
            "run_id": int(os.environ["GITHUB_RUN_ID"]),
            "head_sha": pr["head"]["sha"],
            "commit": commit,
            "tree": git("rev-parse", "HEAD^{tree}"),
        }
        args.proof_dir.mkdir(parents=True, exist_ok=True)
        (args.proof_dir / "proof.json").write_text(json.dumps(proof))
        return
    try:
        mode, reason = plan(os.environ["GITHUB_EVENT_NAME"], event, repo)
    except Exception as exc:
        # This is a cost optimization, never authority to weaken a gate when
        # its evidence reader fails (including an unexpected decoding error).
        mode, reason = "full", f"Reuse evidence unavailable ({type(exc).__name__}); run the complete gate."
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"full={'true' if mode == 'full' else 'false'}\nmode={mode}\n")
    print(reason)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as output:
            output.write(f"CI policy: **{mode}**. {reason}\n")


if __name__ == "__main__":
    main()
