#!/usr/bin/env python3
"""Choose CI checks; only verified merge-queue trees can replace a full gate.

Uses the runner's Python standard library, Git, and authenticated gh CLI.
PR checks become basic only while effective rules require a queue without bypass.
Queue checks and manual runs always execute the complete root gate.
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
_BASE_REFS = {"refs/heads/main", "refs/heads/master"}
_PROOF_SCHEMA = 2
_ACTIONS_APP_ID = 15368


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, timeout=30).strip()


def api(endpoint: str, *, binary: bool = False) -> Any:
    result = subprocess.check_output(["gh", "api", endpoint], timeout=15)
    return result if binary else json.loads(result)


def queue_identity(group: dict[str, Any]) -> bool:
    """Validate the event's queue namespace before using its Git identities."""
    base_ref = group.get("base_ref")
    return (
        base_ref in _BASE_REFS
        and isinstance(group.get("head_ref"), str)
        and group["head_ref"].startswith(f"refs/heads/gh-readonly-queue/{base_ref.removeprefix('refs/heads/')}/")
        and all(isinstance(group.get(key), str) and _SHA.fullmatch(group[key]) for key in ("head_sha", "base_sha"))
    )


def verify_commit(proof: dict[str, Any]) -> bool:
    """Resolve actual tested objects and the captured base, never trust JSON alone."""
    git("fetch", "--no-tags", "origin", proof["commit"])
    return (
        git("rev-parse", f"{proof['commit']}^{{commit}}") == proof["head_sha"]
        and git("rev-parse", f"{proof['commit']}^{{tree}}") == proof["tree"]
        and git("rev-parse", f"{proof['base_sha']}^{{tree}}") == proof["base_tree"]
        and proof["base_sha"] != proof["head_sha"]
        and git("merge-base", "--is-ancestor", proof["base_sha"], proof["commit"]) == ""
    )


def proofs(repo: str, base_ref: str) -> Iterator[tuple[dict[str, Any], str]]:
    query = {"event": "merge_group", "status": "success", "per_page": "20"}
    runs = api(f"repos/{repo}/actions/workflows/ci.yml/runs?{urlencode(query)}")["workflow_runs"]
    for run in runs:
        if (
            run["event"] != "merge_group"
            or run["conclusion"] != "success"
            or run["path"] != ".github/workflows/ci.yml"
            or run["repository"]["full_name"] != repo
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
            proof.get("schema") == _PROOF_SCHEMA
            and proof.get("event") == "merge_group"
            and proof.get("gate") == "full"
            and proof.get("repository") == repo
            and proof.get("run_id") == run["id"]
            and proof.get("base_ref") == base_ref
            and queue_identity(proof)
            and proof["head_ref"].removeprefix("refs/heads/") == run["head_branch"]
            and proof.get("head_sha") == proof.get("commit") == run["head_sha"]
            and all(isinstance(proof.get(key), str) and _SHA.fullmatch(proof[key])
                    for key in ("tree", "commit", "base_tree"))
        ):
            yield proof, run["html_url"]


def queue_without_bypass(repo: str, rule: dict[str, Any]) -> bool:
    ruleset_id = rule["ruleset_id"]
    if (
        type(ruleset_id) is not int or ruleset_id <= 0
        or not isinstance(rule["ruleset_source"], str) or not rule["ruleset_source"]
        or rule["ruleset_source_type"] not in ("Repository", "Organization", "Enterprise")
    ):
        return False
    ruleset = api(f"repos/{repo}/rulesets/{ruleset_id}?includes_parents=true")
    # GitHub omits bypass_actors without write access; absence is not proof.
    return (
        type(ruleset["id"]) is int and ruleset["id"] == ruleset_id
        and ruleset["source"] == rule["ruleset_source"]
        and ruleset["source_type"] == rule["ruleset_source_type"]
        and ruleset["target"] == "branch"
        and ruleset["enforcement"] == "active"
        and any(item["type"] == "merge_queue" for item in ruleset["rules"])
        and ruleset.get("bypass_actors") == []
    )


def plan(event_name: str, event: dict[str, Any], repo: str) -> tuple[str, str]:
    if event_name == "pull_request":
        base = event["pull_request"]["base"]
        if base["repo"]["full_name"] == repo and f"refs/heads/{base['ref']}" in _BASE_REFS:
            # This endpoint returns effective active rules, excluding disabled
            # and evaluate-only rulesets, including inherited organization rules.
            rules = api(f"repos/{repo}/rules/branches/{base['ref']}?per_page=100")
            queue = [rule for rule in rules if rule["type"] == "merge_queue"]
            check = any(
                required["context"] == "check" and required["integration_id"] == _ACTIONS_APP_ID
                for rule in rules if rule["type"] == "required_status_checks"
                for required in rule["parameters"]["required_status_checks"]
            )
            if queue and check and all(queue_without_bypass(repo, rule) for rule in queue):
                return "basic", "Active rules require merge queue without bypass and the Actions check before merge."
        return "full", "Merge queue without bypass is not verified active; run the complete PR gate."
    if event_name == "push" and event["ref"] in _BASE_REFS:
        sha = os.environ["GITHUB_SHA"]
        if git("rev-parse", "HEAD") != sha or event["after"] != sha:
            return "full", "Push checkout identity differs; run the complete gate."
        current = git("rev-parse", "HEAD^{tree}")
        merged = api(f"repos/{repo}/commits/{sha}/pulls")
        if not any(
            pr["merged_at"] and pr["merge_commit_sha"] == sha
            and pr["base"]["repo"]["full_name"] == repo
            and event["ref"] == f"refs/heads/{pr['base']['ref']}"
            for pr in merged
        ):
            return "full", "No matching merged PR for this push; run the complete gate."
        for proof, url in proofs(repo, event["ref"]):
            # Direct pushes cannot borrow an unrelated successful queue tree:
            # require the event's old branch tip to be in the tested history.
            if (
                proof["tree"] == current
                and verify_commit(proof)
                and git("merge-base", "--is-ancestor", event["before"], proof["commit"]) == ""
            ):
                return "reuse", f"Exact pushed tree already passed full merge-queue CI: {url}"
    return "full", "Run the complete root gate."


def record(event_name: str, event: dict[str, Any], repo: str, proof_dir: Path) -> None:
    if event_name != "merge_group" or event["action"] != "checks_requested":
        raise ValueError("Only a full merge-group gate can publish reusable proof")
    group = event["merge_group"]
    if event["repository"]["full_name"] != repo or not queue_identity(group):
        raise ValueError("Invalid merge-group repository or ref identity")
    commit = git("rev-parse", "HEAD")
    if commit != os.environ["GITHUB_SHA"] or commit != group["head_sha"]:
        raise ValueError("The checkout changed identity during CI")
    git("diff", "--exit-code", "HEAD", "--")
    proof = {
        "schema": _PROOF_SCHEMA,
        "event": "merge_group",
        "gate": "full",
        "repository": repo,
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "head_ref": group["head_ref"],
        "head_sha": commit,
        "commit": commit,
        "tree": git("rev-parse", "HEAD^{tree}"),
        "base_ref": group["base_ref"],
        "base_sha": group["base_sha"],
        "base_tree": git("rev-parse", f"{group['base_sha']}^{{tree}}"),
    }
    if proof["base_sha"] == commit or git("merge-base", "--is-ancestor", proof["base_sha"], commit) != "":
        raise ValueError("The merge-group base is not an earlier ancestor")
    proof_dir.mkdir(parents=True, exist_ok=True)
    (proof_dir / "proof.json").write_text(json.dumps(proof))


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
        record(os.environ["GITHUB_EVENT_NAME"], event, repo, args.proof_dir)
        return
    try:
        mode, reason = plan(os.environ["GITHUB_EVENT_NAME"], event, repo)
    except Exception as exc:
        # Cost evidence never authorizes weaker checks when its reader fails.
        mode, reason = "full", f"CI evidence unavailable ({type(exc).__name__}); run the complete gate."
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"full={'true' if mode == 'full' else 'false'}\nmode={mode}\n")
    print(reason)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a") as output:
            output.write(f"CI policy: **{mode}**. {reason}\n")


if __name__ == "__main__":
    main()
