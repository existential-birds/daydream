"""Shared helpers for the review-bot comparison harness.

Generalized over any GitHub review bot (coderabbit, greptile, ...) and any
repo. Pure stdlib + the `gh` and `git` CLIs — no third-party deps so the
harness runs under the system Python.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def run(cmd: list[str], *, cwd: str | Path | None = None, timeout: int | None = None,
        check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, returning the completed process (text mode)."""
    proc = subprocess.run(  # noqa: S603 - args are constructed from trusted CLI inputs
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc


def gh_json(args: list[str]) -> Any:
    """Run `gh <args>` and parse stdout as JSON."""
    proc = run(["gh", *args])
    return json.loads(proc.stdout) if proc.stdout.strip() else None


def gh_paginate(endpoint: str, *, jq: str | None = None) -> list[Any]:
    """Run `gh api --paginate <endpoint>` and return a flat list of items.

    REST list endpoints return a JSON array per page; --paginate concatenates
    pages with `\\n` between top-level arrays unless --slurp is given, so we
    request --slurp and flatten.
    """
    args = ["api", "--paginate", "--slurp", endpoint]
    if jq:
        args += ["--jq", jq]
    data = gh_json(args)
    if data is None:
        return []
    # --slurp wraps the per-page arrays in an outer array; flatten one level.
    flat: list[Any] = []
    for page in data:
        if isinstance(page, list):
            flat.extend(page)
        else:
            flat.append(page)
    return flat


def graphql_review_threads(owner: str, repo: str, number: int) -> tuple[list[dict[str, Any]], bool]:
    """Fetch PR review threads (resolution + first comment author) via GraphQL.

    Returns ``(threads, ok)``, one dict per thread:
    {is_resolved, is_outdated, path, line, author}. Used to derive the "acted
    upon" ground-truth signal. ``ok`` is ``False`` when a transport/shape error
    truncated the set, so the caller can mark that PR's resolved-thread count as
    unreliable rather than reading a partial fetch as a real value. Best-effort:
    never raises on a GraphQL hiccup.
    """
    query = """
    query($owner:String!,$repo:String!,$num:Int!,$cursor:String){
      repository(owner:$owner,name:$repo){
        pullRequest(number:$num){
          reviewThreads(first:100,after:$cursor){
            nodes{
              isResolved isOutdated path line
              comments(first:1){ nodes{ author{login} } }
            }
            pageInfo{ hasNextPage endCursor }
          }
        }
      }
    }
    """
    threads: list[dict[str, Any]] = []
    cursor: str | None = None
    try:
        while True:
            args = [
                "api", "graphql",
                "-f", f"query={query}",
                "-F", f"owner={owner}",
                "-F", f"repo={repo}",
                "-F", f"num={number}",
            ]
            if cursor:
                args += ["-F", f"cursor={cursor}"]
            data = gh_json(args)
            node = data["data"]["repository"]["pullRequest"]["reviewThreads"]
            for t in node["nodes"]:
                comments = t.get("comments", {}).get("nodes", [])
                author = ""
                if comments and comments[0].get("author"):
                    author = comments[0]["author"].get("login", "")
                threads.append({
                    "is_resolved": t.get("isResolved", False),
                    "is_outdated": t.get("isOutdated", False),
                    "path": t.get("path"),
                    "line": t.get("line"),
                    "author": author,
                })
            page = node["pageInfo"]
            if not page["hasNextPage"]:
                break
            cursor = page["endCursor"]
    except (RuntimeError, KeyError, TypeError, json.JSONDecodeError):
        return threads, False
    return threads, True


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_slug(repo: str) -> str:
    """owner/repo -> owner__repo for filesystem-safe dir names."""
    return repo.replace("/", "__")


def bot_login_matches(login: str | None, bot: str) -> bool:
    """Match a bot login tolerant of GitHub's REST/GraphQL `[bot]` mismatch.

    REST `user.login` keeps the `[bot]` suffix (`coderabbitai[bot]`); GraphQL
    `author.login` drops it (`coderabbitai`). Compare on the stripped stem.
    """
    def stem(s: str | None) -> str:
        return (s or "").removesuffix("[bot]").lower()

    return stem(login) == stem(bot)
