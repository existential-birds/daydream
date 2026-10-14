"""Local github.com identity for provenance and repository-scoped App bootstrap.

Unlike archive URL normalization, resolution rejects unsupported or ambiguous
paths and routing. Invalid inputs return no identity and are never echoed.
"""

import os
import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from daydream.git_ops.queries import remote_url

_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
_REPO = re.compile(r"[A-Za-z0-9_.-]{1,100}")


def _slug(value: str) -> tuple[str, str] | None:
    parts = value.split("/")
    if len(parts) != 2:
        return None
    owner, repo = parts
    if not _OWNER.fullmatch(owner) or not _REPO.fullmatch(repo) or repo in (".", ".."):
        return None
    return owner, repo


def _url_identity(value: str, *, remote: bool) -> tuple[str, str] | None:
    if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
        return None
    if remote and value.startswith("git@github.com:"):
        path = value.removeprefix("git@github.com:")
    else:
        try:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in (("https", "ssh") if remote else ("https",))
                or parsed.hostname != "github.com"
                or parsed.port is not None
                or parsed.query
                or parsed.fragment
                or (not remote and parsed.username is not None)
                or (parsed.scheme == "ssh" and (parsed.username != "git" or parsed.password is not None))
            ):
                return None
        except ValueError:
            return None
        if not parsed.path.startswith("/"):
            return None
        path = parsed.path[1:]
    if path.endswith(".git"):
        path = path[:-4]
    return _slug(path)


def resolve_github_repository(
    target_dir: Path,
    pr_repo: str | None = None,
    *,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[str, str] | None:
    """Resolve explicit slug, captured GH_REPO, then target origin without API access.

    Only github.com routing is supported. An invalid selected source never
    falls back to another repository. The caller must bind App subprocess
    routing to the resolved identity before minting or using credentials.
    """
    environment = os.environ if base_environment is None else base_environment
    if environment.get("GH_HOST", "github.com").lower() != "github.com":
        return None
    if pr_repo is not None:
        return _slug(pr_repo)
    if "GH_REPO" in environment:
        selected = environment["GH_REPO"]
        if selected.startswith("https://"):
            return _url_identity(selected, remote=False)
        return _slug(selected.removeprefix("github.com/"))
    origin = remote_url(target_dir)
    if origin is None or any(char.isspace() for char in origin):
        return None
    return _url_identity(origin, remote=True)
