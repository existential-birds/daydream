"""Versioned record identity binds run, trace segment, and host finding identity."""

import hashlib
import json


def record_finding_id(run_id: str, trajectory_id: str, segment_id: str, item_uid: str) -> str:
    """Versioned record identity preserves host findings independently of fingerprints."""
    for name, value in (
        ("run_id", run_id),
        ("trajectory_id", trajectory_id),
        ("segment_id", segment_id),
        ("item_uid", item_uid),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"record_finding_id: missing required component {name!r}")
    payload = json.dumps(["record-snapshot-v1", run_id, trajectory_id, segment_id, item_uid], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_repo_slug(raw: str) -> str:
    """Canonicalize producer URLs, .git suffixes, and whitespace to owner/repo.

    Return an empty string for invalid shapes so spelling cannot bypass benchmark isolation.
    """
    slug = raw.strip()
    if "://" in slug:
        remainder = slug.split("://", 1)[1]
        slug = remainder.split("/", 1)[1] if "/" in remainder else ""
    elif "@" in slug and ":" in slug:
        # SCP spelling 'git@host:owner/repo(.git)' has no scheme. Strip the
        # user@host prefix so the remainder reduces to 'owner/repo' below.
        userhost, __, scp_path = slug.rpartition(":")
        if "@" not in userhost:
            slug = ""
        else:
            slug = scp_path
    if slug.endswith(".git"):
        slug = slug[: -len(".git")]
    owner, sep, repo = slug.strip().partition("/")
    if not sep or not owner.strip() or not repo.strip() or "/" in repo:
        return ""
    return f"{owner.strip()}/{repo.strip()}"
