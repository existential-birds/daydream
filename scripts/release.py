#!/usr/bin/env python3
"""Tag and publish the release that the checked-out commit declares, once.

Run by ``.github/workflows/release.yml`` when a push to main changes ``pyproject.toml``.
Uses the runner's Python standard library, Git, and authenticated gh CLI.
``[project].version`` names the release ``v<version>``. Nothing is tagged unless
``daydream/__init__.py`` agrees and ``CHANGELOG.md`` has the version's dated section with
entries, which becomes the release notes. A tag that already exists, on this or an earlier
commit, is never moved; if its release is missing, it is published from the tagged commit's
CHANGELOG, so a dispatch on main retries a failed publish even after later merges.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

_INIT_VERSION = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, timeout=60).strip()


def gh(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["gh", *args], input=stdin, text=True, capture_output=True, timeout=60, check=False)


def release_notes(changelog: str, version: str) -> str:
    """The body of the version's dated CHANGELOG section, without its heading."""
    heading = re.compile(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", re.MULTILINE)
    found = heading.search(changelog)
    if found is None:
        raise SystemExit(f"CHANGELOG.md has no dated section '## [{version}] - YYYY-MM-DD'")
    following = re.compile(r"^## \[", re.MULTILINE).search(changelog, found.end())
    notes = changelog[found.end():following.start() if following else len(changelog)].strip()
    if not any(line.strip() and not line.lstrip().startswith("#") for line in notes.splitlines()):
        raise SystemExit(f"CHANGELOG.md section {version} has no entries")
    return notes + "\n"


def main() -> None:
    version = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    tag, head = f"v{version}", git("rev-parse", "HEAD")
    try:
        tagged = git("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    except subprocess.CalledProcessError:
        tagged = None
    if tagged is None:
        declared = _INIT_VERSION.search(Path("daydream/__init__.py").read_text())
        if declared is None or declared[1] != version:
            raise SystemExit(f"daydream/__init__.py does not declare __version__ = \"{version}\"")
        notes = release_notes(Path("CHANGELOG.md").read_text(), version)
        git("tag", "--annotate", tag, "--message", f"Release {tag}", head)
        git("push", "origin", f"refs/tags/{tag}")
        print(f"Tagged {tag} at {head}.")
    if gh("release", "view", tag, "--json", "tagName").returncode == 0:
        print(f"Release {tag} already exists.")
        return
    if tagged is not None:
        notes = release_notes(git("show", f"{tagged}:CHANGELOG.md"), version)
    created = gh("release", "create", tag, "--verify-tag", "--title", tag, "--notes-file", "-", stdin=notes)
    if created.returncode:
        raise SystemExit(f"gh release create {tag} failed: {created.stderr.strip()}")
    print(f"Published release {tag}.")


if __name__ == "__main__":
    main()
