"""Release publication at its CLI boundary: real Git repositories and remote, gh stubbed."""

from __future__ import annotations

import runpy
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release.py"
_NOTES = "### Fixed\n\n- A fixture fix (#2).\n"
Calls = list[tuple[tuple[str, ...], str | None]]
Release = Callable[..., Calls]
Repos = tuple[Path, Path]
_CHANGELOG = (f"# Changelog\n\n## [Unreleased]\n\n## [0.2.0] - 2026-11-23\n\n{_NOTES}\n"
              "## [0.1.0] - 2026-11-01\n\n### Added\n\n- The first release.\n")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit(repo: Path, version: str, changelog: str = _CHANGELOG, declared: str | None = None) -> str:
    (repo / "daydream").mkdir(exist_ok=True)
    (repo / "pyproject.toml").write_text(f'[project]\nname = "daydream"\nversion = "{version}"\n')
    (repo / "daydream" / "__init__.py").write_text(f'"""Fixture."""\n\n__version__ = "{declared or version}"\n')
    (repo / "CHANGELOG.md").write_text(changelog)
    _git(repo, "add", "pyproject.toml", "daydream/__init__.py", "CHANGELOG.md")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", f"fixture {version}")
    _git(repo, "push", "--quiet", "origin", "HEAD:main")
    return _git(repo, "rev-parse", "HEAD")


def _tag(repo: Path, name: str, commit: str) -> None:
    _git(repo, "-c", "tag.gpgsign=false", "tag", "--annotate", name, "--message", f"Release {name}", commit)
    _git(repo, "push", "--quiet", "origin", f"refs/tags/{name}")


@pytest.fixture
def repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Repos:
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    subprocess.run(["git", "init", "--quiet", "--bare", "--initial-branch=main", str(origin)], check=True)
    subprocess.run(["git", "clone", "--quiet", str(origin), str(work)], check=True)
    for key, value in {"GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.com",
                       "GIT_COMMITTER_NAME": "Release Bot", "GIT_COMMITTER_EMAIL": "bot@example.com",
                       "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}.items():
        monkeypatch.setenv(key, value)
    _tag(work, "v0.1.0", _commit(work, "0.1.0"))
    monkeypatch.chdir(work)
    return origin, work


@pytest.fixture
def release(monkeypatch: pytest.MonkeyPatch) -> Release:
    """Run the script with a stubbed gh; return each gh call and the notes it was sent."""

    def run(*, published: bool = False, create_fails: bool = False) -> Calls:
        calls: Calls = []

        def gh(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
            calls.append((args, stdin))
            failed = (args[1] == "view" and not published) or (args[1] == "create" and create_fails)
            return subprocess.CompletedProcess(["gh", *args], 1 if failed else 0, "", "HTTP 403" if failed else "")

        namespace: dict[str, Any] = runpy.run_path(str(_SCRIPT))
        monkeypatch.setitem(namespace["main"].__globals__, "gh", gh)
        namespace["main"]()
        return calls

    return run


def _remote_tags(origin: Path) -> dict[str, str]:
    lines = _git(origin, "for-each-ref", "refs/tags", "--format=%(refname:short) %(*objectname)").splitlines()
    return dict(line.split(" ") for line in lines)


def test_a_merged_release_is_tagged_annotated_and_published_with_its_changelog_section(
        repos: Repos, release: Release) -> None:
    origin, work = repos
    head = _commit(work, "0.2.0")

    calls = release()

    assert _remote_tags(origin)["v0.2.0"] == head
    assert _git(origin, "for-each-ref", "refs/tags/v0.2.0",
                "--format=%(objecttype) %(contents:subject) %(taggername)") == "tag Release v0.2.0 Release Bot"
    assert calls == [(("release", "view", "v0.2.0", "--json", "tagName"), None),
                     (("release", "create", "v0.2.0", "--verify-tag", "--title", "v0.2.0", "--notes-file", "-"),
                      _NOTES)]


def test_a_push_that_keeps_a_released_version_tags_and_publishes_nothing(repos: Repos, release: Release) -> None:
    origin, work = repos
    tagged = _remote_tags(origin)
    _commit(work, "0.1.0")

    assert release() == []
    assert _remote_tags(origin) == tagged


@pytest.mark.parametrize("published", [False, True], ids=["release-missing", "release-present"])
def test_a_tag_on_this_commit_gets_its_release_only_when_missing(
        repos: Repos, release: Release, published: bool) -> None:
    origin, work = repos
    head = _commit(work, "0.2.0")
    _tag(work, "v0.2.0", head)

    calls = release(published=published)

    assert _remote_tags(origin)["v0.2.0"] == head
    assert [args[1] for args, _ in calls] == (["view"] if published else ["view", "create"])


def test_a_failed_publish_keeps_the_tag_so_a_rerun_publishes(repos: Repos, release: Release) -> None:
    origin, work = repos
    head = _commit(work, "0.2.0")

    with pytest.raises(SystemExit, match="gh release create v0.2.0 failed: HTTP 403"):
        release(create_fails=True)

    assert _remote_tags(origin)["v0.2.0"] == head
    assert [args[1] for args, _ in release()] == ["view", "create"]


@pytest.mark.parametrize(("changelog", "declared", "message"), [
    (_CHANGELOG, "0.1.9", r'does not declare __version__ = "0.2.0"'),
    (_CHANGELOG.replace("## [0.2.0] - 2026-11-23", "## [0.2.0]"), None, r"no dated section '## \[0.2.0\]"),
    (_CHANGELOG.replace(_NOTES, ""), None, "section 0.2.0 has no entries"),
], ids=["package-version-differs", "section-undated", "section-empty"])
def test_an_inconsistent_release_commit_is_refused_before_anything_is_tagged(
        repos: Repos, release: Release, changelog: str, declared: str | None, message: str) -> None:
    origin, work = repos
    tagged = _remote_tags(origin)
    _commit(work, "0.2.0", changelog, declared)

    with pytest.raises(SystemExit, match=message):
        release()

    assert _remote_tags(origin) == tagged
    assert _git(work, "tag", "--list", "v0.2.0") == ""
