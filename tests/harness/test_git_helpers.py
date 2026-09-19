"""Contract tests for the shared real-git subprocess helpers.

The seed family (``SEED_ENV`` + ``write_and_stage`` + ``seeded_commit``) is what makes
every benchmark suite's fixtures byte-reproducible, so the SHAs it produces are
pinned here: this module is the regression gate for the seed identity and the
commit step. Values were measured on this repository with these helpers.
"""

from __future__ import annotations

from pathlib import Path

from tests.harness.git_helpers import git, init_repo, seeded_commit, write_and_stage

_SHA_BASE1 = "cae67fc3eb4c5d3dd3353ca7fb41f909837bf0a2"
_SHA_BASE1_TREE = "2cd99bd20f7b3bac54014e20db1831d64b2c4fc9"
_SHA_BASE2 = "d35f2cbffc81b6292f67cf891ac1c4256fe948a4"

_SEED_IDENTITY = (
    "Tester <test@example.com> 2026-01-01T00:00:00+00:00 "
    "Tester <test@example.com> 2026-01-01T00:00:00+00:00"
)


def test_seed_family_produces_pinned_shas(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    write_and_stage(repo, "readme.txt", "base1\n")
    base1 = seeded_commit(repo, "base1")
    write_and_stage(repo, "base.py", "BASE = 2\n")
    base2 = seeded_commit(repo, "base2")

    assert base1 == _SHA_BASE1
    assert git(repo, "rev-parse", f"{base1}^{{tree}}") == _SHA_BASE1_TREE
    assert base2 == _SHA_BASE2
    assert git(repo, "log", "-1", "--format=%an <%ae> %aI %cn <%ce> %cI") == _SEED_IDENTITY


def test_write_and_stage_stages_binary_content(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    init_repo(repo)
    write_and_stage(repo, "blob.bin", b"\x00\x01\x02")
    seeded_commit(repo, "binary")

    assert (repo / "blob.bin").read_bytes() == b"\x00\x01\x02"
    assert git(repo, "show", "--name-only", "--format=", "HEAD") == "blob.bin"
