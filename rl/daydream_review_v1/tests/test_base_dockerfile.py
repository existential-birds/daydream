"""Static regression test for the rollout base image's download verification.

Asserts, from the Dockerfile text alone (no Docker daemon), that every pinned
CLI download is checked against a hard-coded SHA-256 digest before any of its
bytes become executable -- and that the protections cannot silently disappear
on a future edit.
"""

from __future__ import annotations

import re

import pytest
from conftest import PROJECT_ROOT

BASE_DOCKERFILE = PROJECT_ROOT / "images" / "base.Dockerfile"

CLAUDE_MANIFEST_URL = (
    "https://downloads.claude.ai/claude-code-releases/${CLAUDE_CODE_VERSION}/manifest.json"
)
CODEX_RELEASE_URL = "https://api.github.com/repos/openai/codex/releases/tags/rust-v${CODEX_VERSION}"
NODE_SHASUMS_URL = "https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt"


def _dockerfile_section(source: str, heading: str) -> str:
    """Return the section beginning at '# --- {heading} ' through the next '# --- ' marker."""
    marker = f"# --- {heading} "
    start = source.index(marker)
    rest = source[start + len(marker) :]
    nxt = rest.find("\n# --- ")
    return rest if nxt == -1 else rest[:nxt]


@pytest.mark.parametrize(
    "heading,prefix,tmp_artifact,use_marker,metadata_url",
    [
        (
            "claude",
            "CLAUDE_CODE",
            "/tmp/claude",
            "install -m 0755 /tmp/claude",
            CLAUDE_MANIFEST_URL,
        ),
        (
            "codex",
            "CODEX",
            "/tmp/codex.tar.gz",
            "tar -xzf /tmp/codex.tar.gz",
            CODEX_RELEASE_URL,
        ),
        (
            "pi (and the node runtime it needs)",
            "NODE",
            "/tmp/node.tar.gz",
            "tar -xzf /tmp/node.tar.gz",
            NODE_SHASUMS_URL,
        ),
    ],
    ids=["claude", "codex", "node"],
)
def test_pinned_tool_downloads_are_verified_before_use(
    heading: str,
    prefix: str,
    tmp_artifact: str,
    use_marker: str,
    metadata_url: str,
) -> None:
    source = BASE_DOCKERFILE.read_text(encoding="utf-8")
    section = _dockerfile_section(source, heading)

    # Both per-architecture digest ARGs are present and well-formed (64 lowercase hex). The
    # Dockerfile -- not this parametrize table -- is the single source of truth for the values,
    # so an edit to one file alone cannot desynchronize them silently.
    for arch in ("AMD64", "ARM64"):
        assert re.search(
            rf"^ARG {prefix}_SHA256_{arch}=[0-9a-f]{{64}}$",
            section,
            re.MULTILINE,
        ), f"{prefix} {arch} digest ARG missing or malformed"

    # Both architecture branches assign the matching checksum var.
    assert f'checksum="${{{prefix}_SHA256_AMD64}}"' in section
    assert f'checksum="${{{prefix}_SHA256_ARM64}}"' in section

    # Provenance comment and temp-artifact cleanup are present.
    assert metadata_url in section
    assert f"rm -f {tmp_artifact}" in section

    # Strict ordering: download < verify < use.
    dl = section.index(f"curl -fsSL -o {tmp_artifact}")
    verify = section.index("sha256sum --check --strict")
    use = section.index(use_marker)
    assert dl < verify < use


def test_base_dockerfile_has_no_unverified_download_pipelines() -> None:
    """Whole-file invariants: exactly one strict check per CLI, no installer/pipe."""

    source = BASE_DOCKERFILE.read_text(encoding="utf-8")
    assert source.count("sha256sum --check --strict") == 3
    assert "https://claude.ai/install.sh" not in source
    assert "| bash" not in source
    assert "| tar" not in source
