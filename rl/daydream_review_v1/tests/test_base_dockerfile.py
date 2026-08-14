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
    "heading,prefix,amd64_digest,arm64_digest,tmp_path,use_marker,metadata_url",
    [
        (
            "claude",
            "CLAUDE_CODE",
            "3c029136f7c81f54ed4a38e9d52e655aad536433dbbde50519c8c31bb646ad14",
            "4c38f26a57a42619ee813f15dc39fc1fa4fe0bb403215c3cdc342b58fa689c3c",
            "/tmp/claude",
            "install -m 0755 /tmp/claude",
            CLAUDE_MANIFEST_URL,
        ),
        (
            "codex",
            "CODEX",
            "bfaf13c9ba34f2ad764e4a916c49cf7177aeba329cf0f719e2227566fc8d662a",
            "d384f90bc842450b42bd675feef06a12a46a3b1ca97efcb22566b270e4a11227",
            "/tmp/codex.tar.gz",
            "tar -xzf /tmp/codex.tar.gz",
            CODEX_RELEASE_URL,
        ),
        (
            "pi (and the node runtime it needs)",
            "NODE",
            "cfb6ac0cf339825fe36efd1f18a79016b02aca19fbfa6c9547c57e27dc09f6ea",
            "f53510706998cf044f634190416f0588e7e1937aecea938768952e0f0ac1f41b",
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
    amd64_digest: str,
    arm64_digest: str,
    tmp_path: str,
    use_marker: str,
    metadata_url: str,
) -> None:
    source = BASE_DOCKERFILE.read_text(encoding="utf-8")
    section = _dockerfile_section(source, heading)

    # Both per-architecture digests are declared exactly, and are 64 lowercase hex.
    assert f"ARG {prefix}_SHA256_AMD64={amd64_digest}" in section
    assert f"ARG {prefix}_SHA256_ARM64={arm64_digest}" in section
    for digest in (amd64_digest, arm64_digest):
        assert re.fullmatch(r"[0-9a-f]{64}", digest), f"{prefix} digest must be 64 lowercase hex"

    # Both architecture branches assign the matching checksum var.
    assert f'checksum="${{{prefix}_SHA256_AMD64}}"' in section
    assert f'checksum="${{{prefix}_SHA256_ARM64}}"' in section

    # Provenance comment and temp-artifact cleanup are present.
    assert metadata_url in section
    assert f"rm -f {tmp_path}" in section

    # Strict ordering: download < verify < use.
    dl = section.index(f"curl -fsSL -o {tmp_path}")
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
