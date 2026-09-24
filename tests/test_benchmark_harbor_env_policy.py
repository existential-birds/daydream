"""Policy declarations and per-layer consumption for the harbor env policy."""

import ast
from pathlib import Path

from daydream.benchmark.harbor import env_policy


def test_declaration_is_a_leaf_module() -> None:
    """The declaration imports stdlib only, not the container entrypoint's dependencies."""
    tree = ast.parse(Path(env_policy.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imported <= {"__future__", "collections.abc", "dataclasses", "types", "typing"}


def test_host_channel_declares_the_host_child_env_sets() -> None:
    """The host channel is today's fail-closed allowlist, verbatim."""
    host = env_policy.HOST
    assert env_policy.REVIEW_CHANNEL_PREFIX == "DAYDREAM_REVIEW_"
    assert host.keep_prefixes == (env_policy.REVIEW_CHANNEL_PREFIX,)
    assert host.required_process_vars == frozenset({"PATH", "HOME", "LANG"})
    assert host.banned_vars == frozenset({
        "GH_TOKEN", "GITHUB_TOKEN", "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY",
        "HF_TOKEN", "DAYDREAM_TRAJECTORY_HUB_REPO", "DAYDREAM_ARCHIVE_DIR",
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "OPENROUTER_API_KEY", "PI_API_KEY",
    })
    assert host.banned_prefixes == frozenset({
        "DAYDREAM_JUDGE_", "ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_", "OPENROUTER_", "PI_",
    })
    assert host.claude_keep_vars == frozenset({
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    })
    assert host.claude_exempt_vars == frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"})
    assert host.claude_exempt_prefix == "ANTHROPIC_"
