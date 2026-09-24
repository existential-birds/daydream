"""Harbor control-plane environment policy declarations.

``HOST`` is the host-side child-environment allowlist read by
``agent.build_child_env``. The container scrub is read by
``entrypoint._sanitize_reviewer_environment``; the renderer and task-TOML
injection views are read by ``package.render_job_config`` and
``package.render_task_toml`` respectively. The judge channel belongs to the
packaged verifier asset. These other channel views are added separately.

The host's ``claude_keep_vars`` are the same Anthropic names as the
``render_job_config`` placeholders: only that trio may survive the host
scrub in Claude mode.
"""

from __future__ import annotations

from dataclasses import dataclass

REVIEW_CHANNEL_PREFIX = "DAYDREAM_REVIEW_"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_BASE_URL_ENV = "ANTHROPIC_BASE_URL"
CLAUDE_KEEP_PREFIX = "ANTHROPIC_"
PI_API_KEY_ENV = "PI_API_KEY"


@dataclass(frozen=True)
class HostChannel:
    """Names allowed, required, or excluded by the host child-env builder."""

    keep_prefixes: tuple[str, ...]
    required_process_vars: frozenset[str]
    banned_vars: frozenset[str]
    banned_prefixes: frozenset[str]
    claude_keep_vars: frozenset[str]
    claude_exempt_vars: frozenset[str]
    claude_exempt_prefix: str


HOST = HostChannel(
    keep_prefixes=(REVIEW_CHANNEL_PREFIX,),
    required_process_vars=frozenset({"PATH", "HOME", "LANG"}),
    banned_vars=frozenset({
        "GH_TOKEN", "GITHUB_TOKEN", "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY",
        "HF_TOKEN", "DAYDREAM_TRAJECTORY_HUB_REPO", "DAYDREAM_ARCHIVE_DIR",
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "OPENROUTER_API_KEY", "PI_API_KEY",
    }),
    banned_prefixes=frozenset({
        "DAYDREAM_JUDGE_", "ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_", "OPENROUTER_", "PI_",
    }),
    claude_keep_vars=frozenset({
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    }),
    claude_exempt_vars=frozenset({"ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"}),
    claude_exempt_prefix=CLAUDE_KEEP_PREFIX,
)
