"""Harbor control-plane environment policy declarations.

``HOST`` records the host-side child-env allowlist for
``agent.build_child_env``; ``CONTAINER`` records the container-side scrub-list
for ``entrypoint._sanitize_reviewer_environment``. The two-stage scrub is
intentional defence in depth: the container can see Harbor-injected names the
host builder never processed. The renderer and task-TOML injection views are
read by ``package.render_job_config`` and ``package.render_task_toml``
respectively. The judge channel belongs to the packaged verifier asset.
These other channel views are added separately.

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
SKILLS_DIR_ENV = "DAYDREAM_SKILLS_DIR"


@dataclass(frozen=True)
class HostChannel:
    """Names allowed, required, or excluded by the host child-env builder."""

    posture: str
    keep_prefixes: tuple[str, ...]
    required_process_vars: frozenset[str]
    banned_vars: frozenset[str]
    banned_prefixes: frozenset[str]
    claude_keep_vars: frozenset[str]
    claude_exempt_vars: frozenset[str]
    claude_exempt_prefix: str


HOST = HostChannel(
    posture="allowlist",
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


@dataclass(frozen=True)
class ContainerChannel:
    """Names removed during the container scrub and before GitHub subprocesses."""

    posture: str
    github_credential_vars: frozenset[str]
    unselected_pi_credentials: frozenset[str]
    scrub_prefixes: frozenset[str]
    control_plane_aliases: frozenset[str]
    github_subprocess_drops: frozenset[str]


CONTAINER = ContainerChannel(
    posture="scrub-list",
    github_credential_vars=frozenset({
        "GITHUB_TOKEN", "GH_TOKEN", "GITHUB_ENTERPRISE_TOKEN", "GH_ENTERPRISE_TOKEN",
        "GH_HOST", "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY",
    }),
    unselected_pi_credentials=frozenset({"ZAI_API_KEY", "NOUS_API_KEY"}),
    scrub_prefixes=frozenset({
        "DAYDREAM_JUDGE_", "ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_", "OPENROUTER_", "PI_",
        "DAYDREAM_APP_",
    }),
    control_plane_aliases=frozenset({
        "DAYDREAM_REVIEW_API_KEY", "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_SKILLS_DIR",
    }),
    github_subprocess_drops=frozenset({
        "PI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    }),
)
