"""Policy declarations and per-layer consumption for the harbor env policy."""

import ast
from pathlib import Path
from types import MappingProxyType

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
    assert host.posture == "allowlist"
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


def test_container_channel_declares_the_container_scrub_sets() -> None:
    """The container channel is today's scrub-list posture, verbatim (spec D3)."""
    container = env_policy.CONTAINER
    assert container.posture == "scrub-list"
    assert container.github_credential_vars == frozenset({
        "GITHUB_TOKEN", "GH_TOKEN", "GITHUB_ENTERPRISE_TOKEN", "GH_ENTERPRISE_TOKEN",
        "GH_HOST", "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY",
    })
    assert container.unselected_pi_credentials == frozenset({"ZAI_API_KEY", "NOUS_API_KEY"})
    assert container.scrub_prefixes == frozenset({
        "DAYDREAM_JUDGE_", "ANTHROPIC_", "CLAUDE_CODE_", "OPENAI_", "OPENROUTER_", "PI_",
        "DAYDREAM_APP_",
    })
    assert container.control_plane_aliases == frozenset({
        "DAYDREAM_REVIEW_API_KEY", "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_SKILLS_DIR",
    })
    assert container.github_subprocess_drops == frozenset({
        "PI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    })


def test_render_task_and_judge_channels_declare_their_names() -> None:
    assert dict(env_policy.RENDERER.reviewer_placeholders) == {
        "DAYDREAM_REVIEW_BACKEND": "${DAYDREAM_REVIEW_BACKEND:-pi}",
        "DAYDREAM_REVIEW_MODEL": "${DAYDREAM_REVIEW_MODEL}",
        "DAYDREAM_REVIEW_API_KEY": "${DAYDREAM_REVIEW_API_KEY:-}",
        "DAYDREAM_REVIEW_BASE_URL": "${DAYDREAM_REVIEW_BASE_URL:-}",
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE": "${DAYDREAM_REVIEW_PROFILE_CANDIDATE:-}",
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-}",
        "ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_AUTH_TOKEN:-}",
        "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL:-}",
    }
    assert dict(env_policy.RENDERER.judge_placeholders) == {
        "DAYDREAM_JUDGE_PROVIDER": "${DAYDREAM_JUDGE_PROVIDER}",
        "DAYDREAM_JUDGE_MODEL": "${DAYDREAM_JUDGE_MODEL}",
        "DAYDREAM_JUDGE_API_KEY": "${DAYDREAM_JUDGE_API_KEY:-}",
        "DAYDREAM_JUDGE_BASE_URL": "${DAYDREAM_JUDGE_BASE_URL:-}",
        "CLAUDE_CODE_OAUTH_TOKEN": "${CLAUDE_CODE_OAUTH_TOKEN:-}",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}",
    }
    assert dict(env_policy.TASK.injected) == {
        "DAYDREAM_REVIEW_CASE_ID": "{opaque_key}",
        "DAYDREAM_REVIEW_BASE_REF": "base",
        "DAYDREAM_REVIEW_HEAD_REF": "head",
    }
    assert env_policy.JUDGE.renderer_emitted == (
        "DAYDREAM_JUDGE_PROVIDER", "DAYDREAM_JUDGE_MODEL", "DAYDREAM_JUDGE_API_KEY",
        "DAYDREAM_JUDGE_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    )
    assert env_policy.JUDGE.host_supplied == frozenset({
        "DAYDREAM_JUDGE_ALLOWED_HOSTS", "DAYDREAM_JUDGE_ARTIFACT_PATH", "DAYDREAM_JUDGE_OUT_PATH",
    })
    assert env_policy.JUDGE.prefix == "DAYDREAM_JUDGE_"
    assert tuple(env_policy.RENDERER.reviewer_placeholders) == (
        "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL", "DAYDREAM_REVIEW_API_KEY",
        "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_REVIEW_PROFILE_CANDIDATE",
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    )
    assert tuple(env_policy.RENDERER.judge_placeholders) == env_policy.JUDGE.renderer_emitted
    assert tuple(env_policy.TASK.injected) == (
        "DAYDREAM_REVIEW_CASE_ID", "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
    )
    assert all(isinstance(view, MappingProxyType) for view in (
        env_policy.REVIEWER_CONTROL_PLANE, env_policy.RENDERER.reviewer_placeholders,
        env_policy.RENDERER.judge_placeholders, env_policy.TASK.injected,
    ))
    assert dict(env_policy.REVIEWER_CONTROL_PLANE) == {
        "DAYDREAM_REVIEW_BACKEND": "operator",
        "DAYDREAM_REVIEW_MODEL": "operator",
        "DAYDREAM_REVIEW_API_KEY": "operator",
        "DAYDREAM_REVIEW_BASE_URL": "operator",
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE": "operator",
        "DAYDREAM_REVIEW_EFFORT": "host",
        "DAYDREAM_REVIEW_REPO_DIR": "defaulted",
        "DAYDREAM_REVIEW_ARTIFACT_PATH": "defaulted",
        "DAYDREAM_REVIEW_TRAJECTORY_PATH": "defaulted",
        "DAYDREAM_REVIEW_CASE_ID": "injected",
        "DAYDREAM_REVIEW_BASE_REF": "injected",
        "DAYDREAM_REVIEW_HEAD_REF": "injected",
    }


def test_declared_names_is_every_name_the_policy_knows_about() -> None:
    assert env_policy.declared_names() == frozenset({
        "PATH", "HOME", "LANG",
        "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_ENTERPRISE_TOKEN", "GH_ENTERPRISE_TOKEN",
        "GH_HOST", "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY", "DAYDREAM_SKILLS_DIR",
        "HF_TOKEN", "DAYDREAM_TRAJECTORY_HUB_REPO", "DAYDREAM_ARCHIVE_DIR",
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "OPENROUTER_API_KEY", "PI_API_KEY", "ZAI_API_KEY", "NOUS_API_KEY",
        "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL", "DAYDREAM_REVIEW_API_KEY",
        "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_REVIEW_PROFILE_CANDIDATE",
        "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR", "DAYDREAM_REVIEW_ARTIFACT_PATH",
        "DAYDREAM_REVIEW_TRAJECTORY_PATH", "DAYDREAM_REVIEW_CASE_ID",
        "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
        "DAYDREAM_JUDGE_PROVIDER", "DAYDREAM_JUDGE_MODEL", "DAYDREAM_JUDGE_API_KEY",
        "DAYDREAM_JUDGE_BASE_URL", "DAYDREAM_JUDGE_ALLOWED_HOSTS",
        "DAYDREAM_JUDGE_ARTIFACT_PATH", "DAYDREAM_JUDGE_OUT_PATH",
        "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    })
