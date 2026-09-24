"""Policy declarations and per-layer consumption for the harbor env policy."""

import ast
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml

from daydream.benchmark.harbor import agent, entrypoint, env_policy, package
from daydream.benchmark.harbor.agent import build_child_env
from daydream.benchmark.harbor.entrypoint import _sanitize_reviewer_environment
from daydream.benchmark.harbor.package import render_job_config, render_task_toml

_PROBES = frozenset({"OPENAI_API_KEY", "DAYDREAM_UNRELATED_PROBE"})


def _synthetic_parent(backend: str) -> dict[str, str]:
    parent = dict.fromkeys(env_policy.declared_names() | _PROBES, "probe")
    parent.update({
        "DAYDREAM_REVIEW_BACKEND": backend,
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
    })
    return parent


_EXPECTED_HOST_KEEP: dict[str, frozenset[str]] = {
    "pi": frozenset({
        "PATH", "HOME", "LANG",
        "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL", "DAYDREAM_REVIEW_API_KEY",
        "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_REVIEW_PROFILE_CANDIDATE",
        "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR", "DAYDREAM_REVIEW_ARTIFACT_PATH",
        "DAYDREAM_REVIEW_TRAJECTORY_PATH", "DAYDREAM_REVIEW_CASE_ID",
        "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
    }),
    "claude": frozenset({
        "PATH", "HOME", "LANG",
        "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL", "DAYDREAM_REVIEW_API_KEY",
        "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_REVIEW_PROFILE_CANDIDATE",
        "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR", "DAYDREAM_REVIEW_ARTIFACT_PATH",
        "DAYDREAM_REVIEW_TRAJECTORY_PATH", "DAYDREAM_REVIEW_CASE_ID",
        "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    }),
}

_EXPECTED_CONTAINER_KEEP: dict[str, frozenset[str]] = {
    "pi": frozenset({
        "PATH", "HOME", "LANG", "PI_API_KEY", "HF_TOKEN",
        "DAYDREAM_TRAJECTORY_HUB_REPO", "DAYDREAM_ARCHIVE_DIR", "DAYDREAM_UNRELATED_PROBE",
        "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL",
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE", "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR",
        "DAYDREAM_REVIEW_ARTIFACT_PATH", "DAYDREAM_REVIEW_TRAJECTORY_PATH",
        "DAYDREAM_REVIEW_CASE_ID", "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
    }),
    "claude": frozenset({
        "PATH", "HOME", "LANG", "HF_TOKEN",
        "DAYDREAM_TRAJECTORY_HUB_REPO", "DAYDREAM_ARCHIVE_DIR", "DAYDREAM_UNRELATED_PROBE",
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL",
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE", "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR",
        "DAYDREAM_REVIEW_ARTIFACT_PATH", "DAYDREAM_REVIEW_TRAJECTORY_PATH",
        "DAYDREAM_REVIEW_CASE_ID", "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
    }),
}

_EXPECTED_RENDERER_PRESENT = frozenset({
    "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL", "DAYDREAM_REVIEW_API_KEY",
    "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_REVIEW_PROFILE_CANDIDATE",
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "DAYDREAM_JUDGE_PROVIDER", "DAYDREAM_JUDGE_MODEL", "DAYDREAM_JUDGE_API_KEY",
    "DAYDREAM_JUDGE_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
})

# Names no consumer keeps: adding one to the declaration without recording its
# outcome fails test_every_declared_name_has_a_recorded_outcome.
_EXPECTED_DROPPED_EVERYWHERE = frozenset({
    "GH_TOKEN", "GITHUB_TOKEN", "GITHUB_ENTERPRISE_TOKEN", "GH_ENTERPRISE_TOKEN", "GH_HOST",
    "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY", "OPENROUTER_API_KEY", "ZAI_API_KEY",
    "NOUS_API_KEY", "DAYDREAM_SKILLS_DIR", "DAYDREAM_JUDGE_ALLOWED_HOSTS",
    "DAYDREAM_JUDGE_ARTIFACT_PATH", "DAYDREAM_JUDGE_OUT_PATH",
})


def test_every_declared_name_has_a_recorded_outcome() -> None:
    """Adding a name to the declaration without recording its outcome fails here."""
    recorded = (
        set().union(*_EXPECTED_HOST_KEEP.values())
        | set().union(*_EXPECTED_CONTAINER_KEEP.values())
        | _EXPECTED_RENDERER_PRESENT
        | _EXPECTED_DROPPED_EVERYWHERE
    ) - _PROBES
    assert recorded == env_policy.declared_names()


@pytest.mark.parametrize("backend", ["pi", "claude"])
def test_host_child_env_table(backend: str) -> None:
    child = build_child_env(_synthetic_parent(backend), backend=backend)
    assert set(child) == _EXPECTED_HOST_KEEP[backend]


@pytest.mark.parametrize("backend", ["pi", "claude"])
def test_container_sanitised_env_table(backend: str) -> None:
    sanitized = _sanitize_reviewer_environment(_synthetic_parent(backend), backend=backend)
    observed = set(sanitized) & (env_policy.declared_names() | _PROBES)
    assert observed == _EXPECTED_CONTAINER_KEEP[backend]


@pytest.mark.parametrize("backend", ["pi", "claude"])
def test_rendered_job_config_table(backend: str) -> None:
    job = yaml.safe_load(render_job_config(oracle=False).decode())
    present = set(job["agents"][0]["env"]) | set(job["verifier"]["env"])
    assert present & (env_policy.declared_names() | _PROBES) == _EXPECTED_RENDERER_PRESENT


def test_task_toml_renderer_derives_its_injected_env_from_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The [environment.env] block is built from the declaration's injection channel."""
    extended = replace(
        env_policy.TASK,
        injected=MappingProxyType(
            {**env_policy.TASK.injected, "DAYDREAM_REVIEW_PROBE": "{opaque_key}"}
        ),
    )
    monkeypatch.setattr(env_policy, "TASK", extended)
    rendered = render_task_toml(
        "case-abc123def456", reviewer_hosts=["api.anthropic.com"], judge_hosts=["openrouter.ai"]
    ).decode()
    assert rendered.index('DAYDREAM_REVIEW_CASE_ID = "case-abc123def456"') < rendered.index(
        'DAYDREAM_REVIEW_PROBE = "case-abc123def456"'
    )


def test_harbour_layers_declare_no_policy_name_literal() -> None:
    """No credential or control-plane name remains a bare literal outside the
    declaration (spec M4) — a re-export from env_policy is allowed, a local
    string equal to a policy name is not."""
    for module in (agent, entrypoint, package):
        assert module.__file__ is not None
        source = Path(module.__file__).read_text(encoding="utf-8")
        offenders = sorted(n for n in env_policy.declared_names() if f'"{n}"' in source)
        assert offenders == [], f"{module.__name__} restates policy names: {offenders}"


def test_job_config_renderer_derives_its_env_from_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A placeholder added to the declaration reaches the rendered bytes (spec M2)."""
    extended = replace(
        env_policy.RENDERER,
        judge_placeholders=MappingProxyType(
            {**env_policy.RENDERER.judge_placeholders, "DAYDREAM_JUDGE_PROBE": "${DAYDREAM_JUDGE_PROBE:-}"}
        ),
    )
    monkeypatch.setattr(env_policy, "RENDERER", extended)
    job = yaml.safe_load(render_job_config(oracle=False).decode())
    assert job["verifier"]["env"]["DAYDREAM_JUDGE_PROBE"] == "${DAYDREAM_JUDGE_PROBE:-}"


def test_rendered_env_key_order_is_unchanged() -> None:
    """The job config bytes are part of the compiled-tree contract; key order is load-bearing."""
    job = yaml.safe_load(render_job_config(oracle=False).decode())
    assert list(job["agents"][0]["env"]) == [
        "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL", "DAYDREAM_REVIEW_API_KEY",
        "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_REVIEW_PROFILE_CANDIDATE",
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    ]
    assert list(job["verifier"]["env"]) == [
        "DAYDREAM_JUDGE_PROVIDER", "DAYDREAM_JUDGE_MODEL", "DAYDREAM_JUDGE_API_KEY",
        "DAYDREAM_JUDGE_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    ]


def test_container_sanitiser_derives_its_sets_from_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the declaration's container view changes the sanitised map (spec M2)."""
    extended = replace(
        env_policy.CONTAINER,
        scrub_prefixes=env_policy.CONTAINER.scrub_prefixes | {"PROBE_"},
    )
    monkeypatch.setattr(env_policy, "CONTAINER", extended)
    sanitized = _sanitize_reviewer_environment(
        {
            "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
            "DAYDREAM_REVIEW_API_KEY": "k",
            "PROBE_X": "leak",
        },
        backend="pi",
    )
    assert "PROBE_X" not in sanitized


def test_host_builder_derives_its_sets_from_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the host view changes the builder's output, without a private copy."""
    extended = replace(
        env_policy.HOST,
        required_process_vars=env_policy.HOST.required_process_vars | {"PROBE_VAR"},
    )
    monkeypatch.setattr(env_policy, "HOST", extended)
    child = build_child_env({"PATH": "/usr/bin", "PROBE_VAR": "kept"})
    assert child.get("PROBE_VAR") == "kept"


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
