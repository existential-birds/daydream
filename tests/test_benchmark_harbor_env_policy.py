"""Actual environment and rendered-job boundaries for the Harbor policy."""

from dataclasses import replace
from types import MappingProxyType

import pytest
import yaml

from daydream.benchmark.harbor import env_policy
from daydream.benchmark.harbor.agent import build_child_env
from daydream.benchmark.harbor.entrypoint import _sanitize_reviewer_environment
from daydream.benchmark.harbor.package import render_job_config, render_task_toml

_PROBES = frozenset({"OPENAI_API_KEY", "DAYDREAM_UNRELATED_PROBE"})


def _synthetic_parent(backend: str) -> dict[str, str]:
    parent = {name: f"value:{name}" for name in env_policy.declared_names() | _PROBES}
    parent.update(
        {
            f"{prefix}PROBE": "unselected-secret"
            for prefix in env_policy.HOST.banned_prefixes | env_policy.CONTAINER.scrub_prefixes
        }
    )
    parent.update(
        {
            "DAYDREAM_REVIEW_BACKEND": backend,
            "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
            "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        }
    )
    return parent


_EXPECTED_HOST_KEEP: dict[str, frozenset[str]] = {"pi": frozenset({
        "PATH", "HOME", "LANG", "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL", "DAYDREAM_REVIEW_API_KEY",
        "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_REVIEW_PROFILE_CANDIDATE",
        "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR", "DAYDREAM_REVIEW_ARTIFACT_PATH",
        "DAYDREAM_REVIEW_TRAJECTORY_PATH", "DAYDREAM_REVIEW_CASE_ID",
        "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
    }), "claude": frozenset({
        "PATH", "HOME", "LANG", "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL", "DAYDREAM_REVIEW_API_KEY",
        "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_REVIEW_PROFILE_CANDIDATE",
        "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR", "DAYDREAM_REVIEW_ARTIFACT_PATH",
        "DAYDREAM_REVIEW_TRAJECTORY_PATH", "DAYDREAM_REVIEW_CASE_ID",
        "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    }),
}

_EXPECTED_CONTAINER_KEEP: dict[str, frozenset[str]] = {"pi": frozenset({
        "PATH", "HOME", "LANG", "PI_API_KEY", "HF_TOKEN",
        "DAYDREAM_TRAJECTORY_HUB_REPO", "DAYDREAM_ARCHIVE_DIR", "DAYDREAM_UNRELATED_PROBE",
        "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL",
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE", "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR",
        "DAYDREAM_REVIEW_ARTIFACT_PATH", "DAYDREAM_REVIEW_TRAJECTORY_PATH",
        "DAYDREAM_REVIEW_CASE_ID", "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
    }), "claude": frozenset({"PATH", "HOME", "LANG", "HF_TOKEN",
        "DAYDREAM_TRAJECTORY_HUB_REPO", "DAYDREAM_ARCHIVE_DIR", "DAYDREAM_UNRELATED_PROBE",
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL",
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE", "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR",
        "DAYDREAM_REVIEW_ARTIFACT_PATH", "DAYDREAM_REVIEW_TRAJECTORY_PATH",
        "DAYDREAM_REVIEW_CASE_ID", "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
    }),
}


def test_oracle_render_path_applies_no_policy_view() -> None:
    oracle = yaml.safe_load(render_job_config(oracle=True).decode())
    assert oracle["agents"] == [{"name": "oracle"}]


@pytest.mark.parametrize("backend", ["pi", "claude"])
def test_host_child_env_table(backend: str) -> None:
    parent = _synthetic_parent(backend)
    before = dict(parent)
    assert build_child_env(parent, backend=backend) == {name: parent[name] for name in _EXPECTED_HOST_KEEP[backend]}
    assert parent == before


@pytest.mark.parametrize("backend", ["pi", "claude"])
def test_container_sanitised_env_table(backend: str) -> None:
    parent = _synthetic_parent(backend)
    before = dict(parent)
    expected = {name: parent[name] for name in _EXPECTED_CONTAINER_KEEP[backend]}
    if backend == "pi":
        expected.update(PI_API_KEY="value:DAYDREAM_REVIEW_API_KEY", PI_PROVIDER="openrouter", PI_TELEMETRY="0")
    assert _sanitize_reviewer_environment(parent, backend=backend) == expected
    assert parent == before


def test_task_toml_renderer_derives_its_injected_env_from_the_declaration(monkeypatch: pytest.MonkeyPatch,) -> None:
    extended = replace(env_policy.TASK,
        injected=MappingProxyType({**env_policy.TASK.injected, "DAYDREAM_REVIEW_PROBE": "{opaque_key}"}),
    )
    monkeypatch.setattr(env_policy, "TASK", extended)
    rendered = render_task_toml("case-abc123def456", reviewer_hosts=["api.anthropic.com"], judge_hosts=["openrouter.ai"]
    ).decode()
    assert rendered.index('DAYDREAM_REVIEW_CASE_ID = "case-abc123def456"') < rendered.index(
        'DAYDREAM_REVIEW_PROBE = "case-abc123def456"'
    )


def test_job_config_renderer_derives_its_env_from_the_declaration(monkeypatch: pytest.MonkeyPatch,) -> None:
    extended = replace(env_policy.RENDERER,
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
    reviewer = {
        "DAYDREAM_REVIEW_BACKEND": "${DAYDREAM_REVIEW_BACKEND:-pi}",
        "DAYDREAM_REVIEW_MODEL": "${DAYDREAM_REVIEW_MODEL}",
        "DAYDREAM_REVIEW_API_KEY": "${DAYDREAM_REVIEW_API_KEY:-}",
        "DAYDREAM_REVIEW_BASE_URL": "${DAYDREAM_REVIEW_BASE_URL:-}",
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE": "${DAYDREAM_REVIEW_PROFILE_CANDIDATE:-}",
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY:-}",
        "ANTHROPIC_AUTH_TOKEN": "${ANTHROPIC_AUTH_TOKEN:-}",
        "ANTHROPIC_BASE_URL": "${ANTHROPIC_BASE_URL:-}",
    }
    judge = {
        "DAYDREAM_JUDGE_PROVIDER": "${DAYDREAM_JUDGE_PROVIDER}",
        "DAYDREAM_JUDGE_MODEL": "${DAYDREAM_JUDGE_MODEL}",
        "DAYDREAM_JUDGE_API_KEY": "${DAYDREAM_JUDGE_API_KEY:-}",
        "DAYDREAM_JUDGE_BASE_URL": "${DAYDREAM_JUDGE_BASE_URL:-}",
        "CLAUDE_CODE_OAUTH_TOKEN": "${CLAUDE_CODE_OAUTH_TOKEN:-}",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}",
    }
    assert list(job["agents"][0]["env"].items()) == list(reviewer.items())
    assert list(job["verifier"]["env"].items()) == list(judge.items())


def test_container_sanitiser_derives_its_sets_from_the_declaration(monkeypatch: pytest.MonkeyPatch,) -> None:
    extended = replace(env_policy.CONTAINER, scrub_prefixes=env_policy.CONTAINER.scrub_prefixes | {"PROBE_"})
    monkeypatch.setattr(env_policy, "CONTAINER", extended)
    sanitized = _sanitize_reviewer_environment(
        {"DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api", "DAYDREAM_REVIEW_API_KEY": "k", "PROBE_X": "leak"},
        backend="pi",
    )
    assert "PROBE_X" not in sanitized


def test_host_builder_derives_its_sets_from_the_declaration(monkeypatch: pytest.MonkeyPatch,) -> None:
    extended = replace(env_policy.HOST, required_process_vars=env_policy.HOST.required_process_vars | {"PROBE_VAR"})
    monkeypatch.setattr(env_policy, "HOST", extended)
    child = build_child_env({"PATH": "/usr/bin", "PROBE_VAR": "kept"})
    assert child.get("PROBE_VAR") == "kept"
