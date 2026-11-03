"""Actual environment and rendered-job boundaries for the Harbor policy."""

import yaml

from daydream.benchmark.harbor import env_policy
from daydream.benchmark.harbor.agent import build_child_env
from daydream.benchmark.harbor.entrypoint import _sanitize_reviewer_environment
from daydream.benchmark.harbor.package import render_job_config

_PROBES = frozenset({"OPENAI_API_KEY", "DAYDREAM_UNRELATED_PROBE"})


def _synthetic_parent() -> dict[str, str]:
    parent = {name: f"value:{name}" for name in env_policy.declared_names() | _PROBES}
    parent.update(
        {
            f"{prefix}PROBE": "unselected-secret"
            for prefix in env_policy.HOST.banned_prefixes | env_policy.CONTAINER.scrub_prefixes
        }
    )
    parent.update(
        {
            "DAYDREAM_REVIEW_BACKEND": "pi",
            "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
            "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        }
    )
    return parent


_EXPECTED_HOST_KEEP = frozenset({
    "PATH", "HOME", "LANG", "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL", "DAYDREAM_REVIEW_API_KEY",
    "DAYDREAM_REVIEW_BASE_URL", "DAYDREAM_REVIEW_PROFILE_CANDIDATE", "DAYDREAM_REVIEW_EFFORT",
    "DAYDREAM_REVIEW_REPO_DIR", "DAYDREAM_REVIEW_ARTIFACT_PATH", "DAYDREAM_REVIEW_TRAJECTORY_PATH",
    "DAYDREAM_REVIEW_CASE_ID", "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF",
    "DAYDREAM_REVIEW_IMMUTABLE_INPUT", "DAYDREAM_REVIEW_BUNDLE", "DAYDREAM_REVIEW_AGENT_DIR",
})

_EXPECTED_CONTAINER_KEEP = frozenset({
    "PATH", "HOME", "LANG", "PI_API_KEY", "HF_TOKEN", "DAYDREAM_TRAJECTORY_HUB_REPO", "DAYDREAM_ARCHIVE_DIR",
    "DAYDREAM_UNRELATED_PROBE", "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL",
    "DAYDREAM_REVIEW_PROFILE_CANDIDATE", "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR",
    "DAYDREAM_REVIEW_ARTIFACT_PATH", "DAYDREAM_REVIEW_TRAJECTORY_PATH", "DAYDREAM_REVIEW_CASE_ID",
    "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF", "DAYDREAM_REVIEW_IMMUTABLE_INPUT",
    "DAYDREAM_REVIEW_BUNDLE", "DAYDREAM_REVIEW_AGENT_DIR",
})


def test_host_and_container_env_tables() -> None:
    """The host builder keeps hub/archive/judge credentials out; the container scrub maps the reviewer key."""
    parent = _synthetic_parent()
    before = dict(parent)
    assert build_child_env(parent) == {name: parent[name] for name in _EXPECTED_HOST_KEEP}
    expected = {name: parent[name] for name in _EXPECTED_CONTAINER_KEEP}
    expected.update(PI_API_KEY="value:DAYDREAM_REVIEW_API_KEY", PI_PROVIDER="openrouter", PI_TELEMETRY="0")
    assert _sanitize_reviewer_environment(parent) == expected
    assert parent == before


def test_rendered_env_key_order_is_unchanged() -> None:
    """The job config bytes are part of the compiled-tree contract; key order is load-bearing.

    The verifier receives only judge variables; unselected alternatives are optional (#979)
    while provider/model selections stay required.
    """
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
    oracle = yaml.safe_load(render_job_config(oracle=True).decode())
    assert oracle["agents"] == [{"name": "oracle"}]
    assert oracle["verifier"] == job["verifier"]
