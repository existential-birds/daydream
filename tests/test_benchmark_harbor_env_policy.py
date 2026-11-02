"""Actual environment and rendered-job boundaries for the Harbor policy."""



from daydream.benchmark.harbor import env_policy
from daydream.benchmark.harbor.entrypoint import _sanitize_reviewer_environment

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


_EXPECTED_CONTAINER_KEEP = frozenset({
    "PATH", "HOME", "LANG", "PI_API_KEY", "HF_TOKEN", "DAYDREAM_TRAJECTORY_HUB_REPO", "DAYDREAM_ARCHIVE_DIR",
    "DAYDREAM_UNRELATED_PROBE", "DAYDREAM_REVIEW_BACKEND", "DAYDREAM_REVIEW_MODEL",
    "DAYDREAM_REVIEW_PROFILE_CANDIDATE", "DAYDREAM_REVIEW_EFFORT", "DAYDREAM_REVIEW_REPO_DIR",
    "DAYDREAM_REVIEW_ARTIFACT_PATH", "DAYDREAM_REVIEW_TRAJECTORY_PATH", "DAYDREAM_REVIEW_CASE_ID",
    "DAYDREAM_REVIEW_BASE_REF", "DAYDREAM_REVIEW_HEAD_REF", "DAYDREAM_REVIEW_IMMUTABLE_INPUT",
    "DAYDREAM_REVIEW_BUNDLE", "DAYDREAM_REVIEW_AGENT_DIR",
})


def test_container_sanitised_env_table() -> None:
    parent = _synthetic_parent()
    before = dict(parent)
    expected = {name: parent[name] for name in _EXPECTED_CONTAINER_KEEP}
    expected.update(PI_API_KEY="value:DAYDREAM_REVIEW_API_KEY", PI_PROVIDER="openrouter", PI_TELEMETRY="0")
    assert _sanitize_reviewer_environment(parent) == expected
    assert parent == before
