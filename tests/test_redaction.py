"""REDA-01..06 unit tests for ``daydream.trajectory.Redactor``.

Pattern: positive case (secret present, asserts replacement token in
output and raw value absent), negative case (clean input, asserts no
``[REDACTED_*]`` tokens appear). Plus REDA-04 surface coverage and
REDA-05 fail-safe.
"""

from __future__ import annotations

import pytest

from daydream.atif import Observation, ObservationResult, Step, ToolCall
from daydream.trajectory import Redactor, now_iso


def _user_step(message: str) -> Step:
    """Construct a minimal user Step with *message* (test helper)."""
    return Step(
        step_id=1,
        timestamp=now_iso(),
        source="user",
        message=message,
        extra={"daydream_phase": "review", "daydream_run_flow": "normal"},
    )


def _agent_step(
    message: str = "ok",
    reasoning_content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    observation: Observation | None = None,
) -> Step:
    """Construct a minimal agent Step (test helper)."""
    return Step(
        step_id=2,
        timestamp=now_iso(),
        source="agent",
        model_name="opus",
        message=message,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        observation=observation,
        extra={"daydream_phase": "review", "daydream_run_flow": "normal"},
    )


# ---- API key patterns (REDA-01) ----


def test_redactor_scrubs_openai_api_key() -> None:
    """REDA-01: sk-* tokens replaced with [REDACTED_API_KEY]."""
    out = Redactor().redact_step(_user_step("token=sk-test-12345abcdef done"))
    assert isinstance(out.message, str)
    assert "sk-test-12345abcdef" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


def test_redactor_scrubs_github_token() -> None:
    """REDA-01: ghp_* tokens replaced with [REDACTED_API_KEY]."""
    out = Redactor().redact_step(_user_step("auth=ghp_test123abcdef bearer"))
    assert isinstance(out.message, str)
    assert "ghp_test123abcdef" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


def test_redactor_scrubs_slack_bot_token() -> None:
    """REDA-01: xoxb-* tokens replaced with [REDACTED_API_KEY]."""
    out = Redactor().redact_step(_user_step("slack=xoxb-test456abcdef"))
    assert isinstance(out.message, str)
    assert "xoxb-test456abcdef" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


def test_redactor_scrubs_aws_access_key() -> None:
    """REDA-01: AKIA-prefixed AWS keys replaced with [REDACTED_API_KEY]."""
    out = Redactor().redact_step(_user_step("aws=AKIA0000TESTKEY00000"))
    assert isinstance(out.message, str)
    assert "AKIA0000TESTKEY00000" not in out.message
    assert "[REDACTED_API_KEY]" in out.message


# ---- JWT pattern (REDA-01) ----


def test_redactor_scrubs_jwt_token() -> None:
    """REDA-01: eyJ JWT tokens replaced with [REDACTED_JWT]."""
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.aBcDeF12345"
    out = Redactor().redact_step(_user_step(f"bearer {jwt}"))
    assert isinstance(out.message, str)
    assert jwt not in out.message
    assert "[REDACTED_JWT]" in out.message


# ---- Git URL credentials (REDA-01: explicit "git remote URLs with embedded credentials") ----


def test_redactor_scrubs_git_url_credentials() -> None:
    """REDA-01: https://user:token@host credentials are scrubbed; host+path preserved."""
    url = "git+https://oauth2:ghp_realtoken123@github.com/user/repo.git"
    out = Redactor().redact_step(_user_step(url))
    assert isinstance(out.message, str)
    # Both username and credential disappear from the output.
    assert "ghp_realtoken123" not in out.message
    assert "oauth2" not in out.message
    # Both replacement tokens appear between https:// and @.
    assert "[REDACTED_USER]" in out.message
    assert "[REDACTED_API_KEY]" in out.message
    # Host and path preserved (debugging/replay value).
    assert "github.com" in out.message
    assert "/user/repo.git" in out.message


# ---- Username path patterns (REDA-02) ----


def test_redactor_scrubs_macos_username_path() -> None:
    """REDA-02: /Users/<name>/ → /Users/[REDACTED_USER]/ (project-relative tail preserved)."""
    out = Redactor().redact_step(_user_step("path=/Users/ka/github/proj/app.py"))
    assert isinstance(out.message, str)
    assert "/Users/ka" not in out.message
    assert "/Users/[REDACTED_USER]" in out.message
    assert "github/proj/app.py" in out.message


def test_redactor_scrubs_linux_username_path() -> None:
    """REDA-02: /home/<name>/ → /home/[REDACTED_USER]/."""
    out = Redactor().redact_step(_user_step("path=/home/alice/foo/bar"))
    assert isinstance(out.message, str)
    assert "/home/alice" not in out.message
    assert "/home/[REDACTED_USER]" in out.message
    assert "foo/bar" in out.message


def test_redactor_scrubs_windows_username_path() -> None:
    """REDA-02: C:\\Users\\<name>\\ → C:\\Users\\[REDACTED_USER]\\."""
    out = Redactor().redact_step(_user_step("path=C:\\Users\\bob\\repo"))
    assert isinstance(out.message, str)
    assert "Users\\bob" not in out.message
    assert "[REDACTED_USER]" in out.message


# ---- Env-var pattern (REDA-03) ----


def test_redactor_scrubs_env_var_secret() -> None:
    """REDA-03: KEY=value secret-keyname env vars get value redacted, key preserved."""
    out = Redactor().redact_step(_user_step("OPENAI_API_KEY=sk-realvalue123"))
    assert isinstance(out.message, str)
    assert "sk-realvalue123" not in out.message
    assert "OPENAI_API_KEY=[REDACTED_ENV_VAR]" in out.message


def test_redactor_scrubs_env_var_password() -> None:
    """REDA-03: PASSWORD= env vars get value redacted, key preserved."""
    out = Redactor().redact_step(_user_step("DB_PASSWORD=hunter2"))
    assert isinstance(out.message, str)
    assert "hunter2" not in out.message
    assert "DB_PASSWORD=[REDACTED_ENV_VAR]" in out.message


# ---- Negative cases ----


def test_redactor_preserves_non_secret_env_vars() -> None:
    """REDA-03 negative: DEBUG=true and APP_NAME=foo pass through unredacted."""
    out = Redactor().redact_step(_user_step("DEBUG=true\nAPP_NAME=myproject"))
    assert isinstance(out.message, str)
    assert "DEBUG=true" in out.message
    assert "APP_NAME=myproject" in out.message
    assert "[REDACTED" not in out.message


def test_redactor_preserves_clean_paths() -> None:
    """Negative: relative paths without /Users//home/ prefix pass through."""
    out = Redactor().redact_step(_user_step("./src/app.py"))
    assert isinstance(out.message, str)
    assert out.message == "./src/app.py"


def test_redactor_preserves_clean_urls() -> None:
    """Negative: URLs without embedded credentials pass through unchanged."""
    out = Redactor().redact_step(_user_step("https://github.com/user/repo"))
    assert isinstance(out.message, str)
    assert out.message == "https://github.com/user/repo"


# ---- Surface coverage (REDA-04) ----


def test_redactor_applies_to_reasoning_content() -> None:
    """REDA-04: secrets inside Step.reasoning_content are redacted."""
    step = _agent_step(reasoning_content="thought: sk-test-secret123abc")
    out = Redactor().redact_step(step)
    assert out.reasoning_content is not None
    assert "sk-test-secret123abc" not in out.reasoning_content
    assert "[REDACTED_API_KEY]" in out.reasoning_content


def test_redactor_applies_to_tool_call_arguments() -> None:
    """REDA-04: secrets inside ToolCall.arguments values are redacted."""
    call = ToolCall(
        tool_call_id="t1",
        function_name="Bash",
        arguments={"command": "echo sk-test-secret123abc"},
    )
    step = _agent_step(tool_calls=[call])
    out = Redactor().redact_step(step)
    assert out.tool_calls is not None
    args_str = str(out.tool_calls[0].arguments)
    assert "sk-test-secret123abc" not in args_str
    assert "[REDACTED_API_KEY]" in args_str


def test_redactor_applies_to_observation_content() -> None:
    """REDA-04: secrets inside ObservationResult.content are redacted."""
    obs = Observation(
        results=[ObservationResult(source_call_id="t1", content="leaked /Users/ka/.ssh/id_rsa")],
    )
    step = _agent_step(observation=obs)
    out = Redactor().redact_step(step)
    assert out.observation is not None
    first_content = out.observation.results[0].content
    assert isinstance(first_content, str)
    assert "/Users/ka" not in first_content
    assert "[REDACTED_USER]" in first_content


# ---- Fail-safe (REDA-05) ----


def test_redactor_failure_mode_replaces_with_redaction_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REDA-05: when an internal regex raises, the field becomes [REDACTION_FAILED] — never raw."""
    from daydream import trajectory as traj_mod

    class _BoomPattern:
        def sub(self, *_args: object, **_kwargs: object) -> str:
            raise RuntimeError("boom")

    # Replace the first rule so _redact_text raises on the first call.
    original_rules = traj_mod._REDACTION_RULES
    boom_rules = ((_BoomPattern(), "[REDACTED_API_KEY]"), *original_rules[1:])
    monkeypatch.setattr(traj_mod, "_REDACTION_RULES", boom_rules)

    out = Redactor().redact_step(_user_step("OPENAI_API_KEY=sk-leakthis123"))
    assert isinstance(out.message, str)
    assert "sk-leakthis123" not in out.message
    assert "[REDACTION_FAILED]" in out.message
