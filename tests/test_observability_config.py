"""Operator observability configuration, independent of reviewed repository settings."""

import sys

import pytest

from daydream.cli import _parse_args, _parse_improve_args
from daydream.observability.config import ObservabilityConfig, ObservabilityError, resolve_observability_config


@pytest.fixture(autouse=True)
def isolated_operator_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("DAYDREAM_TRACE_TO", "DAYDREAM_TRACE_CONTENT", "OTEL_SERVICE_NAME"):
        monkeypatch.delenv(key, raising=False)


def test_tracing_requires_explicit_destination() -> None:
    assert resolve_observability_config(environ={}) == ObservabilityConfig()
    assert resolve_observability_config(environ={"LANGSMITH_API_KEY": "opaque"}).destinations == ()


def test_operator_environment_and_cli_precedence() -> None:
    env = {
        "DAYDREAM_TRACE_TO": "langsmith, honeyhive",
        "DAYDREAM_TRACE_CONTENT": "metadata",
        "OTEL_SERVICE_NAME": "my-daydream",
    }
    assert resolve_observability_config(environ=env) == ObservabilityConfig(
        destinations=("langsmith", "honeyhive"),
        capture_content=False,
        service_name="my-daydream",
    )
    assert resolve_observability_config(destinations=["custom"], content="full", environ=env) == ObservabilityConfig(
        destinations=("custom",),
        service_name="my-daydream",
    )
    assert resolve_observability_config(destinations=[], environ=env).destinations == ()


def test_explicit_off_ignores_invalid_ambient_configuration() -> None:
    assert (
        resolve_observability_config(
            disabled=True,
            environ={
                "DAYDREAM_TRACE_TO": "broken secret value",
                "DAYDREAM_TRACE_CONTENT": "invalid",
                "OTEL_SERVICE_NAME": "\n",
            },
        )
        == ObservabilityConfig()
    )


@pytest.mark.parametrize(
    "env",
    [
        {"DAYDREAM_TRACE_TO": "langsmith,langsmith"},
        {"DAYDREAM_TRACE_TO": "langsmith,,otlp"},
        {"DAYDREAM_TRACE_TO": "https://secret@example.com"},
        {"DAYDREAM_TRACE_CONTENT": "secret"},
        {"OTEL_SERVICE_NAME": "\nsecret"},
    ],
)
def test_invalid_operator_settings_do_not_echo_values(env: dict[str, str]) -> None:
    with pytest.raises(ObservabilityError) as exc:
        resolve_observability_config(environ=env)
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("improve", [False, True])
def test_cli_trace_flags_reach_run_config(improve: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_TRACE_TO", "otlp")
    argv = [
        "/tmp/absent-daydream-config-fixture",
        "--trace-to",
        "langsmith",
        "--trace-to",
        "custom",
        "--trace-content",
        "metadata",
    ]
    if improve:
        config = _parse_improve_args(argv)
    else:
        monkeypatch.setattr(sys, "argv", ["daydream", *argv])
        config = _parse_args()
    assert config.observability == ObservabilityConfig(destinations=("langsmith", "custom"), capture_content=False)


@pytest.mark.parametrize("improve", [False, True])
def test_cli_rejects_conflicting_activation(improve: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    argv = ["/tmp/project", "--no-tracing", "--trace-to", "langsmith"]
    with pytest.raises(SystemExit) as exc:
        if improve:
            _parse_improve_args(argv)
        else:
            monkeypatch.setattr(sys, "argv", ["daydream", *argv])
            _parse_args()
    assert exc.value.code == 2


@pytest.mark.parametrize("improve", [False, True])
def test_cli_invalid_environment_is_parser_error(improve: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_TRACE_CONTENT", "invalid")
    with pytest.raises(SystemExit) as exc:
        if improve:
            _parse_improve_args(["/tmp/project"])
        else:
            monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project"])
            _parse_args()
    assert exc.value.code == 2
