"""Operator observability configuration, independent of reviewed repository settings."""

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from daydream.commands.improve import _parse_improve_args
from daydream.commands.review import _parse_args
from daydream.observability.config import ObservabilityConfig, ObservabilityError, resolve_observability_config
from tests.harness.table_cases import isolated_case


@pytest.fixture(autouse=True)
def isolated_operator_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("DAYDREAM_TRACE_TO", "DAYDREAM_TRACE_CONTENT", "OTEL_SERVICE_NAME"):
        monkeypatch.delenv(key, raising=False)


def test_explicit_off_ignores_invalid_ambient_configuration() -> None:
    assert (resolve_observability_config(disabled=True,
            environ={"DAYDREAM_TRACE_TO": "broken secret value", "DAYDREAM_TRACE_CONTENT": "invalid",
                "OTEL_SERVICE_NAME": "\n",
            },
        )
        == ObservabilityConfig()
    )

def test_invalid_operator_settings_do_not_echo_values(subtests: pytest.Subtests) -> None:
    cases: Sequence[dict[str, str]] = [
        {"DAYDREAM_TRACE_TO": "langsmith,langsmith"},
        {"DAYDREAM_TRACE_TO": "langsmith,,otlp"},
        {"DAYDREAM_TRACE_TO": "https://secret@example.com"},
        {"DAYDREAM_TRACE_CONTENT": "secret"},
        {"OTEL_SERVICE_NAME": "\nsecret"},
    ]
    for case_index, env in enumerate(cases):
        with subtests.test(env=env):
            with pytest.raises(ObservabilityError) as exc:
                resolve_observability_config(environ=env)
            assert "secret" not in str(exc.value)


def test_cli_trace_flags_reach_run_config(tmp_path: Path, subtests: pytest.Subtests) -> None:
    case_root = tmp_path
    cases: Sequence[bool] = [False, True]
    for case_index, improve in enumerate(cases):
        with subtests.test(improve=improve):
            with isolated_case(case_root, str(case_index)) as sandbox:
                monkeypatch = sandbox.monkeypatch
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
                assert config.observability == ObservabilityConfig(
                    destinations=("langsmith", "custom"), capture_content=False
                )


def test_cli_rejects_conflicting_activation(tmp_path: Path, subtests: pytest.Subtests) -> None:
    case_root = tmp_path
    cases: Sequence[bool] = [False, True]
    for case_index, improve in enumerate(cases):
        with subtests.test(improve=improve):
            with isolated_case(case_root, str(case_index)) as sandbox:
                monkeypatch = sandbox.monkeypatch
                argv = ["/tmp/project", "--no-tracing", "--trace-to", "langsmith"]
                with pytest.raises(SystemExit) as exc:
                    if improve:
                        _parse_improve_args(argv)
                    else:
                        monkeypatch.setattr(sys, "argv", ["daydream", *argv])
                        _parse_args()
                assert exc.value.code == 2


def test_cli_invalid_environment_is_parser_error(tmp_path: Path, subtests: pytest.Subtests) -> None:
    case_root = tmp_path
    cases: Sequence[bool] = [False, True]
    for case_index, improve in enumerate(cases):
        with subtests.test(improve=improve):
            with isolated_case(case_root, str(case_index)) as sandbox:
                monkeypatch = sandbox.monkeypatch
                monkeypatch.setenv("DAYDREAM_TRACE_CONTENT", "invalid")
                with pytest.raises(SystemExit) as exc:
                    if improve:
                        _parse_improve_args(["/tmp/project"])
                    else:
                        monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project"])
                        _parse_args()
                assert exc.value.code == 2


# --- P18 Task 3: repository files cannot configure tracing ---------------------
