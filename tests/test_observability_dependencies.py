"""Fresh interpreters: stale editable installs fail tracing actionably without breaking the off path or CLI."""

import os
import subprocess
import sys
import textwrap


def test_missing_tracing_dependency() -> None:
    # A fresh interpreter avoids SDK imports cached by other tests. Blocking the optional
    # tracing packages reproduces an editable checkout whose dependencies were not refreshed.
    result = subprocess.run([sys.executable, "-c",
            textwrap.dedent("""\
                import importlib.abc
                import sys

                blocked = (
                    "opentelemetry.exporter.otlp.proto.grpc", "opentelemetry.exporter.otlp.proto.http", "traceloop"
                )

                class MissingDependency(importlib.abc.MetaPathFinder):
                    def find_spec(self, fullname, path, target=None):
                        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked):
                            raise ModuleNotFoundError("private import diagnostic", name=fullname)

                sys.meta_path.insert(0, MissingDependency())

                # A frozen-evidence consumer imported first must still let the CLI load (no import cycle),
                # and neither may need tracing packages.
                from daydream.training.record_evidence import sessions_from_snapshot
                from daydream.cli import main

                import anyio
                from daydream.extensions import build_registry
                from daydream.observability.config import ObservabilityConfig, ObservabilityError
                from daydream.observability.runtime import current_session, trace_run

                registry = build_registry()
                assert set(registry.trace_exporter_names()) >= {"langsmith", "honeyhive", "otlp"}

                async def run(destination):
                    config = ObservabilityConfig(destinations=() if destination == "off" else (destination,))
                    try:
                        async with trace_run(config, registry, flow="review"):
                            assert destination == "off", "agent work must not start with missing dependencies"
                    except ObservabilityError as exc:
                        assert destination != "off"
                        message = str(exc)
                        assert "tracing dependencies" in message.lower(), message
                        assert "uv tool install --reinstall --editable ." in message, message
                        assert "uv run daydream" in message, message
                        assert "private import diagnostic" not in message, message
                    else:
                        assert destination == "off"
                    assert current_session() is None

                for destination in ("off", "otlp"):
                    anyio.run(run, destination)
                """),
        ], env={key: value for key, value in os.environ.items() if not key.startswith(("OTEL_", "_OTEL_"))},
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
