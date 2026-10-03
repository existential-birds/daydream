"""Fail-closed redaction of the ATIF text and tool-result surfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from daydream import ui
from daydream.atif import Observation, ObservationResult, Step
from daydream.redaction import (
    _is_sensitive_key as _is_sensitive_key,
    redact_structured_text as redact_structured_text,
    redact_value as redact_value,
)

_console = ui.create_console()


class Redactor:
    """Redact ATIF text and nested key-sensitive values; failures replace content with markers."""

    def _redact_optional_text(self, value: str | None) -> str | None:
        """Redact a possibly-None text field; degrade to [REDACTION_FAILED] on error."""
        if value is None:
            return None
        return redact_structured_text(value)

    def _redact_text_parts(self, parts: Sequence[Any]) -> list[Any]:
        """Redact text parts in a message or observation, preserving non-text parts."""
        return [
            part.model_copy(update={"text": self._redact_optional_text(part.text)}) if part.type == "text" else part
            for part in parts
        ]

    def _redact_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Redact nested sensitive keys while preserving shape; failures replace only the affected key."""
        out: dict[str, Any] = {}
        for key, val in arguments.items():
            try:
                out[key] = redact_value(val, _is_sensitive_key(key) if isinstance(key, str) else False)
            except Exception:  # noqa: BLE001 - REDA-05 redact-or-omit
                out[key] = "[REDACTION_FAILED]"
        return out

    def _redact_observation(self, observation: Observation | None) -> Observation | None:
        """Redact every string-valued ObservationResult.content in *observation*."""
        if observation is None:
            return None
        new_results: list[ObservationResult] = []
        for r in observation.results:
            new_content: Any = r.content
            if isinstance(r.content, str):
                try:
                    new_content = redact_structured_text(r.content)
                except Exception:  # noqa: BLE001 - REDA-05 redact-or-omit
                    new_content = "[REDACTION_FAILED]"
            elif isinstance(r.content, list):
                new_content = self._redact_text_parts(r.content)
            new_results.append(r.model_copy(update={"content": new_content}))
        return observation.model_copy(update={"results": new_results})

    def redact_step(self, step: Step) -> Step:
        """Return a redacted copy; any uncaught failure wipes all text-bearing surfaces."""
        try:
            updates: dict[str, Any] = {}
            if isinstance(step.message, str):
                updates["message"] = self._redact_optional_text(step.message)
            elif isinstance(step.message, list):
                updates["message"] = self._redact_text_parts(step.message)
            if step.reasoning_content is not None:
                updates["reasoning_content"] = self._redact_optional_text(step.reasoning_content)
            if step.tool_calls is not None:
                redacted_calls = [
                    tc.model_copy(update={"arguments": self._redact_arguments(tc.arguments)}) for tc in step.tool_calls
                ]
                updates["tool_calls"] = redacted_calls
            if step.observation is not None:
                updates["observation"] = self._redact_observation(step.observation)
            if not updates:
                return step
            return step.model_copy(update=updates)
        except Exception as exc:  # noqa: BLE001 - REDA-05 redact-or-omit (top-level fallback)
            ui.print_warning(_console, f"Redactor failure: {type(exc).__name__}")
            # Wipe every text-bearing surface — partial wipes leak secrets
            # if redaction failed mid-arguments / mid-observation.
            safe_updates: dict[str, Any] = {"message": "[REDACTION_FAILED]"}
            if step.reasoning_content is not None:
                safe_updates["reasoning_content"] = "[REDACTION_FAILED]"
            if step.tool_calls is not None:
                safe_updates["tool_calls"] = [
                    tc.model_copy(update={"arguments": {"_redaction": "[REDACTION_FAILED]"}}) for tc in step.tool_calls
                ]
            if step.observation is not None:
                safe_updates["observation"] = step.observation.model_copy(
                    update={
                        "results": [
                            r.model_copy(update={"content": "[REDACTION_FAILED]"}) for r in step.observation.results
                        ]
                    }
                )
            return step.model_copy(update=safe_updates)
