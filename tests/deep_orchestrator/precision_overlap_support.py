"""Provider-only barriers and findings for precision adjudication runner tests."""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from time import monotonic
from typing import Any

import anyio

from daydream.backends import AgentEvent, ContinuationToken, ResultEvent
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
from tests.test_deep_orchestrator import _record


class PrecisionOverlapBackend(EmptyReviewBackend):
    """Block only external adjudication streams; the runner owns all persisted state."""

    def __init__(self, target: Path, *, blocked: bool = False,
                 fault: tuple[str, str] | None = None) -> None:
        def finding(description: str, file: str, line: int, severity: str) -> dict[str, Any]:
            return _record(description=description, file=file, line=line, severity=severity,
                           confidence="MEDIUM", rationale="Concrete changed behavior", evidence=f"{file}:{line}")

        super().__init__(target, forbid_merge=False, forbid_supervise=False, review_by_stack={
            "generic-fallback": [finding("Dismiss early docs finding", "README.md", 3, "high")],
            "python": [finding("Confirmed Python defect", "api.py", 2, "high"),
                       finding("Drop same-location Python nit", "api.py", 2, "low")],
            "react": [finding("Drop React rendering concern", "App.tsx", 1, "low"),
                      finding("Keep React accessibility defect", "App.tsx", 1, "low")],
            "structure": [finding("Module layering violates the service boundary", "api.py", 1, "low"),
                          finding("Repeated dependency policy needs a shared abstraction", "api.py", 1, "low")],
        })
        self.merge_echo_records = True
        self.arbiter_session_id = "precision-arbiter-session"
        self.fault = fault
        self.started = {kind: anyio.Event() for kind in ("arbiter", "suppression")}
        self.release = {kind: anyio.Event() for kind in self.started}
        self.closed = {kind: anyio.Event() for kind in self.started}
        if not blocked:
            for event in self.release.values():
                event.set()
        self.inputs: dict[str, list[dict[str, Any]]] = {}
        self.input_paths: dict[str, Path] = {}
        self.completed: list[str] = []
        self.intervals: dict[str, tuple[float, float]] = {}

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        lower = prompt.lower()
        kind = ("arbiter" if "you are the arbiter" in lower else
                "suppression" if "you are the suppression reviewer" in lower else None)
        if kind is None:
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event
            return
        self.calls.append({"cwd": cwd, "prompt": prompt, "model": self.model, **kwargs})
        match = re.search(r"listed in (\S*(?:arbiter[-\w]*|suppression)-input\.json)", prompt)
        assert match is not None, "adjudicator must receive its real persisted input"
        path = Path(match.group(1))
        records = json.loads(path.read_text())
        self.inputs[kind] = records
        self.input_paths[kind] = path
        started = monotonic()
        self.started[kind].set()
        try:
            await self.release[kind].wait()
            if self.fault == (kind, "raise"):
                raise RuntimeError(f"unavailable {kind} provider")
            id_field = "arb_id" if kind == "arbiter" else "sup_id"
            verdicts = [{id_field: row[id_field],
                         "keep": (row["description"] != "Dismiss early docs finding" if kind == "arbiter"
                                  else not row["description"].startswith("Drop")),
                         "description": f"Confirmed: {row['description']}",
                         "severity": "low", "confidence": "MEDIUM", "rationale": "Checked source behavior",
                         "evidence": row["evidence"]} for row in records]
            if self.fault == (kind, "omitted"):
                verdicts = []
            elif self.fault == (kind, "foreign-id"):
                verdicts = [dict(row, **{id_field: row[id_field] + 1000}) for row in verdicts]
            # Provider order has no relationship to host record/source identity.
            output: Any = {"findings": "invalid"} if self.fault == (kind, "malformed") else {
                "findings": list(reversed(verdicts))}
            yield ResultEvent(structured_output=output,
                              continuation=ContinuationToken(backend="claude", data={
                                  "session_id": self.arbiter_session_id}) if kind == "arbiter" else None)
            self.completed.append(kind)
        finally:
            self.intervals[kind] = (started, monotonic())
            self.closed[kind].set()
