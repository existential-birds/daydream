"""Cross-stack merge prompt + invocation tests (D-23..D-27, D-38)."""
from __future__ import annotations

import json
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import ResultEvent
from daydream.deep.artifacts import (
    DeepArtifact,
    deep_dir,
)
from daydream.deep.prompts import build_merge_prompt
from daydream.extensions import Registry
from daydream.extensions.builtins import register_builtins
from daydream.phases import phase_cross_stack_merge
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend


def _empty_merge_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create the empty records/alternatives files the empty-merge guard consumes."""
    dd = deep_dir(tmp_path, allow_standalone=True)
    records = dd / "stack-python-records.json"
    records.write_text('{"issues": []}')
    alternatives = dd / "alternatives.json"
    alternatives.write_text("[]")
    return dd, records, alternatives


async def test_empty_merge_requires_completed_inputs_and_builtin_contract(
    tmp_path: Path, make_work: Callable[..., WorkContext], monkeypatch: pytest.MonkeyPatch,
) -> None:
    for input_kind in ("missing-records", "missing-alternatives", "malformed-records",
                       "wrong-alternatives-shape", "custom-builder"):
        case = tmp_path / input_kind
        case.mkdir()
        with monkeypatch.context() as patch:
            dd, records, alternatives = _empty_merge_inputs(case)
            structural: Path | None = None
            if input_kind.startswith("missing-"):
                (records if input_kind == "missing-records" else alternatives).unlink()
            elif input_kind.startswith("malformed-"):
                (records if input_kind == "malformed-records" else alternatives).write_text("{bad json")
            elif input_kind.startswith("wrong-"):
                alternatives.write_text('{"issues": null}')
            elif input_kind == "custom-builder":
                registry = Registry()
                register_builtins(registry)
                structural = dd / "stack-structure-records.json"
                structural.write_text(json.dumps({"issues": [{
                    "id": 1, "uid": "structure:1", "file": "design.py", "line": 0,
                    "description": "Retained structural concern",
                    "evidence": "Dependency violates the documented boundary",
                    "confidence": "HIGH", "severity": "medium", "rationale": "Module coupling",
                }]}))

                def custom_builder(**kwargs: Any) -> str:
                    assert kwargs["structural_records_path"] == structural
                    return build_merge_prompt(**kwargs)

                registry.override_prompt("merge", custom_builder)
                patch.setattr("daydream.extensions.loader._REGISTRY_VAR",
                              ContextVar("merge-registry", default=registry))
            backend = ScriptedBackend(events=[ResultEvent(structured_output={"items": []}, continuation=None)])

            await phase_cross_stack_merge(
                backend, make_work(case), per_stack_records_paths=[records],
                intent_path=dd / "intent.md", alternatives_path=alternatives,
                dedup_candidates_path=dd / "dedup-candidates.json", structural_records_path=structural,
                allow_standalone=True,
            )

            assert backend.call_count == 1
            if input_kind == "custom-builder":
                assert json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text()) == {"items": []}


# The merge agent returns a schema item list; the host renders the report.
