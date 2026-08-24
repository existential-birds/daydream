"""Drift guard: docs/extensions.md must name every registered extension surface."""

from pathlib import Path

import daydream.extensions as extension_api
from daydream.extensions import EXTENSION_API_VERSION, Registry
from daydream.extensions.builtins import register_builtins

CONTRACT_DOC = Path(__file__).resolve().parents[1] / "docs" / "extensions.md"


def test_contract_doc_names_every_registered_surface() -> None:
    doc = CONTRACT_DOC.read_text()
    reg = Registry()
    register_builtins(reg)
    assert f"DAYDREAM_EXT_API = {EXTENSION_API_VERSION}" in doc
    for fragment in (
        "register_tool_supervisor",
        "ToolDecision",
        "items_file",
        "read",
        "rewrite",
        "raise the floor",
        "PlanWriterResult",
        "Sequence[str]",
        "AUTHOR_SCHEMA_INVALID",
        "host-owned",
        "intent_authoritative",
    ):
        assert fragment in doc, f"contract detail {fragment!r} undocumented"
    for symbol in extension_api.__all__:
        assert symbol in doc, f"public symbol {symbol!r} undocumented"
    for flow in ("deep", "improve"):
        assert flow in doc, f"flow {flow!r} undocumented"
        for entry in reg.flow(flow):
            for name in [entry] if isinstance(entry, str) else entry.steps:
                assert name in doc, f"flow step {name!r} undocumented"
    for name in reg.prompt_names():
        assert name in doc, f"prompt {name!r} undocumented"


def test_contract_doc_names_renderer_surface() -> None:
    doc = CONTRACT_DOC.read_text()
    reg = Registry()
    register_builtins(reg)
    assert "override_renderer" in doc
    for name in reg.renderer_names():  # "finding", "summary"
        assert name in doc, f"renderer slot {name!r} undocumented"
    for fragment in ("CommentFinding", "SummaryContext", "host-owned", "falls back"):
        assert fragment in doc
