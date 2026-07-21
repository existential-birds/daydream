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
        "DAYDREAM_EXT_API = 2",
        "raise the floor",
        "PlanWriterResult",
        "Sequence[str]",
        "LEGACY_MARKDOWN_OUTPUT",
        "host-owned",
    ):
        assert fragment in doc, f"contract detail {fragment!r} undocumented"
    for symbol in extension_api.__all__:
        assert symbol in doc, f"public symbol {symbol!r} undocumented"
    for flow in ("deep", "shallow", "review", "pr-feedback"):
        assert flow in doc, f"flow {flow!r} undocumented"
        for entry in reg.flow(flow):
            for name in [entry] if isinstance(entry, str) else entry.steps:
                assert name in doc, f"flow step {name!r} undocumented"
    for slot in reg.skill_slots():
        assert slot in doc, f"skill slot {slot!r} undocumented"
    for name in reg.prompt_names():
        assert name in doc, f"prompt {name!r} undocumented"
