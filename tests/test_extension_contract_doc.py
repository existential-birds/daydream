"""Drift guard: docs/extensions.md must name every registered extension surface."""

import re
from pathlib import Path

import daydream.extensions as extension_api
from daydream.extensions import EXTENSION_API_VERSION, Registry
from daydream.extensions.builtins import register_builtins
from daydream.prompt_budget import (
    SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES,
    SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES,
    SANCTIONED_EXACT_INPUT_MAX_FILES,
    SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES,
)

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
    for flow in ("deep", "improve", "diagram"):
        assert flow in doc, f"flow {flow!r} undocumented"
        for entry in reg.flow(flow):
            for name in [entry] if isinstance(entry, str) else entry.steps:
                assert name in doc, f"flow step {name!r} undocumented"
    for name in reg.prompt_names():
        assert name in doc, f"prompt {name!r} undocumented"


def test_deep_flow_table_matches_registered_step_keys() -> None:
    """The documented registered key is each step's real phase key."""
    doc = CONTRACT_DOC.read_text()
    table = re.search(
        r"#### `deep`.*?(\| # \| Step \| Registered step key \|.*?)(?:\n\n)",
        doc,
        flags=re.DOTALL,
    )
    assert table is not None
    documented = re.findall(
        r"\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|",
        table.group(1),
    )

    reg = Registry()
    register_builtins(reg)
    expected: list[tuple[str, str]] = []
    for entry in reg.flow("deep"):
        names = [entry] if isinstance(entry, str) else entry.steps
        expected.extend((name, reg.phase(name).phase_key) for name in names)

    assert documented == expected


def test_all_flow_tables_and_fix_verify_keys_are_documented_precisely() -> None:
    doc = CONTRACT_DOC.read_text()
    assert doc.count("| # | Step | Registered step key |") == 3
    assert "| # | Step | Config key |" not in doc
    for pattern in (
        r"registered step\s+key `fix-verify`",
        r"`\[tool\.daydream\.phases\.verify\]`",
        r"separately\s+registered `fix-verify` prompt",
    ):
        assert re.search(pattern, doc)


def test_contract_doc_names_renderer_surface() -> None:
    doc = CONTRACT_DOC.read_text()
    reg = Registry()
    register_builtins(reg)
    assert "override_renderer" in doc
    for name in reg.renderer_names():  # "finding", "summary"
        assert name in doc, f"renderer slot {name!r} undocumented"
    for fragment in ("CommentFinding", "SummaryContext", "host-owned", "falls back"):
        assert fragment in doc


def test_contract_doc_names_artifact_lifecycle_contract() -> None:
    """The working-artifact-paths section matches the shipped P10 surface."""
    doc = CONTRACT_DOC.read_text()
    working = doc.index("### Working artifact paths")
    stable = doc.index("### Stable `ctx.data` keys")
    assert working < stable, "artifact section must precede the stable ctx.data keys"
    section = doc[working:stable]

    for fragment in (
        # Real helper/session names, not guessed APIs.
        "`ctx.artifacts`", "artifact_dir_for", "review_output_path_for",
        "prepare_sanctioned_inputs", "FlowStep", "run_agent",
        # Active-session lifetime: bound to the run, frozen at finalization.
        "active artifact session", "freezes at finalization", "after the session ends",
        # Sanctioned-input contract: per-attempt validation, fail closed.
        "remain unchanged before each dispatch attempt", "fail before backend entry",
        "not an OS sandbox",
        # Transport split, then the rendered budgets.
        "Strict Claude audit roots, read-only Codex clones, and sandboxed Osprey",
        "12,288", "512 files", "1 MiB", "4 MiB",
    ):
        assert fragment in section, f"artifact contract detail {fragment!r} undocumented"

    # The rendered budgets stay in sync with the shipped constants.
    assert SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES == 12_288
    assert SANCTIONED_EXACT_INPUT_MAX_FILES == 512
    assert SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES == 1_048_576
    assert SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES == 4_194_304
