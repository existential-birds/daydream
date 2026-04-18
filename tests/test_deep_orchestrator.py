"""Deep-mode orchestrator integration tests (xfail until plan 05-09).

Covers D-07..D-10, D-17, D-19..D-22, D-24..D-26, D-28, D-30, D-31,
D-34, D-35, D-44. Test bodies are intentionally stubbed with
``raise NotImplementedError`` so the xfail(strict=True) contract holds
until plan 05-09 fills them in.
"""

from pathlib import Path

import pytest


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_pipeline_order(multi_stack_target: Path) -> None:
    """D-07: Stage order = TTT -> per-stack -> parse -> merge."""
    # Fill in plan 05-09 when orchestrator exists.
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_fresh_context_per_stage(multi_stack_target: Path) -> None:
    """D-08: Each stage = distinct Backend.execute call."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_artifacts_on_disk(multi_stack_target: Path) -> None:
    """D-09: intent.md, alternatives.json, stack-*-review.md, stack-*-records.json written under .daydream/deep/."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_per_stack_context_isolation(multi_stack_target: Path) -> None:
    """D-10: per-stack agents never see each other's outputs."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_parallel_fan_out(multi_stack_target: Path) -> None:
    """D-17: per-stack fan-out uses anyio task group + CapacityLimiter."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_per_stack_prompt_context(multi_stack_target: Path) -> None:
    """D-19: per-stack prompt references intent + alternatives paths."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_doc_review_notice(multi_stack_target: Path) -> None:
    """D-20: .md routed to generic fallback surfaces doc-review notice."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_pre_merge_parse_per_stack(multi_stack_target: Path) -> None:
    """D-21, D-22: phase_parse_feedback invoked per per-stack output, merger consumes records."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_merged_report_path(multi_stack_target: Path) -> None:
    """D-24: final report at REVIEW_OUTPUT_FILE path."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_report_format_flat_numbered(multi_stack_target: Path) -> None:
    """D-25: flat globally-numbered ## Issues + ## Cross-Stack Issues subsection continuing numbering."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_cross_stack_prefix(multi_stack_target: Path) -> None:
    """D-26: every cross-stack title starts with [cross-stack]."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_fix_gate_prompt(multi_stack_target: Path) -> None:
    """D-28: Y/n prompt after merge."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_preflight_notice(multi_stack_target: Path) -> None:
    """D-30: pre-flight notice lists stages, stacks, skill per stack, total agent count."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_codex_cost_caveat(multi_stack_target: Path) -> None:
    """D-31: Codex backend pre-flight mentions cost_usd=None."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_resume_per_stack_reruns_all(multi_stack_target: Path) -> None:
    """D-34: --start-at per-stack re-runs ALL per-stack reviews."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_resume_overwrites(multi_stack_target: Path) -> None:
    """D-35: resume overwrites stage artifacts."""
    raise NotImplementedError


@pytest.mark.xfail(reason="Wave 5 plan 05-09 not yet implemented", strict=True)
def test_stage_ui_surfacing(multi_stack_target: Path) -> None:
    """D-44: UI prints [stage N/5: ...] at each stage boundary."""
    raise NotImplementedError
