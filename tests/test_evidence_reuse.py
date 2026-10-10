"""Pure tests for the evidence-reuse decision predicate."""

from __future__ import annotations

from typing import Any

import pytest

from daydream.deep.evidence_reuse import EVIDENCE_REUSE_FORMAT, ReuseTarget, decide_reuse
from tests.harness.execution import test_execution_identity as _identity


def _target(**overrides: Any) -> ReuseTarget:
    fields: dict[str, Any] = {"session_id": "s", "tree_key": "t", "argv": ("uv", "run", "pytest"), "cwd_relative": ".",
        "runner": "uv", "interpreter": None, "config_digest": "d" * 64, "absent_components": (),
        "head_sha": "a" * 40, "branch": "feature", "post_commit_verified": False,
    }
    return ReuseTarget(**{**fields, **overrides})

@pytest.mark.parametrize(("component", "override"),
    [("argv", {"argv": ("uv", "run", "pytest", "-k", "one")}), ("cwd_relative", {"cwd_relative": "services/api"}),
        ("runner", {"runner": "poetry"}), ("config_digest", {"config_digest": "e" * 64}),
        ("tree_key", {"tree_key": "other"}),
    ],
)
def test_any_component_mismatch_requires_real_validation(component: str, override: dict[str, Any]) -> None:
    decision = decide_reuse(_identity(), _target(**override))
    assert decision.reused is False
    assert component in decision.mismatched_components

def test_a_bare_tree_key_match_across_a_commit_is_not_enough() -> None:
    """MH13: the tree key is content-only, so a commit moves HEAD without moving it."""
    decision = decide_reuse(_identity(), _target(head_sha="b" * 40, post_commit_verified=False))
    assert decision.reused is False
    assert "head_sha" in decision.mismatched_components

def test_an_absent_component_is_a_named_miss() -> None:
    decision = decide_reuse(_identity(config_digest=None, absent_components=("uv.lock",)),
        _target(config_digest=None, absent_components=("uv.lock",)),
    )
    assert (decision.reused, decision.result) == (False, "absent-component")
    assert "uv.lock" in decision.mismatched_components

@pytest.mark.parametrize("outcome", ["failed", "timed-out", "truncated"])
def test_incomplete_or_red_evidence_requires_real_validation(outcome: str) -> None:
    assert decide_reuse(_identity(outcome=outcome), _target()).reused is False

def test_missing_evidence_requires_real_validation() -> None:
    assert decide_reuse(None, _target()).result == "no-evidence"

def test_the_decision_is_deterministic_and_io_free() -> None:
    assert decide_reuse(_identity(), _target()) == decide_reuse(_identity(), _target())
    assert EVIDENCE_REUSE_FORMAT == 1
