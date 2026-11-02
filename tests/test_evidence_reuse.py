"""Pure tests for the evidence-reuse decision predicate."""

from __future__ import annotations

from typing import Any

from daydream.deep.evidence_reuse import ReuseTarget, decide_reuse
from tests.harness.execution import test_execution_identity as _identity


def _target(**overrides: Any) -> ReuseTarget:
    fields: dict[str, Any] = {"session_id": "s", "tree_key": "t", "argv": ("uv", "run", "pytest"), "cwd_relative": ".",
        "runner": "uv", "interpreter": None, "config_digest": "d" * 64, "absent_components": (),
        "head_sha": "a" * 40, "branch": "feature", "post_commit_verified": False,
    }
    return ReuseTarget(**{**fields, **overrides})

def test_absent_components_and_missing_evidence_cannot_be_reused() -> None:
    decision = decide_reuse(_identity(config_digest=None, absent_components=("uv.lock",)),
        _target(config_digest=None, absent_components=("uv.lock",)),
    )
    assert (decision.reused, decision.result) == (False, "absent-component")
    assert "uv.lock" in decision.mismatched_components
    assert decide_reuse(None, _target()).result == "no-evidence"
