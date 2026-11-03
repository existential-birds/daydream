"""Real deep-flow tests binding finalized host-test evidence to the commit gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.phases import TestAttemptEvidence, phase_commit_push
from daydream.test_execution import TestExecutionIdentity
from tests.harness.backend import ScriptedBackend
from tests.test_deep_orchestrator import MakeConfig


@pytest.mark.asyncio
async def test_commit_gate_refuses_half_formed_or_mismatched_evidence_offers(
    tmp_path: Path, make_work: Any, make_config: MakeConfig,
) -> None:
    """One half of the evidence/tree pair is a caller bug; a disagreeing retained key names no valid offer."""
    work = make_work(tmp_path / "offer")
    config = make_config(work.repo)
    evidence = TestAttemptEvidence(
        session_id="s", kind="host", command=("true",), passed=True, input_tree_key="t", output_tree_key="t",
    )
    with pytest.raises(ValueError, match="together"):
        await phase_commit_push(ScriptedBackend(), work, config=config, evidence=evidence)
    with pytest.raises(ValueError, match="together"):
        await phase_commit_push(ScriptedBackend(), work, config=config, retained_tree_key="t")
    identity = TestExecutionIdentity(session_id="s", argv=("true",), cwd_relative=".", runner=None, interpreter=None,
        config_digest=None, absent_components=(), input_tree_key="t", output_tree_key="t",
        head_sha="a" * 40, branch="feature", kind="host", outcome="passed",
    )
    identified = TestAttemptEvidence(session_id="s", kind="host", command=("true",), passed=True,
        input_tree_key="t", output_tree_key="t", identity=identity,
    )
    with pytest.raises(ValueError, match="does not match"):
        await phase_commit_push(ScriptedBackend(), work, config=config, evidence=identified, retained_tree_key="other")
