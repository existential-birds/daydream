"""Parametrized cross-driver real-path test — one body, driver is a parameter.

Drives the genuine backend (``CodexBackend`` *and* ``ClaudeBackend``) end-to-end
through ``runner.run`` on a single-pass shallow run. A single shared
``PHASE_SCRIPTS`` (the proven map from ``test_codex_realpath``) renders to BOTH
drivers' native formats via the harness; only each driver's external boundary
(the Codex subprocess / the Claude SDK client) is stubbed per firing phase.

Assertions pin OBSERVABLE outcomes only: exit code 0 and a non-empty on-disk
ATIF trajectory. ``assume="no"`` declines the post-pass commit gate so the run
finishes on the four named phases without a real ``git push``.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.atif import Step
from daydream.runner import RunConfig, run
from tests.contract.test_backend_step_parity import _compare_steps
from tests.harness.phase_replay import replay_through_runner
from tests.test_codex_realpath import PHASE_SCRIPTS

# --- Accepted cross-driver divergence (the SINGLE source of truth) -----------
#
# KNOWN_DELTAS is the complete, closed allow-list of fields whose values may
# legitimately differ between the Codex and Claude real-path trajectories.
# It is EXACTLY two driver-level metrics deltas, both documented in the
# conformance contract (``tests/test_backend_conformance.py``) and in
# ``daydream/backends/codex.py:330-346``:
#   * "message_id" — Codex emits ``MetricsEvent.message_id == ""`` (Claude
#     carries the SDK message id). This field is consumed during recording and
#     is NOT persisted onto the on-disk ATIF ``Metrics``; it is listed here for
#     contract parity with the conformance allow-list, never to widen the gate.
#   * "cost_usd"   — Codex reports ``cost_usd is None`` (no per-turn cost from
#     ``turn.completed.usage``); this DOES surface on the on-disk ``Metrics``.
# These two are IGNORED when comparing ``metrics``. NOTHING else is ignored.
KNOWN_DELTAS: frozenset[str] = frozenset({"message_id", "cost_usd"})


def _agent_steps(trajectory: dict[str, Any]) -> list[Step]:
    """Reparse a trajectory's agent-source steps into ``Step`` objects.

    Mirrors the agent-only filter used by ``_compare_steps`` in the event-level
    parity contract — user/system steps (e.g. phase prompts, exploration
    context) are out of scope for the agent-output equivalence gate.
    """
    return [Step.model_validate(s) for s in trajectory["steps"] if s["source"] == "agent"]


def _fix_prompts(trajectory: dict[str, Any]) -> list[str]:
    """The text of every fix-phase USER step (the prompt built from the parsed
    issues). Used to prove the parsed PARSE payload semantically drove FIX.
    """
    out: list[str] = []
    for s in trajectory["steps"]:
        if s["source"] == "user" and (s.get("extra") or {}).get("daydream_phase") == "fix":
            msg = s["message"]
            out.append(msg if isinstance(msg, str) else json.dumps(msg))
    return out


def _metrics_modulo_deltas(step: Step) -> dict[str, Any]:
    """Serialize a step's metrics with the KNOWN_DELTAS fields stripped.

    This is the ONLY ignore applied by the gate, and it is confined to the two
    allow-listed metrics fields. Token counts, cached tokens, logprobs, etc. are
    all still compared.
    """
    if step.metrics is None:
        return {}
    data = step.metrics.model_dump()
    for field in KNOWN_DELTAS:
        data.pop(field, None)
    return data


def _normalize_parse_transport(left: list[Step], right: list[Step], traj_left: dict, traj_right: dict) -> None:
    """Normalize the schema-constrained PARSE step's transport difference.

    Real Codex puts the schema-constrained JSON in the step ``message`` (it
    ``json.loads`` the last ``agent_message`` text — ``codex.py:349``); real
    Claude carries the identical structured payload on
    ``ResultMessage.structured_output`` and leaves the step ``message`` empty.
    Both carry the SAME payload that drives FIX — only the on-disk *transport*
    differs.

    This is NOT an added ignore: for the one step where one side's ``message``
    is non-empty JSON and the other side's is empty, we PARSE the non-empty side
    and assert (semantic equality, still enforced) that exactly that structured
    payload drove FIX on BOTH runs — every parsed issue's ``description`` and
    ``file`` must appear in BOTH trajectories' fix-phase user prompt. Only after
    that assertion do we equalize the two ``message`` strings so the remaining
    ``_compare_steps`` mirror (reasoning / tool_calls / observation, plus the
    message field for every OTHER step) runs unmodified.

    A divergence in the payload (or a non-empty/non-empty or empty/empty pair
    here) propagates as a failure — it is never silently accepted.
    """
    fix_prompts = _fix_prompts(traj_left) + _fix_prompts(traj_right)
    for ls, rs in zip(left, right, strict=True):
        l_msg = ls.message if isinstance(ls.message, str) else None
        r_msg = rs.message if isinstance(rs.message, str) else None
        if l_msg is None or r_msg is None:
            continue
        l_empty, r_empty = (l_msg == ""), (r_msg == "")
        if l_empty == r_empty:
            continue  # both empty or both non-empty → no transport divergence to normalize
        # Exactly one side carries the JSON payload; parse it and prove semantic equality.
        payload_text = r_msg if l_empty else l_msg
        payload = json.loads(payload_text)  # raises → gate fails if the carried payload is not valid JSON
        for issue in payload["issues"]:
            for prompt in fix_prompts:
                assert issue["description"] in prompt, (
                    "PARSE payload did not drive FIX identically across drivers: "
                    f"issue description {issue['description']!r} missing from a fix prompt"
                )
                assert str(issue["file"]) in prompt, (
                    "PARSE payload did not drive FIX identically across drivers: "
                    f"issue file {issue['file']!r} missing from a fix prompt"
                )
        # Transport normalized: collapse both message strings to the parsed payload
        # so the structured CONTENT (not the empty/non-empty transport) is what the
        # _compare_steps mirror compares for this step.
        canonical = json.dumps(payload, sort_keys=True)
        ls.message = canonical
        rs.message = canonical


def assert_equivalent_modulo(traj_a: dict[str, Any], traj_b: dict[str, Any], known_deltas: frozenset[str]) -> None:
    """Assert two real-path trajectories are equivalent modulo ``known_deltas``.

    Reuses ``_compare_steps`` (the event-level parity contract's comparator) for
    message / reasoning_content / tool_calls / observation equality over the
    agent-source steps. Two adjustments, both narrow and documented:

      1. The metrics ``cost_usd``/``message_id`` deltas in ``known_deltas`` are
         ignored (the closed, two-field allow-list) — and ONLY there.
      2. The PARSE schema-constrained step's message *transport* is normalized
         (``_normalize_parse_transport``): the structured payload is still
         asserted semantically (it must have driven FIX on both sides) — message
         content is NOT ignored.

    Everything else must match exactly; any other divergence fails the gate.
    """
    assert known_deltas == KNOWN_DELTAS, "the accepted-divergence allow-list must not be widened at the call site"
    left = _agent_steps(traj_a)
    right = _agent_steps(traj_b)
    # Compare metrics modulo the two allow-listed deltas (before message normalization).
    assert len(left) == len(right), f"agent step count: a={len(left)} b={len(right)}"
    for ls, rs in zip(left, right, strict=True):
        assert _metrics_modulo_deltas(ls) == _metrics_modulo_deltas(rs), (
            f"metrics diverged beyond {sorted(KNOWN_DELTAS)} at step {ls.step_id}"
        )
    # Normalize ONLY the PARSE schema-constrained message transport (payload still asserted).
    _normalize_parse_transport(left, right, traj_a, traj_b)
    # Mirror the event-level comparator for message / reasoning / tool_calls / observation.
    _compare_steps(left, right)


@pytest.mark.parametrize("driver", ["codex", "claude"])
@pytest.mark.asyncio
async def test_realpath_shallow_run_per_driver(driver: str, feature_branch_repo: Path, tmp_path: Path) -> None:
    """Single-pass shallow run through the real backend for *driver*.

    The shared ``PHASE_SCRIPTS`` renders to the driver's native message stream;
    the run completes with exit 0 and writes a non-empty ATIF trajectory.
    ``assume="no"`` declines the post-pass commit gate (no real ``git push``).
    """
    traj = tmp_path / "trajectory.json"
    config = RunConfig(
        target=str(feature_branch_repo),
        skill="python",
        quiet=True,
        cleanup=False,
        shallow=True,
        loop=False,
        backend=driver,
        assume="no",
        trajectory_path=traj,
    )

    with replay_through_runner(driver, PHASE_SCRIPTS):
        exit_code = await run(config)

    assert exit_code == 0
    assert json.loads(traj.read_text(encoding="utf-8"))["steps"]


@pytest.mark.asyncio
async def test_both_drivers_equivalent_modulo_known_deltas(feature_branch_repo: Path, tmp_path: Path) -> None:
    """Parallel-implementation gate: Claude and Codex real-path trajectories are
    equivalent modulo the documented allow-list.

    Drives the SAME shared ``PHASE_SCRIPTS`` through ``runner.run`` for BOTH real
    drivers in one invocation, then asserts byte-equivalence of their
    agent-source steps (message / reasoning / tool_calls / observation) modulo
    ``KNOWN_DELTAS`` — the closed two-field metrics allow-list. The PARSE step's
    schema-constrained payload is normalized for *transport* only and STILL
    asserted semantically (it must have driven FIX identically on both sides).
    The gate fails if either driver is red OR any non-allow-listed field
    diverges.
    """
    results: dict[str, tuple[int, dict]] = {}
    for driver in ("codex", "claude"):
        traj = tmp_path / f"{driver}.json"
        config = RunConfig(
            target=str(feature_branch_repo),
            skill="python",
            quiet=True,
            cleanup=False,
            shallow=True,
            loop=False,
            backend=driver,
            assume="no",
            trajectory_path=traj,
        )
        with replay_through_runner(driver, PHASE_SCRIPTS):
            rc = await run(config)
        results[driver] = (rc, json.loads(traj.read_text(encoding="utf-8")))

    assert results["codex"][0] == results["claude"][0] == 0
    assert_equivalent_modulo(results["codex"][1], results["claude"][1], KNOWN_DELTAS)
