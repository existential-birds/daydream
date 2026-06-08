"""Real-path unit test for the public ``TrajectoryRecorder.current_phase()`` accessor.

Asserts the observable outcome: the accessor reflects the active invocation's
phase while a ``rec.invocation(...)`` scope is open, and returns ``None`` outside
any invocation. No private-list (``_active_invocations``) access in the test —
this is the public read-seam the replay harness keys on.
"""

from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder


async def test_current_phase_tracks_active_invocation(tmp_path):
    rec = TrajectoryRecorder(
        path=tmp_path / "t.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="m",
        session_id="00000000-0000-0000-0000-0000000000ff",
    )
    async with rec:
        assert rec.current_phase() is None
        async with rec.invocation(phase=DaydreamPhase.PARSE):
            assert rec.current_phase() is DaydreamPhase.PARSE
        assert rec.current_phase() is None
