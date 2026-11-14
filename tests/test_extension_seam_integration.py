"""Run fork-mutated flows through runner.run in real temporary repositories.

Only the backend seam is mocked; temporary extension packages modify registered
steps. Assert delivered prompts and exit codes."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream import runner
from daydream.backends import ResultEvent, TextEvent
from daydream.config_file import load_file_config
from daydream.dataset import LocalRecordStore
from daydream.review_result import FinalizedReview
from daydream.run_config import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.dataset import read_records
from tests.harness.stub_backend import StubBackend

KEEP_ME = "KEEP_ME"
DROP_ME = "DROP_ME"

MakeConfig = Callable[..., RunConfig]
InstallBackend = Callable[[object], object]

_EMPTY_TURN = (TextEvent(text=""), ResultEvent(structured_output=None, continuation=None))


async def test_extension_step_rewrites_items_before_the_review_is_sealed(
    ext_dir: ExtDir, multi_stack_target: Path, install_backend: InstallBackend, make_config: MakeConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A step inserted into the deep flow rewrites the canonical items every later consumer reads.

    The reviewer's own sealed record keeps both findings; only the deep flow plans the review, so a
    separately named flow cannot borrow its review phase.
    """
    ext_dir.write_module(
        "import json\n"
        "from daydream.extensions import FlowStep\n"
        "async def _rewrite(ctx):\n"
        "    path = ctx.data['items_file']\n"
        "    items = json.loads(path.read_text())['items']\n"
        f"    kept = [dict(item, description='{KEEP_ME}') for item in items if item['description'] != '{DROP_ME}']\n"
        "    path.write_text(json.dumps({'items': kept}))\n"
        "def register(r):\n"
        "    r.register_phase(FlowStep(name='rewrite', run=_rewrite))\n"
        "    flow = r.flow('deep')\n"
        "    flow.insert(flow.index('load-items') + 1, 'rewrite')\n"
        "    r.set_flow('deep', flow)\n"
        "    r.set_flow('borrowed', ['review', 'load-items'])\n"
    )
    backend = StubBackend(multi_stack_target)
    backend.review_claims = [{"title": "Greeting drift", "file": "api.py", "line": 1},
                             {"title": DROP_ME, "file": "App.tsx", "line": 1}]
    install_backend(backend)
    delivered: list[FinalizedReview] = []

    store = LocalRecordStore(multi_stack_target.parent / "records")
    assert await runner.run(make_config(multi_stack_target, dataset_capture=True, dataset_store_path=store.root),
                            review_result_handler=delivered.append) == 0

    (review,) = delivered
    sealed = json.loads((multi_stack_target / ".daydream/deep/committed-review.json").read_text())
    assert [finding["title"] for finding in sealed["findings"]] == ["Greeting drift", DROP_ME]
    assert review.terminal_result == sealed["terminal_result"] and review.terminal_result["projection_valid"]
    # Identity survives the rewrite; the embedded callback and the captured record agree on what was kept.
    assert [(item["description"], item["item_uid"]) for item in review.items] == [
        (KEEP_ME, sealed["findings"][0]["item_uid"])]
    (captured,) = read_records(store).runs
    assert [(item["description"], item["item_uid"]) for item in captured["findings"]["value"]["items"]] == [
        (KEEP_ME, sealed["findings"][0]["item_uid"])]

    calls = len(backend.calls)
    assert await runner.run(make_config(multi_stack_target, flow_name="borrowed")) == 1
    assert "plans the review" in " ".join(capsys.readouterr().out.replace("║", " ").split())
    assert len(backend.calls) == calls


async def test_bound_custom_flow_rejects_public_artifact_before_backend_entry(
    ext_dir: ExtDir, multi_stack_target: Path, install_backend: InstallBackend, make_config: MakeConfig,
) -> None:
    ext_dir.write_module(
        "from daydream.extensions import FlowStep\n"
        "async def _leak(ctx):\n"
        "    from daydream.agent import run_agent\n"
        "    from daydream.trajectory import DaydreamPhase\n"
        "    public = ctx.work.repo / '.daydream'\n"
        "    public.mkdir()\n"
        "    (public / 'leak.txt').write_text('generated')\n"
        "    await run_agent(ctx.backend_for('review'), ctx.work.repo, 'SECOND-CALL',\n"
        "                    phase=DaydreamPhase.REVIEW)\n"
        "def register(r):\n"
        "    r.register_phase(FlowStep(name='leak', run=_leak))\n"
        "    r.set_flow('leak', ['leak'])\n"
    )
    backend = ScriptedBackend(events=_EMPTY_TURN)
    install_backend(backend)

    rc = await runner.run(make_config(multi_stack_target, flow_name="leak"))

    assert rc == 1
    assert backend.call_count == 0


async def test_builtin_and_fork_tool_supervisor_conflict_fails_loud(
    ext_dir: ExtDir, multi_stack_target: Path, make_config: MakeConfig, capsys: pytest.CaptureFixture[str],
) -> None:
    """Config-enabled built-in and fork supervisors cannot silently compose."""

    ext_dir.write_module(
        "from daydream.extensions import ToolDecision\n"
        "def _fork_supervisor(name, tool_input, *, phase):\n"
        "    return ToolDecision(veto=False)\n"
        "def register(r):\n"
        "    r.register_tool_supervisor(_fork_supervisor)\n"
    )
    (multi_stack_target / ".daydream.toml").write_text('tool_supervisor = "rules"\n')

    rc = await runner.run(make_config(multi_stack_target, file_config=load_file_config(multi_stack_target)))

    assert rc == 1
    output = capsys.readouterr().out.lower()
    assert "tool supervisor conflict" in output
    assert "config-enabled built-in" in output
    assert "extension-registered" in output

async def test_pr_feedback_not_selectable_via_flow(
    multi_stack_target: Path, install_backend: InstallBackend, make_config: MakeConfig,
) -> None:
    """--flow pr-feedback errors (needs PR number + bot)."""
    install_backend(ScriptedBackend(events=_EMPTY_TURN, model="mock-model"))

    rc = await runner.run(make_config(multi_stack_target, flow_name="pr-feedback"))
    assert rc == 1
