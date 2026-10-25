"""Pi discovery uses admitted diff references even for small changes."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ToolResultEvent, ToolStartEvent
from daydream.deep.detection import StackAssignment
from daydream.hunk_index import write_hunk_index
from daydream.run_context import InteractionPolicy, RunContext
from tests.deep_orchestrator.test_review_capture_and_retry import supporting_contents
from tests.deep_orchestrator.test_review_stage_reservations import NativeStageBackend
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.review_result import review_scopes
from tests.harness.stub_backend import review_stage_state
from tests.test_deep_orchestrator import _sanctioned_inputs


@pytest.mark.parametrize('advance_refs', [False, True])
async def test_small_pi_review_keeps_structural_dispatch(
    tmp_path: Path, make_work: Any, advance_refs: bool,
) -> None:
    """Small-diff navigation and source attribution stay bound to captured revisions.

    Installed-Pi transport tests do not cover this phase's persistent Structure
    catalog or dispatch when both named Git refs move after capture.
    """
    (tmp_path / "app.py").write_text("value = 'DIFF_SENTINEL'\n")
    seed_feature_branch(tmp_path, base={'app.py': 'value = 0\n'}, feature={'app.py': "value = 'DIFF_SENTINEL'\n"})
    diff = tmp_path / "diff.patch"
    diff.write_text("diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
                    "@@ -1 +1 @@\n-value = 0\n+value = 'DIFF_SENTINEL'\n")
    write_hunk_index(tmp_path, diff.read_text())
    intent = tmp_path / "intent.md"
    intent.write_text("Change the value")
    calls: list[str] = []
    base, head = git(tmp_path, "rev-parse", "main"), git(tmp_path, "rev-parse", "HEAD")
    work = make_work(tmp_path, base_sha=base, head_sha=head, head_branch="feature")
    if advance_refs:
        git(tmp_path, "checkout", "main")
        (tmp_path / "base-only.txt").write_text("Advanced base tip\n")
        git(tmp_path, "add", "base-only.txt")
        git(tmp_path, "commit", "-m", "advance base")
        git(tmp_path, "checkout", "feature")
        (tmp_path / "head-only.txt").write_text("Advanced head tip\n")
        git(tmp_path, "add", "head-only.txt")
        git(tmp_path, "commit", "-m", "advance head")

    backend = NativeStageBackend(tmp_path)
    backend.model = "fixture"

    @backend.script('python')
    @backend.script('structure')
    def review(state: dict[str, Any], _output: dict[str, Any]) -> Iterable[AgentEvent]:
        request = backend.calls[-1]
        cwd, prompt = request['cwd'], request['prompt']
        assert str(diff) not in prompt
        assert "review-assignment" in _sanctioned_inputs(prompt)
        assert review_stage_state(prompt) == state
        assert state['analyzed_revision']['head_sha'] == head
        assert state['analyzed_revision']['merge_base_sha'] == base
        if state['scope_id'] == 'python':
            captured_diff = supporting_contents(prompt)['diff']
        else:
            # Only persistent instructions survive native compaction; navigation must survive with them.
            guide_line = next(line for line in request['review_instructions'].splitlines()
                              if line.startswith('Persistent exact access guide (supporting metadata): '))
            catalog = json.loads(guide_line.split(': ', 1)[1])['supporting_catalog']
            assert catalog == state['supporting_catalog']
            assert 'before/after patch context' in catalog['guidance']
            catalog_path = catalog['path']
            catalog_body = Path(catalog_path).read_text()
            yield ToolStartEvent(id='supporting-catalog', name='Read', input={'file_path': catalog_path})
            yield ToolResultEvent(id='supporting-catalog', output=catalog_body, is_error=False)
            parts = json.loads(catalog_body)['parts']
            captured_diff = ''.join(Path(part['path']).read_text() for part in parts)
            assert captured_diff == diff.read_text()
        assert "DIFF_SENTINEL" in captured_diff
        assert "DIFF_SENTINEL" not in prompt
        assert request["read_only"] is True
        assert not request.get("tools_disabled")
        assert "Host review stage:" in prompt
        assert "INVESTIGATION HAS ENDED" not in prompt
        calls.append(prompt)
        yield ToolStartEvent(id="source-app", name="Read", input={"file_path": "app.py"})
        yield ToolResultEvent(id="source-app", output=(cwd / "app.py").read_text(), is_error=False)

    results, failures = await review_scopes(backend, work,
        [StackAssignment("python", ["app.py"]), StackAssignment("structure", ["app.py"])],
        diff_path=diff, diff_text=diff.read_text(), intent_path=intent,
        alternatives_path=tmp_path / "alternatives.json", allow_standalone=True,
        run_context=RunContext(InteractionPolicy(interactive=False)),
    )
    assert failures == {}
    assert set(results) == {"python", "structure"}
    stages = [review_stage_state(prompt) for prompt in calls]
    assert sorted((stage["scope_id"], stage["stage"], stage["assigned_files"])
                  for stage in stages if stage is not None) == [
        ("python", "first_pass", ["app.py"]), ("structure", "integration", ["app.py"])]
    expected_digest = hashlib.sha256(diff.read_bytes()).hexdigest()
    for request in backend.calls:
        binding = request['host_tools']
        assert binding.accepted and binding.native_settled and binding.cleanup_complete
        accepted = binding.accepted_result
        assert isinstance(accepted, dict)
        decision, = accepted['targets']
        assert decision['status'] == 'reviewed'
        reference, = decision['evidence_refs']
        assert reference == {
            'artifact_sha256': expected_digest, 'part_sha256': expected_digest, 'revision': base,
            'path': 'app.py', 'side': 'old', 'source_range': {'start': 1, 'end': 1}, 'fragment': None,
            'basis': 'supplied_diff', 'role': 'assigned',
        }
