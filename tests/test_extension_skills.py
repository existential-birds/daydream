"""Tests for registry-routed stack skill resolution and fork stack rules."""

from daydream.deep.detection import detect_stacks
from daydream.extensions import build_registry
from tests.conftest import ExtDir


def test_fork_stack_rule_and_remap_reach_detection(ext_dir: ExtDir) -> None:
    ext_dir.write_module(
        "from daydream.extensions import StackRule\n"
        "DAYDREAM_EXT_API = 1\n"
        "def register(r):\n"
        "    r.add_stack(StackRule('proto', ('*.proto',), 'ro-proto:review-proto'))\n"
        "    r.override_skill('stack:python', 'ro-python:review-python')\n"
    )
    stacks = detect_stacks(
        ["api/v1.proto", "svc/app.py"],
        skill_availability={"python"},
        registry=build_registry(),
    )
    by_name = {s.stack_name: s.skill_invocation for s in stacks}
    assert by_name["proto"] == "ro-proto:review-proto"
    assert by_name["python"] == "ro-python:review-python"
