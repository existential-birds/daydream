"""Exercise model, usage, cost, and phase rows in real deep-run PR comments.

Mock the Claude SDK and GitHub boundary; production phases, trajectory
recording, and comment rendering supply the observable payload.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daydream import pr_review
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockTextBlock,
    MockToolResultBlock,
    MockToolUseBlock,
    MockUserMessage,
    patch_claude_sdk,
)
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import commit as _commit, git as _git, init_repo as _init_repo
from tests.harness.review_result import merge_result
from tests.harness.stub_backend import review_stage_result, review_stage_state

FIXTURE_MODEL_ID = "fixture-model-id"
# SDK fake emits responses and agent-owned review files for downstream phases.

_OUTPUT_PATH_RE = re.compile(r"to ([^\s]+\.(?:md|json))")

def _extract_output_path(prompt: str) -> Path | None:
    """Pull the first ``... to <path>.md|.json`` reference out of a prompt."""
    m = _OUTPUT_PATH_RE.search(prompt)
    if not m:
        return None
    return Path(m.group(1))

_PER_STACK_REPORT = (
    "# Per-stack review\n"
    "\n"
    "## Issues\n"
    "\n"
    "1. [foo.py:1] Use a more descriptive function name\n"
    "   The current name is ambiguous.\n"
)

_MERGED_REPORT = (
    "# Daydream merged review\n"
    "\n"
    "## Issues\n"
    "\n"
    "1. [foo.py:1] Use a more descriptive function name\n"
    "   The current name is ambiguous.\n"
)

# Shared finding; merge adds lens while review and parse use the record shape.
_REVIEW_FINDING = {
    "id": 1, "description": "Use a more descriptive function name", "file": "foo.py", "line": 1, "severity": "medium",
    "confidence": "MEDIUM", "rationale": "The current name is ambiguous.", "evidence": "foo.py:1",
}

class _FakeSDKClient:
    """Per-call canned response + simulated tool-use side effects."""

    def __init__(self, options: Any = None) -> None:
        self.options = options
        self._prompt: str = ""

    async def __aenter__(self) -> "_FakeSDKClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self._prompt = prompt
        self._maybe_write_artifact(prompt)

    @staticmethod
    def _maybe_write_artifact(prompt: str) -> None:
        """Simulate the real agent's Write tool: produce stub report files
        whenever the prompt instructs the agent to write one."""
        out = _extract_output_path(prompt)
        if out is None:
            return
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        name = out.name
        if name == "review-output.md":
            out.write_text(_MERGED_REPORT)
        elif name.startswith("stack-") and name.endswith("-review.md"):
            out.write_text(_PER_STACK_REPORT)

    async def receive_response(self) -> Any:
        for msg in self._build_messages(self._prompt, Path(self.options.cwd)):
            yield msg

    @staticmethod
    def _exploration_messages(structured: dict[str, Any], *, tool_name: str, tool_input: dict[str, Any],) -> list[Any]:
        """Emit paired tool events and SDK model/cost/usage for a forked specialist."""
        tool_id = f"toolu_{tool_name.lower()}_01"
        return [MockAssistantMessage(
                content=[MockToolUseBlock(id=tool_id, name=tool_name, input=tool_input),], model=FIXTURE_MODEL_ID,
            ), MockUserMessage(content=[MockToolResultBlock(tool_use_id=tool_id, content="ok", is_error=False,),],),
            MockAssistantMessage(content=[MockTextBlock(text="exploration complete")], model=FIXTURE_MODEL_ID,),
            MockResultMessage(structured_output=structured, total_cost_usd=0.12,
                usage={"input_tokens": 5000, "output_tokens": 400, "cache_read_input_tokens": 1200,},
            ),
        ]

    @staticmethod
    def _build_messages(prompt: str, cwd: Path) -> list[Any]:
        """Choose canned SDK responses carrying a real model ID and nonzero cost/usage."""
        pl = prompt.lower()

        # Exploration specialists run under maybe_fork; the production-bug path
        # where the 'Exploration' rollup row renders 'unknown' / '$0.00'.
        if "pattern-scanner" in pl:
            return _FakeSDKClient._exploration_messages(
                {"conventions": [], "guidelines": []}, tool_name="Read", tool_input={"file_path": "AGENTS.md"},
            )
        if "dependency-tracer" in pl:
            return _FakeSDKClient._exploration_messages(
                {"affected_files": [], "dependencies": []}, tool_name="Grep", tool_input={"pattern": "import foo"},
            )
        if "test-mapper" in pl:
            return _FakeSDKClient._exploration_messages(
                {"affected_files": []}, tool_name="Read", tool_input={"file_path": "tests/test_foo.py"},
            )

        # phase_alternative_review (ALTERNATIVE_REVIEW_SCHEMA): ResultMessage
        # must carry structured_output of the right shape or it returns [].
        if "architectural alternatives" in pl or ("alternative" in pl and "intent" in pl and "given" in pl):
            return [
                MockAssistantMessage(content=[MockTextBlock(text="evaluating alternatives")], model=FIXTURE_MODEL_ID,),
                MockResultMessage(structured_output={"issues": []}, total_cost_usd=0.10,
                    usage={"input_tokens": 3000, "output_tokens": 250, "cache_read_input_tokens": 1000,},
                ),
            ]

        # phase_understand_intent: free-form text.
        if "understand" in pl and "intent" in pl and review_stage_state(prompt) is None:
            return [MockAssistantMessage(
                    content=[MockTextBlock(text="The PR refactors foo() for clarity.")], model=FIXTURE_MODEL_ID,
                ),
                MockResultMessage(total_cost_usd=0.08,
                    usage={"input_tokens": 2000, "output_tokens": 150, "cache_read_input_tokens": 800,},
                ),
            ]

        # Return one merged item so the host-rendered report and PR comment are nonempty.
        if "cross-stack merge agent" in pl:
            return [MockAssistantMessage(content=[MockTextBlock(text="merging")], model=FIXTURE_MODEL_ID,),
                MockResultMessage(
                    structured_output=merge_result([{**_REVIEW_FINDING, "lens": "per-stack"}]), total_cost_usd=0.20,
                    usage={"input_tokens": 4000, "output_tokens": 600, "cache_read_input_tokens": 1500,},
                ),
            ]

        # Reviewers return records directly: structure empty, language nonempty.
        if "structural reviewer" in pl:
            review_issues: list[Any] = []
        else:
            review_issues = [dict(_REVIEW_FINDING)]
        return [MockAssistantMessage(content=[MockTextBlock(text="ok, wrote the review")], model=FIXTURE_MODEL_ID),
            MockResultMessage(structured_output=review_stage_result(prompt, review_issues), total_cost_usd=0.20,
                usage={"input_tokens": 4000, "output_tokens": 600, "cache_read_input_tokens": 1500,},
            ),
        ]

# Repo + monkeypatch fixtures.

@pytest.fixture
def deep_target_multi(tmp_path: Path) -> Path:
    """Change four Python files to exercise parallel exploration and forked trajectories."""
    repo = tmp_path / "deep_repo_multi"
    _init_repo(repo)
    # Seed five Python files on main so the tree-sitter index can resolve them.
    for name in ("foo.py", "bar.py", "baz.py", "qux.py", "quux.py"):
        (repo / name).write_text(f"def {name[:-3]}():\n    return 1\n")
    _git(repo, "add", ".")
    _commit(repo, "init")
    _git(repo, "checkout", "-b", "feature")
    # Mutate four so count_changed_files >= 4 -> tier="parallel".
    for name in ("foo.py", "bar.py", "baz.py", "qux.py"):
        (repo / name).write_text(f"def {name[:-3]}():\n    return 2\n")
    _git(repo, "add", ".")
    _commit(repo, "tweak four files")
    # Assert diff count up-front so a threshold-breaking refactor trips here.
    diff_out = subprocess.run(  # noqa: S603 - controlled args
        ["git", "diff", "--name-only", "main..HEAD"],  # noqa: S607
        cwd=repo, capture_output=True, check=True, text=True,
    )
    changed = [ln for ln in diff_out.stdout.splitlines() if ln]
    assert len(changed) >= 4, (f"deep_target_multi fixture should produce >=4 changed files; "
        f"got {len(changed)}: {changed!r}"
    )
    return repo

@pytest.fixture
def patch_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch every SDK symbol that ClaudeBackend.execute does isinstance on."""
    patch_claude_sdk(monkeypatch, _FakeSDKClient)

# Fake PR lookup and gh transport capture the production-rendered payload.

@dataclass
class _CapturedPost:
    gh: FakeGh

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [call.payload for call in self.gh.calls("POST", "repos/test-owner/test-repo/pulls/123/reviews")]

@pytest.fixture
def captured_post(monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh) -> _CapturedPost:
    """Wire PR discovery and fake gh so the complete submission runs and we see
    the rendered markdown without ever touching GitHub."""

    captured = _CapturedPost(fake_gh)
    fake_pr = pr_review.PRInfo(
        number=123, head_sha="0" * 40, base_sha="1" * 40, base_ref="main", head_ref="feature", owner="test-owner",
        repo="test-repo", url="https://example/pr/123",
    )

    monkeypatch.setattr("daydream.pr_review.find_open_pr", lambda target_dir, **_kwargs: fake_pr,)

    fake_gh.set_response("POST", "repos/test-owner/test-repo/pulls/123/reviews",
        {"html_url": "https://example/pr/123#review-1"},
    )
    return captured

# Misc: silence Rich UI noise + answer interactive prompts.

# Markdown extraction helpers.

# The test.
