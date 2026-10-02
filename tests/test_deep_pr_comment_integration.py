"""Exercise model, usage, cost, and phase rows in real deep-run PR comments.

Mock the Claude SDK and GitHub boundary; production phases, trajectory
recording, and comment rendering supply the observable payload.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daydream import pr_review
from daydream.exploration import ExplorationContext
from daydream.run_config import RunConfig
from daydream.runner import run
from daydream.trajectory import TrajectoryDocumentSnapshot, get_current_recorder
from tests.conftest import silence_module_console
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

FIXTURE_MODEL_ID = "fixture-model-id"
_PARTIAL_MODEL = "partial-only-model-must-not-be-posted"

def _write_live_sibling_canary(*, malformed: bool) -> Path | None:
    """Exercise the real session sink from the fake external backend boundary."""

    recorder = get_current_recorder()
    assert recorder is not None
    while recorder.parent is not None:
        recorder = recorder.parent
    if not recorder.steps:
        return None
    assert recorder.artifact_run_dir is not None
    assert recorder.document_writer is not None
    trajectory_id = f"{recorder.session_id}-live-post-canary"
    path = recorder.artifact_run_dir / "trajectories" / "live-post-canary.json"
    trajectory = recorder.build_trajectory().to_json_dict()
    trajectory["trajectory_id"] = trajectory_id
    trajectory["agent"]["model_name"] = _PARTIAL_MODEL
    json_bytes = (b"private-prompt-canary: malformed trajectory"
        if malformed
        else json.dumps(trajectory).encode("utf-8")
    )
    recorder.document_writer(
        TrajectoryDocumentSnapshot(trajectory_id, path, json_bytes), "complete" if malformed else "partial",
    )
    return path

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
        for msg in self._build_messages(self._prompt):
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
    def _build_messages(prompt: str) -> list[Any]:
        """Choose canned SDK responses carrying a real model ID and nonzero cost/usage."""
        pl = prompt.lower()

        # Exploration specialists run under maybe_fork; the production-bug path
        # where the 'Exploration' rollup row renders 'unknown' / '$0.00'.
        if "pattern-scanner" in pl:
            return _FakeSDKClient._exploration_messages(
                {"conventions": [], "guidelines": []}, tool_name="Read", tool_input={"file_path": "CLAUDE.md"},
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
        if "understand" in pl and "intent" in pl:
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
        return [MockAssistantMessage(content=[MockTextBlock(text="ok, wrote the review")], model=FIXTURE_MODEL_ID,),
            MockResultMessage(structured_output={"issues": review_issues}, total_cost_usd=0.20,
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

def _silence_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ("daydream.deep.orchestrator", "daydream.phases", "daydream.runner", "daydream.pr_review",):
        silence_module_console(monkeypatch, module)

def _answer_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept intent and PR posting through the shared gateway; decline fixes."""
    monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)
    def answer(_console: Any, message: str, default: str = "") -> str:
        return "n" if "apply fix" in message.lower() else "y"
    monkeypatch.setattr("daydream.run_context._prompt_user", answer)

# Markdown extraction helpers.

def _line_starting(markdown: str, prefix: str) -> str:
    for line in markdown.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(
        f"No line starting with {prefix!r} in markdown:\n{markdown}"
    )

def _phase_rows(markdown: str) -> list[str]:
    rows: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("|") and not line.startswith("| Phase") and not line.startswith("|---"):
            rows.append(line)
    return rows

def _row_cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip("|").split("|")]

# The test.

async def test_deep_run_produces_pr_comment_with_real_model_and_metrics(
    deep_target: Path, patch_sdk: None, captured_post: _CapturedPost, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render a deep-run PR comment with real model and nonzero usage/cost.

    Only SDK messages/client, PR lookup, and gh transport are mocked; the
    real API adapter handles its request file and the full pipeline records
    and renders the results.
    """
    _silence_ui(monkeypatch)
    _answer_prompts(monkeypatch)

    config = RunConfig(target=str(deep_target), cleanup=False, archive=False,)
    # Single-file diff -> select_tier "skip": the orchestrator's natural
    # pre-scan path runs unpopulated. (ExplorationContext kept for the linter.)
    _ = ExplorationContext
    exit_code = await run(config)

    assert exit_code == 0, f"run() returned {exit_code}"
    assert captured_post.payloads, ("build_payload was never called: deep flow did not reach _post")
    payload = captured_post.payloads[-1]
    body = payload["body"]
    assert isinstance(body, str)

    # M3/M4: deep --comment path names the fixture PR's head SHA.
    assert ("- **Reviewed commit:** [`0000000`]"
        "(https://github.com/test-owner/test-repo/commit/" + "0" * 40 + ")"
    ) in body, f"reviewed-commit line missing from payload body.\n\nfull body:\n{body}"

    # --- Mode line: removed everywhere ----------------------------------
    assert "**Mode:**" not in body, (
        f"BUG: Mode line should be gone from PR-comment body.\n\nfull body:\n{body}"
    )

    # --- Model line: real SDK id, not 'unknown' / backend alias ---------
    model_line = _line_starting(body, "- **Model:**")
    assert FIXTURE_MODEL_ID in model_line, (f"BUG: rollup Model line is missing the real SDK model id "
        f"({FIXTURE_MODEL_ID!r}).\n  got: {model_line!r}\n\n"
        f"full body:\n{body}"
    )
    assert "unknown" not in model_line, (f"BUG: rollup Model line still says 'unknown' (no model id "
        f"propagated from SDK).\n  got: {model_line!r}"
    )

    # --- Cost line: non-zero ---------------------------------------------
    cost_line = _line_starting(body, "- **Cost:**")
    assert "$0.00" not in cost_line, (f"BUG: rollup Cost line shows $0.00 — per-step cost never landed "
        f"on Step.metrics.\n  got: {cost_line!r}"
    )

    # --- Tokens line: non-zero in / out ---------------------------------
    tokens_line = _line_starting(body, "- **Tokens:**")
    assert not re.search(r"(?<!\d)0 in\b", tokens_line), (
        f"BUG: rollup Tokens line shows '0 in'.\n  got: {tokens_line!r}"
    )
    assert not re.search(r"(?<!\d)0 out\b", tokens_line), (
        f"BUG: rollup Tokens line shows '0 out'.\n  got: {tokens_line!r}"
    )

    # --- Per-phase breakdown: at least 2 rows, each with real model + cost
    rows = _phase_rows(body)
    assert len(rows) >= 2, (
        f"expected >= 2 per-phase rows, got {len(rows)}.\n"
        f"  rows: {rows}\n\nfull body:\n{body}"
    )
    for row in rows:
        cells = _row_cells(row)
        # Layout: | Phase | Model | Tools | Input (cached) | Output | Cost |
        assert len(cells) >= 6, f"unexpected row layout: {row!r}"
        phase_name, model_cell, _tools, input_cell, _out, cost_cell = cells[:6]
        assert input_cell != "0", (f"BUG: row {phase_name!r} has Input='0' "
            f"(per-step token metrics never propagated).\n  row: {row!r}"
        )
        assert model_cell != "unknown", (f"BUG: row {phase_name!r} has Model='unknown' "
            f"(SDK model id never propagated to the per-phase rollup).\n"
            f"  row: {row!r}"
        )
        assert FIXTURE_MODEL_ID in model_cell, (f"BUG: row {phase_name!r} Model cell missing real SDK id "
            f"{FIXTURE_MODEL_ID!r}.\n  row: {row!r}"
        )
        assert cost_cell != "$0.00", (f"BUG: row {phase_name!r} has Cost=$0.00 "
            f"(per-step cost never landed).\n  row: {row!r}"
        )

# Exploration-row reproduction test: forces the parallel tier (which the
# single-file fixture skips) so the broken Exploration rollup row is exercised.

async def test_deep_run_exploration_row_has_real_model_and_metrics(
    deep_target_multi: Path, patch_sdk: None, captured_post: _CapturedPost, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forked exploration trajectories receive SDK model IDs and usage/cost.

    Four changed files trigger the three parallel specialists; their child
    recorders must upgrade the inherited empty model when cost events arrive.
    """
    _silence_ui(monkeypatch)
    _answer_prompts(monkeypatch)

    partial_paths: list[Path] = []

    class PartialSiblingSDK(_FakeSDKClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            if not partial_paths:
                path = _write_live_sibling_canary(malformed=False)
                if path is not None:
                    partial_paths.append(path)

    monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", PartialSiblingSDK)

    config = RunConfig(target=str(deep_target_multi), cleanup=False, archive=False,)
    exit_code = await run(config)

    assert exit_code == 0, f"run() returned {exit_code}"
    assert captured_post.payloads, ("build_payload was never called: deep flow did not reach _post")
    payload = captured_post.payloads[-1]
    body = payload["body"]
    assert isinstance(body, str)
    assert len(partial_paths) == 1
    assert _PARTIAL_MODEL not in body

    # Print rendered markdown so a future failure trace shows what we observed.
    print("\n=== RENDERED PR COMMENT BODY ===")
    print(body)
    print("=== END RENDERED PR COMMENT BODY ===\n")

    # --- Top-level rollup must reflect a real model + non-zero cost -------
    model_line = _line_starting(body, "- **Model:**")
    assert "unknown" not in model_line.lower(), (f"BUG: rollup Model line still says 'unknown' even when exploration "
        f"forks observed real model ids.\n  got: {model_line!r}\n\n"
        f"full body:\n{body}"
    )

    cost_line = _line_starting(body, "- **Cost:**")
    assert "$0.00" not in cost_line and "—" not in cost_line, (
        f"BUG: rollup Cost line is degraded.\n  got: {cost_line!r}"
    )

    # --- Locate the Exploration row in the per-phase breakdown ------------
    rows = _phase_rows(body)
    exploration_rows = [
        r for r in rows if re.search(r"\|\s*Exploration\s*\|", r, re.IGNORECASE)
    ]
    assert exploration_rows, (
        "BUG: per-phase breakdown is missing an 'Exploration' row entirely.\n"
        f"  rows: {rows!r}\n\nfull body:\n{body}"
    )
    assert len(exploration_rows) == 1, (f"unexpected: {len(exploration_rows)} Exploration rows in breakdown.")
    exploration_row = exploration_rows[0]
    cells = _row_cells(exploration_row)
    # Layout: | Phase | Model | Tools | Input (cached) | Output | Cost |
    assert len(cells) >= 6, f"unexpected row layout: {exploration_row!r}"
    (_phase_name, model_cell, tools_cell, _input_cell, _output_cell, cost_cell,) = cells[:6]

    # --- The three production-bug symptoms, asserted one at a time. -------

    # 1. Model column — production shows 'unknown'. The fork's child
    #    recorder should have upgraded this to the SDK id.
    assert model_cell.lower() != "unknown", (
        f"BUG: Exploration row Model='unknown' (matches production bug).\n"
        f"  row: {exploration_row!r}\n\nfull body:\n{body}"
    )
    assert FIXTURE_MODEL_ID in model_cell, (f"BUG: Exploration row Model cell is missing real SDK id "
        f"{FIXTURE_MODEL_ID!r}.\n  got Model cell: {model_cell!r}\n"
        f"  row: {exploration_row!r}"
    )

    # 2. Cost column — production shows '$0.00' (or '—'). Per-step cost
    #    should be aggregated from the fork CostEvents.
    assert cost_cell != "$0.00", (f"BUG: Exploration row Cost='$0.00' (matches production bug — "
        f"per-step cost from fork trajectories never aggregated).\n"
        f"  row: {exploration_row!r}"
    )
    assert cost_cell != "—", (
        f"BUG: Exploration row Cost='—' (cost_unknown flipped True).\n"
        f"  row: {exploration_row!r}"
    )

    # 3. Tools column — production shows 53; even one tool call per fork
    #    should produce a non-zero count here.
    assert tools_cell != "0", (
        f"BUG: Exploration row Tools='0' but the forks issued tool calls.\n"
        f"  row: {exploration_row!r}"
    )
    assert tools_cell == "3"
    assert cost_cell == "$0.36"

async def test_deep_run_posts_safe_fallback_when_completed_sibling_is_malformed(
    deep_target_multi: Path, patch_sdk: None, captured_post: _CapturedPost, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bad retained document degrades run details, not the authorized post."""

    _silence_ui(monkeypatch)
    _answer_prompts(monkeypatch)
    malformed_paths: list[Path] = []

    class MalformedSiblingSDK(_FakeSDKClient):
        async def query(self, prompt: str) -> None:
            await super().query(prompt)
            # Inject at merge so live post acquisition encounters the
            # malformed completed child.
            if not malformed_paths and "cross-stack merge agent" in prompt.lower():
                path = _write_live_sibling_canary(malformed=True)
                if path is not None:
                    malformed_paths.append(path)

    monkeypatch.setattr("daydream.backends.claude.ClaudeSDKClient", MalformedSiblingSDK)
    exit_code = await run(RunConfig(target=str(deep_target_multi), cleanup=False, archive=False))

    assert exit_code == 0
    assert len(malformed_paths) == 1
    assert len(captured_post.payloads) == 1
    body = captured_post.payloads[0]["body"]
    assert "*run details unavailable*" in body
    assert "- **Reviewed commit:** [`0000000`]" in body
    output = capsys.readouterr().out
    assert "run info: trajectory document invalid" in output
    assert "private-prompt-canary" not in body + output
    assert str(malformed_paths[0]) not in body + output
