"""The Antigravity guard hook and the per-run directory that installs it."""

from __future__ import annotations

import json
import shlex
import stat
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import _antigravity_run as run_directory
from daydream.backends.antigravity_guard import decide

_BEARER = "callback-secret-123"


@pytest.fixture
def mutating(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """A mutating run directory with host tools, the widest policy the generator writes."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    created = run_directory.create_run_directory(
        read_only=False, review_instructions=None, mcp_url="http://127.0.0.1:1/mcp", mcp_token=_BEARER)
    yield created
    created.remove()


def _payload(name: Any, args: Any, cwd: Any) -> dict[str, Any]:
    return {"hookEventName": "PreToolUse", "cwd": cwd, "toolCall": {"id": "call-1", "name": name, "args": args}}


@pytest.mark.parametrize(("name", "args", "denied"), [
    # Recorded hook alias of a registered tool, and the MCP descriptor read that precedes a host call.
    ("read_file", {"Path": "/agy-home/.gemini/antigravity-cli/mcp/daydream/structured_output.json"}, None),
    ("grep_search", {"Query": "needle", "SearchPath": "."}, None),
    ("ask_permission", {}, "not allowed"),
    ("unknown_alias", {"Path": "calc.py"}, "not allowed"),
    # Path confinement: absolute, relative to the hook cwd, through a symlink, and nested in a list.
    ("read_file", {"Path": "{root}/.agents/mcp_config.json"}, "private run directory"),
    ("read_file", {"Path": "../{root_name}/guard-policy.json"}, "private run directory"),
    ("list_dir", {"DirectoryPath": "{root}"}, "private run directory"),
    ("read_file", {"Path": "link/.agents/hooks.json"}, "private run directory"),
    ("grep_search", {"Query": "x", "Options": [{"Includes": ["{root}/guard.py"]}]}, "private run directory"),
    ("read_file", {"Path": "file://{root}/.agents/mcp_config.json"}, "private run directory"),
    ("read_file", {"Path": "bad\x00path"}, "could not be checked"),
    # Only this invocation's own MCP server is reachable.
    ("call_mcp_tool", {"ServerName": "daydream", "ToolName": "structured_output", "Arguments": {"a": 1}}, None),
    ("call_mcp_tool", {"ServerName": "operator", "ToolName": "search", "Arguments": {}}, "own MCP server"),
    ("call_mcp_tool", {"ToolName": "structured_output"}, "own MCP server"),
    # Shell: the shared dangerous and background rules, and no route to the run directory by text.
    ("run_command", {"CommandLine": "make test 2>&1 && git commit -m 'fix & tidy'"}, None),
    ("run_command", {"CommandLine": "rm -rf /"}, "dangerous command blocked"),
    ("run_command", {"CommandLine": "cd / && rm -rf --no-preserve-root /"}, "dangerous command blocked"),
    ("run_command", {"CommandLine": "make build |& tee log; ls"}, None),
    ("run_command", {"CommandLine": "sleep 600 &"}, "background command blocked"),
    ("run_command", {"CommandLine": "cat $(echo {root})/guard-policy.json"}, "private run directory"),
    ("run_command", {"CommandLine": ["ls"]}, "command line is malformed"),
    ("run_command", {}, "command line is malformed"),
    # Malformed payloads never allow.
    (7, {}, "payload is malformed"),
    ("read_file", ["calc.py"], "payload is malformed"),
])
def test_guard_decides_each_tool_call_under_the_generated_policy(
    mutating: Any, tmp_path: Path, name: Any, args: Any, denied: str | None,
) -> None:
    """Regression: a tool outside the policy, a path into the run directory (which holds the bearer and
    the guard's own policy), another MCP server, or a dangerous/background command is allowed."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "link").symlink_to(mutating.path)
    policy = json.loads((mutating.path / "guard-policy.json").read_text())
    resolved = json.loads(json.dumps(args).replace("{root}", str(mutating.path)).replace(
        "{root_name}", mutating.path.name))

    reason = decide(_payload(name, resolved, str(workspace)), policy)

    assert (reason is None) if denied is None else (reason is not None and denied in reason), reason


@pytest.mark.parametrize("policy", [
    None, {"allowed_tools": "read_file", "protected_root": "/r", "mcp_server": None, "denied_commands": []},
    {"allowed_tools": ["run_command"], "protected_root": "/r", "mcp_server": None,
     "denied_commands": [{"pattern": "(", "reason": "x"}]},
    {"allowed_tools": ["read_file"], "protected_root": "", "mcp_server": None, "denied_commands": []},
])
def test_guard_denies_under_a_policy_it_cannot_apply(policy: Any) -> None:
    """Regression: a truncated or hand-edited policy file (wrong types, an uncompilable rule, no
    protected root) turns the guard into an allow-all hook instead of a denial."""
    for payload in (_payload("read_file", {"Path": "calc.py"}, "/ws"), _payload("run_command", {"CommandLine": "ls"},
                                                                                "/ws"), "not an object"):
        assert decide(payload, policy) is not None


def test_generated_hook_command_never_exits_zero_silently_except_to_allow(mutating: Any, tmp_path: Path) -> None:
    """Regression: agy treats exit 0 with empty stdout as allow, so a crash, garbage stdin or an unreadable
    policy that ends that way silently disables the guard. Only the real process proves the exit contract
    of the command hooks.json names."""
    hooks = json.loads((mutating.path / ".agents" / "hooks.json").read_text())
    (hook,) = hooks["daydream"]["PreToolUse"]
    assert hook["matcher"] == ".*"
    argv = shlex.split(hook["hooks"][0]["command"])

    def run(stdin: str) -> tuple[int, str]:
        done = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=30, check=False)
        return done.returncode, done.stdout

    assert run(json.dumps(_payload("read_file", {"Path": "calc.py"}, str(tmp_path)))) == (0, "")
    code, out = run(json.dumps(_payload("run_command", {"CommandLine": "find / -name x"}, str(tmp_path))))
    assert code == 0 and json.loads(out) == {
        "decision": "deny", "reason": "daydream guard: dangerous command blocked (always-on guard)"}
    code, out = run("not json")
    assert code != 0 and json.loads(out)["decision"] == "deny"
    (mutating.path / "guard-policy.json").write_text("{")
    code, out = run(json.dumps(_payload("read_file", {"Path": "calc.py"}, str(tmp_path))))
    assert code != 0 and json.loads(out)["decision"] == "deny"
    argv[-1] = str(mutating.path / "missing.json")
    code, out = run(json.dumps(_payload("read_file", {"Path": "calc.py"}, str(tmp_path))))
    assert code != 0 and json.loads(out)["decision"] == "deny"


def test_run_directory_files_match_the_invocation_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Regression: a read-only invocation is given shell or edit tools, the instructions miss the agent's
    system prompt or an H1 line inside them opens another section, the default prompt is replaced, or the
    bearer lands outside the private MCP config."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    review = run_directory.create_run_directory(
        read_only=True, review_instructions="Review policy: cite lines.\n# Output\nUse #7 style.")
    hosted = run_directory.create_run_directory(
        read_only=True, review_instructions=None, mcp_url="http://127.0.0.1:9/mcp", mcp_token=_BEARER)
    edit = run_directory.create_run_directory(read_only=False, review_instructions=None)

    def files(created: Any) -> dict[str, str]:
        return {path.relative_to(created.path).as_posix(): path.read_text()
                for path in created.path.rglob("*") if path.is_file()}

    def frontmatter_tools(created: Any) -> list[str]:
        head = files(created)[f".agents/agents/{created.agent}.md"].split("---")[1].splitlines()
        return [line.removeprefix("  - ") for line in head if line.startswith("  - ")]

    for created in (review, hosted, edit):
        assert stat.S_IMODE(created.path.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in created.path.rglob("*") if path.is_file())
        assert frontmatter_tools(created) == list(created.tools)
        agent = files(created)[f".agents/agents/{created.agent}.md"]
        assert f"name: {created.agent}\n" in agent and "mainAgent: true\ninheritMcp: true\n" in agent
        assert "excludeDefaultComponents" not in agent

    assert (review.agent, hosted.agent, edit.agent) == ("daydream-review", "daydream-review", "daydream-edit")
    assert review.tools == run_directory.READ_ONLY_TOOLS
    # agy's registry has no call_mcp_tool: listing it fails startup, so only the guard policy names it.
    assert hosted.tools == run_directory.READ_ONLY_TOOLS
    assert 'Call one as call_mcp_tool with ServerName "daydream"' in files(hosted)[
        ".agents/agents/daydream-review.md"].split("# System Prompt")[1]
    assert edit.tools == run_directory.MUTATING_TOOLS
    assert files(review)[".agents/agents/daydream-review.md"].endswith(
        "\n# System Prompt\n\nReview policy: cite lines.\n\\# Output\nUse #7 style.\n")
    assert sorted(files(review)) == [".agents/agents/daydream-review.md", ".agents/hooks.json", "guard-policy.json",
                                     "guard.py"]

    review_policy = json.loads(files(review)["guard-policy.json"])
    assert review_policy == {
        "allowed_tools": ["view_file", "grep_search", "find_by_name", "list_dir", "read_file"],
        "protected_root": str(review.path), "mcp_server": None, "denied_commands": []}
    hosted_policy = json.loads(files(hosted)["guard-policy.json"])
    assert hosted_policy["mcp_server"] == "daydream" and "call_mcp_tool" in hosted_policy["allowed_tools"]
    edit_policy = json.loads(files(edit)["guard-policy.json"])
    assert {"run_command", "write_file", "edit_file", "multi_replace_file_content", "command_status"} <= set(
        edit_policy["allowed_tools"])
    assert "call_mcp_tool" not in edit_policy["allowed_tools"] and len(edit_policy["denied_commands"]) == 4

    assert json.loads(files(hosted)[".agents/mcp_config.json"]) == {"mcpServers": {"daydream": {
        "url": "http://127.0.0.1:9/mcp", "headers": {"Authorization": f"Bearer {_BEARER}"}}}}
    assert [name for name, text in files(hosted).items() if _BEARER in text] == [".agents/mcp_config.json"]

    for created in (review, hosted, edit):
        created.remove()
        created.remove()
        assert not created.path.exists()
