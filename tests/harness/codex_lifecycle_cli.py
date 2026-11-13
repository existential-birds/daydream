"""Synthetic public Codex JSONL from a real, controllable external process.

Fixture observations and FIFOs stay outside the repository and are never lifecycle
diagnostics. No native session or private rollout content is used. When its argv
registers the invocation's ``daydream`` MCP server, the process is a real MCP client
of that server and prints the public shapes the Codex CLI prints for those calls.
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LifecycleMode = Literal[
    "healthy", "terminal_open_stdout", "terminal_eof_live_child", "descendant_stdout",
    "nonzero_exit", "preterminal_silence", "repeated_terminal", "unfinished_next_turn",
    "completed_next_turn", "delayed_shutdown", "completed_chatter",
]
_FIXTURE_CONFIG: dict[str, Any] = globals().get("_FIXTURE_CONFIG", {})
FIX_MARKER = "\n# synthetic lifecycle fix\n"


@dataclass(frozen=True)
class CodexLifecycleCli:
    root: Path
    bin_dir: Path

    def observations(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text()) for path in sorted((self.root / "observations").glob("*.json"))]

    async def wait_for_stage(self, stage: str, *, timeout_s: float = 5.0) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while asyncio.get_running_loop().time() < deadline:
            for row in self.observations():
                if row["stage"] == stage:
                    return row
            await asyncio.sleep(0.01)
        raise AssertionError(f"synthetic Codex did not reach {stage}")

    def release(self, pid: int) -> None:
        """Release a blocked child after observing its reader-ready stage."""
        descriptor = os.open(self.root / "controls" / f"{pid}.fifo", os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(descriptor, b"release\n")
        finally:
            os.close(descriptor)


def install_codex_lifecycle_cli(
    root: Path, *, mode: LifecycleMode = "healthy", fix_modes: dict[str, LifecycleMode] | None = None,
    structure_mode: LifecycleMode | None = None, shutdown_delay_s: float = 0.0,
    empty_reviews: bool = False, spawn_on_term: bool = False,
    tool_output_path: Path | None = None,
) -> CodexLifecycleCli:
    root = root.resolve()
    for name in ("bin", "observations", "controls", "fix_started"):
        (root / name).mkdir(parents=True)
    config = {"root": str(root), "mode": mode, "fix_modes": fix_modes or {},
              "structure_mode": structure_mode, "shutdown_delay_s": shutdown_delay_s, "empty_reviews": empty_reviews,
              "spawn_on_term": spawn_on_term,
              "tool_output_path": None if tool_output_path is None else str(tool_output_path)}
    executable = root / "bin" / "codex"
    executable.write_text(
        f"#!{sys.executable}\n_FIXTURE_CONFIG = {config!r}\n" + Path(__file__).read_text(), encoding="utf-8",
    )
    executable.chmod(0o700)
    return CodexLifecycleCli(root, root / "bin")


def _emit(event: dict[str, Any]) -> None:
    print(json.dumps(event), flush=True)


def _observe(stage: str, *, mode: str, file: str | None = None, descendant: int | None = None,
             host: dict[str, Any] | None = None) -> None:
    path = Path(_FIXTURE_CONFIG["root"]) / "observations" / f"{os.getpid()}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "stage": stage, "mode": mode,
                                   "file": file, "cwd": str(Path.cwd()), "descendant": descendant, "host": host}))
    os.replace(temporary, path)


def _block(stage: str, *, mode: str, file: str | None = None) -> None:
    """Publish readiness only after this process holds an actual FIFO reader.

    POSIX read/write open avoids waiting for a writer on Darwin and Linux. A
    single-byte read then blocks until ``release`` supplies the readiness token.
    """
    fifo = Path(_FIXTURE_CONFIG["root"]) / "controls" / f"{os.getpid()}.fifo"
    os.mkfifo(fifo, 0o600)
    descriptor = os.open(fifo, os.O_RDWR)
    try:
        _observe(stage, mode=mode, file=file)
        os.read(descriptor, 1)
    finally:
        os.close(descriptor)


def _issue(file: str, ordinal: int) -> dict[str, Any]:
    return {"id": ordinal, "file": file, "line": 1, "severity": "high", "confidence": "HIGH",
            "description": f"Synthetic defect in {file}", "rationale": "fixture contract",
            "evidence": f"{file}:1"}


def _response(prompt: str) -> tuple[str, str | None]:
    """Answer real host prompts; only fix calls edit their assigned source file."""
    lower = prompt.lower()
    match = re.search(r"^File: (.+)$", prompt, re.M) or re.search(r"^Fix these \d+ issues in (.+):$", prompt, re.M)
    if lower.startswith(("fix this issue", "fix these")) and match:
        path = Path(match[1].strip())
        path = path if path.is_absolute() else Path.cwd() / path
        path.write_text(path.read_text() + FIX_MARKER)
        return f"Synthetic final answer for {path.name}", path.name
    _, marker, state_payload = prompt.partition("Host review stage:\n")
    if marker:
        state = json.JSONDecoder().raw_decode(state_payload)[0]
        candidates: list[dict[str, Any]] = []
        if state["stage"] in {"first_pass", "integration"} and not state["progress"]:
            candidates = [{"candidate_id": "", "file": file, "line": 1,
                           "trigger": "The changed path is exercised", "consequence": f"Synthetic defect in {file}",
                           "grounds": f"{file}:1", "disposition": "confirmed", "finding": _issue(file, ordinal)}
                          for ordinal, file in enumerate(state["assigned_files"], 1)]
        elif state["stage"] == "triage":
            candidates = [dict(row, disposition="confirmed") for row in state["candidates"]
                          if row["candidate_id"] in state["assigned_candidate_ids"]]
        # The structural reviewer contributes no duplicated file finding.
        if "you are the structural reviewer" in lower or _FIXTURE_CONFIG.get("empty_reviews"):
            candidates = []
        return json.dumps({"targets": [{"target_id": target, "status": "reviewed", "reason": "",
                                        "evidence_refs": [
                                            {"source_id": option["source_id"], "basis": "supplied_diff",
                                             "source_range": None}
                                            for option in state["source_options"][target][:1]]}
                                       for target in state["assigned_target_ids"]],
                           "notes": "Synthetic assigned review complete", "supporting_files": [],
                           "candidates": candidates,
                           "contradictions": []}), None
    if "cross-stack merge agent" in lower:
        items = []
        for ordinal, file in enumerate(("App.tsx", "api.py", "README.md"), 1):
            items.append({**_issue(file, ordinal), "lens": "per-stack", "related_files": None, "source_uids": None})
        return json.dumps({"items": items}), None
    if "post-fix fix-verifier agent" in lower:
        ids = [int(value) for value in re.findall(r"(?m)^(\d+)\. \[", prompt)]
        return json.dumps({"verdicts": [{"issue_id": value, "verdict": "resolved", "reason": "Synthetic verification"}
                                       for value in ids]}), None
    if "you are the recommendation-verifier agent" in lower:
        return json.dumps({"verdicts": [{"issue_id": 1, "verdict": "consistent", "evidence": "Synthetic verification",
                                        "unverified_assumptions": []}]}), None
    if "would you have done this differently" in lower or "evaluate the implementation" in lower:
        return json.dumps({"issues": []}), None
    if "you are the arbiter" in lower:
        match = re.search(r"listed in (\S*arbiter[-\w]*input\.json)", prompt)
        rows = json.loads(Path(match[1]).read_text()) if match else []
        return json.dumps({"findings": [{"arb_id": row["arb_id"], "keep": True,
                                         "severity": row["severity"], "confidence": row["confidence"],
                                         "description": row["description"], "rationale": "Synthetic confirmation",
                                         "evidence": row["evidence"]} for row in rows]}), None
    if "you are the suppression reviewer" in lower:
        return json.dumps({"findings": []}), None
    return "Synthetic lifecycle response", None


def _overrides() -> dict[str, Any]:
    """``-c`` overrides as Codex reads them: TOML values, or the raw text when that does not parse."""
    values: dict[str, Any] = {}
    for flag, text in zip(sys.argv, sys.argv[1:]):
        if flag == "-c":
            key, _, raw = text.partition("=")
            try:
                values[key] = tomllib.loads("value = " + raw)["value"]
            except tomllib.TOMLDecodeError:
                values[key] = raw
    return values


async def _host_tools(overrides: dict[str, Any], prompt: str, mode: str) -> None:
    """Drive the invocation's host tools over MCP, as a scripted model would.

    A committed-review assignment in the developer instructions is reviewed (lookup, two
    Codex-owned starts, a rejected then corrected submission, a call after acceptance)
    or challenged; otherwise the user prompt is the JSON submission.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import TextContent

    url = overrides["mcp_servers.daydream.url"]
    token = os.environ[overrides["mcp_servers.daydream.bearer_token_env_var"]]
    instructions = overrides.get("developer_instructions", "")

    def status(**request: Any) -> int:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, **request)) as response:
                return int(response.status)
        except urllib.error.HTTPError as exc:
            return exc.code

    # A request without the secret, then an authenticated request for a server-initiated stream.
    refused = [status(data=b"{}", method="POST"), status(headers={"Authorization": "Bearer " + token})]
    calls: list[list[Any]] = []
    async with streamablehttp_client(url, headers={"Authorization": "Bearer " + token}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = sorted(tool.name for tool in (await session.list_tools()).tools)
            _emit({"type": "thread.started", "thread_id": f"synthetic-{os.getpid()}"})
            _emit({"type": "turn.started"})

            async def call(name: str, arguments: dict[str, Any]) -> list[str]:
                item = {"id": f"item_{len(calls)}", "type": "mcp_tool_call", "server": "daydream", "tool": name,
                        "arguments": arguments, "result": None, "error": None, "status": "in_progress"}
                _emit({"type": "item.started", "item": item})
                result = await session.call_tool(name, arguments)
                texts = [block.text for block in result.content if isinstance(block, TextContent)]
                _emit({"type": "item.completed", "item": {
                    **item, "status": "failed" if result.isError else "completed",
                    "result": {"content": [{"type": "text", "text": text} for text in texts],
                               "structured_content": None}}})
                calls.append([name, result.isError, [text[:160] for text in texts]])
                return texts

            _, marker, assignment = instructions.partition("\nCommitted review assignment:\n")
            if not marker:
                await call("structured_output", json.loads(prompt))
            elif (work := json.JSONDecoder().raw_decode(assignment)[0])["kind"] == "challenge":
                claim = work["claims"][0]
                await call("structured_output", {"verdicts": [{
                    "uid": claim["uid"], "revision": claim["revision"], "verdict": "confirmed",
                    "grounds": "The caller divides by the changed value.", "evidence_refs": []}], "claims": []})
            else:
                targets = work["work"]["targets"]["ids"]
                decisions = [{"target_id": target, "status": "reviewed", "reason": "Captured change inspected.",
                              "evidence_refs": []} for target in targets]
                lookup = {"target_id": targets[0], "path": None, "cursor": None}
                texts = await call("review_source", lookup)
                option = next(item for item in json.loads(texts[0])["target"]["source_options"]
                              if item["side"] == "new")
                command = {"type": "command_execution", "id": "command", "command": "git diff --check"}
                _emit({"type": "item.started", "item": {**command, "status": "in_progress"}})
                _emit({"type": "item.completed", "item": {**command, "status": "completed",
                                                            "aggregated_output": "", "exit_code": 0}})
                other = {"type": "mcp_tool_call", "id": "other", "server": "user", "tool": "search", "arguments": {}}
                _emit({"type": "item.started", "item": {**other, "status": "in_progress"}})
                _emit({"type": "item.completed", "item": {**other, "status": "completed", "error": None,
                                                            "result": {"content": [], "structured_content": None}}})
                await call("structured_output", {"targets": [{**decisions[0], "target_id": "unplanned"}],
                                                 "claims": []})
                await call("structured_output", {"targets": decisions, "claims": [{
                    "uid": "", "revision": 0, "source_uids": [], "title": "Zero divides the caller", "payload": {
                        "claim": "VALUE becomes zero.", "trigger": "The caller divides by it.",
                        "consequence": "Requests fail.", "grounds": "The captured change sets zero.",
                        "file": option["path"], "line": 1, "evidence_refs": [
                            {key: option[key] for key in ("source_id", "basis", "source_range")}]}}]})
                await call("review_source", lookup)
    _emit({"type": "item.completed", "item": {"id": "final", "type": "agent_message", "text": "Assignment complete."}})
    # A native result field must never stand in for the host-accepted submission.
    _emit({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}, "result": {"native": True}})
    _observe("terminal", mode=mode, host={
        "overrides": {key: value for key, value in overrides.items() if key != "developer_instructions"},
        "instructions": instructions, "output_schema": "--output-schema" in sys.argv,
        "token_exposed": any(token in text for text in (*sys.argv, prompt)),
        "no_proxy": [os.environ.get(name) for name in ("NO_PROXY", "no_proxy")],
        "refused": refused, "tools": tools, "calls": calls})


def _run() -> int:
    if "--fixture-descendant" in sys.argv:
        _block("descendant_ready", mode="descendant_stdout")
        return 0
    if "--cd" in sys.argv:
        os.chdir(sys.argv[sys.argv.index("--cd") + 1])
    prompt = sys.stdin.read()
    overrides = _overrides()
    if "mcp_servers.daydream.url" in overrides:
        asyncio.run(_host_tools(overrides, prompt, _FIXTURE_CONFIG["mode"]))
        return 7 if _FIXTURE_CONFIG["mode"] == "nonzero_exit" else 0
    text, file = _response(prompt)
    mode = _FIXTURE_CONFIG["fix_modes"].get(file, _FIXTURE_CONFIG["mode"])
    if "you are the structural reviewer" in prompt.lower() and _FIXTURE_CONFIG["structure_mode"]:
        mode = _FIXTURE_CONFIG["structure_mode"]
    if file and _FIXTURE_CONFIG["fix_modes"]:
        started = Path(_FIXTURE_CONFIG["root"]) / "fix_started"
        (started / file).touch()
        deadline = time.monotonic() + 10
        while len(list(started.iterdir())) < len(_FIXTURE_CONFIG["fix_modes"]):
            if time.monotonic() >= deadline:
                raise RuntimeError("concurrent fix barrier was not reached")
            time.sleep(0.01)
    _emit({"type": "thread.started", "thread_id": f"synthetic-{os.getpid()}"})
    _emit({"type": "item.completed", "item": {"type": "agent_message", "id": "initial",
                                               "text": "Synthetic initial commentary", "phase": "commentary"}})
    if input_path := _FIXTURE_CONFIG.get("tool_output_path"):
        # This operator-owned fixture input is read by the real CLI process;
        # its public tool event must traverse native decoding and recording.
        command = "cat " + str(input_path)
        _emit({"type": "item.started", "item": {"type": "command_execution", "id": "fixture-input",
                                                  "command": command, "status": "in_progress"}})
        payload = {"data": json.loads(Path(input_path).read_text()),
                   "public": {"type": "record", "key": "ordinary-public-key"},
                   "usage": {"input": 0, "output": None}}
        _emit({"type": "item.completed", "item": {"type": "command_execution", "id": "fixture-input",
                "command": command, "status": "completed", "aggregated_output": json.dumps(payload), "exit_code": 0}})
    if mode == "preterminal_silence":
        if _FIXTURE_CONFIG.get("spawn_on_term"):
            def spawn_and_exit(signum: int, frame: Any) -> None:
                del signum, frame
                escaped = subprocess.Popen([sys.executable, "-I", "-c", "import time;time.sleep(3600)"],
                    start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                (Path(_FIXTURE_CONFIG["root"]) / "escaped.json").write_text(json.dumps({"pid": escaped.pid}))
                os._exit(0)

            signal.signal(signal.SIGTERM, spawn_and_exit)
        _block("initial", mode=mode, file=file)
        return 0
    _observe("initial", mode=mode, file=file)
    _emit({"type": "item.started", "item": {"type": "command_execution", "id": "command",
                                              "command": "git diff --check", "status": "in_progress"}})
    _emit({"type": "item.completed", "item": {"type": "command_execution", "id": "command",
                                                "command": "git diff --check", "status": "completed",
                                                "aggregated_output": "", "exit_code": 0}})
    _emit({"type": "item.completed", "item": {"type": "agent_message", "id": "final", "text": text,
                                                "phase": "final_answer"}})
    terminal = {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}
    _emit(terminal)
    if mode == "terminal_eof_live_child":
        os.close(1)
        os.close(2)
        _block("stdout_closed", mode=mode, file=file)
    elif mode == "terminal_open_stdout":
        _block("terminal", mode=mode, file=file)
    else:
        _observe("terminal", mode=mode, file=file)
    if mode == "unfinished_next_turn":
        _emit({"type": "turn.started"})
        _observe("unfinished_turn_exit", mode=mode, file=file)
    elif mode == "completed_next_turn":
        _emit({"type": "turn.started"})
        _emit({"type": "item.completed", "item": {"type": "agent_message", "id": "next-final",
                                                   "text": "Latest public turn", "phase": "final_answer"}})
        _emit(terminal)
    elif mode == "delayed_shutdown":
        time.sleep(_FIXTURE_CONFIG["shutdown_delay_s"])
    elif mode == "completed_chatter":
        for _ in range(1100):
            _emit({"type": "fixture.chatter"})
    elif mode == "descendant_stdout":
        descendant = subprocess.Popen([sys.executable, __file__, "--fixture-descendant"])
        deadline = time.monotonic() + 5
        ready = Path(_FIXTURE_CONFIG["root"]) / "observations" / f"{descendant.pid}.json"
        while not ready.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("descendant failed readiness")
            time.sleep(0.01)
        _observe("direct_exit", mode=mode, file=file, descendant=descendant.pid)
    elif mode == "repeated_terminal":
        while True:
            _emit(terminal)
            _emit({"type": "thread.started", "thread_id": "synthetic-chatter"})
            time.sleep(0.01)
    return 7 if mode == "nonzero_exit" else 0


if __name__ == "__main__":
    raise SystemExit(_run())
