"""Synthetic public Codex JSONL from a real, controllable external process.

Fixture observations and FIFOs stay outside the repository and are never lifecycle
diagnostics. No native session or private rollout content is used.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
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
) -> CodexLifecycleCli:
    root = root.resolve()
    for name in ("bin", "observations", "controls", "fix_started"):
        (root / name).mkdir(parents=True)
    config = {"root": str(root), "mode": mode, "fix_modes": fix_modes or {},
              "structure_mode": structure_mode, "shutdown_delay_s": shutdown_delay_s}
    executable = root / "bin" / "codex"
    executable.write_text(
        f"#!{sys.executable}\n_FIXTURE_CONFIG = {config!r}\n" + Path(__file__).read_text(), encoding="utf-8",
    )
    executable.chmod(0o700)
    return CodexLifecycleCli(root, root / "bin")


def _emit(event: dict[str, Any]) -> None:
    print(json.dumps(event), flush=True)


def _observe(stage: str, *, mode: str, file: str | None = None, descendant: int | None = None) -> None:
    path = Path(_FIXTURE_CONFIG["root"]) / "observations" / f"{os.getpid()}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "stage": stage, "mode": mode,
                                   "file": file, "cwd": str(Path.cwd()), "descendant": descendant}))
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
        if "you are the structural reviewer" in lower:
            candidates = []
        return json.dumps({"targets": [{"target_id": target, "status": "reviewed", "reason": "",
                                        "evidence_refs": state["source_refs"][target][:1]}
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


def _run() -> int:
    if "--fixture-descendant" in sys.argv:
        _block("descendant_ready", mode="descendant_stdout")
        return 0
    if "--cd" in sys.argv:
        os.chdir(sys.argv[sys.argv.index("--cd") + 1])
    prompt = sys.stdin.read()
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
    if mode == "preterminal_silence":
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
