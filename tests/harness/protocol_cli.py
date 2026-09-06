"""External, stdlib-only CLI fixtures that exercise real adapter transports.

The installer copies this module into a disposable executable. Observation
configuration is embedded there, not passed through the model's environment or
prompt. Only test canary booleans and hashes are recorded, never file contents.
"""

import hashlib
import json
import os
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_CANARIES = (
    "SOURCE_CANARY", "PRIOR_REASONING_CANARY", "CURRENT_REASONING_CANARY",
    "SIBLING_REASONING_CANARY", "RESUME_CACHE_CANARY",
)
_FIXTURE_CONFIG: dict[str, Any] = globals().get("_FIXTURE_CONFIG", {})


@dataclass(frozen=True)
class ProtocolCli:
    bin_dir: Path
    executable: Path
    observations: Path
    entered: Path
    release: Path

    def read_observations(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.observations.glob("*.json"))]


def install_protocol_cli(
    root: Path,
    backend: Literal["codex", "pi", "osprey"],
    *,
    response_mode: Literal["success", "model_error", "process_error", "block"] = "success",
    sanctioned_files: tuple[Path, ...] = (),
) -> ProtocolCli:
    """Install a real executable; block mode releases through a FIFO write."""
    root = root.resolve()
    bin_dir = root / "bin"
    observations = root / "observations"
    bin_dir.mkdir(parents=True)
    observations.mkdir()
    release = root / "release"
    if response_mode == "block":
        os.mkfifo(release, mode=0o600)
    executable = bin_dir / backend
    config = {
        "root": str(root), "backend": backend, "response_mode": response_mode,
        "sanctioned_files": [str(path) for path in sanctioned_files],
    }
    executable.write_text(
        f"#!{sys.executable}\n_FIXTURE_CONFIG = {config!r}\n"
        + Path(__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return ProtocolCli(bin_dir, executable, observations, root / "entered", release)


def _atomic_observation(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _cwd_observation(cwd: Path) -> dict[str, Any]:
    hits = dict.fromkeys(_CANARIES, False)
    entries: list[str] = []
    truncated = False
    pending = [cwd]
    budget_exhausted = False
    while pending and not budget_exhausted:
        with os.scandir(pending.pop()) as children:
            for entry in children:
                if len(entries) >= 256:
                    truncated = budget_exhausted = True
                    break
                path = Path(entry.path)
                entries.append(path.relative_to(cwd).as_posix())
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                if not entry.is_file(follow_symlinks=False):
                    continue
                with path.open("rb") as handle:
                    payload = handle.read(131_073)
                if len(payload) > 131_072:
                    truncated = True
                    continue
                for canary in hits:
                    hits[canary] = hits[canary] or canary.encode() in payload
    return {"cwd_entries": sorted(entries), "cwd_canaries": hits, "walk_truncated": truncated}


def _observed_argv(backend: str, argv: list[str]) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Admit known fixture flags only; content values become lengths/hashes."""
    switches = {
        "codex": {"exec", "--experimental-json"},
        "pi": {"--no-session", "--no-skills"},
        "osprey": {"agent", "--events-jsonl", "--sandbox", "--read-only", "--ultracode",
                   "--atif-system-prompt-plaintext", "--immutable-runtime-surface",
                   "--compress-context=true", "--compress-context=false"},
    }[backend]
    values = {
        "codex": {"--model", "--sandbox", "--cd", "--output-schema", "resume"},
        "pi": {"--mode", "--model", "--provider", "--thinking", "--tools", "--session-id"},
        "osprey": {
            "--model", "--toolset", "--temperature", "--atif-output", "--max-turns", "--turn-timeout",
            "--stream-idle-timeout-secs", "--streaming-timeout-secs", "--empty-completion-threshold",
            "--driver-max-retries", "--approval", "--allowed-root", "--compress-min-bytes",
            "--tool-result-cap", "--tool-result-head", "--tool-result-tail", "--tool-result-max-lines",
            "--tool-result-raw-dir", "--retry-failure-threshold", "--no-progress-family-threshold",
            "--no-progress-family-window", "--no-progress-artifact-threshold", "--no-progress-suppression-window",
            "--fork-from", "--resume", "--output-schema", "--max-subagents", "--llm-rpm", "--effort",
            "--observation-budget-update-bytes", "--observation-budget-inline-bytes",
            "--observation-budget-admission-bytes",
        },
    }[backend]
    content = {"codex": {"-c"}, "pi": {"--append-system-prompt"}, "osprey": {"--persona", "--var"}}[backend]
    recorded: list[str] = []
    hashes: dict[str, list[dict[str, Any]]] = {}
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag in switches:
            recorded.append(flag)
            index += 1
            continue
        if flag not in values | content or index + 1 >= len(argv):
            raise ValueError("unsupported or incomplete fixture argument")
        value = argv[index + 1]
        recorded.extend([flag, "[content omitted]" if flag in content else value])
        if flag in content:
            payload = value.encode()
            hashes.setdefault(flag, []).append({"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        index += 2
    return recorded, hashes


def _option(argv: list[str], name: str, default: str) -> str:
    return argv[argv.index(name) + 1] if name in argv else default


def _emit(event: dict[str, Any]) -> None:
    print(json.dumps(event), flush=True)


def _codex_events(text: str, *, failed: bool) -> None:
    _emit({"type": "thread.started", "thread_id": "fixture-thread"})
    _emit({"type": "item.started", "item": {"type": "agent_message", "id": "msg-1", "content": []}})
    _emit({"type": "item.completed", "item": {
        "type": "agent_message", "id": "msg-1", "content": [{"type": "text", "text": text}],
    }})
    if failed:
        _emit({"type": "turn.failed", "error": {"message": "fixture failure"}})
    else:
        _emit({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}})


def _pi_events(text: str, model: str, *, failed: bool) -> None:
    message = {
        "role": "assistant", "content": [{"type": "text", "text": text}],
        "model": model, "provider": "nous", "timestamp": int(time.time() * 1000),
    }
    _emit({"type": "session", "sessionId": "fixture-pi-session"})
    _emit({"type": "agent_start"})
    _emit({"type": "turn_start"})
    _emit({"type": "message_start", "message": message})
    completed = {
        **message, "usage": {"input": 10, "output": 5, "cacheRead": 0},
        "stopReason": "error" if failed else "stop",
    }
    if failed:
        completed["errorMessage"] = "fixture failure"
    _emit({"type": "message_end", "message": completed})
    _emit({"type": "turn_end", "message": completed})
    _emit({"type": "agent_end", "messages": []})


def _osprey_events(text: str, model: str, *, failed: bool, exit_code: int) -> None:
    _emit({"event": "protocol", "version": 2})
    _emit({
        "event": "session_start", "session_id": "fixture-osprey-session",
        "started_at": "2026-09-06T00:00:00Z", "model": model, "provider": "fixture-provider",
    })
    _emit({"event": "turn_start", "turn_id": "turn-1", "timestamp": "2026-09-06T00:00:00Z"})
    _emit({"event": "text_delta", "content": text})
    _emit({"event": "turn_end", "turn_id": "turn-1", "usage_reported": True,
           "duration_ms": 1, "prompt_tokens": 10, "completion_tokens": 5,
           "cached_tokens": 0, "thinking_tokens": 0, "cost_usd": None, "model": model})
    _emit({
        "event": "session_end", "total_turns": 1, "session_wallclock_ms": 1,
        "total_cost_usd": None, "total_prompt_tokens": 10, "total_completion_tokens": 5,
        "total_cached_tokens": 0, "total_cache_write_tokens": 0, "total_thinking_tokens": 0,
        "total_oom_kills": 0, "p50_turn_ms": 1, "p99_turn_ms": 1, "avg_turn_cost_usd": None,
        "structured_output": None, "outcome": "failed" if failed else "completed",
        "verification": None, "exit_code": exit_code,
    })


def _run_cli() -> int:
    config = _FIXTURE_CONFIG
    root = Path(config["root"])
    backend = config["backend"]
    mode = config["response_mode"]
    argv = sys.argv[1:]
    stdin = sys.stdin.buffer.read()
    prompt = stdin.decode("utf-8") if backend == "codex" else argv[-1]
    args_without_prompt = argv if backend == "codex" else argv[:-1]
    recorded_argv, content_arguments = _observed_argv(backend, args_without_prompt)
    cwd = Path(_option(argv, "--cd", str(Path.cwd()))).resolve()
    opened: dict[str, str] = {}
    for requested in config["sanctioned_files"]:
        if requested in prompt:
            path = Path(requested)
            if stat.S_ISREG(path.lstat().st_mode):
                with path.open("rb") as handle:
                    opened[requested] = hashlib.sha256(handle.read(12_289)).hexdigest()
    observation = {
        "backend": backend, "response_mode": mode, "pid": os.getpid(),
        "argv": recorded_argv, "content_arguments": content_arguments,
        "inherited_cwd": str(Path.cwd()), "effective_cwd": str(cwd),
        "stdin_bytes": len(stdin), "stdin_sha256": hashlib.sha256(stdin).hexdigest(),
        "prompt_bytes": len(prompt.encode()), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_canaries": {canary: canary in prompt for canary in _CANARIES},
        "sanctioned_reads": opened, "process_outcome": "entered", **_cwd_observation(cwd),
    }
    path = root / "observations" / f"{backend}-{os.getpid()}-{time.monotonic_ns()}.json"
    _atomic_observation(path, observation)
    _atomic_observation(root / "entered", {"pid": os.getpid()})
    if mode == "block":
        with (root / "release").open("r", encoding="utf-8") as release:
            release.read()
    failed = mode in ("model_error", "process_error")
    exit_code = 7 if mode == "process_error" else 0
    text = "CURRENT_REASONING_CANARY"
    model = _option(argv, "--model", "fixture-model")
    if backend == "codex":
        _codex_events(text, failed=failed)
    elif backend == "pi":
        _pi_events(text, model, failed=failed)
    else:
        _osprey_events(text, model, failed=failed, exit_code=exit_code)
    observation["process_outcome"] = mode
    _atomic_observation(path, observation)
    if exit_code:
        print("fixture process failure", file=sys.stderr, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_run_cli())
