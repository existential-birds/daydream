"""Replay recorded ``agy`` stream-json through the shared transport seam.

Each backend instance first launches ``agy --version`` and then one process per
invocation, so every replay scripts both launches. The per-run directory is removed at
teardown; its files are captured when the invocation's process is spawned.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.harness.fake_cli_process import FakeCliScript, FakeCliSpawner, install_fake_cli_process

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "antigravity_jsonl"


def fixture_lines(name: str) -> list[str]:
    return (FIXTURES_DIR / name).read_text().strip().split("\n")


@dataclass
class AgyReplay:
    """What the adapter launched, and each invocation's run directory as agy would have read it."""

    spawner: FakeCliSpawner
    run_dirs: list[Path] = field(default_factory=list)
    run_files: list[dict[str, str]] = field(default_factory=list)

    @property
    def invocations(self) -> list[tuple[str, ...]]:
        """Argv of every launch but the version checks."""
        return [argv for argv in self.spawner.argvs if "--version" not in argv]


def install_agy(
    monkeypatch: pytest.MonkeyPatch, *invocations: list[str], version: str = "1.5.15", exit_code: int = 0,
    stderr_lines: list[str] | None = None, hang: bool = False,
) -> AgyReplay:
    """Script one version check followed by one launch per given stdout transcript."""
    spawner = install_fake_cli_process(
        monkeypatch, "agy", lines=[], stderr_lines=stderr_lines,
        scripts=[FakeCliScript([version]),
                 *(FakeCliScript(lines, hang=hang, exit_code=exit_code) for lines in invocations)])
    replay = AgyReplay(spawner)
    launch = asyncio.create_subprocess_exec

    async def observing(*args: Any, **kwargs: Any) -> Any:
        if "--add-dir" in args:
            root = Path(args[args.index("--add-dir") + 1])
            replay.run_dirs.append(root)
            replay.run_files.append({path.relative_to(root).as_posix(): path.read_text()
                                     for path in sorted(root.rglob("*")) if path.is_file()})
        return await launch(*args, **kwargs)

    monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", observing)
    return replay
