"""Shared fixtures: the deterministic fixture repo, a real runtime, a stub upstream.

The fixture repository itself lives in the package
(:mod:`daydream_review_v1.fixture`) because the image builder needs it too.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import AsyncIterator, Iterator

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime, SubprocessRuntimeInfo

from daydream_review_v1.fixture import FixtureRepo, build_fixture_repo
from daydream_review_v1.stub_upstream import serve

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fixture_repo(tmp_path: Path) -> FixtureRepo:
    """A freshly built deterministic fixture repository."""
    return build_fixture_repo(tmp_path / "daydream-rl-fixture")


@pytest.fixture(scope="session")
def corpus_mini_dir() -> Path:
    """The committed 2-PR harvested-format corpus over the fixture repo."""
    return PROJECT_ROOT / "tests" / "fixtures" / "corpus-mini"


@pytest.fixture(scope="session")
def fixture_manifest_path() -> Path:
    """The committed images manifest, which carries the fixture repo entry."""
    return PROJECT_ROOT / "images" / "manifest.toml"


@pytest.fixture(scope="session")
def rundir_golden() -> Path:
    """An archived run dir from a real local daydream run against the fixture repo."""
    return PROJECT_ROOT / "tests" / "fixtures" / "rundir-golden"


@pytest.fixture
async def runtime() -> AsyncIterator[SubprocessRuntime]:
    """A real verifiers subprocess runtime — not a double.

    It shares the host filesystem, so absolute sandbox paths in the tests are
    plain host temp dirs and ``read``/``run`` behave exactly as they do in a real
    rollout (``verifiers/v1/runtimes/subprocess.py``).
    """
    rt = SubprocessRuntime(SubprocessConfig())
    await rt.start()
    try:
        yield rt
    finally:
        rt.cleanup()


class FakeRuntime(vf.Runtime):
    """Records what a strategy or the harness does, without touching the host.

    Used only where a real runtime would have side effects the test must not
    have: ``pi install`` mutating the developer's own ``~/.pi``, or a full
    daydream run. Everything that can use the real subprocess runtime does.
    """

    is_local = True

    def __init__(self, *, exit_code: int = 0, files: dict[str, bytes] | None = None) -> None:
        super().__init__()
        self.config = SubprocessConfig()
        self.info = SubprocessRuntimeInfo(**self.config.model_dump())
        self.exit_code = exit_code
        self.files: dict[str, bytes] = dict(files or {})
        self.writes: dict[str, bytes] = {}
        self.commands: list[list[str]] = []
        self.programs: list[tuple[list[str], dict[str, str]]] = []

    async def start(self) -> None:
        return None

    async def run(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
        self.commands.append(argv)
        return vf.ProgramResult(exit_code=0, stdout="", stderr="")

    async def run_program(self, argv: list[str], env: dict[str, str]) -> vf.ProgramResult:
        self.programs.append((argv, env))
        return vf.ProgramResult(exit_code=self.exit_code, stdout="", stderr="")

    async def read(self, path: str) -> bytes:
        return self.files[path]

    async def write(self, path: str, data: bytes) -> None:
        self.writes[path] = data


@pytest.fixture
def stub_upstream() -> Iterator[str]:
    """Run a canned OpenAI-compatible upstream; yield its ``/v1`` base URL."""
    server = serve()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
