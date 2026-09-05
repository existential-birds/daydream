"""Shared fixtures: the deterministic fixture repo, a real runtime, a stub upstream.

The fixture repository itself lives in the package
(:mod:`daydream_review_v1.fixture`) because the image builder needs it too.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime, SubprocessRuntimeInfo

from daydream_review_v1.fixture import FixtureRepo, build_fixture_repo
from daydream_review_v1.stub_upstream import serve
from images import build_images

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def docker_daemon_is_available() -> bool:
    """Whether the Docker daemon is reachable, determined by ``docker info``.

    Returns ``True`` only when ``docker info`` exits with return code 0.
    If the client binary cannot be launched (``OSError``) or the daemon is
    unreachable (non-zero exit), returns ``False``.
    """
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, check=False)
    except OSError:
        return False
    return result.returncode == 0


@pytest.fixture(scope="session")
def base_image() -> str:
    """The versioned base image every PR-snapshot image is ``FROM``, built if absent.

    The versioned tag (``base_tag()``, e.g. ``daydream-rl/base:v0.1.2-3-g5ce4c0e``)
    is returned rather than the mutable ``latest`` alias, so every snapshot build
    the slow tests drive pins an explicit immutable base identity.
    """
    tag = build_images.base_tag()
    present = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False)
    if present.returncode != 0:
        subprocess.run(
            ["uv", "run", "python", "images/build_images.py", "--base-only"],
            cwd=PROJECT_ROOT,
            check=True,
        )
    return tag


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


_OUTCOME_MODEL_STATE: dict[str, Any] = {
    "weights": {"bug": 1.0, "race": 0.5, "regression": 0.75},
    "bias": -0.25,
    "split_digest": "fixture-split-digest",
    "label_ratio_reported": 0.5,
    "train_rows": 10,
    "held_out_rows": 4,
    "held_out_accuracy": 0.75,
    "model_fingerprint": "",
}

_GATE_EVIDENCE: dict[str, Any] = {
    "split_digest": _OUTCOME_MODEL_STATE["split_digest"],
    "model_fingerprint": _OUTCOME_MODEL_STATE["model_fingerprint"],
    "thresholds": {"min_separation": 0.1, "min_calibration": 0.5},
    "held_out_rows": _OUTCOME_MODEL_STATE["held_out_rows"],
    "separation": 0.2,
    "calibration": 0.75,
    "accepted_ratio": 0.5,
}


def _evidence_digest(payload: dict[str, Any]) -> str:
    """SHA-256 over the sorted evidence payload (mirror of ``gate.py``)."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@pytest.fixture
def stage0_gate_report(tmp_path: Path) -> Path:
    """A PASSED Stage-0 gate report, bound to the fixture outcome model (M4).

    The evidence_digest is recomputed over the same payload the offline gate
    hashes (``{split_digest, model_fingerprint, thresholds, held_out_rows,
    separation, calibration, accepted_ratio}``), so the taskset load path's
    gateway binding accepts it alongside the ``outcome_model_path`` fixture.
    """
    p = tmp_path / "stage0-gate.json"
    p.write_text(
        json.dumps(
            {
                "passed": True,
                "separation": _GATE_EVIDENCE["separation"],
                "calibration": _GATE_EVIDENCE["calibration"],
                "accepted_ratio": _GATE_EVIDENCE["accepted_ratio"],
                "evidence_digest": _evidence_digest(_GATE_EVIDENCE),
                "thresholds": dict(_GATE_EVIDENCE["thresholds"]),
                "held_out_rows": _GATE_EVIDENCE["held_out_rows"],
            }
        ),
        encoding="utf-8",
    )
    return p


@pytest.fixture
def outcome_model_path(tmp_path: Path) -> Path:
    """A trained Stage-0 outcome model checkpoint (OutcomeModel.state_dict shape)."""
    p = tmp_path / "outcome-model.json"
    p.write_text(json.dumps(_OUTCOME_MODEL_STATE), encoding="utf-8")
    return p


@pytest.fixture(scope="session")
def rundir_golden() -> Path:
    """An archived run dir from a real local daydream run against the fixture repo.

    UNTRUSTED, model-directed test-only data: the retained root
    ``trajectory.json`` carries operational text captured during the run. It is
    never a model-input source — the scoring projection excludes it at the
    ``fetch_run_dir`` boundary (see
    tests/test_rundir.py::test_fetch_run_dir_excludes_fixture_trajectories).
    """
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

    async def _read(self, path: str) -> bytes:
        return await self.read(path)

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


def assert_docstring_guards(
    func: Callable[..., object], *, gone: tuple[str, ...] = (), present: tuple[str, ...] = ()
) -> None:
    """Assert a docstring no longer claims the stale ``gone`` phrases and still
    describes the accurate ``present`` phrases. Keeps the docker docstring-guard
    tests (test_harness.py / test_images.py) DRY and the phrase lists the only
    difference between them."""
    doc = getattr(func, "__doc__", None)
    assert doc is not None, f"{func.__qualname__} has no docstring to guard"
    for phrase in gone:
        assert phrase not in doc, (
            f"{func.__qualname__} docstring still contains stale phrase {phrase!r}"
        )
    for phrase in present:
        assert phrase in doc, (
            f"{func.__qualname__} docstring missing accurate phrase {phrase!r}"
        )
