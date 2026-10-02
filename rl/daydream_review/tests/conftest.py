"""Shared deterministic repository, real runtime, and stub upstream; image builds use the packaged fixture."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime, SubprocessRuntimeInfo

from daydream_review.gate_refusal import _evidence_digest
from daydream_review.stub_upstream import serve
from images import build_images

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def docker_daemon_is_available() -> bool:
    """Return whether docker info succeeds; missing clients and unreachable daemons return False."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, check=False)
    except OSError:
        return False
    return result.returncode == 0


@pytest.fixture(scope="session")
def base_image() -> str:
    """Build the base image if absent and return its versioned tag, never mutable latest."""
    tag = build_images.base_tag()
    present = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False)
    if present.returncode != 0:
        subprocess.run(["uv", "run", "python", "images/build_images.py", "--base-only"], cwd=PROJECT_ROOT, check=True)
    return tag


@pytest.fixture(scope="session")
def fixture_manifest_path() -> Path:
    """The committed images manifest, which carries the fixture repo entry."""
    return PROJECT_ROOT / "images" / "manifest.toml"

_OUTCOME_MODEL_STATE: dict[str, Any] = {
    "weights": {"bug": 1.0, "race": 0.5, "regression": 0.75}, "bias": -0.25, "split_digest": "fixture-split-digest",
    "label_ratio_reported": 0.5, "train_rows": 10, "held_out_rows": 4, "held_out_accuracy": 0.75,
    "model_fingerprint": "",
}

_GATE_EVIDENCE: dict[str, Any] = {"split_digest": _OUTCOME_MODEL_STATE["split_digest"],
    "model_fingerprint": _OUTCOME_MODEL_STATE["model_fingerprint"],
    "thresholds": {"min_separation": 0.1, "min_calibration": 0.5},
    "held_out_rows": _OUTCOME_MODEL_STATE["held_out_rows"], "separation": 0.2, "calibration": 0.75,
    "accepted_ratio": 0.5,
}


@contextmanager
def passed_gate_report() -> Iterator[Path]:
    """Yield a minimal PASSED Stage-0 gate report path, removed on exit."""
    fd, gate_name = tempfile.mkstemp(suffix="-stage0-gate.json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"passed": True, "separation": 0.2, "evidence_digest": "test"}, fh)
    try:
        yield Path(gate_name)
    finally:
        os.unlink(gate_name)


@pytest.fixture
def stage0_gate_report(tmp_path: Path) -> Path:
    """Build a passing Stage-0 gate with the offline evidence digest bound to the fixture model."""
    p = tmp_path / "stage0-gate.json"
    p.write_text(json.dumps({"passed": True, "separation": _GATE_EVIDENCE["separation"],
                "calibration": _GATE_EVIDENCE["calibration"], "accepted_ratio": _GATE_EVIDENCE["accepted_ratio"],
                "evidence_digest": _evidence_digest(_GATE_EVIDENCE), "thresholds": dict(_GATE_EVIDENCE["thresholds"]),
                "held_out_rows": _GATE_EVIDENCE["held_out_rows"],
            }
        ), encoding="utf-8",
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
    """A real archived run retained as untrusted test data, never model input.

    fetch_run_dir excludes its root trajectory from the scoring projection.
    """
    return PROJECT_ROOT / "tests" / "fixtures" / "rundir-golden"


@pytest.fixture
async def runtime() -> AsyncIterator[SubprocessRuntime]:
    """Use real SubprocessRuntime; absolute sandbox paths map to host temporary directories."""
    rt = SubprocessRuntime(SubprocessConfig())
    await rt.start()
    try:
        yield rt
    finally:
        rt.cleanup()


class FakeRuntime(vf.Runtime):
    """Record calls where real execution would mutate host Pi config or launch a full model run."""

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

