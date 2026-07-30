"""Real-path tests: broken extensions fail fast with named, actionable errors.

Enters from the production entrypoint (``runner.run``) with a real temp git
repo, mocking ONLY the backend seam (``daydream.runner.create_backend``) per
the testing standard. A ``daydream_ext`` package written by the ``ext_dir``
fixture is deliberately broken (dangling flow reference / wrong API version);
assertions pin the CLI-visible outcome: exit code 1, zero agents run, and
error output naming the broken piece (Task 16 of the extension-seam plan).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream import runner
from daydream.extensions import (
    EXTENSION_API_VERSION,
    MIN_SUPPORTED_EXTENSION_API_VERSION,
)
from daydream.runner import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend


async def test_broken_flow_ref_fails_before_any_agent(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A flow entry naming an unregistered phase aborts before any agent runs.

    ``insert_after`` accepts the dangling name at mutation time (entry names
    resolve at ``run_flow``'s pre-flight pass), so this pins the CLI-visible
    outcome through ``runner.run``: exit 1, zero agents, named broken piece.
    """
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 4\n"
        "def register(r):\n"
        "    r.insert_after('deep', anchor='intent', step='ghost_phase')\n"
    )
    backend = ScriptedBackend()
    install_backend(backend)

    rc = await runner.run(make_config(multi_stack_target))

    assert rc == 1
    assert backend.prompts == []                      # zero agents ran
    assert "ghost_phase" in capsys.readouterr().out   # error names the broken piece


async def test_version_mismatch_exits_1_naming_versions(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A DAYDREAM_EXT_API mismatch exits 1 naming both versions, before any git work."""
    # 99 is above the ceiling.
    ext_dir.write_module("DAYDREAM_EXT_API = 99\ndef register(r): ...\n")
    install_backend(ScriptedBackend())

    rc = await runner.run(make_config(multi_stack_target))

    assert rc == 1
    # Collapse the Rich panel's borders and wrapping: the message is wrapped at
    # width 80, so a bare substring check can straddle a line break.
    out = " ".join(re.sub(r"[\u2550-\u256c]", " ", capsys.readouterr().out).split())
    assert "DAYDREAM_EXT_API = 99" in out
    assert (
        f"supports {MIN_SUPPORTED_EXTENSION_API_VERSION}..{EXTENSION_API_VERSION}"
        in out
    )
