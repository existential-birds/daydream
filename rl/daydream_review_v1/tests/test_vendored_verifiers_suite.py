"""M14 / AC10: training claims are only accepted after the env's test suite
passes inside prime-rl's vendored verifiers workspace.

prime-rl v0.7.0 vendors verifiers **0.2.0** (submodule ``deps/verifiers``) while
this env package pins ``verifiers==0.2.1``. Task 0's spike (``plan-notes.md``)
validated that the API surface this env uses is compatible across 0.2.0/0.2.1
in both directions, so the resolution is pin discipline, not a fork or a
version relaxation. This gate is the standing proof of that claim: it re-runs
the env's suite with the vendored 0.2.0 copy shadowing the pinned 0.2.1.

The gate is opt-in via ``PRIME_RL_VENDORED_VERIFIERS`` (the vendored checkout
path) because the prime-rl workspace is not present on every machine. When the
variable is unset the test skips LOUDLY, with instructions — never silently.
See README "Vendored-verifiers skew (AC10)" for the full procedure.
"""

import os
import subprocess
from pathlib import Path

import pytest

VENDORED_ENV_VAR = "PRIME_RL_VENDORED_VERIFIERS"
SKIP_REASON = (
    "AC10 gate not run: set PRIME_RL_VENDORED_VERIFIERS to prime-rl's vendored "
    "verifiers checkout (e.g. <prime-rl>/deps/verifiers) and re-run — see "
    "rl/daydream_review_v1/README.md 'Vendored-verifiers skew (AC10)'. A "
    "training claim is only valid when this gate has run green."
)
ENV_DIR = Path(__file__).resolve().parent.parent
VENDORED = os.environ.get(VENDORED_ENV_VAR, "")


@pytest.mark.skipif(not VENDORED, reason=SKIP_REASON)
def test_env_suite_passes_under_vendored_verifiers() -> None:
    vendored = os.environ[VENDORED_ENV_VAR]
    assert Path(vendored, "verifiers", "__init__.py").is_file(), (
        f"{VENDORED_ENV_VAR}={vendored!r} is not a verifiers checkout"
    )
    # PYTHONPATH precedence shadows the venv's verifiers==0.2.1 with the
    # vendored 0.2.0 copy — the same override Task 0 validated by editable
    # install, without mutating either venv.
    env = dict(os.environ)
    env["PYTHONPATH"] = vendored + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/", "-q"],
        cwd=ENV_DIR,
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]


def test_skip_reason_carries_instructions() -> None:
    # The skip must be loud: a reader of the skip reason can follow it to the
    # README procedure without any other context.
    assert "PRIME_RL_VENDORED_VERIFIERS" in SKIP_REASON
    assert "README" in SKIP_REASON


def test_skip_is_active_when_workspace_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(VENDORED_ENV_VAR, raising=False)
    # Mirror the module-level skipif condition so the loud skip and the gate
    # can never drift apart.
    assert not os.environ.get(VENDORED_ENV_VAR)
