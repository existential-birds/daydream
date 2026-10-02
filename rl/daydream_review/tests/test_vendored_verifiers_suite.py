"""Verify the env suite against prime-rl's vendored verifiers 0.2.0, which shadows this package's pinned
0.2.1. Set PRIME_RL_VENDORED_VERIFIERS to the checkout; absent workspaces skip with instructions. See
README "Vendored-verifiers skew (AC10)".
"""

import os
import subprocess
from pathlib import Path

import pytest

VENDORED_ENV_VAR = "PRIME_RL_VENDORED_VERIFIERS"
SKIP_REASON = ("AC10 gate not run: set PRIME_RL_VENDORED_VERIFIERS to prime-rl's vendored "
    "verifiers checkout (e.g. <prime-rl>/deps/verifiers) and re-run — see "
    "rl/daydream_review/README.md 'Vendored-verifiers skew (AC10)'. A "
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
    # Use PYTHONPATH to shadow the pinned verifiers 0.2.1 with the vendored 0.2.0 without mutating
    # the environment.
    env = dict(os.environ)
    env["PYTHONPATH"] = vendored + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        ["uv", "run", "pytest", "tests/", "-q"], cwd=ENV_DIR, capture_output=True, text=True, timeout=1800, env=env,
    )
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
