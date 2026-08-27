"""Dead-code parity: every gate surface (makefile, CI, pre-push hook) executes
identically-scoped vulture invocations.

Divergence points:
- Site A (Makefile root scan): runs at repo root, resolves root pyproject.toml.
- Site B (Makefile rl scan): runs via ``cd`` into ``rl/daydream_review_v1``,
  resolves THAT dir's pyproject.toml via ``--config``.
- Site C (ci.yml ``check`` job): inherits repo root cwd, byte-identical to Site A.
- Site D (ci.yml ``rl-check`` job): uses ``defaults.run.working-directory`` set to
  ``rl/daydream_review_v1``, so ``--config pyproject.toml`` resolves inside the
  subdir. Same file as Site B.
- Site E (pre-push): combines Sites A+B inline; ``set -e`` + subshell propagation
  confirmed safe (non-zero exits propagate).

Failure of any assertion here means a later edit to one surface forgot the
others — the local-parity invariant is broken.
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
_MK = (REPO / "Makefile").read_text()
_CI = (REPO / ".github" / "workflows" / "ci.yml").read_text()
_HOOK = (REPO / "scripts" / "hooks" / "pre-push").read_text()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _makefile_deadcode_block() -> str:
    """Return the indented recipe body under the ``deadcode:`` target."""
    after = _MK.split("deadcode:")[1]
    lines = after.splitlines()
    body_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip leading empty strings from the split.
        if stripped == "" and not body_lines:
            continue
        # Stop at a completely blank line that breaks the recipe.
        if stripped == "":
            break
        # Also stop if we hit another target (no leading tab).
        if not line.startswith("\t"):
            break
        body_lines.append(stripped)
    return "\n".join(body_lines)


def _ci_step_run(job_name: str, step_name: str) -> str:
    """Return the ``run`` string of a named step inside a named job."""
    data = yaml.safe_load(_CI)
    for step in data["jobs"][job_name]["steps"]:
        if step.get("name") == step_name:
            return str(step["run"])
    raise RuntimeError(f"step {step_name!r} not found in job {job_name!r}")


# ---------------------------------------------------------------------------
# Parity assertions
# ---------------------------------------------------------------------------

def test_root_argv_parity_makefile_and_ci() -> None:
    """Root scan invocation is byte-identical in Makefile and CI check job."""
    mk_body = _makefile_deadcode_block()
    root_line = mk_body.splitlines()[0]
    assert root_line == "uv run vulture --config pyproject.toml daydream tests"
    ci_run = _ci_step_run("check", "Run vulture")
    assert ci_run == "uv run vulture --config pyproject.toml daydream tests"


def test_rl_argv_parity_makefile_and_ci() -> None:
    """RL scan invocation matches between Makefile and CI rl-check job."""
    mk_body = _makefile_deadcode_block()
    rl_line = mk_body.splitlines()[1]
    assert rl_line == (
        "cd rl/daydream_review_v1 && "
        "uv run vulture --config pyproject.toml daydream_review_v1 tests"
    )
    ci_run = _ci_step_run("rl-check", "Run vulture")
    assert ci_run == "uv run vulture --config pyproject.toml daydream_review_v1 tests"


def test_hook_contains_both_invocations_verbatim() -> None:
    """Pre-push script contains the exact two vulture command lines."""
    assert "uv run vulture --config pyproject.toml daydream tests" in _HOOK
    assert "cd rl/daydream_review_v1 &&" in _HOOK
    assert "uv run vulture --config pyproject.toml daydream_review_v1 tests" in _HOOK


def test_all_vulture_lines_use_explicit_config() -> None:
    """No vulture invocation anywhere omits the explicit ``--config pyproject.toml``."""
    for line in _MK.splitlines():
        if "vulture" in line and not line.strip().startswith("#"):
            assert "--config pyproject.toml" in line, (
                f"Makefile line omits --config: {line!r}"
            )
    for line in _HOOK.splitlines():
        if "vulture" in line and not line.strip().startswith("#"):
            # Only check actual command lines, not echo banners.
            if "uv run" not in line:
                continue
            assert "--config pyproject.toml" in line, (
                f"Hook line omits --config: {line!r}"
            )
    ci_data = yaml.safe_load(_CI)
    for job_name, job in ci_data["jobs"].items():
        for step in job.get("steps", []):
            run = step.get("run", "")
            if "vulture" in run:
                assert "--config pyproject.toml" in run, (
                    f"CI job {job_name} step {step.get('name')!r} omits --config"
                )
