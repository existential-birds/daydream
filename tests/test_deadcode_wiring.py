from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

def _ci():
    wf = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text())
    return {name.lower(): job for name, job in wf["jobs"].items()}

def _root_vulture_step(steps):
    return [s for s in steps if "vulture" in s.get("run", "") and "rl/" not in s.get("run", "")]

def test_ci_check_job_has_deadcode_step_mirroring_makefile():
    steps = _ci()["check"]["steps"]
    found = _root_vulture_step(steps)
    assert len(found) == 1
    run = found[0]["run"].strip()
    assert run == "uv run vulture --config pyproject.toml daydream tests"
    lint_idx = next(i for i, s in enumerate(steps) if "ruff" in s.get("run", ""))
    assert steps.index(found[0]) > lint_idx  # slots after lint, mirroring check order

def test_ci_rl_job_has_rl_scoped_deadcode_step():
    job = next(j for name, j in _ci().items() if "rl" in name)
    rl_steps = [
        s for s in job["steps"]
        if "vulture" in s.get("run", "") and "daydream_review_v1 tests" in s.get("run", "")
    ]
    assert len(rl_steps) == 1
    # rl job sets working-directory defaults; asserted below per-job
    assert "working-directory" in job.get("defaults", {}).get("run", {})
