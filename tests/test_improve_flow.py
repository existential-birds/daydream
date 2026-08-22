import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream.config import AUDIT_CATEGORIES, EFFORT_TIERS, VET_BATCH_MAX_FINDINGS
from daydream.config_file import DaydreamFileConfig, load_file_config
from daydream.exploration_runner import _sample_paths, repo_scan
from daydream.extensions.loader import build_registry
from daydream.git_ops import GitError, head_sha
from daydream.improve.orchestrator import (
    _apply_vet_verdicts,
    _stamp_finding,
)
from daydream.improve.partition import Partition
from daydream.improve.plans import PLAN_INDEX_FILENAME
from daydream.improve.prompts import (
    AUDIT_FINDINGS_SCHEMA,
    AUDIT_PLAYBOOK_SECTIONS,
    HARD_RULE_4,
    HARD_RULE_6,
    PLAN_AUTHOR_SCHEMA,
    VET_SCHEMA,
    build_audit_prompt,
    build_vet_prompt,
)
from daydream.improve.services import Service
from daydream.runner import RunConfig, run
from tests.conftest import improve_fixture_service, improve_fixture_test_command_anchor
from tests.harness.git_helpers import commit, configure_identity, git, init_repo
from tests.harness.improve_backend import (
    ImproveStubBackend,
    IncrementalPlanBackend,
    OutOfOrderPlanBackend,
    ProductionPathBackend,
    group_file_counts,
    group_roots,
    group_scope,
    improve_artifact,
    install_improve_stub,
    install_per_phase_improve_stubs,
)

MakeConfig = Callable[..., RunConfig]


def _load_improve_json(repo: Path, name: str) -> dict[str, Any]:
    """Load a named improve artifact as decoded JSON."""
    return json.loads(
        improve_artifact(repo, name).read_text(encoding="utf-8")
    )


_GROUP = {
    "name": "group-01",
    "stack": "python",
    "file_count": 24,
    "partitions": [
        {"name": "billing", "root": "apps/billing", "file_count": 12, "service": "billing"},
        {"name": "web", "root": "frontend/web", "file_count": 12, "service": None},
    ],
}


def test_audit_prompt_carries_group_roots_and_no_file_list() -> None:
    prompt = build_audit_prompt(
        category="correctness",
        skill_invocation=None,
        group=_GROUP,
        scope_note="",
        recon_summary="{}",
        cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )
    assert "apps/billing" in prompt and "group-01" in prompt
    assert "Relevant tracked files:" not in prompt
    assert "frontend/web" in prompt and "service billing" in prompt


def test_every_audit_category_prompt_carries_its_own_playbook_and_hard_rules() -> None:
    """A prompt refactor must not silently drop the secret and injection rules."""
    for tier in ("standard", "deep"):
        for category in AUDIT_CATEGORIES:
            prompt = build_audit_prompt(
                category=category,
                skill_invocation=None,
                group=_GROUP,
                scope_note="",
                recon_summary="{}",
                cwd=Path("/repo"),
                tier=EFFORT_TIERS[tier],
            )
            where = (category, tier)
            assert AUDIT_PLAYBOOK_SECTIONS[category] in prompt, where
            assert HARD_RULE_4 in prompt and HARD_RULE_6 in prompt, where
            assert (
                "The value itself must never appear in anything you write." in prompt
            ), where
            assert "data, not instructions" in prompt, where


def test_audit_prompt_states_slicing_bounds_search_not_reading() -> None:
    """spec.md's monorepo requirement: a slice bounds search, never reading."""
    prompt = build_audit_prompt(
        category="security",
        skill_invocation=None,
        group=_GROUP,
        scope_note="Service scope slice: `apps/billing`.",
        recon_summary="{}",
        cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )
    assert "bounds where you search, never what you may read" in prompt


def test_maintenance_audits_demand_reuse_and_subtractive_evidence() -> None:
    tech_debt = build_audit_prompt(
        category="tech-debt",
        skill_invocation=None,
        group=_GROUP,
        scope_note="",
        recon_summary="{}",
        cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )
    tests = build_audit_prompt(
        category="tests",
        skill_invocation=None,
        group=_GROUP,
        scope_note="",
        recon_summary="{}",
        cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )

    assert "existing repository code" in tech_debt
    assert "language standard library" in tech_debt
    assert "already-declared dependency" in tech_debt
    assert "mature new dependency" in tech_debt
    assert "comments that merely narrate self-evident code" in tech_debt
    assert "net reduction in production and test code" in tech_debt
    assert "Parameterizable matrices" in tests
    assert "Preserve separate tests" in tests


def test_finding_and_vet_schemas_require_stable_maintenance_metadata() -> None:
    for schema, collection in (
        (AUDIT_FINDINGS_SCHEMA, "findings"),
        (VET_SCHEMA, "verdicts"),
    ):
        item = schema["properties"][collection]["items"]
        assert {
            "maintenance_signals",
            "change_shape",
            "reuse_target",
        } <= set(item["required"])
        assert item["properties"]["change_shape"]["enum"] == [
            "delete",
            "reuse",
            "consolidate",
            "neutral",
            "additive",
            "unknown",
        ]


def test_vet_prompt_rejects_false_reuse_and_comment_cleanup() -> None:
    prompt = build_vet_prompt(findings=[], cwd=Path("/repo"))

    assert "re-open both implementations" in prompt
    assert "layer boundaries" in prompt
    assert "Similar-looking tests" in prompt
    assert "preserve rationale, invariants, security" in prompt


def _index_numbers_by_fingerprint(index_text: str) -> dict[str, int]:
    return {
        match.group(2): int(match.group(1))
        for match in re.finditer(
            r"\|\s+\[(\d{3})\]\([^)]+\)\s+<!-- fingerprint:([^ ]+) -->",
            index_text,
        )
    }


@pytest.fixture
def tmp_git_repo(improve_monorepo_target: Path) -> Path:
    return improve_monorepo_target



def _tiers_by_marker(calls: list[dict[str, Any]]) -> dict[str, set[tuple[str, str | None]]]:
    tiers: dict[str, set[tuple[str, str | None]]] = {}
    for call in calls:
        tiers.setdefault(str(call["marker"]), set()).add(
            (str(call["model"]), call["reasoning_effort"])
        )
    return tiers


def _git_status_porcelain(repo: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _untracked(repo: Path) -> list[str]:
    return subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _scan_trajectory_extra(run_root: Path, traj: Path, key: str) -> list[str]:
    values: list[str] = []
    for trajectory_file in list(run_root.rglob("*.json")) + (
        [traj] if traj.exists() else []
    ):
        try:
            payload = json.loads(trajectory_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        for step in payload.get("steps", []):
            value = (step.get("extra") or {}).get(key)
            if value:
                values.append(value)
    return values


def _improve_observable_texts(repo: Path) -> list[str]:
    return [
        path.read_text(encoding="utf-8")
        for root in (
            repo / ".daydream" / "improve",
            repo / ".daydream" / "runs",
            repo / "daydream_plans",
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]


def _raise_enumeration_failure(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
    raise RuntimeError("unparseable repository manifest")


def _forbidden_input(*_args: Any, **_kwargs: Any) -> str:
    raise AssertionError(
        "input() was called in non-interactive mode -- stdin must not be touched"
    )


def _force_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)


@pytest.mark.anyio
async def test_repo_scan_seeds_specialists_from_tracked_files(tmp_git_repo: Path) -> None:
    stub = ImproveStubBackend(tmp_git_repo)
    ctx = await repo_scan(stub, tmp_git_repo, max_files=500)
    assert any(c.name == "OpenAPI First" for c in ctx.conventions)
    prompt = stub.calls[0]["prompt"]
    assert stub.calls[0]["marker"] == "repo-scan"
    assert "api.py" in prompt
    assert stub.calls[0]["read_only"] is True


@pytest.mark.anyio
async def test_repo_scan_prompt_carries_no_diff_framing(tmp_git_repo: Path) -> None:
    """A repo-scoped scan has no change set, so it must not be described as one."""
    stub = ImproveStubBackend(tmp_git_repo)
    ctx = await repo_scan(stub, tmp_git_repo, max_files=500)
    prompt = stub.calls[0]["prompt"]
    assert "pattern-scanner" not in prompt
    assert "git diff" not in prompt
    assert "affected_files" not in prompt
    assert "relevant to the changes" not in prompt
    assert "no change set here" in prompt.lower()
    # A repo-scoped context has no affected files -- emitting the tracked tree
    # would relabel the whole repository as change-relevant downstream.
    assert ctx.affected_files == []
    assert "Affected Files" not in ctx.to_prompt_section()


@pytest.mark.anyio
async def test_repo_scan_sample_spans_the_tree_not_the_alphabetical_head(
    tmp_git_repo: Path,
) -> None:
    """A capped sample must still reach real source, not just dotfile dirs."""
    dotdir = tmp_git_repo / ".agents" / "skills"
    dotdir.mkdir(parents=True)
    for i in range(40):
        (dotdir / f"skill-{i:02d}.md").write_text(f"# skill {i}\n")
    git(tmp_git_repo, "add", ".")
    commit(tmp_git_repo, "add skills")

    stub = ImproveStubBackend(tmp_git_repo)
    await repo_scan(stub, tmp_git_repo, max_files=10)
    prompt = stub.calls[0]["prompt"]
    sample = prompt.split("<tracked_file_sample>")[1].split("</tracked_file_sample>")[0]
    sampled = [line[2:] for line in sample.strip().splitlines()]
    # Head-truncation would return .agents/ entries exclusively -- the source
    # tree sorts after them and never made it into the prompt.
    assert any(path.startswith("apps/") for path in sampled)
    assert "10 of 47 tracked files" in prompt


def test_sample_paths_spreads_across_a_capped_list() -> None:
    paths = [f"f{i:03d}" for i in range(100)]
    sample = _sample_paths(paths, 10)
    assert len(sample) == 10
    assert sample[0] == "f000"
    assert sample[-1] == "f090"
    assert _sample_paths(paths, 200) == paths
    assert _sample_paths(paths, 0) == []
    assert _sample_paths([], 10) == []


def test_registry_seeds_audit_slots_and_improve_prompts() -> None:
    r = build_registry()
    assert r.skill("audit:correctness:python") == "beagle-python:review-python"
    assert r.skill("audit:security:elixir") == "beagle-elixir:elixir-security-review"
    assert r.skill_if_registered("audit:dx") is None
    for name in ("audit", "vet", "plan-writer"):
        assert callable(r.prompt(name))


@pytest.mark.anyio
async def test_credentials_never_reach_improve_observables(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    secret = "OPENAI_API_KEY=sk-secret123456"
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.inject_credential = True

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 1
    console_output = capsys.readouterr().out
    artifact_texts = [
        path.read_text(encoding="utf-8")
        for root in (
            improve_monorepo_target / ".daydream" / "improve",
            improve_monorepo_target / ".daydream" / "runs",
            improve_monorepo_target / "daydream_plans",
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]
    observables = [console_output, *artifact_texts]
    assert all(secret not in observable for observable in observables)
    assert "[REDACTED" in "\n".join(observables)


@pytest.mark.anyio
async def test_improve_recon_writes_artifacts_and_never_mutates_source(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    before = _git_status_porcelain(improve_monorepo_target)
    code = await run(make_config(improve_monorepo_target, flow_name="improve"))
    assert code == 0
    dd = improve_monorepo_target / ".daydream" / "improve"
    assert (dd / "report.md").is_file()
    report_text = (dd / "report.md").read_text()
    assert "- **billing** — `apps/billing`" in report_text
    assert "- **catalog** — `apps/catalog`" in report_text
    trajectories = list(
        (improve_monorepo_target / ".daydream" / "runs").glob(
            "*/trajectory.json"
        )
    )
    assert len(trajectories) == 1
    trajectory = json.loads(trajectories[0].read_text())
    assert trajectory["steps"]
    assert all(
        step["extra"]["daydream_run_flow"] == "improve"
        for step in trajectory["steps"]
    )
    assert any(
        step["extra"]["daydream_phase"] == "recon"
        for step in trajectory["steps"]
    )
    run_marker = f"Daydream run: `{trajectory['session_id']}`"
    assert run_marker in (dd / "report.md").read_text()
    plans_dir = improve_monorepo_target / "daydream_plans"
    assert run_marker in (plans_dir / "README.md").read_text()
    assert all(
        run_marker in plan.read_text()
        for plan in plans_dir.glob("[0-9][0-9][0-9]-*.md")
    )
    plan_diagnostics = json.loads(
        (dd / "plan-write-diagnostics.json").read_text()
    )
    assert (
        plan_diagnostics["artifact_provenance"]["session_id"]
        == trajectory["session_id"]
    )
    assert all(call["read_only"] for call in stub.calls)
    assert _git_status_porcelain(improve_monorepo_target) == before


def _pin_stack_availability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin stack routing: an absent plugin registry means optimistic availability.

    Without this the detected stacks -- and so the partition-group count --
    depend on which Beagle plugins the developer happens to have installed.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config-absent"))


def _append_improve_config(target: Path, body: str) -> DaydreamFileConfig:
    """Append an ``[tool.daydream.improve]`` block, re-commit, and load it.

    The CLI is what reads the repo's config file (``cli.py`` calls
    ``load_file_config`` before building the RunConfig), so a runner-entry test
    loads it the same way instead of hand-building the dataclass.
    """
    pyproject = target / "pyproject.toml"
    pyproject.write_text(pyproject.read_text() + body)
    git(target, "add", "pyproject.toml")
    commit(target, "configure improve partition bounds")
    return load_file_config(target)


@pytest.mark.anyio
async def test_recon_prompt_names_audited_subtrees_for_per_service_commands(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    """One recon pass carries the audited roots, so it can return per-service commands."""
    _pin_stack_availability(monkeypatch, tmp_path)
    stub = install_improve_stub(
        monkeypatch, improve_scaled_monorepo_target, n_findings=0
    )

    code = await run(make_config(improve_scaled_monorepo_target, flow_name="improve"))

    assert code == 0
    recon_calls = [call for call in stub.calls if call["marker"] == "recon"]
    assert len(recon_calls) == 1
    prompt = recon_calls[0]["prompt"]
    assert "`apps/svc00`" in prompt and "`frontend`" in prompt
    assert "Return the per-subtree build, test, and lint commands" in prompt
    assert "`in-scope-paths`" in prompt
    from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY

    assert UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY in prompt
    assert prompt.index(UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY) < prompt.index("Existing repository scan:")
    # The exploration summary embedded below the recon header already opened with
    # the boundary; dedup must leave exactly one occurrence in the full prompt.
    assert prompt.count(UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY) == 1
    # Roots only: the recon prompt never inlines an individual tracked file.
    assert "apps/svc00/api.py" not in prompt
    # Service roots are printed anchored at the audit snapshot (the model's
    # cwd), never as bare relative paths a model could resolve against the
    # live target, so resolving a service root reads the snapshot.
    audit_cwd = recon_calls[0]["cwd"]
    assert f"- svc00: {(audit_cwd / 'apps/svc00').as_posix()}" in prompt
    assert f"- svc01: {(audit_cwd / 'apps/svc01').as_posix()}" in prompt


@pytest.mark.anyio
async def test_audit_fans_out_per_partition_group_on_scaled_monorepo(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    # Default bound (400): partitions = 12 services + frontend + residue, which
    # pack into exactly 3 stack-homogeneous groups (python / react / generic).
    stub = install_improve_stub(
        monkeypatch, improve_scaled_monorepo_target, n_findings=0
    )

    code = await run(make_config(improve_scaled_monorepo_target, flow_name="improve"))

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert len(audit_calls) == 3 * len(AUDIT_CATEGORIES)
    assert all("Relevant tracked files" not in call["prompt"] for call in audit_calls)
    # Roots only: no prompt names an individual tracked file.
    assert all("apps/svc00/api.py" not in call["prompt"] for call in audit_calls)
    assert {group_scope(call["prompt"])[0] for call in audit_calls} == {
        "group-01",
        "group-02",
        "group-03",
    }
    coverage = json.loads(improve_artifact(improve_scaled_monorepo_target, "coverage.json").read_text())
    assert len(coverage["groups"]) == 3
    assert {entry["name"] for entry in coverage["partitions"]} == {
        *(f"svc{index:02d}" for index in range(12)),
        "frontend",
        "residue",
    }
    assert coverage["not_audited"] == []


@pytest.mark.anyio
async def test_partition_bound_splits_oversized_trees_via_config(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    # max-partition-groups is raised out of the way so the bound alone is measured.
    file_config = _append_improve_config(
        improve_scaled_monorepo_target,
        "\n[tool.daydream.improve]\npartition-max-files = 5\nmax-partition-groups = 20\n",
    )
    stub = install_improve_stub(monkeypatch, improve_scaled_monorepo_target, n_findings=0)

    code = await run(
        make_config(
        improve_scaled_monorepo_target,
        flow_name="improve",
        file_config=file_config,
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    groups = {group_scope(call["prompt"])[0] for call in audit_calls}
    # frontend/src (12 files) splits into 3 partitions of 4; the 12 two-file
    # services pack 2-per-group; and the mixed generic/python residue is
    # routed to both specialists -> 11 groups.
    assert len(groups) == 11
    assert len(audit_calls) == 11 * len(AUDIT_CATEGORIES)
    for call in audit_calls:
        assert sum(group_file_counts(call["prompt"])) <= 5
    coverage = json.loads(improve_artifact(improve_scaled_monorepo_target, "coverage.json").read_text())
    assert {"frontend/src/alpha", "frontend/src/beta", "frontend/src/gamma"} <= {
        entry["name"] for entry in coverage["partitions"]
    }
    assert coverage["not_audited"] == []


@pytest.mark.anyio
async def test_group_ceiling_reports_full_and_partial_stack_coverage(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    file_config = _append_improve_config(
        improve_scaled_monorepo_target,
        "\n[tool.daydream.improve]\nmax-partition-groups = 1\n",
    )
    stub = install_improve_stub(monkeypatch, improve_scaled_monorepo_target, n_findings=0)

    code = await run(
        make_config(
        improve_scaled_monorepo_target,
        flow_name="improve",
        file_config=file_config,
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    # Only the largest group (the 24-file python service group) is audited.
    assert len(audit_calls) == len(AUDIT_CATEGORIES)
    assert {group_scope(call["prompt"])[0] for call in audit_calls} == {"group-01"}
    coverage = json.loads(improve_artifact(improve_scaled_monorepo_target, "coverage.json").read_text())
    not_audited = {entry["partition"]: entry for entry in coverage["not_audited"]}
    assert set(not_audited) == {"frontend"}
    assert not_audited["frontend"]["reason"] == "group-ceiling"
    assert not_audited["frontend"]["omitted_stacks"] == ["react"]
    partially = {entry["partition"]: entry for entry in coverage["partially_audited"]}
    assert set(partially) == {"residue"}
    assert partially["residue"]["reason"] == "group-ceiling"
    assert partially["residue"]["audited_stacks"] == ["python"]
    assert partially["residue"]["omitted_stacks"] == ["generic"]


@pytest.mark.anyio
async def test_quick_tier_audits_whole_repo_in_one_group(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    stub = install_improve_stub(monkeypatch, improve_scaled_monorepo_target, n_findings=0)

    code = await run(
        make_config(
        improve_scaled_monorepo_target,
        flow_name="improve",
        improve_effort="quick",
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert len(audit_calls) == 4  # quick also hunts tech-debt/code bloat
    for call in audit_calls:
        assert group_roots(call["prompt"]) == ["."]
    coverage = json.loads(improve_artifact(improve_scaled_monorepo_target, "coverage.json").read_text())
    assert coverage["not_audited"] == []
    assert [entry["name"] for entry in coverage["partitions"]] == ["repository"]


@pytest.mark.anyio
async def test_all_audit_assignments_failing_exits_nonzero(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.fail_categories = set(AUDIT_CATEGORIES)

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 1
    assert not improve_artifact(improve_monorepo_target, "report.md").exists()


@pytest.mark.anyio
async def test_small_repo_collapses_to_bounded_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    target = tmp_path / "single_package"
    target.mkdir()
    (target / "api.py").write_text("def handler():\n    return 1\n")
    (target / "pyproject.toml").write_text('[project]\nname = "single-package"\n')
    init_repo(target)
    git(target, "add", ".")
    commit(target, "initial")
    stub = install_improve_stub(monkeypatch, target, n_findings=0)

    code = await run(make_config(target, flow_name="improve"))

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert len(audit_calls) == len(AUDIT_CATEGORIES)
    assert all(group_roots(call["prompt"]) == ["."] for call in audit_calls)
    coverage = json.loads(improve_artifact(target, "coverage.json").read_text())
    assert [entry["name"] for entry in coverage["partitions"]] == ["residue"]
    assert len(coverage["groups"]) == 1


def _not_audited_section(report: str) -> str:
    return report.split("## What was not audited")[1].split("## ")[0]


@pytest.mark.anyio
async def test_report_names_unaudited_partitions_and_failed_groups(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    file_config = _append_improve_config(
        improve_scaled_monorepo_target,
        "\n[tool.daydream.improve]\nmax-partition-groups = 1\n",
    )
    stub = install_improve_stub(monkeypatch, improve_scaled_monorepo_target, n_findings=0)
    stub.fail_categories = {"docs"}

    code = await run(
        make_config(
        improve_scaled_monorepo_target,
        flow_name="improve",
        file_config=file_config,
        )
    )

    assert code == 0
    report = improve_artifact(improve_scaled_monorepo_target, "report.md").read_text()
    section = _not_audited_section(report)
    # Every ceiling-skipped partition is named with its root and reason.
    assert "**frontend**" in section and "group-ceiling" in section
    assert "`frontend/`" in section
    # residue's only retained group (group-01) failed its docs audit, so the
    # python stack is not counted as audited: the partition is reported as not
    # audited, naming both the ceiling-omitted and the failed stacks.
    assert "**residue**" in section
    assert "not audited" in section
    assert "omitted: generic, python" in section
    assert "partially audited" not in section
    failed = report.split("### Failed audit assignments")[1].split("## ")[0]
    # The failed assignment resolves to its group's roots, not just a key.
    assert "**docs / group-01**" in failed and "apps/svc00/" in failed
    coverage = json.loads(
        improve_artifact(improve_scaled_monorepo_target, "coverage.json").read_text()
    )
    assert {entry["reason"] for entry in coverage["not_audited"]} == {
        "group-ceiling",
        "group-failed",
    }
    assert coverage["partially_audited"] == []
    assert coverage["groups"] and coverage["partitions"]
    assert [entry["status"] for entry in coverage["groups"]] == ["failed"]
    assert "docs:group-01" in coverage["failed_assignments"]


@pytest.mark.anyio
async def test_clean_full_coverage_reports_nothing_skipped(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    install_improve_stub(monkeypatch, improve_monorepo_target, n_findings=0)

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    coverage = json.loads(improve_artifact(improve_monorepo_target, "coverage.json").read_text())
    assert coverage["not_audited"] == []
    assert {entry["status"] for entry in coverage["groups"]} == {"audited"}
    section = _not_audited_section(
        improve_artifact(improve_monorepo_target, "report.md").read_text()
    )
    assert "hotspot-weighted" in section  # the standard tier statement
    assert "All 4 partitions were audited." in section


@pytest.mark.anyio
async def test_top_offenders_name_directory_partitions_and_survive_artifacts(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    stub = install_improve_stub(
        monkeypatch, improve_monorepo_target, n_findings=1
    )
    # Each group's agent cites a file inside its own group, so the react group's
    # finding lands in the uncovered `web/` tree, which no service covers.
    stub.group_scoped_findings = True

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    audit = json.loads(improve_artifact(improve_monorepo_target, "audit-findings.json").read_text())
    assert {finding["partition"] for finding in audit["findings"]} == {
        "billing",
        "web",
        "residue",
    }
    vetted = json.loads(
        improve_artifact(improve_monorepo_target, "vetted-findings.json").read_text()
    )
    # The same pattern in three disjoint partitions aggregates into one finding
    # that names every location it was found in.
    assert len(vetted["findings"]) == 1
    assert set(vetted["findings"][0]["partitions"]) == {"billing", "web", "residue"}
    report = improve_artifact(improve_monorepo_target, "report.md").read_text()
    offenders = report.split("## Top offenders")[1].split("## ")[0]
    assert "**web**" in offenders and "**billing**" in offenders


@pytest.mark.anyio
async def test_vet_batches_are_bounded_and_parallel(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    # One partition group and one category keep the batch math exact: 45
    # candidates -> 20 + 20 + 5.
    file_config = _append_improve_config(
        improve_monorepo_target,
        "\n[tool.daydream.improve]\nmax-partition-groups = 1\n",
    )
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.findings_per_category = 45
    stub.vet_reject_titles = {"Security finding 45"}

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_focus="security",
        file_config=file_config,
        )
    )

    assert code == 0
    vet_calls = [call for call in stub.calls if call["marker"] == "vet"]
    assert len(vet_calls) == 3
    for call in vet_calls:
        payload = json.loads(call["prompt"].split("```json\n")[1].split("```")[0])
        assert len(payload) <= VET_BATCH_MAX_FINDINGS
    vetted = json.loads(improve_artifact(improve_monorepo_target, "vetted-findings.json").read_text())
    members = [member for package in vetted["findings"] for member in package.get("members", [package])]
    titles = {finding["title"] for finding in members}
    # Verdicts from every batch apply: the last batch's rejection is honored
    # and the other 44 survive inside nine manageable work packages.
    assert len(members) == 44
    assert len(vetted["findings"]) == 9
    assert "Security finding 45" not in titles
    assert "Security finding 01" in titles and "Security finding 44" in titles
    rejected = json.loads((improve_monorepo_target / "daydream_plans" / "rejected.json").read_text())
    assert any(entry["title"] == "Security finding 45" for entry in rejected["rejected"])


@pytest.mark.anyio
async def test_vet_batch_failure_fails_closed_per_batch(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    file_config = _append_improve_config(
        improve_monorepo_target,
        "\n[tool.daydream.improve]\nmax-partition-groups = 1\n",
    )
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.findings_per_category = 45
    stub.fail_vet_titles = {"Security finding 41"}  # the third batch's agent raises

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_focus="security",
        file_config=file_config,
        )
    )

    assert code == 0
    vetted = json.loads(improve_artifact(improve_monorepo_target, "vetted-findings.json").read_text())
    members = [member for package in vetted["findings"] for member in package.get("members", [package])]
    titles = {finding["title"] for finding in members}
    # Only the failed batch's five findings drop; the other two batches keep theirs.
    assert len(members) == 40
    assert len(vetted["findings"]) == 8
    assert "Security finding 41" not in titles
    assert "Security finding 01" in titles and "Security finding 40" in titles


@pytest.mark.anyio
async def test_run_with_no_findings_writes_report_and_empty_plan_diagnostics(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=0,
    )

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    assert [call for call in stub.calls if call["marker"] == "audit"]
    assert not [call for call in stub.calls if call["marker"] == "plan-writer"]
    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert diagnostics["attempts"] == []
    assert "Plans written: 0" in improve_artifact(
        improve_monorepo_target,
        "report.md",
    ).read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_improve_continues_audit_and_planning_when_recon_has_no_valid_commands(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.all_recon_commands_invalid = True

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert [call for call in stub.calls if call["marker"] == "audit"]
    assert [call for call in stub.calls if call["marker"] == "plan-writer"]
    recon_path = improve_artifact(improve_monorepo_target, "recon.json")
    recon_text = recon_path.read_text()
    recon = json.loads(recon_text)
    assert recon["commands"] == []
    assert recon["command_rejections"] == [
        {
            "code": "RECON_APPLICABILITY_INVALID",
            "pointer": f"/commands/{index}/applicability/scope/kind",
        }
        for index in range(2)
    ]
    # No rejected candidate's content survives into the persisted artifact.
    assert "verbatim_excerpt" not in recon_text
    assert "uv run pytest" not in recon_text
    report = improve_artifact(improve_monorepo_target, "report.md")
    plans = sorted(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    plan_diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert code == 0
    assert report.is_file()
    assert plans
    assert all("**Command**" not in plan.read_text() for plan in plans)
    assert any(
        attempt["disposition"] == "success"
        for attempt in plan_diagnostics["attempts"]
    )
    assert all(call["read_only"] for call in stub.calls)
    trajectories = list(
        (improve_monorepo_target / ".daydream" / "runs").glob(
            "*/trajectory.json"
        )
    )
    trajectory = json.loads(trajectories[0].read_text())
    validation_events = [
        event
        for event in trajectory["extra"]["phase_events"]
        if event["event"] == "command_validation"
    ]
    assert validation_events[0]["metadata"]["counts"] == {
        "total_candidates": 2,
        "accepted": 0,
        "rejected": 2,
    }
    assert validation_events[0]["metadata"]["reasons"] == {
        "RECON_APPLICABILITY_INVALID": 2
    }
    assert recon["artifact_provenance"]["session_id"] == trajectory["session_id"]


@pytest.mark.anyio
async def test_makefile_and_manifest_gate_plans_when_the_model_cites_nothing(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """The host enumerates Make/manifest commands the model is told to skip.

    Recon is read-only, so it can never run a command to confirm one exists.
    A repository whose only test gate lives in a Makefile or package.json --
    written nowhere in prose for a model to cite -- must still hand the
    executor a real verification command instead of the manual fallback.
    """
    (improve_monorepo_target / "Makefile").write_text(
        "check: ## Run the full gate\n\tuv run pytest\n",
        encoding="utf-8",
    )
    (improve_monorepo_target / "web" / "package.json").write_text(
        json.dumps({"name": "web", "scripts": {"test": "vitest run"}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (improve_monorepo_target / "web" / "pnpm-lock.yaml").write_text(
        "lockfileVersion: '9.0'\n",
        encoding="utf-8",
    )
    git(
        improve_monorepo_target,
        "add",
        "Makefile",
        "web/package.json",
        "web/pnpm-lock.yaml",
    )
    commit(improve_monorepo_target, "add build tooling")
    # Neither invocation is written verbatim anywhere, so `literal-command`
    # evidence for them cannot exist: only host enumeration can supply them.
    for tracked in improve_monorepo_target.rglob("*"):
        if not tracked.is_file() or ".git" in tracked.parts:
            continue
        source = tracked.read_text(encoding="utf-8", errors="ignore")
        assert "make check" not in source and "pnpm test" not in source

    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.recon_output_override = {
        "languages": ["python", "typescript"],
        "commands": [],
        "conventions": ["OpenAPI First"],
        "intent_docs": ["README.md"],
    }
    stub.plan_gate_on_first_menu_id = True

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    recon = json.loads(
        improve_artifact(improve_monorepo_target, "recon.json").read_text(encoding="utf-8")
    )
    by_command = {command["command"]: command for command in recon["commands"]}
    assert "make check" in by_command and "pnpm test" in by_command
    assert by_command["make check"]["id"] == "make-check"
    assert by_command["make check"]["working_directory"] == "."
    assert by_command["make check"]["evidence"] == {
        "kind": "host-derived",
        "source_path": "Makefile",
        "line_anchor": {"start_line": 1, "end_line": 1},
        "verbatim_excerpt": "check: ## Run the full gate",
    }
    assert by_command["pnpm test"]["working_directory"] == "web"
    assert by_command["pnpm test"]["applicability"]["scope"] == {
        "kind": "in-scope-paths",
        "paths": ["web"],
    }

    plans = sorted(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    )
    assert plans
    texts = [plan.read_text(encoding="utf-8") for plan in plans]
    assert all("**Command**: `make check`" in text for text in texts)
    assert all(
        "No repository command was verified during planning" not in text
        for text in texts
    )
    assert all(call["read_only"] for call in stub.calls)
    assert _git_status_porcelain(improve_monorepo_target) == ""


def _dedup_test_command(
    *,
    command_id: str,
    purpose: str,
    working_directory: str,
    scope: dict[str, Any],
    rationale: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build one recon command record for the absolute-wd dedup fixture.

    The host-enumerated and model-cited sides of the dedup share the same
    command, expected-success block, and applicability skeleton; only the id,
    working-directory spelling, scope, rationale, and evidence differ.
    """
    return {
        "id": command_id,
        "purpose": purpose,
        "command": "uv run pytest",
        "working_directory": working_directory,
        "expected_success": {
            "exit_code": 0,
            "observable_result": "exit 0 and the tests pass",
        },
        "applicability": {
            "scope": scope,
            "preconditions": [],
            "rationale": rationale,
        },
        "evidence": evidence,
    }


@pytest.mark.anyio
async def test_host_enumeration_dedups_absolute_model_wd(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A model-cited command whose working_directory is spelled absolutely is
    deduped against the host record for the same directory (relative spelling):
    exactly one command record, no rejection noise (issue #654).

    Fixture contract (mirrors tests/test_command_contract.py): the service dir
    and the test-command anchor are derived from the fixture's actual content
    via the shared fixture-contract helpers rather than hardcoded, so a future
    fixture edit surfaces as an attributable error instead of a silent meaning
    shift.
    """
    # Derive a real service dir (one containing a pyproject.toml) from the
    # fixture, and the root test-command declaration line, via the shared
    # fixture-contract helpers.
    service = improve_fixture_service(improve_monorepo_target / "apps")
    rel = f"apps/{service}"
    anchor_line = improve_fixture_test_command_anchor(
        improve_monorepo_target / "pyproject.toml"
    )
    monkeypatch.setattr(
        "daydream.improve.orchestrator.enumerate_repository_commands",
        lambda repo, *, directories=(".",), reserved_ids=(): [
            _dedup_test_command(
                command_id="make-check",
                purpose="Run the repository test suite",
                working_directory=rel,
                scope={"kind": "in-scope-paths", "paths": [rel]},
                rationale=f"The {service} service declares the test command.",
                evidence={
                    "kind": "host-derived",
                    "source_path": f"{rel}/pyproject.toml",
                    "line_anchor": {"start_line": 1, "end_line": 1},
                    "verbatim_excerpt": "[project]",
                },
            )
        ],
    )
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.recon_output_override = {
        "languages": ["python"],
        "commands": [
            _dedup_test_command(
                command_id="model-check",
                purpose="Run the repository test suite",
                # Absolute spelling of the SAME directory the host enumerates
                # relative.
                working_directory=f"{improve_monorepo_target}/{rel}",
                scope={"kind": "whole-repository"},
                rationale="The root configuration declares the test command.",
                evidence={
                    "kind": "literal-command",
                    "source_path": "pyproject.toml",
                    "line_anchor": {"start_line": anchor_line, "end_line": anchor_line},
                    "verbatim_excerpt": None,
                },
            )
        ],
        "conventions": ["OpenAPI First"],
        "intent_docs": ["README.md"],
    }
    stub.plan_gate_on_first_menu_id = True

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    recon = _load_improve_json(improve_monorepo_target, "recon.json")
    # The host command was deduped against the absolute model wd: no
    # re-admitted make-check record, no rejection noise.
    assert [command["id"] for command in recon["commands"]] == ["model-check"]
    assert recon["command_rejections"] == []


@pytest.mark.anyio
async def test_host_enumeration_failure_is_visible_and_keeps_model_commands(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    """A broken enumerator degrades to the model's commands, never silently."""
    monkeypatch.setattr(
        "daydream.improve.orchestrator.enumerate_repository_commands",
        _raise_enumeration_failure,
    )
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    recon = json.loads(
        improve_artifact(improve_monorepo_target, "recon.json").read_text(encoding="utf-8")
    )
    assert [command["id"] for command in recon["commands"]] == [
        "test-suite",
        "git-diff",
    ]
    assert recon["command_rejections"] == [
        {"code": "HOST_COMMAND_ENUMERATION_FAILED", "pointer": "/host_commands"}
    ]
    assert "Host command enumeration failed" in capsys.readouterr().out
    assert [call for call in stub.calls if call["marker"] == "plan-writer"]


@pytest.mark.anyio
async def test_unrelated_recon_container_error_preserves_valid_commands(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.recon_languages_override = {"secret-model-prose": "must not persist"}

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    assert [call for call in stub.calls if call["marker"] == "audit"]
    recon_text = improve_artifact(improve_monorepo_target, "recon.json").read_text()
    # The malformed `languages` value is replaced wholesale, never persisted.
    assert "secret-model-prose" not in recon_text
    recon = json.loads(recon_text)
    assert recon["languages"] == []
    assert len(recon["commands"]) == 2
    assert all(
        command["evidence"]["verbatim_excerpt"]
        in {
            'test-command = "uv run pytest"',
            'scope-command = "git diff --exit-code"',
        }
        for command in recon["commands"]
    )
    assert recon["command_rejections"] == []


@pytest.mark.anyio
async def test_non_array_commands_preserve_diagnostics_and_continue_audit(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    secret = "OPENAI_API_KEY=sk-secret123456"
    model_prose = "private arbitrary model explanation"
    rejected_command = "uv run pytest --private-selection"
    stub.recon_output_override = {
        "languages": [model_prose],
        "commands": {
            "command": rejected_command,
            "verbatim_excerpt": secret,
        },
        "conventions": [model_prose],
        "intent_docs": [model_prose],
    }

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 1
    assert [call for call in stub.calls if call["marker"] == "audit"]
    assert [call for call in stub.calls if call["marker"] == "plan-writer"]
    assert improve_artifact(improve_monorepo_target, "report.md").is_file()
    recon_text = improve_artifact(improve_monorepo_target, "recon.json").read_text()
    recon = json.loads(recon_text)
    assert recon["commands"] == []
    assert recon["command_rejections"] == [
        {"code": "RECON_COMMANDS_INVALID", "pointer": "/commands"}
    ]
    # `model_prose` is legitimate recon prose and stays; the rejected
    # candidate's own content must never reach the persisted artifact.
    for private_value in (secret, rejected_command, "verbatim_excerpt"):
        assert private_value not in recon_text

    console_output = capsys.readouterr().out
    for private_value in (secret, model_prose, rejected_command):
        assert private_value not in console_output


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("effort", "focus", "expected_categories"),
    [
        pytest.param("standard", None, sorted(AUDIT_CATEGORIES), id="standard-effort"),
        pytest.param(
            "quick",
            None,
            ["correctness", "security", "tech-debt", "tests"],
            id="quick-effort",
        ),
        pytest.param("standard", "security", ["security"], id="focus-security"),
    ],
)
async def test_effort_and_focus_select_the_audited_categories_read_only(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
    focus: str | None,
    expected_categories: list[str],
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_effort=effort,
        improve_focus=focus,
    )
    )
    audited = json.loads(improve_artifact(improve_monorepo_target, "audit-findings.json").read_text())
    assert sorted(audited["categories_run"]) == expected_categories
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert audit_calls and all(call["read_only"] for call in audit_calls)


@pytest.mark.anyio
async def test_repo_with_no_test_files_still_receives_a_plan(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A repository with zero tests must still be plannable.

    The audit playbook tells the auditor that "if there is no one-command way
    to know the codebase works, that is a prerequisite finding" — so the plan
    writer has to be able to author the plan that fixes it, in a repository
    with no existing test to point at as an exemplar.
    """
    assert not list(improve_monorepo_target.rglob("test_*.py"))
    assert not list(improve_monorepo_target.rglob("*_test.py"))
    stub = install_improve_stub(
        monkeypatch, improve_monorepo_target, n_findings=1
    )
    stub.plan_no_test_exemplars = True

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    plans = sorted(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )

    assert code == 0
    assert plans
    body = plans[0].read_text(encoding="utf-8")
    assert "## Test plan" in body
    assert "### Named cases" in body
    assert "test_service_name_preserves_contract" in body
    assert "no existing test" in body
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["success"]


@pytest.mark.anyio
async def test_pi_improve_retains_valid_commands_and_avoids_provider_overload(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "daydream.agent.prompt_user",
        lambda *args, **kwargs: "1-5",
    )
    backend = ProductionPathBackend(improve_monorepo_target)
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
    )

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        non_interactive=False,
        )
    )

    recon = json.loads(improve_artifact(improve_monorepo_target, "recon.json").read_text(encoding="utf-8"))
    plan_files = sorted((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    console_output = capsys.readouterr().out
    observables = [
        console_output,
        *_improve_observable_texts(improve_monorepo_target),
    ]

    assert code == 0
    assert recon["commands"]
    assert recon["command_rejections"] == [
        {
            "code": "RECON_MALFORMED_COMMAND",
            "pointer": "/commands/42/command",
        }
    ]
    assert plan_files
    assert all(backend.recon_secret not in text for text in observables)
    assert all(backend.planner_secret not in text for text in observables)


@pytest.mark.anyio
async def test_pi_improve_partial_failure_is_successful_and_safe(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "daydream.agent.prompt_user",
        lambda *args, **kwargs: "1-5",
    )
    backend = ProductionPathBackend(
        improve_monorepo_target,
        failed_title="Production finding 03",
    )
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
    )

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        non_interactive=False,
        )
    )

    plan_files = sorted((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    diagnostics_text = improve_artifact(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    ).read_text(encoding="utf-8")
    diagnostics = json.loads(diagnostics_text)
    failed = [attempt for attempt in diagnostics["attempts"] if attempt["disposition"] == "blocked"]
    console_output = capsys.readouterr().out
    observables = [
        console_output,
        *_improve_observable_texts(improve_monorepo_target),
    ]

    assert code == 0
    assert len(plan_files) == 1
    assert len(failed) == 1
    # The failed member is represented by its aggregate package title.
    assert failed[0]["finding"]["title"] == "Production finding 01"
    assert failed[0]["errors"] == [{"code": "PROCESS_EXIT", "pointer": "/"}]
    assert "Plan blocked for Production finding 01: PROCESS_EXIT at /." in console_output
    report = improve_artifact(improve_monorepo_target, "report.md").read_text(encoding="utf-8")
    assert "Plans written: 1" in report
    assert "Plans blocked by plan-writing failure: 1" in report
    assert all(backend.recon_secret not in text for text in observables)
    assert all(backend.planner_secret not in text for text in observables)


@pytest.mark.anyio
async def test_branch_focus_scopes_audit_to_merge_base_diff_and_tags_provenance(
    improve_branch_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_branch_target)
    await run(make_config(improve_branch_target, flow_name="improve", improve_focus="branch"))
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert all(
        "apps/billing/api.py" in call["prompt"] for call in audit_calls
    )
    # Branch focus bypasses partitioning: one synthetic group over the changed
    # files, so the fan-out stays one serial agent per category.
    assert len(audit_calls) == len(AUDIT_CATEGORIES)
    assert all(group_roots(call["prompt"]) == ["."] for call in audit_calls)
    coverage = json.loads(improve_artifact(improve_branch_target, "coverage.json").read_text())
    assert [entry["name"] for entry in coverage["partitions"]] == ["branch"]
    assert coverage["not_audited"] == []
    vetted = json.loads(
        improve_artifact(improve_branch_target, "vetted-findings.json").read_text()
    )
    assert {finding["provenance"] for finding in vetted["findings"]} <= {
        "introduced",
        "inherited",
    }


@pytest.mark.anyio
async def test_branch_focus_with_scope_excludes_out_of_scope_service_diff(
    improve_branch_two_services_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_branch_two_services_target)
    code = await run(
        make_config(
        improve_branch_two_services_target,
        flow_name="improve",
        improve_focus="branch",
        improve_scope="apps/billing",
        )
    )
    assert code == 0
    # Isolate the embedded ```diff fenced block so the assertion targets the
    # branch diff itself, not the whole-repo recon context that legitimately
    # names other services.
    diff_blocks = [
        prompt.split("```diff\n", 1)[1].split("\n```", 1)[0]
        for call in stub.calls
        if call["marker"] in ("audit", "vet")
        for prompt in [call["prompt"]]
        if "```diff\n" in prompt
    ]
    assert diff_blocks
    assert all("apps/billing/api.py" in block for block in diff_blocks)
    assert all("apps/catalog/api.py" not in block for block in diff_blocks)
    assert all("catalog-v2" not in block for block in diff_blocks)


@pytest.mark.anyio
async def test_branch_focus_on_base_branch_reports_and_exits_cleanly(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_focus="branch",
        )
    )
    assert code == 1


@pytest.mark.anyio
async def test_failed_category_is_reported_not_silently_dropped(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.fail_categories = {"performance"}
    code = await run(make_config(improve_monorepo_target, flow_name="improve"))
    assert code == 0
    report = improve_artifact(improve_monorepo_target, "report.md").read_text()
    assert "performance" in report.lower()
    assert "not audited" in report.lower()


@pytest.mark.anyio
async def test_vet_rejects_unconfirmed_finding_with_reason_and_persists(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.vet_reject_titles = {"Phantom N+1"}
    await run(make_config(improve_monorepo_target, flow_name="improve"))

    vetted = json.loads(
        improve_artifact(improve_monorepo_target, "vetted-findings.json").read_text()
    )
    assert all(
        finding["title"] != "Phantom N+1" for finding in vetted["findings"]
    )
    rejected = json.loads(
        (
            improve_monorepo_target / "daydream_plans" / "rejected.json"
        ).read_text()
    )
    assert rejected["rejected"][0]["title"] == "Phantom N+1"
    assert rejected["rejected"][0]["reason"]


@pytest.mark.anyio
async def test_previously_rejected_finding_is_not_revetted_or_rereported(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.vet_reject_titles = {"Phantom N+1"}
    config =make_config(improve_monorepo_target, flow_name="improve")
    await run(config)

    stub.calls.clear()
    await run(config)

    vet_calls = [call for call in stub.calls if call["marker"] == "vet"]
    assert all("Phantom N+1" not in call["prompt"] for call in vet_calls)
    report = improve_artifact(improve_monorepo_target, "report.md").read_text()
    assert "previously rejected" in report.lower()


@pytest.mark.anyio
async def test_non_interactive_run_selects_top_findings_and_writes_plans(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    monkeypatch.setattr("builtins.input", _forbidden_input)
    install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=8,
    )
    code = await run(make_config(improve_monorepo_target, flow_name="improve"))
    selected = json.loads(
        improve_artifact(improve_monorepo_target, "selected.json").read_text()
    )
    assert len(selected["selected"]) == 5
    assert selected["mode"] == "non-interactive-default"
    plans_dir = improve_monorepo_target / "daydream_plans"
    plan_files = sorted(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert code == 0
    assert 1 <= len(plan_files) <= 5
    index = (plans_dir / "README.md").read_text()
    assert "non-interactive default" in index.lower()
    assert head_sha(improve_monorepo_target)[:7] in plan_files[0].read_text()


# Selection order is the leverage order of the two audited findings; the plan
# number each one gets is claimed from that order before any writer runs.
_PLAN_FILE_BY_TITLE = {
    "Security finding": "001-security-finding.md",
    "high-leverage-title": "002-high-leverage-title.md",
}


@pytest.mark.anyio
@pytest.mark.parametrize("slow_title", sorted(_PLAN_FILE_BY_TITLE))
async def test_finished_plan_is_on_disk_while_a_slower_writer_still_runs(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    slow_title: str,
    make_config: MakeConfig,
) -> None:
    """Each plan lands as its writer completes, numbered by selection order.

    Parametrizing which writer finishes last also proves the numbering: both
    completion orders produce the same title-to-number mapping.
    """
    backend = IncrementalPlanBackend(
        improve_monorepo_target,
        slow_title=slow_title,
    )
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
    )

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    plans_dir = improve_monorepo_target / "daydream_plans"
    fast_title = next(
        title for title in _PLAN_FILE_BY_TITLE if title != slow_title
    )
    assert code == 0
    assert backend.observed_while_slow_writer_ran == [
        _PLAN_FILE_BY_TITLE[fast_title]
    ]
    # An interrupt at that moment would leave a resumable index, not an
    # orphaned plan file.
    assert (
        f"({_PLAN_FILE_BY_TITLE[fast_title]})"
        in backend.observed_index_while_slow_writer_ran
    )
    assert sorted(
        path.name for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
    ) == sorted(_PLAN_FILE_BY_TITLE.values())
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    for filename in _PLAN_FILE_BY_TITLE.values():
        assert f"({filename})" in index


@pytest.mark.anyio
async def test_plan_numbers_track_selection_order_when_writers_finish_out_of_order(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Three writers, completion order rotated away from selection order.

    The first-selected finding's writer returns last, so if numbers were
    claimed as writers finished it would end up with the highest number. It
    keeps 001 because numbers are reserved before any writer runs.
    """
    backend = OutOfOrderPlanBackend(improve_monorepo_target, n_findings=3)
    backend.vet_reject_titles = {"Phantom N+1"}
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
    )

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    plans_dir = improve_monorepo_target / "daydream_plans"
    selected = json.loads(
        improve_artifact(improve_monorepo_target, "selected.json").read_text()
    )["selected"]
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert code == 0
    assert len(selected) == 3
    assert backend.completion_order == [1, 2, 0]
    assert _index_numbers_by_fingerprint(index) == {
        fingerprint: rank + 1 for rank, fingerprint in enumerate(selected)
    }
    assert len(list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))) == 3
    assert index.count("| TODO |") == 3
    # The durable record is the sidecar; the README above is rendered from it.
    sidecar = json.loads(
        (plans_dir / PLAN_INDEX_FILENAME).read_text(encoding="utf-8")
    )
    assert sidecar["schema_version"] == 1
    assert sidecar["artifact_type"] == "daydream.plan-index"
    assert [
        (entry["number"], entry["fingerprint"], entry["status"])
        for entry in sidecar["plans"]
    ] == [
        (rank + 1, fingerprint, "TODO")
        for rank, fingerprint in enumerate(selected)
    ]
    assert all(
        (plans_dir / f"{entry['number']:03d}-{entry['slug']}.md").is_file()
        and entry["planned_at"] == head_sha(improve_monorepo_target)
        for entry in sidecar["plans"]
    )


@pytest.mark.anyio
async def test_plan_writer_crash_leaves_the_finished_plan_on_disk(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    backend = IncrementalPlanBackend(
        improve_monorepo_target,
        slow_title="Security finding",
        crash=True,
    )
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
    )

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    plans_dir = improve_monorepo_target / "daydream_plans"
    assert code == 0
    assert backend.observed_while_slow_writer_ran == [
        _PLAN_FILE_BY_TITLE["high-leverage-title"]
    ]
    assert [
        path.name for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
    ] == [_PLAN_FILE_BY_TITLE["high-leverage-title"]]
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert "BLOCKED (PLAN_WRITER_FAILED: PROCESS_EXIT)" in index
    assert "002" in index


@pytest.mark.anyio
async def test_all_legacy_plan_results_block_and_return_failure(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.return_legacy_plan = True

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    plans_dir = improve_monorepo_target / "daydream_plans"
    assert code == 1
    assert stub.plan_writer_calls == 2
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert "BLOCKED (PLAN_VALIDATION_FAILED: " in (
        plans_dir / "README.md"
    ).read_text()
    report = improve_artifact(improve_monorepo_target, "report.md").read_text()
    assert "Plans written: 0" in report
    diagnostics_text = improve_artifact(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    ).read_text(encoding="utf-8")
    assert "AUTHOR_SCHEMA_INVALID" in diagnostics_text
    assert "Make the change." not in diagnostics_text


@pytest.mark.anyio
async def test_real_improve_flow_plans_from_live_dirty_source_without_running_candidates(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    source_path = improve_monorepo_target / "apps/billing/api.py"
    user_edit = (
        "def service_name():\n"
        '    return "billing-from-live-working-tree"\n'
        "# verify manually: touch candidate-command-ran\n"
    )
    source_path.write_text(user_edit, encoding="utf-8")
    dirty_status = _git_status_porcelain(improve_monorepo_target)
    plans_dir = improve_monorepo_target / "daydream_plans"
    plans_dir.mkdir()
    (plans_dir / "001-billing-foundation.md").write_text(
        "# Plan 001: Establish billing foundation\n",
        encoding="utf-8",
    )
    (plans_dir / "README.md").write_text(
        "# Implementation Plans\n\n"
        "## Execution order & status\n\n"
        "| Plan | Title | Priority | Effort | Status |\n"
        "|------|-------|----------|--------|--------|\n"
        "| [001](001-billing-foundation.md) "
        "<!-- fingerprint:trusted-existing-foundation --> | "
        "Establish billing foundation | P1 | S | TODO |\n",
        encoding="utf-8",
    )
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.recon_commands_extra = [
        {
            "id": "manual-sentinel",
            "purpose": "Manually verify candidate commands are never auto-executed",
            "command": "touch candidate-command-ran",
            "working_directory": ".",
            "expected_success": {
                "exit_code": 0,
                "observable_result": "exit 0 and the manual sentinel exists",
            },
            "applicability": {
                "scope": {"kind": "whole-repository"},
                "preconditions": [],
                "rationale": "The live source records this manual-only command.",
            },
            "evidence": {
                "kind": "literal-command",
                "source_path": "apps/billing/api.py",
                "line_anchor": {"start_line": 3, "end_line": 3},
                "verbatim_excerpt": "touch candidate-command-ran",
            },
        }
    ]

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    generated = sorted(plans_dir.glob("[0-9][0-9][0-9]-high-leverage-title.md"))
    assert len(generated) == 1
    plan = generated[0].read_text(encoding="utf-8")
    assert (
        'def service_name():\n    return "billing-from-live-working-tree"'
        in plan
    )
    assert "uv run pytest apps/billing/test_api.py -q" in plan
    assert source_path.read_text(encoding="utf-8") == user_edit
    assert _git_status_porcelain(improve_monorepo_target) == dirty_status
    assert not (improve_monorepo_target / "candidate-command-ran").exists()

    report = improve_artifact(improve_monorepo_target, "report.md").read_text(
        encoding="utf-8"
    )
    assert "high-leverage-title" in report
    recon = json.loads(
        improve_artifact(improve_monorepo_target, "recon.json").read_text(encoding="utf-8")
    )
    sentinel = next(
        command
        for command in recon["commands"]
        if command["id"] == "manual-sentinel"
    )
    assert sentinel["evidence"]["verbatim_excerpt"] == (
        "# verify manually: touch candidate-command-ran"
    )

    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert "| high-leverage-title | P1 | S | TODO |" in index

    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert diagnostics["artifact_type"] == "daydream.plan-write-diagnostics"
    assert any(
        attempt["disposition"] == "success"
        and attempt["artifact"]["path"] == generated[0].name
        for attempt in diagnostics["attempts"]
    )


@pytest.mark.anyio
async def test_schema_invalid_planner_metadata_never_reaches_observables(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.return_secret_invalid_enum = True

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 1
    index = (
        improve_monorepo_target / "daydream_plans/README.md"
    ).read_text(encoding="utf-8")
    sidecar = (
        improve_monorepo_target / "daydream_plans" / PLAN_INDEX_FILENAME
    ).read_text(encoding="utf-8")
    report = improve_artifact(improve_monorepo_target, "report.md").read_text(encoding="utf-8")
    diagnostics = improve_artifact(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    ).read_text(encoding="utf-8")
    console_output = capsys.readouterr().out
    for observable in (index, sidecar, report, console_output, diagnostics):
        assert "PRIVATE_SCHEMA_SECRET" not in observable
        assert "TOKEN=" not in observable
        assert "SCHEMA_INVALID" in observable


@pytest.mark.anyio
async def test_interactive_selection_honors_user_choice(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "2")
    install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=8,
    )
    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        non_interactive=False,
    )
    )
    assert code == 0
    selected = json.loads(improve_artifact(improve_monorepo_target, "selected.json").read_text())
    assert len(selected["selected"]) == 1
    assert selected["mode"] == "interactive"


@pytest.mark.anyio
async def test_report_orders_by_leverage_without_non_actionable_direction_section(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=9,
    )
    code = await run(make_config(improve_monorepo_target, flow_name="improve"))
    assert code == 0
    report = improve_artifact(improve_monorepo_target, "report.md").read_text()
    assert report.index("high-leverage-title") < report.index("low-leverage-title")
    assert "## Direction" not in report
    assert "## Cleanup pressure" in report
    assert "not audited" in report.lower()


@pytest.mark.anyio
async def test_scope_slices_search_but_report_names_the_unaudited_rest(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_scope="apps/billing",
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert all("apps/billing" in call["prompt"] for call in audit_calls)
    # A slice bounds where the audit searches, never what it may read: a
    # cross-service finding must stay reachable (spec.md monorepo requirement).
    assert all("bounds where you search, never what you may read" in call["prompt"] for call in audit_calls)
    assert any("bounds where the audit searches" in call["prompt"].lower() for call in audit_calls)
    report = improve_artifact(improve_monorepo_target, "report.md").read_text()
    assert "catalog" in report and "not audited" in report.lower()


@pytest.mark.anyio
async def test_group_scope_expands_named_service_group_to_all_members(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_scope="core",
            file_config=DaydreamFileConfig(improve_service_groups={"core": ["apps/billing", "apps/catalog"]}),
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    audited_paths = {call["prompt"] for call in audit_calls}
    assert any("apps/billing" in p for p in audited_paths)
    assert any("apps/catalog" in p for p in audited_paths)
    report = improve_artifact(improve_monorepo_target, "report.md").read_text()
    # the group covered every detected service, so the unaudited list is empty
    assert "No other detected service directories." in report


@pytest.mark.anyio
async def test_plan_subverb_skips_audit_and_writes_single_plan(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    assert code == 0
    assert not improve_artifact(improve_monorepo_target, "audit-findings.json").exists()
    plans = list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert len(plans) == 1
    assert "rate limiting" in plans[0].read_text().lower()


@pytest.mark.anyio
async def test_plan_subverb_repairs_schema_invalid_plan_once(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.return_secret_invalid_enum_once = True

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plan_calls = [call for call in stub.calls if call["marker"] == "plan-writer"]
    assert code == 0
    assert len(plan_calls) == 2
    repair_prompt = plan_calls[1]["prompt"]
    assert "PRIVATE_SCHEMA_SECRET" not in repair_prompt
    assert "TOKEN=" not in repair_prompt
    assert all(
        call["output_schema"] == PLAN_AUTHOR_SCHEMA and call["read_only"] is True and call["persist_session"] is False
        for call in plan_calls
    )
    plans = list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert len(plans) == 1
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (
            improve_monorepo_target / ".daydream",
            improve_monorepo_target / "daydream_plans",
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    )
    assert "PRIVATE_SCHEMA_SECRET" not in persisted
    assert "TOKEN=" not in persisted


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("stub_attr", "stub_value", "error_code", "error_pointer"),
    [
        pytest.param(
            "return_secret_invalid_enum",
            True,
            "AUTHOR_SCHEMA_INVALID",
            "/steps/0/changes/0/operation",
            id="schema-invalid-on-every-attempt",
        ),
        pytest.param(
            "plan_bad_recon_id_attempts",
            99,
            "RECON_COMMAND_UNKNOWN",
            None,
            id="unknown-recon-command-on-every-attempt",
        ),
    ],
)
async def test_persistent_authoring_failure_blocks_after_one_repair(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_attr: str,
    stub_value: bool | int,
    error_code: str,
    error_pointer: str | None,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    setattr(stub, stub_attr, stub_value)

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plan_calls = [call for call in stub.calls if call["marker"] == "plan-writer"]
    plans_dir = improve_monorepo_target / "daydream_plans"
    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert code == 1
    assert len(plan_calls) == 2
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    blocked_rows = [
        line
        for line in index.splitlines()
        if "BLOCKED (PLAN_VALIDATION_FAILED: " in line
    ]
    assert len(blocked_rows) == 1
    assert re.search(r"\| 001 <!-- fingerprint:", blocked_rows[0])
    dispositions = [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ]
    assert dispositions == ["retried", "blocked"]
    for attempt in diagnostics["attempts"]:
        assert attempt["stage"] == "authoring"
        assert any(
            error["code"] == error_code
            and (error_pointer is None or error["pointer"] == error_pointer)
            for error in attempt["errors"]
        )


@pytest.mark.anyio
async def test_plan_subverb_clamps_over_length_prose_without_repair(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    over_length_role = "Billing role " + "x" * 293
    assert len(over_length_role) == 306
    stub.plan_file_role_override = over_length_role

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plan_calls = [call for call in stub.calls if call["marker"] == "plan-writer"]
    assert code == 0
    assert len(plan_calls) == 1
    plans = list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    assert over_length_role[:299] + "…" in plan_text
    assert over_length_role not in plan_text
    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["success"]


@pytest.mark.anyio
async def test_plan_subverb_accepts_placeholder_secret_syntax(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_problem_override = (
        "Callers must send X-Internal-Service-Secret: <internalSecret> in "
        "production and X-Internal-Service-Secret: test-secret in tests."
    )

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plan_calls = [call for call in stub.calls if call["marker"] == "plan-writer"]
    assert code == 0
    assert len(plan_calls) == 1
    plans = list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    assert "X-Internal-Service-Secret: <internalSecret>" in plan_text
    assert "X-Internal-Service-Secret: test-secret" in plan_text


@pytest.mark.anyio
async def test_repository_secret_in_quoted_source_is_redacted_not_blocked(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A credential on a quoted source line must not reach the plan on disk.

    The excerpt is spliced from raw repository bytes after the authored-string
    redaction has already run. The secret shape is lowercase on purpose:
    ``trajectory.redact_text`` does not match it, so this exercises the
    improve-side redaction rather than pre-existing coverage.
    """
    source_path = improve_monorepo_target / "apps/billing/api.py"
    source_path.write_text(  # the stub quotes lines 1-2 of this file
        'password = "s3cr3tplaintext"\n'
        "def service_name():\n"
        '    return "billing"\n',
        encoding="utf-8",
    )
    install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert code == 0
    # A plan was written: the secret is redacted, not a reason to block.
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    assert "password = <redacted>" in plan_text
    assert "s3cr3tplaintext" not in plan_text
    assert all(
        "s3cr3tplaintext" not in observable
        for observable in _improve_observable_texts(improve_monorepo_target)
    )


@pytest.mark.anyio
async def test_secret_value_never_reaches_any_artifact(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.plan_problem_override = (
        "Document the rotation runbook including secret: hunter2realvalue "
        "before the credential expires."
    )
    stub.plan_bad_recon_id_attempts = 1

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert code == 0
    assert len(plan_calls) == 2
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    assert "<redacted>" in plan_text
    assert "hunter2realvalue" not in plan_text
    assert all(
        "hunter2realvalue" not in call["prompt"] for call in stub.calls
    )
    diagnostics_text = improve_artifact(
        improve_monorepo_target, "plan-write-diagnostics.json"
    ).read_text(encoding="utf-8")
    assert "hunter2realvalue" not in diagnostics_text
    diagnostics = json.loads(diagnostics_text)
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["retried", "success"]
    assert all(
        "hunter2realvalue" not in observable
        for observable in _improve_observable_texts(improve_monorepo_target)
    )


@pytest.mark.anyio
async def test_sloppy_but_salvageable_output_is_normalized_and_written(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_sloppy = True

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plan_calls = [call for call in stub.calls if call["marker"] == "plan-writer"]
    plans = list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert code == 0
    assert len(plan_calls) == 1
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    sloppy_role = "Billing role " + "x" * 293
    assert sloppy_role[:299] + "…" in plan_text
    assert sloppy_role not in plan_text
    assert "<redacted>" in plan_text
    assert "hunter2realvalue" not in plan_text
    assert "Make the change." not in plan_text
    assert "Planner scratch notes" not in plan_text
    assert "The billing implementation file must not change further." not in plan_text
    assert "The billing implementation does not alter documentation." in plan_text
    assert all(
        "hunter2realvalue" not in observable
        and "Planner scratch notes" not in observable
        for observable in _improve_observable_texts(improve_monorepo_target)
    )
    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["success"]


@pytest.mark.anyio
async def test_n_selected_findings_produce_n_plans_first_attempt(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=3,
    )
    stub.vet_reject_titles = {"Phantom N+1"}

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    plans_dir = improve_monorepo_target / "daydream_plans"
    plans = sorted(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert code == 0
    assert stub.plan_writer_calls == 3
    assert len(plans) == 3
    for plan_path in plans:
        plan_text = plan_path.read_text(encoding="utf-8")
        assert "`uv run pytest apps/billing/test_api.py -q`" in plan_text
        assert "`git diff --exit-code -- README.md`" in plan_text
        assert "exit 0 and the selected pytest tests pass" in plan_text
        assert "recon_command_id" not in plan_text
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert index.count("| TODO |") == 3
    assert "BLOCKED (PLAN_" not in index


@pytest.mark.anyio
async def test_a_finding_audited_by_several_stack_groups_yields_one_plan(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    make_config: MakeConfig,
) -> None:
    """One finding, one plan -- no matter how many groups re-audit its code.

    A partition whose files span stacks is bundled into one audit group per
    stack, so the same code is audited more than once and returns the identical
    finding each time. Pinning optimistic stack availability forces that
    multi-group fan-out (the CI condition; ambient dev environments collapse to
    a single generic group and never exercise it). The audit must collapse those
    byte-identical findings by fingerprint, or every extra group mints a
    duplicate plan and the plan numbers march past their true count.
    """
    _pin_stack_availability(monkeypatch, tmp_path)
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=3,
    )
    stub.vet_reject_titles = {"Phantom N+1"}

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    # The fan-out really did span multiple groups -- otherwise the dedup path is
    # never taken and the assertions below prove nothing.
    coverage = json.loads(
        improve_artifact(improve_monorepo_target, "coverage.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(coverage["groups"]) > 1
    audit_findings = json.loads(
        improve_artifact(
            improve_monorepo_target, "audit-findings.json"
        ).read_text(encoding="utf-8")
    )["findings"]
    fingerprints = [finding["fingerprint"] for finding in audit_findings]
    assert len(fingerprints) == len(set(fingerprints))

    plans = sorted(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    )
    assert stub.plan_writer_calls == 3
    assert [plan.name for plan in plans] == [
        "001-security-finding.md",
        "002-high-leverage-title.md",
        "003-performance-finding.md",
    ]


@pytest.mark.anyio
async def test_generalist_fallback_audits_and_plans_with_no_stack_skills(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Zero stack skills → generic routing → the audit still runs and plans land.

    This is the "works for everyone" baseline the generalist fallback exists to
    guarantee. It relies on the autouse ``_hermetic_skill_availability`` fixture
    (empty plugin registry → ``get_installed_skills()`` returns an empty set), so
    it deliberately does NOT call ``_pin_stack_availability``. Every stack falls
    back to generic, collapsing the monorepo into a single generic audit group;
    the flow must still produce one plan per selected finding.
    """
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=3,
    )
    stub.vet_reject_titles = {"Phantom N+1"}

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    # Generalist routing: one group, and its stack is the generic fallback value.
    coverage = json.loads(
        improve_artifact(improve_monorepo_target, "coverage.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(coverage["groups"]) == 1
    assert coverage["groups"][0]["stack"] == "generic"

    plans = sorted(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    )
    assert stub.plan_writer_calls == 3
    assert len(plans) == 3
    for plan_path in plans:
        plan_text = plan_path.read_text(encoding="utf-8")
        assert "`uv run pytest apps/billing/test_api.py -q`" in plan_text
        assert "exit 0 and the selected pytest tests pass" in plan_text
    index = (improve_monorepo_target / "daydream_plans" / "README.md").read_text(
        encoding="utf-8"
    )
    assert index.count("| TODO |") == 3


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("target_fixture", "injected", "expected_group_stacks"),
    [
        # An empty set collapses every stack into a single generic audit group.
        pytest.param(
            "improve_monorepo_target", frozenset[str](), ["generic"], id="empty-generalist"
        ),
        # A non-empty set splits the python services into their own group while
        # react (no injected skill) falls back to generic.
        pytest.param(
            "improve_scaled_monorepo_target",
            frozenset({"python"}),
            ["generic", "python"],
            id="nonempty-multistack",
        ),
    ],
)
async def test_injected_skill_availability_drives_routing_without_env(
    target_fixture: str,
    injected: frozenset[str],
    expected_group_stacks: list[str],
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """``RunConfig.skill_availability`` alone drives audit routing, with no probe.

    Proves availability is data carried on RunConfig, not ambient filesystem I/O:
    the field is injected directly (this test manipulates no CLAUDE_CONFIG_DIR /
    env and patches no ``get_installed_skills``). The autouse hermetic fixture
    makes the probe return an empty set, so a non-empty injected set producing
    multi-stack routing can only have come from the injected field.
    """
    target: Path = request.getfixturevalue(target_fixture)
    install_improve_stub(monkeypatch, target, n_findings=0)

    code = await run(
        make_config(
        target,
        flow_name="improve",
        skill_availability=injected,
        )
    )

    assert code == 0
    coverage = json.loads(improve_artifact(target, "coverage.json").read_text(encoding="utf-8"))
    # One group per expected stack: no collapse, and no stack split in two.
    assert len(coverage["groups"]) == len(expected_group_stacks)
    assert sorted(group["stack"] for group in coverage["groups"]) == expected_group_stacks


@pytest.mark.anyio
async def test_bad_recon_id_gets_named_feedback_and_retry_succeeds(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_bad_recon_id_attempts = 1
    stub.plan_missing_path_attempts = 1

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plan_calls = [call for call in stub.calls if call["marker"] == "plan-writer"]
    assert code == 0
    assert len(plan_calls) == 2
    repair_prompt = plan_calls[1]["prompt"]
    assert "RECON_COMMAND_UNKNOWN" in repair_prompt
    assert "/steps/0/verification" in repair_prompt
    assert "valid recon command ids: test-suite, git-diff" in repair_prompt
    assert "make-tests" not in repair_prompt
    assert "EXISTING_PATH_MISSING" in repair_prompt
    assert "/scope/existing_paths/1/path" in repair_prompt
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    assert "uv run pytest apps/billing/test_api.py -q" in plan_text
    assert "apps/billing/legacy_api.py" not in plan_text
    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["retried", "success"]
    first_attempt_codes = {
        error["code"] for error in diagnostics["attempts"][0]["errors"]
    }
    assert {"RECON_COMMAND_UNKNOWN", "EXISTING_PATH_MISSING"} <= first_attempt_codes


@pytest.mark.anyio
async def test_an_edited_file_left_unquoted_is_repaired_before_the_plan_lands(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """The drift STOP condition must never ship without text to compare.

    A first attempt that edits a file it never quotes is rejected and named
    back to the writer; the landed plan quotes every path drift lists.
    """
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_unquoted_path_attempts = 1

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plan_calls = [call for call in stub.calls if call["marker"] == "plan-writer"]
    assert code == 0
    assert len(plan_calls) == 2
    repair_prompt = plan_calls[1]["prompt"]
    assert "EXISTING_PATH_NOT_QUOTED" in repair_prompt
    assert "/scope/existing_paths/0/path" in repair_prompt
    assert "context_excerpts" in repair_prompt
    plans = list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    current_state = plan_text.split("## Current state\n\n", 1)[1].split("\n\n## Commands")[0]
    assert "- `apps/billing/api.py:1-2`" in current_state
    assert "def service_name" in current_state
    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert [attempt["disposition"] for attempt in diagnostics["attempts"]] == ["retried", "success"]
    first_attempt = diagnostics["attempts"][0]
    assert first_attempt["stage"] == "authoring"
    assert {(error["code"], error["pointer"]) for error in first_attempt["errors"]} == {
        ("EXISTING_PATH_NOT_QUOTED", "/scope/existing_paths/0/path")
    }


def _out_of_scope_section(plan_text: str) -> str:
    return plan_text.split("**Out of scope**\n\n", 1)[1].split("\n\n## ", 1)[0]


@pytest.mark.anyio
async def test_undeclared_stop_condition_path_lands_in_the_out_of_scope_section(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    deleted = "apps/billing/legacy_loader.py"
    stub.plan_stop_condition_path = deleted

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plan_calls = [call for call in stub.calls if call["marker"] == "plan-writer"]
    plans = list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert code == 0
    assert len(plan_calls) == 1
    assert len(plans) == 1
    assert not (improve_monorepo_target / deleted).exists()
    plan_text = plans[0].read_text(encoding="utf-8")
    assert (
        f"- `{deleted}` — Referenced by a stop condition for context only; "
        "do not create, modify, or depend on this path."
    ) in _out_of_scope_section(plan_text)
    assert "STOP_PATH_UNKNOWN" not in plan_text
    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["success"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "failure_attr",
    [
        pytest.param("plan_crash_attempts", id="transport-crash"),
        pytest.param("plan_stall_attempts", id="stream-stall"),
    ],
)
async def test_plan_writer_transient_failure_is_retried_and_the_plan_lands(
    failure_attr: str,
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A transient plan-writer failure is retried, not terminal, and the plan lands.

    A transport crash or a ``StreamStalledError`` is the usual symptom of a flaky
    endpoint. It is retryable, so ``run_agent`` re-arms a fresh subprocess and the
    plan writer completes on the second attempt — one blip must never sink a finding.
    """
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    setattr(stub, failure_attr, 1)

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plans = list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert code == 0
    assert stub.plan_writer_calls == 2
    assert len(plans) == 1
    assert "## Steps" in plans[0].read_text(encoding="utf-8")
    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["success"]


@pytest.mark.anyio
async def test_persistent_retryable_failure_does_not_restart_the_retry_budget(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """``run_agent`` owns the budget for retryable errors; the writer adds none.

    A persistently rate-limited plan writer must burn exactly one attempt budget
    and then block the finding — re-entering ``run_agent`` would hand the same
    dead endpoint a second full budget.
    """
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_rate_limit_always = True
    stub.retry_attempts = 2

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plans_dir = improve_monorepo_target / "daydream_plans"
    assert code == 1
    assert stub.plan_writer_calls == 3
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert "BLOCKED (PLAN_WRITER_FAILED: RATE_LIMIT)" in index


@pytest.mark.anyio
async def test_two_consecutive_transport_crashes_block_the_finding(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_crash_attempts = 2

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    plans_dir = improve_monorepo_target / "daydream_plans"
    assert code == 1
    assert stub.plan_writer_calls == 2
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert "BLOCKED (PLAN_WRITER_FAILED: PROCESS_EXIT)" in index
    diagnostics = _load_improve_json(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    assert [
        (attempt["disposition"], attempt["stage"])
        for attempt in diagnostics["attempts"]
    ] == [("blocked", "transport")]
    assert diagnostics["attempts"][0]["errors"] == [
        {"code": "PROCESS_EXIT", "pointer": "/"}
    ]


@pytest.mark.anyio
async def test_improve_run_leaves_no_stray_audit_worktree(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(make_config(improve_monorepo_target, flow_name="improve"))
    assert code == 0
    # After a full improve run, no audit worktree remains linked to the target.
    worktrees = git(improve_monorepo_target, "worktree", "list", "--porcelain")
    assert ".daydream/audit/" not in worktrees


@pytest.mark.anyio
async def test_improve_run_prunes_stale_audit_worktree_from_crashed_run(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A hard-killed improve run's locked audit worktree is reclaimed by the
    next run (the ``*-reanchor``-only prune cannot see it)."""
    install_improve_stub(monkeypatch, improve_monorepo_target)
    # Simulate a crashed prior run: the audit worktree was created locked and
    # only the owning run's finally-block removes it, so it survives the kill.
    stale_dir = improve_monorepo_target / ".daydream" / "audit" / "run-crashed"
    git(
        improve_monorepo_target,
        "worktree",
        "add",
        "--detach",
        "--lock",
        "--reason",
        "run-crashed",
        str(stale_dir),
        "HEAD",
    )
    common = Path(git(improve_monorepo_target, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = improve_monorepo_target / common
    old = time.time() - 48 * 3600
    os.utime(common / "worktrees" / stale_dir.name / "locked", (old, old))

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    worktrees = git(improve_monorepo_target, "worktree", "list", "--porcelain")
    assert "run-crashed" not in worktrees
    assert ".daydream/audit/" not in worktrees
    assert not stale_dir.exists()


@pytest.mark.anyio
async def test_improve_model_calls_run_in_audit_worktree_not_target(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    before_status = _git_status_porcelain(improve_monorepo_target)

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    # Every model turn's cwd must be a detached audit worktree under .daydream/audit/,
    # never the target worktree itself.
    assert stub.calls, "expected at least one model call"
    for call in stub.calls:
        assert call["cwd"] != improve_monorepo_target
        assert ".daydream/audit/" in str(call["cwd"]), str(call["cwd"])
    # The target tree is untouched (host artifacts under gitignored paths only).
    assert _git_status_porcelain(improve_monorepo_target) == before_status


class _AuditCommittingBackend(ImproveStubBackend):
    """Stub that performs a REAL git commit in every distinct cwd it is given.

    Simulates the Codex read-only 'git commit' residual: the model turn commits
    into whatever working directory it runs in, so isolation must confine those
    commits to the audit worktree by construction. A unique per-call scratch
    file is written first so the commit always has something to commit (the
    audit worktree is materialized exactly at the snapshot, so a plain
    add+commit on a clean tree would be a no-op failure).

    Each turn also exercises the escape class: it attempts to commit against
    the reachable parent target (the target worktree, via the relative path
    ``../../..`` from the audit cwd) instead of the confined cwd. git rejects
    a pathspec outside the audit repository, so the escape fails cleanly and
    never reaches the target's HEAD, refs, or index.
    """

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self._commit_count = 0
        self.escape_attempts = 0

    async def execute(
        self, cwd, prompt, output_schema=None, continuation=None,
        agents=None, max_turns=None, read_only=False, persist_session=True,
    ):
        self._commit_count += 1
        (cwd / "model-scratch.txt").write_text(
            f"model residual {self._commit_count}\n"
        )
        git(cwd, "add", "-A")
        git(cwd, "commit", "-m", "model residual commit")
        # Escape attempt: commit against the reachable parent target instead of
        # the confined cwd. The detached-worktree construction rejects the
        # relative pathspec as outside the audit repository, so both commands
        # fail cleanly and the target stays pristine.
        self.escape_attempts += 1
        escape_target = os.path.relpath(self._target, cwd)
        git(cwd, "add", escape_target, check=False)
        git(cwd, "commit", "-m", "escaped model commit", check=False)
        async for event in super().execute(
            cwd, prompt, output_schema=output_schema, continuation=continuation,
            agents=agents, max_turns=max_turns, read_only=read_only,
            persist_session=persist_session,
        ):
            yield event


@pytest.mark.anyio
async def test_improve_model_commit_is_confined_to_audit_worktree(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    configure_identity(improve_monorepo_target)
    stub = _AuditCommittingBackend(improve_monorepo_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *a, **k: stub)

    before_head = git(improve_monorepo_target, "rev-parse", "HEAD")
    before_refs = git(improve_monorepo_target, "show-ref")
    before_status = _git_status_porcelain(improve_monorepo_target)

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    assert stub.calls
    # Every model turn also attempted the escape, so the escape-class coverage
    # is real, not vacuous; the target assertions below prove it was confined.
    assert stub.escape_attempts == len(stub.calls)
    # Every model turn ran in ONE detached audit worktree (a single non-target path).
    audit_cwds = {str(call["cwd"]) for call in stub.calls}
    assert len(audit_cwds) == 1, audit_cwds
    audit_path = next(iter(audit_cwds))
    assert ".daydream/audit/" in audit_path and audit_path != str(improve_monorepo_target)
    # Target HEAD, named refs, and staged index/diff are unchanged after the full run.
    assert git(improve_monorepo_target, "rev-parse", "HEAD") == before_head
    assert git(improve_monorepo_target, "show-ref") == before_refs
    assert _git_status_porcelain(improve_monorepo_target) == before_status
    # The audit worktree is gone (the model committed into it, yet it was removed).
    worktrees = git(improve_monorepo_target, "worktree", "list", "--porcelain")
    assert ".daydream/audit/" not in worktrees


@pytest.mark.anyio
async def test_full_run_leaves_tracked_tree_and_untracked_set_untouched(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        attempt_write=True,
    )
    before_status = _git_status_porcelain(improve_monorepo_target)

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    assert _git_status_porcelain(improve_monorepo_target) == before_status
    new_untracked = _untracked(improve_monorepo_target)
    assert all(
        path.startswith(("daydream_plans/", ".daydream/"))
        for path in new_untracked
    )


@pytest.mark.anyio
async def test_every_agent_call_in_every_mode_is_read_only(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    configs = (
        make_config(improve_monorepo_target, flow_name="improve"),
        make_config(improve_monorepo_target, flow_name="improve", improve_plan_description="x"),
    )
    for config in configs:
        stub = install_improve_stub(monkeypatch, improve_monorepo_target)
        code = await run(config)
        assert code == 0
        assert stub.calls and all(call["read_only"] for call in stub.calls)


@pytest.mark.anyio
async def test_trajectory_records_improve_flow_and_phases(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    install_improve_stub(monkeypatch, improve_monorepo_target)

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    trajectories = list(
        (improve_monorepo_target / ".daydream" / "runs").glob(
            "*/trajectory.json"
        )
    )
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    run_root = trajectory.parent
    flows = _scan_trajectory_extra(
        run_root,
        trajectory,
        "daydream_run_flow",
    )
    phases = _scan_trajectory_extra(
        run_root,
        trajectory,
        "daydream_phase",
    )
    assert flows and set(flows) == {"improve"}
    assert {"recon", "audit", "vet", "plan_write"} <= set(phases)


_VET_FINDINGS = [
    {"fingerprint": "a", "title": "A", "path": "a.py", "line": 1},
    {"fingerprint": "b", "title": "B", "path": "b.py", "line": 2},
    {"fingerprint": "c", "title": "C", "path": "c.py", "line": 3},
]


@pytest.mark.parametrize(
    ("findings", "verdicts", "expected_kept", "expected_rejected"),
    [
        pytest.param(
            _VET_FINDINGS,
            # Returned in reverse order with 1-based vet_ids: the model may
            # return verdicts in any order, and matching must be by vet_id.
            [
                {"vet_id": 3, "keep": True, "reason": "ok"},
                {"vet_id": 1, "keep": False, "reason": "rejected"},
                {"vet_id": 2, "keep": True, "reason": "ok"},
            ],
            {"b", "c"},
            {"a"},
            id="reordered-verdicts-match-by-vet-id",
        ),
        pytest.param(
            _VET_FINDINGS[:2],
            # Only vet_id=2 is provided; vet_id=1 (a model obeying the old
            # zero-based prose would emit vet_id=0) is dropped, not kept.
            [{"vet_id": 2, "keep": True, "reason": "ok"}],
            {"b"},
            set(),
            id="missing-verdict-drops-finding",
        ),
    ],
)
def test_apply_vet_verdicts_matches_by_vet_id(
    findings: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    expected_kept: set[str],
    expected_rejected: set[str],
) -> None:
    kept, rejected = _apply_vet_verdicts(
        findings, verdicts, rejected_at_sha="sha"
    )
    assert {f["fingerprint"] for f in kept} == expected_kept
    assert {f["fingerprint"] for f in rejected} == expected_rejected


def test_apply_vet_verdicts_normalizes_schema_severity_for_prioritization() -> None:
    kept, _ = _apply_vet_verdicts(
        [_VET_FINDINGS[0]],
        [
            {
                "vet_id": 1,
                "keep": True,
                "reason": "confirmed",
                "severity": "medium",
            }
        ],
        rejected_at_sha="sha",
    )

    assert kept[0]["severity"] == "MED"


def test_apply_vet_verdicts_restamps_a_corrected_location(tmp_path: Path) -> None:
    findings = [
        {
            "fingerprint": "original",
            "title": "Misattributed finding",
            "category": "correctness",
            "path": "apps/billing/api.py",
            "line": 10,
            "body": "The evidence belongs to catalog.",
            "evidence": ["`apps/billing/api.py:10`"],
        }
    ]
    (tmp_path / "apps/billing").mkdir(parents=True)
    (tmp_path / "apps/catalog").mkdir(parents=True)
    (tmp_path / "apps/billing/api.py").write_text("first\n" * 10)
    (tmp_path / "apps/catalog/api.py").write_text("first\n" * 20)
    services = [
        Service("billing", Path("apps/billing"), "test"),
        Service("catalog", Path("apps/catalog"), "test"),
    ]
    partitions = [
        Partition("billing", "apps/billing", "test", "billing", ()),
        Partition("catalog", "apps/catalog", "test", "catalog", ()),
    ]

    kept, _ = _apply_vet_verdicts(
        findings,
        [{"vet_id": 1, "keep": True, "path": "apps/catalog/api.py", "line": 20}],
        rejected_at_sha="sha",
        repo=tmp_path,
        services=services,
        partitions=partitions,
    )

    assert len(kept) == 1
    assert kept[0]["evidence"] == ["`apps/catalog/api.py:20`"]
    assert kept[0]["services"] == ["catalog"]
    assert kept[0]["partition"] == "catalog"
    assert kept[0]["fingerprint"] != "original"


@pytest.mark.parametrize(
    "citation",
    [
        "../outside.py:1",
        "missing.py:1",
        "valid.py:3",
        "valid.py:2:3",
    ],
)
def test_stamp_finding_rejects_unconfined_missing_and_out_of_range_evidence(
    tmp_path: Path,
    citation: str,
) -> None:
    (tmp_path / "valid.py").write_text("first\nsecond\n")

    stamped = _stamp_finding(
        {"evidence": [citation]},
        "correctness",
        [],
        [],
        repo=tmp_path,
    )

    assert stamped is None


def test_stamp_finding_rejects_evidence_crossing_a_symlink(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir()
    (outside / "secret.py").write_text("secret\n")
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    stamped = _stamp_finding(
        {"evidence": ["escape/secret.py:1"]},
        "security",
        [],
        [],
        repo=tmp_path,
    )

    assert stamped is None

def test_stamp_finding_attributes_dot_slash_evidence_to_partition_and_service(
    tmp_path: Path,
) -> None:
    """``./``-prefixed evidence (legal since the grammar relaxed) must still be
    attributed to its partition and service, not silently dropped."""
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "app.py").write_text("x = 1\n")

    partition = Partition(
        name="frontend",
        root="frontend",
        source="directory",
        service=None,
        files=("frontend/app.py",),
    )
    service = Service(name="frontend", root=Path("frontend"), source="config")

    stamped = _stamp_finding(
        {"evidence": ["./frontend/app.py:1"]},
        "correctness",
        [service],
        [partition],
        repo=tmp_path,
    )

    assert stamped is not None
    assert stamped["partition"] == "frontend"
    assert stamped["services"] == ["frontend"]


@pytest.mark.anyio
async def test_improve_pi_calls_are_ephemeral(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    improve_calls = [
        call
        for call in stub.calls
        if call["marker"] in {"recon", "audit", "vet", "plan-writer"}
    ]
    assert {call["marker"] for call in improve_calls} == {
        "recon",
        "audit",
        "vet",
        "plan-writer",
    }
    assert all(call["persist_session"] is False for call in improve_calls)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("file_config", "expected_tiers"),
    [
        # Plan authoring runs on the top model tier at max reasoning; recon does not.
        pytest.param(
            None,
            {
                "plan-writer": ("claude-opus-5", "max"),
                "vet": ("claude-opus-5", "xhigh"),
                "audit": ("claude-sonnet-5", "high"),
                "recon": ("claude-sonnet-5", "low"),
            },
            id="built-in-defaults",
        ),
        # A ``[tool.daydream.phases.plan_write]`` table still wins over those
        # defaults, and unrelated phases keep their own.
        pytest.param(
            DaydreamFileConfig(
                phases={"plan_write": {"model": "claude-sonnet-5", "reasoning_effort": "medium"}}
            ),
            {
                "plan-writer": ("claude-sonnet-5", "medium"),
                "recon": ("claude-sonnet-5", "low"),
                "vet": ("claude-opus-5", "xhigh"),
            },
            id="phase-table-override",
        ),
    ],
)
async def test_improve_phases_resolve_their_own_model_and_reasoning_tier(
    file_config: DaydreamFileConfig | None,
    expected_tiers: dict[str, tuple[str, str]],
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Observed at the ``Backend.execute`` seam: each recorded turn carries the
    model and reasoning effort the backend serving it was constructed with.
    """
    calls = install_per_phase_improve_stubs(monkeypatch, improve_monorepo_target)

    code = await run(
        make_config(improve_monorepo_target, flow_name="improve", file_config=file_config)
    )

    assert code == 0
    tiers = _tiers_by_marker(calls)
    for marker, expected in expected_tiers.items():
        assert tiers[marker] == {expected}, marker


@pytest.mark.anyio
async def test_improve_runs_unbudgeted_so_a_long_turn_is_never_truncated(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A plan turn spending 200 tool calls completes and writes its plan.

    The flow used to cap every phase at 50 calls / 1800 s; ten of 49 archived
    audit turns recorded a real ``tool_call_budget_exceeded`` abort under it,
    and a budget abort returns partial output the flow reads as complete.
    """
    stub = install_improve_stub(
        monkeypatch, improve_monorepo_target, n_findings=1
    )
    stub.plan_tool_calls_before_result = 200

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    )
    assert len(plans) == 1
    diagnostics = json.loads(
        improve_artifact(improve_monorepo_target, "plan-write-diagnostics.json").read_text()
    )
    assert not any(
        error["code"] in ("TOOL_CALL_BUDGET_EXCEEDED", "WALL_BUDGET_EXCEEDED")
        for attempt in diagnostics["attempts"]
        for error in attempt["errors"]
    )


@pytest.mark.anyio
async def test_long_step_instruction_reaches_the_plan_whole(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A 2000-char step instruction renders in full, ending on its last word.

    The old 1500-char prose clamp cut real plan instructions off mid-sentence,
    handing the executor an order that stopped in the middle of a requirement.
    """
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    instruction = (
        "In apps/billing/api.py, replace the body of service_name. "
        + "Keep every existing caller working. " * 45
        + "Do NOT modify any other file in this step."
    )
    assert 1500 < len(instruction) <= 4000
    stub.plan_instruction_override = instruction

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    assert code == 0
    plan_text = next((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")).read_text(
        encoding="utf-8"
    )
    assert instruction in plan_text
    assert "…" not in plan_text


@pytest.mark.anyio
async def test_over_length_instruction_is_repaired_not_silently_truncated(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Past the schema ceiling the host asks for a rewrite instead of cutting."""
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_instruction_override = "Replace service_name. " + "x" * 4000

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="add rate limiting",
        )
    )

    assert code == 1
    diagnostics = json.loads(improve_artifact(improve_monorepo_target, "plan-write-diagnostics.json").read_text())
    errors = [error for attempt in diagnostics["attempts"] for error in attempt["errors"]]
    assert any(error["pointer"] == "/steps/0/changes/0/instruction" for error in errors), errors
    # The plan writer was asked again rather than a mangled plan being written.
    assert stub.plan_writer_calls == 2
    assert not list((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))


@pytest.mark.anyio
async def test_empty_secret_named_assignments_do_not_eat_the_next_line(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Redaction must not delete plan content it mistakes for a secret value.

    An instruction naming empty ``.env`` placeholders lost two of its five
    lines: each empty ``*_SECRET=``/``*_TOKEN=`` consumed the following line as
    its "value" and the replacement dropped the newline with it.
    """
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_instruction_override = (
        "Create .env.dev.example at the repository root with exactly these "
        "five empty assignment lines, in this order and with no value after "
        "any equals sign:\n"
        "CLERK_SECRET_KEY=\n"
        "CLOUDFLARE_ACCOUNT_ID=\n"
        "CLOUDFLARE_API_TOKEN=\n"
        "CLOUDFLARE_ACCOUNT_HASH=\n"
        "INTERNAL_SERVICE_SECRET="
    )

    code = await run(
        make_config(
        improve_monorepo_target,
        flow_name="improve",
        improve_plan_description="repoint dev env secrets",
        )
    )

    assert code == 0
    plan_text = next((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")).read_text(
        encoding="utf-8"
    )
    for key in (
        "CLERK_SECRET_KEY=",
        "CLOUDFLARE_ACCOUNT_ID=",
        "CLOUDFLARE_API_TOKEN=",
        "CLOUDFLARE_ACCOUNT_HASH=",
        "INTERNAL_SERVICE_SECRET=",
    ):
        assert key in plan_text, key
    assert "[REDACTED_ENV_VAR]" not in plan_text


@pytest.mark.anyio
async def test_rendered_plan_gives_a_literal_executor_no_room_to_guess(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Walk the rendered artifact for the points a zero-context agent stalls on."""
    install_improve_stub(monkeypatch, improve_monorepo_target, n_findings=1)

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    plan_path = next(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    )
    text = plan_path.read_text(encoding="utf-8")
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=improve_monorepo_target,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Preconditions: the executor is told where it must be standing, with the
    # full commit id and an exact expected result per command.
    assert "## Before you start" in text
    assert f"`git cat-file -e {head_sha}^{{commit}}`" in text
    assert "`git status --porcelain` — expected: no output at all." in text
    assert head_sha in text  # full sha, not only the 7-char Status stamp
    # A moved HEAD is expected, not a stop: drift is scoped to this plan's own
    # files, because plans are executed days or weeks after they are written.
    assert "You are expected to be running it later, from a HEAD" in text
    assert "that has moved on — that is normal and is not by itself a reason to" in text
    assert f"`git diff --name-only {head_sha} HEAD --" in text
    assert "Files outside this list do not matter." in text
    # The branch comes off the executor's current HEAD, not the planned-at sha.
    assert "`git switch --create improve/" in text
    assert f"`git switch --create improve/batch-billing-contract {head_sha}`" not in text
    assert "branches from your current HEAD, which is what you want." in text

    # Why-this-matters is labelled, so the intended outcome cannot be misread
    # as a statement about the code as it stands.
    assert "- **Problem**:" in text
    assert "- **Cost of leaving it**:" in text
    assert "- **Intended outcome (does not describe the code today)**:" in text

    # No judgement calls in host-owned wording.
    assert "unless a reviewer maintains the index" not in text
    assert "Do not skip a\n> step, reorder steps, or substitute your own judgement" in text

    # Every command says where to run it.
    assert "| Purpose | Run from | Command | Expected on success |" in text
    assert "**Run from**: the repository root" in text
    assert "Run this now, before starting the next step." in text

    # Ordering and section relationships are stated, not implied.
    assert "Do these in the order they are numbered." in text
    assert "write it once, not twice." in text

    # Finishing is literal, and never `git add -A`.
    assert "## Finishing" in text
    assert "never `git add -A`" in text
    assert "git add apps/billing/api.py apps/billing/test_api.py" in text
    assert "4. Do not push and do not open a pull request." in text
    # Issue publication copies the plan without committing the local index, so
    # an executor must not be told to edit daydream_plans/README.md.
    assert "`TODO` to `DONE`" not in text

    # The two previously unactionable STOP conditions now name the check.
    assert "Before editing a file, read the exact line range quoted for it in the Current state section" in text
    assert "two failures total for the same verification" in text


@pytest.mark.anyio
async def test_ungated_step_and_scope_criterion_still_get_a_real_check(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A step the model left ungated must not render a dead end.

    Five of six steps in a real replayed plan carried no command and rendered
    only "No host-verified command is attached to this step."
    """
    stub = install_improve_stub(
        monkeypatch, improve_monorepo_target, n_findings=1
    )
    stub.plan_ungate_steps = True

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    text = next(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    ).read_text(encoding="utf-8")

    assert "No host-verified command is attached to this step." not in text
    assert "No repository command was verified during planning for this step." in text
    assert (
        "confirm every **Target state** sentence above is now literally true"
        in text
    )
    assert "From the repository root run `git status --porcelain`." in text
    # The host-injected scope-integrity criterion is always ungated by the
    # model, and is exactly the one the host can always check itself.
    assert "(scope-integrity)" in text
    assert (
        "**Check**: from the repository root run `git status --porcelain`."
        in text
    )
    assert "No host-verified command is attached." not in text
    # An ungated test case names the symbol to run and forbids guessing a runner.
    assert (
        "run only `test_service_name_preserves_contract` in "
        "`apps/billing/test_api.py` using this repository's own test runner"
        in text
    )
    assert "stop and report that — do not guess a command." in text


@pytest.mark.anyio
async def test_plan_writer_is_told_to_leave_the_executor_no_decisions(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """The anti-ambiguity contract and per-field guidance reach the writer.

    Observed at the ``Backend.execute`` seam: the prompt text and the schema
    the plan-writer call actually received.
    """
    stub = install_improve_stub(
        monkeypatch, improve_monorepo_target, n_findings=1
    )

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    call = next(c for c in stub.calls if c["marker"] == "plan-writer")
    prompt = call["prompt"]
    assert "cannot infer and will not look" in prompt
    assert "has never seen this repository" in prompt
    for banned in ("the relevant handler", "as appropriate", "update accordingly"):
        assert banned in prompt, banned
    assert "Length is never a reason to compress." in prompt

    changes = call["output_schema"]["properties"]["steps"]["items"][
        "properties"
    ]["changes"]["items"]["properties"]
    assert "Banned:" in changes["instruction"]["description"]
    assert changes["instruction"]["maxLength"] == 4000
    assert "re-read the file" in changes["target_state"]["description"]
    assert "verbatim from the file" in changes["symbol"]["description"]
    done = call["output_schema"]["properties"]["done_criteria"]["items"][
        "properties"
    ]["description"]["description"]
    assert "without judgement" in done


@pytest.mark.anyio
async def test_configured_headless_publish_selects_all_and_embeds_local_plans(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Real runner path: configured CI mode writes no branches and files every plan."""
    install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=len(AUDIT_CATEGORIES),
    )
    monkeypatch.setattr(
        "daydream.git_ops.gh_issue_list_strict",
        lambda *args, **kwargs: [],
    )
    created: list[dict[str, Any]] = []

    def _create_issue(*args: Any, **kwargs: Any) -> str:
        created.append(dict(kwargs))
        return f"https://github.com/acme/widgets/issues/{len(created)}"

    monkeypatch.setattr("daydream.git_ops.gh_issue_create", _create_issue)
    branches_before = git(
        improve_monorepo_target,
        "for-each-ref",
        "--format=%(refname)",
        "refs/heads",
    )

    code = await run(
        make_config(
            improve_monorepo_target,
            flow_name="improve",
            pr_repo="acme/widgets",
            file_config=DaydreamFileConfig(
                improve_github_publish_issues=True,
            ),
        )
    )

    assert code == 0
    selected = _load_improve_json(improve_monorepo_target, "selected.json")
    vetted = _load_improve_json(
        improve_monorepo_target,
        "vetted-findings.json",
    )
    assert len(selected["selected"]) == len(vetted["findings"])
    assert len(selected["selected"]) > 5
    plan_files = sorted((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert len(created) == len(plan_files) == len(selected["selected"])
    assert {str(issue["body"]).partition("\n\n")[2] for issue in created} == {
        path.read_text(encoding="utf-8") for path in plan_files
    }
    assert all(str(issue["body"]).startswith("<!-- daydream-improve: package=") for issue in created)
    assert (
        git(
            improve_monorepo_target,
            "for-each-ref",
            "--format=%(refname)",
            "refs/heads",
        )
        == branches_before
    )


@pytest.mark.anyio
async def test_disabled_publication_overwrites_stale_current_run_artifact(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A disabled run cannot leave an earlier run looking current."""
    artifact = improve_artifact(
        improve_monorepo_target,
        "published-issues.json",
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "enabled": True,
                "status": "complete",
                "published": [{"issue_url": "https://example.test/stale"}],
            }
        ),
        encoding="utf-8",
    )
    install_improve_stub(monkeypatch, improve_monorepo_target, n_findings=0)

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    publication = _load_improve_json(
        improve_monorepo_target,
        "published-issues.json",
    )
    assert code == 0
    assert publication["enabled"] is False
    assert publication["status"] == "disabled"
    assert publication["published"] == []
    assert "example.test/stale" not in artifact.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_configured_publish_records_a_pathless_reconciled_plan(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    """An indexed package with no plan file is an explicit hard failure."""
    install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    first_code = await run(make_config(improve_monorepo_target, flow_name="improve"))
    plan_path = next((improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    plan_path.unlink()
    capsys.readouterr()
    monkeypatch.setattr(
        "daydream.git_ops.gh_issue_list_strict",
        lambda *args, **kwargs: pytest.fail("a pathless package must fail before GitHub reconciliation"),
    )

    second_code = await run(
        make_config(
            improve_monorepo_target,
            flow_name="improve",
            pr_repo="acme/widgets",
            file_config=DaydreamFileConfig(
                improve_github_publish_issues=True,
            ),
        )
    )

    publication = _load_improve_json(
        improve_monorepo_target,
        "published-issues.json",
    )
    assert first_code == 0
    assert second_code == 1
    assert publication["status"] == "failed"
    assert publication["published"] == []
    assert len(publication["failed"]) == 1
    assert publication["failed"][0]["stage"] == "plan-reconciliation"
    assert publication["failed"][0]["plan_path"] is None
    output = capsys.readouterr().out
    assert "Improve issue publishing failed" in output
    assert "Improve planning failed" not in output


@pytest.mark.anyio
async def test_reused_plan_publishes_its_stored_package_and_member_identities(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """A skipped current finding publishes the complete stored plan identity."""
    stub = install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    first_code = await run(
        make_config(improve_monorepo_target, flow_name="improve")
    )
    sidecar_path = (
        improve_monorepo_target / "daydream_plans" / PLAN_INDEX_FILENAME
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    stored = sidecar["plans"][0]
    stored["package_fingerprint"] = "stored-package"
    # Preserve multiplicity: duplicate semantic aliases tell the publisher it
    # must reconcile on raw member fingerprints instead.
    stored["member_aliases"] = ["stored-alias", "stored-alias"]
    stored_fingerprints = list(stored["member_fingerprints"])
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "daydream.git_ops.gh_issue_list_strict",
        lambda *args, **kwargs: [],
    )
    created: list[dict[str, Any]] = []

    def _create_issue(*args: Any, **kwargs: Any) -> str:
        created.append(dict(kwargs))
        return "https://github.com/acme/widgets/issues/1"

    monkeypatch.setattr("daydream.git_ops.gh_issue_create", _create_issue)

    second_code = await run(
        make_config(
            improve_monorepo_target,
            flow_name="improve",
            pr_repo="acme/widgets",
            file_config=DaydreamFileConfig(
                improve_github_publish_issues=True,
            ),
        )
    )

    publication = _load_improve_json(
        improve_monorepo_target,
        "published-issues.json",
    )
    assert first_code == second_code == 0
    assert stub.plan_writer_calls == 1
    assert len(created) == 1
    body = str(created[0]["body"])
    assert body.startswith(
        "<!-- daydream-improve: package=stored-package -->\n"
        "<!-- daydream-improve-member: alias=stored-alias -->\n"
    )
    assert all(
        f"<!-- daydream-improve-member: fingerprint={fingerprint} -->"
        in body
        for fingerprint in stored_fingerprints
    )
    assert publication["published"][0]["package_id"] == "stored-package"
    assert publication["published"][0]["member_aliases"] == [
        "stored-alias",
        "stored-alias",
    ]
    assert publication["published"][0]["member_fingerprints"] == (
        stored_fingerprints
    )


@pytest.mark.anyio
async def test_configured_publish_records_partial_plan_write_failure(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Every selected package remains accounted for when one writer fails."""
    backend = ProductionPathBackend(
        improve_monorepo_target,
        failed_title="Production finding 03",
    )
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
    )
    monkeypatch.setattr(
        "daydream.git_ops.gh_issue_list_strict",
        lambda *args, **kwargs: [],
    )
    created: list[str] = []

    def _create_issue(*args: Any, **kwargs: Any) -> str:
        created.append(str(kwargs["title"]))
        return "https://github.com/acme/widgets/issues/1"

    monkeypatch.setattr("daydream.git_ops.gh_issue_create", _create_issue)

    code = await run(
        make_config(
            improve_monorepo_target,
            flow_name="improve",
            pr_repo="acme/widgets",
            file_config=DaydreamFileConfig(
                improve_github_publish_issues=True,
            ),
        )
    )

    publication = _load_improve_json(
        improve_monorepo_target,
        "published-issues.json",
    )
    selected = _load_improve_json(improve_monorepo_target, "selected.json")
    assert code == 2
    assert publication["status"] == "partial"
    assert len(publication["published"]) == len(created) == 1
    assert len(publication["failed"]) == 1
    assert publication["failed"][0]["stage"] == "plan-write"
    assert publication["failed"][0]["plan_path"] is None
    assert len(publication["published"]) + len(publication["failed"]) == len(selected["selected"])


@pytest.mark.anyio
async def test_publication_only_failure_is_not_reported_as_planning_failure(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    make_config: MakeConfig,
) -> None:
    install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    monkeypatch.setattr(
        "daydream.git_ops.gh_issue_list_strict",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "daydream.git_ops.gh_issue_create",
        lambda *args, **kwargs: (_ for _ in ()).throw(GitError("offline")),
    )

    code = await run(
        make_config(
            improve_monorepo_target,
            flow_name="improve",
            pr_repo="acme/widgets",
            file_config=DaydreamFileConfig(
                improve_github_publish_issues=True,
            ),
        )
    )

    publication = _load_improve_json(
        improve_monorepo_target,
        "published-issues.json",
    )
    output = capsys.readouterr().out
    report = improve_artifact(
        improve_monorepo_target,
        "report.md",
    ).read_text(encoding="utf-8")
    assert code == 1
    assert publication["status"] == "failed"
    assert publication["published"] == []
    assert [entry["stage"] for entry in publication["failed"]] == ["issue-create"]
    assert "Improve issue publishing failed" in output
    assert "Improve planning failed" not in output
    assert "GitHub publication failures: 1" in report
