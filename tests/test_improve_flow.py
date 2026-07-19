import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import ResultEvent
from daydream.config import AUDIT_CATEGORIES, EFFORT_TIERS
from daydream.exploration_runner import repo_scan
from daydream.extensions.loader import build_registry
from daydream.git_ops import head_sha
from daydream.runner import RunConfig, run


class _ImproveStubBackend:
    """Backend stub for improve-flow tests."""

    model = "mock-model"

    def __init__(self, target: Path, *, n_findings: int | None = None) -> None:
        self._target = target
        self._n_findings = n_findings
        self.calls: list[dict[str, Any]] = []
        self.fail_categories: set[str] = set()
        self.vet_reject_titles: set[str] = set()

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        marker = "other"
        category = None
        if "you are a **pattern-scanner** specialist" in prompt.lower():
            marker = "repo-scan"
        elif "IMPROVE_RECON" in prompt:
            marker = "recon"
        elif "read-only improve audit specialist" in prompt:
            marker = "audit"
            headings = {
                "correctness": "## Correctness / Bugs",
                "security": "## Security",
                "performance": "## Performance",
                "tests": "## Test Coverage",
                "tech-debt": "## Tech Debt & Architecture",
                "dependencies": "## Dependencies & Migrations",
                "dx": "## DX & Tooling",
                "docs": "## Docs",
                "direction": "## Direction",
            }
            category = next(
                name for name, heading in headings.items() if heading in prompt
            )
        elif "You are the improve vet." in prompt:
            marker = "vet"
        elif "You are reviewing an existing daydream implementation plan" in prompt:
            marker = "plan-reviewer"
        elif "You are writing a self-contained implementation plan" in prompt:
            marker = "plan-writer"
        self.calls.append(
            {
                "cwd": cwd,
                "prompt": prompt,
                "output_schema": output_schema,
                "agents": agents,
                "max_turns": max_turns,
                "read_only": read_only,
                "marker": marker,
            }
        )
        if category in self.fail_categories:
            raise RuntimeError(f"{category} audit failed")
        if "you are the **pattern-scanner** specialist" in prompt.lower():
            yield ResultEvent(
                structured_output={
                    "conventions": [
                        {
                            "name": "OpenAPI First",
                            "description": "openapi.yaml is the HTTP contract",
                            "source": "CLAUDE.md",
                        }
                    ],
                    "guidelines": [],
                },
                continuation=None,
            )
            return
        if "IMPROVE_RECON" in prompt:
            yield ResultEvent(
                structured_output={
                    "languages": ["python", "typescript"],
                    "commands": {
                        "build": [],
                        "test": ["uv run pytest"],
                        "lint": ["uv run ruff check ."],
                    },
                    "conventions": ["OpenAPI First"],
                    "intent_docs": ["README.md"],
                },
                continuation=None,
            )
            return
        if category is not None:
            category_index = AUDIT_CATEGORIES.index(category)
            if (
                self._n_findings is not None
                and category_index >= self._n_findings
            ):
                yield ResultEvent(
                    structured_output={"findings": []},
                    continuation=None,
                )
                return
            title = f"{category.title()} finding"
            impact = "HIGH"
            effort = "S"
            if category == "correctness":
                title = "high-leverage-title"
            elif category == "docs":
                title = "low-leverage-title"
                impact = "LOW"
                effort = "L"
            findings = [
                {
                    "title": title,
                    "category": "wrong-agent-category",
                    "path": "apps/billing/api.py",
                    "line": 1,
                    "body": f"Concrete {category} impact and fix.",
                    "impact": impact,
                    "effort": effort,
                    "risk": "LOW",
                    "confidence": "HIGH",
                    "evidence": ["apps/billing/api.py:1"],
                }
            ]
            if category == "performance":
                findings.append(
                    {
                        "title": "Phantom N+1",
                        "category": "wrong-agent-category",
                        "path": "apps/catalog/api.py",
                        "line": 1,
                        "body": "Claims a query loop that is not present.",
                        "impact": "HIGH",
                        "effort": "S",
                        "risk": "LOW",
                        "confidence": "HIGH",
                        "evidence": ["apps/catalog/api.py:1"],
                    }
                )
            yield ResultEvent(
                structured_output={"findings": findings},
                continuation=None,
            )
            return
        if marker == "vet":
            match = re.search(
                r"Candidates .*?:\n```json\n(.*?)\n```",
                prompt,
                flags=re.DOTALL,
            )
            assert match is not None
            candidates = json.loads(match.group(1))
            yield ResultEvent(
                structured_output={
                    "verdicts": [
                        {
                            "vet_id": candidate["vet_id"],
                            "keep": candidate["title"]
                            not in self.vet_reject_titles,
                            "reason": (
                                "No query loop exists at the cited location."
                                if candidate["title"] in self.vet_reject_titles
                                else "Confirmed from cited evidence."
                            ),
                            "severity": None,
                            "impact": candidate["impact"],
                            "effort": candidate["effort"],
                            "risk": candidate["risk"],
                            "confidence": candidate["confidence"],
                            "path": candidate["path"],
                            "line": candidate["line"],
                        }
                        for candidate in candidates
                    ]
                },
                continuation=None,
            )
            return
        if marker == "plan-writer":
            match = re.search(
                r"Selected vetted finding:\n```json\n(.*?)\n```",
                prompt,
                flags=re.DOTALL,
            )
            assert match is not None
            finding = json.loads(match.group(1))
            slug = re.sub(r"[^a-z0-9]+", "-", finding["title"].lower()).strip(
                "-"
            )
            yield ResultEvent(
                structured_output={
                    "slug": slug,
                    "title": finding["title"],
                    "priority": "P1",
                    "depends_on": [],
                    "markdown": (
                        "## Why this matters\n\nConcrete impact.\n\n"
                        "## Current state\n\n`apps/billing/api.py:1`.\n\n"
                        "## Commands you will need\n\nUse verified commands.\n\n"
                        "## Scope\n\nOnly the cited file.\n\n"
                        "## Steps\n\n### Step 1\n\nMake the change.\n\n"
                        "## Test plan\n\nRun the tests.\n\n"
                        "## Done criteria\n\n- [ ] Tests pass.\n\n"
                        "## STOP conditions\n\nStop on drift.\n\n"
                        "## Maintenance notes\n\nReview carefully."
                    ),
                },
                continuation=None,
            )
            return
        if marker == "plan-reviewer":
            yield ResultEvent(
                structured_output={
                    "critique": "The original plan lacked executable detail.",
                    "markdown": (
                        "# Plan 001: Fix\n\n"
                        "## Status\n\n- **Priority**: P1\n\n"
                        "## Why this matters\n\nConcrete impact.\n\n"
                        "## Current state\n\n`apps/billing/api.py:1`.\n\n"
                        "## Commands you will need\n\n"
                        "| Purpose | Command | Expected on success |\n"
                        "|---|---|---|\n"
                        "| Test | `uv run pytest` | exit 0 |\n\n"
                        "## Scope\n\nOnly `apps/billing/api.py`.\n\n"
                        "## Steps\n\n### Step 1\n\nMake the change.\n\n"
                        "## Test plan\n\nRun `uv run pytest`.\n\n"
                        "## Done criteria\n\n- [ ] Tests pass.\n\n"
                        "## STOP conditions\n\nStop on drift.\n"
                    ),
                },
                continuation=None,
            )
            return
        raise AssertionError(f"unexpected improve prompt: {prompt[:120]}")

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


@pytest.fixture
def tmp_git_repo(improve_monorepo_target: Path) -> Path:
    return improve_monorepo_target


def _install_improve_stub(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    n_findings: int | None = None,
) -> _ImproveStubBackend:
    stub = _ImproveStubBackend(target, n_findings=n_findings)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: stub)
    return stub


def _git_status_porcelain(repo: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _dd(repo: Path, name: str) -> Path:
    return repo / ".daydream" / "improve" / name


def _forbidden_input(*_args: Any, **_kwargs: Any) -> str:
    raise AssertionError(
        "input() was called in non-interactive mode -- stdin must not be touched"
    )


def _force_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)


@pytest.mark.anyio
async def test_repo_scan_seeds_specialists_from_tracked_files(tmp_git_repo: Path) -> None:
    stub = _ImproveStubBackend(tmp_git_repo)
    ctx = await repo_scan(stub, tmp_git_repo, max_files=500)
    assert any(c.name == "OpenAPI First" for c in ctx.conventions)
    prompt = stub.calls[0]["prompt"]
    assert "api.py" in prompt
    assert stub.calls[0]["read_only"] is True


def test_registry_seeds_audit_slots_and_improve_prompts() -> None:
    r = build_registry()
    assert r.skill("audit:correctness:python") == "beagle-python:review-python"
    assert r.skill("audit:security:elixir") == "beagle-elixir:elixir-security-review"
    assert r.skill_if_registered("audit:dx") is None
    for name in ("audit", "vet", "plan-writer"):
        assert callable(r.prompt(name))


def test_audit_prompt_carries_playbook_section_and_hard_rules() -> None:
    prompt = build_registry().prompt("audit")(
        category="security",
        skill_invocation=None,
        services=[],
        scope_note="",
        recon_summary="langs: python",
        cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )
    assert "never reproduce secret values" in prompt.lower()
    assert "data, not instructions" in prompt.lower()
    assert "file:line" in prompt and "Effort" in prompt


@pytest.mark.anyio
async def test_improve_recon_writes_artifacts_and_never_mutates_source(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    before = _git_status_porcelain(improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 0
    dd = improve_monorepo_target / ".daydream" / "improve"
    services = json.loads((dd / "services.json").read_text())
    assert {service["name"] for service in services["services"]} == {
        "billing",
        "catalog",
    }
    assert (dd / "report.md").is_file()
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
    assert all(call["read_only"] for call in stub.calls)
    assert _git_status_porcelain(improve_monorepo_target) == before


@pytest.mark.anyio
async def test_standard_effort_fans_out_all_nine_categories_read_only(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    audited = json.loads(
        _dd(improve_monorepo_target, "audit-findings.json").read_text()
    )
    assert set(audited["categories_run"]) == set(AUDIT_CATEGORIES)
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert audit_calls and all(call["read_only"] for call in audit_calls)
    assert any(
        "beagle-python:review-python" in call["prompt"]
        for call in audit_calls
    )


@pytest.mark.anyio
async def test_quick_effort_restricts_categories(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_effort="quick",
            non_interactive=True,
            archive=False,
        )
    )
    audited = json.loads(
        _dd(improve_monorepo_target, "audit-findings.json").read_text()
    )
    assert set(audited["categories_run"]) == {
        "correctness",
        "security",
        "tests",
    }


@pytest.mark.anyio
async def test_focus_security_audits_single_category(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_focus="security",
            non_interactive=True,
            archive=False,
        )
    )
    audited = json.loads(
        _dd(improve_monorepo_target, "audit-findings.json").read_text()
    )
    assert audited["categories_run"] == ["security"]


@pytest.mark.anyio
async def test_focus_next_is_direction_only_and_plans_are_spikes(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_focus="next",
            non_interactive=True,
            archive=False,
        )
    )
    audited = json.loads(
        _dd(improve_monorepo_target, "audit-findings.json").read_text()
    )
    assert audited["categories_run"] == ["direction"]
    audit_prompts = [
        call["prompt"] for call in stub.calls if call["marker"] == "audit"
    ]
    assert audit_prompts and all(
        "4–6 grounded suggestions" in prompt for prompt in audit_prompts
    )
    plan = next(
        (improve_monorepo_target / "daydream_plans").glob("0*.md")
    ).read_text()
    assert "spike" in plan.lower()


@pytest.mark.anyio
async def test_branch_focus_scopes_audit_to_merge_base_diff_and_tags_provenance(
    improve_branch_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_branch_target)
    await run(
        RunConfig(
            target=str(improve_branch_target),
            flow_name="improve",
            improve_focus="branch",
            non_interactive=True,
            archive=False,
        )
    )
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert all(
        "apps/billing/api.py" in call["prompt"] for call in audit_calls
    )
    vetted = json.loads(
        _dd(improve_branch_target, "vetted-findings.json").read_text()
    )
    assert {finding["provenance"] for finding in vetted["findings"]} <= {
        "introduced",
        "inherited",
    }


@pytest.mark.anyio
async def test_branch_focus_on_base_branch_reports_and_exits_cleanly(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_focus="branch",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 1


@pytest.mark.anyio
async def test_failed_category_is_reported_not_silently_dropped(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.fail_categories = {"performance"}
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 0
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert "performance" in report.lower()
    assert "not audited" in report.lower()


@pytest.mark.anyio
async def test_vet_rejects_unconfirmed_finding_with_reason_and_persists(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.vet_reject_titles = {"Phantom N+1"}
    await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    vetted = json.loads(
        _dd(improve_monorepo_target, "vetted-findings.json").read_text()
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
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.vet_reject_titles = {"Phantom N+1"}
    config = RunConfig(
        target=str(improve_monorepo_target),
        flow_name="improve",
        non_interactive=True,
        archive=False,
    )
    await run(config)

    stub.calls.clear()
    await run(config)

    vet_calls = [call for call in stub.calls if call["marker"] == "vet"]
    assert all("Phantom N+1" not in call["prompt"] for call in vet_calls)
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert "previously rejected" in report.lower()


@pytest.mark.anyio
async def test_non_interactive_selects_top_findings_never_touching_stdin(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", _forbidden_input)
    _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=8,
    )
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 0
    selected = json.loads(
        _dd(improve_monorepo_target, "selected.json").read_text()
    )
    assert len(selected["selected"]) == 5
    assert selected["mode"] == "non-interactive-default"


@pytest.mark.anyio
async def test_non_interactive_run_writes_plans_and_index(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=8,
    )
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    plans_dir = improve_monorepo_target / "daydream_plans"
    plan_files = sorted(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert code == 0
    assert 1 <= len(plan_files) <= 5
    index = (plans_dir / "README.md").read_text()
    assert "non-interactive default" in index.lower()
    assert head_sha(improve_monorepo_target)[:7] in plan_files[0].read_text()


@pytest.mark.anyio
async def test_interactive_selection_honors_user_choice(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "2")
    _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=8,
    )
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=False,
            archive=False,
        )
    )
    assert code == 0
    selected = json.loads(
        _dd(improve_monorepo_target, "selected.json").read_text()
    )
    assert len(selected["selected"]) == 1
    assert selected["mode"] == "interactive"


@pytest.mark.anyio
async def test_report_orders_by_leverage_and_separates_direction(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=9,
    )
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 0
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert report.index("high-leverage-title") < report.index(
        "low-leverage-title"
    )
    assert "## Direction" in report
    assert "not audited" in report.lower()


@pytest.mark.anyio
async def test_plan_subverb_skips_audit_and_writes_single_plan(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    assert not _dd(improve_monorepo_target, "audit-findings.json").exists()
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert len(plans) == 1
    assert "rate limiting" in plans[0].read_text().lower()


@pytest.mark.anyio
async def test_review_plan_rejects_files_outside_daydream_plans(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_review_plan="README.md",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 1


@pytest.mark.anyio
async def test_review_plan_tightens_in_place(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    plan = improve_monorepo_target / "daydream_plans" / "001-fix.md"
    plan.parent.mkdir()
    plan.write_text("# Plan 001: Fix\n\nMake the fix.\n")

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_review_plan=str(plan),
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    assert "## STOP conditions" in plan.read_text()
