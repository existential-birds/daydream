"""Pure Markdown renderers for improve-advisor plan artifacts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from typing import Any

from daydream.improve.prioritize import plan_priority
from daydream.trajectory import redact_text

_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")


def plan_slug(title: Any) -> str:
    """Derive a plan's filename and branch slug from its title."""
    derived = _SLUG_SEPARATOR.sub("-", str(title or "").lower()).strip("-")
    return derived[:60].rstrip("-") or "plan"


def _redact_model_value(value: Any) -> Any:
    """Redact nested model-authored strings before durable host rendering."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_model_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_model_value(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _redact_model_value(item)
            for key, item in value.items()
        }
    return value


def markdown_cell(value: Any) -> str:
    """Render a value safely inside a Markdown table cell."""
    return redact_text(str(value or "—")).replace("|", "\\|").replace("\n", " ")


# Horizontal-only separator whitespace: see ``_ENV_VAR_PATTERN`` in
# daydream/trajectory.py — crossing a newline makes an empty assignment eat the
# following line.
# The secret token is matched as a whole ``_``/``-`` separated segment of the
# key name, not as a ``\b``-delimited word: ``_`` is itself a word character, so
# ``\bsecret\b`` never matched inside ``aws_secret_access_key`` and a live AWS
# key survived both this pass and ``trajectory.redact_text``. Segment anchoring
# is what keeps ``tokenizer:``/``passwordless:`` out of the match.
_SECRET_VALUE = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])(?:[A-Za-z0-9]{1,40}[_-]){0,4}"
    r"(?:token|password|secret|api[_-]?key)"
    r"(?:[_-][A-Za-z0-9]{1,40}){0,4}"
    r"[^\S\n\r]*[:=][^\S\n\r]*([^\s]+)"
)
# Structural placeholders are not secret values: angle-bracket slots,
# shell/env references, and obvious placeholder words. Anything else after a
# secret-named key still fails closed.
_SECRET_PLACEHOLDER = re.compile(
    r"(?i)^(?:"
    r"<[^<>]{1,80}>"
    r"|\$\{[^{}]{1,80}\}"
    r"|\$[A-Za-z_][A-Za-z0-9_]*"
    r"|(?:test|example|changeme|dummy|placeholder|redacted|sample|fake|stub"
    r"|mock|your)[A-Za-z0-9_.\-]*"
    r"|x{3,}|\*{3,}|\.{3,}|…"
    r")$"
)
_SECRET_VALUE_TRIM = "`'\".,;:()*"


def redact_secret_values(text: str) -> str:
    """Deterministically replace literal secret values with ``<redacted>``."""

    def _replace(match: re.Match[str]) -> str:
        candidate = match.group(1).strip(_SECRET_VALUE_TRIM)
        if candidate and not _SECRET_PLACEHOLDER.fullmatch(candidate):
            prefix_length = match.start(1) - match.start(0)
            return match.group(0)[:prefix_length] + "<redacted>"
        return match.group(0)

    return _SECRET_VALUE.sub(_replace, text)


def _run_from(command: dict[str, Any]) -> str:
    """Human-readable working directory for a command record."""
    directory = str(command.get("working_directory") or ".").strip() or "."
    return "the repository root" if directory == "." else f"`{directory}`"


def _path_list(paths: Sequence[str]) -> str:
    return ", ".join(f"`{path}`" for path in paths) or "none"


def _commands_table(commands: Sequence[dict[str, Any]]) -> str:
    if not commands:
        return "No host-verified repository commands were available during planning."
    lines = [
        "| Purpose | Run from | Command | Expected on success |",
        "|---------|----------|---------|---------------------|",
    ]
    lines.extend(
        f"| {markdown_cell(command['purpose'])} | {_run_from(command)} | "
        f"`{command['command']}` | "
        f"exit {command['expected_success']['exit_code']}; "
        f"{markdown_cell(command['expected_success']['observable_result'])} |"
        for command in commands
    )
    return "\n".join(lines)


def _manual_step_check(changed_paths: Sequence[str]) -> str:
    """Host-owned fallback when the model attached no command to a step.

    Never leave the executor with nothing to do: both checks below are
    deterministic, need no repository knowledge, and have a stated expected
    result.
    """
    listed = _path_list(changed_paths)
    return (
        "No repository command was verified during planning for this step. "
        "Verify it by hand instead, in this order:\n\n"
        f"1. Re-read {listed} and confirm every **Target state** sentence "
        "above is now literally true of the file contents. Expected: each one "
        "describes what the file now says.\n"
        "2. From the repository root run `git status --porcelain`. Expected: "
        f"the only paths listed are {listed}.\n\n"
        "If either check fails, that is a failed verification — apply the "
        "\"repeated-verification-failure\" STOP condition."
    )


def _render_verification(
    command: dict[str, Any] | None,
    *,
    changed_paths: Sequence[str],
) -> str:
    if command is None:
        return _manual_step_check(changed_paths)
    expected = command["expected_success"]
    exit_prefix = f"exit {expected['exit_code']} and "
    observable = expected["observable_result"]
    expected_text = (
        observable
        if observable.casefold().startswith(exit_prefix.casefold())
        else f"exit {expected['exit_code']}; {observable}"
    )
    note = command.get("note")
    return (
        "Run this now, before starting the next step.\n\n"
        f"**Purpose**: {command['purpose']}\n\n"
        f"**Run from**: {_run_from(command)}\n\n"
        f"**Command**: `{command['command']}`\n\n"
        f"**Expected**: {expected_text}"
        + (f"\n\n**Why this gate**: {note}" if note else "")
    )


def _render_verification_list(
    command: dict[str, Any] | None,
    *,
    indent: str,
    fallback: str,
) -> str:
    if command is None:
        return f"{indent}- {fallback}"
    expected = command["expected_success"]
    note = command.get("note")
    return "\n".join(
        (
            f"{indent}- **Purpose**: {command['purpose']}",
            f"{indent}- **Run from**: {_run_from(command)}",
            f"{indent}- **Command**: `{command['command']}`",
            (
                f"{indent}- **Expected**: exit {expected['exit_code']}; "
                f"{expected['observable_result']}"
            ),
            *(
                [f"{indent}- **Why this gate**: {note}"]
                if note
                else []
            ),
        )
    )


def _test_case_fallback(test_file: str, test_symbol: str) -> str:
    """Actionable stand-in when the model gated a test case with no command."""
    return (
        "**Check**: run only `{symbol}` in `{file}` using this repository's "
        "own test runner and confirm it passes. **If you cannot determine the "
        "runner command from the repository**, stop and report that — do not "
        "guess a command.".format(symbol=test_symbol, file=test_file)
    )


def _done_criterion_fallback(kind: str, in_scope_paths: Sequence[str]) -> str:
    """Actionable stand-in for a done criterion the model left ungated.

    ``scope-integrity`` is host-injected and always arrives ungated, yet it is
    the one criterion the host can always check: it is a git diff.
    """
    if kind == "scope-integrity":
        return (
            "**Check**: from the repository root run "
            "`git status --porcelain`. **Expected**: every listed path is one "
            f"of {_path_list(in_scope_paths)}, and no other path appears."
        )
    return (
        "No repository command was verified during planning for this "
        "criterion. Confirm it by re-reading the files named in it; if you "
        "cannot confirm it from the files alone, stop and report."
    )


def render_plan(
    finding: dict[str, Any],
    *,
    plan: dict[str, Any],
    planned_at: str,
    planned_on: date,
    number: int,
    run_session_id: str | None = None,
) -> str:
    """Render validated PlanWriterResult data without authored Markdown."""
    finding = _redact_model_value(finding)
    plan = _redact_model_value(plan)
    scope = plan["scope"]
    why = plan["why_this_matters"]
    current_state: list[str] = []
    for excerpt in plan["current_state_excerpts"]:
        anchor = excerpt["line_anchor"]
        current_state.extend(
            [
                (
                    f"- `{excerpt['path']}:{anchor['start_line']}-"
                    f"{anchor['end_line']}` — {excerpt['file_role']}"
                ),
                "",
                "```text",
                excerpt["verbatim_excerpt"],
                "```",
            ]
        )

    in_scope_lines = [
        *[
            f"- `{entry['path']}` (existing) — {entry['role']}"
            for entry in scope["existing_paths"]
        ],
        *[
            f"- `{entry['path']}` (create) — {entry['role']}"
            for entry in scope["new_paths"]
        ],
    ]
    out_scope_lines = [
        *[
            f"- `{entry['path']}` — {entry['reason']}"
            for entry in scope["out_of_scope_paths"]
        ],
        *[
            f"- {entry['behavior']} — {entry['reason']}"
            for entry in scope["out_of_scope_behaviors"]
        ],
    ]
    workflow = plan["git_workflow"]
    in_scope_paths = [
        *[entry["path"] for entry in scope["existing_paths"]],
        *[entry["path"] for entry in scope["new_paths"]],
    ]
    step_sections: list[str] = []
    for step in sorted(plan["steps"], key=lambda item: item["order"]):
        changes = "\n".join(
            (
                f"- `{change['path']}` — `{change['symbol']}` "
                f"({change['operation']}): {change['instruction']} "
                f"Target state: {change['target_state']}"
            )
            for change in step["changes"]
        )
        step_sections.append(
            f"### Step {step['order']}: {step['title']}\n\n"
            f"{changes}\n\n"
            "**Verify**\n\n"
            + _render_verification(
                step["verification"],
                changed_paths=[change["path"] for change in step["changes"]],
            )
        )
    test_plan = plan["test_plan"]
    exemplar_section = (
        "Copy the shape of these existing tests; do not invent a new style.\n\n"
        + "\n".join(
            f"- `{item['path']}` — `{item['symbol']}`: "
            f"{item['pattern_to_copy']}"
            for item in test_plan["exemplars"]
        )
        if test_plan["exemplars"]
        else (
            "This repository has no existing test to copy. Write each named "
            "case below from its own specification only, and do not go looking "
            "for a house style that is not there."
        )
    )
    case_lines = "\n\n".join(
        (
            f"- **{case['name']}** — `{case['test_file']}::"
            f"{case['test_symbol']}` ({case['kind']})\n"
            f"  - Setup: {case['setup']}\n"
            f"  - Action: {case['action']}\n"
            + "\n".join(
                f"  - Assert: {assertion}" for assertion in case["assertions"]
            )
            + "\n  - Verification:\n"
            + _render_verification_list(
                case["verification"],
                indent="    ",
                fallback=_test_case_fallback(
                    case["test_file"], case["test_symbol"]
                ),
            )
        )
        for case in test_plan["cases"]
    )
    done_lines = "\n".join(
        f"- [ ] **{criterion['id']} ({criterion['kind']})**: "
        f"{criterion['description']}\n"
        + _render_verification_list(
            criterion["verification"],
            indent="  ",
            fallback=_done_criterion_fallback(
                str(criterion["kind"]), in_scope_paths
            ),
        )
        for criterion in plan["done_criteria"]
    )
    stop_lines = "\n".join(
        f"- **{condition['kind']}** — {condition['condition']} "
        f"STOP and report: {condition['evidence_to_report']}"
        for condition in plan["stop_conditions"]
    )
    return (
        f"# Plan {number:03d}: {plan['title']}\n\n"
        "> **Executor instructions**: Do the \"Before you start\" checks first, then\n"
        "> work through the Steps in the order they are numbered. After each step run\n"
        "> its **Verify** block and confirm the stated expected result before starting\n"
        "> the next step. Change only the files listed under \"In scope\". Do not skip a\n"
        "> step, reorder steps, or substitute your own judgement for an instruction. If\n"
        "> anything in the \"STOP conditions\" section occurs, stop immediately and report\n"
        "> it — do not improvise and do not work around it. When every done criterion is\n"
        "> checked, follow the \"Finishing\" section at the end of this file.\n"
        "\n"
        "## Status\n\n"
        f"- **Priority**: {plan_priority(finding)}\n"
        f"- **Effort**: {finding.get('effort', '—')}\n"
        f"- **Risk**: {finding.get('risk', '—')}\n"
        f"- **Category**: {finding.get('category', '—')}\n"
        f"- **Planned at**: commit `{planned_at[:7]}`, {planned_on.isoformat()}\n\n"
        + (
            f"Daydream run: `{run_session_id}`\n\n"
            if run_session_id is not None
            else ""
        )
        + "## Before you start\n\n"
        "Run these from the repository root, in this order, before Step 1. Each\n"
        "one has an exact expected result.\n\n"
        "This plan was written against commit "
        f"`{planned_at}`. You are expected to be running it later, from a HEAD\n"
        "that has moved on — that is normal and is not by itself a reason to\n"
        "stop. What matters is only whether the files this plan edits have\n"
        "changed since then, which step 3 checks.\n\n"
        f"1. `git cat-file -e {planned_at}^{{commit}}` — expected: exit 0 and no\n"
        "   output. A failure means this clone does not contain the commit the\n"
        "   plan was written against (wrong repository, shallow clone, or the\n"
        "   commit was rewritten). Stop and report; do not continue.\n"
        "2. `git status --porcelain` — expected: no output at all. If anything\n"
        "   is listed, the working tree is dirty; stop and report the output.\n"
        f"3. `git diff --name-only {planned_at} HEAD -- "
        f"{' '.join(in_scope_paths)}` — expected: no output, meaning every file\n"
        "   this plan touches is byte-for-byte what it was at planning time.\n"
        "   Any path listed here changed since the plan was written, so the line\n"
        "   numbers and quoted text in \"Current state\" may be stale for that\n"
        "   file: before you edit it, re-read the line range quoted for it and\n"
        "   compare. If a quoted excerpt no longer matches, that is the `drift`\n"
        "   STOP condition. Files outside this list do not matter.\n"
        f"4. `git switch --create {workflow['branch_name']}` — expected:\n"
        f"   `Switched to a new branch '{workflow['branch_name']}'`. This\n"
        "   branches from your current HEAD, which is what you want. If the\n"
        "   branch already exists, stop and report; do not reuse or delete it.\n\n"
        "## Why this matters\n\n"
        f"- **Problem**: {why['problem']}\n"
        f"- **Cost of leaving it**: {why['concrete_cost']}\n"
        f"- **Intended outcome (does not describe the code today)**: "
        f"{why['intended_outcome']}\n\n"
        "## Current state\n\n"
        + "\n".join(current_state)
        + "\n\n## Commands you will need\n\n"
        + _commands_table(plan["commands_you_will_need"])
        + "\n\n## Scope\n\n**In scope**\n\n"
        + "\n".join(in_scope_lines)
        + "\n\n**Out of scope**\n\n"
        + "\n".join(out_scope_lines)
        + "\n\n## Git workflow\n\n"
        f"- **Branch**: {workflow['branch_name']} ({workflow['branch_basis']})\n"
        f"- **Commit boundaries**: {workflow['commit_boundaries']}\n"
        f"- **Commit example**: `{workflow['commit_message_example']}`\n"
        "- **Push**: never without operator instruction\n"
        "- **Pull request**: never without operator instruction\n\n"
        "## Steps\n\n"
        "Do these in the order they are numbered. Finish and verify each one "
        "before reading the next.\n\n"
        + "\n\n".join(step_sections)
        + "\n\n## Test plan\n\n"
        "These are the tests this plan requires. Where a step above already "
        "creates one, this section is that test's specification — write it "
        "once, not twice.\n\n"
        "### Exemplars\n\n"
        + exemplar_section
        + "\n\n### Named cases\n\n"
        + case_lines
        + "\n\n## Done criteria\n\n"
        "Every box must be checked before the plan is done.\n\n"
        + done_lines
        + "\n\n## STOP conditions\n\n"
        "If any of these happens, stop work immediately and report it. Do not "
        "attempt a workaround and do not continue to the next step.\n\n"
        + stop_lines
        + "\n\n## Finishing\n\n"
        "Only after every box under \"Done criteria\" is checked:\n\n"
        f"1. Stage exactly the in-scope paths — never `git add -A`, never "
        f"`git add .`: `git add {' '.join(in_scope_paths)}`\n"
        "2. Confirm nothing else is staged: `git status --porcelain` — "
        f"expected: every line is one of {_path_list(in_scope_paths)}.\n"
        "3. Commit, following the **Commit boundaries** line under \"Git "
        f"workflow\": `git commit -m \"{workflow['commit_message_example']}\"`\n"
        "4. Do not push and do not open a pull request.\n"
        f"5. Set this plan's Status cell in `daydream_plans/README.md` from "
        "`TODO` to `DONE`.\n"
    )
