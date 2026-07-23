"""Persistent plan-directory state for the improve advisor flow."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from daydream.improve.prioritize import plan_priority
from daydream.trajectory import redact_text

REJECTIONS_SCHEMA_VERSION = 1
PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION = 1
_FINGERPRINT_MARKER = re.compile(
    r"<!--\s*fingerprint:([^\s>]+)\s*-->"
)
_NUMBERED_PLAN = re.compile(r"^(\d{3})-[a-z0-9-]+\.md$")
_SAFE_ERROR_DETAIL = re.compile(r"^[A-Za-z0-9_.;=-]{1,80}$")
_HOST_BLOCKED_STATUS = re.compile(
    r"^BLOCKED \(PLAN_(?:WRITER|VALIDATION)_FAILED: [^()\r\n]+\)$"
)
# Plan | Title | Priority | Effort | Status
_INDEX_COLUMNS = 5
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


def load_rejections(plans_dir: Path) -> dict[str, dict[str, Any]]:
    """Load durable rejections keyed by fingerprint.

    An absent, unreadable, malformed, or structurally invalid file is treated
    as empty so stale user-authored state cannot prevent a fresh audit.
    """
    path = plans_dir / "rejected.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != REJECTIONS_SCHEMA_VERSION
        or not isinstance(payload.get("rejected"), list)
    ):
        return {}

    rejections: dict[str, dict[str, Any]] = {}
    for entry in payload["rejected"]:
        if not isinstance(entry, dict):
            continue
        fingerprint = entry.get("fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            rejections[fingerprint] = entry
    return rejections


def record_rejections(
    plans_dir: Path, entries: Sequence[dict[str, Any]]
) -> None:
    """Append rejection entries to the versioned durable envelope."""
    if not entries:
        return
    rejected = [
        _redact_model_value(entry)
        for entry in load_rejections(plans_dir).values()
    ]
    rejected.extend(
        _redact_model_value(dict(entry))
        for entry in entries
    )
    plans_dir.mkdir(parents=True, exist_ok=True)
    (plans_dir / "rejected.json").write_text(
        json.dumps(
            {
                "schema_version": REJECTIONS_SCHEMA_VERSION,
                "rejected": rejected,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _markdown_cell(value: Any) -> str:
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
_SAFE_METADATA_LABEL = re.compile(r"^[A-Za-z0-9._:/-]{1,160}$")


def redact_secret_values(text: str) -> str:
    """Deterministically replace literal secret values with ``<redacted>``."""

    def _replace(match: re.Match[str]) -> str:
        candidate = match.group(1).strip(_SECRET_VALUE_TRIM)
        if candidate and not _SECRET_PLACEHOLDER.fullmatch(candidate):
            prefix_length = match.start(1) - match.start(0)
            return match.group(0)[:prefix_length] + "<redacted>"
        return match.group(0)

    return _SECRET_VALUE.sub(_replace, text)


def _safe_metadata_label(value: Any, *, fallback: str) -> str:
    text = redact_text(str(value or "").strip())
    if not _SAFE_METADATA_LABEL.fullmatch(text):
        return fallback
    return text


def _received_metadata(value: Any) -> dict[str, Any]:
    received_type = (
        "null"
        if value is None
        else "object"
        if isinstance(value, dict)
        else "array"
        if isinstance(value, list)
        else type(value).__name__
    )
    metadata: dict[str, Any] = {
        "type": received_type,
        "object_count": 0,
        "array_count": 0,
        "string_count": 0,
        "string_length": 0,
        "top_level_count": (
            len(value) if isinstance(value, (dict, list)) else None
        ),
    }

    def count_shape(item: Any) -> None:
        if isinstance(item, dict):
            metadata["object_count"] += 1
            for child in item.values():
                count_shape(child)
        elif isinstance(item, list):
            metadata["array_count"] += 1
            for child in item:
                count_shape(child)
        elif isinstance(item, str):
            metadata["string_count"] += 1
            metadata["string_length"] += len(item)

    count_shape(value)
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        metadata["sha256"] = None
        metadata["serialized_length"] = None
    else:
        metadata["sha256"] = hashlib.sha256(serialized).hexdigest()
        metadata["serialized_length"] = len(serialized)
    return metadata


def _validation_error(code_with_pointer: str) -> dict[str, str]:
    code, separator, remainder = code_with_pointer.partition("@")
    embedded_pointer, _, detail = remainder.partition("#")
    # Assembly issues always carry their own pointer; the host codes raised
    # around them are plan-wide.
    pointer = (
        embedded_pointer
        if separator and embedded_pointer.startswith("/")
        else "/"
    )
    if detail and _SAFE_ERROR_DETAIL.fullmatch(detail):
        return {"code": code, "pointer": pointer, "detail": detail}
    return {"code": code, "pointer": pointer}


# Codes emitted by assemble._collect_issues (seam 2, model authoring defects).
_AUTHORING_CODES = frozenset(
    {
        "AUTHOR_SCHEMA_INVALID",
        "MALFORMED_APPENDED_ARGS",
        "MALFORMED_PATH",
        "PATH_OUTSIDE_REPOSITORY",
        "EMPTY_SCOPE",
        "EXISTING_PATH_MISSING",
        "EXISTING_PATH_NOT_QUOTED",
        "NEW_PATH_ALREADY_EXISTS",
        "EXCERPT_ANCHOR_INVALID",
        "EXCERPT_PATH_MISSING",
        "RECON_COMMAND_UNKNOWN",
        "CREATE_PATH_NOT_NEW",
        "CHANGE_PATH_NOT_EXISTING",
        "TEST_EXEMPLAR_INVALID",
        "STOP_PATH_UNKNOWN",
    }
)


def _validation_stage(errors: Sequence[str]) -> str:
    codes = [error.partition("@")[0] for error in errors]
    if any(code == "RENDER_FAILED" for code in codes):
        return "render"
    if any(code in _AUTHORING_CODES for code in codes):
        return "authoring"
    return "semantic"


def _attempt_diagnostic(
    *,
    finding: dict[str, Any],
    attempt: dict[str, Any] | None,
    received: Any,
    disposition: str,
    stage: str,
    errors: Sequence[str] = (),
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attempt = attempt or {}
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "finding": {
            "fingerprint": str(finding.get("fingerprint") or ""),
            "title": redact_text(
                str(finding.get("title") or "Selected finding")
            ),
        },
        "planner": {
            "descriptor": _safe_metadata_label(
                attempt.get("descriptor"),
                fallback="plan-writer",
            ),
            "backend": _safe_metadata_label(
                attempt.get("backend"),
                fallback="unknown-backend",
            ),
            "model": _safe_metadata_label(
                attempt.get("model"),
                fallback="unknown-model",
            ),
        },
        "disposition": disposition,
        "stage": stage,
        "errors": [_validation_error(error) for error in errors],
        "validation_errors": [_validation_error(error) for error in errors],
        "received": _received_metadata(received),
        "artifact": artifact,
    }


def record_plan_write_diagnostics(
    path: Path,
    attempts: Sequence[dict[str, Any]],
    *,
    artifact_provenance: dict[str, str] | None = None,
) -> None:
    """Append sanitized plan-attempt metadata without retaining model content."""
    existing_attempts: list[dict[str, Any]] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("schema_version")
            == PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION
            and isinstance(existing.get("attempts"), list)
            and (
                artifact_provenance is None
                or existing.get("artifact_provenance")
                == artifact_provenance
            )
        ):
            existing_attempts = [
                _redact_model_value(item)
                for item in existing["attempts"]
                if isinstance(item, dict)
            ]
    payload = {
        "schema_version": PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION,
        "artifact_type": "daydream.plan-write-diagnostics",
        **(
            {"artifact_provenance": dict(artifact_provenance)}
            if artifact_provenance is not None
            else {}
        ),
        "attempts": [
            _redact_model_value(item)
            for item in [*existing_attempts, *attempts]
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _head_matches(repo: Path, planned_at: str) -> bool:
    planned = _git(repo, "rev-parse", "--verify", f"{planned_at}^{{commit}}")
    head = _git(repo, "rev-parse", "--verify", "HEAD")
    return (
        planned.returncode == 0
        and head.returncode == 0
        and planned.stdout.strip() == head.stdout.strip()
    )


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
        f"| {_markdown_cell(command['purpose'])} | {_run_from(command)} | "
        f"`{command['command']}` | "
        f"exit {command['expected_success']['exit_code']}; "
        f"{_markdown_cell(command['expected_success']['observable_result'])} |"
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
    number: int,
    planned_on: date | None = None,
    run_session_id: str | None = None,
) -> str:
    """Render validated PlanWriterResult data without authored Markdown."""
    finding = _redact_model_value(finding)
    plan = _redact_model_value(plan)
    planned_date = planned_on or date.today()
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
        f"- **Planned at**: commit `{planned_at[:7]}`, {planned_date.isoformat()}\n\n"
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


def _existing_index_rows(index_text: str) -> list[str]:
    rows: list[str] = []
    for line in index_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != _INDEX_COLUMNS or cells[0] in {"Plan", "------"}:
            continue
        if set(cells[0]) == {"-"}:
            continue
        rows.append(line)
    return rows


def _row_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip("|").split("|")]


def _row_number(row: str) -> int | None:
    cells = _row_cells(row)
    if not cells:
        return None
    match = re.search(r"\b(\d{3})\b", cells[0])
    return int(match.group(1)) if match is not None else None


def _row_plan_path(row: str) -> str | None:
    cells = _row_cells(row)
    if not cells:
        return None
    match = re.search(r"\[\d{3}\]\(([^)]+)\)", cells[0])
    if match is None:
        return None
    filename = match.group(1)
    if Path(filename).name != filename or _NUMBERED_PLAN.fullmatch(filename) is None:
        return None
    return filename


def _row_has_plan_artifact(row: str, plans_dir: Path) -> bool:
    filename = _row_plan_path(row)
    if filename is not None and (plans_dir / filename).is_file():
        return True
    number = _row_number(row)
    return number is not None and any(
        plans_dir.glob(f"{number:03d}-*.md")
    )


def _retryable_host_blocked_row(row: str, plans_dir: Path) -> bool:
    cells = _row_cells(row)
    if len(cells) != _INDEX_COLUMNS or _row_has_plan_artifact(row, plans_dir):
        return False
    return _HOST_BLOCKED_STATUS.fullmatch(cells[-1]) is not None


def planned_fingerprints(plans_dir: Path) -> set[str]:
    """Return fingerprints with durable executable/non-transient status."""
    index_path = plans_dir / "README.md"
    if not index_path.is_file():
        return set()
    try:
        index_text = index_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return set()
    rows = _existing_index_rows(index_text)
    return {
        match.group(1)
        for row in rows
        if not _retryable_host_blocked_row(row, plans_dir)
        if (match := _FINGERPRINT_MARKER.search(row)) is not None
    }


def _highest_plan_number(plans_dir: Path, rows: Sequence[str]) -> int:
    numbers = [
        int(match.group(1))
        for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
        if (match := _NUMBERED_PLAN.match(path.name)) is not None
    ]
    for row in rows:
        match = re.search(r"\b(\d{3})\b", row)
        if match is not None:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0)


def _render_index(
    rows: Sequence[str],
    *,
    plans_dir: Path,
    non_interactive_default: bool,
    run_session_id: str | None,
) -> str:
    rejections = load_rejections(plans_dir)
    default_note = (
        "\nThe non-interactive default selected the top-N vetted defect "
        "findings by leverage.\n"
        if non_interactive_default
        else ""
    )
    rejected_lines = [
        f"- {_markdown_cell(entry.get('title'))}: "
        f"{_markdown_cell(entry.get('reason') or 'rejected during vetting')} "
        f"<!-- fingerprint:{fingerprint} -->"
        for fingerprint, entry in rejections.items()
    ]
    return (
        "# Implementation Plans\n\n"
        f"Generated by daydream improve on {date.today().isoformat()}. Execute "
        "in the order below. Read each plan fully, honor its STOP conditions, "
        "and update its row when done.\n"
        + (
            f"\nDaydream run: `{run_session_id}`\n"
            if run_session_id is not None
            else ""
        )
        +
        f"{default_note}\n"
        "## Execution order & status\n\n"
        "| Plan | Title | Priority | Effort | Status |\n"
        "|------|-------|----------|--------|--------|\n"
        + ("\n".join(rows) if rows else "| — | No plans written. | — | — | — |")
        + "\n\nStatus values: TODO | IN PROGRESS | DONE | BLOCKED "
        "(with one-line reason) | REJECTED (with one-line rationale)\n\n"
        "## Findings considered and rejected\n\n"
        + ("\n".join(rejected_lines) if rejected_lines else "- None.")
        + "\n"
    )


def _blocked_index_row(
    *,
    number: int,
    marker: str,
    finding: dict[str, Any],
    status: str,
) -> str:
    """Render a blocked row without consulting rejected planner metadata."""
    trusted_title = str(finding.get("title") or "Selected finding")
    return (
        f"| {number:03d} {marker} | {_markdown_cell(trusted_title)} | "
        f"{plan_priority(finding)} | "
        f"{_markdown_cell(finding.get('effort'))} | {status} |"
    )


@dataclass(frozen=True)
class PlanReservation:
    """A plan number claimed before any plan writer has produced output.

    Numbers are handed out in the order the caller reserves them, so the
    filename a finding gets never depends on which writer finishes first.
    ``number`` is ``None`` when the finding is already planned or rejected and
    therefore consumes no number.
    """

    index: int
    fingerprint: str
    number: int | None


@dataclass(frozen=True)
class PlanOutcome:
    """What a single :meth:`PlanWriteSession.commit` did on disk."""

    status: str
    number: int | None
    path: str | None
    title: str


class PlanWriteSession:
    """Reconcile plan-writer results into files and the durable index.

    The session owns every piece of plan-directory state: number reservation
    (including reuse of a host-blocked attempt's number), validation,
    rendering, blocked-attempt rows, and index reconciliation. Callers reserve
    numbers once in a deterministic order, then commit each result as its
    writer completes, so a finished plan is on disk while slower writers are
    still running.

    ``commit`` is synchronous on purpose: called from concurrent async tasks it
    runs to completion without an await point, so the shared row/number state
    needs no lock.
    """

    def __init__(
        self,
        plans_dir: Path,
        *,
        planned_at: str,
        non_interactive_default: bool = False,
        run_session_id: str | None = None,
    ) -> None:
        self._plans_dir = plans_dir
        self._repo = plans_dir.parent
        self._planned_at = planned_at
        self._run_session_id = run_session_id
        plans_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = plans_dir / "README.md"
        index_text = (
            self._index_path.read_text(encoding="utf-8")
            if self._index_path.is_file()
            else ""
        )
        self._non_interactive_default = (
            non_interactive_default
            or "non-interactive default" in index_text.lower()
        )
        self._rows = _existing_index_rows(index_text)
        self._fingerprints = planned_fingerprints(plans_dir)
        self._rejected = load_rejections(plans_dir)
        self._next_number = _highest_plan_number(plans_dir, self._rows) + 1
        self._reserved_count = 0
        self._written: list[tuple[int, dict[str, Any]]] = []
        self._skipped: list[tuple[int, dict[str, Any]]] = []
        self._failed: list[tuple[int, dict[str, Any]]] = []
        self._diagnostics: list[tuple[int, dict[str, Any]]] = []
        self._planned_at_errors: tuple[str, ...] = ()
        commit = _git(self._repo, "cat-file", "-e", f"{planned_at}^{{commit}}")
        if commit.returncode != 0:
            self._planned_at_errors = ("PLANNED_AT_INVALID",)
        else:
            ancestor = _git(
                self._repo, "merge-base", "--is-ancestor", planned_at, "HEAD"
            )
            if ancestor.returncode != 0:
                self._planned_at_errors = ("PLANNED_AT_NOT_ANCESTOR",)

    def reserve(
        self, findings: Sequence[dict[str, Any] | None]
    ) -> list[PlanReservation]:
        """Claim one plan number per finding, in the order given."""
        reservations: list[PlanReservation] = []
        for finding in findings:
            index = self._reserved_count
            self._reserved_count += 1
            if not isinstance(finding, dict):
                reservations.append(PlanReservation(index, "", None))
                continue
            fingerprint = str(finding.get("fingerprint") or "")
            if fingerprint in self._fingerprints or fingerprint in self._rejected:
                reservations.append(
                    PlanReservation(index, fingerprint, None)
                )
                continue
            retry_rows = [
                row
                for row in self._rows
                if (
                    (match := _FINGERPRINT_MARKER.search(row)) is not None
                    and match.group(1) == fingerprint
                    and _retryable_host_blocked_row(row, self._plans_dir)
                )
            ]
            reserved_numbers = [
                number
                for row in retry_rows
                if (number := _row_number(row)) is not None
            ]
            self._rows = [row for row in self._rows if row not in retry_rows]
            if reserved_numbers:
                number = min(reserved_numbers)
            else:
                number = self._next_number
                self._next_number += 1
            reservations.append(PlanReservation(index, fingerprint, number))
        return reservations

    def commit(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
    ) -> PlanOutcome:
        """Land one plan-writer result, writing its file when it is complete."""
        safe = _redact_model_value(selection)
        if not isinstance(safe, dict):
            return PlanOutcome("ignored", None, None, "")
        finding = safe.get("finding")
        if not isinstance(finding, dict):
            return PlanOutcome("ignored", None, None, "")
        title = str(finding.get("title") or "Selected finding")
        if reservation.number is None:
            self._skipped.append((reservation.index, finding))
            attempt = self._attempt_of(safe)
            if attempt is not None:
                self._diagnostics.append(
                    (
                        reservation.index,
                        _attempt_diagnostic(
                            finding=finding,
                            attempt=attempt,
                            received=_plan_payload(safe),
                            disposition="skipped",
                            stage="reconciliation",
                            errors=("ALREADY_PLANNED_OR_REJECTED",),
                        ),
                    )
                )
            return PlanOutcome("skipped", None, None, title)
        return self._land(reservation, safe)

    def finish(self) -> dict[str, list[dict[str, Any]]]:
        """Reconcile the index and return what this session landed."""
        self._write_index()
        return {
            "written": _by_reservation(self._written),
            "skipped": _by_reservation(self._skipped),
            "failed": _by_reservation(self._failed),
            "diagnostics": _by_reservation(self._diagnostics),
        }

    @staticmethod
    def _attempt_of(selection: dict[str, Any]) -> dict[str, Any] | None:
        attempt = selection.get("_attempt")
        return attempt if isinstance(attempt, dict) else None

    def _block(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
        *,
        number: int,
        finding: dict[str, Any],
        status: str,
        stage: str,
        errors: Sequence[str],
        received: Any,
    ) -> PlanOutcome:
        self._rows.append(
            _blocked_index_row(
                number=number,
                marker=f"<!-- fingerprint:{reservation.fingerprint} -->",
                finding=finding,
                status=status,
            )
        )
        self._failed.append((reservation.index, finding))
        self._diagnostics.append(
            (
                reservation.index,
                _attempt_diagnostic(
                    finding=finding,
                    attempt=self._attempt_of(selection),
                    received=received,
                    disposition="blocked",
                    stage=stage,
                    errors=errors,
                ),
            )
        )
        self._write_index()
        return PlanOutcome(
            "blocked",
            number,
            None,
            str(finding.get("title") or "Selected finding"),
        )

    def _land(
        self,
        reservation: PlanReservation,
        selection: dict[str, Any],
    ) -> PlanOutcome:
        finding = selection["finding"]
        assert reservation.number is not None  # commit() gates on the number
        number = reservation.number
        title = str(finding.get("title") or "Selected finding")
        attempt = self._attempt_of(selection)
        slug = plan_slug(selection.get("title"))
        if selection.get("error"):
            raw_errors = attempt.get("errors") if attempt is not None else None
            if not isinstance(raw_errors, (list, tuple)) and attempt is not None:
                legacy_code = attempt.get("transport_error_code")
                raw_errors = (legacy_code,) if isinstance(legacy_code, str) else ()
            error_entries = tuple(
                entry
                for entry in (
                    raw_errors if isinstance(raw_errors, (list, tuple)) else ()
                )
                if isinstance(entry, str)
                and re.fullmatch(
                    r"[A-Z][A-Z0-9_]{1,63}", entry.partition("@")[0]
                )
            )
            if not error_entries:
                error_entries = ("UNKNOWN",)
            error_codes = tuple(
                entry.partition("@")[0] for entry in error_entries
            )
            if attempt is not None and attempt.get("validation"):
                status = (
                    "BLOCKED (PLAN_VALIDATION_FAILED: "
                    f"{','.join(error_codes)})"
                )
                stage = _validation_stage(error_entries)
            else:
                status = f"BLOCKED (PLAN_WRITER_FAILED: {error_codes[0]})"
                stage = "transport"
            return self._block(
                reservation,
                selection,
                number=number,
                finding=finding,
                status=status,
                stage=stage,
                errors=error_entries,
                received=(
                    attempt.get("received_result")
                    if attempt is not None
                    else None
                ),
            )

        plan_result = _plan_payload(selection)
        errors = self._planned_at_errors
        if not errors and not _head_matches(self._repo, self._planned_at):
            errors = ("PLAN_HEAD_CHANGED",)
        if errors:
            return self._block(
                reservation,
                selection,
                number=number,
                finding=finding,
                status=(
                    "BLOCKED (PLAN_VALIDATION_FAILED: "
                    f"{','.join(errors)})"
                ),
                stage=_validation_stage(errors),
                errors=errors,
                received=plan_result,
            )

        filename = f"{number:03d}-{slug}.md"
        try:
            text = render_plan(
                finding,
                plan=plan_result,
                planned_at=self._planned_at,
                number=number,
                run_session_id=self._run_session_id,
            )
        except Exception:  # noqa: BLE001 - persist a safe render disposition
            return self._block(
                reservation,
                selection,
                number=number,
                finding=finding,
                status="BLOCKED (PLAN_VALIDATION_FAILED: RENDER_FAILED)",
                stage="render",
                errors=("RENDER_FAILED",),
                received=plan_result,
            )
        if not _head_matches(self._repo, self._planned_at):
            return self._block(
                reservation,
                selection,
                number=number,
                finding=finding,
                status=(
                    "BLOCKED (PLAN_VALIDATION_FAILED: PLAN_HEAD_CHANGED)"
                ),
                stage=_validation_stage(("PLAN_HEAD_CHANGED",)),
                errors=("PLAN_HEAD_CHANGED",),
                received=plan_result,
            )
        (self._plans_dir / filename).write_text(text, encoding="utf-8")
        self._rows.append(
            f"| [{number:03d}]({filename}) "
            f"<!-- fingerprint:{reservation.fingerprint} --> | "
            f"{_markdown_cell(selection.get('title') or title)} | "
            f"{plan_priority(finding)} | "
            f"{_markdown_cell(finding.get('effort'))} | TODO |"
        )
        self._written.append(
            (
                reservation.index,
                {**selection, "number": number, "path": filename},
            )
        )
        self._diagnostics.append(
            (
                reservation.index,
                _attempt_diagnostic(
                    finding=finding,
                    attempt=attempt,
                    received=plan_result,
                    disposition="success",
                    stage="success",
                    artifact={"path": filename, "status": "TODO"},
                ),
            )
        )
        self._fingerprints.add(reservation.fingerprint)
        self._write_index()
        return PlanOutcome("written", number, filename, title)

    def _write_index(self) -> None:
        """Rewrite the index from the rows landed so far.

        Rewriting on every landing costs one small file write and leaves an
        interrupted run with an index that matches the plans already on disk.
        """
        self._rows.sort(
            key=lambda row: (
                _row_number(row) is None,
                _row_number(row) or 0,
            )
        )
        self._index_path.write_text(
            _render_index(
                self._rows,
                plans_dir=self._plans_dir,
                non_interactive_default=self._non_interactive_default,
                run_session_id=self._run_session_id,
            ),
            encoding="utf-8",
        )


def _plan_payload(selection: dict[str, Any]) -> dict[str, Any]:
    """Return the authored plan fields, without host bookkeeping keys."""
    return {
        key: value
        for key, value in selection.items()
        if key not in {"finding", "error"} and not key.startswith("_")
    }


def _by_reservation(
    entries: Sequence[tuple[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [entry for _, entry in sorted(entries, key=lambda item: item[0])]
