"""Documentation-contract tests for the private PR Harbor benchmark runbook (issue #784).

Pins Daydream-owned docs/CLI/schema/privacy contracts only. No Harbor runtime,
Docker, or paid hosted-model command is ever executed here; paid commands
(calibrate-judge, run --oracle) are only asserted to be visibly marked as paid
gates and are never run in CI. Mirrors the existing doc-contract patterns in
tests/test_docs_contract.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.cli import _build_benchmark_parser

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "benchmark.md"
README = ROOT / "README.md"
CLAUDE = ROOT / "CLAUDE.md"

# The 12 shipped subcommands (MH-3), matching _build_benchmark_parser()
# in daydream/benchmark/cli.py:469-613.
EXPECTED_SUBCOMMANDS = {
    "init", "status", "validate", "build-harbor", "upgrade", "import-prs",
    "curate", "calibrate-judge", "run", "clean", "objective", "aggregate",
}

# Legacy runtime tokens MH-2 forbids as active instructions in the runbook.
FORBIDDEN_LEGACY_TOKENS = ("martian", "MARTIAN", "CodeRabbit", "anthropic-direct")


def _parser_choices() -> set[str]:
    subparsers = _build_benchmark_parser()._subparsers
    if subparsers is None:
        raise AssertionError("benchmark parser has no subparsers")
    choices = subparsers._group_actions[0].choices
    if choices is None:
        raise AssertionError("benchmark parser subcommands are empty")
    return set(choices)


def _code_lines(text: str) -> list[str]:
    """Nonblank, non-comment lines inside fenced ```bash/```sh blocks."""
    out: list[str] = []
    in_block = False
    for raw in text.splitlines():
        if raw.strip().startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            line = raw.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def _json_blocks(text: str) -> list[dict[str, Any]]:
    blocks = re.findall(r"```json\n(.*?)```", text, re.S)
    return [json.loads(b) for b in blocks if b.strip()]


def _objective_example() -> dict[str, Any]:
    for block in _json_blocks(RUNBOOK.read_text(encoding="utf-8")):
        if "objective" in block and "identity" in block:
            return block
    pytest.fail("runbook has no ```json objective example with an identity block")


def _suite_manifest_example() -> dict[str, Any]:
    for block in _json_blocks(RUNBOOK.read_text(encoding="utf-8")):
        if "entries" in block:
            return block
    pytest.fail("runbook has no ```json suite manifest example with entries")


# --- CLI / runbook command set (MH-3, MH-15) ---

def test_runbook_documents_every_shipped_subcommand() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    for sub in EXPECTED_SUBCOMMANDS:
        assert re.search(rf"`daydream benchmark {sub}(`|\s|$)", text), (
            f"runbook must document `daydream benchmark {sub}`"
        )


def test_every_documented_verb_exists_in_parser() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    documented = {m.group(1) for m in re.finditer(r"daydream benchmark ([\w-]+)", text)}
    assert documented <= _parser_choices()


def test_free_command_examples_parse() -> None:
    parser = _build_benchmark_parser()
    bad: list[str] = []
    for line in _code_lines(RUNBOOK.read_text(encoding="utf-8")):
        if not line.startswith("daydream benchmark "):
            continue
        argv = [t for t in line.split() if t != "\\"]
        if "#" in argv:
            argv = argv[: argv.index("#")]
        try:
            parser.parse_args(argv[2:])
        except SystemExit:
            bad.append(line)
    assert not bad, "documented command examples failed to parse:\n" + "\n".join(bad)


def test_benchmark_help_lists_all_subcommands() -> None:
    help_text = _build_benchmark_parser().format_help()
    for sub in EXPECTED_SUBCOMMANDS:
        assert sub in help_text, f"`daydream benchmark --help` must list {sub}"


# --- Legacy cutover (MH-2, MH-12) ---

def test_no_legacy_transition_note() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert len(re.findall(r"\bdaydream bench\b", text)) == 0


def test_no_active_legacy_instructions() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    for token in FORBIDDEN_LEGACY_TOKENS:
        assert token not in text, f"legacy token {token!r} must not appear in the runbook"


# --- Privacy (MH-7) ---

def test_privacy_placeholders_only() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "OWNER/REPO" in text
    assert re.search(r"sk-[A-Za-z0-9]", text) is None
    assert re.search(r"ghp_[A-Za-z0-9]", text) is None
    assert re.search(r"github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/pull/\d+", text) is None


# --- Objectives & suites (MH-9, MH-10) ---

def test_objective_example_matches_shipped_schema() -> None:
    obj = _objective_example()
    assert obj["schema_version"] == 1
    assert set(obj) == {"run_id", "mode", "schema_version", "identity", "objective"}
    assert obj["mode"] in {"oracle", "benchmark"}
    o = obj["objective"]
    for key in ("tp", "fp", "fn", "precision", "recall", "f1",
                "comparison_eligible", "task_count", "candidate_count", "gold_count"):
        assert key in o, f"objective JSON missing {key!r}"


def test_objective_example_is_privacy_safe() -> None:
    blob = json.dumps(_objective_example())
    assert "OWNER" not in blob and "github.com" not in blob
    assert not re.search(r"pull/\d+|pr-\d", blob)


def test_suite_example_pools_counts_never_averages_f1() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "pool" in text.lower()
    assert "micro" in text.lower()
    assert re.search(r"never.{0,40}averag|averag.{0,40}never", text, re.I)
    suite = _suite_manifest_example()
    assert suite["schema_version"] == 1 and "entries" in suite


def test_objective_aggregate_are_read_only() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert re.search(r"read-only|does not run Harbor|does not call a judge", text, re.I)


# --- Candidate-profile trust boundary (MH-8) ---

def test_profile_trust_boundary_documented() -> None:
    text = RUNBOOK.read_text(encoding="utf-8").lower()
    assert "packaged default" in text
    assert "control plane" in text or "control-plane" in text
    for token in ("schema version", "profile name", "source kind", "digest"):
        assert token in text, f"attribution field {token!r} not documented"
    assert ".daydream.toml" in text


# --- Upgrade path (#857, MH-11) ---

def test_upgrade_path_documented() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "--dry-run" in text
    assert "requested_base_sha" in text
    assert "original_base_sha" in text
    assert "finding_id" in text
    assert "unreplayable" in text
    assert re.search(r"no-op|idempotent|already upgraded", text, re.I)


# --- Prerequisites & privacy boundary (MH-4, MH-5, MH-6) ---

def test_prerequisites_documented() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "0.22" in text and "0.23" in text
    assert "Docker Desktop" in text
    assert "nftables" in text
    assert "openrouter.ai" in text
    assert "OpenRouter" in text and "Pi" in text
    for token in ("no-network", "allowlist", "frozen", "Oracle"):
        assert token in text, f"prereq/privacy token {token!r} not documented"


# --- Paid gates (MH-13) ---

def test_paid_gates_visibly_marked() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert text.lower().count("paid") >= 2
    assert "calibrate-judge" in text and "run --oracle" in text
    assert "never executed in CI" in text


# --- Reconciliation (MH-14) ---

def test_readme_harbor_is_supported_evaluator() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "Harbor" in readme
    assert "offline benchmark" not in readme
    assert "daydream_review_profile" in readme.lower()  # #889-owned wording preserved


def test_claude_has_no_active_legacy_benchmark_reference() -> None:
    text = CLAUDE.read_text(encoding="utf-8")
    assert re.search(r"\bdaydream bench\b", text) is None
    assert "docs/benchmark.md" in text
    for token in FORBIDDEN_LEGACY_TOKENS:
        assert token not in text, f"legacy token {token!r} must not appear in CLAUDE.md"


# --- Authoring anchors (issue-826) ---

def test_docs_describe_anchor_model_and_reasons() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    for needle in ("authoring", "re-anchored", "history-unavailable", "path-unavailable",
                   "range-unavailable"):
        assert needle in text, f"runbook must describe the authoring anchor ({needle!r})"
    assert "original head SHA" in text  # the runbook names authoring_anchor.commit_id, not commit_id
# --- Reported location/severity axes (issue #971) ---

def test_reward_example_documents_reported_axes() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    # Per-task reward keys (24-key schema, TEMPLATE_VERSION 4)
    for key in (
        "location_exact", "location_near", "location_file", "location_miss",
        "location_credit", "location_present", "severity_exact",
        "severity_within_1", "severity_mean_distance", "severity_credit",
        "severity_pairs", "severity_present",
    ):
        assert key in text, f"reward-key example missing {key!r}"
    # Pooled aggregate axis keys
    for key in (
        "location_exact_rate", "location_near_rate", "location_miss_rate",
        "location_pairs_scored", "severity_pairs_scored",
    ):
        assert key in text, f"aggregate example missing {key!r}"


def test_reported_axes_contract_documented() -> None:
    text = RUNBOOK.read_text(encoding="utf-8").lower()
    assert re.search(r"reported.{0,120}(never|do not|don't).{0,60}(gate|change)", text, re.S), (
        "runbook must state that the reported axes never gate tp or reward"
    )
    assert "axis" in text


# --- Change-scope guards (out-of-scope) ---

def test_removed_docs_not_recreated() -> None:
    assert not (ROOT / "docs" / "evaluation-framework.md").exists()


def test_no_project_local_skill_instruction() -> None:
    assert "SKILL.md" not in RUNBOOK.read_text(encoding="utf-8")


# --- Prioritized curation contract (issue-879) ---

def test_prioritization_contract_documented() -> None:
    runbook = (ROOT / "docs" / "benchmark.md").read_text(encoding="utf-8")
    for phrase in (
        "prioritized", "advisory", "all evidence is retained",
        "does not establish semantic correctness", "only explicit curator actions create gold",
        "resolution", "outdated",
    ):
        assert phrase in runbook, f"runbook must describe the prioritization contract ({phrase!r})"
