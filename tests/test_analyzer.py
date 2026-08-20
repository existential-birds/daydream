"""Tests for daydream.eval.analyzer trajectory loading and grounding.

Focused on session-id resolution semantics inside ``load_trajectories``:
ambiguous prefixes must raise instead of silently picking one, exact
matches must take precedence over prefix matches, and unique prefixes
must still resolve.

Also covers ``analyze_grounding``'s rate arithmetic, including the
undefined (zero-findings) case, which must not report a perfect score.
"""

import json
import math
from pathlib import Path

import pytest

from daydream.backends import MetricsEvent, ResultEvent, TextEvent
from daydream.eval.analyzer import (
    _files_read,
    _tokenize_command,
    analyze_costs,
    analyze_coverage,
    analyze_grounding,
    analyze_quality,
    analyze_session,
    load_trajectories,
)
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder


def _write_run(daydream_dir: Path, session_id: str, marker: str) -> Path:
    """Create a minimal ``runs/<session_id>/trajectory.json`` fixture.

    ``load_trajectories`` only reads the file with ``json.loads`` and
    stuffs a ``_source_file`` key onto the returned dict, so any valid
    JSON object is enough for the resolution path we're exercising.
    """
    run_dir = daydream_dir / "runs" / session_id
    run_dir.mkdir(parents=True)
    traj = run_dir / "trajectory.json"
    traj.write_text(json.dumps({"session_id": session_id, "marker": marker}))
    return run_dir


def test_ambiguous_prefix_raises(tmp_path: Path):
    daydream_dir = tmp_path / ".daydream"
    _write_run(daydream_dir, "abcd1234-0000-0000-0000-000000000001", "first")
    _write_run(daydream_dir, "abcd1234-0000-0000-0000-000000000002", "second")

    with pytest.raises(ValueError, match="matches multiple runs"):
        load_trajectories(daydream_dir, session_id="abcd1234")


def test_unique_prefix_resolves(tmp_path: Path):
    daydream_dir = tmp_path / ".daydream"
    _write_run(daydream_dir, "abcd1234-0000-0000-0000-000000000001", "first")
    _write_run(daydream_dir, "ffff0000-0000-0000-0000-000000000002", "second")

    result = load_trajectories(daydream_dir, session_id="abcd1234")

    assert result["main"] is not None
    assert result["main"]["marker"] == "first"
    assert result["forked"] == []


def test_exact_match_takes_precedence(tmp_path: Path):
    """An exact dir name must win even if a longer dir would also prefix-match."""
    daydream_dir = tmp_path / ".daydream"
    # Exact id and a sibling whose name starts with the same string.
    _write_run(daydream_dir, "abcd1234", "exact")
    _write_run(daydream_dir, "abcd1234-extra", "prefix-only")

    result = load_trajectories(daydream_dir, session_id="abcd1234")

    assert result["main"] is not None
    assert result["main"]["marker"] == "exact"


def test_analyze_costs_preserves_fractional_aggregate_precision():
    trajectories = {
        "main": {
            "_source_file": "trajectory.json",
            "final_metrics": {"total_cost_usd": 0.00006},
        },
        "forked": [],
    }

    result = analyze_costs(trajectories)

    assert result["total_cost_usd"] == 0.00006
    assert sum(agent["cost_usd"] for agent in result["by_agent"]) == result["total_cost_usd"]


def test_analyze_costs_includes_cached_tokens_when_prompt_dominates():
    trajectories = {
        "main": {
            "_source_file": "trajectory.json",
            "final_metrics": {
                "total_prompt_tokens": 140,
                "total_cached_tokens": 14,
            },
        },
        "forked": [],
    }

    result = analyze_costs(trajectories)

    assert result["total_input_tokens"] == 140
    assert result["cache_hit_rate"] == 0.1


def test_analyze_costs_aggregates_legacy_fork_metrics():
    trajectories = {
        "main": {
            "_source_file": "trajectory.json",
            "final_metrics": {"total_cost_usd": 1.0},
        },
        "forked": [
            {
                "_source_file": "fork.json",
                "final_metrics": {"total_cost_usd": 0.5},
            }
        ],
    }

    result = analyze_costs(trajectories)

    assert result["total_cost_usd"] == 1.5


async def test_analyze_costs_assigns_nested_forks_their_own_metrics(tmp_path: Path):
    session = "nested-forks"
    daydream_dir = tmp_path / ".daydream"
    recorder = TrajectoryRecorder(
        path=daydream_dir / "runs" / session / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id=session,
    )

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="main"))
            inv.observe(MetricsEvent("main", 10, 1, 0, 0.1))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.fork("outer") as outer:
            async with outer.invocation(phase=DaydreamPhase.DEEP) as inv:
                inv.observe(TextEvent(text="outer"))
                inv.observe(MetricsEvent("outer", 20, 2, 1, 0.2))
                inv.observe(ResultEvent(structured_output=None, continuation=None))
            async with outer.fork("inner") as inner:
                async with inner.invocation(phase=DaydreamPhase.DEEP) as inv:
                    inv.observe(TextEvent(text="inner"))
                    inv.observe(MetricsEvent("inner", 30, 3, 1, 0.3))
                    inv.observe(ResultEvent(structured_output=None, continuation=None))

    trajectories = load_trajectories(daydream_dir, session)

    result = analyze_costs(trajectories)

    by_agent = {agent["agent"]: agent for agent in result["by_agent"]}
    assert by_agent["main"]["cost_usd"] == pytest.approx(0.1)
    assert by_agent["outer"]["cost_usd"] == pytest.approx(0.2)
    assert by_agent["inner"]["cost_usd"] == pytest.approx(0.3)
    assert by_agent["main"]["steps"] == 1
    assert by_agent["outer"]["steps"] == 1
    assert by_agent["inner"]["steps"] == 1
    assert sum(agent["cost_usd"] for agent in result["by_agent"]) == pytest.approx(
        result["total_cost_usd"]
    )
    assert sum(agent["steps"] for agent in result["by_agent"]) == 3


# --- analyze_grounding ---


def _read_traj(source_file: str, *read_paths: str, pi_style: bool = False) -> dict:
    """Forked-trajectory fixture whose agent Read each of ``read_paths``.

    Shaped for ``_extract_tool_calls``: every step needs a ``step_id`` and its
    ``tool_calls`` need ``function_name``/``arguments``. ``_files_read`` keeps
    only the ``file_path`` of ``Read`` calls, and ``_agent_label`` derives the
    ``deep-<stack>`` key from ``_source_file``. With ``fn_style=True`` the
    calls use the pi style ``read``/``arguments.path`` shape.
    """
    steps = []
    for i, path in enumerate(read_paths):
        if pi_style:
            tc = {"function_name": "read", "arguments": {"path": path}}
        else:
            tc = {"function_name": "Read", "arguments": {"file_path": path}}
        steps.append({"step_id": f"s{i}", "tool_calls": [tc]})
    return {"_source_file": source_file, "steps": steps}


def test_exploration_utilization_counts_only_deterministic_artifact():
    from daydream.eval.analyzer import analyze_exploration_utilization

    trajectories = {
        "main": None,
        "forked": [
            _read_traj("deep-python.json", "/repo/.daydream/exploration/summary.md"),
            _read_traj("deep-ts.json", "/repo/.daydream/exploration/affected_files.md"),
            _read_traj("deep-rust.json", "/repo/.daydream/exploration/conventions.md"),
            _read_traj(
                "deep-c.json", "/repo/.daydream/exploration/affected_files.md", pi_style=True
            ),
        ],
    }
    result = analyze_exploration_utilization(trajectories)
    by_agent = {agent["agent"]: agent for agent in result["by_agent"]}
    assert by_agent["deep-python"]["utilized"] is False
    assert by_agent["deep-ts"]["utilized"] is True
    assert by_agent["deep-rust"]["utilized"] is False
    assert by_agent["deep-c"]["utilized"] is True
    assert result["reviewers_utilizing_exploration"] == 2


def test_grounding_rate_is_undefined_with_zero_findings():
    """A review that produced NO findings has an undefined grounding rate.

    Reporting 1.0 here would hand a review that found nothing a perfect
    grounding score, which flows into the manifest and becomes a top RL
    reward. ``None`` is the only honest value for 0/0.
    """
    trajectories = {"main": None, "forked": [_read_traj("deep-python.json", "/repo/api.py")]}

    result = analyze_grounding(trajectories, [])

    assert result["total_findings"] == 0
    assert result["grounded_count"] == 0
    assert result["ungrounded_count"] == 0
    assert result["grounding_rate"] is None


# --- cross-backend read extraction (issue #307) ---


CODEX_READ_COMMAND = (
    "sed -n '1,240p' .daydream/exploration/summary.md && "
    "sed -n '1,280p' .daydream/diff.patch; "
    "rg -n -C 3 'cache_write_tokens|total_cache_write_tokens' "
    "core/osprey-cli docs README.md 2>/dev/null && "
    "cat some/file.rb; nl -ba pkg/services/cleanup.go"
)


def test_files_read_extracts_codex_shell_paths():
    calls = [{"function_name": "shell", "arguments": {"command": CODEX_READ_COMMAND}}]

    paths = _files_read(calls)

    assert ".daydream/exploration/summary.md" in paths
    assert ".daydream/diff.patch" in paths
    assert "core/osprey-cli" in paths
    assert "docs" in paths
    assert "README.md" in paths
    assert "some/file.rb" in paths
    assert "pkg/services/cleanup.go" in paths
    assert "1,240p" not in paths
    assert "1,280p" not in paths
    assert "cache_write_tokens|total_cache_write_tokens" not in paths
    assert "2>/dev/null" not in paths


def test_files_read_extracts_pi_read_and_bash_paths():
    calls = [
        {"function_name": "read", "arguments": {"path": "/repo/pkg/services/cleanup.go"}},
        {"function_name": "bash", "arguments": {"command": "cat README.md && nl -ba core/osprey-cli"}},
    ]

    paths = _files_read(calls)

    assert "/repo/pkg/services/cleanup.go" in paths
    assert "README.md" in paths
    assert "core/osprey-cli" in paths


def _shell_reads(command: str) -> set[str]:
    """Extract read paths from a single codex ``shell`` call."""
    return _files_read([{"function_name": "shell", "arguments": {"command": command}}])


def test_files_read_skips_separated_redirect_target():
    paths = _shell_reads("cat source.txt > target.py")

    assert "source.txt" in paths
    assert "target.py" not in paths


def test_files_read_skips_redirect_and_pattern_for_rg():
    paths = _shell_reads("rg -n 'pat' a.py b.py 2>/dev/null")

    assert "a.py" in paths
    assert "b.py" in paths
    assert "/dev/null" not in paths
    assert "pat" not in paths


def test_files_read_preserves_quoted_paths_with_spaces():
    assert "my file.py" in _shell_reads("cat 'my file.py'")
    assert "my file.py" in _shell_reads('cat "my file.py"')


def test_files_read_skips_rg_option_values():
    paths = _shell_reads("rg -C 3 --glob '*.py' 'needle' src/app.py")

    assert paths == {"src/app.py"}
    assert "3" not in paths
    assert "*.py" not in paths
    assert "needle" not in paths


def test_files_read_claude_read_and_grep_unchanged():
    calls = [
        {"function_name": "Read", "arguments": {"file_path": "/repo/api.py"}},
        {"function_name": "Grep", "arguments": {"path": "src/"}},
    ]

    paths = _files_read(calls)

    assert paths == {"/repo/api.py", "src/"}


# --- unbalanced-quote tokenization (issue #327) ---


# Only a quote opened and never closed makes shlex raise; the two other
# former candidates had balanced/escaped quotes and tokenize cleanly.


def test_tokenize_command_never_raises_on_unbalanced_quotes() -> None:
    # must not raise
    tokens = _tokenize_command("rg -l '\"unclosed")
    assert isinstance(tokens, list)


def test_tokenize_command_whitespace_fallback_keeps_recoverable_paths() -> None:
    # the unclosed quote makes shlex raise mid-token; the whitespace-split
    # fallback still surfaces the ``a.py`` operand preceding it
    paths = _shell_reads("rg -n 'pat' a.py '\"unclosed")
    assert "a.py" in paths


def test_analyze_coverage_counts_codex_and_pi_reads(tmp_path: Path):
    daydream_dir = tmp_path / ".daydream"
    daydream_dir.mkdir()
    (daydream_dir / "diff.patch").write_text(
        "diff --git a/pkg/services/cleanup.go b/pkg/services/cleanup.go\n"
        "diff --git a/core/osprey-cli b/core/osprey-cli\n"
    )
    trajectories = {
        "main": {
            "_source_file": "trajectory.json",
            "steps": [
                {
                    "step_id": "m0",
                    "extra": {"daydream_phase": "deep"},
                    "tool_calls": [
                        {"function_name": "shell", "arguments": {"command": "sed -n '1,240p' .daydream/diff.patch"}}
                    ],
                }
            ],
        },
        "forked": [
            {
                "_source_file": "deep-python.json",
                "steps": [
                    {
                        "step_id": "s0",
                        "tool_calls": [
                            {"function_name": "read", "arguments": {"path": "/repo/pkg/services/cleanup.go"}},
                            {"function_name": "bash", "arguments": {"command": "cat core/osprey-cli"}},
                        ],
                    }
                ],
            }
        ],
    }

    result = analyze_coverage(trajectories, daydream_dir)

    assert result["coverage_ratio"] == 1.0
    assert result["files_read_by_reviewers"] == 2
    assert result["uncovered_files"] == []


def test_analyze_grounding_counts_codex_and_pi_reads():
    trajectories = {
        "main": None,
        "forked": [
            {
                "_source_file": "deep-python.json",
                "steps": [
                    {
                        "step_id": "s0",
                        "tool_calls": [
                            {"function_name": "shell", "arguments": {"command": "cat /repo/api.py"}},
                            {"function_name": "read", "arguments": {"path": "/repo/api.py"}},
                        ],
                    }
                ],
            }
        ],
    }
    findings = [
        {
            "id": "py-1",
            "_stack": "python",
            "file": "api.py",
            "rationale": "Read api.py; flag the missing validation.",
            "confidence": "HIGH",
        }
    ]

    result = analyze_grounding(trajectories, findings)

    entry = result["grounded"][0]
    assert entry["file_was_read"] is True
    assert result["grounded_count"] == 1
    assert result["ungrounded_count"] == 0
    assert result["grounding_rate"] == 1.0


# --- quality metrics (issue #316) ---


def _quality_workspace(tmp_path: Path, files: dict[str, str], name: str = "workspace") -> Path:
    """Create a temp workspace with the given ``{relative_path: content}`` files."""
    ws = tmp_path / name
    ws.mkdir(parents=True)
    for rel, content in files.items():
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return ws


def _big_function(max_x: int) -> str:
    """A single-function if/elif chain reaching ``max_x``.

    ``big`` has ``max_x + 1`` cyclomatic complexity (1 + the ``if`` + every
    ``elif``) and ``2 * max_x + 2`` sloc lines.
    """
    lines = ["def big(x):", "    if x == 1:", "        return 1"]
    for i in range(2, max_x + 1):
        lines.append(f"    elif x == {i}:")
        lines.append(f"        return {i}")
    lines.append("    return 0")
    return "\n".join(lines) + "\n"


def _mass(cc: int, sloc: int) -> float:
    return cc * math.sqrt(sloc)


def test_quality_erosion_computes_cc_mass_share(tmp_path: Path):
    """Pooled erosion is the high-CC mass share, hand-computed from the file."""
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def small(x):\n    return x * 2\n\n" + _big_function(11)},
    )

    result = analyze_quality(ws / ".daydream")

    small_mass = _mass(1, 2)
    big_mass = _mass(12, 24)
    expected = round(big_mass / (small_mass + big_mass), 4)
    assert result["erosion"] == pytest.approx(expected)
    entry = result["per_file"]["app.py"]
    assert entry["erosion"] == pytest.approx(expected)
    assert entry["functions"] == 2
    assert entry["high_cc_functions"] == 1


def test_quality_erosion_zero_when_no_high_cc(tmp_path: Path):
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def one(x):\n    return x + 1\n\ndef two(x, y):\n    return x + y\n"},
    )

    result = analyze_quality(ws / ".daydream")

    assert result["erosion"] == 0.0
    assert result["per_file"]["app.py"]["erosion"] == 0.0
    assert result["per_file"]["app.py"]["high_cc_functions"] == 0


def test_quality_verbosity_flags_identity_comprehension(tmp_path: Path):
    ws = _quality_workspace(tmp_path, {"app.py": "def f(items):\n    return [x for x in items]\n"})

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["verbosity"] > 0
    assert entry["verbosity"] == pytest.approx(1 / 2)


def test_quality_verbosity_flags_empty_list_guard(tmp_path: Path):
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def process(items):\n"
                "    for x in items:\n"
                "        if len(items) == 0:\n"
                "            return None\n"
                "        print(x)\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["verbosity"] > 0
    assert entry["verbosity"] == pytest.approx(2 / 5)


def test_quality_verbosity_flags_single_use_variable(tmp_path: Path):
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def compute(x):\n"
                "    intermediate = x + 1\n"
                "    return intermediate * 2\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["verbosity"] > 0
    assert entry["verbosity"] == pytest.approx(round(1 / 3, 4))


def test_quality_verbosity_flags_trivial_wrapper(tmp_path: Path):
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def inner(x, y):\n"
                "    return x + y\n"
                "\n"
                "def outer(x, y):\n"
                "    return inner(x, y)\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["verbosity"] > 0
    assert entry["verbosity"] == pytest.approx(2 / 4)


def test_quality_verbosity_flags_nested_ladder(tmp_path: Path):
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def f(a, b, c):\n"
                "    if a:\n"
                "        if b:\n"
                "            if c:\n"
                "                return 1\n"
                "    return 0\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["verbosity"] > 0
    assert entry["verbosity"] == pytest.approx(round(2 / 6, 4))


def test_quality_verbosity_flags_clone_block(tmp_path: Path):
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def a():\n"
                "    if x > 1:\n"
                "        return 1\n"
                "    return 0\n"
                "\n"
                "def b():\n"
                "    if x > 1:\n"
                "        return 1\n"
                "    return 0\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["verbosity"] > 0
    assert entry["verbosity"] == pytest.approx(6 / 8)


def test_quality_per_file_keyed_by_relative_path(tmp_path: Path):
    ws = _quality_workspace(tmp_path, {"pkg/mod.py": "def f(x):\n    return x\n"})

    result = analyze_quality(ws / ".daydream")

    assert "pkg/mod.py" in result["per_file"]
    assert result["per_file"]["pkg/mod.py"]["functions"] == 1


def test_quality_returns_none_when_no_python_files(tmp_path: Path):
    ws = _quality_workspace(tmp_path, {"README.md": "# nothing here\n"})

    result = analyze_quality(ws / ".daydream")

    assert result["erosion"] is None
    assert result["verbosity"] is None
    assert result["scoped_files"] == 0
    assert result["per_file"] == {}
    assert result["calibration"] == {
        "human_verbosity": 0.19,
        "human_erosion": 0.34,
        "paper": "arXiv:2603.24755",
    }


def test_quality_excludes_vendored_and_internal_dirs(tmp_path: Path):
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": "def f(x):\n    return x\n",
            ".daydream/deep/fixture.py": "def g(x):\n    return x\n",
            "node_modules/pkg/index.py": "def h(x):\n    return x\n",
            "sub/venv/lib/x.py": "def i(x):\n    return x\n",
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert result["scoped_files"] == 1
    assert list(result["per_file"]) == ["app.py"]


def test_quality_monotone_across_eroding_fix(tmp_path: Path):
    """An eroding fix to an already-large function raises erosion (verbosity holds)."""
    clean_ws = _quality_workspace(
        tmp_path,
        {"app.py": "def small(x):\n    return x * 2\n\n" + _big_function(11)},
        name="clean",
    )
    eroded_ws = _quality_workspace(
        tmp_path,
        {"app.py": "def small(x):\n    return x * 2\n\n" + _big_function(13)},
        name="eroded",
    )

    clean = analyze_quality(clean_ws / ".daydream")
    eroded = analyze_quality(eroded_ws / ".daydream")

    assert eroded["erosion"] > clean["erosion"]
    assert eroded["verbosity"] >= clean["verbosity"]


def test_analyze_session_includes_quality_for_post_fix_workspace(tmp_path: Path):
    """Real-path: analyze_session computes quality on the live workspace tree."""
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def small(x):\n    return x * 2\n\n" + _big_function(11)},
    )
    daydream_dir = ws / ".daydream"
    run_dir = daydream_dir / "runs" / "quality-real"
    run_dir.mkdir(parents=True)
    (run_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "quality-real",
                "agent": {"name": "daydream", "model_name": "claude-sonnet-4-5"},
                "steps": [],
            }
        )
    )

    result = analyze_session(daydream_dir, session_id="quality-real")

    quality = result["quality"]
    assert set(quality) == {"erosion", "verbosity", "per_file", "calibration", "scoped_files"}
    assert quality["scoped_files"] == 1
    assert quality["calibration"]["human_erosion"] == 0.34
    assert quality["calibration"]["human_verbosity"] == 0.19
    entry = quality["per_file"]["app.py"]
    assert entry["functions"] == 2
    assert entry["high_cc_functions"] == 1
    expected = round(_mass(12, 24) / (_mass(1, 2) + _mass(12, 24)), 4)
    assert quality["erosion"] == pytest.approx(expected)


# --- review round 1 fix regressions (#316) ---


def test_quality_verbosity_stays_within_zero_one_when_spans_include_blank_lines(tmp_path: Path):
    """Blank rows inside a flagged span must not count toward the ratio.

    A trivial wrapper's span covers the whole function, blank lines included;
    previously ``verbosity`` divided those rows by non-blank LOC and could
    exceed 1.0, corrupting the per-file and workspace aggregates.
    """
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def outer(x, y):\n"
                "\n"
                "\n"
                "    return inner(x, y)\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert 0.0 <= entry["verbosity"] <= 1.0
    assert entry["verbosity"] == pytest.approx(1.0)


def test_quality_erosion_counts_comprehension_filters_toward_cc(tmp_path: Path):
    """A comprehension's generators and filters are real branch paths.

    A function whose only decision points live inside a comprehension must
    cross the CC>10 erosion threshold; previously they were invisible.
    """
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def f(xs):\n"
                "    return [x for x in xs "
                + " ".join(f"if x != {i}" for i in range(10))
                + "]\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["high_cc_functions"] == 1
    assert entry["erosion"] == 1.0


def test_quality_erosion_ignores_wildcard_match_case(tmp_path: Path):
    """``case _:`` matches any value and adds no decision path."""
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def f(x):\n"
                "    match x:\n"
                "        case _:\n"
                "            return 0\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["high_cc_functions"] == 0
    assert result["erosion"] == 0.0


def test_quality_erosion_counts_real_match_cases_toward_cc(tmp_path: Path):
    """Each real ``case <value>:`` adds a decision path; 11 cross the threshold."""
    lines = ["def f(x):", "    match x:"]
    for i in range(1, 12):
        lines.append(f"        case {i}:")
        lines.append(f"            return {i}")
    ws = _quality_workspace(tmp_path, {"app.py": "\n".join(lines) + "\n"})

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["high_cc_functions"] == 1
    assert entry["erosion"] == 1.0


def test_quality_verbosity_filtered_comprehension_not_flagged(tmp_path: Path):
    """``[x for x in items if x > 0]`` filters, so it is not an identity."""
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def f(items):\n    return [x for x in items if x > 0]\n"},
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] == 0.0


def test_quality_verbosity_multi_generator_comprehension_not_flagged(tmp_path: Path):
    """``[x for x in a for y in b]`` is a product, not a passthrough."""
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def f(a, b):\n    return [x for x in a for y in b]\n"},
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] == 0.0


def test_quality_verbosity_while_predicate_guard_not_flagged(tmp_path: Path):
    """A predicate merely receiving the collection does not prove it nonempty."""
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def f(items):\n"
                "    while should_continue(items):\n"
                "        if not items:\n"
                "            break\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] == 0.0


def test_quality_verbosity_while_bare_collection_guard_flagged(tmp_path: Path):
    """``while items:`` proves nonemptiness, so the guard is redundant."""
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def f(items):\n"
                "    while items:\n"
                "        if not items:\n"
                "            break\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["verbosity"] > 0
    assert entry["verbosity"] == pytest.approx(2 / 4)


def test_quality_verbosity_while_len_comparison_guard_flagged(tmp_path: Path):
    """``while len(items) > 0:`` proves nonemptiness, so the guard is redundant."""
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def f(items):\n"
                "    while len(items) > 0:\n"
                "        if not items:\n"
                "            break\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["verbosity"] > 0
    assert entry["verbosity"] == pytest.approx(2 / 4)


def test_quality_verbosity_trivial_wrapper_with_literal_not_flagged(tmp_path: Path):
    """``g(x, 42)`` supplies a literal, so the wrapper is not a pure passthrough."""
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def f(x):\n    return g(x, 42)\n"},
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] == 0.0


def test_quality_verbosity_trivial_wrapper_with_keyword_arg_not_flagged(tmp_path: Path):
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def f(x):\n    return g(x=x)\n"},
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] == 0.0


def test_quality_verbosity_trivial_wrapper_with_starred_args_not_flagged(tmp_path: Path):
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def f(*xs):\n    return g(*xs)\n"},
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] == 0.0


def test_quality_verbosity_trivial_wrapper_typed_param_still_flagged(tmp_path: Path):
    """A type annotation adds no behavior, so ``def f(x: int): return g(x)`` is a wrapper."""
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def f(x: int):\n    return g(x)\n"},
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] == 1.0


def test_quality_verbosity_trivial_wrapper_with_default_not_flagged(tmp_path: Path):
    """A default supplies behavior, so ``def f(x=1): return g(x)`` is not a wrapper."""
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def f(x=1):\n    return g(x)\n"},
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] == 0.0


# --- review round 2 fix regressions (#316) ---


_TEN_COMPREHENSION_FILTERS = " ".join(f"if x != {i}" for i in range(10))


def test_quality_verbosity_flags_clones_across_files(tmp_path: Path):
    """An exact block copied from ``a.py`` into ``b.py`` flags BOTH files.

    Per-file clone detection alone never sees the duplicate — each per-file
    invocation observes a single occurrence. The cross-file pass must index
    blocks across scoped files and attribute them back.
    """
    block = "    if x > 1:\n        return 1\n    return 0\n"
    ws = _quality_workspace(
        tmp_path,
        {"a.py": "def a():\n" + block, "b.py": "def b():\n" + block},
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["a.py"]["verbosity"] > 0
    assert result["per_file"]["b.py"]["verbosity"] > 0
    assert result["verbosity"] > 0


def test_quality_candidate_scope_indexes_valid_peers_for_clones(
    tmp_path: Path,
) -> None:
    """Candidate mode indexes valid peer text for cross-file clone attribution."""
    block = "    if x > 1:\n        return 1\n    return 0\n"
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": "def a():\n" + block,
            "peer.py": "def b():\n" + block,  # clone source, NOT a candidate
            "other.py": "def c():\n    return 3\n",  # neither candidate nor peer source
        },
    )

    result = analyze_quality(ws / ".daydream", candidate_paths={"app.py"})

    assert result["scoped_files"] == 1
    assert set(result["per_file"]) == {"app.py"}
    assert result["per_file"]["app.py"]["verbosity"] > 0  # clone from peer.py indexed
    assert result["erosion"] is not None


def test_quality_candidate_none_preserves_whole_workspace_result(tmp_path: Path) -> None:
    """candidate_paths=None is byte-identical to the default whole-workspace call."""
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def a():\n    return 1\n", "b.py": "def b(y):\n    return y * 2\n"},
    )
    default = analyze_quality(ws / ".daydream")
    explicit_none = analyze_quality(ws / ".daydream", candidate_paths=None)
    assert explicit_none == default


def test_quality_candidate_empty_set_returns_empty_without_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly empty candidate set reports zero files without walking the workspace."""
    from daydream.eval import analyzer as analyzer_mod

    ws = _quality_workspace(tmp_path, {"app.py": "def a():\n    return 1\n"})

    def _boom(_workspace: Path):
        raise AssertionError("workspace must not be enumerated for an empty candidate set")

    monkeypatch.setattr(analyzer_mod, "_scoped_python_files", _boom)
    result = analyze_quality(ws / ".daydream", candidate_paths=set())

    assert result["scoped_files"] == 0
    assert result["per_file"] == {}
    assert result["erosion"] is None
    assert result["verbosity"] is None


def test_quality_candidate_ineligible_path_not_reported(tmp_path: Path) -> None:
    """A candidate that fails the generated-file eligibility rule is not reported."""
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": "def a():\n    return 1\n",
            "schema_generated.py": "def x():\n    return 1\n",  # *_generated.py glob excludes it
        },
    )
    result = analyze_quality(ws / ".daydream", candidate_paths={"schema_generated.py", "app.py"})
    assert result["scoped_files"] == 1
    assert set(result["per_file"]) == {"app.py"}


def test_quality_verbosity_cross_file_clone_needs_two_files(tmp_path: Path):
    """A block present in only one file flags neither file."""
    ws = _quality_workspace(
        tmp_path,
        {
            "a.py": "def a():\n    if x > 1:\n        return 1\n    return 0\n",
            "b.py": "def b(y):\n    return y * 2\n",
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["a.py"]["verbosity"] == 0.0
    assert result["per_file"]["b.py"]["verbosity"] == 0.0


def test_quality_verbosity_within_file_clones_still_count_across_pass(tmp_path: Path):
    """Within-file duplicates keep counting now that the cross-file pass exists."""
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def a():\n"
                "    if x > 1:\n"
                "        return 1\n"
                "    return 0\n"
                "\n"
                "def b():\n"
                "    if x > 1:\n"
                "        return 1\n"
                "    return 0\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] == pytest.approx(6 / 8)


def test_quality_erosion_generator_expression_filters_count_toward_cc(tmp_path: Path):
    """Generator-expression filters are real branch paths, like list comprehensions."""
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def f(xs):\n"
                "    return (x for x in xs "
                + _TEN_COMPREHENSION_FILTERS
                + ")\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["high_cc_functions"] == 1
    assert entry["erosion"] == 1.0


@pytest.mark.parametrize(
    ("comprehension", "label"),
    [
        (f"[x for x in xs {_TEN_COMPREHENSION_FILTERS}]", "list"),
        (f"{{x for x in xs {_TEN_COMPREHENSION_FILTERS}}}", "set"),
        (f"{{x: x for x in xs {_TEN_COMPREHENSION_FILTERS}}}", "dict"),
        (f"(x for x in xs {_TEN_COMPREHENSION_FILTERS})", "generator"),
    ],
)
def test_quality_erosion_comprehension_types_cc_parity(
    tmp_path: Path, comprehension: str, label: str
):
    """List/set/dict/generator comprehensions count generators + filters identically."""
    ws = _quality_workspace(tmp_path, {"app.py": f"def f(xs):\n    return {comprehension}\n"})

    result = analyze_quality(ws / ".daydream")

    entry = result["per_file"]["app.py"]
    assert entry["high_cc_functions"] == 1, label
    assert entry["erosion"] == 1.0


def test_quality_verbosity_unfiltered_generator_expression_is_identity(tmp_path: Path):
    """``(x for x in items)`` is an identity comprehension, exactly like a list one."""
    ws = _quality_workspace(tmp_path, {"app.py": "def f(items):\n    return (x for x in items)\n"})

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] > 0


def test_quality_excludes_generated_and_vendored_files(tmp_path: Path):
    """Generated and vendored Python is out of metric scope (Finding #8).

    Path-based generated files (``*_generated.py``, ``*.pb.py``,
    ``migrations/*.py``), the generated-file header marker, and vendored
    trees (``vendor``/``third_party``) must not reach ``per_file``,
    ``scoped_files``, or the aggregate denominators.
    """
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": "def f(x):\n    return x\n",
            "api_generated.py": "def g(x):\n    return x\n",
            "svc.pb.py": "def h(x):\n    return x\n",
            "vendor/lib/v.py": "def i(x):\n    return x\n",
            "third_party/lib/t.py": "def j(x):\n    return x\n",
            "migrations/0001_init.py": "def k(x):\n    return x\n",
            "gen_tool.py": "# Code generated by protoc. DO NOT EDIT.\ndef m(x):\n    return x\n",
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert list(result["per_file"]) == ["app.py"]
    assert result["scoped_files"] == 1


def test_quality_syntax_error_file_excluded_from_aggregates(tmp_path: Path):
    """A malformed file stays in scoped_files but not per_file or the ratios."""
    ws = _quality_workspace(
        tmp_path,
        {
            "good.py": "def f(x):\n    return x\n",
            "broken.py": "def broken(:\n    return 1\n",
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert result["scoped_files"] == 2
    assert list(result["per_file"]) == ["good.py"]
    assert result["verbosity"] == 0.0


# --- review round 3 fix regressions (#316) ---


def test_quality_unparseable_file_does_not_contaminate_cross_file_clones(tmp_path: Path):
    """A malformed file's lines must never flag matching blocks in valid files.

    ``analyze_quality`` indexes the cross-file clone pass over successfully
    parsed files only. A broken file holding a block that also appears in a
    valid file would otherwise be a second occurrence and flag the valid
    file, shifting its verbosity (Finding #1).
    """
    block = "    if x > 1:\n        return 1\n    return 0\n"
    clean_ws = _quality_workspace(tmp_path, {"app.py": "def a():\n" + block}, name="clean")
    dirty_ws = _quality_workspace(
        tmp_path,
        {"app.py": "def a():\n" + block, "broken.py": "def broken(:\n" + block},
        name="dirty",
    )

    clean = analyze_quality(clean_ws / ".daydream")
    dirty = analyze_quality(dirty_ws / ".daydream")

    assert dirty["scoped_files"] == 2
    assert list(dirty["per_file"]) == ["app.py"]
    assert clean["per_file"]["app.py"]["verbosity"] == dirty["per_file"]["app.py"]["verbosity"]
    assert clean["verbosity"] == dirty["verbosity"]


def test_quality_candidate_malformed_peer_does_not_contaminate_cross_file_clones(
    tmp_path: Path,
) -> None:
    """Finding #1 holds in candidate mode: a malformed peer never flags a valid candidate.

    Candidate mode indexes scoped peers as clone sources, but a peer that fails to
    parse must be excluded from the cross-file clone index so its garbage lines
    cannot inflate a valid candidate's verbosity (regression #457 round 2).
    """
    block = "    if x > 1:\n        return 1\n    return 0\n"
    clean_ws = _quality_workspace(tmp_path, {"app.py": "def a():\n" + block}, name="clean")
    dirty_ws = _quality_workspace(
        tmp_path,
        {"app.py": "def a():\n" + block, "broken.py": "def broken(:\n" + block},
        name="dirty",
    )

    clean = analyze_quality(clean_ws / ".daydream", candidate_paths={"app.py"})
    dirty = analyze_quality(dirty_ws / ".daydream", candidate_paths={"app.py"})

    assert set(dirty["per_file"]) == {"app.py"}
    assert clean["per_file"]["app.py"]["verbosity"] == dirty["per_file"]["app.py"]["verbosity"]
    assert clean["verbosity"] == dirty["verbosity"]


def test_quality_per_file_erosion_none_without_functions(tmp_path: Path):
    """A module with no functions has no mass, so its erosion ratio is None.

    Zero is a meaningful value (no high-CC mass), so undefined must be None;
    the workspace aggregate pools mass across files and stays numeric.
    """
    ws = _quality_workspace(
        tmp_path,
        {
            "m.py": "import os\nX = 1\n",
            "app.py": "def f(x):\n    return x\n",
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["m.py"]["erosion"] is None
    assert result["per_file"]["m.py"]["functions"] == 0
    assert result["per_file"]["app.py"]["erosion"] == 0.0
    assert result["erosion"] == 0.0


def test_quality_per_file_verbosity_none_on_blank_only_file(tmp_path: Path):
    """A file with no non-blank lines has an undefined verbosity ratio.

    ``None`` signals the undefined denominator; the workspace aggregate keeps
    pooling the zero lines harmlessly and stays ``None`` for verbosity too.
    """
    ws = _quality_workspace(tmp_path, {"blank.py": "\n\n\n"})

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["blank.py"]["verbosity"] is None
    assert result["per_file"]["blank.py"]["sloc"] == 0
    assert result["verbosity"] is None


@pytest.mark.parametrize(
    ("mutation", "label"),
    [
        ("        item = items.pop()\n", "pop"),
        ("        items.clear()\n", "clear"),
    ],
)
def test_quality_verbosity_guard_after_mutation_not_flagged(
    tmp_path: Path, mutation: str, label: str
):
    """A post-mutation empty check is a necessary termination guard.

    ``while items:`` proves nonemptiness only at the header; once the body
    mutates the collection, ``if not items:`` is meaningful and must not be
    counted as redundant slop (Finding #3).
    """
    ws = _quality_workspace(
        tmp_path,
        {"app.py": "def f(items):\n    while items:\n" + mutation + "        if not items:\n            break\n"},
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] == 0.0, label


def test_quality_verbosity_guard_without_prior_mutation_still_flagged(tmp_path: Path):
    """A non-mutating statement between header and guard keeps it redundant.

    ``while items: x = f(); if not items: break`` — nothing touches ``items``,
    so the header's nonemptiness proof still holds at the guard, which stays
    flagged (Finding #3).
    """
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def f(items):\n"
                "    while items:\n"
                "        x = f()\n"
                "        if not items:\n"
                "            break\n"
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] > 0


@pytest.mark.parametrize(
    ("wrapper_body", "label"),
    [
        ('    """Pass through."""\n    return inner(x, y)\n', "documented"),
        ("    return inner(x, y)\n", "undocumented"),
    ],
)
def test_quality_verbosity_wrapper_docstring_does_not_hide_wrapper(
    tmp_path: Path, wrapper_body: str, label: str
):
    """A leading docstring is not an executable statement for wrapper detection.

    ``_trivial_wrapper`` must count only the real body statement, so a
    documented pass-through wrapper is flagged exactly like an undocumented
    one (Finding #4).
    """
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": (
                "def inner(x, y):\n"
                "    return x + y\n"
                "\n"
                "def outer(x, y):\n"
                + wrapper_body
            )
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert result["per_file"]["app.py"]["verbosity"] > 0, label


def test_quality_excludes_explicitly_vendored_subtree(tmp_path: Path):
    """A vendored dir whose basename is not vendor/third_party is still excluded.

    ``daydream/atif/`` is explicitly vendored from Harbor (see
    daydream/atif/NOTICE) but its basename ``atif`` is not in the obvious
    vendored set — it must not reach ``per_file`` or the aggregates
    (Finding #5).
    """
    ws = _quality_workspace(
        tmp_path,
        {
            "app.py": "def f(x):\n    return x\n",
            "daydream/atif/models.py": "def g(x):\n    return x\n",
            "daydream/atif/validator.py": "def h(x):\n    return x\n",
        },
    )

    result = analyze_quality(ws / ".daydream")

    assert result["scoped_files"] == 1
    assert list(result["per_file"]) == ["app.py"]
    assert result["erosion"] == 0.0
