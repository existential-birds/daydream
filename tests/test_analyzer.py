"""Trajectory resolution, finding/location arithmetic, and source-quality metrics."""
import json
import math
import uuid
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from daydream import _tree_sitter_safety as safety
from daydream.artifact_visibility import ArtifactEvidenceProvenance
from daydream.backends import MetricsEvent, ResultEvent, TextEvent
from daydream.deep.records import RECORD_SOURCE_UIDS_KEY, mint_record_uid
from daydream.eval import quality as quality_mod
from daydream.eval.analyzer import (
    _agent_label,
    _latest_main_trajectory,
    analyze_costs,
    analyze_findings,
    analyze_location,
    analyze_quality,
    analyze_session,
    analyze_shipped_duplication,
    analyze_timing,
    analyze_tools,
    analyze_training_signals,
    collect_trajectory_paths,
)
from daydream.eval.quality import _quality_python_parser
from daydream.trajectory import (
    RUN_DOCUMENT_NAME,
    DaydreamPhase,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    run_directory,
    run_document_path,
    sibling_document_path,
    snapshot_trajectories,
)
from tests.harness.trajectory import make_recorder, trajectory_payload


def _snapshot(root: dict[str, Any], *forks: dict[str, Any], cutoff_at: str = "2026-01-01T00:00:10Z",
    status: Literal["complete", "partial"] = "complete",
) -> RunWriteSnapshot:
    """Prepare explicitly identified fixture documents without reading trajectory files."""
    return RunWriteSnapshot(status=status, cutoff_at=cutoff_at, root_trajectory_id=root["trajectory_id"],
        documents=tuple(TrajectoryDocumentSnapshot(payload["trajectory_id"], Path(payload["_source_file"]),
                json.dumps({key: value for key, value in payload.items() if key != "_source_file"}).encode(),
            ) for payload in (root, *forks)),
    )


def test_analyze_timing_prefers_root_lifecycle_over_misleading_steps() -> None:
    trajectories: dict[str, Any] = {"main": {"session_id": "session", "trajectory_id": "session",
            "steps": [{"timestamp": "2026-01-01T00:00:40Z"}, {"timestamp": "2026-01-01T00:00:41Z"}],
            "extra": {"run_started_at": "2026-01-01T00:00:00Z", "run_ended_at": "2026-01-01T00:00:10Z"},
            "_source_file": "trajectory.json",
        }, "forked": [{"session_id": "session", "trajectory_id": "child",
                "steps": [{"timestamp": "2026-01-01T00:01:00Z"}, {"timestamp": "2026-01-01T00:01:02Z"}], "extra": {},
                "_source_file": "child.json",
            }
        ],
    }

    snapshot = _snapshot(trajectories["main"], *trajectories["forked"], cutoff_at="2026-01-01T00:00:10Z")
    assert analyze_timing(snapshot_trajectories(snapshot), snapshot)["total_wall_clock_seconds"] == 10.0

def test_analyze_timing_without_lifecycle_is_unavailable() -> None:
    snapshot = _snapshot({"session_id": "session", "trajectory_id": "session",
        "_source_file": "trajectory.json", "extra": {},
        "steps": [{"timestamp": "2026-01-01T00:00:02Z"}, {"timestamp": "2026-01-01T00:00:08Z"}],
    }, cutoff_at="2026-01-01T00:00:10Z")
    snapshot.validate("session")

    timing = analyze_timing(snapshot_trajectories(snapshot), snapshot)

    assert timing == {"total_wall_clock_seconds": None,
        "by_agent": [{"agent": "main", "duration_seconds": 6.0}],
    }


@pytest.mark.parametrize(("status", "cutoff_at", "wall"), [
    ("partial", "2026-01-01T00:00:04Z", 4.0),
    ("complete", "2026-01-01T00:00:10Z", 10.0),
    ("partial", "2026-01-01T00:00:10Z", None),
    ("complete", "2026-01-01T00:00:04Z", None),
])
def test_evaluation_uses_frozen_bytes_status_and_cutoff(tmp_path: Path,
    status: Literal["complete", "partial"], cutoff_at: str, wall: float | None,
) -> None:
    root: dict[str, Any] = {"session_id": "frozen", "trajectory_id": "frozen", "_source_file": "trajectory.json",
        "steps": [{"timestamp": "2026-01-01T00:00:02Z"}, {"timestamp": "2026-01-01T00:00:03Z"}],
        "final_metrics": {"total_cost_usd": 0.5},
        "extra": {"partial": False, "run_started_at": "2026-01-01T00:00:00Z",
            "run_ended_at": "2026-01-01T00:00:10Z", "snapshot_at": "2026-01-01T00:00:04Z"},
    }
    snapshot = _snapshot(root, status=status, cutoff_at=cutoff_at)
    snapshot.validate("frozen")
    live_path = tmp_path / ".daydream" / "runs" / "frozen" / "trajectory.json"
    live_path.parent.mkdir(parents=True)
    live_path.write_bytes(snapshot.documents[0].json_bytes)
    root["final_metrics"]["total_cost_usd"] = 99.0
    root["extra"]["run_ended_at"] = "2026-01-01T00:01:00Z"
    live_path.write_text(json.dumps(root))

    result = analyze_session(tmp_path / ".daydream", write_snapshot=snapshot)

    assert result["cost"]["total_cost_usd"] == 0.5
    assert result["timing"]["total_wall_clock_seconds"] == wall
    assert result["timing"]["by_agent"] == [{"agent": "main", "duration_seconds": 1.0}]


SESSION = "11111111-2222-3333-4444-555555555555"

def test_analyzer_resolution_keys_off_the_owned_names(tmp_path: Path) -> None:
    daydream_dir = tmp_path / ".daydream"
    run_dir = run_directory(daydream_dir, SESSION)
    run_document_path(run_dir).parent.mkdir(parents=True)
    run_document_path(run_dir).write_bytes(trajectory_payload(SESSION))
    sibling = sibling_document_path(run_dir, "deep-python.json")
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(trajectory_payload("fork-1"))

    assert [p.name for p in collect_trajectory_paths(run_dir)] == [RUN_DOCUMENT_NAME, "deep-python.json"]
    assert [p.name for p in collect_trajectory_paths(daydream_dir)] == [RUN_DOCUMENT_NAME, "deep-python.json"]
    assert _latest_main_trajectory(daydream_dir) == run_document_path(run_dir)

def test_analyze_costs_preserves_fractional_aggregate_precision() -> None:
    trajectories = {
        "main": {"_source_file": "trajectory.json", "final_metrics": {"total_cost_usd": 0.00006}}, "forked": [],
    }

    result = analyze_costs(trajectories)

    assert result["total_cost_usd"] == 0.00006
    assert sum(agent["cost_usd"] for agent in result["by_agent"]) == result["total_cost_usd"]

@pytest.mark.parametrize(("name", "expected"),
    [("Write", "write"), ("write", "write"), ("Edit", "write"), ("edit", "write"), ("MultiEdit", "write"),
        ("multiedit", "write"), ("NotebookEdit", "write"), ("notebookedit", "write"), ("patch", "write"),
        ("apply_patch", "write"), ("Read", "read"), ("read", "read"), ("shell", "other"), ("bash", "other"),
        ("custom", "other"),
    ],
)
def test_write_tool_counts_are_backend_neutral(name: str, expected: str) -> None:
    result = analyze_tools({"main": None,
        "forked": [{"_source_file": "deep-python.json", "steps": [{
            "step_id": 1, "tool_calls": [{"function_name": name, "arguments": {}}],
        }]}],
    })
    assert result["by_type"] == {name: 1}
    assert result["write_ratio"] == (1.0 if expected == "write" else 0.0)

def test_analyze_tools_uses_semantic_writes_and_preserves_raw_names() -> None:
    shell_calls = [{"function_name": "shell", "arguments": {"command": f"echo {i}"}} for i in range(311)]
    patch_calls = [{"function_name": "patch", "arguments": {"patch": f"change {i}"}} for i in range(15)]
    trajectories = {"main": None,
        "forked": [{"_source_file": "deep-python.json",
                "steps": [{"step_id": 1, "tool_calls": [*shell_calls, *patch_calls]}],
            }
        ],
    }

    result = analyze_tools(trajectories)

    assert result["total_calls"] == 326
    assert result["by_type"] == {"shell": 311, "patch": 15}
    assert result["by_agent"] == {"deep-python": {"shell": 311, "patch": 15}}
    assert result["write_ratio"] == 0.046


def _training_flags(steps: list[dict[str, Any]]) -> list[str]:
    result = analyze_training_signals({"main": None, "forked": [{"_source_file": "deep-python.json", "steps": steps}]},
    )
    return cast(list[str], result["trajectories"][0]["noise_flags"])

@pytest.mark.parametrize(("result_extra", "expected"),
    [({"is_error": True}, "failed_tool_result"), ({"status": "interrupted"}, "incomplete_tool_call"),
        ({"cancelled": True}, "incomplete_tool_call"),
    ],
)
def test_training_flags_linked_tool_failures(result_extra: dict[str, Any], expected: str) -> None:
    steps = [{"step_id": 1, "tool_calls": [{"tool_call_id": "call-1", "function_name": "Read", "arguments": {}}],
            "observation": {"results": [{"source_call_id": "call-1", "content": "output", "extra": result_extra}]},
        }
    ]
    assert _training_flags(steps) == [expected]

def test_training_flags_unpaired_calls_and_null_interrupted_marker_once() -> None:
    steps = [{"step_id": 1, "tool_calls": [{"tool_call_id": "call-1", "function_name": "Read", "arguments": {}}],
            "observation": {"results": [{"source_call_id": None, "content": "interrupted",
                        "extra": {"is_error": True, "status": "interrupted"},
                    }
                ]
            },
        }
    ]
    assert _training_flags(steps) == ["incomplete_tool_call"]

def test_training_correlations_are_step_local_and_detect_unmatched_results() -> None:
    steps = [{"step_id": 1, "tool_calls": [{"tool_call_id": "same", "function_name": "Read", "arguments": {}}]},
        {"step_id": 2,
            "observation": {"results": [{"source_call_id": "same", "content": "late", "extra": {"is_error": False}}]
            }, "extra": {"unmatched_tool_results": ["another"]},
        },
    ]
    assert _training_flags(steps) == ["incomplete_tool_call", "unmatched_tool_result"]

def test_training_diagnostics_map_only_recognized_codes_in_fixed_order() -> None:
    steps = [{"step_id": 1,
            "extra": {"backend_diagnostics": [{"code": "codex_parser_coverage"}, {"code": "arbitrary_backend_note"},
                    {"code": "codex_transport_coverage"}, {"code": "codex_parser_coverage"},
                ]
            },
        }
    ]
    assert _training_flags(steps) == ["incomplete_telemetry", "parser_coverage_gap"]

@pytest.mark.parametrize("legacy_extra", [None, {}, {"is_error": "true"}, {"cancelled": 1}, []])
def test_clean_training_tolerates_missing_or_malformed_legacy_result_metadata(legacy_extra: Any,) -> None:
    result: dict[str, Any] = {"source_call_id": "call-1", "content": "ok"}
    if legacy_extra is not None:
        result["extra"] = legacy_extra
    steps = [{"step_id": 1, "tool_calls": [{"tool_call_id": "call-1", "function_name": "read", "arguments": {}}],
            "observation": {"results": [result]},
        }
    ]
    assert _training_flags(steps) == []

def test_analyze_costs_includes_cached_tokens_when_prompt_dominates() -> None:
    trajectories = {"main": {
            "_source_file": "trajectory.json", "final_metrics": {"total_prompt_tokens": 140, "total_cached_tokens": 14},
        }, "forked": [],
    }

    result = analyze_costs(trajectories)

    assert result["total_input_tokens"] == 140
    assert result["cache_hit_rate"] == 0.1

def test_analyze_costs_aggregates_legacy_fork_metrics() -> None:
    trajectories = {"main": {"_source_file": "trajectory.json", "final_metrics": {"total_cost_usd": 1.0}},
        "forked": [{"_source_file": "fork.json", "final_metrics": {"total_cost_usd": 0.5}}],
    }

    result = analyze_costs(trajectories)

    assert result["total_cost_usd"] == 1.5

async def test_analyze_costs_assigns_nested_forks_their_own_metrics(tmp_path: Path,) -> None:
    session = "nested-forks"
    daydream_dir = tmp_path / ".daydream"
    snapshots: list[RunWriteSnapshot] = []
    recorder = make_recorder(
        tmp_path, on_write=lambda _rec, snapshot: snapshots.append(snapshot),
        path=daydream_dir / "runs" / session / "trajectory.json",
        agent_model_name="opus", session_id=session,
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

    trajectories = snapshot_trajectories(snapshots[-1])

    result = analyze_costs(trajectories)

    by_agent = {agent["agent"]: agent for agent in result["by_agent"]}
    assert by_agent["main"]["cost_usd"] == pytest.approx(0.1)
    assert by_agent["outer"]["cost_usd"] == pytest.approx(0.2)
    assert by_agent["inner"]["cost_usd"] == pytest.approx(0.3)
    assert by_agent["main"]["steps"] == 1
    assert by_agent["outer"]["steps"] == 1
    assert by_agent["inner"]["steps"] == 1
    assert sum(agent["cost_usd"] for agent in result["by_agent"]) == pytest.approx(result["total_cost_usd"])
    assert sum(agent["steps"] for agent in result["by_agent"]) == 3


def _read_traj(source_file: str, *read_paths: str, pi_style: bool = False) -> dict[str, Any]:
    """Build a fork with Read calls; pi_style uses read/arguments.path instead."""
    steps = []
    for i, path in enumerate(read_paths):
        if pi_style:
            tc = {"function_name": "read", "arguments": {"path": path}}
        else:
            tc = {"function_name": "Read", "arguments": {"file_path": path}}
        tc["tool_call_id"] = f"read-{i}"
        steps.append({"step_id": f"s{i}", "tool_calls": [tc],
            "observation": {"results": [{"source_call_id": f"read-{i}", "content": "source"}]},
        })
    return {"_source_file": source_file, "steps": steps}


def _artifact_provenance(
    *, public_source: Path, private_base: Path, workspace_key: str = "workspace-key", session_id: str = "session-id",
) -> Any:
    """Construct exact current-owner provenance without assuming a default root."""

    live = private_base / workspace_key / "runs" / session_id / "live"
    return ArtifactEvidenceProvenance(
        workspace_key=workspace_key, session_id=session_id, public_source=public_source, live_root=live,
    )


def _owned_source(tmp_path: Path, **kwargs: Any) -> tuple[Path, Path, Any]:
    """(public source, its ``.daydream`` dir, current-owner provenance) for one run."""
    public_source = tmp_path / "source"
    provenance = _artifact_provenance(public_source=public_source, private_base=tmp_path / "private", **kwargs)
    return public_source, public_source / ".daydream", provenance


def _deep_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create and return ``(daydream_dir, deep_dir)`` under ``tmp_path``."""
    dd = tmp_path / ".daydream"
    deep = dd / "deep"
    deep.mkdir(parents=True)
    return dd, deep


def _seed_diff(daydream_dir: Path, *files: str) -> None:
    """Write a ``diff.patch`` naming exactly *files* as the reviewed diff."""
    daydream_dir.mkdir(parents=True, exist_ok=True)
    (daydream_dir / "diff.patch").write_text(
        "".join(f"diff --git a/{path} b/{path}\n" for path in files), encoding="utf-8"
    )


# Other workspaces and other sessions under the same key remain eligible repository reads.


def _write_records(deep: Path, **overrides: Any) -> None:
    """Write one per-stack python record, defaulted to a grounded src/api.py claim."""
    record: dict[str, Any] = {
        "id": "py-1", "file": "src/api.py", "line": 1, "confidence": "HIGH", "rationale": "src/api.py needs a guard",
    }
    record.update(overrides)
    (deep / "stack-python-records.json").write_text(json.dumps([record]), encoding="utf-8")


def _root_trajectory(session_id: str) -> dict[str, Any]:
    """A minimal ATIF root trajectory ``analyze_session`` accepts as frozen input."""
    return {"_source_file": "trajectory.json", "session_id": session_id, "trajectory_id": session_id,
        "agent": {"name": "daydream", "model_name": "test"}, "steps": [], "final_metrics": {},
    }


_READ_TOOL_SHAPES = {"claude": ("Read", "file_path"), "pi": ("read", "path"), "osprey": ("read", "path")}

def test_analyze_session_preserves_source_quality_without_read_metrics(tmp_path: Path,) -> None:
    public_source, _, provenance = _owned_source(tmp_path, session_id="session")
    (public_source / "src").mkdir(parents=True)
    (public_source / "src/api.py").write_text("def api(value):\n    return value\n", encoding="utf-8")
    # Frozen evaluation input lives apart from the post-fix source workspace.
    daydream_dir = tmp_path / "frozen" / ".daydream"
    deep = daydream_dir / "deep"
    deep.mkdir(parents=True)
    _seed_diff(daydream_dir, "src/api.py")
    private_live = provenance.live_root
    artifact_ref = str(private_live / ".daydream/deep/stack-python-review.md")
    exploration_ref = str(private_live / ".daydream/exploration/summary.md")
    _write_records(deep, rationale=f"Evidence came from {artifact_ref}")
    trajectories: dict[str, Any] = {"main": _root_trajectory("session"),
        "forked": [{**_read_traj("deep-python.json", str(public_source / "src/api.py"), artifact_ref, exploration_ref,),
            "session_id": "session", "trajectory_id": "deep-python"}],
    }

    result = analyze_session(
        daydream_dir, write_snapshot=_snapshot(trajectories["main"], *trajectories["forked"]),
        artifact_provenance=provenance,
        code_workspace=public_source,
    )

    assert not {"coverage", "grounding", "exploration_utilization"} & result.keys()
    assert result["tools"]["total_calls"] == 3


def _quality_workspace(tmp_path: Path, files: dict[str, str], name: str = "workspace") -> Path:
    """Create a temp workspace with the given ``{relative_path: content}`` files."""
    ws = tmp_path / name
    ws.mkdir(parents=True)
    for rel, content in files.items():
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return ws


def _quality(tmp_path: Path, files: dict[str, str], *, name: str = "workspace", **kwargs: Any,) -> dict[str, Any]:
    """Build a workspace with ``files`` and analyze its ``.daydream`` quality."""
    return analyze_quality(_quality_workspace(tmp_path, files, name) / ".daydream", **kwargs)


def _big_function(max_x: int) -> str:
    """Build an if/elif chain with max_x+1 complexity and 2*max_x+2 SLOC."""
    lines = ["def big(x):", "    if x == 1:", "        return 1"]
    for i in range(2, max_x + 1):
        lines.append(f"    elif x == {i}:")
        lines.append(f"        return {i}")
    lines.append("    return 0")
    return "\n".join(lines) + "\n"


def _mass(cc: int, sloc: int) -> float:
    return cc * math.sqrt(sloc)

def test_quality_erosion_computes_cc_mass_share(tmp_path: Path) -> None:
    """Pooled erosion is the high-CC mass share, hand-computed from the file."""
    result = _quality(tmp_path, {"app.py": "def small(x):\n    return x * 2\n\n" + _big_function(11)},)

    small_mass = _mass(1, 2)
    big_mass = _mass(12, 24)
    expected = round(big_mass / (small_mass + big_mass), 4)
    assert result["erosion"] == pytest.approx(expected)
    entry = result["per_file"]["app.py"]
    assert entry["erosion"] == pytest.approx(expected)
    assert entry["functions"] == 2
    assert entry["high_cc_functions"] == 1

def test_quality_erosion_zero_when_no_high_cc(tmp_path: Path) -> None:
    result = _quality(tmp_path, {"app.py": "def one(x):\n    return x + 1\n\ndef two(x, y):\n    return x + y\n"},)

    assert result["erosion"] == 0.0
    assert result["per_file"]["app.py"]["erosion"] == 0.0
    assert result["per_file"]["app.py"]["high_cc_functions"] == 0

@pytest.mark.parametrize(("source", "expected"),
    [
        pytest.param("def f(items):\n    return [x for x in items]\n", 1 / 2, id="identity-comprehension"),
        pytest.param(
            "def process(items):\n    for x in items:\n        if len(items) == 0:\n"
            "            return None\n        print(x)\n",
            2 / 5, id="empty-list-guard",
        ),
        pytest.param(
            "def compute(x):\n    intermediate = x + 1\n    return intermediate * 2\n",
            round(1 / 3, 4), id="single-use-variable",
        ),
        pytest.param(
            "def inner(x, y):\n    return x + y\n\ndef outer(x, y):\n    return inner(x, y)\n",
            2 / 4, id="trivial-wrapper",
        ),
        pytest.param(
            "def f(a, b, c):\n    if a:\n        if b:\n            if c:\n"
            "                return 1\n    return 0\n",
            round(2 / 6, 4), id="nested-ladder",
        ),
        pytest.param(
            "def a():\n    if x > 1:\n        return 1\n    return 0\n\n"
            "def b():\n    if x > 1:\n        return 1\n    return 0\n",
            6 / 8, id="clone-block",
        ),
        pytest.param("def f(items):\n    return [x for x in items if x > 0]\n", 0.0, id="filtered-comprehension",),
        pytest.param(
            "def f(a, b):\n    return [x for x in a for y in b]\n",
            0.0, id="multi-generator-comprehension",
        ),
        pytest.param(
            "def f(items):\n    while should_continue(items):\n"
            "        if not items:\n            break\n",
            0.0, id="predicate-guard",
        ),
        pytest.param(
            "def f(items):\n    while items:\n        if not items:\n            break\n",
            2 / 4, id="bare-collection-guard",
        ),
        pytest.param(
            "def f(items):\n    while len(items) > 0:\n        if not items:\n"
            "            break\n",
            2 / 4, id="len-comparison-guard",
        ),
    ],
)
def test_quality_verbosity_detects_redundancy(tmp_path: Path, source: str, expected: float) -> None:
    entry = _quality(tmp_path, {"app.py": source})["per_file"]["app.py"]
    if expected:
        assert entry["verbosity"] > 0
    assert entry["verbosity"] == pytest.approx(expected)

def test_quality_per_file_keyed_by_relative_path(tmp_path: Path) -> None:
    result = _quality(tmp_path, {"pkg/mod.py": "def f(x):\n    return x\n"})

    assert "pkg/mod.py" in result["per_file"]
    assert result["per_file"]["pkg/mod.py"]["functions"] == 1

def test_quality_returns_none_when_no_python_files(tmp_path: Path) -> None:
    result = _quality(tmp_path, {"README.md": "# nothing here\n"})

    assert result["erosion"] is None
    assert result["verbosity"] is None
    assert result["scoped_files"] == 0
    assert result["per_file"] == {}
    assert result["calibration"] == {"human_verbosity": 0.19, "human_erosion": 0.34, "paper": "arXiv:2603.24755"}

def test_quality_excludes_vendored_and_internal_dirs(tmp_path: Path) -> None:
    result = _quality(tmp_path,
        {
            "app.py": "def f(x):\n    return x\n",
            ".daydream/deep/fixture.py": "def g(x):\n    return x\n",
            "node_modules/pkg/index.py": "def h(x):\n    return x\n",
            "sub/venv/lib/x.py": "def i(x):\n    return x\n",
        },
    )

    assert result["scoped_files"] == 1
    assert list(result["per_file"]) == ["app.py"]

def test_quality_monotone_across_eroding_fix(tmp_path: Path) -> None:
    clean = _quality(tmp_path, {"app.py": "def small(x):\n    return x * 2\n\n" + _big_function(11)}, name="clean",)
    eroded = _quality(tmp_path, {"app.py": "def small(x):\n    return x * 2\n\n" + _big_function(13)}, name="eroded",)

    assert eroded["erosion"] > clean["erosion"]
    assert eroded["verbosity"] >= clean["verbosity"]

def test_analyze_session_includes_quality_for_post_fix_workspace(tmp_path: Path,) -> None:
    ws = _quality_workspace(tmp_path, {"app.py": "def small(x):\n    return x * 2\n\n" + _big_function(11)},)
    daydream_dir = ws / ".daydream"
    snapshot = run_snapshot("quality-real", schema_version="ATIF-v1.6", model_name="claude-sonnet-4-5",)

    result = analyze_session(daydream_dir, write_snapshot=snapshot)

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

def test_analyze_session_reads_quality_from_explicit_code_workspace(tmp_path: Path,) -> None:
    """Frozen artifact inputs and post-fix source quality use distinct roots."""
    daydream_dir = tmp_path / "frozen" / ".daydream"
    snapshot = run_snapshot("quality-split", schema_version="ATIF-v1.6", model_name="test",)
    code_workspace = _quality_workspace(tmp_path,
        {"app.py": "def changed(x):\n    return x + 1\n"},
        name="operational",
    )
    public_source = tmp_path / "public-source"
    provenance = _artifact_provenance(
        public_source=public_source, private_base=tmp_path / "private", workspace_key="workspace",
        session_id="quality-split",
    )

    result = analyze_session(
        daydream_dir, write_snapshot=snapshot, artifact_provenance=provenance, code_workspace=code_workspace,
    )

    assert result["quality"]["scoped_files"] == 1
    assert list(result["quality"]["per_file"]) == ["app.py"]
    assert result["daydream_dir"] == str(public_source / ".daydream")

def test_quality_verbosity_stays_within_zero_one_when_spans_include_blank_lines(tmp_path: Path,) -> None:
    result = _quality(tmp_path, {"app.py": ("def outer(x, y):\n" "\n" "\n" "    return inner(x, y)\n")},)

    entry = result["per_file"]["app.py"]
    assert 0.0 <= entry["verbosity"] <= 1.0
    assert entry["verbosity"] == pytest.approx(1.0)

def test_quality_erosion_ignores_wildcard_match_case(tmp_path: Path) -> None:
    """``case _:`` matches any value and adds no decision path."""
    result = _quality(tmp_path,
        {"app.py": ("def f(x):\n" "    match x:\n" "        case _:\n" "            return 0\n")},
    )

    assert result["per_file"]["app.py"]["high_cc_functions"] == 0
    assert result["erosion"] == 0.0

def test_quality_erosion_counts_real_match_cases_toward_cc(tmp_path: Path) -> None:
    """Each real ``case <value>:`` adds a decision path; 11 cross the threshold."""
    lines = ["def f(x):", "    match x:"]
    for i in range(1, 12):
        lines.append(f"        case {i}:")
        lines.append(f"            return {i}")
    result = _quality(tmp_path, {"app.py": "\n".join(lines) + "\n"})

    entry = result["per_file"]["app.py"]
    assert entry["high_cc_functions"] == 1
    assert entry["erosion"] == 1.0

@pytest.mark.parametrize(("source", "verbosity"),
    [
        ("def f(x):\n    return g(x, 42)\n", 0.0),  # literal argument
        ("def f(x):\n    return g(x=x)\n", 0.0),  # keyword argument
        ("def f(*xs):\n    return g(*xs)\n", 0.0),  # starred argument
        ("def f(x: int):\n    return g(x)\n", 1.0),  # annotation adds no behavior
        ("def f(x=1):\n    return g(x)\n", 0.0),  # default adds behavior
    ],
)
def test_quality_verbosity_wrapper_cases(tmp_path: Path, source: str, verbosity: float) -> None:
    result = _quality(tmp_path, {"app.py": source})
    assert result["per_file"]["app.py"]["verbosity"] == verbosity


_TEN_COMPREHENSION_FILTERS = " ".join(f"if x != {i}" for i in range(10))

def test_quality_verbosity_flags_clones_across_files(tmp_path: Path) -> None:
    block = "    if x > 1:\n        return 1\n    return 0\n"
    result = _quality(tmp_path, {"a.py": "def a():\n" + block, "b.py": "def b():\n" + block},)

    assert result["per_file"]["a.py"]["verbosity"] > 0
    assert result["per_file"]["b.py"]["verbosity"] > 0
    assert result["verbosity"] > 0

def test_quality_candidate_scope_indexes_valid_peers_for_clones(tmp_path: Path,) -> None:
    block = "    if x > 1:\n        return 1\n    return 0\n"
    result = _quality(tmp_path,
        {
            "app.py": "def a():\n" + block,
            "peer.py": "def b():\n" + block,  # clone source, NOT a candidate
            "other.py": "def c():\n    return 3\n",  # neither candidate nor peer source
        }, candidate_paths={"app.py"},
    )

    assert result["scoped_files"] == 1
    assert set(result["per_file"]) == {"app.py"}
    assert result["per_file"]["app.py"]["verbosity"] > 0  # clone from peer.py indexed
    assert result["erosion"] is not None

def test_quality_candidate_none_preserves_whole_workspace_result(tmp_path: Path,) -> None:
    ws = _quality_workspace(tmp_path, {"app.py": "def a():\n    return 1\n", "b.py": "def b(y):\n    return y * 2\n"},)
    default = analyze_quality(ws / ".daydream")
    explicit_none = analyze_quality(ws / ".daydream", candidate_paths=None)
    assert explicit_none == default

def test_quality_candidate_empty_set_returns_empty_without_enumeration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_workspace: Path) -> None:
        raise AssertionError("workspace must not be enumerated for an empty candidate set")

    monkeypatch.setattr(quality_mod, "_scoped_python_files", _boom)
    result = _quality(tmp_path, {"app.py": "def a():\n    return 1\n"}, candidate_paths=set())

    assert result["scoped_files"] == 0
    assert result["per_file"] == {}
    assert result["erosion"] is None
    assert result["verbosity"] is None

def test_quality_candidate_ineligible_path_not_reported(tmp_path: Path) -> None:
    result = _quality(tmp_path,
        {
            "app.py": "def a():\n    return 1\n",
            "schema_generated.py": "def x():\n    return 1\n",  # *_generated.py glob excludes it
        }, candidate_paths={"schema_generated.py", "app.py"},
    )
    assert result["scoped_files"] == 1
    assert set(result["per_file"]) == {"app.py"}

def test_quality_verbosity_cross_file_clone_needs_two_files(tmp_path: Path) -> None:
    result = _quality(tmp_path,
        {
            "a.py": "def a():\n    if x > 1:\n        return 1\n    return 0\n",
            "b.py": "def b(y):\n    return y * 2\n",
        },
    )

    assert result["per_file"]["a.py"]["verbosity"] == 0.0
    assert result["per_file"]["b.py"]["verbosity"] == 0.0

def test_quality_verbosity_within_file_clones_still_count_across_pass(tmp_path: Path,) -> None:
    result = _quality(tmp_path,
        {"app.py": (
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

    assert result["per_file"]["app.py"]["verbosity"] == pytest.approx(6 / 8)

@pytest.mark.parametrize(("comprehension", "label"),
    [(f"[x for x in xs {_TEN_COMPREHENSION_FILTERS}]", "list"),
        (f"{{x for x in xs {_TEN_COMPREHENSION_FILTERS}}}", "set"),
        (f"{{x: x for x in xs {_TEN_COMPREHENSION_FILTERS}}}", "dict"),
        (f"(x for x in xs {_TEN_COMPREHENSION_FILTERS})", "generator"),
    ],
)
def test_quality_erosion_comprehension_types_cc_parity(tmp_path: Path, comprehension: str, label: str) -> None:
    """Comprehension generators and filters must all contribute to the CC>10 threshold."""
    result = _quality(tmp_path, {"app.py": f"def f(xs):\n    return {comprehension}\n"})

    entry = result["per_file"]["app.py"]
    assert entry["high_cc_functions"] == 1, label
    assert entry["erosion"] == 1.0

def test_quality_verbosity_unfiltered_generator_expression_is_identity(tmp_path: Path,) -> None:
    result = _quality(tmp_path, {"app.py": "def f(items):\n    return (x for x in items)\n"})

    assert result["per_file"]["app.py"]["verbosity"] > 0

def test_quality_excludes_generated_and_vendored_files(tmp_path: Path) -> None:
    """Generated paths/headers and vendored trees must be absent from files and denominators."""
    result = _quality(tmp_path,
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

    assert list(result["per_file"]) == ["app.py"]
    assert result["scoped_files"] == 1

def test_quality_syntax_error_file_excluded_from_aggregates(tmp_path: Path) -> None:
    """A malformed file stays in scoped_files but not per_file or the ratios."""
    result = _quality(tmp_path, {"good.py": "def f(x):\n    return x\n", "broken.py": "def broken(:\n    return 1\n"},)

    assert result["scoped_files"] == 2
    assert list(result["per_file"]) == ["good.py"]
    assert result["verbosity"] == 0.0

def test_quality_unparseable_file_does_not_contaminate_cross_file_clones(tmp_path: Path,) -> None:
    block = "    if x > 1:\n        return 1\n    return 0\n"
    clean = _quality(tmp_path, {"app.py": "def a():\n" + block}, name="clean")
    dirty = _quality(tmp_path, {"app.py": "def a():\n" + block, "broken.py": "def broken(:\n" + block}, name="dirty",)

    assert dirty["scoped_files"] == 2
    assert list(dirty["per_file"]) == ["app.py"]
    assert clean["per_file"]["app.py"]["verbosity"] == dirty["per_file"]["app.py"]["verbosity"]
    assert clean["verbosity"] == dirty["verbosity"]

def test_quality_candidate_malformed_peer_does_not_contaminate_cross_file_clones(tmp_path: Path,) -> None:
    block = "    if x > 1:\n        return 1\n    return 0\n"
    clean = _quality(tmp_path, {"app.py": "def a():\n" + block}, name="clean", candidate_paths={"app.py"})
    dirty = _quality(tmp_path,
        {"app.py": "def a():\n" + block, "broken.py": "def broken(:\n" + block},
        name="dirty", candidate_paths={"app.py"},
    )

    assert set(dirty["per_file"]) == {"app.py"}
    assert clean["per_file"]["app.py"]["verbosity"] == dirty["per_file"]["app.py"]["verbosity"]
    assert clean["verbosity"] == dirty["verbosity"]

def test_quality_per_file_erosion_none_without_functions(tmp_path: Path) -> None:
    """No function mass means undefined erosion, distinct from zero high-complexity mass."""
    result = _quality(tmp_path, {"m.py": "import os\nX = 1\n", "app.py": "def f(x):\n    return x\n"},)

    assert result["per_file"]["m.py"]["erosion"] is None
    assert result["per_file"]["m.py"]["functions"] == 0
    assert result["per_file"]["app.py"]["erosion"] == 0.0
    assert result["erosion"] == 0.0

def test_quality_per_file_verbosity_none_on_blank_only_file(tmp_path: Path) -> None:
    """A zero nonblank denominator remains undefined in file and workspace verbosity."""
    result = _quality(tmp_path, {"blank.py": "\n\n\n"})

    assert result["per_file"]["blank.py"]["verbosity"] is None
    assert result["per_file"]["blank.py"]["sloc"] == 0
    assert result["verbosity"] is None

@pytest.mark.parametrize(("mutation", "label"),
    [("        item = items.pop()\n", "pop"), ("        items.clear()\n", "clear")],
)
def test_quality_verbosity_guard_after_mutation_not_flagged(tmp_path: Path, mutation: str, label: str) -> None:
    """Mutation invalidates the loop-header nonempty proof, so its termination guard is needed."""
    result = _quality(tmp_path,
        {"app.py": "def f(items):\n    while items:\n" + mutation + "        if not items:\n            break\n"},
    )

    assert result["per_file"]["app.py"]["verbosity"] == 0.0, label

def test_quality_verbosity_guard_without_prior_mutation_still_flagged(tmp_path: Path,) -> None:
    """An intervening non-mutating statement leaves the loop-header nonempty proof valid."""
    result = _quality(tmp_path,
        {"app.py": (
                "def f(items):\n"
                "    while items:\n"
                "        x = f()\n"
                "        if not items:\n"
                "            break\n"
            )
        },
    )

    assert result["per_file"]["app.py"]["verbosity"] > 0

@pytest.mark.parametrize(("wrapper_body", "label"),
    [
        ('    """Pass through."""\n    return inner(x, y)\n', "documented"),
        ("    return inner(x, y)\n", "undocumented"),
    ],
)
def test_quality_verbosity_wrapper_docstring_does_not_hide_wrapper(tmp_path: Path, wrapper_body: str, label: str,
) -> None:
    result = _quality(tmp_path,
        {"app.py": ("def inner(x, y):\n" "    return x + y\n" "\n" "def outer(x, y):\n" + wrapper_body)},
    )

    assert result["per_file"]["app.py"]["verbosity"] > 0, label

def test_quality_excludes_explicitly_vendored_subtree(tmp_path: Path) -> None:
    """The explicit atif vendor exclusion must work without a conventional vendor basename."""
    result = _quality(tmp_path,
        {
            "app.py": "def f(x):\n    return x\n",
            "daydream/atif/models.py": "def g(x):\n    return x\n",
            "daydream/atif/validator.py": "def h(x):\n    return x\n",
        },
    )

    assert result["scoped_files"] == 1
    assert list(result["per_file"]) == ["app.py"]
    assert result["erosion"] == 0.0


# Count authoritative merged items, falling back to review text, then pre-merge stack totals.

def seed_shipped_items(deep: Path, *, high: int, med: int) -> None:
    """Write deep/\"merged-items.json\" = {\"items\": [high+med schema-valid items]}."""
    items = [_item(i, file="a.py", line=i + 1, description=f"high-{i}", confidence="HIGH", severity="high")
        for i in range(high)
    ]
    items += [_item(high + i, file="b.py", line=i + 1, description=f"med-{i}") for i in range(med)]
    seed_merged_items(deep, items)


def seed_stack_records(deep: Path, stack_name: str, *, n: int) -> None:
    """Seed HIGH records with host UIDs using 1-based birth ordinals.

    Reviewer ids deliberately remain 0-based: display ids are not record identity."""
    records: list[dict[str, Any]] = [{"id": i, "confidence": "HIGH", "uid": mint_record_uid(stack_name, i + 1)}
        for i in range(n)
    ]
    (deep / f"stack-{stack_name}-records.json").write_text(json.dumps({"issues": records}))


def seed_review_output(deep: Path, *, count: int) -> None:
    """Write deep/\"review-output.md\" with `count` lines matching ^\\d+\\.\\s+\\[."""
    lines = [f"{i}. [HIGH] finding {i}" for i in range(1, count + 1)]
    (deep / "review-output.md").write_text("\n".join(lines) + "\n")

def test_shipped_count_wins_over_per_stack_records(tmp_path: Path) -> None:
    dd, deep = _deep_dirs(tmp_path)
    seed_shipped_items(deep, high=4, med=4)     # merged-items.json: 8 items (4 HIGH, 4 MEDIUM)
    seed_stack_records(deep, "python", n=4)     # stack-python-records.json: 4 HIGH
    out = analyze_findings(dd)
    assert out["total"] == 8
    assert out["by_confidence"] == {"HIGH": 4, "MEDIUM": 4}

def test_shipped_count_includes_wonder_lens_items(tmp_path: Path) -> None:
    # Wonder findings belong to the shipped denominator too; excluding them inflates cost per finding.
    dd, deep = _deep_dirs(tmp_path)
    seed_shipped_items(deep, high=4, med=0)
    shipped = json.loads((deep / "merged-items.json").read_text())["items"]
    shipped += [_item(4 + i, file="w.py", line=i + 1, description=f"wonder-{i}", lens="wonder") for i in range(4)]
    seed_merged_items(deep, shipped)
    out = analyze_findings(dd)
    assert out["total"] == 8                     # wonder items are counted (issue #741)
    assert out["by_confidence"] == {"HIGH": 4, "MEDIUM": 4}




def test_shipped_count_falls_back_to_regex_when_merged_items_absent(tmp_path: Path,) -> None:
    dd, deep = _deep_dirs(tmp_path)
    seed_review_output(deep, count=8)           # review-output.md: 8 numbered [ items
    seed_stack_records(deep, "python", n=4)
    out = analyze_findings(dd)
    assert out["total"] == 8                    # from merged_finding_count regex, not per-stack

def test_shipped_count_never_zero_without_artifacts(tmp_path: Path) -> None:
    dd, deep = _deep_dirs(tmp_path)
    seed_stack_records(deep, "python", n=4)
    out = analyze_findings(dd)
    assert out["total"] == 4                    # pre-merge fallback, never 0

def test_per_lens_attribution_reads_alternatives_and_stack_buckets(tmp_path: Path,) -> None:
    dd, deep = _deep_dirs(tmp_path)
    (deep / "alternatives.json").write_text(json.dumps([{"id": 1}, {"id": 2}]))
    seed_stack_records(deep, "python", n=3)
    seed_stack_records(deep, "uncovered", n=1)
    seed_stack_records(deep, "structure", n=2)
    out = analyze_findings(dd)
    assert all(stack["name"] != "uncovered" for stack in out["stacks"])
    assert out["per_lens"] == {"wonder": 2, "per-stack": 3, "structure": 2}

def test_per_lens_malformed_alternatives_does_not_crash(tmp_path: Path) -> None:
    dd, deep = _deep_dirs(tmp_path)
    (deep / "alternatives.json").write_text("{not json")
    out = analyze_findings(dd)
    assert out["per_lens"]["wonder"] == 0

def test_per_lens_wonder_only_run_reports_nonzero_wonder(tmp_path: Path) -> None:
    dd, deep = _deep_dirs(tmp_path)
    (deep / "alternatives.json").write_text(json.dumps([{"id": i} for i in range(6)]))
    out = analyze_findings(dd)
    assert out["per_lens"]["wonder"] == 6
    assert out["per_lens"]["per-stack"] == 0


def run_snapshot(
    session_id: str, *, total_cost_usd: float | None = None, schema_version: str = "ATIF-v1.7",
    model_name: str | None = None, steps: list[dict[str, Any]] | None = None,
) -> RunWriteSnapshot:
    """Prepare immutable run bytes with the requested fields."""
    agent: dict[str, Any] = {"name": "test"}
    if model_name is not None:
        agent["model_name"] = model_name
    document: dict[str, Any] = {"schema_version": schema_version, "session_id": session_id, "trajectory_id": session_id,
        "_source_file": "trajectory.json", "agent": agent,
        "steps": [] if steps is None else steps, "extra": {},
    }
    if total_cost_usd is not None:
        document["final_metrics"] = {"total_cost_usd": total_cost_usd}
    return _snapshot(document)

def test_analyze_session_shipped_metrics_match_a80b9373(tmp_path: Path) -> None:
    sid = "a80b9373-56d6-4062-9ab5-4c75e475ab67"
    dd = tmp_path / ".daydream"
    snapshot = run_snapshot(sid, total_cost_usd=18.2056)
    deep = dd / "deep"
    deep.mkdir(parents=True)
    seed_shipped_items(deep, high=4, med=4)     # 8 shipped items, the a80b9373 shape
    (deep / "alternatives.json").write_text(json.dumps([{"id": i} for i in range(6)]))
    res = analyze_session(dd, write_snapshot=snapshot)
    assert res["findings"]["total"] == 8
    assert res["findings"]["by_confidence"] == {"HIGH": 4, "MEDIUM": 4}
    assert res["findings"]["per_lens"]["wonder"] == 6
    assert res["derived"]["cost_per_finding_usd"] == pytest.approx(18.2056 / 8, rel=1e-4)

def test_analyze_quality_refuses_known_bad_tree_sitter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """#1087: on a known-bad tree-sitter install the analyzer refuses native
    analysis by raising the typed guard error — the orchestrator's fail-open
    wrapper converts that into (None, reason); it must not silently skip.
    """

    monkeypatch.setattr(safety, "installed_tree_sitter_version", lambda: "0.26.0")
    # Clear the parser cache so the bad-version guard sees the patched installation.
    _quality_python_parser.cache_clear()
    with pytest.raises(safety.TreeSitterBadVersionError):
        _quality(tmp_path, {"mod.py": "def f():\n    return 1\n"})

def test_analyze_quality_unchanged_on_good_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """#1087 (M5): the guard is a no-op on valid installs — behavior identical
    to pre-regression, including the parser cache being consulted.
    """

    monkeypatch.setattr(safety, "installed_tree_sitter_version", lambda: "0.25.2")
    result = _quality(tmp_path, {"mod.py": "def f():\n    return 1\n"})
    assert result["scoped_files"] == 1
    entry = result["per_file"]["mod.py"]
    assert entry["functions"] == 1
    assert entry["sloc"] > 0

def test_analyze_session_degrades_quality_on_known_bad_tree_sitter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#1087: a known-bad install degrades only the quality section of
    analyze_session -- the rest of the evaluation (and evaluation.json)
    survives instead of the typed escape dropping the whole run.
    """

    monkeypatch.setattr(safety, "installed_tree_sitter_version", lambda: "0.26.0")
    ws = _quality_workspace(tmp_path, {"app.py": "def small(x):\n    return x * 2\n\n" + _big_function(11)},)
    daydream_dir = ws / ".daydream"
    snapshot = run_snapshot("quality-bad", schema_version="ATIF-v1.6", model_name="claude-sonnet-4-5",)

    result = analyze_session(daydream_dir, write_snapshot=snapshot)

    quality = result["quality"]
    assert quality["unavailable"] is True
    assert quality["error"]
    assert quality["per_file"] == {}
    assert quality["scoped_files"] == 0
    assert result["session_id"] == "quality-bad"
    assert result["trajectory_count"] == 1


# Seed the exact production artifact shapes for location and duplication analysis.

WORKED_A = "New helper duplicates the existing loader"
WORKED_B = "Config reading is implemented twice in this module"


def seed_diff_patch(dd: Path, file: str = "svc/loader.py", *, start: int = 85, count: int = 8) -> None:
    """Write one hunk with inclusive new-side range [start, start+count-1]."""
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "diff.patch").write_text(
        f"diff --git a/{file} b/{file}\n"
        f"--- a/{file}\n"
        f"+++ b/{file}\n"
        f"@@ -{start},{count} +{start},{count} @@ def load():\n"
        "-old\n"
        "+new\n"
    )


def seed_hunk_index(dd: Path, ranges: dict[str, list[tuple[int, int]]]) -> None:
    """Write ``.daydream/hunk-index.json`` in the persisted production shape."""
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "hunk-index.json").write_text(json.dumps({path: {"hunks": [{
                            "old_start": start, "old_end": end, "new_start": start, "new_end": end, "added": 1,
                            "removed": 1,
                        }
                        for start, end in file_ranges
                    ], "added_total": len(file_ranges), "removed_total": len(file_ranges),
                }
                for path, file_ranges in ranges.items()
            }
        )
    )


def seed_merged_items(deep: Path, items: list[dict[str, Any]]) -> None:
    """Write ``deep/merged-items.json`` from an explicit item list."""
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "merged-items.json").write_text(json.dumps({"items": items}))


def seed_dedup_candidates(deep: Path, *, record_alt_pairs: list[dict[str, Any]] | None = None,
    record_duplicate_pairs: list[dict[str, Any]] | None = None,
) -> None:
    """Write production-shaped pre-merge candidate pairs, which are inputs rather than escapes."""
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "dedup-candidates.json").write_text(json.dumps(
            {"record_alt_pairs": record_alt_pairs or [], "record_duplicate_pairs": record_duplicate_pairs or []}
        )
    )


def _provenance(*uids: str) -> dict[str, Any]:
    """Provide a typed source_uids keyword payload for unpacking into _item.

    No arguments means explicit attribution refusal, distinct from an absent key."""
    return {RECORD_SOURCE_UIDS_KEY: list(uids)}


def _item(
    item_id: int, *, file: str = "svc/loader.py", line: Any = 88, description: str = WORKED_A, lens: str = "per-stack",
    **extra: Any,
) -> dict[str, Any]:
    """A shipped ``merged-items.json`` item in the canonical shape."""
    item: dict[str, Any] = {
        "id": item_id, "file": file, "line": line, "lens": lens, "severity": "medium", "confidence": "MEDIUM",
        "description": description, "rationale": f"see {file}", "evidence": f"{file}:{line}",
    }
    item.update(extra)
    return item


def _worked_example_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """``(.daydream, .daydream/deep)`` with the issue's diff already seeded."""
    dd, deep = _deep_dirs(tmp_path)
    seed_diff_patch(dd)  # svc/loader.py, single hunk (85, 92)
    return dd, deep

def test_issue_1106_worked_example_is_distinguishable_from_the_clean_run(tmp_path: Path,) -> None:
    """Distinguish a correct line-88 finding from its line-4 structural restatement.

    The sole hunk is lines 85–92; location must expose the 81-line miss, and duplication
    must expose the pair even though similarity 0.1538 is below the 0.5 threshold."""
    dd, deep = _worked_example_dirs(tmp_path)
    seed_merged_items(deep,
        [_item(1, line=88, description=WORKED_A),
            _item(2, line=4, description=WORKED_B, lens="structural", location_distrust=True, location_cited_line=4,),
        ],
    )

    location = analyze_location(dd)
    assert location["hunk_source"] == "diff.patch"
    assert location["shipped_items"] == 2
    assert location["scored_items"] == 2
    assert location["tiers"] == {"in_hunk": 1, "within_tolerance": 0,
        "beyond_tolerance": 1,   # the mis-anchored twin
        "file_absent": 0,
    }
    assert location["in_hunk_rate"] == 0.5    # NOT 1.0 -- the run is no longer perfect
    assert location["distrusted_items"] == 1
    beyond = [row for row in location["items"] if row["tier"] == "beyond_tolerance"]
    assert [row["id"] for row in beyond] == [2]
    assert beyond[0]["distance"] == 81        # exactly the issue's arithmetic

    duplication = analyze_shipped_duplication(dd)
    assert duplication["shipped_items"] == 2
    assert duplication["comparable_pairs"] == 1
    assert duplication["same_file_pairs"] == 1           # the escape IS visible
    assert duplication["near_duplicate_pairs"] == 0      # ...but under the 0.5 bar
    assert duplication["max_similarity"] == 0.1538       # the issue's own number
    pair = duplication["pairs"][0]
    assert (pair["a_id"], pair["b_id"]) == ("1", "2")
    assert (pair["a_lens"], pair["b_lens"]) == ("per-stack", "structural")
    assert pair["same_file"] is True
    # Unattributed merge-authored items keep empty provenance; the duplicate pair still counts.
    assert (pair["a_source_uids"], pair["b_source_uids"]) == ([], [])

    clean_dd, clean_deep = _worked_example_dirs(tmp_path / "clean")
    seed_merged_items(clean_deep, [_item(1, line=88, description=WORKED_A)])
    clean_location = analyze_location(clean_dd)
    assert clean_location["in_hunk_rate"] == 1.0
    assert clean_location["tiers"]["beyond_tolerance"] == 0
    clean_duplication = analyze_shipped_duplication(clean_dd)
    assert clean_duplication["comparable_pairs"] == 0
    assert clean_duplication["same_file_pairs"] == 0
    assert clean_duplication["max_similarity"] is None
    assert clean_location["in_hunk_rate"] > location["in_hunk_rate"]

@pytest.mark.parametrize(("file", "line", "expected_tier", "expected_distance"),
    [pytest.param("svc/loader.py", 88, "in_hunk", 0, id="in_hunk"),
        pytest.param("svc/loader.py", 94, "within_tolerance", 2, id="within_tolerance"),
        pytest.param("svc/loader.py", 4, "beyond_tolerance", 81, id="beyond_tolerance"),
        pytest.param("other/untouched.py", 88, "file_absent", None, id="file_absent"),
    ],
)
def test_location_tiers_are_each_reachable(
    tmp_path: Path, file: str, line: int, expected_tier: str, expected_distance: int | None,
) -> None:
    dd, deep = _worked_example_dirs(tmp_path)
    seed_merged_items(deep, [_item(1, file=file, line=line)])

    location = analyze_location(dd)

    assert location["scored_items"] == 1
    assert location["tiers"][expected_tier] == 1
    assert location["tier_rates"][expected_tier] == 1.0
    assert location["items"][0]["tier"] == expected_tier
    assert location["items"][0]["distance"] == expected_distance

def test_location_scores_the_cited_line_not_the_snapped_line(tmp_path: Path) -> None:
    dd, deep = _worked_example_dirs(tmp_path)
    seed_merged_items(deep,
        [_item(1, line=92, location_cited_line=94)],  # snapped to 92, cited 94
    )

    location = analyze_location(dd)

    assert location["tiers"]["within_tolerance"] == 1   # the CITED line's tier
    assert location["tiers"]["in_hunk"] == 0            # not the snapped line's
    assert location["in_hunk_rate"] == 0.0
    assert location["relocated_items"] == 1
    row = location["items"][0]
    assert (row["line"], row["cited_line"], row["distance"]) == (92, 94, 2)

def test_location_structural_whole_file_anchor_does_not_pollute_tiers(tmp_path: Path,) -> None:
    """Structural line 0 counts as a whole-file anchor and never lowers scored-line rates."""
    dd, deep = _worked_example_dirs(tmp_path)
    seed_merged_items(deep,
        [_item(1, line=88), _item(2, line=0, lens="structural", description=WORKED_B),
            _item(3, line="not-an-int", description="unscorable citation"),
            _item(4, line=True, description="bool is not a line"),
        ],
    )

    location = analyze_location(dd)

    assert location["shipped_items"] == 4
    assert location["whole_file_anchors"] == 1
    assert location["unscorable_items"] == 2      # the string and the bool
    assert location["scored_items"] == 1
    assert location["tiers"] == {"in_hunk": 1, "within_tolerance": 0, "beyond_tolerance": 0, "file_absent": 0}
    assert location["in_hunk_rate"] == 1.0        # the exemptions cost nothing

def test_location_prefers_persisted_hunk_index_over_the_diff(tmp_path: Path) -> None:
    """Use ranges absent from the diff to prove the persisted index remains authoritative."""
    dd, deep = _worked_example_dirs(tmp_path)
    seed_hunk_index(dd, {"svc/loader.py": [(200, 210)]})
    seed_merged_items(deep, [_item(1, line=205)])

    location = analyze_location(dd)

    assert location["hunk_source"] == "hunk-index.json"
    assert location["tiers"]["in_hunk"] == 1

def test_location_hunk_source_falls_back_to_diff_patch(tmp_path: Path) -> None:
    """Archived runs carry only ``diff.patch``; the axis still scores them."""
    dd, deep = _worked_example_dirs(tmp_path)
    assert not (dd / "hunk-index.json").exists()
    seed_merged_items(deep, [_item(1, line=88)])

    location = analyze_location(dd)

    assert location["hunk_source"] == "diff.patch"
    assert location["in_hunk_rate"] == 1.0

def test_location_hunk_source_none_reports_not_measured(tmp_path: Path) -> None:
    """With neither artifact, "not measured" must be distinguishable from "clean"."""
    dd, deep = _deep_dirs(tmp_path)
    seed_merged_items(deep, [_item(1, line=4)])   # would be beyond_tolerance

    location = analyze_location(dd)

    assert location["hunk_source"] == "none"
    assert location["shipped_items"] == 1
    assert location["scored_items"] == 0
    assert location["tiers"] == {"in_hunk": 0, "within_tolerance": 0, "beyond_tolerance": 0, "file_absent": 0}
    assert location["tier_rates"] == {}
    assert location["in_hunk_rate"] is None       # undefined, not perfect
    assert location["items"] == []

def test_location_and_duplication_are_zeroed_when_merged_items_absent(tmp_path: Path,) -> None:
    """An existing deep directory does not prove merge produced a shipped set.

    Absent merged-items keeps duplicate counts undefined; zero means merge completed
    with a genuinely empty set."""
    dd, _deep = _worked_example_dirs(tmp_path)

    location = analyze_location(dd)
    duplication = analyze_shipped_duplication(dd)

    assert location["shipped_items"] == 0
    assert location["scored_items"] == 0
    assert location["in_hunk_rate"] is None
    assert location["hunk_source"] == "diff.patch"   # honest: hunks WERE readable
    assert duplication["shipped_items"] == 0
    assert duplication["comparable_pairs"] == 0
    assert duplication["near_duplicate_pairs"] is None
    assert duplication["max_similarity"] is None
    assert duplication["mean_similarity"] is None
    assert duplication["pairs"] == []

def test_duplication_near_duplicate_pairs_is_none_when_deep_never_ran(tmp_path: Path,) -> None:
    dd = tmp_path / ".daydream"
    dd.mkdir(parents=True)

    duplication = analyze_shipped_duplication(dd)

    assert duplication["shipped_items"] == 0
    assert duplication["comparable_pairs"] == 0
    assert duplication["near_duplicate_pairs"] is None
    assert duplication["max_similarity"] is None
    assert duplication["mean_similarity"] is None
    assert duplication["pairs"] == []

def test_shipped_duplication_input_is_capped_to_bound_the_on2_scan(tmp_path: Path,) -> None:
    """Place the only duplicate pair beyond the 200-item quadratic-comparison cap.

    It must remain uncounted while shipped_items still reports all 202 entries."""
    dd, deep = _worked_example_dirs(tmp_path)
    # Random hex descriptions avoid accidentally crossing the 0.5 similarity threshold.
    items = [_item(i, description=uuid.uuid4().hex) for i in range(200)]
    items.append(_item(200, description="the exact same duplicate description text"))
    items.append(_item(201, description="the exact same duplicate description text"))
    seed_merged_items(deep, items)

    duplication = analyze_shipped_duplication(dd)

    assert duplication["shipped_items"] == 202
    assert duplication["near_duplicate_pairs"] == 0   # the tail pair is beyond the cap

@pytest.mark.parametrize(("payload", "expected_error"),
    [pytest.param("{not json", json.JSONDecodeError, id="syntax-invalid"),
        pytest.param('{"items": {"a": 1}}', ValueError, id="wrong-shape"),
        pytest.param("{}", ValueError, id="missing-items"),
    ],
)
def test_location_and_duplication_propagate_corrupt_merged_items(
    tmp_path: Path, payload: str, expected_error: type[Exception]
) -> None:
    """The shared loader's raise-on-corrupt contract holds for the new axes too.

    A bogus shipped set is never silently counted -- and never silently scored."""
    dd, deep = _worked_example_dirs(tmp_path)
    seed_stack_records(deep, "python", n=4)
    (deep / "merged-items.json").write_text(payload)

    with pytest.raises(expected_error):
        analyze_findings(dd)
    with pytest.raises(expected_error):
        analyze_location(dd)
    with pytest.raises(expected_error):
        analyze_shipped_duplication(dd)

def test_shipped_duplication_counts_a_genuine_near_duplicate_pair(tmp_path: Path,) -> None:
    dd, deep = _worked_example_dirs(tmp_path)
    seed_merged_items(deep,
        [_item(1, line=88, description="The loader does not validate its config path"),
            _item(2, line=90, description="The loader fails to validate the config path"),
        ],
    )

    duplication = analyze_shipped_duplication(dd)

    assert duplication["comparable_pairs"] == 1
    assert duplication["near_duplicate_pairs"] == 1
    assert duplication["same_file_pairs"] == 1
    assert duplication["same_file_near_duplicate_pairs"] == 1
    assert duplication["max_similarity"] is not None
    assert duplication["max_similarity"] >= 0.5
    assert duplication["mean_similarity"] == duplication["max_similarity"]
    assert duplication["pairs"][0]["similarity"] == duplication["max_similarity"]

def test_shipped_duplication_pairs_carry_the_item_source_uids(tmp_path: Path) -> None:
    dd, deep = _worked_example_dirs(tmp_path)
    seed_merged_items(deep,
        [_item(1, line=88, description="The loader does not validate its config path", lens="structural",
                uid=mint_record_uid("structure", 3),
            ), _item(2, line=90, description="The loader fails to validate the config path",
                **_provenance(mint_record_uid("python", 4)),
            ),
        ],
    )

    duplication = analyze_shipped_duplication(dd)

    assert duplication["comparable_pairs"] == 1
    pair = duplication["pairs"][0]
    assert (pair["a_id"], pair["b_id"]) == ("1", "2")
    assert pair["a_source_uids"] == ["structure:3"]
    assert pair["b_source_uids"] == ["python:4"]
    # Remapping through the synthetic sources index must preserve each item's lens.
    assert (pair["a_lens"], pair["b_lens"]) == ("structural", "per-stack")
    # Provenance is reporting-only; numeric metrics remain unchanged.
    assert duplication["near_duplicate_pairs"] == 1
    assert duplication["same_file_pairs"] == 1

def test_shipped_duplication_reports_every_consolidated_source_uid_in_order(tmp_path: Path,) -> None:
    dd, deep = _worked_example_dirs(tmp_path)
    seed_merged_items(deep,
        [_item(1, line=88, description="The loader does not validate its config path", lens="cross-stack",
                **_provenance("python:1", "react:2"),
            ),
            _item(2, line=90, description="The loader fails to validate the config path", **_provenance("python:7"),),
        ],
    )

    duplication = analyze_shipped_duplication(dd)

    pair = duplication["pairs"][0]
    assert pair["a_source_uids"] == ["python:1", "react:2"]
    assert pair["b_source_uids"] == ["python:7"]

def test_shipped_duplication_reports_empty_provenance_rather_than_fabricating_one(tmp_path: Path,) -> None:
    """Explicit [] attribution must not be replaced with a display id, lens, or invented UID."""
    dd, deep = _worked_example_dirs(tmp_path)
    seed_merged_items(deep,
        [_item(1, line=88, description="The loader does not validate its config path", **_provenance(),),
            _item(2, line=90, description="The loader fails to validate the config path", **_provenance(),),
        ],
    )

    duplication = analyze_shipped_duplication(dd)

    pair = duplication["pairs"][0]
    assert pair["a_source_uids"] == []
    assert pair["b_source_uids"] == []
    # Missing provenance cannot hide an otherwise known duplicate.
    assert duplication["near_duplicate_pairs"] == 1

def test_shipped_duplication_falls_back_to_the_birth_uid_without_source_uids(tmp_path: Path,) -> None:
    dd, deep = _worked_example_dirs(tmp_path)
    legacy_items = [_item(
            1, line=88, description="The loader does not validate its config path", uid=mint_record_uid("python", 2),
        ),
        _item(
            2, line=90, description="The loader fails to validate the config path", uid=mint_record_uid("structure", 5),
        ),
    ]
    # Require the newer key to be absent so the legacy fallback is actually exercised.
    assert all(RECORD_SOURCE_UIDS_KEY not in item for item in legacy_items)
    seed_merged_items(deep, legacy_items)

    duplication = analyze_shipped_duplication(dd)

    pair = duplication["pairs"][0]
    assert pair["a_source_uids"] == ["python:2"]
    assert pair["b_source_uids"] == ["structure:5"]

def test_shipped_duplication_reveals_a_misset_threshold(tmp_path: Path) -> None:
    """Keep subthreshold pairs and descending similarity visible so threshold mistakes are diagnosable."""
    dd, deep = _worked_example_dirs(tmp_path)
    seed_merged_items(deep,
        [_item(1, line=88, description=WORKED_A), _item(2, line=4, description=WORKED_B, lens="structural"),
            _item(3, file="other/untouched.py", line=1, description="A wholly unrelated concern",),
        ],
    )

    duplication = analyze_shipped_duplication(dd)

    assert duplication["comparable_pairs"] == 3          # every pair, threshold=0.0
    assert duplication["near_duplicate_pairs"] == 0      # none clears 0.5
    assert duplication["same_file_pairs"] == 1           # (1, 2) share svc/loader.py
    assert duplication["max_similarity"] == 0.1538
    similarities = [pair["similarity"] for pair in duplication["pairs"]]
    assert similarities == sorted(similarities, reverse=True)
    assert similarities[0] == duplication["max_similarity"]

def test_record_duplicate_candidates_is_the_input_counter_under_findings_dedup(tmp_path: Path,) -> None:
    dd, deep = _deep_dirs(tmp_path)
    seed_dedup_candidates(deep, record_alt_pairs=[{"similarity": 0.75}, {"similarity": 0.55}],
        record_duplicate_pairs=[{"similarity": 0.9}],
    )

    dedup = analyze_findings(dd)["dedup"]

    assert dedup["record_duplicate_candidates"] == 1
    assert "record_duplicates" not in dedup
    assert dedup["record_alt_overlaps"] == 2
    assert dedup["avg_overlap_similarity"] == 0.65


def _grounding_finding(**extra: Any) -> dict[str, Any]:
    """A pre-merge per-stack record tagged for the ``deep-python`` reader."""
    finding: dict[str, Any] = {
        "id": "py-1", "_stack": "python", "file": "svc/loader.py", "line": 88, "confidence": "HIGH",
        "rationale": "svc/loader.py needs a guard",
    }
    finding.update(extra)
    return finding

def test_analyze_session_reports_location_and_shipped_duplication(tmp_path: Path,) -> None:
    dd, deep = _deep_dirs(tmp_path)
    seed_diff_patch(dd)
    seed_hunk_index(dd, {"svc/loader.py": [(85, 92)]})
    seed_merged_items(
        deep, [_item(1, line=88, description=WORKED_A), _item(2, line=4, description=WORKED_B, lens="structural")],
    )
    snapshot = run_snapshot("loc-session", schema_version="ATIF-v1.7", model_name="claude-sonnet-4-5",)

    result = analyze_session(dd, write_snapshot=snapshot)

    assert result["location"]["hunk_source"] == "hunk-index.json"
    assert result["location"]["in_hunk_rate"] == 0.5
    assert result["location"]["tiers"]["beyond_tolerance"] == 1
    assert result["findings"]["shipped_duplication"]["same_file_pairs"] == 1
    assert result["findings"]["shipped_duplication"]["near_duplicate_pairs"] == 0

@pytest.mark.parametrize(("filename", "label"),
    [("trajectory.json", "main"), ("deep-python.json", "deep-python"),
        ("trajectory-20260101T000000-abc123.json", "main"), ("deadbeef.deep-python.json", "deep-python"),
    ],
)
def test_agent_label_keeps_its_legacy_filename_tolerance(filename: str, label: str) -> None:
    """The legacy shapes are retained deliberately — pin them instead of guessing they are dead."""
    assert _agent_label(filename) == label
