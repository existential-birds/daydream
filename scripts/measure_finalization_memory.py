"""Offline memory characterization of trajectory serialization and finalization consumers (#1714).

A standalone ``TrajectoryRecorder`` receives synthetic backend events, with no models:
``--forks`` completed children of ``--turns`` turns whose tool result is ``--body-kb`` KB,
plus ``--root-invocations`` root invocations. Each scenario runs in a fresh subprocess.
Per stage it reports the tracemalloc peak above the stage's starting allocation (Python
heap), the kernel RSS high-water delta (``/proc/self/clear_refs``; includes native
pydantic-core buffers), bytes read (``/proc/self/io`` rchar; the table prints the archive
stage's) and seconds:

- ``write``: the root's final write on recorder exit (serialization and snapshot callback);
- ``runinfo``: ``render_live_run_info`` while the recorder is live;
- ``summarize``: ``render_run_info_block`` over the written run directory (``daydream summarize``);
- ``archive``: ``finalize_archive_run`` with archive and evaluation enabled;
- ``capture``: ``capture_run_record`` (trajectory evidence only; no review sidecars).

RSS and bytes read need Linux ``/proc`` and read 0 elsewhere. After:
``uv run --frozen python scripts/measure_finalization_memory.py``. Before: extract the base with
``git archive <base> daydream | tar -x -C <base tree>``, then, from outside the checkout,
``PYTHONPATH=<base tree> uv run --frozen --project <checkout> python <checkout>/scripts/<this file>``.
Add ``--forks 10 --body-kb 1 --turns 1000`` for a metadata-heavy row.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MIB = 1024 * 1024
SESSION = "0d1714aa-0000-4000-8000-000000000000"
STAGES = ("write", "runinfo", "summarize", "archive", "capture")


PROC = Path("/proc/self")


def _proc_value(path: str, field: str) -> int:
    if not PROC.is_dir():
        return 0
    for line in Path(path).read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1])
    raise RuntimeError(f"{path} has no {field}")


class Stage:
    """Python heap peak, RSS high-water delta, bytes read and seconds for one stage."""

    def __init__(self, result: dict[str, float], name: str) -> None:
        self.result, self.name = result, name

    def __enter__(self) -> Stage:
        if PROC.is_dir():
            (PROC / "clear_refs").write_text("5")  # reset VmHWM to the current VmRSS
        self.rss = _proc_value("/proc/self/status", "VmRSS")
        self.traced = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        self.read = _proc_value("/proc/self/io", "rchar")
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.result[f"{self.name}_s"] = time.perf_counter() - self.started
        self.result[f"{self.name}_peak"] = (tracemalloc.get_traced_memory()[1] - self.traced) / MIB
        self.result[f"{self.name}_rss"] = (_proc_value("/proc/self/status", "VmHWM") - self.rss) / 1024
        self.result[f"{self.name}_read"] = (_proc_value("/proc/self/io", "rchar") - self.read) / MIB


def _drive(inv: Any, turns: int, body: str) -> None:
    from daydream.backends import TextEvent, ToolResultEvent, ToolStartEvent, TurnEndEvent

    for turn in range(turns):
        inv.observe(TextEvent(text=f"turn {turn}"))
        inv.observe(ToolStartEvent(id=f"t{turn}", name="Read", input={"file_path": f"f{turn}.py"}))
        inv.observe(ToolResultEvent(id=f"t{turn}", output=body, is_error=False))
        inv.observe(TurnEndEvent(message_id=f"m{turn}"))


def _scenario(forks: int, body_kb: int, turns: int, root_invocations: int) -> dict[str, float]:
    import anyio

    # Import lazily loaded consumers up front so stage figures exclude module loading.
    import daydream.archive.provenance  # noqa: F401
    import daydream.dataset  # noqa: F401
    import daydream.dataset_scoring  # noqa: F401
    import daydream.deep.dedup  # noqa: F401
    import daydream.eval.analyzer  # noqa: F401
    import daydream.git_ops.immutable  # noqa: F401
    import daydream.immutable_review  # noqa: F401
    import daydream.retry_policy  # noqa: F401
    from daydream.archive import finalize_archive_run, get_archive_dir
    from daydream.archive.manifest import archive_recorder_provenance_from_snapshot
    from daydream.artifact_visibility import ArtifactEvidenceProvenance, ArtifactTreeSnapshot, manifest_tree
    from daydream.dataset_capture import capture_run_record
    from daydream.eval.analyzer import collect_trajectory_paths
    from daydream.pr_comment_renderer import render_run_info_block
    from daydream.pr_run_info import LiveRunInfoSource, render_live_run_info
    from daydream.run_config import RunConfig
    from daydream.run_snapshot import ArchiveRunSnapshot, ManifestRunIdentity, RunPhaseCapabilities
    from daydream.trajectory import DaydreamPhase, DaydreamRunFlow, TrajectoryRecorder, default_trajectory_path
    from daydream.trajectory.scopes import fork_scope, invocation_scope

    body = "x" * (body_kb * 1024)
    result: dict[str, float] = {}
    snapshots: list[Any] = []

    async def record(frozen: Path) -> None:
        recorder = TrajectoryRecorder(
            path=default_trajectory_path(frozen, SESSION), run_flow=DaydreamRunFlow.DEEP, target_dir=frozen,
            agent_model_name="synthetic", session_id=SESSION,
            on_write=lambda _recorder, snapshot: snapshots.append(snapshot),
        )
        write = Stage(result, "write")
        async with recorder:
            for index in range(forks):
                async with fork_scope(recorder, f"reviewer-{index}") as child:
                    async with invocation_scope(child, DaydreamPhase.REVIEW) as inv:
                        _drive(inv, turns, body)
            for _ in range(root_invocations):
                async with invocation_scope(recorder, DaydreamPhase.REVIEW) as inv:
                    _drive(inv, turns, body)
            with Stage(result, "runinfo"):
                info = render_live_run_info(LiveRunInfoSource(recorder, object()))  # type: ignore[arg-type]
            assert info.diagnostic is None, info.diagnostic
            write.__enter__()
        write.__exit__()

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        frozen = tmp / "run" / "frozen"  # the dataset store must not overlap the frozen tree's parent
        tracemalloc.start()
        anyio.run(record, frozen)
        final = snapshots[-1]
        written = sorted(frozen.glob(".daydream/runs/*/**/*.json"))
        result["docs"] = len(final.documents)
        result["run_mib"] = sum(path.stat().st_size for path in written) / MIB
        result["largest_mib"] = max(path.stat().st_size for path in written) / MIB
        with Stage(result, "summarize"):
            render_run_info_block(collect_trajectory_paths(default_trajectory_path(frozen, SESSION).parent))
        (tmp / "source").mkdir()
        identity = ManifestRunIdentity(
            skill=None, model=None, backend="claude", review_backend=None, fix_backend=None, test_backend=None,
            per_stack_review_backend=None, per_stack_review_model=None, review_only=True, deep=True, profile=None,
            phases=RunPhaseCapabilities(per_stack_review=True, merge=True, fix=False, test=False, push=False,
                                        remote_ci=False),
        )
        provenance = archive_recorder_provenance_from_snapshot(write_snapshot=final, run_flow=DaydreamRunFlow.DEEP)
        with Stage(result, "archive"):
            finalize_archive_run(
                run=ArchiveRunSnapshot(provenance, identity, final),
                artifacts=ArtifactTreeSnapshot(SESSION, "workspace", frozen, manifest_tree(frozen), ()),
                artifact_provenance=ArtifactEvidenceProvenance("workspace", SESSION, tmp / "source", tmp / "live"),
                config=RunConfig(target=str(frozen), archive=True, run_eval=True), work=None,
            )
        assert (get_archive_dir() / "runs" / SESSION / "manifest.json").is_file()
        # identity=None skips review sidecars; a captured source supplies the final head without Git.
        work = SimpleNamespace(source=tmp / "source", repo=tmp / "source", head_sha="0" * 40, head_branch="main",
                               base_branch="main", captured_source=SimpleNamespace(source_id=None))
        artifacts = ArtifactTreeSnapshot(SESSION, "workspace", frozen, manifest_tree(frozen), ())
        with Stage(result, "capture"):
            capture_run_record(
                artifacts=artifacts, artifact_session=SimpleNamespace(code_workspace=None),  # type: ignore[arg-type]
                selected=final, original_task=None, identity=None, work=work,  # type: ignore[arg-type]
                config=RunConfig(dataset_store_path=tmp / "store"), outcome="success",
            )
        assert len(list((tmp / "store" / "runs").glob("*.jsonl"))) == 1
        tracemalloc.stop()
    result["maxrss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--forks", type=int, nargs="+", default=[10, 40])
    parser.add_argument("--body-kb", type=int, nargs="+", default=[10, 50])
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--root-invocations", type=int, default=4)
    parser.add_argument("--scenario", nargs=2, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.scenario:
        from daydream.dataset import LocalRecordStore

        store_defaults = LocalRecordStore.__init__.__kwdefaults__
        assert store_defaults is not None
        store_defaults["max_record_bytes"] = 1 << 34  # admit every scenario
        forks, body_kb = args.scenario
        with tempfile.TemporaryDirectory() as archive:
            os.environ["DAYDREAM_ARCHIVE_DIR"] = archive
            print(json.dumps(_scenario(forks, body_kb, args.turns, args.root_invocations)))
        return
    header = ["forks", "body KB", "docs", "run MiB", "largest MiB"]
    header += [f"{stage} {kind}" for stage in STAGES for kind in ("peak MiB", "RSS MiB", "s")]
    header += ["archive read MiB", "maxrss MiB"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for body_kb in args.body_kb:
        for forks in args.forks:
            output = subprocess.run(
                [sys.executable, __file__, "--scenario", str(forks), str(body_kb), "--turns", str(args.turns),
                 "--root-invocations", str(args.root_invocations)],
                check=True, capture_output=True, text=True,
            ).stdout.strip().splitlines()[-1]
            row = json.loads(output)
            cells = [str(forks), str(body_kb), f"{row['docs']:.0f}", f"{row['run_mib']:.2f}",
                     f"{row['largest_mib']:.2f}"]
            for stage in STAGES:
                cells += [f"{row[stage + '_peak']:.2f}", f"{row[stage + '_rss']:.1f}", f"{row[stage + '_s']:.3f}"]
            cells += [f"{row['archive_read']:.2f}", f"{row['maxrss']:.1f}"]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
