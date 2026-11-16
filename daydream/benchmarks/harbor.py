"""Export a run's local OpenTelemetry evidence as Harbor-compatible ATIF v1.7.

Benchmark-only: no live recorder, remote tracing destination or credential is
involved. The exporter reduces each evidence document to its view and maps it to
the vendored ATIF models, writing into one output directory:

- ``trajectory.json``: the root document, with every fork embedded under
  ``subagent_trajectories`` following fork parentage, so Harbor's usage loader
  totals per-model usage once per step (dispatch steps carry no usage);
- ``trajectory.subagent-<document>.json``: each fork as a flat file, where
  Harbor's trace extraction discovers subagent conversations.

Dispatch steps' subagent references point at those flat files. A document with
no steps exports one ``system`` step marking a host-only or partial snapshot,
because ATIF requires a step; the evidence itself never contains it. Exported
trajectories are validated with Harbor's (vendored) validator; an export never
creates benchmark completion receipts or findings.

Run ``python -m daydream.benchmarks.harbor RUN_DIR OUT_DIR`` on a run directory
holding ``evidence.otlp.jsonl`` and its ``forks/`` siblings.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from daydream.benchmarks.atif import TrajectoryValidator
from daydream.benchmarks.atif.models import Agent, FinalMetrics, Step, Trajectory
from daydream.trajectory import (
    DOCUMENT_SUFFIX,
    RUN_DOCUMENT_NAME,
    RunWriteSnapshot,
    read_document_view,
    run_document_path,
    siblings_directory,
)

ROOT_FILE = "trajectory.json"
SCHEMA_VERSION: Literal["ATIF-v1.7"] = "ATIF-v1.7"


class AtifExportError(ValueError):
    """The evidence cannot be exported as valid ATIF."""


@dataclass(frozen=True)
class AtifExport:
    """Exported root trajectory and the flat subagent files, keyed by file name."""

    root: Trajectory
    subagents: dict[str, Trajectory]


def _subagent_file(view: dict[str, Any], source_name: str) -> str:
    return f"trajectory.subagent-{source_name.removesuffix(DOCUMENT_SUFFIX)}.json"


def _steps(view: dict[str, Any], files: dict[str, str]) -> list[Step]:
    steps: list[dict[str, Any]] = view["steps"]
    if not steps:
        partial = bool(view["extra"].get("partial"))
        return [Step(step_id=1, timestamp=view["extra"].get("snapshot_at") or view["extra"].get("run_ended_at"),
                     source="system",
                     message="Daydream run snapshot" if partial else "Daydream host-only run snapshot",
                     extra={"daydream_run_flow": view.get("run_flow"),
                            "host_event": "partial_snapshot" if partial else "host_only_final_snapshot"})]
    exported: list[Step] = []
    for step in steps:
        observation = step.get("observation")
        if observation is not None:
            results = []
            for result in observation["results"]:
                refs = result.get("subagent_trajectory_ref")
                if refs:
                    result = {**result, "subagent_trajectory_ref": [
                        {**ref, "trajectory_path": files.get(ref.get("trajectory_id"), ref.get("trajectory_path"))}
                        for ref in refs]}
                results.append(result)
            step = {**step, "observation": {**observation, "results": results}}
        exported.append(Step.model_validate(step))
    return exported


def _trajectory(view: dict[str, Any], files: dict[str, str], children: list[Trajectory] | None) -> Trajectory:
    return Trajectory(
        schema_version=SCHEMA_VERSION,
        session_id=view["session_id"],
        trajectory_id=view["trajectory_id"],
        agent=Agent(**view["agent"]),
        steps=_steps(view, files),
        final_metrics=FinalMetrics(**view["final_metrics"]),
        extra=view["extra"],
        subagent_trajectories=children or None,
    )


def export_views(root: dict[str, Any], forks: Sequence[tuple[str, dict[str, Any]]]) -> AtifExport:
    """Map a root view and ``(source file name, view)`` forks to validated ATIF."""
    files = {view["trajectory_id"]: _subagent_file(view, name) for name, view in forks}
    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for _name, view in forks:
        by_parent.setdefault(view.get("parent_trajectory_id"), []).append(view)

    def embedded(trajectory_id: str) -> list[Trajectory]:
        return [_trajectory(child, files, embedded(child["trajectory_id"]))
                for child in by_parent.get(trajectory_id, [])]

    exported_root = _trajectory(root, files, embedded(root["trajectory_id"]))
    exported = AtifExport(exported_root, {
        files[view["trajectory_id"]]: _trajectory(view, files, None) for _name, view in forks
    })
    validator = TrajectoryValidator()  # type: ignore[no-untyped-call]  # vendored, untyped
    for name, trajectory in (("root", exported.root), *exported.subagents.items()):
        if not validator.validate(trajectory.model_dump(mode="json", exclude_none=True), validate_images=False):
            raise AtifExportError(f"exported {name} trajectory is not valid ATIF: {validator.errors[:3]}")
    return exported


def _write(trajectory: Trajectory, path: Path) -> None:
    path.write_text(json.dumps(trajectory.model_dump(mode="json", exclude_none=True), indent=2) + "\n")


def write_export(export: AtifExport, out_dir: Path) -> Path:
    """Write the root and flat subagent files; return the root path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    root_path = out_dir / ROOT_FILE
    _write(export.root, root_path)
    for name, trajectory in export.subagents.items():
        _write(trajectory, out_dir / name)
    return root_path


def _split(views: Iterable[tuple[str, dict[str, Any]]], session_id: str) -> AtifExport:
    root: dict[str, Any] | None = None
    forks: list[tuple[str, dict[str, Any]]] = []
    for name, view in views:
        if view["session_id"] != session_id:
            raise AtifExportError("evidence document belongs to another run")
        if view["trajectory_id"] == session_id:
            root = view
        else:
            forks.append((name, view))
    if root is None:
        raise AtifExportError("run evidence has no root document")
    return export_views(root, forks)


def export_snapshot(snapshot: RunWriteSnapshot) -> AtifExport:
    """Export a frozen run snapshot's checked evidence documents."""
    return _split(((document.path.name, document.validated_payload(snapshot.root_trajectory_id))
                   for document in snapshot.documents), snapshot.root_trajectory_id)


def export_run_directory(run_dir: Path) -> AtifExport:
    """Export a run directory holding the root evidence document and its siblings."""
    root_view = read_document_view(run_document_path(run_dir).read_bytes())
    siblings = siblings_directory(run_dir)
    paths = sorted(siblings.glob(f"*{DOCUMENT_SUFFIX}")) if siblings.is_dir() else []
    views = [(RUN_DOCUMENT_NAME, root_view), *((path.name, read_document_view(path.read_bytes())) for path in paths)]
    return _split(views, root_view["session_id"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m daydream.benchmarks.harbor", description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="run directory holding evidence.otlp.jsonl")
    parser.add_argument("out_dir", type=Path, help="directory receiving trajectory.json and subagent files")
    args = parser.parse_args(argv)
    print(write_export(export_run_directory(args.run_dir), args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
