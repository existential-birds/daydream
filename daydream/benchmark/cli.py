"""Argument parsing and command dispatch for daydream benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only; objective is imported lazily at runtime
    from daydream.benchmark.harbor import objective

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from daydream import git_ops


def _fail(message: str) -> int:
    """Print ``message`` to stderr and return exit code ``1`` (never a bare traceback)."""
    print(message, file=sys.stderr)
    return 1


def _build_benchmark_parser() -> argparse.ArgumentParser:
    """Build the ``daydream benchmark`` subcommand parser."""
    parser = argparse.ArgumentParser(
        prog="daydream benchmark",
        description=(
            "Private PR benchmark workspace: init/status/validate/build-harbor/upgrade/"
            "import-prs/curate/calibrate-judge/run/clean/objective/aggregate."
        ),
    )
    sub = parser.add_subparsers(dest="subcommand")

    init_p = sub.add_parser("init", help="create a private benchmark workspace")
    init_p.add_argument("dir", type=Path, help="workspace directory (must be empty/absent)")
    init_p.add_argument("--repo", required=True, help="OWNER/REPO repository")
    init_p.add_argument(
        "--reviewer-host", action="append", default=[], help="reviewer egress host (repeatable)"
    )
    init_p.add_argument(
        "--judge-host", action="append", default=[], help="judge egress host (repeatable)"
    )

    status_p = sub.add_parser("status", help="show read-only derived workspace state")
    status_p.add_argument("dir", type=Path, help="workspace directory")

    validate_p = sub.add_parser("validate", help="validate the workspace (0/2/1 exit codes)")
    validate_p.add_argument("dir", type=Path, help="workspace directory")
    validate_p.add_argument("--compiled", action="store_true", help="validate emitted tasks with Harbor 0.22")

    build_p = sub.add_parser("build-harbor", help="package a validated workspace for Harbor 0.22")
    build_p.add_argument("dir", type=Path, help="workspace directory")
    build_p.add_argument("--daydream-wheel", required=True, type=Path, help="wheel for this Daydream version")

    upgrade_p = sub.add_parser(
        "upgrade",
        help="upgrade legacy cases and verify ready-snapshot base provenance",
    )
    upgrade_p.add_argument("dir", type=Path, help="workspace directory")
    upgrade_p.add_argument("--dry-run", action="store_true", help="report the upgrade without writing")

    import_prs_p = sub.add_parser(
        "import-prs", help="import explicit private GitHub PR evidence into the workspace"
    )
    import_prs_p.add_argument("dir", type=Path, help="workspace directory")
    import_prs_p.add_argument(
        "--pr",
        action="append",
        default=[],
        metavar="N|URL",
        help="PR number or https://github.com/OWNER/REPO/pull/N (repeatable)",
    )
    import_prs_p.add_argument(
        "--pr-file", action="append", default=[], type=Path, metavar="FILE",
        help="file listing PR numbers/URLs, one per line (repeatable)",
    )
    import_prs_p.add_argument(
        "--head", action="append", default=[], metavar="PR=<40-hex>",
        help=(
            "explicit head SHA of PR N (PR=<40-hex>, repeatable); a bare 40-hex "
            "is accepted for back-compat and treated as the sole requested PR"
        ),
    )
    import_prs_p.add_argument(
        "--refresh", action="store_true",
        help="re-fetch already-imported PRs",
    )

    curate_p = sub.add_parser("curate", help="curate a case's golden review")
    curate_p.add_argument("dir", type=Path, help="workspace directory")
    curate_p.add_argument(
        "--case", metavar="CASE-ID", help="case id to curate"
    )
    curate_p.add_argument(
        "--apply-gold",
        type=Path,
        default=None,
        metavar="FILE",
        help="apply a reviewed gold YAML draft (derive all forbidden fields, never ready)",
    )

    calibrate_p = sub.add_parser(
        "calibrate-judge",
        help="diagnostic: measure the configured judge's agreement with the unverified fixture",
        description=(
            "diagnostic: measure the configured judge's agreement with the unverified "
            "labeled fixture. A passing result only means the judge agrees with this "
            "unverified fixture — it is not calibrated or correct."
        ),
    )
    calibrate_p.add_argument("dir", type=Path, help="workspace directory")
    calibrate_p.add_argument(
        "--yes", action="store_true", help="confirm the paid 72-call calibration run"
    )

    run_p = sub.add_parser(
        "run", help="supervise a Harbor run behind the Oracle self-match gate"
    )
    run_p.add_argument("dir", type=Path, help="workspace directory")
    run_p.add_argument(
        "--oracle", action="store_true", help="run the Oracle self-match pass"
    )
    run_p.add_argument(
        "--yes", action="store_true", help="confirm the paid run without prompting"
    )

    clean_p = sub.add_parser(
        "clean", help="remove ledger-derived disposable artifacts (issue #782)"
    )
    clean_p.add_argument("dir", type=Path, help="workspace directory")
    clean_p.add_argument(
        "--cache", action="store_true",
        help="remove the disposable clone + build stage under cache/"
    )
    clean_p.add_argument(
        "--jobs", action="store_true",
        help="remove ledgered Harbor job dirs + their recorded Docker images"
    )
    clean_p.add_argument(
        "--trajectories", action="store_true",
        help="remove contained agent/trajectory.json files in ledgered job dirs"
    )
    clean_p.add_argument(
        "--derived", action="store_true",
        help="union of --cache --jobs --trajectories (preserves curated source/gold)"
    )
    clean_p.add_argument(
        "--all", action="store_true",
        help="delete every deletable artifact including curated source/gold (needs --yes)"
    )
    clean_p.add_argument(
        "--yes", action="store_true", help="confirm --all without prompting"
    )

    objective_p = sub.add_parser(
        "objective", help="resolve an exact completed run as machine-readable JSON"
    )
    objective_p.add_argument("dir", type=Path, help="workspace directory")
    objective_p.add_argument("--run-id", required=True, help="exact ledgered run id")
    objective_p.add_argument(
        "--json",
        default=None,
        metavar="PATH|-",
        help="write the strict objective JSON to this path ('-' writes to stdout)",
    )

    aggregate_p = sub.add_parser(
        "aggregate", help="pool a suite manifest of exact runs into one compatible objective JSON"
    )
    aggregate_p.add_argument(
        "manifest", type=Path, help="suite manifest file (schema_version + entries of workspace/run_id)"
    )
    aggregate_p.add_argument(
        "--json",
        default=None,
        metavar="PATH|-",
        help="write the pooled suite objective JSON to this path ('-' writes to stdout)",
    )

    return parser


def _handle_benchmark_import_prs(args: argparse.Namespace) -> int:
    """Import explicit private PRs: parse targets, preflight, then run the import."""
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.workspace import WorkspaceCorrupt

    try:
        targets = gi.parse_import_targets(args.pr, args.pr_file, args.head)
    except gi.ImportTargetError as exc:
        return _fail(str(exc))
    try:
        return gi.run_import_prs(
            args.dir,
            targets.pr_numbers,
            pr_heads=targets.pr_heads,
            refresh=args.refresh,
        )
    except gi.PreflightError as exc:
        return _fail(f"{exc.code}: {exc.message}")
    except WorkspaceCorrupt as exc:
        return _fail(str(exc))


def _handle_benchmark_init(dir_path: Path, repo: str, reviewer_hosts: list[str], judge_hosts: list[str]) -> int:
    """Run ``init_workspace`` and report classification + egress boundary."""
    from daydream.benchmark.workspace import InitError, init_workspace

    try:
        manifest = init_workspace(dir_path, repo, reviewer_hosts, judge_hosts)
    except InitError as exc:
        return _fail(str(exc))
    classification = manifest.privacy.classification
    egress = " ".join(manifest.privacy.reviewer_allowed_hosts + manifest.privacy.judge_allowed_hosts)
    print(f"classification: {classification}")
    print(f"egress boundary: {egress}")
    return 0


def _handle_benchmark_status(dir_path: Path) -> int:
    """Print the read-only derived workspace state + unresolved identity."""
    from daydream.benchmark.workspace import WorkspaceCorrupt, workspace_status

    try:
        status = workspace_status(dir_path)
    except WorkspaceCorrupt as exc:
        return _fail(str(exc))
    unresolved = "unresolved" if not status.repository_identity_resolved else "resolved"
    print(f"workspace state: {status.workspace_state}")
    print(f"repository identity: {unresolved}")
    if status.last_preflight_verified_at:
        print(f"repository identity/access verification: ran ({status.last_preflight_verified_at})")
    else:
        print("repository identity/access verification: not yet run")
    print(f"ledger entries: {len(status.ledger.pull_requests)}")
    for summary in status.case_snapshots:
        head = summary.get("head_prefix") or "-"
        snapshot_status = summary.get("snapshot_status", "imported")
        reason = summary.get("error_reason") or ""
        if reason:
            snapshot_status += f" ({reason})"
        print(
            f"  case {summary.get('case_id', '')}: "
            f"snapshot {snapshot_status} @ {head}; "
            f"task-spec approval {summary.get('task_spec_approval', 'not-required')}"
        )
    return 0


def _handle_benchmark_validate(args: argparse.Namespace) -> int:
    """Print the human-readable classification and return the numeric code."""
    if args.compiled:
        from daydream.benchmark.harbor.build import CompileError
        from daydream.benchmark.harbor.package import validate_compiled
        from daydream.benchmark.workspace import WorkspaceCorrupt

        try:
            code = validate_compiled(args.dir)
        except (CompileError, WorkspaceCorrupt) as exc:
            return _fail(str(exc))
        print("validation: compiled-ready")
        return code
    from daydream.benchmark.workspace import validate_workspace

    code, label = validate_workspace(args.dir)
    print(f"validation: {label}")
    return code


def _handle_benchmark_build_harbor(args: argparse.Namespace) -> int:
    """Package a validated authoring workspace as a runnable Harbor dataset."""
    from daydream.benchmark.harbor.build import CompileError
    from daydream.benchmark.harbor.package import build_harbor
    from daydream.benchmark.workspace import WorkspaceCorrupt

    try:
        lock = build_harbor(args.dir, wheel=args.daydream_wheel)
    except (CompileError, WorkspaceCorrupt) as exc:
        return _fail(str(exc))
    print(f"built Harbor dataset with {len(lock.get('cases', {}))} case(s)")
    return 0


def _handle_benchmark_upgrade(args: argparse.Namespace) -> int:
    """Upgrade legacy cases and verify snapshot provenance; report per-case failures with
    exit 1.
    """
    from daydream.benchmark import migrate

    report = migrate.migrate_workspace(args.dir, dry_run=args.dry_run)
    for c in report.cases:
        print(
            f"case {c.case_id}: finding_ids_recomputed={c.finding_ids_recomputed} "
            f"changed=True"
        )
    for e in report.errors:
        print(f"error: {e}", file=sys.stderr)
    if report.errors:
        return 1
    return 0


def _is_interactive_tty() -> bool:
    """True only when both sys.stdin and stdout are real interactive terminals."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _handle_benchmark_calibrate(args: argparse.Namespace) -> int:
    """Run the unverified-fixture agreement diagnostic after --yes or TTY confirmation.
    Expected failures and refusals report to stderr with exit 1.
    """
    if not args.yes and not _is_interactive_tty():
        return _fail(
            "calibrate-judge: requires TTY confirmation or --yes before any paid judge call"
        )
    from daydream.benchmark.harbor import calibrate

    env = {
        name: os.environ.get(name)
        for name in (
            "DAYDREAM_JUDGE_PROVIDER",
            "DAYDREAM_JUDGE_MODEL",
            "DAYDREAM_JUDGE_API_KEY",
            "DAYDREAM_JUDGE_BASE_URL",
            "DAYDREAM_JUDGE_ALLOWED_HOSTS",
            # Thread Claude judge credentials through its presence gate; the CLI itself
            # inherits the ambient OAuth token from os.environ.
            "CLAUDE_CODE_OAUTH_TOKEN",
        )
    }

    # Bind diagnostic receipts to the candidate; reject invalid profiles.
    # None preserves default identity. The run gate never reads this diagnostic receipt.
    env["DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST"] = _candidate_profile_digest()

    return calibrate.run_calibration(
        args.dir,
        yes=args.yes,
        env=env,
        http=None,
    )


def _handle_benchmark_run(args: argparse.Namespace) -> int:
    """Pass control-plane reviewer/judge settings to the Oracle-gated Harbor supervisor."""
    from daydream.benchmark.harbor import run as run_mod

    env = {
        name: os.environ.get(name)
        for name in (
            "DAYDREAM_REVIEW_BACKEND",
            "DAYDREAM_REVIEW_MODEL",
            "DAYDREAM_REVIEW_API_KEY",
            "DAYDREAM_REVIEW_BASE_URL",
            "DAYDREAM_REVIEW_EFFORT",
            # Thread Claude credentials so host-side preflight can resolve its base URL.
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "DAYDREAM_JUDGE_PROVIDER",
            "DAYDREAM_JUDGE_MODEL",
            "DAYDREAM_JUDGE_API_KEY",
            "DAYDREAM_JUDGE_BASE_URL",
        )
    }

    # Bind the tested profile digest before the supervisor writes its ledger.
    # Invalid candidates fail before a paid run; containers cannot supply provenance.
    env["DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST"] = _candidate_profile_digest()

    return run_mod.run_run(args.dir, oracle=args.oracle, yes=args.yes, env=env)


def _candidate_profile_digest() -> str | None:
    """Resolve the trusted profile candidate exactly as the container entrypoint does.
    Absence preserves legacy None identity; invalid candidates raise ProfileError.
    """
    from daydream import review_profile as rp

    if not os.environ.get("DAYDREAM_REVIEW_PROFILE_CANDIDATE"):
        return None
    resolved = rp.resolve_harbor_profile()  # env=None -> os.environ (trusted)
    return resolved.digest


def _handle_benchmark_clean(args: argparse.Namespace) -> int:
    """Resolve derived selections before cleanup; total deletion requires --yes or TTY
    confirmation. Expected cleanup failures report to stderr with exit 1.
    """
    from daydream.benchmark.harbor import clean as clean_mod, run as run_mod
    from daydream.benchmark.storage import WorkspaceCorrupt

    if args.all and not args.yes and not _is_interactive_tty():
        return _fail(
            "clean --all: requires TTY confirmation or --yes before deleting curated source/gold"
        )
    cache = args.cache or args.derived
    jobs = args.jobs or args.derived
    trajectories = args.trajectories or args.derived
    try:
        report = clean_mod.clean_workspace(
            args.dir,
            cache=cache,
            jobs=jobs,
            trajectories=trajectories,
            all_=args.all,
            yes=args.yes,
        )
    except (run_mod.RunError, WorkspaceCorrupt) as exc:
        return _fail(str(exc))
    for line in report.summary_lines():
        print(line)
    return report.exit_code


def _handle_benchmark_curate(args: argparse.Namespace) -> int:
    """Open interactive curation or require a reviewed --apply-gold draft. All mutations go
    through curation services; errors return exit 1.
    """
    from pydantic import ValidationError

    from daydream.benchmark import curation as cu
    from daydream.benchmark.storage import WorkspaceCorrupt, load_yaml_strict

    if args.apply_gold is None:
        if _is_interactive_tty():
            from daydream.benchmark.curate_tui import run_curate_tui
            return run_curate_tui(args.dir, args.case)
        return _fail(
            "curate: interactive curation requires a TTY; pass --apply-gold <file> to apply "
            "a reviewed gold draft"
        )
    try:
        fragment = load_yaml_strict(args.apply_gold)
        cu.CaseEditor(args.dir, args.case).apply_gold_fragment(fragment)
    except (cu.CurationError, WorkspaceCorrupt, git_ops.GitError, ValidationError, KeyError, TypeError) as exc:
        return _fail(str(exc))
    return 0


def _handle_benchmark_objective(args: argparse.Namespace) -> int:
    """Emit an exact completed run as a local summary or opaque JSON. Validation failures
    preserve existing output bytes; successful file writes are atomic.
    """
    from daydream.benchmark.harbor import objective
    from daydream.benchmark.storage import atomic_write_json

    try:
        run = objective.read_completed_run(args.dir, args.run_id, env=dict(os.environ))
    except objective.ObjectiveError as exc:
        return _fail(str(exc))

    if args.json is not None:
        blob = objective.objective_to_json(run)
        if args.json == "-":
            print(json.dumps(blob, indent=2))
        else:
            atomic_write_json(Path(args.json), blob)

    obj = run.objective
    # In ``--json -`` mode stdout must stay pure JSON (issue #888 machine-readable
    # contract); route the human summary to stderr so ``jq``/``> file.json`` sees
    # only the blob.
    out_stream = sys.stderr if args.json == "-" else sys.stdout
    if obj is not None:
        print(
            f"objective {run.run_id}: comparison_eligible={obj.comparison_eligible} "
            f"micro_f1={obj.metrics['micro_f1']:.4f} tasks={obj.metrics['task_count']} "
            f"scored={obj.metrics['scored_task_count']} infra={obj.metrics['infra_error_task_count']}",
            file=out_stream,
        )
    else:
        print(f"objective {run.run_id}: no objective (unscored run)", file=out_stream)
    return 0


def _suite_objective_to_json(suite: objective.SuiteObjective) -> dict[str, object]:
    """Project opaque experiment identity, shared compatibility, and canonical pooled
    metrics. Repository paths, PR data, text, reasoning, and source code remain
    excluded.
    """
    from daydream.benchmark.harbor import objective

    identity = suite.identity
    objective_json = dict(suite.objective.metrics)
    return {
        "experiment_id": suite.experiment_id,
        "profile_digest": suite.profile_digest,
        "identity": objective.identity_to_dict(identity),
        "objective": objective_json,
    }


def _handle_benchmark_aggregate(args: argparse.Namespace) -> int:
    """Strictly load and pool the entire compatible suite; failed entries abort without
    replacing output. Emit shared identity/profile and optionally atomic JSON.
    """
    from daydream.benchmark.harbor import objective
    from daydream.benchmark.storage import WorkspaceCorrupt, atomic_write_json, load_json_strict

    try:
        manifest = load_json_strict(args.manifest)
        suite = objective.aggregate_suite(manifest, env=dict(os.environ))
    except objective.ObjectiveError as exc:
        return _fail(str(exc))
    except WorkspaceCorrupt as exc:
        return _fail(str(exc))

    blob = _suite_objective_to_json(suite)
    if args.json is not None:
        if args.json == "-":
            print(json.dumps(blob, indent=2))
        else:
            atomic_write_json(Path(args.json), blob)

    identity = suite.identity
    # In ``--json -`` mode stdout must stay pure JSON; route the human summary
    # to stderr so a ``jq``/file-redirect consumer sees only the blob.
    out_stream = sys.stderr if args.json == "-" else sys.stdout
    print(f"profile digest: {suite.profile_digest or ''}", file=out_stream)
    print(
        "identity: "
        f"profile={identity.profile_name} "
        f"reviewer={identity.reviewer_backend}/{identity.reviewer_model} "
        f"judge={identity.judge_provider}/{identity.judge_model}",
        file=out_stream,
    )
    print(
        f"aggregate {suite.objective.metrics['task_count']} tasks, "
        f"micro_f1={suite.objective.metrics['micro_f1']:.4f}, experiment_id={suite.experiment_id}",
        file=out_stream,
    )
    return 0


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "init": lambda a: _handle_benchmark_init(a.dir, a.repo, a.reviewer_host, a.judge_host),
    "status": lambda a: _handle_benchmark_status(a.dir),
    "validate": _handle_benchmark_validate,
    "build-harbor": _handle_benchmark_build_harbor,
    "upgrade": _handle_benchmark_upgrade,
    "import-prs": _handle_benchmark_import_prs,
    "curate": _handle_benchmark_curate,
    "calibrate-judge": _handle_benchmark_calibrate,
    "run": _handle_benchmark_run,
    "clean": _handle_benchmark_clean,
    "objective": _handle_benchmark_objective,
    "aggregate": _handle_benchmark_aggregate,
}


def _handle_benchmark_command(argv: list[str]) -> int:
    """Dispatch benchmark commands and translate expected workspace failures into stderr
    plus exit 1.
    """

    parser = _build_benchmark_parser()
    args = parser.parse_args(argv)
    sub = args.subcommand
    if sub is None:
        parser.print_help()
        return 0
    handler = _HANDLERS.get(sub)
    if handler is not None:
        return handler(args)
    parser.print_help(file=sys.stderr)
    return 2
