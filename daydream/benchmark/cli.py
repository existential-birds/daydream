"""CLI helpers for the ``daydream benchmark`` subcommand.

These helpers are called from :func:`daydream.cli.main` when ``benchmark`` is
the first argv token. They live here rather than in the top-level ``daydream.cli``
module to keep that file below the 1 000-line threshold and to co-locate the
benchmark argument-parsing logic with the rest of the benchmark package.

``benchmark`` carries the sub-verbs ``init``, ``status``, ``validate``,
``build-harbor``, ``upgrade``, ``import-prs``, ``curate``, ``calibrate-judge``,
``run``, ``clean``, ``objective``, and ``aggregate``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _build_benchmark_parser() -> argparse.ArgumentParser:
    """Build the ``daydream benchmark`` subcommand parser.

    Sub-verbs: ``init``, ``status``, ``validate``, ``build-harbor``, ``upgrade``, ``import-prs``,
    ``curate``, ``calibrate-judge``, ``run``, ``clean``, ``objective``, ``aggregate``.
    """
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
        "upgrade", help="deterministically upgrade legacy case documents (finding_id + schema_version)"
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


def _handle_benchmark_import_prs(args) -> int:
    """Import explicit private PRs: parse targets, preflight, then run the import.

    Expected errors (mis-tokenized targets, preflight failure) print a message
    to stderr and return exit ``1`` — never a bare traceback.
    """
    from daydream.benchmark import github_import as gi
    from daydream.benchmark.workspace import WorkspaceCorrupt

    try:
        targets = gi.parse_import_targets(args.pr, args.pr_file, args.head)
    except gi.ImportTargetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        return gi.run_import_prs(
            args.dir,
            targets.pr_numbers,
            pr_heads=targets.pr_heads,
            refresh=args.refresh,
        )
    except gi.PreflightError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except WorkspaceCorrupt as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _handle_benchmark_init(dir_path: Path, repo: str, reviewer_hosts: list[str], judge_hosts: list[str]) -> int:
    """Run ``init_workspace`` and report classification + egress boundary."""
    from daydream.benchmark.workspace import InitError, init_workspace

    try:
        manifest = init_workspace(dir_path, repo, reviewer_hosts, judge_hosts)
    except InitError as exc:
        print(str(exc), file=sys.stderr)
        return 1
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
        print(str(exc), file=sys.stderr)
        return 1
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
        print(
            f"  case {summary.get('case_id', '')}: "
            f"snapshot {summary.get('snapshot_status', 'imported')} @ {head}"
        )
    return 0


def _handle_benchmark_validate(args) -> int:
    """Print the human-readable classification and return the numeric code."""
    if args.compiled:
        from daydream.benchmark.harbor.build import CompileError
        from daydream.benchmark.harbor.package import validate_compiled
        from daydream.benchmark.workspace import WorkspaceCorrupt

        try:
            code = validate_compiled(args.dir)
        except (CompileError, WorkspaceCorrupt) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("validation: compiled-ready")
        return code
    from daydream.benchmark.workspace import validate_workspace

    code, label = validate_workspace(args.dir)
    print(f"validation: {label}")
    return code


def _handle_benchmark_build_harbor(args) -> int:
    """Package a validated authoring workspace as a runnable Harbor dataset."""
    from daydream.benchmark.harbor.build import CompileError
    from daydream.benchmark.harbor.package import build_harbor
    from daydream.benchmark.workspace import WorkspaceCorrupt

    try:
        lock = build_harbor(args.dir, wheel=args.daydream_wheel)
    except (CompileError, WorkspaceCorrupt) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"built Harbor dataset with {len(lock.get('cases', {}))} case(s)")
    return 0


def _handle_benchmark_upgrade(args) -> int:
    """Deterministically upgrade legacy v1 case documents to v2 in place.

    Prints the per-case report plus any surfaced errors. Returns ``0`` on a
    successful upgrade (including an idempotent no-op second run) and ``1``
    when a case errored.
    """
    from daydream.benchmark import migrate

    report = migrate.migrate_workspace(args.dir, dry_run=args.dry_run)
    for c in report.cases:
        print(
            f"case {c.case_id}: finding_ids_recomputed={c.finding_ids_recomputed} "
            f"changed={c.changed}"
        )
    for e in report.errors:
        print(f"error: {e}", file=sys.stderr)
    if report.errors:
        return 1
    return 0


def _is_interactive_tty() -> bool:
    """True only when both sys.stdin and stdout are real interactive terminals."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _handle_benchmark_calibrate(args) -> int:
    """Diagnostic: measure the configured judge's agreement with the unverified fixture.

    A passing result only means the judge agrees with this unverified labeled
    fixture — it is not calibrated or correct. Requires ``--yes`` or an
    interactive TTY before any paid judge call. Lazy imports the calibrate
    module (mirroring the other handlers); build failures and refusals print to
    stderr and return exit ``1`` — never a bare traceback.
    """
    if not args.yes and not _is_interactive_tty():
        print(
            "calibrate-judge: requires TTY confirmation or --yes before any paid judge call",
            file=sys.stderr,
        )
        return 1
    from daydream.benchmark.harbor import calibrate

    env = {
        name: os.environ.get(name)
        for name in (
            "DAYDREAM_JUDGE_PROVIDER",
            "DAYDREAM_JUDGE_MODEL",
            "DAYDREAM_JUDGE_API_KEY",
            "DAYDREAM_JUDGE_BASE_URL",
            "DAYDREAM_JUDGE_ALLOWED_HOSTS",
        )
    }

    # Issue #885/R12: thread the control-plane candidate profile digest so a
    # candidate-scoped diagnostic receipt can be produced (its invalidation
    # inputs fold the digest). Fail-closed on an invalid candidate. None for
    # default runs. The receipt is diagnostic-only — it is not read by run.py.
    env["DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST"] = _candidate_profile_digest()

    return calibrate.run_calibration(
        args.dir,
        yes=args.yes,
        env=env,
        http=None,
    )


def _handle_benchmark_run(args) -> int:
    """Supervise a Harbor run in the workspace behind the Oracle gate.

    Threads the reviewer/judge env vars into the supervisor (mirroring
    :func:`_handle_benchmark_calibrate`); expected supervisor errors already
    print to stderr inside ``run_run`` and return a nonzero code — this
    handler does not wrap them in a traceback.
    """
    from daydream.benchmark.harbor import run as run_mod

    env = {
        name: os.environ.get(name)
        for name in (
            "DAYDREAM_REVIEW_BACKEND",
            "DAYDREAM_REVIEW_MODEL",
            "DAYDREAM_REVIEW_API_KEY",
            "DAYDREAM_REVIEW_BASE_URL",
            "DAYDREAM_REVIEW_EFFORT",
            "DAYDREAM_JUDGE_PROVIDER",
            "DAYDREAM_JUDGE_MODEL",
            "DAYDREAM_JUDGE_API_KEY",
            "DAYDREAM_JUDGE_BASE_URL",
        )
    }

    # Issue #885/R12: thread the control-plane candidate profile digest into
    # the supervisor env so run.py's ledger/receipt provenance can attribute
    # the run to exactly the tested candidate. run.py reads
    # DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST from this env dict; the in-container
    # entrypoint cannot set it (different process, runs after the ledger row).
    # Fail-closed: an invalid candidate raises ProfileError here, before any
    # paid run -- matching the entrypoint's own fail-closed validation.
    env["DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST"] = _candidate_profile_digest()

    return run_mod.run_run(args.dir, oracle=args.oracle, yes=args.yes, env=env)


def _candidate_profile_digest() -> str | None:
    """Canonical digest of the control-plane Harbor candidate (R12), or None.

    Reads ``DAYDREAM_REVIEW_PROFILE_CANDIDATE`` from the trusted control-plane
    environment and resolves it exactly as the in-container entrypoint will,
    so the ledger/receipt ``profile_digest`` matches the tested profile. No
    candidate -> ``None`` (legacy default-profile runs stay byte-stable).
    Raises ``ProfileError`` on an invalid candidate (fail-closed).
    """
    from daydream import review_profile as rp

    if not os.environ.get("DAYDREAM_REVIEW_PROFILE_CANDIDATE"):
        return None
    resolved = rp.resolve_harbor_profile()  # env=None -> os.environ (trusted)
    return resolved.digest


def _handle_benchmark_clean(args) -> int:
    """Handle ``daydream benchmark clean <dir> [--cache] [--jobs] [...]``.

    Resolves the ``--derived`` union into the three selection flags *before*
    calling ``clean_workspace`` (the contract the routing test pins) and runs
    contracted deletion entirely inside ``clean_workspace``. An explicit
    ``--all`` without ``--yes`` needs a TTY (mirroring ``run --yes`` /

    ``calibrate-judge``); expected ``RunError``/``WorkspaceCorrupt`` print to
    stderr and return exit ``1`` — never a bare traceback.
    """
    from daydream.benchmark.harbor import clean as clean_mod
    from daydream.benchmark.harbor import run as run_mod
    from daydream.benchmark.storage import WorkspaceCorrupt

    if args.all and not args.yes and not _is_interactive_tty():
        print(
            "clean --all: requires TTY confirmation or --yes before deleting "
            "curated source/gold",
            file=sys.stderr,
        )
        return 1
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
        print(str(exc), file=sys.stderr)
        return 1
    for line in report.summary_lines():
        print(line)
    return report.exit_code


def _handle_benchmark_curate(args) -> int:
    """Curate a case: derive everything, never attests to ready.

    On an interactive TTY, ``curate`` dispatches into the resumable terminal
    client (:func:`daydream.benchmark.curate_tui.run_curate_tui`); otherwise it
    requires ``--apply-gold <file>`` (a reviewed gold YAML draft) and routes
    it through :func:`daydream.benchmark.curation.apply_gold_fragment`. Expected
    workspace errors print to stderr and return exit ``1`` — never a bare traceback.
    """
    from pydantic import ValidationError

    from daydream import git_ops
    from daydream.benchmark import curation as cu
    from daydream.benchmark.storage import WorkspaceCorrupt, load_yaml_strict

    if args.apply_gold is None:
        if _is_interactive_tty():
            from daydream.benchmark.curate_tui import run_curate_tui
            return run_curate_tui(args.dir, args.case)
        print(
            "curate: interactive curation requires a TTY; pass --apply-gold <file> to apply "
            "a reviewed gold draft",
            file=sys.stderr,
        )
        return 1
    try:
        fragment = load_yaml_strict(args.apply_gold)
        cu.apply_gold_fragment(args.dir, args.case, fragment)
    except (cu.CurationError, WorkspaceCorrupt, git_ops.GitError, ValidationError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _handle_benchmark_objective(args) -> int:
    """Resolve an exact completed run and emit its machine-readable objective.

    ``--json`` serializes the opaque privacy-safe objective via
    ``objective.objective_to_json`` and writes it through
    ``storage.atomic_write_json`` (or prints it directly on ``-``); a parse/
    compat failure leaves an existing output file byte-identical. Without
    ``--json``, prints a concise local summary (run_id, comparison_eligible,
    micro F1, task/infra counts) to stdout. Expected ``ObjectiveError`` prints
    to stderr and returns exit ``1`` — never a bare traceback.
    """
    from daydream.benchmark.harbor import objective
    from daydream.benchmark.storage import atomic_write_json

    try:
        run = objective.read_completed_run(args.dir, args.run_id, env=dict(os.environ))
    except objective.ObjectiveError as exc:
        print(str(exc), file=sys.stderr)
        return 1

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
            f"micro_f1={obj.f1:.4f} tasks={obj.task_count} "
            f"scored={obj.scored_task_count} infra={obj.infra_error_task_count}",
            file=out_stream,
        )
    else:
        print(f"objective {run.run_id}: no objective (unscored run)", file=out_stream)
    return 0


def _suite_objective_to_json(suite) -> dict[str, object]:
    """Project a pooled ``SuiteObjective`` into opaque machine-readable JSON.

    Produces the stable ``experiment_id``, the shared ``profile_digest``, the
    full ``identity`` dict (identical across every pooled entry), and the
    count-derived ``objective`` dict projected in the authoritative
    ``aggregate_metrics`` key/shaper set. No repository slug, PR number, source
    path, sample text, judge reasoning, or source code is emitted; only opaque
    ids and counts pass through (privacy must-have).
    """
    from daydream.benchmark.harbor import objective

    identity = suite.identity
    objective_json = suite.objective._as_metric_dict()
    return {
        "experiment_id": suite.experiment_id,
        "profile_digest": suite.profile_digest,
        "identity": objective.identity_to_dict(identity),
        "objective": objective_json,
    }


def _handle_benchmark_aggregate(args) -> int:
    """Pool a suite manifest of exact runs into one compatible objective JSON (issue #888).

    Loads the manifest through ``storage.load_json_strict``, then drives
    ``objective.aggregate_suite`` (which fails closed on any missing/incomplete/
    incompatible/malformed/duplicated entry — never a silently-subsetted pool).
    When ``--json`` is set, the strict suite objective is written through
    ``storage.atomic_write_json`` (or printed on ``-``); an expected
    ``ObjectiveError``/``WorkspaceCorrupt`` prints to stderr and returns exit
    ``1`` without touching an existing output file — never a bare traceback.
    The shared profile digest and full compatibility identity are always printed
    to stdout.
    """
    from daydream.benchmark.harbor import objective
    from daydream.benchmark.storage import WorkspaceCorrupt, atomic_write_json, load_json_strict

    try:
        manifest = load_json_strict(args.manifest)
        suite = objective.aggregate_suite(manifest, env=dict(os.environ))
    except objective.ObjectiveError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except WorkspaceCorrupt as exc:
        print(str(exc), file=sys.stderr)
        return 1

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
        f"aggregate {suite.objective.task_count} tasks, "
        f"micro_f1={suite.objective.f1:.4f}, experiment_id={suite.experiment_id}",
        file=out_stream,
    )
    return 0


def _handle_benchmark_command(argv: list[str]) -> int:
    """Handle ``daydream benchmark init|status|validate|build-harbor|upgrade|import-prs|curate``.

    Returns an exit code; ``daydream.cli.main`` translates it to a
    process exit. Expected workspace errors (``InitError``/``WorkspaceCorrupt``/
    ``ImportTargetError``/``PreflightError``/``CurationError``) are printed to
    stderr and mapped to exit ``1`` — never a bare traceback. ``run`` dispatches
    to :func:`_handle_benchmark_run` (the supervised Harbor runner),
    ``objective`` to :func:`_handle_benchmark_objective` (the read-only
    machine-readable run resolution), and ``aggregate`` to
    :func:`_handle_benchmark_aggregate` (the pooled suite objective).
    """

    parser = _build_benchmark_parser()
    args = parser.parse_args(argv)
    sub = args.subcommand
    if sub is None:
        parser.print_help()
        return 0
    if sub == "init":
        return _handle_benchmark_init(args.dir, args.repo, args.reviewer_host, args.judge_host)
    if sub == "status":
        return _handle_benchmark_status(args.dir)
    if sub == "validate":
        return _handle_benchmark_validate(args)
    if sub == "build-harbor":
        return _handle_benchmark_build_harbor(args)
    if sub == "upgrade":
        return _handle_benchmark_upgrade(args)
    if sub == "import-prs":
        return _handle_benchmark_import_prs(args)
    if sub == "curate":
        return _handle_benchmark_curate(args)
    if sub == "calibrate-judge":
        return _handle_benchmark_calibrate(args)
    if sub == "run":
        return _handle_benchmark_run(args)
    if sub == "clean":
        return _handle_benchmark_clean(args)
    if sub == "objective":
        return _handle_benchmark_objective(args)
    if sub == "aggregate":
        return _handle_benchmark_aggregate(args)
    parser.print_help(file=sys.stderr)
    return 2
