"""Common commands and their argument parsing."""

import argparse
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

from rich.console import Console

from daydream import git_ops
from daydream.config_file import DaydreamFileConfig, load_file_config
from daydream.observability.config import ObservabilityConfig, ObservabilityError, resolve_observability_config
from daydream.run_config import RunConfig
from daydream.trajectory import RUN_DOCUMENT_NAME, RUNS_DIRNAME
from daydream.ui import NEON_THEME


def _auto_detect_pr_number(repo: Path) -> int | None:
    """Find the PR for the target checkout branch, independent of the invoking cwd."""
    try:
        data = git_ops.gh_pr_view(repo, None, auth=git_ops.INHERIT_GITHUB_AUTH)
    except git_ops.GitError:
        return None
    if not data:
        return None
    number = data.get("number")
    return int(number) if isinstance(number, int) else None


def _detect_repo_slug(repo: Path) -> str | None:
    """Detect owner/repo from the target checkout so archives and trajectories retain its provenance."""
    try:
        slug = git_ops.gh_repo_view(repo, auth=git_ops.INHERIT_GITHUB_AUTH)
    except git_ops.GitError:
        return None
    if slug is None:
        return None
    owner, name = slug
    return f"{owner}/{name}"


def _resolve_target_provenance(target: str | None) -> tuple[Path, str | None, DaydreamFileConfig]:
    """Return the target repository, slug, and low-precedence file policy; never infer from invoking cwd."""
    target_repo = Path(target) if target else Path.cwd()
    pr_repo = _detect_repo_slug(target_repo)
    file_config = load_file_config(target_repo)
    return target_repo, pr_repo, file_config


def _add_shared_arguments(parser: argparse.ArgumentParser, *, full_help: bool = True) -> None:
    """Add flags shared by review and improve.

    CLI globals outrank file phase overrides, file globals, and backend defaults.
    Per-phase settings live in [tool.daydream.phases.<phase>]; review's pre-parse
    scan rejects their removed flags with a config pointer. With full_help=False,
    advanced flags still parse but hide their help until --help-all.
    """
    tracing = parser.add_mutually_exclusive_group()
    tracing.add_argument(
        "--trace-to", action="append", default=None, metavar="NAME",
        help="Export OpenLLMetry traces to a destination (langsmith, honeyhive, otlp, or an extension); repeatable"
        if full_help else argparse.SUPPRESS,
    )
    tracing.add_argument(
        "--no-tracing", action="store_true",
        help="Disable tracing, including DAYDREAM_TRACE_TO" if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trace-content", choices=("full", "metadata"), default=None,
        help="Trace content: full captures redacted messages and tools; metadata omits content"
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trajectory",
        metavar="PATH",
        type=Path,
        dest="trajectory_path",
        help=(
            "Write ATIF v1.7 trajectory JSON to PATH (default: publish to "
            f"<target>/.daydream/{RUNS_DIRNAME}/<session_id>/{RUN_DOCUMENT_NAME} after finalization; "
            "explicit external paths receive live updates)"
        ) if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Disable automatic archival to ~/.daydream/archive/" if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--capture-data", action="store_true", dest="dataset_capture",
        help="Capture validated run evidence in the private local JSONL store (opt-in)"
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-capture-data", action="store_true", dest="dataset_capture_disabled",
        help="Disable JSONL collection and upload, including an explicitly configured Hub destination"
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dataset-store", type=Path, dest="dataset_store_path", metavar="DIR",
        help="Local JSONL store directory (default: ~/.daydream/dataset); enables data capture"
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-eval",
        action="store_false",
        dest="run_eval",
        help="Skip the deterministic evaluation analysis during archive "
        "(eval runs by default: it is file-based and cheap)"
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dump-artifacts",
        metavar="DIR",
        help="Merge the finalized run bundle (ATIF trajectory, review output, deep artifacts, diffs, "
             "findings, manifest, evaluation) into DIR for CI upload. Preserves unrelated destination "
             "files. Always copies exact assembled bytes, including credentials and binary files, "
             "without scanning or sanitizing. "
             "Works on every flow."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--review-profile",
        metavar="PATH",
        dest="review_profile_path",
        help=(
            "Explicit review-profile TOML path (highest-precedence source; "
            "beats DAYDREAM_REVIEW_PROFILE, the repo-committed "
            "file_config.review_profile, and the packaged default)"
        ),
    )
    parser.add_argument(
        "--trajectory-hub-repo",
        metavar="REPO",
        help="Capture and upload immutable JSONL evidence to this private Hugging Face dataset repo "
             "(owner/repo). Requires HF credentials; rejects public destinations. "
             "Always refuses blocking credentials and scanner failures. Advisory findings are allowed. "
             "Unpublished evidence remains in the local retry queue."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["claude", "codex", "pi", "osprey"],
        help="Agent backend: claude, codex, pi, or osprey "
             "(default: config file, then claude)",
    )
    parser.add_argument(
        "--model", "-m",
        metavar="MODEL",
        help="Global default model across phases "
             "(default: config file, then the per-backend table). "
             "This global --model takes precedence over any per-phase config-file override.",
    )
    parser.add_argument(
        "--reasoning-effort",
        metavar="EFFORT",
        help="Global reasoning-effort override (e.g. low, medium, high). "
             "Consumed by every backend through its native knob: Codex as "
             "-c model_reasoning_effort=<EFFORT>, Claude as --effort, Pi as "
             "--thinking. Takes precedence over any per-phase config-file override.",
    )
    parser.add_argument(
        "--latency-profile",
        metavar="PROFILE",
        help="Wonder/arbiter effort profile: fast, balanced (default), or "
             "forensic. Sets the effort floor for the wonder and arbiter "
             "phases; risk may raise it, never lower it. Codex-only effort "
             "effects apply where the effort table has an entry. An "
             "unrecognised value resolves fail-safe to forensic.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run without prompting; take each prompt's safe default "
             "(confirm intent, decline fixes, exit the test/heal loop)."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        dest="log_mode",
        help="Bypass Rich UI and emit redacted agent events as plain text to stdout; "
             "on unexpected fatal errors, additionally write redacted chained "
             "diagnostics to stderr."
        if full_help else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--yes",
        action="store_const",
        const="yes",
        dest="assume",
        help="Auto-answer every yes/no gate with yes (apply fixes, commit). "
             "Orthogonal to --non-interactive: --yes pre-decides the answer, "
             "--non-interactive controls whether we may block on stdin.",
    )


def _resolve_cli_observability(parser: argparse.ArgumentParser, args: argparse.Namespace) -> ObservabilityConfig:
    """Turn invalid operator settings into the same actionable CLI errors as invalid flags."""
    try:
        return resolve_observability_config(
            destinations=args.trace_to, disabled=args.no_tracing, content=args.trace_content,
        )
    except ObservabilityError as exc:
        parser.error(str(exc))


def _add_dry_run_argument(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the shared ``--dry-run`` option to a corpus/train subcommand parser."""
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=help_text,
    )


def _add_license_arguments(
    parser: argparse.ArgumentParser, *, policy_help: str, copyleft_help: str
) -> None:
    """Add the shared ``--license-policy``/``--allow-copyleft`` options."""
    parser.add_argument(
        "--license-policy", type=Path, default=None, dest="license_policy",
        metavar="PATH", help=policy_help,
    )
    parser.add_argument(
        "--allow-copyleft", action="append", default=[], dest="allow_copyleft",
        metavar="OWNER/REPO", help=copyleft_help,
    )


def _print_namespace_help(usage: str, *, error: bool = False) -> None:
    """Print usage to stderr for errors, otherwise stdout."""


    Console(stderr=error, theme=NEON_THEME).print(usage)


def _dispatch_namespace(argv: list[str], subverbs: dict[str, Callable[[list[str]], int]], usage: str) -> int:
    """Dispatch sub-handlers unchanged; bare/unknown verbs return 2 with usage on stdout/stderr
    respectively.
    """
    if not argv:
        _print_namespace_help(usage)
        return 2
    handler = subverbs.get(argv[0])
    if handler is None:
        _print_namespace_help(usage, error=True)
        return 2
    return int(handler(argv[1:]))


def config_from_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *, detect_pr: bool = False, **overrides: Any,
) -> RunConfig:
    """Copy only RunConfig destinations; resolve operator settings before target provenance.
    Command-specific transformations stay at callers and parser-only flags never enter runtime
    policy.
    """
    observability = _resolve_cli_observability(parser, args)
    target_repo, pr_repo, file_config = _resolve_target_provenance(args.target)
    values = {item.name: getattr(args, item.name) for item in fields(RunConfig) if hasattr(args, item.name)}
    if args.dataset_store_path is not None:
        values["dataset_capture"] = True
    if args.dataset_capture_disabled:
        values["dataset_capture"] = False
    if detect_pr:
        values["pr_number"] = args.pr_number if args.pr_number is not None else _auto_detect_pr_number(target_repo)
    values.update(observability=observability, file_config=file_config, pr_repo=pr_repo, archive=not args.no_archive)
    return RunConfig(**(values | overrides))
