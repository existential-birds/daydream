"""Admin commands and their argument parsing."""

import argparse
from pathlib import Path

from daydream import git_ops
from daydream.agent import console
from daydream.commands import common
from daydream.config_file import load_file_config
from daydream.extensions import Registry
from daydream.ui import create_console, print_error, print_warning


def _build_summarize_parser() -> argparse.ArgumentParser:
    """Build the summarize parser before review parsing can consume its positional path."""
    parser = argparse.ArgumentParser(
        prog="daydream summarize",
        description=(
            "Print run-info markdown (rollup + per-phase breakdown table) "
            "for a trajectory file or run directory."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        metavar="PATH",
        help=(
            "Either a trajectory JSON file or a run directory containing "
            "trajectory.json (and optional trajectories/ siblings)."
        ),
    )
    return parser


def _handle_summarize_command(argv: list[str]) -> int:
    """Dispatch ``daydream summarize`` to the summarize module."""
    from daydream.summarize import summarize

    return summarize(_build_summarize_parser().parse_args(argv).path)


_EXT_USAGE = (
    "usage: daydream ext {validate} ...\n"
    "\n"
    "Extension sub-verbs:\n"
    "  validate   load the daydream_ext extension and resolve-check the registry"
)


def _ext_resolve_failure(registry: "Registry") -> str | None:
    """Return the first phase/config/callable registry failure, including loop-group bodies in run_flow
    preflight.
    """
    from daydream.extensions import UnresolvedExtensionError
    from daydream.flows.engine import _resolve_steps

    for flow_name in registry.flow_names():
        try:
            _resolve_steps(registry, flow_name, registry.flow(flow_name))
        except UnresolvedExtensionError as exc:
            return str(exc)
    for name in registry.phase_names():
        phase_key = registry.phase(name).phase_key
        if not isinstance(phase_key, str):
            return f"phase '{name}' has a non-string config key: {phase_key!r}"
    for name in registry.prompt_names():
        if not callable(registry.prompt(name)):
            return f"prompt slot '{name}' does not resolve to a callable"
    for name in registry.renderer_names():
        if not callable(registry.renderer(name)):
            return f"renderer slot '{name}' does not resolve to a callable"
    return None


def _handle_ext_validate_command() -> int:
    """Resolve the registry without a target repo; report source, API version, and supervisor. Load or
    resolution failures return 1.
    """
    import importlib.util
    import os

    from daydream.extensions import (
        EXTENSION_API_VERSION,
        MIN_SUPPORTED_EXTENSION_API_VERSION,
        ExtensionError,
        build_registry,
    )

    try:
        registry = build_registry()
    except ExtensionError as exc:
        print_error(console, "Extension Error", str(exc))
        return 1

    ext_dir = os.environ.get("DAYDREAM_EXT_DIR")
    if ext_dir:
        source = f"extension source: $DAYDREAM_EXT_DIR = {ext_dir}"
    elif importlib.util.find_spec("daydream_ext") is not None:
        source = "extension source: import daydream_ext"
    else:
        source = "extension source: no extension found (builtins only)"
    console.print(source, soft_wrap=True)
    console.print(
        f"extension API version {EXTENSION_API_VERSION} "
        f"(supported: {MIN_SUPPORTED_EXTENSION_API_VERSION}..{EXTENSION_API_VERSION})"
    )
    supervisor_status = "registered" if registry.tool_supervisor_if_registered() is not None else "none"
    console.print(f"tool supervisor: {supervisor_status}")
    console.print(f"trace exporters: {', '.join(registry.trace_exporter_names()) or 'none'}")

    failure = _ext_resolve_failure(registry)
    if failure is not None:
        print_error(console, "Extension Error", failure)
        return 1

    console.print(
        f"registry OK: {len(registry.phase_names())} phases, "
        f"{len(registry.flow_names())} flows, "
        f"{len(registry.prompt_names())} prompts, "
        f"{len(registry.renderer_names())} renderers"
    )
    return 0


def _handle_ext_command(argv: list[str]) -> int:
    """Dispatch ext with usage exit 2: bare help goes to stdout, invalid arguments to stderr. Validate
    accepts no trailing arguments.
    """
    if argv[:1] == ["validate"] and argv[1:]:
        common._print_namespace_help(_EXT_USAGE, error=True)
        return 2
    return common._dispatch_namespace(argv, {"validate": lambda _argv: _handle_ext_validate_command()}, _EXT_USAGE)


def _build_post_findings_parser() -> argparse.ArgumentParser:
    """Parse the unattended Phase B artifact and event-derived PR/head/repository target."""
    parser = argparse.ArgumentParser(
        prog="daydream post-findings",
        description=(
            "Validate a Phase A findings artifact against event-derived facts and "
            "post new findings to the PR (privileged Phase B poster; unattended)."
        ),
    )
    parser.add_argument(
        "artifact",
        type=Path,
        metavar="ARTIFACT",
        help="Path to the findings artifact written by --findings-out.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        metavar="PATH",
        help="Checkout directory for configuration, diagram evidence, and GitHub operations (default: cwd).",
    )
    parser.add_argument(
        "--pr",
        type=int,
        required=True,
        dest="pr_number",
        metavar="N",
        help="Event-derived target PR number.",
    )
    parser.add_argument(
        "--head-sha",
        required=True,
        metavar="SHA",
        help="Event-derived PR head SHA the artifact must declare.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        metavar="OWNER/REPO",
        help="Event-derived repository slug the artifact must declare.",
    )
    parser.add_argument(
        "--bot-login",
        metavar="LOGIN",
        help="Bot login (App slug) for prior-finding author filtering. "
        "Defaults to $DAYDREAM_BOT_HANDLE.",
    )
    parser.add_argument(
        "--approve-on-clean",
        action="store_true",
        help="Approve the PR when the posted findings contain no high/medium "
        "severity issues (issue #343): post event: 'APPROVE' instead of "
        "'COMMENT'. Also settable via [tool.daydream] approve_on_clean in a "
        "repo config file.",
    )
    return parser


def _handle_post_findings_command(argv: list[str]) -> int:
    """Validate the artifact before GitHub writes, then reconcile and post findings.

    Rejected diagrams warn and are dropped without failing valid findings. Return
    1 for validation, inventory, or posting failures; otherwise 0. No agent runs.
    """
    from daydream import pr_review

    parser = _build_post_findings_parser()
    args = parser.parse_args(argv)
    if "/" not in args.repo:
        parser.error(f"--repo must be an OWNER/REPO slug, got {args.repo!r}")

    target_dir = (args.target if args.target is not None else Path.cwd()).resolve()
    if not target_dir.is_dir():
        parser.error(f"--target must be an existing directory: {target_dir}")

    console = create_console()
    # Malformed repo config must not abort unattended posting; warn and fall back to the CLI flag.
    approve = args.approve_on_clean
    try:
        approve = approve or bool(load_file_config(target_dir).approve_on_clean)
    except ValueError as exc:
        print_warning(console, f"Ignoring malformed repo config: {exc}")
    return pr_review.post_findings_from_artifact(
        target_dir,
        args.artifact,
        pr_number=args.pr_number,
        head_sha=args.head_sha,
        repo=args.repo,
        console=console,
        bot_login=args.bot_login,
        approve_on_clean=approve, auth=git_ops.INHERIT_GITHUB_AUTH,
    )


def _build_setup_parser() -> argparse.ArgumentParser:
    """Parse setup or read-only --verify for exactly one repo/org scope; setup registers the App,
    deposits secrets, and proposes workflows.
    """
    parser = argparse.ArgumentParser(
        prog="daydream setup",
        description=(
            "Set up a self-hosted Daydream review bot: register the GitHub App, "
            "deposit credentials as Actions secrets, and land the workflows via a PR."
        ),
    )
    parser.add_argument(
        "target",
        type=Path,
        metavar="TARGET",
        help="Path to the repository working directory to set the bot up in.",
    )
    scope_group = parser.add_mutually_exclusive_group(required=True)
    scope_group.add_argument(
        "--repo",
        metavar="OWNER/REPO",
        help="Deposit secrets/variables and install at repository scope.",
    )
    scope_group.add_argument(
        "--org",
        metavar="NAME",
        help="Deposit secrets/variables and install at organization scope.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the read-only setup doctor instead of performing setup.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-register the App even if credentials are already deposited.",
    )
    return parser


def _handle_setup_command(argv: list[str]) -> int:
    """Run setup or --verify without an agent trajectory; render GitHub errors as panels and return 1
    on failure.
    """
    from daydream import bot_setup
    from daydream.git_ops import GitError
    from daydream.github_app import GitHubAppError

    parser = _build_setup_parser()
    args = parser.parse_args(argv)
    if args.repo is not None and "/" not in args.repo:
        parser.error(f"--repo must be an OWNER/REPO slug, got {args.repo!r}")

    scope = bot_setup.Scope(repo=args.repo, org=args.org)
    target = args.target

    try:
        if args.verify:
            result = bot_setup.run_verify(target, scope=scope)
            bot_setup.print_verify_result(result)
            return 0 if result.ok else 1
        return bot_setup.run_setup(target, scope=scope, force=args.force, anthropic_key=None)
    except (GitHubAppError, GitError) as exc:
        print_error(console, "Setup failed", str(exc))
        return 1
