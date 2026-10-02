"""Register, configure, and verify the self-hosted review bot."""

from __future__ import annotations

import getpass
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream import config, git_ops
from daydream.agent import console
from daydream.bot_manifest import register_app_via_manifest
from daydream.git_ops import GitError
from daydream.github_app import (
    APP_ID_ENV,
    APP_PRIVATE_KEY_ENV,
    AppCredentials,
    GitHubAppError,
    build_app_jwt_auth,
    get_app_metadata,
    resolve_credentials,
)
from daydream.templates import workflow_template_files
from daydream.ui import print_error, print_info, print_success, print_warning

_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
_SETUP_BRANCH = "daydream/setup-bot"

@dataclass(frozen=True)
class Scope:
    """Exactly one repository or organization target for Actions secrets and variables."""

    repo: str | None = None
    org: str | None = None

    def __post_init__(self) -> None:
        if bool(self.repo) == bool(self.org):
            raise ValueError("Scope requires exactly one of repo='owner/repo' or org='name'")

    def _secret_kwargs(self) -> dict[str, str]:
        """Map the scope to the ``gh_secret_set``/``gh_variable_set`` keyword."""
        if self.repo:
            return {"repo_slug": self.repo}
        return {"org": self.org} if self.org else {}


def deposit_secrets(
    repo_dir: Path,
    creds: AppCredentials,
    *,
    anthropic_key: str,
    bot_handle: str,
    scope: Scope,
) -> None:
    """Overwrite the three setup secrets and bot-handle variable at ``scope``.

    Secret values, including the PEM, travel on stdin and are never logged.
    GitHubAppError names any failed list/set operation; partial deposits cannot
    be reported as success.
    """
    scope_kwargs = scope._secret_kwargs()
    secret_values = {
        "DAYDREAM_APP_ID": str(creds.app_id),
        "DAYDREAM_APP_PRIVATE_KEY": creds.private_key,
        "ANTHROPIC_API_KEY": anthropic_key,
    }

    try:
        existing = set(git_ops.gh_secret_list(repo_dir, **scope_kwargs, auth=git_ops.INHERIT_GITHUB_AUTH))
    except GitError as exc:
        raise GitHubAppError(f"Could not list existing secrets: {exc}") from exc

    already = [name for name in config.SETUP_SECRET_NAMES if name in existing]
    if already:
        print_info(console, f"Overwriting existing secrets: {', '.join(already)}")

    for name in config.SETUP_SECRET_NAMES:
        try:
            git_ops.gh_secret_set(repo_dir, name, secret_values[name], **scope_kwargs, auth=git_ops.INHERIT_GITHUB_AUTH)
        except GitError as exc:
            raise GitHubAppError(f"Failed to set secret {name}: {exc}") from exc

    try:
        git_ops.gh_variable_set(
            repo_dir, config.BOT_HANDLE_VAR, bot_handle, **scope_kwargs, auth=git_ops.INHERIT_GITHUB_AUTH,
        )
    except GitError as exc:
        raise GitHubAppError(f"Failed to set variable {config.BOT_HANDLE_VAR}: {exc}") from exc


# Returned by :func:`land_workflows` when all three workflow files already exist
# verbatim, so no branch/PR was opened. Deliberately not a URL (does not start
# with ``http``) so the caller can distinguish a no-op from a freshly opened PR.
WORKFLOWS_ALREADY_INSTALLED = "already-installed"

_WORKFLOWS_DIR = ".github/workflows"

_PR_TITLE = "Add Daydream review-bot workflows"
_PR_BODY = (
    "Adds the Daydream self-hosted review-bot GitHub Actions workflows.\n\n"
    "These workflows run code review in your own Actions runners under your "
    "GitHub App identity. Review the setup guide before merging:\n\n"
    "- Setup guide: `docs/self-hosted-bot-setup.md`\n"
    "- Security model: see the *Security model* section of that guide.\n\n"
    "After merging, a trusted maintainer can request a head-bound review by "
    "commenting `@<bot> review` on a pull request."
)


def land_workflows(repo_dir: Path, *, branch: str) -> str:
    """Land packaged workflows through a branch and reviewable PR.

    Commit only changed workflow files. Customized files are replaced with a
    warning. If every file already matches, return WORKFLOWS_ALREADY_INSTALLED
    without creating a branch or PR. Git/gh failures propagate.
    """
    workflows_dir = repo_dir / _WORKFLOWS_DIR
    workflows_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for template in workflow_template_files():
        target = workflows_dir / template.name
        content = template.read_text()
        if target.exists() and target.read_text() == content:
            continue
        if target.exists():
            print_warning(
                console,
                f"{_WORKFLOWS_DIR}/{template.name} is a customized workflow — customization is "
                "intentionally unsupported and it will be replaced by the packaged template.",
            )
        target.write_text(content)
        copied.append(Path(_WORKFLOWS_DIR) / template.name)

    if not copied:
        print_info(console, "Workflow files already present; skipping branch/PR.")
        return WORKFLOWS_ALREADY_INSTALLED

    base = git_ops.default_branch(repo_dir)
    if git_ops.branch_exists(repo_dir, branch):
        git_ops.checkout_branch(repo_dir, branch)
    else:
        git_ops.create_branch(repo_dir, branch)
    git_ops.commit_paths(repo_dir, copied, _PR_TITLE)
    git_ops.push_branch(repo_dir, branch)
    existing_prs = git_ops.gh_pr_list_for_branch(repo_dir, branch, auth=git_ops.INHERIT_GITHUB_AUTH)
    if existing_prs:
        return str(existing_prs[0]["url"])
    return git_ops.gh_pr_create(
        repo_dir, head=branch, base=base, title=_PR_TITLE, body=_PR_BODY, auth=git_ops.INHERIT_GITHUB_AUTH,
    )


@dataclass(frozen=True)
class Check:
    """One doctor result, with remediation in ``detail`` when it fails.

    Non-required checks are informational and do not affect VerifyResult.ok.
    """

    name: str
    passed: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class VerifyResult:
    """Ordered doctor results; success requires every required check to pass."""

    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        """True when every required check passed (skipped optionals ignored)."""
        return all(c.passed for c in self.checks if c.required)


def _owner_of(scope: Scope) -> str:
    """Resolve the target owner login from a repo/org scope."""
    if scope.org:
        return scope.org
    # scope.repo is "owner/repo" (Scope validates exactly one is set).
    parsed = git_ops.split_owner_repo(scope.repo or "")
    return parsed[0] if parsed is not None else ""


def _bot_handle_for(slug: str | None, scope: Scope) -> str:
    """Return the bot handle to deposit: the App slug when known, else the scope owner."""
    return slug if slug else _owner_of(scope)


def _installed_owner_logins(repo_dir: Path, creds: AppCredentials) -> set[str | None]:
    """Read installed account logins under App JWT authentication; GitError propagates."""
    jwt_auth, bearer = build_app_jwt_auth(creds.app_id, creds.private_key)
    installations = git_ops.gh_api(
        repo_dir, "/app/installations", headers=bearer, idempotent=True, auth=jwt_auth,
    )
    return {
        (inst.get("account") or {}).get("login")
        for inst in (installations if isinstance(installations, list) else [])
    }


def _skipped_no_app_credentials(name: str, purpose: str) -> Check:
    """Report an optional App-level check as skipped when no local credentials exist."""
    return Check(
        name=name,
        passed=True,
        detail=(
            "Skipped: no local App credentials "
            f"({APP_ID_ENV}/{APP_PRIVATE_KEY_ENV}) {purpose}"
        ),
        required=False,
    )


def _check_app_installed(repo_dir: Path, scope: Scope, creds: AppCredentials | None) -> Check:
    """Check the target owner installation, or skip without local App credentials."""
    if creds is None:
        return _skipped_no_app_credentials(
            "app_installed",
            "to read installations. Export them to verify the App is installed on the target.",
        )
    owner = _owner_of(scope)
    try:
        logins = _installed_owner_logins(repo_dir, creds)
    except GitError as exc:
        return Check(
            name="app_installed",
            passed=False,
            detail=f"Could not read App installations: {exc}. Confirm the App credentials are valid.",
        )
    if owner in logins:
        return Check(name="app_installed", passed=True, detail=f"App is installed on '{owner}'.")
    return Check(
        name="app_installed",
        passed=False,
        detail=(
            f"App is not installed on '{owner}'. Install it via "
            f"https://github.com/settings/installations (or the org's settings)."
        ),
    )


def _check_secrets_and_var(repo_dir: Path, scope: Scope) -> Check:
    """Check (2): all required secrets + the bot-handle variable are present."""
    scope_kwargs = scope._secret_kwargs()
    try:
        secrets = set(git_ops.gh_secret_list(repo_dir, **scope_kwargs, auth=git_ops.INHERIT_GITHUB_AUTH))
        variables = set(git_ops.gh_variable_list(repo_dir, **scope_kwargs, auth=git_ops.INHERIT_GITHUB_AUTH))
    except GitError as exc:
        return Check(
            name="secrets",
            passed=False,
            detail=f"Could not list secrets/variables: {exc}. Confirm gh is authenticated for the target.",
        )

    missing_secrets = [name for name in config.SETUP_SECRET_NAMES if name not in secrets]
    var_missing = config.BOT_HANDLE_VAR not in variables
    if not missing_secrets and not var_missing:
        return Check(
            name="secrets",
            passed=True,
            detail=f"All {len(config.SETUP_SECRET_NAMES)} secrets and {config.BOT_HANDLE_VAR} are set.",
        )

    parts: list[str] = []
    if missing_secrets:
        parts.append(
            "Missing secret(s): "
            + ", ".join(missing_secrets)
            + f" — set via `gh secret set <NAME> {_scope_flag(scope)}`."
        )
    if var_missing:
        parts.append(
            f"Missing variable {config.BOT_HANDLE_VAR} — "
            f"set via `gh variable set {config.BOT_HANDLE_VAR} {_scope_flag(scope)}`."
        )
    return Check(name="secrets", passed=False, detail=" ".join(parts))


def _check_permissions(repo_dir: Path, creds: AppCredentials | None) -> Check:
    """Check (3): App permissions are a superset of the required set."""
    if creds is None:
        return _skipped_no_app_credentials("permissions", "to read App permissions.")
    try:
        meta = get_app_metadata(repo_dir, creds.app_id, creds.private_key)
    except GitHubAppError as exc:
        return Check(
            name="permissions",
            passed=False,
            detail=f"Could not read App permissions: {exc}. Confirm the App credentials are valid.",
        )
    granted = meta.get("permissions") or {}
    _LEVELS = {"none": 0, "read": 1, "write": 2, "admin": 3}
    missing = [
        name
        for name in config.APP_PERMISSIONS
        if _LEVELS.get(granted.get(name, "none"), 0)
        < _LEVELS.get(config.APP_PERMISSIONS[name], 0)
    ]
    if not missing:
        return Check(name="permissions", passed=True, detail="App grants all required permissions.")
    detail = ", ".join(f"{name}={config.APP_PERMISSIONS[name]}" for name in missing)
    return Check(
        name="permissions",
        passed=False,
        detail=(
            f"App is missing required permission(s): {detail}. "
            "Update the App's permissions in its GitHub settings and re-accept the install."
        ),
    )


#: The exact vocabulary of the review workflow's optional ``command`` dispatch
#: input. Mirrors ``options: [review, sequence, flowchart]`` in the packaged
#: ``daydream-review.yml``: `review` is the full review, the other two are
#: diagram-only passes over the same approved head.
_WORKFLOW_COMMAND_OPTIONS = ("review", "sequence", "flowchart")


def _command_input_intact(spec: Any) -> bool:
    """Allow an absent optional selector, or a choice restricted to approved commands.

    A free-form selector or extra choice could dispatch a run the maintainer
    did not approve.
    """
    if spec is None:
        return True
    if not isinstance(spec, dict) or spec.get("type") != "choice":
        return False
    options = spec.get("options")
    if not isinstance(options, list):
        return False
    return sorted(str(option) for option in options) == sorted(_WORKFLOW_COMMAND_OPTIONS)


def _workflow_contract_intact(name: str, content: str) -> bool:
    """Check the parsed workflow's approval boundary; invalid/unknown files fail.

    Reviews require approved_head_sha and may not trigger on pull_request.
    An optional command selector must retain exactly the approved choices.
    Every command dispatch must bind the live API-resolved HEAD_SHA; post runs
    must follow workflow_run completion. Comments cannot satisfy these checks.
    """
    try:
        import yaml  # lazy: pyyaml is a dev-only dependency (see pyproject.toml)
    except ImportError:
        return False
    try:
        wf = yaml.safe_load(content)
    except yaml.YAMLError:
        return False
    if not isinstance(wf, dict):
        return False
    # PyYAML parses the bare ``on:`` key as boolean ``True``; normalize the
    # trigger map the same way the test harness's ``_wf_triggers`` does.
    on: Any = wf.get("on")
    if on is None:
        on = wf.get(True)
    triggers = on if isinstance(on, dict) else {}

    if name == "daydream-review.yml":
        if "pull_request" in triggers:
            return False
        dispatch = triggers.get("workflow_dispatch")
        inputs = dispatch.get("inputs") if isinstance(dispatch, dict) else None
        if not isinstance(inputs, dict) or "approved_head_sha" not in inputs:
            return False
        return _command_input_intact(inputs.get("command"))
    if name == "daydream-command.yml":
        # The single approval point: every review dispatch must bind the live
        # PR head to the approved_head_sha input. Require the binding itself —
        # the input set to the gh-api-resolved $HEAD_SHA variable (the test
        # harness asserts the same) — so a drift that keeps the literal token
        # in a hardcoded value, echo, or comment fails instead of passing as
        # gate-intact.
        dispatch_steps = [
            step
            for job in wf.get("jobs", {}).values()
            if isinstance(job, dict)
            for step in job.get("steps", [])
            if isinstance(step, dict)
            and "gh workflow run daydream-review.yml" in step.get("run", "")
        ]
        return bool(dispatch_steps) and all(
            '-f approved_head_sha="$HEAD_SHA"' in step.get("run", "")
            and "gh api" in step.get("run", "")
            and ".head.sha" in step.get("run", "")
            for step in dispatch_steps
        )
    if name == "daydream-post.yml":
        return "workflow_run" in triggers
    return False


def _check_workflows(repo_dir: Path) -> Check:
    """Check default-branch workflows against templates and the approval contract.

    Missing or gate-breaking files fail. Intact customized variants pass with
    an unsupported-customization warning.
    """
    try:
        base = git_ops.default_branch(repo_dir)
    except git_ops.BranchNotFoundError as exc:
        return Check(
            name="workflows",
            passed=False,
            detail=f"Could not resolve the default branch: {exc}.",
        )

    missing: list[str] = []
    outdated: list[str] = []
    customized: list[str] = []
    for template in workflow_template_files():
        path = f"{_WORKFLOWS_DIR}/{template.name}"
        try:
            installed = git_ops.show(repo_dir, f"origin/{base}", path)
        except GitError:
            missing.append(path)
            continue
        if installed == template.read_bytes():
            continue
        if _workflow_contract_intact(template.name, installed.decode("utf-8", errors="replace")):
            customized.append(path)
        else:
            outdated.append(path)

    if not missing and not outdated:
        if customized:
            return Check(
                name="workflows",
                passed=True,
                detail=(
                    f"Workflow file(s) on '{base}' are customized variants: "
                    + ", ".join(customized)
                    + ". Customization is intentionally unsupported — `daydream setup` "
                    "replaces them with the packaged templates."
                ),
            )
        return Check(name="workflows", passed=True, detail=f"All workflow files match on '{base}'.")

    problems: list[str] = []
    if missing:
        problems.append("Missing workflow file(s): " + ", ".join(missing))
    if outdated:
        problems.append("Out of date workflow file(s): " + ", ".join(outdated))
    return Check(
        name="workflows",
        passed=False,
        detail=(
            f"Workflow check failed on '{base}': "
            + "; ".join(problems)
            + " — run `daydream setup` (or merge the setup PR) to install the packaged versions."
        ),
    )


def _scope_flag(scope: Scope) -> str:
    """Render the ``gh`` scope flag for a remediation hint."""
    if scope.org:
        return f"--org {scope.org}"
    return f"--repo {scope.repo}"


def run_verify(repo_dir: Path, *, scope: Scope) -> VerifyResult:
    """Read installation, secret/variable, permission and workflow state.

    App-level checks are optional without local credentials. Required failures
    name the missing element and remediation; this doctor never mutates state.
    """
    creds = resolve_credentials()
    checks = (
        _check_app_installed(repo_dir, scope, creds),
        _check_secrets_and_var(repo_dir, scope),
        _check_permissions(repo_dir, creds),
        _check_workflows(repo_dir),
    )
    return VerifyResult(checks=checks)


def print_verify_result(result: VerifyResult) -> None:
    """Render successes, required failures and informational skips; the CLI sets the exit code."""
    for check in result.checks:
        if check.passed and check.required:
            print_success(console, f"[{check.name}] {check.detail}")
        elif check.passed:
            print_info(console, f"[{check.name}] {check.detail}")
        else:
            print_error(console, f"[{check.name}] check failed", check.detail)
    if result.ok:
        print_success(console, "All required checks passed; the bot is configured.")
    else:
        print_warning(console, "Setup is incomplete; address the failed checks above.")


def _confirm_installation(repo_dir: Path, scope: Scope, creds: AppCredentials) -> bool:
    """Check installation, prompt for the manual Install click if needed, then re-check once."""
    owner = _owner_of(scope)
    if _owner_installed(repo_dir, owner, creds):
        return True

    install_url = (
        f"https://github.com/organizations/{scope.org}/settings/installations"
        if scope.org
        else "https://github.com/settings/installations"
    )
    print_info(
        console,
        f"Install the new App on '{owner}' to finish: {install_url}",
    )
    _wait_for_install_click()
    return _owner_installed(repo_dir, owner, creds)


def _owner_installed(repo_dir: Path, owner: str, creds: AppCredentials) -> bool:
    """Read installation state; transport errors raise GitHubAppError instead of returning False."""
    try:
        logins = _installed_owner_logins(repo_dir, creds)
    except GitError as exc:
        raise GitHubAppError(f"Could not read App installations: {exc}") from exc
    return owner in logins


def _wait_for_install_click() -> None:
    """Wait for the operator to confirm installation."""
    input("Press Enter once you have installed the App on the target...")


def _prompt_for_anthropic_key() -> str | None:
    """Read a hidden, stripped key on a TTY; return None for cancellation or non-TTY input."""
    if not sys.stdin.isatty():
        return None
    print_info(
        console,
        f"{_ANTHROPIC_KEY_ENV} is not set. Enter it now — it will be stored as an Actions secret.",
    )
    try:
        entered = getpass.getpass(f"{_ANTHROPIC_KEY_ENV} (input hidden): ")
    except (EOFError, KeyboardInterrupt):
        return None
    return entered.strip() or None


def run_setup(
    target_dir: Path,
    *,
    scope: Scope,
    force: bool,
    anthropic_key: str | None,
) -> int:
    """Resolve the API key, register/reuse the App, confirm installation and deposit secrets.

    Workflow files land through a reviewable PR. ``force`` re-registers even when
    secrets exist. Return 0 for success/no-op and 1 for recoverable preflight or
    installation failures; registration, deposit and Git errors propagate.
    """
    if shutil.which("gh") is None:
        print_error(
            console,
            "gh not found",
            "The GitHub CLI (`gh`) must be installed and authenticated. See https://cli.github.com/.",
        )
        return 1

    resolved_key = anthropic_key or os.environ.get(_ANTHROPIC_KEY_ENV)
    if not resolved_key:
        resolved_key = _prompt_for_anthropic_key()
    if not resolved_key:
        print_error(
            console,
            "ANTHROPIC_API_KEY missing",
            f"Set {_ANTHROPIC_KEY_ENV} in the environment (or pass it) before running setup, "
            "or run interactively to be prompted for it.",
        )
        return 1

    already = set(git_ops.gh_secret_list(target_dir, **scope._secret_kwargs(), auth=git_ops.INHERIT_GITHUB_AUTH))
    creds_present = all(name in already for name in config.SETUP_SECRET_NAMES)

    if creds_present and not force:
        print_info(
            console,
            "App credentials already deposited; skipping registration (use --force to re-register).",
        )
        creds = resolve_credentials()
        if creds is None:
            print_error(
                console,
                "Cannot reuse credentials",
                (
                    f"Secrets exist at the target but {APP_ID_ENV}/{APP_PRIVATE_KEY_ENV} are not "
                    "set locally to confirm the install. Re-run with --force, or export them."
                ),
            )
            return 1
        try:
            slug = get_app_metadata(target_dir, creds.app_id, creds.private_key).get("slug") or None
        except GitHubAppError:
            slug = None
    else:
        creds, slug = register_app_via_manifest(target_dir, org=scope.org)
        print_success(console, f"Registered GitHub App '{slug}'.")

    if not _confirm_installation(target_dir, scope, creds):
        print_error(
            console,
            "App not installed",
            f"The App is still not installed on '{_owner_of(scope)}'. Install it, then re-run setup.",
        )
        return 1

    bot_handle = _bot_handle_for(slug, scope)
    deposit_secrets(
        target_dir,
        creds,
        anthropic_key=resolved_key,
        bot_handle=bot_handle,
        scope=scope,
    )
    print_success(console, "Deposited App credentials and bot handle as Actions secrets/variables.")

    pr_url = land_workflows(target_dir, branch=_SETUP_BRANCH)
    if pr_url == WORKFLOWS_ALREADY_INSTALLED:
        print_info(console, "Workflow files already present on this repository; nothing to land.")
    else:
        print_success(console, f"Opened workflow PR: {pr_url}")
        print_info(
            console,
            "Merge the PR, then request reviews with a trusted `@<bot> review` comment.",
        )
    return 0
