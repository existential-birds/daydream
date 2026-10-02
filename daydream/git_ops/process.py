"""The sole subprocess boundary for repository Git and GitHub CLI commands."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, overload
from urllib.parse import urlparse

from daydream.backends._subprocess import terminate_process
from daydream.git_ops.auth import INHERIT_GITHUB_AUTH, GitHubAuth
from daydream.git_ops.models import GitError, GitHubRequestBudget, GitTimeoutError, RateLimitError
from daydream.redaction import redact_text

_logger = logging.getLogger(__name__)

# Match HTTP 429 separately at word boundaries: URLs, SHAs, and sizes can
# contain those digits without indicating a rate limit.
_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "rate limit",
    "secondary rate limit",
)


# Header values must be masked before argv reaches logs or errors.
_SENSITIVE_HEADER_PREFIXES = ("authorization:",)

# Manifest-conversion codes are credentials embedded in the endpoint path.
# Diagnostics retain the route while masking its credential segment.
_APP_MANIFEST_CONVERSION_CODE_RE = re.compile(r"(/app-manifests/)[^/\s]+(/conversions)")
_GITHUB_TOKEN_RE = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")
# Start at the fixed authority delimiter. Searching for an arbitrary-length
# scheme at every input character makes long non-URL diagnostics quadratic.
_URL_USERINFO_RE = re.compile(r"://[^/@\s]+@")


def _redact_sensitive_text(text: str) -> str:
    """Mask manifest-conversion credentials in diagnostic strings only. Never apply this
    transform to live request arguments.
    """
    redacted = _APP_MANIFEST_CONVERSION_CODE_RE.sub(r"\1***\2", text)
    redacted = _GITHUB_TOKEN_RE.sub("***", redacted)
    redacted = _URL_USERINFO_RE.sub("://***@", redacted)
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(key)
        if token and len(token) >= 8:
            redacted = redacted.replace(token, "***")
    return redacted


def _redact_args(args: list[str]) -> list[str]:
    """Copy arguments with sensitive header values and manifest-conversion codes masked for
    diagnostics. Preserve header names and never mutate live request arguments.
    """
    redacted: list[str] = []
    for arg in args:
        if any(arg.lower().startswith(prefix) for prefix in _SENSITIVE_HEADER_PREFIXES):
            redacted.append(f"{arg.split(':', 1)[0]}: ***")
        else:
            redacted.append(arg)
    return [_redact_sensitive_text(a) for a in redacted]


def _require_ok(proc: subprocess.CompletedProcess[Any], context: str) -> None:
    """Raise ``GitError`` as ``context: stderr`` for failed text or binary captures."""
    if proc.returncode != 0:
        raise GitError(f"{context}: {os.fsdecode(proc.stderr).strip()}")


# --- Internal subprocess helpers --------------------------------------------


# Bounded retries let read-only queries survive temporary host starvation.
_GIT_TIMEOUT_RETRIES = 2

# Git invokes this helper with get/store/erase and protocol/host on stdin.
# Ambient gh auth keeps tokens out of argv, URLs, files, and Git config.
# Read at call time so the harness can substitute a recording wrapper.
GH_CREDENTIAL_HELPER = "!gh auth git-credential"

# Retries require caller-declared reads: GraphQL queries and mutations both
# use POST. Read both budgets at call time so the fake-gh harness can bound
# subprocess stalls without inheriting production network timeouts.
_GH_DEFAULT_TIMEOUT = 60
_GH_DEFAULT_RETRIES = 2


def _gh_int_env(name: str, default: int, *, is_valid: Callable[[int], bool], invalid_msg: str) -> int:
    """Read an integer setting; invalid configured values warn and fall back."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        _logger.warning(
            "%s=%r is not a valid integer; using default %d",
            name,
            raw,
            default,
        )
        return default
    if not is_valid(value):
        _logger.warning(
            "%s=%r %s; using default %d",
            name,
            raw,
            invalid_msg,
            default,
        )
        return default
    return value


def _gh_timeout() -> int:
    """Default ``gh`` subprocess timeout in seconds (env-overridable)."""
    return _gh_int_env(
        "DAYDREAM_GH_TIMEOUT_SECONDS",
        _GH_DEFAULT_TIMEOUT,
        is_valid=lambda value: value > 0,
        invalid_msg="must be positive",
    )


def _gh_retries() -> int:
    """Read-only ``gh`` timeout-retry budget (env-overridable)."""
    return _gh_int_env(
        "DAYDREAM_GH_TIMEOUT_RETRIES",
        _GH_DEFAULT_RETRIES,
        is_valid=lambda value: value >= 0,
        invalid_msg="is negative",
    )


def _retrying_subprocess(
    argv: list[str],
    *,
    display: str,
    cwd: Path,
    text: bool,
    input_data: str | bytes | None,
    env: Any | None,
    timeout: int | float,
    retries: int,
    scrub: Callable[[str], str] = str,
    suffix: str = "",
) -> subprocess.CompletedProcess[Any]:
    """Run *argv*, retrying only timeouts; *scrub* redacts failure text, *suffix* labels the exhausted error."""
    program = argv[0]
    last_timeout: subprocess.TimeoutExpired | None = None
    for attempt in range(retries + 1):
        try:
            return subprocess.run(  # noqa: S603 - arguments are not user-controlled
                argv,
                cwd=cwd,
                capture_output=True,
                text=text,
                timeout=timeout,
                shell=False,
                check=False,
                input=input_data,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            last_timeout = exc
            if attempt < retries:
                _logger.warning(
                    "%s timed out after %ss (attempt %d/%d); retrying",
                    program,
                    timeout,
                    attempt + 1,
                    retries + 1,
                )
        except (subprocess.SubprocessError, OSError) as exc:
            raise GitError(f"{program} {display} failed: {type(exc).__name__}: {scrub(str(exc))}") from exc

    raise GitTimeoutError(f"{program} {display} timed out after {timeout}s{suffix}") from last_timeout


@overload
def _run_git(
    repo: Path,
    args: list[str],
    *,
    timeout: int | float = 5,
    capture_bytes: Literal[True],
    retries: int = _GIT_TIMEOUT_RETRIES,
    input_text: str | None = None,
    input_bytes: bytes | None = None,
    env_cmd: Any | None = None,
    error_context: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Binary-capture variant: ``capture_bytes=True`` reads bytes stdout."""


@overload
def _run_git(
    repo: Path,
    args: list[str],
    *,
    timeout: int | float = 5,
    capture_bytes: Literal[False] = False,
    retries: int = _GIT_TIMEOUT_RETRIES,
    input_text: str | None = None,
    input_bytes: bytes | None = None,
    env_cmd: Any | None = None,
    error_context: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Text-capture variant (the default): decoded ``str`` stdout."""


def _run_git(
    repo: Path,
    args: list[str],
    *,
    timeout: int | float = 5,
    capture_bytes: bool = False,
    retries: int = _GIT_TIMEOUT_RETRIES,
    input_text: str | None = None,
    input_bytes: bytes | None = None,
    env_cmd: Any | None = None,
    error_context: str | None = None,
) -> subprocess.CompletedProcess[Any]:
    """Run ``git`` from *repo*, returning its completed process without checking its exit code.

    Text input is UTF-8 encoded for binary capture; raw byte input requires
    binary capture. Only timeouts are retried. Mutating callers pass
    ``retries=0`` to avoid repeating a partially completed command. Subprocess
    failures raise ``GitError``; exhausted timeouts raise ``GitTimeoutError``.
    """
    if input_text is not None and input_bytes is not None:
        raise GitError("git subprocess input must be text or bytes, not both")
    if input_bytes is not None and not capture_bytes:
        raise GitError("binary git subprocess input requires binary capture")
    display = " ".join(args)
    proc = _retrying_subprocess(
        ["git", *args],
        display=display,
        cwd=repo,
        text=not capture_bytes,
        input_data=(
            input_bytes
            if input_bytes is not None
            else input_text.encode("utf-8")
            if capture_bytes and input_text is not None
            else input_text
        ),
        env=env_cmd,
        timeout=timeout,
        retries=retries,
        suffix=f" ({retries + 1} attempts)",
    )
    if error_context is not None:
        _require_ok(proc, error_context)
    return proc


def _run_gh(
    repo: Path,
    args: list[str],
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    timeout: int | None = None,
    input_text: str | None = None,
    retries: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Run ``gh`` from *repo*, returning its text-decoded completed process.

    Authentication is resolved once per retry sequence. The default timeout
    comes from :func:`_gh_timeout`; mutating calls do not retry. Sensitive
    values can be sent through ``input_text`` rather than argv. Subprocess
    failures raise ``GitError``; exhausted timeouts raise ``GitTimeoutError``.
    """
    if timeout is None:
        timeout = _gh_timeout()
    environment = auth.environment_for_request()
    env = dict(environment) if environment is not None else None
    display = " ".join(_redact_args(args))
    suffix = f" ({retries + 1} attempts)" if retries else ""
    return _retrying_subprocess(
        ["gh", *args],
        display=display,
        cwd=repo,
        text=True,
        input_data=input_text,
        env=env,
        timeout=timeout,
        retries=retries,
        scrub=_redact_sensitive_text,
        suffix=suffix,
    )


def _credential_helper_args(helper: str | None = None) -> list[str]:
    """Select the call-time helper for one command without changing Git config."""
    return ["-c", f"credential.helper={helper or GH_CREDENTIAL_HELPER}"]


def _gh_error_for(message: str, stderr: str) -> GitError:
    """Classify stderr rate-limit indicators, including 429 or rate-associated 403, and
    parse an integer retry hint when present. Other failures remain ordinary GitError
    values.
    """
    lowered = stderr.lower()
    redacted_message = _redact_sensitive_text(message)
    is_rate_limit = (
        any(marker in lowered for marker in _RATE_LIMIT_MARKERS)
        or re.search(r"\b429\b", lowered) is not None
        or ("403" in lowered and "rate" in lowered)
    )
    if not is_rate_limit:
        return GitError(redacted_message)
    retry_after: float | None = None
    match = re.search(r"retry[- ]after[:\s]+(\d+)", lowered)
    if match:
        retry_after = float(match.group(1))
    return RateLimitError(redacted_message, retry_after=retry_after)


async def _run_gh_async(
    repo: Path,
    args: list[str],
    *,
    auth: GitHubAuth = INHERIT_GITHUB_AUTH,
    budget: GitHubRequestBudget,
) -> subprocess.CompletedProcess[str]:
    """Run one cancellable ``gh`` process within a shared float deadline.

    Authentication resolution runs in a worker so a contended refresh lock
    cannot block the event loop. Deadline expiry or task cancellation prevents
    this request from spawning ``gh``. A synchronous refresh already running
    in the worker may still finish for its own auth session afterward.
    """
    auth_timeout = budget.next_timeout()
    try:
        environment = await asyncio.wait_for(
            asyncio.to_thread(auth.environment_for_request),
            timeout=auth_timeout,
        )
    except TimeoutError:
        raise GitTimeoutError(f"GitHub authentication timed out after {auth_timeout:g}s") from None
    env = dict(environment) if environment is not None else None
    timeout = budget.next_timeout()
    try:
        proc = await asyncio.create_subprocess_exec(
            "gh",
            *args,
            cwd=repo,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        command = " ".join(_redact_args(args))
        detail = _redact_sensitive_text(str(exc))
        raise GitError(f"gh {command} failed: {type(exc).__name__}: {detail}") from exc

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as exc:
        await terminate_process(proc)
        command = " ".join(_redact_args(args))
        raise GitTimeoutError(f"gh {command} timed out after {timeout:g}s") from exc
    except BaseException:
        await terminate_process(proc)
        raise

    try:
        stdout = stdout_bytes.decode("utf-8")
        stderr = stderr_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        await terminate_process(proc)
        raise GitError("gh API request returned invalid UTF-8") from exc
    return subprocess.CompletedProcess(args, proc.returncode or 0, stdout, stderr)


def _safe_url_desc(url: str) -> str:
    """Describe a URL as host/path; redact local paths and scp-form remotes."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return redact_text(url)
    if parsed.scheme and parsed.hostname:
        return f"{parsed.hostname}{parsed.path}"
    return redact_text(url)


def _run_clone(remote_url: str, cmd: list[str], timeout: int, env: dict[str, str] | None = None) -> None:
    """Run a clone with redacted GitErrors; ``env=None`` inherits the parent environment."""
    try:
        proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
            cmd,  # noqa: S607 - git is a trusted command
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise GitError(
            f"git clone {_safe_url_desc(remote_url)} failed: {type(exc).__name__}: {redact_text(str(exc))}"
        ) from exc
    if proc.returncode != 0:
        raise GitError(f"git clone {_safe_url_desc(remote_url)} failed: {redact_text(proc.stderr.strip())}")
