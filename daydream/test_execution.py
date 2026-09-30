"""Bounded host-side test runner.

Executes shell test commands as real subprocesses (issue #726): streaming +
redacting merged output, wall-budget timeout that kills the whole process
group via the existing ``backends/_subprocess.terminate_process``, returning a
typed :class:`TestExecutionResult` whose ``passed`` is derived only from exit
status. "Green" means the subprocess exited 0 — nothing else.

The merged output buffer and the post-exit pipe drain are both capped, so a
suite that floods output or leaves a grandchild holding the pipe write ends
cannot balloon the host process or hang the run.
"""

import asyncio
import hashlib
import json
import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from daydream.backends._subprocess import terminate_process
from daydream.repository_paths import (
    canonicalize_working_directory,
    path_is_confined,
)
from daydream.trajectory import DaydreamPhase, host_phase_scope, redact_structured_text

_REDACTED_ENV_VAR = "[REDACTED_ENV_VAR]"

# Env values shorter than this are not secret-shaped: inherited trivial values
# (e.g. ``SHLVL=1``, ``CI=true``) collide with ordinary output ("1 failed"),
# so blanket-replacing them mangles exactly the text that feeds the failure
# gate and the fix prompt. Only longer (secret-shaped) values are scrubbed.
_MIN_REDACTED_ENV_VALUE_LENGTH = 8

# Cap on the merged output buffer retained, in characters. A chatty suite
# (``pytest -s -v`` with logging) must not balloon the host process or feed an
# unbounded string into the next agent turn; the prompt-build-time tail
# truncation runs after this cap, so the capped buffer is the outer bound.
_MERGED_OUTPUT_LIMIT_CHARS = 512 * 1024


class MissingTestCommandError(RuntimeError):
    """Raised when no canonical test command is configured.

    Issue #726: a "green" daydream run must mean the target repo's test
    command really exited 0 as a host-side subprocess — the agent never
    guesses the command. When neither the CLI flag nor a config file declares
    one, :func:`canonical_test_command` raises this rather than return an
    empty or unknown command. A configured value that cannot be parsed as
    shell argv (an unbalanced/unterminated quote fails ``shlex.split``) is the
    same config-error class: it is wrapped into this exception instead of
    letting ``ValueError`` escape at the call sites. Production call sites
    currently catch it in :func:`daydream.phases._canonical_test_cmd` and fall
    back to the warned, deprecated agent-run path during the #726 transition;
    the exception still fails closed anywhere it is let through.
    """


def _select_raw_test_command(
    config: object, run_config: object
) -> tuple[str | None, Literal["cli", "config"]]:
    """Apply CLI-over-config precedence, returning the raw value and its origin.

    The selection is the existing ``or`` chain: a truthy CLI value wins, else
    the config-file value (``None``/empty included). Whether the selected value
    is usable is the caller's concern — see :func:`canonical_test_command`.
    """
    cli_value = getattr(run_config, "test_command", None)
    if cli_value:
        return cli_value, "cli"
    return getattr(config, "test_command", None), "config"


@dataclass(frozen=True)
class ResolvedFact:
    """One resolved run fact together with where it came from.

    ``value`` is ``None`` exactly when the fact could not be resolved, and
    ``source == "unresolved"`` names that absence (Pattern C: a named miss,
    never a placeholder). ``resolved`` is the single predicate callers gate on.
    """

    value: str | tuple[str, ...] | None
    source: Literal["cli", "config", "admitted", "derived", "unresolved"]

    @property
    def resolved(self) -> bool:
        """True only for a non-``None`` value with a real provenance."""
        return self.value is not None and self.source != "unresolved"


def canonical_test_command(config: object, run_config: object) -> list[str]:
    """Resolve the canonical test command as shell-word-split argv.

    Precedence (highest first): the CLI ``--test-command`` flag
    (``run_config.test_command``), then the config-file ``test_command`` key
    (``.daydream.toml`` root keys override ``[tool.daydream]`` in
    ``pyproject.toml`` — that merge already happened in
    :func:`daydream.config_file.load_file_config`). When neither is set,
    raise :class:`MissingTestCommandError` naming the key, the precedence
    sources checked, and exactly what to set — never fall back to an empty or
    unknown command.
    """
    raw, _ = _select_raw_test_command(config, run_config)
    if not raw or not raw.strip():
        raise MissingTestCommandError(
            "No canonical test command is configured; refusing to run tests "
            "without one (issue #726). Set it via the --test-command flag, or "
            "in a config file under the `test_command` key: either the root of "
            ".daydream.toml (highest precedence) or the [tool.daydream] table "
            "in pyproject.toml. "
            "Example: daydream --test-command 'uv run pytest -n auto' /path/to/project"
        )
    try:
        return shlex.split(raw)
    except ValueError as exc:
        # Unbalanced/unterminated quotes: ``shlex.split`` cannot build an argv.
        # Same config-error class as a missing command — the call sites fail
        # soft (warn + fall back) instead of crashing the run (issue #726).
        raise MissingTestCommandError(
            "The configured test_command could not be parsed as shell argv "
            f"(unbalanced or unterminated quote): {raw!r}. Set --test-command "
            "or the `test_command` config key to a valid shell command."
        ) from exc


def resolve_test_command_fact(config: object, run_config: object) -> ResolvedFact:
    """Resolve the test command into a provenance-bearing :class:`ResolvedFact`.

    Non-raising counterpart of :func:`canonical_test_command`: it applies the
    same CLI-over-config precedence but a missing, empty/whitespace, or
    ``shlex``-unparseable value yields the unresolved fact instead of an
    exception. The unresolved fact is the named miss — there is no guessed or
    placeholder command.
    """
    raw, source = _select_raw_test_command(config, run_config)
    if not raw or not raw.strip():
        return ResolvedFact(value=None, source="unresolved")
    try:
        return ResolvedFact(value=tuple(shlex.split(raw)), source=source)
    except ValueError:
        return ResolvedFact(value=None, source="unresolved")


class RecipeConfinementError(ValueError):
    """Raised when a package directory resolves outside the worktree.

    A package cwd is a security boundary: the host test runner will execute in
    it, so a start path that escapes the worktree (``..`` traversal, an
    absolute path elsewhere) is rejected rather than silently clamped to the
    root. Confinement is decided by :func:`path_is_confined`, the single
    primitive that also settles symlinked components.
    """


#: The closed set of package manifests that make a directory a package root.
#: The digest inputs and the nearest-package walk both read this one constant.
_PACKAGE_MANIFESTS: tuple[str, ...] = ("pyproject.toml", "setup.cfg", "setup.py")

#: Interpreter declaration read before ``requires-python``.
_PYTHON_VERSION_FILE = ".python-version"

#: The closed runner-input set, first match wins: a lockfile names the package
#: manager that owns the package (``requirements.txt`` is the ``pip`` fallback).
_RUNNER_LOCKFILES: tuple[tuple[str, str], ...] = (
    ("uv.lock", "uv"),
    ("poetry.lock", "poetry"),
    ("Pipfile.lock", "pipenv"),
    ("requirements.txt", "pip"),
)


@dataclass(frozen=True)
class PackageResolution:
    """Resolved facts about the package a test command will run within.

    ``cwd_relative`` is the repo-relative posix directory (``"."`` for the
    worktree root) the command runs in. ``runner`` names the package manager
    discovered from the closed lockfile set, or ``None`` when none is declared.
    ``interpreter`` is the declared Python version (``.python-version`` first,
    then the manifest's ``requires-python``), or ``None``. ``config_digest``
    digests the package's existing config inputs; any input that exists but
    cannot be read makes the digest ``None`` and names it in
    ``absent_components`` (Pattern C: a named miss, never a placeholder).
    """

    cwd_relative: str
    runner: str | None
    interpreter: str | None
    config_digest: str | None
    absent_components: tuple[str, ...]


def _nearest_package_dir(repo_root: Path, cwd_relative: str) -> Path:
    """Walk up from ``cwd_relative`` to the nearest manifest-bearing directory.

    The walk is bounded by the worktree root: an ancestor above the root is
    never inspected, so the root remains the floor and maps to the root itself.
    """
    current = repo_root if cwd_relative == "." else repo_root / cwd_relative
    while True:
        if any((current / name).exists() for name in _PACKAGE_MANIFESTS):
            return current
        if current == repo_root:
            return current
        current = current.parent


def _resolve_runner(package_dir: Path) -> str | None:
    """Return the first lockfile-named package manager present, else ``None``."""
    for name, runner in _RUNNER_LOCKFILES:
        if (package_dir / name).exists():
            return runner
    return None


def _resolve_interpreter(package_dir: Path) -> str | None:
    """Read ``.python-version`` first, then the manifest's ``requires-python``."""
    pinned = package_dir / _PYTHON_VERSION_FILE
    if pinned.exists():
        try:
            value = pinned.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None
    manifest = package_dir / "pyproject.toml"
    if manifest.exists():
        try:
            data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return None
        value = data.get("project", {}).get("requires-python")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _config_input_names(package_dir: Path, runner: str | None) -> tuple[str, ...]:
    """Name the package's existing config inputs in a deterministic order.

    The set is closed: the present manifests, the resolved runner lockfile,
    and ``.python-version`` when it exists. Inputs are keyed by existence (not
    file-ness) so a malformed directory-shaped input still becomes an absent
    component rather than silently vanishing from the identity.
    """
    names = [name for name in _PACKAGE_MANIFESTS if (package_dir / name).exists()]
    if runner is not None:
        for name, _ in _RUNNER_LOCKFILES:
            if (package_dir / name).exists():
                names.append(name)
                break
    if (package_dir / _PYTHON_VERSION_FILE).exists():
        names.append(_PYTHON_VERSION_FILE)
    return tuple(sorted(set(names)))


def _config_digest(
    package_dir: Path, names: tuple[str, ...]
) -> tuple[str | None, tuple[str, ...]]:
    """Digest the named config inputs, or return a named miss.

    The discipline follows ``blob_map_digest`` (``deep/reuse_key.py``): each
    input contributes ``sha256(name + b"\\0" + bytes)`` to a canonical map, and
    any unreadable input makes the whole digest ``None`` with the input named
    in ``absent_components``. There is no placeholder digest and no empty-string
    fallback.
    """
    entries: dict[str, str] = {}
    for name in names:
        try:
            payload = (package_dir / name).read_bytes()
        except OSError:
            return None, (name,)
        entries[name] = hashlib.sha256(
            name.encode("utf-8") + b"\0" + payload
        ).hexdigest()
    if not entries:
        # No config inputs at all is a real value, not a miss.
        return hashlib.sha256(b"{}").hexdigest(), ()
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), ()


def resolve_package(repo_root: Path, start: Path) -> PackageResolution:
    """Resolve the package a test command runs within, bounded by the worktree.

    ``start`` is the directory the command was anchored to (the worktree root
    for the repo-root case). The nearest manifest-bearing directory at or above
    it names the package; the worktree root maps to ``cwd_relative == "."``.
    The resolved cwd is confinement-checked with :func:`path_is_confined` and
    rejected with :class:`RecipeConfinementError` when it escapes — never
    silently clamped. Runner, interpreter, and config-input identity are read
    from the package directory.
    """
    start_abs = start if start.is_absolute() else repo_root / start
    relative = os.path.relpath(str(start_abs), str(repo_root))
    if not path_is_confined(repo_root, relative):
        raise RecipeConfinementError(
            f"Package directory {start!s} resolves outside the worktree "
            f"{repo_root!s}; refusing to run tests there."
        )
    cwd_relative = canonicalize_working_directory(repo_root, relative)
    package_dir = _nearest_package_dir(repo_root, cwd_relative)
    runner = _resolve_runner(package_dir)
    interpreter = _resolve_interpreter(package_dir)
    config_digest, absent = _config_digest(
        package_dir, _config_input_names(package_dir, runner)
    )
    return PackageResolution(
        cwd_relative=cwd_relative,
        runner=runner,
        interpreter=interpreter,
        config_digest=config_digest,
        absent_components=absent,
    )


@dataclass
class TestExecutionResult:
    """Outcome of one host-side test-command run."""

    # Not a pytest test class despite the name prefix.
    __test__ = False

    exit_status: int
    timed_out: bool
    merged_output: str

    @property
    def passed(self) -> bool:
        """Single source of truth: exit status (a timeout is never a pass)."""
        return self.exit_status == 0 and not self.timed_out


def _redact_merged(output: str, env: dict[str, str] | None) -> str:
    """Scrub secret-shaped content and the literal env values from the buffer.

    The env vars handed to the test command are treated as sensitive: their
    values are redacted wherever they appear in the merged output, in addition
    to the structured pattern-based scrub. Only secret-shaped values (at least
    ``_MIN_REDACTED_ENV_VALUE_LENGTH`` characters) are scrubbed: inherited
    trivial values like ``SHLVL=1`` or ``CI=true`` collide with ordinary output
    and are deliberately left alone. The fail-closed gate keys off the
    PRE-replacement buffer: the replace loop removes every occurrence, so a
    membership test run after it could never observe a survivor. A value that
    was present before replacement and still survives after it (the blanket
    replace cannot clear a value the replacement marker itself carries, e.g.
    ``REDACTED``) degrades the whole field to ``[REDACTION_FAILED]``.
    """
    redacted = redact_structured_text(output)
    env_values = [
        value
        for value in (env or {}).values()
        if value and len(value) >= _MIN_REDACTED_ENV_VALUE_LENGTH
    ]
    # Membership test against the pre-replacement buffer: the loop below can
    # only be asked to remove values that are present here.
    found = [value for value in env_values if value in redacted]
    for value in found:
        redacted = redacted.replace(value, _REDACTED_ENV_VAR)
    # Fail closed if a value the scrub was asked to remove still survives.
    if any(value in redacted for value in found):
        return "[REDACTION_FAILED]"
    return redacted


async def run_test_command(
    cmd: list[str],
    *,
    cwd: Path,
    wall_budget_s: float,
    env: dict[str, str] | None = None,
) -> TestExecutionResult:
    """Run *cmd* as a real subprocess with a wall-budget group kill.

    Spawns with ``start_new_session=True`` so the budget timeout can terminate
    the whole process group (reusing the shielded ``terminate_process``).
    Both pipes are streamed concurrently into one merged buffer; the merged
    output is redacted before it lands in the result. Spawn errors propagate —
    never a bogus default result.

    With no ``env`` given, the subprocess inherits the parent environment and
    the scrub covers exactly those inherited values (the only env values that
    can appear in the merged output); with ``env`` given, the scrub covers
    exactly that dict.

    With an active trajectory recorder, the run is bracketed by
    ``test-execution`` phase events carrying ``duration_ms`` and a
    ``stop_reason`` of ``completed`` / ``timed_out`` / ``failed`` (issue #726).
    """
    effective_env = env if env is not None else dict(os.environ)
    async with host_phase_scope(DaydreamPhase.TEST_EXECUTION) as phase:
        return await _run_test_command_inner(cmd, cwd=cwd, wall_budget_s=wall_budget_s,
                                             env=effective_env, phase=phase)


async def _run_test_command_inner(
    cmd: list[str],
    *,
    cwd: Path,
    wall_budget_s: float,
    env: dict[str, str] | None,
    phase: Any,
) -> TestExecutionResult:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    chunks: list[str] = []
    buffered = 0

    async def _pump(stream: asyncio.StreamReader | None) -> None:
        nonlocal buffered
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            if buffered >= _MERGED_OUTPUT_LIMIT_CHARS:
                continue
            text = line.decode(errors="replace")
            remaining = _MERGED_OUTPUT_LIMIT_CHARS - buffered
            if len(text) > remaining:
                text = text[:remaining]
            buffered += len(text)
            chunks.append(text)

    pump_stdout = asyncio.ensure_future(_pump(proc.stdout))
    pump_stderr = asyncio.ensure_future(_pump(proc.stderr))

    timed_out = False
    try:
        exit_status = await asyncio.wait_for(proc.wait(), wall_budget_s)
    except TimeoutError:
        timed_out = True
        await terminate_process(proc)
        exit_status = proc.returncode if proc.returncode is not None else -1

    # Bound the post-exit drain with a salvage window. After a group kill the
    # pipes hit EOF; a suite-spawned grandchild that inherited the pipe write
    # ends can keep them open past a direct child's exit (a live-server
    # fixture, a daemonizing helper), so the success path needs the same bound.
    # A CancelledError here is only reachable after a clean exit, so it
    # propagates; the timed-out path swallows it as before.
    try:
        await asyncio.wait_for(
            asyncio.gather(pump_stdout, pump_stderr), timeout=5.0
        )
    except TimeoutError:
        pass
    except asyncio.CancelledError:
        if not timed_out:
            raise
    if timed_out:
        phase.stop_reason = "timed_out"
    return TestExecutionResult(
        exit_status=exit_status,
        timed_out=timed_out,
        merged_output=_redact_merged("".join(chunks), env),
    )
