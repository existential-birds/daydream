"""Resolve test recipes and execute bounded host-side test commands.

Passing requires exit status zero without timeout. Output and post-exit
pipe drains are capped; timeouts kill the entire process group.
"""

import asyncio
import codecs
import hashlib
import json
import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, TypeAdapter

from daydream.backends._subprocess import terminate_process
from daydream.json_utils import atomic_write_json, dataclass_payload, read_json_object
from daydream.redaction import redact_structured_text
from daydream.repository_paths import (
    canonicalize_working_directory,
    path_is_confined,
)
from daydream.trajectory import DaydreamPhase, host_phase_scope

_REDACTED_ENV_VAR = "[REDACTED_ENV_VAR]"

# Env values shorter than this are not secret-shaped: inherited trivial values
# (e.g. ``SHLVL=1``, ``CI=true``) collide with ordinary output ("1 failed"),
# so blanket-replacing them mangles exactly the text that feeds the failure
# gate and the fix prompt. Only longer (secret-shaped) values are scrubbed.
_MIN_REDACTED_ENV_VALUE_LENGTH = 8

# Cap on the merged output buffer retained, in characters. A chatty suite
# (``pytest -s -v`` with logging) must not balloon the host process or feed an
# unbounded string into the next agent turn; the prompt-build-time tail
# truncation runs after this cap, so the capped buffer is the outer bound. A
# cap hit is recorded explicitly as ``output_truncated`` -- never a silent drop.
_MERGED_OUTPUT_LIMIT_CHARS = 512 * 1024

#: Bump whenever the recipe payload shape changes so a stale persisted recipe
#: can never be read as the current contract (mirrors ``REUSE_KEY_FORMAT``).
RECIPE_FORMAT: int = 2

#: The persisted recipe's filename inside the run's deep directory.
TEST_RECIPE_FILENAME = "test-recipe.json"

#: The closed provenance domain of a resolved fact.
FactSource = Literal["cli", "config", "admitted", "derived", "unresolved"]


class MissingTestCommandError(RuntimeError):
    """No configured test command, or its shell-word syntax is invalid."""


def _select_raw_test_command(
    config: object, run_config: object
) -> tuple[str | None, Literal["cli", "config"]]:
    """Select a truthy CLI command before the config value; parsing happens separately."""
    cli_value = getattr(run_config, "test_command", None)
    if cli_value:
        return cli_value, "cli"
    return getattr(config, "test_command", None), "config"


def _string_tuple(value: object) -> tuple[str, ...]:
    """Require a JSON array containing only strings."""
    if isinstance(value, list) and all(isinstance(entry, str) for entry in value):
        return tuple(value)
    raise ValueError("expected an array of strings")



def _admit_recipe_command(raw: Any) -> Any:
    """Persisted unresolved commands require a value key but never admit its contents."""
    if not isinstance(raw, dict):
        return raw
    value, source = raw["value"], raw["source"]
    return {**raw, "value": None if source == "unresolved" else _string_tuple(value)}


_RecipeString = Annotated[str, Field(strict=True)]
_RecipeStrings = Annotated[tuple[str, ...], BeforeValidator(_string_tuple)]

@dataclass(frozen=True)
class ResolvedFact:
    """A resolved value and provenance, or None with source="unresolved"."""

    value: str | tuple[str, ...] | None
    source: FactSource

    @property
    def resolved(self) -> bool:
        """True only for a non-``None`` value with a real provenance."""
        return self.value is not None and self.source != "unresolved"


def _parse_test_command(
    config: object, run_config: object
) -> tuple[tuple[str, ...], Literal["cli", "config"]]:
    """Parse CLI-over-config argv, with actionable errors for missing or invalid commands."""
    raw, source = _select_raw_test_command(config, run_config)
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
        return tuple(shlex.split(raw)), source
    except ValueError as exc:
        raise MissingTestCommandError(
            "The configured test_command could not be parsed as shell argv "
            f"(unbalanced or unterminated quote): {raw!r}. Set --test-command "
            "or the `test_command` config key to a valid shell command."
        ) from exc


def canonical_test_command(config: object, run_config: object) -> list[str]:
    """Split CLI-over-config test argv; missing or invalid commands raise MissingTestCommandError."""
    return list(_parse_test_command(config, run_config)[0])


def resolve_test_command_fact(config: object, run_config: object) -> ResolvedFact:
    """Resolve command argv and provenance; missing or invalid input returns an unresolved fact."""
    try:
        argv, source = _parse_test_command(config, run_config)
        return ResolvedFact(value=argv, source=source)
    except MissingTestCommandError:
        return ResolvedFact(value=None, source="unresolved")


class RecipeConfinementError(ValueError):
    """The resolved package cwd escapes the worktree, including through symlinks."""


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
    """Package identity for a repo-relative cwd ("." at the root).

    The first declared lockfile selects the runner; .python-version precedes
    requires-python. Unreadable config inputs produce a None digest and named
    absent_components, never a placeholder.
    """

    cwd_relative: _RecipeString
    runner: _RecipeString | None
    interpreter: _RecipeString | None
    config_digest: _RecipeString | None
    absent_components: _RecipeStrings


def _nearest_package_dir(repo_root: Path, cwd_relative: str) -> Path:
    """Find the nearest manifest-bearing ancestor, bounded by the worktree root."""
    current = repo_root if cwd_relative == "." else repo_root / cwd_relative
    while True:
        if any((current / name).exists() for name in _PACKAGE_MANIFESTS):
            return current
        if current == repo_root:
            return current
        current = current.parent


def _package_input_text(payload: bytes) -> str:
    """Preserve text-file newline translation while hashing the exact admitted bytes."""
    return payload.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _package_inputs(package_dir: Path, cwd_relative: str) -> PackageResolution:
    """Derive interpreter and digest from one bounded per-file input observation."""
    names = [name for name in _PACKAGE_MANIFESTS if (package_dir / name).exists()]
    selected = next(((name, runner) for name, runner in _RUNNER_LOCKFILES if (package_dir / name).exists()), None)
    if selected is not None:
        names.append(selected[0])
    pinned = (package_dir / _PYTHON_VERSION_FILE).exists()
    if pinned:
        names.append(_PYTHON_VERSION_FILE)
    interpreter: str | None = None
    entries: dict[str, str] = {}
    absent: tuple[str, ...] = ()
    for name in sorted(set(names)):
        try:
            payload = (package_dir / name).read_bytes()
        except OSError:
            if not absent:
                absent = (name,)
            continue
        if name == _PYTHON_VERSION_FILE:
            interpreter = _package_input_text(payload).strip() or None
        elif name == "pyproject.toml" and not pinned:
            try:
                data = tomllib.loads(_package_input_text(payload))
            except tomllib.TOMLDecodeError:
                pass
            else:
                value = data.get("project", {}).get("requires-python")
                if isinstance(value, str) and value.strip():
                    interpreter = value.strip()
        entries[name] = hashlib.sha256(name.encode("utf-8") + b"\0" + payload).hexdigest()
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return PackageResolution(
        cwd_relative=cwd_relative,
        runner=None if selected is None else selected[1],
        interpreter=interpreter,
        config_digest=None if absent else hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        absent_components=absent,
    )


def resolve_package(repo_root: Path, start: Path) -> PackageResolution:
    """Resolve package metadata from the nearest manifest ancestor of the command cwd.

    The cwd remains repo-relative and confinement-checked; outward paths raise
    RecipeConfinementError. Metadata lookup never walks above repo_root.
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
    return _package_inputs(package_dir, cwd_relative)


@dataclass
class TestExecutionResult:
    """Host test outcome: completed excludes timeouts; incomplete includes timeout or capped output.

    These evidence flags do not change passed.
    """

    # Not a pytest test class despite the name prefix.
    __test__ = False

    exit_status: int
    timed_out: bool
    merged_output: str
    completed: bool = True
    output_truncated: bool = False
    incomplete: bool = False

    @property
    def passed(self) -> bool:
        """Single source of truth: exit status (a timeout is never a pass)."""
        return self.exit_status == 0 and not self.timed_out


@dataclass(frozen=True)
class TestExecutionIdentity:
    """Complete execution identity for pure evidence-reuse comparison.

    Only passed host runs authorize reuse. Agent verdicts, timeouts, and
    truncated runs are non-authoritative; payload() persists all fields.
    """

    # Not a pytest test class despite the name prefix.
    __test__ = False

    session_id: str
    argv: tuple[str, ...]
    cwd_relative: str
    runner: str | None
    interpreter: str | None
    config_digest: str | None
    absent_components: tuple[str, ...]
    input_tree_key: str
    output_tree_key: str
    head_sha: str
    branch: str
    kind: Literal["host", "agent"]
    outcome: Literal["passed", "failed", "timed-out", "truncated"]

    @property
    def reusable(self) -> bool:
        """True only for a host execution that reached a passing exit."""
        return self.kind == "host" and self.outcome == "passed"

    def payload(self) -> dict[str, Any]:
        """Return the strict-JSON projection of every identity component."""
        return dataclass_payload(self)


@dataclass(frozen=True)
class RecipeCandidate:
    """A suggestion from the closed manifest runner set, never an authoritative command."""

    argv: _RecipeStrings
    provenance: Literal["manifest"]


@dataclass(frozen=True)
class TestRecipe:
    """Command, package and suite facts resolved once for all consumers in a run."""

    command: Annotated[ResolvedFact, BeforeValidator(_admit_recipe_command)]
    package: PackageResolution
    declared: _RecipeStrings
    candidate: RecipeCandidate | None

    # Not a pytest test class despite the name prefix.
    __test__ = False


def _manifest_candidate(runner: str | None) -> RecipeCandidate | None:
    """Suggest pytest through a recognized runner, or bare pytest for pip."""
    if runner in {"uv", "poetry", "pipenv"}:
        return RecipeCandidate(argv=(runner, "run", "pytest"), provenance="manifest")
    if runner == "pip":
        return RecipeCandidate(argv=("pytest",), provenance="manifest")
    return None


def resolve_test_recipe(
    config: object,
    run_config: object,
    *,
    repo_root: Path,
    cwd: Path | None = None,
) -> TestRecipe:
    """Resolve CLI-over-config command and suites, confined package and manifest suggestion.

    An unresolved command cannot run; a candidate is only a suggestion.
    """
    command = resolve_test_command_fact(config, run_config)
    package = resolve_package(repo_root, cwd if cwd is not None else repo_root)
    declared = tuple(
        getattr(run_config, "test_required_suites", None)
        or getattr(config, "test_required_suites", None)
        or ()
    )
    return TestRecipe(
        command=command,
        package=package,
        declared=declared,
        candidate=_manifest_candidate(package.runner),
    )


def recipe_to_payload(recipe: TestRecipe) -> dict[str, Any]:
    """Persist every recipe field as a JSON-safe value, including its format version."""
    return {"format_version": RECIPE_FORMAT, **dataclass_payload(recipe)}



_RECIPE_ADAPTER = TypeAdapter(TestRecipe)


def _recipe_from_payload(payload: dict[str, Any]) -> TestRecipe | None:
    """Admit persisted native records; malformed input never supplies a partial recipe."""
    try:
        return _RECIPE_ADAPTER.validate_python({**payload, "candidate": payload.get("candidate")})
    except (KeyError, TypeError, ValueError):
        return None


def persist_test_recipe(deep_dir: Path, recipe: TestRecipe) -> Path:
    """Atomically persist *recipe* under *deep_dir*, returning its path."""
    path = deep_dir / TEST_RECIPE_FILENAME
    atomic_write_json(path, recipe_to_payload(recipe))
    return path


def load_test_recipe(deep_dir: Path) -> TestRecipe | None:
    """Load a current-format recipe; absent, unreadable, malformed or stale files return None."""
    payload = read_json_object(deep_dir / TEST_RECIPE_FILENAME)
    if payload.get("format_version") != RECIPE_FORMAT:
        return None
    return _recipe_from_payload(payload)


def _redact_merged(output: str, env: dict[str, str] | None) -> str:
    """Scrub structured secrets and literal environment values of at least eight characters.

    Short values collide with ordinary output. Record matches before replacement;
    if any survives (including within the marker), discard the entire field.
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
    """Run argv with concurrent, capped stdout/stderr capture and a wall-budget group kill.

    Redact the effective environment before retaining output. Bound pipe drain
    after exit, including pipes held by surviving descendants. Spawn errors
    propagate; trajectory records completion, timeout or failure.
    """
    effective_env = env if env is not None else dict(os.environ)
    async with host_phase_scope(DaydreamPhase.TEST_EXECUTION) as phase:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            env=effective_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        chunks: list[str] = []
        buffered = 0
        truncated = False

        async def _pump(stream: asyncio.StreamReader | None) -> None:
            nonlocal buffered, truncated
            if stream is None:
                return
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            while True:
                data = await stream.read(64 * 1024)
                text = decoder.decode(data, final=not data)
                remaining = _MERGED_OUTPUT_LIMIT_CHARS - buffered
                if len(text) > remaining:
                    truncated = True
                chunk = text[:remaining]
                if chunk:
                    chunks.append(chunk)
                    buffered += len(chunk)
                if not data:
                    return

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
            merged_output=_redact_merged("".join(chunks), effective_env),
            completed=not timed_out,
            output_truncated=truncated,
            incomplete=timed_out or truncated,
        )
