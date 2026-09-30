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
import codecs
import hashlib
import json
import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from daydream.backends._subprocess import terminate_process
from daydream.json_utils import atomic_write_json
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
# truncation runs after this cap, so the capped buffer is the outer bound. A
# cap hit is recorded explicitly as ``output_truncated`` -- never a silent drop.
_MERGED_OUTPUT_LIMIT_CHARS = 512 * 1024

#: Bump whenever the recipe payload shape changes so a stale persisted recipe
#: can never be read as the current contract (mirrors ``REUSE_KEY_FORMAT``).
RECIPE_FORMAT: int = 1

#: The persisted recipe's filename inside the run's deep directory.
TEST_RECIPE_FILENAME = "test-recipe.json"

#: The closed provenance domain of a resolved fact.
FactSource = Literal["cli", "config", "admitted", "derived", "unresolved"]


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
    source: FactSource

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
    """Outcome of one host-side test-command run.

    ``completed`` records that the process reached its own exit (a timeout is
    the one thing that makes it ``False``), and ``incomplete`` records that the
    retained evidence is missing something (a timeout or a capped output
    buffer). Neither flag changes ``passed``.
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
class RequiredRun:
    """A run of the single configured command, eligible to satisfy a contract.

    Only this type may be passed to :meth:`RequiredContract.satisfied_by`;
    a :class:`TargetedCheckRun` is structurally barred because a narrowed
    ``-k``/selector check is not the authoritative full-suite gate (MH6).
    """

    argv: tuple[str, ...]
    cwd_relative: str
    result: "TestExecutionResult"


@dataclass(frozen=True)
class TargetedCheckRun:
    """A narrowed check (``-k``/single file) — never a required-suite gate."""

    argv: tuple[str, ...]
    selector: str | None
    result: "TestExecutionResult"
    cwd_relative: str = "."


@dataclass(frozen=True)
class RequiredContract:
    """The declared required-suite contract for the configured command.

    ``declared`` names the suite ids the single configured command is the
    authoritative gate for (declaration only — no second runner executes).
    ``argv`` is the resolved command, or ``None`` when nothing is configured,
    in which case the contract is never satisfied. ``satisfied_by`` accepts
    only a :class:`RequiredRun`; anything else raises ``TypeError`` so a
    targeted check can never stand in for the required suite.
    """

    declared: tuple[str, ...]
    argv: tuple[str, ...] | None
    source: Literal["cli", "config", "unresolved"]

    def satisfied_by(self, run: RequiredRun) -> bool:
        """True only for a passing run of exactly this contract's command."""
        if not isinstance(run, RequiredRun):
            raise TypeError(
                "RequiredContract.satisfied_by accepts only a RequiredRun; "
                "a targeted check cannot satisfy the required contract."
            )
        if self.argv is None:
            return False
        return run.argv == self.argv and run.result.passed


@dataclass(frozen=True)
class RecipeCandidate:
    """A manifest-derived suggestion, never the authoritative command.

    Produced only from the closed runner set (:data:`_RUNNER_LOCKFILES`); no
    arbitrary repository text ever becomes a candidate (spec MH4).
    """

    argv: tuple[str, ...]
    provenance: Literal["manifest"]


@dataclass(frozen=True)
class RecipeIdentity:
    """The versioned identity of one resolved recipe.

    ``digest`` is a content digest over the command/package facts plus the
    per-package config-input digest, or ``None`` when a config input could not
    be read — in which case ``absent_components`` names it (Pattern C: a named
    miss, never a placeholder digest).
    """

    digest: str | None
    absent_components: tuple[str, ...]
    format_version: int = RECIPE_FORMAT


@dataclass(frozen=True)
class TestRecipe:
    """One resolved test recipe: everything a run needs to run tests.

    Every fact is resolved exactly once in the deep preamble and then consumed
    from this single value — no consumer re-derives the command, the package,
    or the identity.
    """

    command: ResolvedFact
    package: PackageResolution
    required: RequiredContract
    candidate: RecipeCandidate | None
    identity: RecipeIdentity
    format_version: int = RECIPE_FORMAT

    # Not a pytest test class despite the name prefix.
    __test__ = False


def _manifest_candidate(runner: str | None) -> RecipeCandidate | None:
    """Map a resolved runner to the closed manifest candidate, or ``None``.

    The mapping is total over :data:`_RUNNER_LOCKFILES`: ``uv``/``poetry``/
    ``pipenv`` run pytest through the runner; ``pip`` runs a bare ``pytest``.
    An unresolved runner produces no candidate at all.
    """
    if runner in {"uv", "poetry", "pipenv"}:
        return RecipeCandidate(argv=(runner, "run", "pytest"), provenance="manifest")
    if runner == "pip":
        return RecipeCandidate(argv=("pytest",), provenance="manifest")
    return None


def _compose_identity(
    command: ResolvedFact, package: PackageResolution, repo_root: Path
) -> RecipeIdentity:
    """Recompute the versioned recipe identity from the live config inputs.

    The config-input digest is read from disk on every call so a changed
    lockfile/manifest moves the identity; an unreadable input is a named miss.
    """
    package_dir = _nearest_package_dir(repo_root, package.cwd_relative)
    config_digest, absent = _config_digest(
        package_dir, _config_input_names(package_dir, package.runner)
    )
    if config_digest is None:
        return RecipeIdentity(digest=None, absent_components=absent)
    components = {
        "format": RECIPE_FORMAT,
        "command": list(command.value) if command.value is not None else None,
        "command_source": command.source,
        "cwd_relative": package.cwd_relative,
        "runner": package.runner,
        "interpreter": package.interpreter,
        "config_digest": config_digest,
    }
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return RecipeIdentity(
        digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        absent_components=(),
    )


def recipe_identity(recipe: TestRecipe, repo_root: Path) -> RecipeIdentity:
    """Recompute a recipe's identity against the current config inputs.

    ``recipe.identity`` is the identity captured at resolution time; this
    function recomputes it so a decision gate can tell whether the recipe's
    inputs still hold, returning a named miss rather than raising.
    """
    return _compose_identity(recipe.command, recipe.package, repo_root)


def resolve_test_recipe(
    config: object,
    run_config: object,
    *,
    repo_root: Path,
    cwd: Path | None = None,
) -> TestRecipe:
    """Resolve the run's single test recipe, each fact exactly once.

    The command fact applies CLI-over-config precedence; the package is
    resolved from ``cwd`` (defaulting to the worktree root) and confined to
    ``repo_root``; the candidate comes only from the closed manifest set; and
    the identity binds the command/package facts to the per-package config
    inputs. An unresolved command never authorizes a required argv and never
    runs anything — the candidate is a suggestion only (spec MH4).
    """
    command = resolve_test_command_fact(config, run_config)
    package = resolve_package(repo_root, cwd if cwd is not None else repo_root)
    declared = tuple(
        getattr(run_config, "test_required_suites", None)
        or getattr(config, "test_required_suites", None)
        or ()
    )
    required = RequiredContract(
        declared=declared,
        argv=command.value if isinstance(command.value, tuple) else None,
        source="unresolved" if not command.resolved else _command_source(command.source),
    )
    return TestRecipe(
        command=command,
        package=package,
        required=required,
        candidate=_manifest_candidate(package.runner),
        identity=_compose_identity(command, package, repo_root),
    )


def _command_source(source: str) -> Literal["cli", "config"]:
    """Narrow a resolved command provenance to the required-contract domain."""
    return "config" if source == "config" else "cli"


def _tuple_of_str(value: object) -> tuple[str, ...] | None:
    """Return *value* as a tuple of strings, or ``None`` when it is not one."""
    if isinstance(value, list) and all(isinstance(entry, str) for entry in value):
        return tuple(value)
    return None


def recipe_to_payload(recipe: TestRecipe) -> dict[str, Any]:
    """Serialise a recipe to its persisted, JSON-safe payload.

    Every fact the recipe carries is emitted, so :func:`load_test_recipe`
    reconstructs the same typed value a resumed run resolves.
    """
    command_value = recipe.command.value
    candidate_payload: dict[str, Any] | None = None
    if recipe.candidate is not None:
        candidate_payload = {
            "argv": list(recipe.candidate.argv),
            "provenance": recipe.candidate.provenance,
        }
    return {
        "format_version": RECIPE_FORMAT,
        "command": {
            "value": list(command_value) if isinstance(command_value, tuple) else command_value,
            "source": recipe.command.source,
        },
        "package": {
            "cwd_relative": recipe.package.cwd_relative,
            "runner": recipe.package.runner,
            "interpreter": recipe.package.interpreter,
            "config_digest": recipe.package.config_digest,
            "absent_components": list(recipe.package.absent_components),
        },
        "required": {
            "declared": list(recipe.required.declared),
            "argv": list(recipe.required.argv) if recipe.required.argv is not None else None,
            "source": recipe.required.source,
        },
        "candidate": candidate_payload,
        "identity": {
            "digest": recipe.identity.digest,
            "absent_components": list(recipe.identity.absent_components),
        },
    }


def _recipe_from_payload(payload: dict[str, Any]) -> TestRecipe | None:
    """Rebuild a recipe from its payload, or ``None`` when it is malformed.

    The persisted artifact is untrusted input (Pattern B): every field is
    validated before it becomes a typed value, and any missing, mistyped, or
    out-of-domain field is a named absence, never a partially-built recipe.
    """
    try:
        command_payload = payload["command"]
        package_payload = payload["package"]
        required_payload = payload["required"]
        identity_payload = payload["identity"]
        if not all(
            isinstance(part, dict)
            for part in (command_payload, package_payload, required_payload, identity_payload)
        ):
            return None
        command_source = command_payload["source"]
        command_values = _tuple_of_str(command_payload["value"])
        if command_source == "unresolved":
            command = ResolvedFact(value=None, source="unresolved")
        elif command_source in ("cli", "config", "admitted", "derived") and command_values is not None:
            command = ResolvedFact(value=command_values, source=cast(FactSource, command_source))
        else:
            return None

        runner = package_payload["runner"]
        interpreter = package_payload["interpreter"]
        config_digest = package_payload["config_digest"]
        if (runner is not None and not isinstance(runner, str)) or (
            interpreter is not None and not isinstance(interpreter, str)
        ) or (config_digest is not None and not isinstance(config_digest, str)):
            return None
        cwd_relative = package_payload["cwd_relative"]
        if not isinstance(cwd_relative, str):
            return None
        absent_components = _tuple_of_str(package_payload["absent_components"])
        if absent_components is None:
            return None
        package = PackageResolution(
            cwd_relative=cwd_relative,
            runner=runner,
            interpreter=interpreter,
            config_digest=config_digest,
            absent_components=absent_components,
        )

        required_source = required_payload["source"]
        declared = _tuple_of_str(required_payload["declared"])
        required_argv = required_payload["argv"]
        if declared is None or required_source not in ("cli", "config", "unresolved"):
            return None
        if required_argv is None:
            argv: tuple[str, ...] | None = None
        else:
            argv = _tuple_of_str(required_argv)
            if argv is None:
                return None
        required = RequiredContract(
            declared=declared,
            argv=argv,
            source=cast(Literal["cli", "config", "unresolved"], required_source),
        )

        candidate: RecipeCandidate | None = None
        candidate_payload = payload.get("candidate")
        if candidate_payload is not None:
            if not isinstance(candidate_payload, dict):
                return None
            candidate_argv = _tuple_of_str(candidate_payload["argv"])
            if candidate_argv is None or candidate_payload["provenance"] != "manifest":
                return None
            candidate = RecipeCandidate(argv=candidate_argv, provenance="manifest")

        digest = identity_payload["digest"]
        identity_absent = _tuple_of_str(identity_payload["absent_components"])
        if (digest is not None and not isinstance(digest, str)) or identity_absent is None:
            return None
        identity = RecipeIdentity(
            digest=digest, absent_components=identity_absent, format_version=RECIPE_FORMAT
        )
    except (KeyError, TypeError, ValueError):
        return None
    return TestRecipe(
        command=command,
        package=package,
        required=required,
        candidate=candidate,
        identity=identity,
    )


def persist_test_recipe(deep_dir: Path, recipe: TestRecipe) -> Path:
    """Atomically persist *recipe* under *deep_dir*, returning its path."""
    path = deep_dir / TEST_RECIPE_FILENAME
    atomic_write_json(path, recipe_to_payload(recipe))
    return path


def load_test_recipe(deep_dir: Path) -> TestRecipe | None:
    """Read the persisted recipe, fail-open (Pattern B).

    A missing file, an unreadable file, malformed JSON, or a payload whose
    ``format_version`` is not :data:`RECIPE_FORMAT` all return ``None`` and
    never raise; a stale-format recipe must not be read as the current one.
    """
    try:
        payload = json.loads((deep_dir / TEST_RECIPE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format_version") != RECIPE_FORMAT:
        return None
    return _recipe_from_payload(payload)


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
    truncated = False

    async def _pump(stream: asyncio.StreamReader | None) -> None:
        nonlocal buffered, truncated
        if stream is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            data = await stream.read(64 * 1024)
            text = decoder.decode(data, final=not data)
            if not text:
                if not data:
                    return
                continue
            if buffered >= _MERGED_OUTPUT_LIMIT_CHARS:
                truncated = True
                if not data:
                    return
                continue
            remaining = _MERGED_OUTPUT_LIMIT_CHARS - buffered
            if len(text) > remaining:
                text = text[:remaining]
                truncated = True
            buffered += len(text)
            chunks.append(text)
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
        merged_output=_redact_merged("".join(chunks), env),
        completed=not timed_out,
        output_truncated=truncated,
        incomplete=timed_out or truncated,
    )
