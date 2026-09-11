"""Run-local interaction policy and active-backend lifecycle state."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

if TYPE_CHECKING:
    from rich.console import Console

    from daydream.backends import Backend
    from daydream.github_app import GitHubIdentity


@dataclass(frozen=True, kw_only=True)
class InteractionPolicy:
    """Immutable interaction and presentation choices for one run."""

    assume: str | None = None
    interactive: bool = True
    quiet: bool = False
    log_mode: bool = False


@dataclass
class _BackendRegistration:
    run_token: object
    backend: Backend


# Signal handling needs a process-wide view across task-local run contexts. The
# registry deliberately stores only an opaque run token and backend lifecycle
# entries; interaction and presentation policy never enter this structure.
_registry_lock = RLock()
_active_registrations: dict[object, _BackendRegistration] = {}
_current_context: ContextVar[RunContext | None] = ContextVar(
    "daydream_run_context", default=None
)
_P = ParamSpec("_P")
_T = TypeVar("_T")


class RunContext:
    """Own the immutable policy and active backend counts for one run."""

    __slots__ = ("_policy", "_run_token", "github_identity")

    def __init__(
        self, policy: InteractionPolicy, *, github_identity: GitHubIdentity | None = None,
    ) -> None:
        from daydream.github_app import GitHubIdentity

        self._policy = policy
        self._run_token = object()
        self.github_identity = github_identity if github_identity is not None else GitHubIdentity("unknown")

    @property
    def policy(self) -> InteractionPolicy:
        """Return this run's immutable interaction policy."""
        return self._policy

    def confirm(
        self,
        question: str,
        *,
        safe_default: bool,
        default: str = "n",
        console: Console | None = None,
    ) -> bool:
        """Resolve a yes/no gate and prompt only when policy permits it."""
        decision = resolve_gate(
            assume=self.policy.assume,
            interactive=self.policy.interactive,
            safe_default=safe_default,
        )
        if decision is not None:
            return decision
        with bind_run_context(self):
            response = _prompt_user(console, question, default)
        return response.strip().lower() in ("y", "yes")

    def choice(
        self,
        question: str,
        *,
        default: str,
        safe_default: str,
        assume_yes: str | None = None,
        assume_no: str | None = None,
        console: Console | None = None,
    ) -> str:
        """Resolve a menu or free-form choice through this run's policy."""
        if self.policy.assume == "yes" and assume_yes is not None:
            return assume_yes
        if self.policy.assume == "no" and assume_no is not None:
            return assume_no
        if not self.policy.interactive:
            return safe_default
        with bind_run_context(self):
            return _prompt_user(console, question, default)

    @contextmanager
    def backend_registration(self, backend: Backend) -> Iterator[None]:
        """Register one backend invocation until its entire lifecycle exits."""
        registration_token = object()
        try:
            with _registry_lock:
                _active_registrations[registration_token] = _BackendRegistration(
                    self._run_token, backend
                )
            yield
        finally:
            with _registry_lock:
                _active_registrations.pop(registration_token, None)

    def active_backends(self) -> tuple[Backend, ...]:
        """Return the unique backend identities active in this run."""
        with _registry_lock:
            return _unique_backends(
                entry.backend
                for entry in _active_registrations.values()
                if entry.run_token is self._run_token
            )


def current_run_context() -> RunContext | None:
    """Return the context bound to this task, if any."""
    return _current_context.get()


def resolve_run_context(context: RunContext | None = None) -> RunContext:
    """Resolve an explicit, bound, or fresh standalone run context."""
    if context is not None:
        return context
    bound = current_run_context()
    if bound is not None:
        return bound
    return RunContext(InteractionPolicy())


@contextmanager
def bind_run_context(context: RunContext) -> Iterator[RunContext]:
    """Bind a context for the current task and restore the prior binding."""
    token = _current_context.set(context)
    try:
        yield context
    finally:
        _current_context.reset(token)


def bind_resolved_run_context(
    function: Callable[_P, Awaitable[_T]],
) -> Callable[_P, Coroutine[Any, Any, _T]]:
    """Bind a standalone async entry's explicit or ambient runtime for its body."""

    @wraps(function)
    async def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        run_context = resolve_run_context(cast("RunContext | None", kwargs.get("run_context")))
        with bind_run_context(run_context):
            return await function(*args, **kwargs)

    return wrapped


def active_backends() -> tuple[Backend, ...]:
    """Return a process-wide snapshot of unique active backend identities."""
    with _registry_lock:
        return _unique_backends(
            entry.backend for entry in _active_registrations.values()
        )


def _unique_backends(backends: Iterator[Backend]) -> tuple[Backend, ...]:
    """Deduplicate a registration snapshot by object identity."""
    seen: set[int] = set()
    snapshot: list[Backend] = []
    for backend in backends:
        identity = id(backend)
        if identity not in seen:
            seen.add(identity)
            snapshot.append(backend)
    return tuple(snapshot)


def resolve_gate(*, assume: str | None, interactive: bool, safe_default: bool) -> bool | None:
    """Resolve a yes/no interaction gate without performing I/O."""
    if assume is not None:
        return assume == "yes"
    if not interactive:
        return safe_default
    return None


def _prompt_user(console: Console | None, message: str, default: str) -> str:
    """Call the sole raw prompt reader, resolving the shared console lazily."""
    if console is None:
        from daydream.agent import console as agent_console

        console = agent_console
    from daydream.ui.messages import _read_user_input

    return _read_user_input(console, message, default)
