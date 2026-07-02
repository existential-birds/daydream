"""Extension discovery, version gate, and per-run registry construction.

Discovery precedence: ``$DAYDREAM_EXT_DIR`` (explicit path override, matching
the ``$DAYDREAM_SKILLS_DIR`` convention; also the test seam) → ``import
daydream_ext`` (the fork extension package pre-declared in the wheel) → no
extension (builtins-only registry). Absence is fine; a present-but-broken
extension is a loud, named error.

Registry propagation uses a ``ContextVar`` (mirroring the trajectory
recorder's ``_RECORDER_VAR`` in ``daydream/trajectory.py``): access via
:func:`get_registry` / :func:`set_registry` only.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType

from daydream.extensions.api import (
    EXTENSION_API_VERSION,
    ExtensionError,
    ExtensionVersionError,
)
from daydream.extensions.builtins import register_builtins
from daydream.extensions.registry import Registry

_EXT_DIR_ENV = "DAYDREAM_EXT_DIR"
_EXT_MODULE_NAME = "daydream_ext"
_CONTRACT_DOC = "docs/extensions.md"


def _load_from_dir(ext_dir: str) -> tuple[ModuleType, str]:
    """Load ``<ext_dir>/__init__.py`` fresh, never touching ``sys.modules``."""
    init_path = Path(ext_dir) / "__init__.py"
    # A non-sys.modules name keeps every load fresh: repeat runs and tests
    # never see a stale module.
    spec = importlib.util.spec_from_file_location("_daydream_ext_from_dir", init_path)
    if spec is None or spec.loader is None:
        raise ExtensionError(f"cannot load extension module from {init_path} (${_EXT_DIR_ENV}={ext_dir})")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ExtensionError(f"failed to import extension module {init_path}: {exc}") from exc
    return module, str(init_path)


def _load_extension_module() -> tuple[ModuleType, str] | None:
    """Discover the extension module, or return None when none is present."""
    ext_dir = os.environ.get(_EXT_DIR_ENV)
    if ext_dir:
        return _load_from_dir(ext_dir)
    try:
        module = importlib.import_module(_EXT_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name == _EXT_MODULE_NAME:
            return None  # silent absence: upstream ships no daydream_ext
        raise ExtensionError(f"failed to import {_EXT_MODULE_NAME}: {exc}") from exc
    except Exception as exc:
        raise ExtensionError(f"failed to import {_EXT_MODULE_NAME}: {exc}") from exc
    return module, getattr(module, "__file__", None) or _EXT_MODULE_NAME


def _require_version(module: ModuleType, source: str) -> None:
    version = getattr(module, "DAYDREAM_EXT_API", None)
    if version is None:
        raise ExtensionVersionError(
            f"extension module at {source} does not export DAYDREAM_EXT_API; "
            f"this daydream expects {EXTENSION_API_VERSION} (see {_CONTRACT_DOC})"
        )
    if version != EXTENSION_API_VERSION:
        raise ExtensionVersionError(
            f"extension module at {source} declares DAYDREAM_EXT_API = {version!r}; "
            f"this daydream expects {EXTENSION_API_VERSION} (see {_CONTRACT_DOC})"
        )


def _require_register(module: ModuleType, source: str) -> Callable[[Registry], None]:
    register = getattr(module, "register", None)
    if not callable(register):
        raise ExtensionError(
            f"extension module at {source} does not export a callable 'register(registry)' (see {_CONTRACT_DOC})"
        )
    return register


def build_registry() -> Registry:
    """Build a per-run registry: builtins seeded, then the extension applied."""
    registry = Registry()
    register_builtins(registry)
    loaded = _load_extension_module()
    if loaded is None:
        return registry
    module, source = loaded
    _require_version(module, source)
    register = _require_register(module, source)
    try:
        register(registry)
    except ExtensionError:
        raise
    except Exception as exc:
        raise ExtensionError(f"extension module at {source} raised during register(): {exc}") from exc
    return registry


_REGISTRY_VAR: ContextVar[Registry | None] = ContextVar("_REGISTRY_VAR", default=None)


def set_registry(registry: Registry) -> None:
    """Set the current async context's registry (called once per run)."""
    _REGISTRY_VAR.set(registry)


def get_registry() -> Registry:
    """Return the current registry, lazily building a builtins-only one when unset.

    The lazy fallback lets direct phase calls in unit tests resolve built-ins
    without any runner setup; it deliberately skips extension discovery —
    extensions apply only through :func:`build_registry` at run entry. It also
    deliberately does NOT cache into the ContextVar: a sync caller before
    :func:`set_registry` would otherwise pin a stale builtins snapshot on the
    process-level context, leaking across pytest tests.
    """
    registry = _REGISTRY_VAR.get()
    if registry is None:
        registry = Registry()
        register_builtins(registry)
    return registry
