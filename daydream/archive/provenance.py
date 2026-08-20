"""Executable provenance capture for archived runs.

Records the immutable Daydream executable that produced a run: version,
install source, resolved commit, dirty state, and opt-in container digest.
This lives in its own ``daydream.*`` namespace and is never conflated with
the target-repository ``git.*`` / ``code_context.*`` blocks.

Each fallible field is a best-effort capture with an explicit ``"unknown"``
sentinel — a failure to resolve a field reports ``"unknown"`` rather than a
raise or a fabricated value (mirroring ``git_context.capture_git_context``'s
independent-fields pattern, degrading to ``"unknown"`` instead of ``None``).

Exports:
    ExecutableProvenance: Dataclass holding captured executable provenance.
    capture_executable_provenance: Capture provenance from the installed
        package dir + opt-in env.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import daydream
from daydream import git_ops
from daydream.git_ops import GitError


@dataclass
class ExecutableProvenance:
    """The immutable Daydream executable that produced an archived run.

    Attributes:
        version: Package version (``daydream.__version__``).
        install_source: ``editable`` (uv/editable install), ``git``
            (VCS-derived direct_url), ``package`` (no direct_url), or
            ``unknown`` when the distribution cannot be resolved.
        commit: Full SHA of the package dir's git HEAD, or ``"unknown"``.
        dirty: ``True``/``False`` when the package-dir git state resolves,
            else ``"unknown"``.
        container_digest: The opt-in ``DAYDREAM_IMAGE_DIGEST`` value, or
            ``"unknown"``. Never auto-detected.
    """

    version: str
    install_source: str
    commit: str = "unknown"
    dirty: bool | str = "unknown"
    container_digest: str = "unknown"

    def to_dict(self) -> dict[str, str | bool]:
        """Return exactly the five provenance fields (unknowns as strings)."""
        return {
            "version": self.version,
            "install_source": self.install_source,
            "commit": self.commit,
            "dirty": self.dirty,
            "container_digest": self.container_digest,
        }


def _resolve_install_source() -> str:
    """Resolve the package install source from ``importlib.metadata``.

    Returns ``"unknown"`` on any lookup failure — never raises.
    """
    try:
        from importlib.metadata import distribution

        dist = distribution("daydream")
        if dist is None:
            return "unknown"
        dist = distribution("daydream")
        if dist is None:
            return "unknown"
        try:
            direct_url_json = dist.read_text("direct_url.json")
            if direct_url_json is None:
                # No direct_url.json -> conventional package install
                # (site-packages, no install-time VCS/editable marker).
                return "package"
            info = json.loads(direct_url_json)
        except Exception:  # noqa: BLE001 - unreadable/malformed direct_url, never raise
            return "unknown"
        dir_info = info.get("dir_info")
        if isinstance(dir_info, dict) and dir_info.get("editable"):
            return "editable"
        if info.get("vcs_info"):
            return "git"
        # direct_url present but no dir_info/vcs_info -> install-time info
        # unavailable, reported honestly as unknown, never a fabricated value.
        return "unknown"
    except Exception:  # noqa: BLE001 - unknown install source, never raise
        return "unknown"


def capture_executable_provenance() -> ExecutableProvenance:
    """Capture the Daydream executable provenance.

    Best-effort per field; never raises. The package dir is the directory
    holding ``daydream/__file__``. Commit/dirty resolve from that dir's git
    state (walking up to the enclosing checkout) or degrade to ``"unknown"``.
    Container digest is opt-in via ``DAYDREAM_IMAGE_DIGEST``.
    """
    pkg_dir = Path(daydream.__file__).parent

    try:
        commit = git_ops.head_sha(pkg_dir)
    except GitError:
        commit = "unknown"

    dirty: bool | str
    try:
        dirty = bool(git_ops.status_porcelain(pkg_dir))
    except GitError:
        dirty = "unknown"

    return ExecutableProvenance(
        version=daydream.__version__,
        install_source=_resolve_install_source(),
        commit=commit,
        dirty=dirty,
        container_digest=os.environ.get("DAYDREAM_IMAGE_DIGEST") or "unknown",
    )
