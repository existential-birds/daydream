"""Stdlib-only seal/verify primitive for reward artifact isolation.

The reward must consume only state the rollout agent cannot rewrite. The trusted
supervisor seals the staged run-dir artifacts together with the candidate diff
(:func:`seal_artifacts`), and the reward verifies the seal against the staged
copy before trusting any value (:func:`verify`). An attempted tamper — a
rewritten artifact, an altered candidate diff, a missing artifact — makes
verification fail, which must zero the reward rather than crash scoring:
:func:`verify` never raises and returns ``False`` on any ``OSError``.

Constraint: standard library only (``hashlib``, ``dataclasses``, ``pathlib``,
``os``, ``json``). The standalone RL project deliberately keeps verifiers/prime-rl
out of daydream's lockfile, and this primitive must not disturb that boundary.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_ALGORITHM: Literal["sha256"] = "sha256"


@dataclass(frozen=True)
class SealResult:
    """An integrity seal over a set of artifacts and the candidate diff.

    ``artifact_digests`` maps each artifact's path relative to the common parent
    of the sealed paths (posix form) to its sha256 hex digest; the algorithm is
    pinned as ``"sha256"`` so a verify can never silently downgrade.
    ``candidate_diff`` carries the raw diff bytes as an audit record of what
    was sealed. Verification does not trust this embedded copy: the verifying
    side re-derives the diff from the sandbox at scoring time and compares its
    digest against ``candidate_diff_digest``, so the seal is bound to the diff
    the verifier checkout actually applies. Diff bytes are arbitrary (git diff
    output is not guaranteed UTF-8), so the record is serialized to JSON as
    base64, never as a UTF-8 decode. Serialized by the supervisor
    (:meth:`model_dump_json`) and parsed back at verification time
    (:meth:`model_validate_json`).
    """

    algorithm: Literal["sha256"] = _ALGORITHM
    artifact_digests: dict[str, str] = field(default_factory=dict)
    candidate_diff_digest: str = ""
    candidate_diff: bytes = b""

    def model_dump_json(self) -> str:
        """Serialize the seal to a JSON string (deterministic key order).

        ``candidate_diff`` is base64-encoded: the diff bytes are arbitrary
        (git diff output is not guaranteed UTF-8), and a UTF-8 decode here
        would make seal production raise on a non-UTF-8 diff — leaving the run
        unsealed instead of sealed.
        """
        return json.dumps(
            {
                "algorithm": self.algorithm,
                "artifact_digests": self.artifact_digests,
                "candidate_diff_digest": self.candidate_diff_digest,
                "candidate_diff": base64.b64encode(self.candidate_diff).decode("ascii"),
            },
            sort_keys=True,
        )

    @classmethod
    def model_validate_json(cls, raw: str) -> "SealResult":
        """Parse and validate a ``seal.json`` produced by :meth:`model_dump_json`.

        Raises:
            ValueError: If the payload is not the sealed shape (malformed JSON,
                wrong algorithm, or non-string digest values). A tamper must
                surface as a verification failure, so callers treat this as a
                failed seal — never as a pass.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"seal.json is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("seal.json must be a JSON object")
        algorithm = data.get("algorithm")
        if algorithm != _ALGORITHM:
            raise ValueError(f"unsupported seal algorithm {algorithm!r}; expected {_ALGORITHM!r}")
        digests = data.get("artifact_digests")
        if not isinstance(digests, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in digests.items()
        ):
            raise ValueError("seal.json artifact_digests must map path -> hex digest")
        diff = data.get("candidate_diff_digest")
        if not isinstance(diff, str):
            raise ValueError("seal.json candidate_diff_digest must be a string")
        raw_diff = data.get("candidate_diff")
        if not isinstance(raw_diff, str):
            raise ValueError("seal.json candidate_diff must be a string")
        try:
            diff_bytes = base64.b64decode(raw_diff.encode("ascii"))
        except (UnicodeError, ValueError) as exc:
            raise ValueError("seal.json candidate_diff is not valid base64") from exc
        return cls(
            algorithm=algorithm,
            artifact_digests=digests,
            candidate_diff_digest=diff,
            candidate_diff=diff_bytes,
        )


def _relative_keys(paths: list[Path]) -> dict[str, Path]:
    """Map each path to its posix path relative to the paths' common parent.

    An empty list yields no keys. A single path is keyed by its own name. Paths
    with no common parent (``os.path.commonpath`` raises) propagate as a
    ``ValueError`` — a caller error, not a tamper signal.
    """
    if not paths:
        return {}
    parents = [str(path.parent) for path in paths]
    common = Path(os.path.commonpath(parents))
    return {path.relative_to(common).as_posix(): path for path in paths}


def read_paths(paths: list[Path]) -> dict[str, bytes]:
    """Read each path's raw bytes, keyed by its posix path relative to the common parent.

    Raises:
        OSError: If an artifact cannot be read. Callers choose the policy:
            :func:`seal_artifacts` lets it propagate (a supervisor-side
            programming error), :func:`verify` absorbs it as a failed seal.
        ValueError: If the paths share no common parent (see :func:`_relative_keys`).
    """
    return {rel: path.read_bytes() for rel, path in _relative_keys(paths).items()}


def seal_artifacts(paths: list[Path], candidate_diff: bytes) -> SealResult:
    """Seal *paths* (sha256 of each artifact's raw bytes) plus *candidate_diff*.

    The raw diff is carried in the seal record as an audit copy; verification
    re-derives the diff from the sandbox at scoring time instead of trusting
    this embedded copy (:func:`rundir.verify_seal`).

    Raises:
        OSError: If an artifact cannot be read. Sealing is the supervisor's job
            over a known-good staged copy, so a missing artifact here is a
            programming error, not a tamper to absorb.
    """
    return seal_bytes(read_paths(paths), candidate_diff)


def seal_bytes(artifacts: dict[str, bytes], candidate_diff: bytes) -> SealResult:
    """Seal artifacts given as ``{relative posix path: raw bytes}`` plus *candidate_diff*.

    The sandbox-side twin of :func:`seal_artifacts`: the supervisor reads the
    archived run-dir members as bytes out of the runtime and seals them without
    staging a host copy first.
    """
    return SealResult(
        algorithm=_ALGORITHM,
        artifact_digests={rel: hashlib.sha256(data).hexdigest() for rel, data in artifacts.items()},
        candidate_diff_digest=hashlib.sha256(candidate_diff).hexdigest(),
        candidate_diff=candidate_diff,
    )


def verify(seal: SealResult, paths: list[Path], candidate_diff: bytes) -> bool:
    """Return True iff every artifact matches its recorded digest and no seal member is missing.

    ``True`` requires all three: every artifact's current bytes hash to its
    recorded digest, the candidate diff hashes to ``seal.candidate_diff_digest``,
    and the set of presented paths exactly matches the sealed set (a deleted or
    added artifact is a tamper). Never raises: any ``OSError`` (a missing or
    unreadable artifact) returns ``False``.
    """
    try:
        digests = {
            rel: hashlib.sha256(data).hexdigest() for rel, data in read_paths(paths).items()
        }
        if hashlib.sha256(candidate_diff).hexdigest() != seal.candidate_diff_digest:
            return False
        if set(digests) != set(seal.artifact_digests):
            return False
        return all(digests[rel] == digest for rel, digest in seal.artifact_digests.items())
    except OSError:
        return False
