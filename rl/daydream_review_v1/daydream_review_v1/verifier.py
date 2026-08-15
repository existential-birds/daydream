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
    pinned as ``"sha256"`` so a verify can never silently downgrade. Serialized
    to JSON by the supervisor (:meth:`model_dump_json`) and parsed back at
    verification time (:meth:`model_validate_json`).
    """

    algorithm: Literal["sha256"] = _ALGORITHM
    artifact_digests: dict[str, str] = field(default_factory=dict)
    candidate_diff_digest: str = ""

    def model_dump_json(self) -> str:
        """Serialize the seal to a JSON string (deterministic key order)."""
        return json.dumps(
            {
                "algorithm": self.algorithm,
                "artifact_digests": self.artifact_digests,
                "candidate_diff_digest": self.candidate_diff_digest,
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
        return cls(algorithm=algorithm, artifact_digests=digests, candidate_diff_digest=diff)


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


def seal_artifacts(paths: list[Path], candidate_diff: bytes) -> SealResult:
    """Seal *paths* (sha256 of each artifact's raw bytes) plus *candidate_diff*.

    Raises:
        OSError: If an artifact cannot be read. Sealing is the supervisor's job
            over a known-good staged copy, so a missing artifact here is a
            programming error, not a tamper to absorb.
    """
    digests = {
        rel: hashlib.sha256(path.read_bytes()).hexdigest() for rel, path in _relative_keys(paths).items()
    }
    return SealResult(
        algorithm=_ALGORITHM,
        artifact_digests=digests,
        candidate_diff_digest=hashlib.sha256(candidate_diff).hexdigest(),
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
            rel: hashlib.sha256(path.read_bytes()).hexdigest() for rel, path in _relative_keys(paths).items()
        }
        if hashlib.sha256(candidate_diff).hexdigest() != seal.candidate_diff_digest:
            return False
        if set(digests) != set(seal.artifact_digests):
            return False
        return all(digests[rel] == digest for rel, digest in seal.artifact_digests.items())
    except OSError:
        return False
