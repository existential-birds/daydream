"""Standard-library-only integrity seals for reward artifacts and candidate diffs, keeping
verifiers/prime-rl out of Daydream's lockfile. The supervisor seals after the agent write window;
scoring verifies staged inputs before trust. Changed/missing inputs fail verification and zero
reward. verify never raises and treats OSError as failure.
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
    """Seal artifacts by relative POSIX path under their common parent, using pinned SHA-256 digests to
    prevent algorithm downgrade. candidate_diff stores raw bytes as base64 for audit, including
    non-UTF-8 diffs. Verification re-derives the sandbox diff and checks candidate_diff_digest; it
    must never trust the embedded audit copy.
    """

    algorithm: Literal["sha256"] = _ALGORITHM
    artifact_digests: dict[str, str] = field(default_factory=dict)
    candidate_diff_digest: str = ""
    candidate_diff: bytes = b""

    def model_dump_json(self) -> str:
        """Serialize deterministically, with candidate_diff base64-encoded as specified by SealResult.

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
        """Parse a seal, raising ValueError for malformed JSON, wrong algorithm, or invalid digest
        types. Callers must treat invalid payloads as failed verification.
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
    """Read raw bytes under the relative keys defined by _relative_keys, propagating its ValueError.
    Unreadable artifacts raise OSError: seal_artifacts propagates this supervisor error, while
    verify treats it as a failed seal.
    """
    return {rel: path.read_bytes() for rel, path in _relative_keys(paths).items()}


def seal_artifacts(paths: list[Path], candidate_diff: bytes) -> SealResult:
    """Seal raw artifact bytes and candidate_diff under the SealResult contract. Unreadable artifacts
    raise OSError: the supervisor expects a known-good staged copy, so read failures are programming
    errors.
    """
    return seal_bytes(read_paths(paths), candidate_diff)


def seal_bytes(artifacts: dict[str, bytes], candidate_diff: bytes) -> SealResult:
    """Seal {relative POSIX path: raw bytes} plus candidate_diff directly from the runtime, without
    staging a host copy for seal_artifacts.
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
        return digests == seal.artifact_digests
    except OSError:
        return False
