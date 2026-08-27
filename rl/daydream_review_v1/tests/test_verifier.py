"""Seal/verify primitive for reward artifact isolation — stdlib only.

The reward must consume only state the rollout agent cannot rewrite. The
supervisor seals the staged run-dir artifacts (with the candidate diff) and the
reward verifies the seal before trusting any value; an attempted tamper must
make verification fail. These tests pin the round-trip and every tamper shape.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_seal_verify_roundtrip(tmp_path: Path) -> None:
    from daydream_review_v1.verifier import seal_artifacts, verify

    a = tmp_path / "a.json"
    a.write_text('{"x": 1}', encoding="utf-8")
    d = tmp_path / "deep"
    d.mkdir()
    b = d / "b.json"
    b.write_text('{"y": 2}', encoding="utf-8")
    diff = b"diff --git a/a b/a"

    seal = seal_artifacts([a, b], candidate_diff=diff)
    assert verify(seal, [a, b], candidate_diff=diff) is True


def test_verify_detects_tampered_artifact(tmp_path: Path) -> None:
    from daydream_review_v1.verifier import seal_artifacts, verify

    a = tmp_path / "a.json"
    a.write_text('{"x": 1}', encoding="utf-8")
    seal = seal_artifacts([a], candidate_diff=b"")
    a.write_text('{"x": 999}', encoding="utf-8")  # agent rewrote it

    assert verify(seal, [a], candidate_diff=b"") is False


def test_verify_detects_altered_candidate_diff(tmp_path: Path) -> None:
    from daydream_review_v1.verifier import seal_artifacts, verify

    a = tmp_path / "a.json"
    a.write_text('{"x": 1}', encoding="utf-8")
    seal = seal_artifacts([a], candidate_diff=b"patch-v1")

    assert verify(seal, [a], candidate_diff=b"patch-v2") is False


def test_verify_detects_missing_artifact(tmp_path: Path) -> None:
    from daydream_review_v1.verifier import seal_artifacts, verify

    a = tmp_path / "a.json"
    a.write_text('{"x": 1}', encoding="utf-8")
    seal = seal_artifacts([a], candidate_diff=b"")
    a.unlink()

    assert verify(seal, [a], candidate_diff=b"") is False


def test_seal_json_roundtrip(tmp_path: Path) -> None:
    """The seal serializes to JSON and parses back to the same verification result."""
    from daydream_review_v1.verifier import SealResult, seal_artifacts, verify

    a = tmp_path / "a.json"
    a.write_text('{"x": 1}', encoding="utf-8")
    seal = seal_artifacts([a], candidate_diff=b"candidate-diff")

    raw = seal.model_dump_json()
    parsed = SealResult.model_validate_json(raw)
    assert parsed == seal
    assert json.loads(raw)["algorithm"] == "sha256"
    assert verify(parsed, [a], candidate_diff=b"candidate-diff") is True


def test_validate_rejects_unsupported_algorithm() -> None:
    """A seal.json with a downgraded algorithm (e.g. md5) must fail closed."""
    import pytest

    from daydream_review_v1.verifier import SealResult

    raw = json.dumps(
        {
            "algorithm": "md5",
            "artifact_digests": {"a.json": "0" * 32},
            "candidate_diff_digest": "0" * 64,
            "candidate_diff": "",
        }
    )
    with pytest.raises(ValueError, match="unsupported seal algorithm"):
        SealResult.model_validate_json(raw)


def test_validate_rejects_malformed_json() -> None:
    """Garbage seal.json content must fail closed as a verification failure."""
    import pytest

    from daydream_review_v1.verifier import SealResult

    with pytest.raises(ValueError, match="not valid JSON"):
        SealResult.model_validate_json("this is not json{")
    with pytest.raises(ValueError, match="must be a JSON object"):
        SealResult.model_validate_json('["not", "an", "object"]')
