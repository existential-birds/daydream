"""Tests for the stdlib-only harbor verifier core module."""


def test_harbor_package_imports_stdlib_only():
    import daydream.benchmark.harbor.verifier_core as vc

    assert vc.MAX_GOLD_FINDINGS == 50
    assert vc.MAX_CANDIDATE_FINDINGS == 100
    assert vc.MAX_ARTIFACT_BYTES == 1_048_576
    assert vc.CONFIDENCE_THRESHOLD == 0.7
    assert issubclass(vc.VerifierError, Exception)