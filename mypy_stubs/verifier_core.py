"""Type-checking shim for the sandbox template's bare ``import verifier_core``.

The templates/tests/score_review.py sandbox is copied into an isolated Harbor
verifier image at build time, where ``verifier_core`` (the byte-parity twin of
in-repo :mod:`daydream.benchmark.harbor.verifier_core`) sits beside it. At
type-check time that top-level bare name is not on ``sys.path``; this shim
(pinned to ``[tool.mypy] mypy_path``) re-exports the in-repo twin's public API
so the template is checked against the real signatures instead of degrading to
``Any``. Runtime never imports this module -- it is not packaged (see
``[tool.hatch.build.targets.wheel]``) and no code path references it.
"""

from daydream.benchmark.harbor.verifier_core import (
    GoldFinding as GoldFinding,
    MAX_ARTIFACT_BYTES as MAX_ARTIFACT_BYTES,
    Reward as Reward,
    Verdict as Verdict,
    VerifierError as VerifierError,
    maximum_matching as maximum_matching,
    retained_edges as retained_edges,
    reward_details as reward_details,
    reward_to_json as reward_to_json,
    score_review as score_review,
    validate_candidate_artifact as validate_candidate_artifact,
    validate_exact_keys as validate_exact_keys,
    validate_gold_set as validate_gold_set,
)