"""Type-checking shim for the sandbox template's bare ``import verifier_core``.

The templates/tests/score_review.py sandbox is copied into an isolated Harbor
verifier image at build time, where ``verifier_core`` (the canonical module
from :mod:`daydream.benchmark.harbor.verifier_core`, deployed verbatim) sits
beside it. At type-check time that top-level bare name is not on ``sys.path``;
this shim (pinned to ``[tool.mypy] mypy_path``) re-exports the canonical
module's public API so the template is checked against the real signatures
instead of degrading to ``Any``. Runtime never imports this module -- it is
not packaged (see ``[tool.hatch.build.targets.wheel]``) and no code path
references it.

Listing the names in ``__all__`` is what makes each one a re-export under
mypy's implicit-reexport rule (``strict`` disables bare re-export).
"""

from daydream.benchmark.harbor.verifier_core import (
    MAX_ARTIFACT_BYTES,
    GoldFinding,
    Reward,
    Verdict,
    VerifierError,
    maximum_matching,
    retained_edges,
    reward_details,
    reward_to_json,
    score_review,
    validate_candidate_artifact,
    validate_exact_keys,
    validate_gold_set,
)

__all__ = [
    "MAX_ARTIFACT_BYTES",
    "GoldFinding",
    "Reward",
    "Verdict",
    "VerifierError",
    "maximum_matching",
    "retained_edges",
    "reward_details",
    "reward_to_json",
    "score_review",
    "validate_candidate_artifact",
    "validate_exact_keys",
    "validate_gold_set",
]
