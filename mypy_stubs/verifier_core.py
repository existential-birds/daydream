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
    MAX_ARTIFACT_BYTES as MAX_ARTIFACT_BYTES,
)
from daydream.benchmark.harbor.verifier_core import (
    GoldFinding as GoldFinding,
)
from daydream.benchmark.harbor.verifier_core import (
    Reward as Reward,
)
from daydream.benchmark.harbor.verifier_core import (
    Verdict as Verdict,
)
from daydream.benchmark.harbor.verifier_core import (
    VerifierError as VerifierError,
)
from daydream.benchmark.harbor.verifier_core import (
    maximum_matching as maximum_matching,
)
from daydream.benchmark.harbor.verifier_core import (
    retained_edges as retained_edges,
)
from daydream.benchmark.harbor.verifier_core import (
    reward_details as reward_details,
)
from daydream.benchmark.harbor.verifier_core import (
    reward_to_json as reward_to_json,
)
from daydream.benchmark.harbor.verifier_core import (
    score_review as score_review,
)
from daydream.benchmark.harbor.verifier_core import (
    validate_candidate_artifact as validate_candidate_artifact,
)
from daydream.benchmark.harbor.verifier_core import (
    validate_exact_keys as validate_exact_keys,
)
from daydream.benchmark.harbor.verifier_core import (
    validate_gold_set as validate_gold_set,
)
