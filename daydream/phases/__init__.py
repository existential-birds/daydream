"""Public review and fix phases; implementation modules follow the pipeline lifecycle."""

from daydream.phases.findings import (
    severity_sorted as severity_sorted,
)
from daydream.phases.fix import (
    UnconfinedFindingError as UnconfinedFindingError,
    group_items_by_footprint as group_items_by_footprint,
    phase_fix as phase_fix,
    phase_fix_batched as phase_fix_batched,
)
from daydream.phases.fix_fanout import (
    phase_fix_parallel as phase_fix_parallel,
)
from daydream.phases.handoff import (
    FAILURE_SUMMARIZER_SCHEMA as FAILURE_SUMMARIZER_SCHEMA,
)
from daydream.phases.inputs import (
    TEST_OUTPUT_TAIL_LINES as TEST_OUTPUT_TAIL_LINES,
    append_extended_facts as append_extended_facts,
)
from daydream.phases.publish import (
    CommitPushResult as CommitPushResult,
    PushAttemptError as PushAttemptError,
    PushReceipt as PushReceipt,
    build_commit_message as build_commit_message,
    phase_commit_push as phase_commit_push,
    require_empty_staged_index as require_empty_staged_index,
)
from daydream.phases.review_prompts import (
    _exploration_pointer as _exploration_pointer,
)
from daydream.phases.schemas import (
    FIX_VERIFY_ACTIONABLE_VERDICTS as FIX_VERIFY_ACTIONABLE_VERDICTS,
    FIX_VERIFY_RETARGETABLE_VERDICTS as FIX_VERIFY_RETARGETABLE_VERDICTS,
    FIX_VERIFY_VERDICTS as FIX_VERIFY_VERDICTS,
    FIX_VERIFY_VERDICTS_SCHEMA as FIX_VERIFY_VERDICTS_SCHEMA,
    MERGED_ITEMS_SCHEMA as MERGED_ITEMS_SCHEMA,
    RECOMMENDATION_VERDICTS_SCHEMA as RECOMMENDATION_VERDICTS_SCHEMA,
)
from daydream.phases.test_evidence import (
    RepairAttemptEvidence as RepairAttemptEvidence,
    TestAndHealResult as TestAndHealResult,
    TestAttemptEvidence as TestAttemptEvidence,
    phase_test_once as phase_test_once,
    reuse_target as reuse_target,
)
from daydream.phases.testing import (
    SETUP_INVESTIGATOR_SCHEMA as SETUP_INVESTIGATOR_SCHEMA,
    phase_test_and_heal as phase_test_and_heal,
)
from daydream.phases.verify import (
    phase_fix_verify as phase_fix_verify,
    phase_verify_recommendations as phase_verify_recommendations,
)
