"""Session-bound remote CI verification.

Strict parsing produces normalized evidence; pure evaluation drives deadline-
bounded polling. Artifacts contain only validated identities and observations.
"""

from daydream.remote_ci.artifacts import (
    local_host_facts as local_host_facts,
    remote_ci_handoff_payload as remote_ci_handoff_payload,
    remote_ci_verdict_payload as remote_ci_verdict_payload,
    write_remote_ci_handoff as write_remote_ci_handoff,
    write_remote_ci_verdict as write_remote_ci_verdict,
)
from daydream.remote_ci.evaluation import (
    evaluate_remote_ci as evaluate_remote_ci,
    pending_remote_ci_verdict as pending_remote_ci_verdict,
    required_context_label as required_context_label,
    required_context_matches as required_context_matches,
    unavailable_remote_ci_verdict as unavailable_remote_ci_verdict,
)
from daydream.remote_ci.evidence import (
    DEFAULT_LIMITS as DEFAULT_LIMITS,
    CIObservation as CIObservation,
    ObservationState as ObservationState,
    PRCIBinding as PRCIBinding,
    RemoteCIIdentityMismatch as RemoteCIIdentityMismatch,
    RemoteCILimits as RemoteCILimits,
    RemoteCISnapshot as RemoteCISnapshot,
    RemoteCIStatus as RemoteCIStatus,
    RemoteCITarget as RemoteCITarget,
    RemoteCIVerdict as RemoteCIVerdict,
    RequiredContext as RequiredContext,
    RequiredPolicy as RequiredPolicy,
)
from daydream.remote_ci.github import (
    GitHubRemoteCIFetcher as GitHubRemoteCIFetcher,
    RemoteCIFetcher as RemoteCIFetcher,
)
from daydream.remote_ci.parsing import (
    parse_active_workflows as parse_active_workflows,
    parse_observations as parse_observations,
    parse_pr_ci_binding as parse_pr_ci_binding,
    parse_required_policy as parse_required_policy,
)
from daydream.remote_ci.polling import (
    wait_for_remote_ci as wait_for_remote_ci,
)
