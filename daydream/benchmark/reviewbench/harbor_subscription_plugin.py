"""Queue one account's complete public trials before any timed Harbor phase.

The container refresh lease protects rotating credentials. It cannot be the
scheduler: waiting for a different trial's judge inside agent.run would consume
that reviewer's frozen wall budget. The pinned Harbor queue already owns the
whole-trial permit and releases it across failures and cancellation, so cap that
permit before submission instead of adding another trial lifecycle.
"""
from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any

ADMISSION_POLICY = {
    'scope': 'whole-trial-before-create-and-phase-timers',
    'effective_concurrency': 1,
    'reason': 'one-subscription-account-with-exclusive-refresh-ownership',
}


class SubscriptionAdmissionPlugin:
    """Preserve each phase budget by admitting one complete trial per account."""

    async def on_job_start(self, job: Any) -> None:
        TrialQueue = importlib.import_module('harbor.trial.queue').TrialQueue

        from daydream.benchmark.reviewbench.harbor_cli import verify_runtime

        verify_runtime()
        queue = job._trial_queue
        if not isinstance(queue, TrialQueue):
            raise ValueError('Unsupported pinned Harbor trial queue')
        policy = {'schema': 1, **ADMISSION_POLICY,
                  'requested_concurrency': job.config.n_concurrent_trials}
        path = job.job_dir / 'daydream-subscription-admission.json'
        if path.exists():
            if json.loads(path.read_text()) != policy:
                raise ValueError('Existing subscription admission policy differs; use a new job')
        else:
            with path.open('x') as stream:
                json.dump(policy, stream, sort_keys=True, indent=2)
                stream.write('\n')
        # Narrow pinned SDK compatibility boundary, like upstream's run planner.
        # TrialQueue._run_trial acquires this before Trial.create and retains it
        # through verifier finalization; its async-with owns all cleanup paths.
        queue._semaphore = asyncio.Semaphore(1)

    async def on_job_end(self, _job_result: Any) -> None:
        """Harbor owns permit cleanup; no account lease survives the job."""
