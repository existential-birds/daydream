"""Account admission at Harbor's real whole-trial queue and event-loop boundary."""
from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.benchmark.reviewbench import harbor_cli


@pytest.fixture
def queue_types(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    pytest.importorskip('harbor')
    queue = importlib.import_module('harbor.trial.queue').TrialQueue
    trial = importlib.import_module('harbor.trial.trial').Trial
    monkeypatch.setattr(harbor_cli, 'verify_runtime', lambda: None)
    return queue, trial


def trial_config(name: str) -> Any:
    return SimpleNamespace(trial_name=name, agent=SimpleNamespace(n_concurrent=None))


async def admitted_queue(queue_type: Any, tmp_path: Path) -> Any:
    queue = queue_type(n_concurrent=2)
    job = SimpleNamespace(_trial_queue=queue, config=SimpleNamespace(n_concurrent_trials=2), job_dir=tmp_path)
    attach = importlib.import_module('harbor.cli.job_plugins').attach_job_plugin
    await attach(job, 'daydream.benchmark.reviewbench.harbor_subscription_plugin:SubscriptionAdmissionPlugin')
    return queue


@pytest.mark.parametrize('account_admission', [False, True])
async def test_judge_occupancy_does_not_consume_queued_reviewers_frozen_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queue_types: tuple[Any, Any], account_admission: bool,
) -> None:
    """Real Harbor queue admission must happen before phase timers.

    The existing refresh-lock tests prove credential integrity, but miss healthy
    reviewers timing out while a different trial's judge owns that lock. Replace
    only Trial.create's external workload; keep Harbor submission, concurrency,
    retries and cancellation scopes plus actual event-loop timers and locks.
    The unadmitted control reproduces that timeout with the identical workload.
    """
    queue_type, trial_type = queue_types
    queue = (await admitted_queue(queue_type, tmp_path) if account_admission else queue_type(n_concurrent=2))
    account = asyncio.Lock()
    judge_entered, release_judge, second_created = asyncio.Event(), asyncio.Event(), asyncio.Event()
    result = SimpleNamespace(exception_info=None)

    class TimedTrial:
        def __init__(self, config: Any) -> None:
            self.config = config

        def add_hook(self, *args: Any) -> None:
            raise AssertionError('No custom lifecycle hooks are needed for admission')

        async def run(self) -> Any:
            # These are the frozen reviewer and verifier phase timeout scopes.
            async with asyncio.timeout(0.05):
                async with account:
                    await asyncio.sleep(0)
            if self.config.trial_name == 'first':
                async with asyncio.timeout(1):
                    async with account:
                        judge_entered.set()
                        await release_judge.wait()
            return result

    async def create(config: Any) -> TimedTrial:
        if config.trial_name == 'second':
            second_created.set()
        return TimedTrial(config)

    monkeypatch.setattr(trial_type, 'create', create)
    first = asyncio.create_task(queue.submit(trial_config('first')))
    await asyncio.wait_for(judge_entered.wait(), timeout=1)
    second = asyncio.create_task(queue.submit(trial_config('second')))
    try:
        if account_admission:
            # Allow the queued workload more than an entire reviewer budget.
            # Its Trial has not even been created, so no phase budget is spent.
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(second_created.wait(), timeout=0.1)
            assert not second.done()
            release_judge.set()
            assert await asyncio.wait_for(second, timeout=1) is result
            assert second_created.is_set()
            policy = json.loads((tmp_path / 'daydream-subscription-admission.json').read_text())
            assert policy['requested_concurrency'] == 2 and policy['effective_concurrency'] == 1
        else:
            await asyncio.wait_for(second_created.wait(), timeout=1)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(second, timeout=1)
    finally:
        release_judge.set()
        await asyncio.wait_for(first, timeout=1)
        if not second.done():
            second.cancel()
            await asyncio.gather(second, return_exceptions=True)


@pytest.mark.parametrize('interruption', ['active-cancel', 'queued-cancel', 'creation-failure'])
async def test_harbor_owns_admission_release_on_cancellation_and_start_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, queue_types: tuple[Any, Any], interruption: str,
) -> None:
    """A failed/cancelled trial must not strand the account's remaining work.

    Custom START/END hook permits can leak because START is outside Harbor's
    try/finally and earlier END hooks may fail. The real queue's async-with must
    release its whole-trial permit even when creation or running is interrupted.
    """
    queue_type, trial_type = queue_types
    queue = await admitted_queue(queue_type, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    completed: list[str] = []

    class Workload:
        def __init__(self, config: Any) -> None:
            self.config = config

        def add_hook(self, *args: Any) -> None:
            raise AssertionError('Admission must use Harbor queue cleanup')

        async def run(self) -> Any:
            if self.config.trial_name == 'first':
                entered.set()
                await release.wait()
            completed.append(self.config.trial_name)
            return SimpleNamespace(exception_info=None)

    async def create(config: Any) -> Workload:
        if interruption == 'creation-failure' and config.trial_name == 'first':
            entered.set()
            raise RuntimeError('external trial creation failed')
        return Workload(config)

    monkeypatch.setattr(trial_type, 'create', create)
    first = asyncio.create_task(queue.submit(trial_config('first')))
    await asyncio.wait_for(entered.wait(), timeout=1)
    if interruption == 'creation-failure':
        with pytest.raises(RuntimeError, match='external trial creation failed'):
            await first
    elif interruption == 'active-cancel':
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    else:
        queued = asyncio.create_task(queue.submit(trial_config('cancelled-waiter')))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        release.set()
        await asyncio.wait_for(first, timeout=1)
    await asyncio.wait_for(queue.submit(trial_config('remaining')), timeout=1)
    assert completed[-1] == 'remaining'
    assert 'cancelled-waiter' not in completed
