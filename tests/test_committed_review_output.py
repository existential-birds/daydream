"""Real installation faults for the committed findings/coverage output owner.

The native entry journeys prove ordinary review output. These cases protect the
new bundle's staging and publication, which legacy findings-output tests miss.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from daydream.artifact_visibility import (
    ArtifactDisposition,
    ArtifactVisibilityError,
    OutputLabel,
    open_artifact_session,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.committed_review_model import planned_work_id
from daydream.committed_review_output import write_committed_review_output
from daydream.review_result import AnalyzedRevision, ReviewCoverage
from tests.harness.git_helpers import git, work_context


def _empty_result(repo: Path, run_id: str) -> dict[str, Any]:
    head = git(repo, 'rev-parse', 'HEAD')
    revision = AnalyzedRevision(head, head, '0' * 64)
    coverage = ReviewCoverage(run_id, revision, (), ('no_diff',))
    coverage.record_phase('no_diff', 'complete', noop=True)
    return {
        'schema_version': 1, 'run_id': run_id, 'analyzed_revision': revision.to_dict(),
        'assignments': [], 'planned_work_id': planned_work_id(revision.to_dict(), []),
        'terminal_result': coverage.finalize('completed'), 'claims': [], 'findings': [],
        'settings': {}, 'execution': [],
    }


@pytest.mark.parametrize('fault', ['report-stage', 'bundle-stage', 'install', 'competing-report'])
async def test_failed_current_output_preserves_prior_or_competing_bytes(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """A failed new output cannot expose an old bundle as a successful new run.

    Staging and installation faults restore prior bytes. A competing public
    report keeps the prior bundle in the owner's recovery tree until the
    conflict can be resolved without replacing the competing bytes.
    """
    owner = resolve_private_workspace_owner(git_repo, locations=private_root_locations())
    work = work_context(git_repo, run_id='output')
    async with open_artifact_session(work, session_id='prior', owner=owner) as session:
        session.register_destination(git_repo / '.daydream', label=OutputLabel.PUBLIC_DAYDREAM)
        session.register_destination(git_repo / '.review-output.md', label=OutputLabel.PUBLIC_REVIEW_OUTPUT)
        session.register_trajectory_output(None)
        write_committed_review_output(session, _empty_result(git_repo, 'prior'))
        session.finalize_frozen(session.freeze_unproduced(), disposition=ArtifactDisposition.COMPLETE)
    bundle = git_repo / '.daydream' / 'committed-review.json'
    report = git_repo / '.review-output.md'
    prior_bundle, prior_report = bundle.read_bytes(), report.read_bytes()
    injected = False
    allocate, link = tempfile.mkstemp, os.link
    with pytest.raises((OSError, ArtifactVisibilityError)):
        async with open_artifact_session(work, session_id='current', owner=owner) as session:
            session.register_destination(git_repo / '.daydream', label=OutputLabel.PUBLIC_DAYDREAM)
            session.register_destination(git_repo / '.review-output.md', label=OutputLabel.PUBLIC_REVIEW_OUTPUT)
            session.register_trajectory_output(None)

            def fail_allocation(*args: Any, **kwargs: Any) -> tuple[int, str]:
                nonlocal injected
                target = session.review_output.parent if fault == 'report-stage' else session.daydream_dir
                if fault.endswith('-stage') and Path(kwargs['dir']) == target and not injected:
                    injected = True
                    raise OSError('committed output staging failed')
                return allocate(*args, **kwargs)

            def fail_install(source: Any, destination: Any, **kwargs: Any) -> None:
                nonlocal injected
                if fault == 'install' and Path(destination) == report and not injected:
                    injected = True
                    raise OSError('committed output installation failed')
                link(source, destination, **kwargs)

            monkeypatch.setattr(tempfile, 'mkstemp', fail_allocation)
            monkeypatch.setattr(os, 'link', fail_install)
            write_committed_review_output(session, _empty_result(git_repo, 'current'))
            frozen = session.freeze_unproduced()
            if fault == 'competing-report':
                report.write_bytes(b'competing report\n')
                injected = True
            session.finalize_frozen(frozen, disposition=ArtifactDisposition.COMPLETE)
    assert injected
    retained_bundle = (
        owner.artifact_state_root / 'canonical' / '.daydream' / 'committed-review.json'
        if fault == 'competing-report' else bundle
    )
    assert retained_bundle.read_bytes() == prior_bundle
    assert json.loads(retained_bundle.read_text())['run_id'] == 'prior'
    if fault == 'competing-report':
        assert not bundle.exists()
        assert json.loads((frozen.root / '.daydream' / 'committed-review.json').read_text())['run_id'] == 'current'
    assert report.read_bytes() == (b'competing report\n' if fault == 'competing-report' else prior_report)
