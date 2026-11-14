"""Record Identity And Retained Tree."""

from __future__ import annotations


def test_persisted_record_artifacts_admit_only_current_host_identity() -> None:
    """Loaded records must carry this scope, revision, run lineage and canonical UIDs.

    The runner fault journeys corrupt bytes or shape; foreign identity metadata is
    otherwise only rejected implicitly through those slower whole-run paths.
    """
    from daydream.phases.review import valid_record_artifact

    revision = {'head_sha': 'h' * 40}
    valid = {'issues': [], 'scope_id': 'python', 'analyzed_revision': revision, 'originating_run_id': 'run-1'}
    assert valid_record_artifact(valid, scope_id='python', analyzed_revision=revision)
    faults = [{**valid, 'unexpected': True}, {**valid, 'scope_id': 'foreign'},
              {**valid, 'analyzed_revision': {'head_sha': 'f' * 40}}, {**valid, 'originating_run_id': ''},
              {**valid, 'incomplete': False}, {**valid, 'issues': {}}, {**valid, 'issues': [{'uid': 1}]},
              {**valid, 'issues': [{'uid': 'react:1'}]}, {**valid, 'issues': [{'uid': 'python:01'}]}]
    assert not any(valid_record_artifact(fault, scope_id='python', analyzed_revision=revision) for fault in faults)
