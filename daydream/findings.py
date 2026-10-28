"""Strict findings handoff from unprivileged analysis to privileged posting.

Artifacts carry raw finding fields and placement, never rendered comments. The
poster validates size, schema and event identity before rendering without PR
checkout access. Dependencies flow from findings to pr_review.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import jsonschema

from daydream import git_ops, pr_review
from daydream.immutable_review import IMMUTABLE_REVIEW_INPUT_SCHEMA, validate_immutable_input
from daydream.json_utils import atomic_write_bytes
from daydream.output_schema import strict_object
from daydream.pr_review import ParsedIssue, PRInfo
from daydream.review_rationale import valid_review_context
from daydream.review_reporting import REVIEW_CONTEXT_SCHEMA
from daydream.review_result import (
    RUN_TARGET_DECISIONS_MAX_BYTES,
    TERMINAL_RESULT_SCHEMA,
    published_attribution_bytes,
    validate_terminal_result,
)

FINDINGS_SCHEMA_VERSION = 2
IMMUTABLE_FINDINGS_SCHEMA_VERSION = 3

MAX_ARTIFACT_BYTES = 1_048_576
MAX_RAW_ARTIFACT_BYTES = MAX_ARTIFACT_BYTES + RUN_TARGET_DECISIONS_MAX_BYTES


def _enforce_max_artifact_bytes(size: int, limit: int = MAX_ARTIFACT_BYTES) -> None:
    if size > limit:
        raise FindingsValidationError(
            f"artifact size check failed: {size} bytes exceeds the {limit}-byte cap"
        )

_FINDING_PROPERTIES = {
    'fingerprint': {'type': 'string', 'pattern': '^[0-9a-f]{64}$'},
    **{name: {'type': 'string'} for name in ('path', 'title', 'body')},
    'line': {'type': ['integer', 'null']}, 'placement': {'enum': ['inline', 'file', 'body']},
    **{name: {'type': ['string', 'null']} for name in ('severity', 'confidence', 'severity_before_demotion')},
    **{name: {'type': 'boolean'} for name in ('is_cross_stack', 'location_distrust', 'severity_off_vocabulary')},
}
FINDINGS_SCHEMA: dict[str, Any] = strict_object({
    'schema_version': {'const': FINDINGS_SCHEMA_VERSION}, 'repo': {'type': 'string'},
    'pr_number': {'type': 'integer'}, 'head_sha': {'type': 'string'}, 'kind': {'enum': ['review', 'diagram']},
    'run_info': {'type': ['string', 'null']},
    'review_warnings': {'type': 'array', 'items': {'type': 'string'}},
    'diagrams': {'type': ['object', 'null']},
    'findings': {'type': 'array', 'items': strict_object(_FINDING_PROPERTIES)},
    'terminal_result': TERMINAL_RESULT_SCHEMA,
    'review_context': REVIEW_CONTEXT_SCHEMA,
})
FINDINGS_SCHEMA['required'].remove('terminal_result')
FINDINGS_SCHEMA['required'].remove('review_context')
FINDINGS_SCHEMA['allOf'] = [{
    'if': {'properties': {'kind': {'const': 'review'}}},
    'then': {'required': ['terminal_result']},
    'else': {'not': {'required': ['terminal_result']}, 'properties': {'findings': {'maxItems': 0}}},
}]
IMMUTABLE_FINDINGS_SCHEMA = copy.deepcopy(FINDINGS_SCHEMA)
IMMUTABLE_FINDINGS_SCHEMA['properties'].update({
    'schema_version': {'const': IMMUTABLE_FINDINGS_SCHEMA_VERSION},
    'kind': {'const': 'review'},
    'immutable_input': IMMUTABLE_REVIEW_INPUT_SCHEMA,
})
IMMUTABLE_FINDINGS_SCHEMA['required'].append('immutable_input')


def _validate_immutable_binding(data: dict[str, Any]) -> None:
    """Keep captured PR provenance distinct from the locally analyzed revision."""
    immutable = data['immutable_input']
    validate_immutable_input(immutable)
    provenance, materialization = immutable['provenance'], immutable['materialization']
    for outer, captured in (('repo', 'repository'), ('pr_number', 'pr_number'), ('head_sha', 'head_sha')):
        if data[outer] != provenance[captured]:
            raise ValueError(f'findings {outer} differs from captured provenance')
    revision = data['terminal_result']['analyzed_revision']
    for analyzed, local in (('head_sha', 'head_sha'), ('merge_base_sha', 'comparison_base_sha'),
                            ('diff_key', 'review_diff_key')):
        if revision[analyzed] != materialization[local]:
            raise ValueError(f'findings analyzed {analyzed} differs from immutable materialization')


def _validate_artifact(data: Any) -> None:
    """Validate the current kind-specific envelope and its coverage semantics."""
    version = data.get("schema_version") if isinstance(data, dict) else None
    if type(version) is not int or version not in (FINDINGS_SCHEMA_VERSION, IMMUTABLE_FINDINGS_SCHEMA_VERSION):
        raise FindingsValidationError(f"artifact failed schema validation: unsupported schema version {version!r}")
    try:
        jsonschema.validate(data, IMMUTABLE_FINDINGS_SCHEMA if version == IMMUTABLE_FINDINGS_SCHEMA_VERSION
                            else FINDINGS_SCHEMA)
        if data["kind"] == "review":
            immutable = version == IMMUTABLE_FINDINGS_SCHEMA_VERSION
            if immutable:
                _validate_immutable_binding(data)
            validate_terminal_result(data["terminal_result"],
                                     expected_head_sha=None if immutable else data["head_sha"])
            if not data["terminal_result"]["projection_valid"] and data["findings"]:
                raise ValueError("untrustworthy projection must not contain findings")
            if 'review_context' in data:
                terminal = data['terminal_result']
                planned = {scope['scope_id'] for scope in terminal['planned_scopes']}
                entries = data['review_context']['scopes']
                if len({entry['scope_id'] for entry in entries}) != len(entries):
                    raise ValueError('duplicate report scope identities')
                for entry in entries:
                    missing_source = (entry['analyzed_revision'] is None
                                      and entry['originating_run_id'] is None
                                      and entry['review_context'] is None)
                    if entry['scope_id'] not in planned or (
                        not missing_source and entry['analyzed_revision'] != terminal['analyzed_revision']
                    ):
                        raise ValueError('report context differs from terminal snapshot inventory')
                    if entry['review_context'] is not None and not valid_review_context(entry['review_context']):
                        raise ValueError('invalid source rationale counts or redaction')

    except (jsonschema.ValidationError, ValueError) as exc:
        message = exc.message if isinstance(exc, jsonschema.ValidationError) else str(exc)
        raise FindingsValidationError(f"artifact failed schema validation: {message}") from exc


class FindingsValidationError(Exception):
    """Artifact read/write size, parse, schema or event-binding validation failed."""


@dataclass
class ArtifactFinding:
    """Validated finding with placement, identity and approval-gate provenance.

    Location and severity provenance preserve the current approval gate inputs.
    severity_before_demotion preserves whether a demoted finding was blocking.
    Non-inline placements have no line; body remains raw until posting.
    """

    fingerprint: str
    path: str
    line: int | None
    placement: str
    title: str
    body: str
    severity: str | None
    confidence: str | None
    is_cross_stack: bool
    location_distrust: bool = False
    severity_before_demotion: str | None = None
    severity_off_vocabulary: bool = False


@dataclass
class FindingsArtifact:
    """Event-bound findings and optional diagrams for the privileged poster.

    review_warnings block approval/stale resolution but permit surviving findings.
    Diagram payloads omit stored Mermaid;
    the poster validates and renders their final specs.
    """

    repo: str
    pr_number: int
    head_sha: str
    run_info: str | None
    findings: list[ArtifactFinding]
    kind: str = "review"
    diagrams: dict[str, Any] | None = None
    review_warnings: tuple[str, ...] = ()
    schema_version: int = FINDINGS_SCHEMA_VERSION
    terminal_result: dict[str, Any] | None = None
    review_context: dict[str, Any] | None = None
    immutable_input: dict[str, Any] | None = None

    @property
    def analysis_complete(self) -> bool:
        """Only explicit validated code review evidence establishes completeness."""
        return self.terminal_result is not None and self.terminal_result["analysis_state"] == "complete"

    @property
    def coverage_notice(self) -> tuple[str, ...]:
        """Render typed incomplete/failed coverage for review notices."""
        if self.terminal_result is None or self.analysis_complete:
            return ()
        state = self.terminal_result["analysis_state"]
        reasons = ", ".join(self.terminal_result["reason_codes"])
        return (f"Review analysis is {state}; required coverage was not completed ({reasons}).",)


def _finding_dict(issue: ParsedIssue, *, placement: str, line: int | None) -> dict[str, Any]:
    """Map one classified issue onto an artifact finding entry."""
    finding = {field.name: getattr(issue, field.name) for field in fields(ArtifactFinding)
               if field.name not in {'placement', 'line'}}
    return {**finding, 'placement': placement, 'line': line}


def build_findings_artifact(
    target_dir: Path,
    pr: PRInfo,
    issues: list[ParsedIssue],
    *,
    run_info: str | None,
    review_warnings: tuple[str, ...] = (),
    kind: str = "review",
    diagrams: dict[str, Any] | None = None,
    renderers: pr_review.ReviewRenderers | None = None,
    review_context: dict[str, Any] | None = None,
    terminal_result: dict[str, Any] | None = None,
    snapshot_diff: str | None = None,
    frozen_only: bool = False,
    immutable_input: dict[str, Any] | None = None,
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
) -> dict[str, Any]:
    """Classify against the PR diff before producing the handoff artifact.

    Inline findings retain snapped lines; file/body findings have line=None.
    Diagram-only callers provide empty issues and specs without stored Mermaid.
    Classification runs where PR Git objects are available, before privileged posting.
    Immutable callers supply the validated input and keep the prepared source's
    inert Git context active; classification uses its local commits, while the
    envelope retains the original captured publication identity.
    """
    if (kind == 'review') != (terminal_result is not None):
        raise FindingsValidationError('review requires terminal_result; diagram forbids code coverage')
    if kind == 'review' and snapshot_diff is None:
        raise FindingsValidationError('review requires the captured snapshot diff')
    if frozen_only and (terminal_result is None or snapshot_diff is None
            or hashlib.sha256(snapshot_diff.encode()).hexdigest() != terminal_result['analyzed_revision']['diff_key']):
        raise FindingsValidationError('frozen findings placement requires the bound snapshot diff')
    artifact: dict[str, Any] = {
        "schema_version": IMMUTABLE_FINDINGS_SCHEMA_VERSION if immutable_input is not None else FINDINGS_SCHEMA_VERSION,
        "repo": f"{pr.owner}/{pr.repo}",
        "pr_number": pr.number,
        "head_sha": pr.head_sha,
        "run_info": run_info,
        "kind": kind,
        "diagrams": diagrams,
        "findings": [],
        "review_warnings": list(review_warnings),
        **({"terminal_result": copy.deepcopy(terminal_result)} if terminal_result is not None else {}),
        **({"review_context": copy.deepcopy(review_context)} if review_context is not None else {}),
        **({"immutable_input": copy.deepcopy(immutable_input)} if immutable_input is not None else {}),
    }
    if immutable_input is not None:
        _validate_artifact(artifact)
        local = immutable_input['materialization']
        pr = replace(pr, head_sha=local['head_sha'], base_sha=local['comparison_base_sha'])
    classified = pr_review.classify(target_dir, pr, issues, auth=auth, renderers=renderers,
                                    snapshot_diff=snapshot_diff, frozen_only=frozen_only)
    findings = [
        _finding_dict(issue, placement="inline", line=entry.line)
        for entry, issue in zip(classified.inline, classified.inline_issues, strict=True)
    ]
    findings.extend(_finding_dict(issue, placement="file", line=None) for issue in classified.file_level)
    findings.extend(_finding_dict(issue, placement="body", line=None) for issue in classified.body_only)
    artifact['findings'] = findings
    return artifact


def build_committed_findings_artifact(
    target_dir: Path, payload: dict[str, Any], *, snapshot_diff: str,
) -> dict[str, Any]:
    """Project a sealed immutable review through snapshot-local finding placement.

    The caller retains the prepared source's inert Git context and its artifact
    session. Unresolved concerns and native diagnostics remain in the paired
    committed result; this handoff retains confirmed findings and typed coverage.
    """
    from daydream.committed_review_model import validate_committed_result

    validate_committed_result(payload)
    if payload['schema_version'] != 2:
        raise FindingsValidationError('committed findings require an immutable result')
    if hashlib.sha256(snapshot_diff.encode()).hexdigest() != payload['analyzed_revision']['diff_key']:
        raise FindingsValidationError('committed findings differ from the captured review diff')
    immutable = payload['immutable_input']
    original = immutable['provenance']
    owner, repo = original['repository'].split('/', 1)
    pr = PRInfo(original['pr_number'], original['head_sha'], original['comparison_base_sha'],
                '', '', owner, repo, '')
    issues = []
    for finding in payload['findings']:
        semantic = finding['payload']
        body = '\n\n'.join([semantic['claim'], *(
            f'**{label}:** {semantic[key]}'
            for key, label in (('trigger', 'Trigger'), ('consequence', 'Consequence'),
                               ('grounds', 'Supporting grounds'))
        )])
        issues.append(ParsedIssue(
            path=semantic['file'], line=semantic['line'], title=finding['title'], body=body,
            severity=None, confidence=None, is_cross_stack=False,
            fingerprint=pr_review.compute_fingerprint(semantic['file'], finding['title'], body),
        ))
    return build_findings_artifact(
        target_dir, pr, issues, run_info=f"Committed review {payload['run_id']}",
        terminal_result=payload['terminal_result'], snapshot_diff=snapshot_diff, immutable_input=immutable,
    )


def write_findings_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Write UTF-8 JSON with parent creation, enforcing the cap on exact output bytes."""
    try:
        _validate_artifact(artifact)
        payload = _artifact_payload(artifact)
        attribution = published_attribution_bytes(artifact.get('terminal_result', {}))
        if len(payload) - attribution > MAX_ARTIFACT_BYTES and 'review_context' in artifact:
            core = {key: value for key, value in artifact.items() if key != 'review_context'}
            core_payload = _artifact_payload(core)
            _enforce_max_artifact_bytes(len(core_payload) - attribution)
            fitted = {**core, 'review_context': copy.deepcopy(artifact['review_context'])}
            context = fitted['review_context']
            while len(payload) - attribution > MAX_ARTIFACT_BYTES:
                verification = context['verification']
                if context['author_reported'] is not None:
                    context['author_reported'] = None
                elif context['scopes']:
                    context['scopes'].pop()
                    context['scopes_omitted'] += 1
                elif verification['value']:
                    priority = {'test-verdict.json': 2, 'remote-ci-verdict.json': 1}
                    name = min(verification['value'], key=lambda key: (priority.get(key, 0), key))
                    omitted = verification.setdefault('omitted_artifacts', [])
                    if name not in omitted:
                        if len(omitted) == 16:
                            payload = core_payload
                            break
                        omitted.append(name)
                    verification['value'].pop(name)
                    verification['value'] = verification['value'] or None
                    verification['status'] = 'failed'
                else:
                    payload = core_payload
                    break
                payload = _artifact_payload(fitted)
            logging.getLogger(__name__).warning('Optional review context omitted to fit the findings artifact budget')
        _enforce_max_artifact_bytes(len(payload) - attribution)
        atomic_write_bytes(path, payload)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise FindingsValidationError(f"artifact write failed: {path}: {exc}") from exc


def _artifact_payload(artifact: dict[str, Any]) -> bytes:
    """Use the exact published encoding for both capacity checks and final bytes."""
    return (json.dumps(artifact, indent=2, ensure_ascii=False) + '\n').encode('utf-8')


def load_findings_artifact(
    path: Path,
    *,
    expected_repo: str,
    expected_pr_number: int,
    expected_head_sha: str,
    expected_run_id: str | None = None,
    expected_base_sha: str | None = None,
    expected_comparison_sha: str | None = None,
) -> FindingsArtifact:
    """Validate untrusted output against event facts before privileged posting.

    Check size before reading, then JSON, schema and independently trusted identity.
    Immutable artifacts additionally require requested base and comparison facts;
    their captured provenance never supplies publisher authority.
    """
    try:
        size = path.stat().st_size
        _enforce_max_artifact_bytes(size, MAX_RAW_ARTIFACT_BYTES)
        with path.open('rb', buffering=0) as stream:
            captured = stream.read(MAX_RAW_ARTIFACT_BYTES + 1)
        _enforce_max_artifact_bytes(len(captured), MAX_RAW_ARTIFACT_BYTES)
        raw = captured.decode('utf-8')
    except (OSError, UnicodeError) as exc:
        raise FindingsValidationError(f"artifact read failed: {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise FindingsValidationError(f"artifact JSON parse failed: {exc}") from exc

    _validate_artifact(data)
    try:
        _enforce_max_artifact_bytes(
            len(_artifact_payload(data)) - published_attribution_bytes(data.get('terminal_result', {})))
    except (ValueError, TypeError, RecursionError) as exc:
        raise FindingsValidationError(f'artifact size encoding failed: {exc}') from exc

    if data['schema_version'] == IMMUTABLE_FINDINGS_SCHEMA_VERSION and (
        type(expected_pr_number) is not int or expected_pr_number <= 0
    ):
        raise FindingsValidationError('immutable publisher requires an exact positive trusted PR number')

    for field_name, expected in (
        ("repo", expected_repo),
        ("pr_number", expected_pr_number),
        ("head_sha", expected_head_sha),
    ):
        declared = data[field_name]
        if declared != expected:
            raise FindingsValidationError(
                f"artifact {field_name} {declared!r} does not match event-derived {field_name} {expected!r}"
            )

    if data['schema_version'] == IMMUTABLE_FINDINGS_SCHEMA_VERSION:
        provenance = data['immutable_input']['provenance']
        for field_name, expected_binding in (('requested_base_sha', expected_base_sha),
                                             ('comparison_base_sha', expected_comparison_sha)):
            if expected_binding is None:
                raise FindingsValidationError(f'immutable artifact requires independently trusted {field_name}')
            if provenance[field_name] != expected_binding:
                raise FindingsValidationError(
                    f'artifact {field_name} {provenance[field_name]!r} does not match trusted '
                    f'{field_name} {expected_binding!r}'
                )

    if expected_run_id is not None:
        declared = data.get("terminal_result", {}).get("run_id")
        if declared != expected_run_id:
            raise FindingsValidationError(
                f"artifact run_id {declared!r} does not match expected run_id {expected_run_id!r}"
            )

    return FindingsArtifact(**{**data, 'findings': [ArtifactFinding(**f) for f in data['findings']],
                               'review_warnings': tuple(data['review_warnings'])})
