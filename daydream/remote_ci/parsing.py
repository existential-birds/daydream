"""Strictly normalize untrusted GitHub REST responses before CI evaluation."""

from __future__ import annotations

from typing import Sequence, cast

from daydream.remote_ci.evidence import (
    _CHECK_FAIL,
    _CHECK_PASS,
    _CHECK_PENDING,
    _URL_CHARS,
    _WORKFLOW_STATES,
    CIObservation,
    ObservationState,
    PRCIBinding,
    RemoteCIIdentityMismatch,
    RemoteCITarget,
    RequiredContext,
    RequiredPolicy,
    _field,
    _is_positive_int,
    _mapping,
    _normalize_repository,
    _optional_url,
    _require_sha,
    _required_text,
    _safe_diagnostic,
    _safe_url,
    _sequence,
)


def parse_pr_ci_binding(raw: object, target: RemoteCITarget) -> PRCIBinding:
    """Validate the latest REST PR row against the one fixed target."""
    row = _mapping(raw, "pull request")
    number = _field(row, "number", "pull request")
    if not _is_positive_int(number):
        raise ValueError("pull request number is invalid")
    if number != target.pr_number:
        raise RemoteCIIdentityMismatch("pull request number changed")
    pr_url = _safe_url(_field(row, "html_url", "pull request"), _URL_CHARS, "PR URL")
    if pr_url != target.pr_url:
        raise RemoteCIIdentityMismatch("pull request URL changed")
    state = _field(row, "state", "pull request")
    if state not in {"open", "closed"}:
        raise ValueError("pull request state is invalid")

    base = _mapping(_field(row, "base", "pull request"), "pull request base")
    head = _mapping(_field(row, "head", "pull request"), "pull request head")
    base_repo = _mapping(_field(base, "repo", "pull request base"), "base repository")
    head_repo = _mapping(_field(head, "repo", "pull request head"), "head repository")
    base_repository = _normalize_repository(
        _field(base_repo, "full_name", "base repository"), "base repository"
    )
    head_repository = _normalize_repository(
        _field(head_repo, "full_name", "head repository"), "head repository"
    )
    base_ref = _required_text(_field(base, "ref", "pull request base"), "base ref")
    head_ref = _required_text(_field(head, "ref", "pull request head"), "head ref")
    if (
        base_repository != target.base_repository
        or base_ref != target.base_ref
        or head_repository != target.head_repository
        or head_ref != target.head_ref
    ):
        raise RemoteCIIdentityMismatch("pull request base/head identity changed")

    head_sha = _require_sha(_field(head, "sha", "pull request head"), "head SHA")
    raw_merge = _field(row, "merge_commit_sha", "pull request")
    merge_sha = None if raw_merge is None else _require_sha(raw_merge, "merge SHA")
    return PRCIBinding(
        pr_number=number,
        pr_url=pr_url,
        base_repository=base_repository,
        base_ref=base_ref,
        head_repository=head_repository,
        head_ref=head_ref,
        head_sha=head_sha,
        merge_sha=merge_sha,
        state=state,
    )


def _parse_required_entry(
    raw: object,
    *,
    app_key: str,
    app_key_optional: bool = False,
) -> RequiredContext:
    row = _mapping(raw, "required status check")
    context = _required_text(_field(row, "context", "required status check"), "required context")
    app_id = row.get(app_key) if app_key_optional else _field(
        row, app_key, "required status check"
    )
    if app_id is not None and not _is_positive_int(app_id):
        raise ValueError("required status-check app id must be positive or null")
    return RequiredContext(context=context, app_id=app_id)


def _normalized_policy(contexts: Sequence[RequiredContext], strict: bool) -> RequiredPolicy:
    by_name: dict[str, set[int | None]] = {}
    for item in contexts:
        by_name.setdefault(item.context, set()).add(item.app_id)
    normalized: list[RequiredContext] = []
    for context, app_ids in by_name.items():
        pinned = sorted(item for item in app_ids if item is not None)
        if pinned:
            normalized.extend(RequiredContext(context, item) for item in pinned)
        else:
            normalized.append(RequiredContext(context, None))
    normalized.sort(key=lambda item: (item.context, -1 if item.app_id is None else item.app_id))
    return RequiredPolicy(tuple(normalized), strict)


def parse_required_policy(active_rules: object, classic: object | None) -> RequiredPolicy:
    """Union active ruleset and classic branch-protection requirements."""
    contexts: list[RequiredContext] = []
    strict = False
    for raw_rule in _sequence(active_rules, "active branch rules"):
        rule = _mapping(raw_rule, "active branch rule")
        rule_type = _required_text(_field(rule, "type", "active branch rule"), "rule type")
        if rule_type != "required_status_checks":
            continue
        parameters = _mapping(
            _field(rule, "parameters", "required status-check rule"),
            "required status-check parameters",
        )
        raw_strict = _field(
            parameters,
            "strict_required_status_checks_policy",
            "required status-check parameters",
        )
        if not isinstance(raw_strict, bool):
            raise ValueError("ruleset strict policy must be boolean")
        strict = strict or raw_strict
        for item in _sequence(
            _field(parameters, "required_status_checks", "required status-check parameters"),
            "required status checks",
        ):
            contexts.append(
                _parse_required_entry(
                    item,
                    app_key="integration_id",
                    app_key_optional=True,
                )
            )

    if classic is not None:
        classic_row = _mapping(classic, "classic required checks")
        raw_strict = _field(classic_row, "strict", "classic required checks")
        if not isinstance(raw_strict, bool):
            raise ValueError("classic strict policy must be boolean")
        strict = strict or raw_strict
        for raw_context in _sequence(
            _field(classic_row, "contexts", "classic required checks"), "classic contexts"
        ):
            contexts.append(RequiredContext(_required_text(raw_context, "classic context"), None))
        for item in _sequence(
            _field(classic_row, "checks", "classic required checks"), "classic checks"
        ):
            contexts.append(_parse_required_entry(item, app_key="app_id"))
    return _normalized_policy(contexts, strict)


def parse_active_workflows(rows: object) -> tuple[dict[str, str | int], ...]:
    """Validate the Actions inventory and retain active workflow identity only."""
    active: list[dict[str, str | int]] = []
    seen_ids: set[int] = set()
    for raw in _sequence(rows, "workflow inventory"):
        row = _mapping(raw, "workflow")
        row_id = _field(row, "id", "workflow")
        if not _is_positive_int(row_id) or row_id in seen_ids:
            raise ValueError("workflow id must be unique and positive")
        seen_ids.add(row_id)
        name = _required_text(_field(row, "name", "workflow"), "workflow name")
        path = _required_text(_field(row, "path", "workflow"), "workflow path")
        state = _field(row, "state", "workflow")
        if not isinstance(state, str) or state not in _WORKFLOW_STATES:
            raise ValueError("workflow state is unsupported")
        if state == "active":
            active.append({"id": row_id, "name": name, "path": path, "state": state})
    active.sort(key=lambda row: cast(int, row["id"]))
    return tuple(active)


def _parse_check_state(status: object, conclusion: object) -> tuple[ObservationState, str]:
    if not isinstance(status, str):
        raise ValueError("check-run status must be a string")
    if status == "completed":
        if not isinstance(conclusion, str):
            raise ValueError("completed check run requires a conclusion")
        if conclusion in _CHECK_PASS:
            return "pass", conclusion
        if conclusion in _CHECK_FAIL:
            return "fail", conclusion
        raise ValueError("check-run conclusion is unsupported")
    if status in _CHECK_PENDING:
        if conclusion is not None:
            raise ValueError("incomplete check run cannot have a conclusion")
        return "pending", status
    raise ValueError("check-run status is unsupported")


def _parse_status_state(state: object) -> ObservationState:
    if state == "success":
        return "pass"
    if state == "pending":
        return "pending"
    if state in {"failure", "error"}:
        return "fail"
    raise ValueError("legacy status state is unsupported")


def parse_observations(
    check_runs: object,
    statuses: object,
    *,
    expected_sha: str,
    limit: int,
) -> tuple[CIObservation, ...]:
    """Strictly parse and reduce observations for exactly one commit SHA."""
    _require_sha(expected_sha, "expected SHA")
    if not _is_positive_int(limit):
        raise ValueError("diagnostic limit must be positive")
    checks_by_producer: dict[tuple[str, int], tuple[int, CIObservation]] = {}
    statuses_by_context: dict[str, CIObservation] = {}
    seen_check_ids: set[int] = set()
    seen_status_ids: set[int] = set()

    for raw in _sequence(check_runs, "check runs"):
        row = _mapping(raw, "check run")
        row_id = _field(row, "id", "check run")
        if not _is_positive_int(row_id) or row_id in seen_check_ids:
            raise ValueError("check-run id must be unique and positive")
        seen_check_ids.add(row_id)
        name = _required_text(_field(row, "name", "check run"), "check-run name")
        if _require_sha(_field(row, "head_sha", "check run"), "check-run SHA") != expected_sha:
            raise ValueError("check-run evidence belongs to a different SHA")
        app = _mapping(_field(row, "app", "check run"), "check-run app")
        app_id = _field(app, "id", "check-run app")
        if not _is_positive_int(app_id):
            raise ValueError("check-run app id must be positive")
        state, raw_state = _parse_check_state(
            _field(row, "status", "check run"), _field(row, "conclusion", "check run")
        )
        output = _mapping(_field(row, "output", "check run"), "check-run output")
        diagnostic = _safe_diagnostic(
            (
                _field(output, "title", "check-run output"),
                _field(output, "summary", "check-run output"),
            ),
            limit,
        )
        observation = CIObservation(
            source="check_run",
            context=name,
            app_id=app_id,
            state=state,
            raw_state=raw_state,
            url=_optional_url(
                _field(row, "details_url", "check run"), _URL_CHARS, "check-run URL"
            ),
            diagnostic=diagnostic,
        )
        key = (name, app_id)
        prior = checks_by_producer.get(key)
        if prior is None or row_id > prior[0]:
            checks_by_producer[key] = (row_id, observation)

    for raw in _sequence(statuses, "legacy statuses"):
        row = _mapping(raw, "legacy status")
        row_id = _field(row, "id", "legacy status")
        if not _is_positive_int(row_id) or row_id in seen_status_ids:
            raise ValueError("legacy status id must be unique and positive")
        seen_status_ids.add(row_id)
        context = _required_text(_field(row, "context", "legacy status"), "status context")
        if _require_sha(_field(row, "sha", "legacy status"), "status SHA") != expected_sha:
            raise ValueError("legacy status evidence belongs to a different SHA")
        status_key = context.casefold()
        if status_key in statuses_by_context:
            continue
        legacy_raw_state = _field(row, "state", "legacy status")
        state = _parse_status_state(legacy_raw_state)
        statuses_by_context[status_key] = CIObservation(
            source="status",
            context=context,
            app_id=None,
            state=state,
            raw_state=cast(str, legacy_raw_state),
            url=_optional_url(
                _field(row, "target_url", "legacy status"), _URL_CHARS, "status URL"
            ),
            diagnostic=_safe_diagnostic(
                (_field(row, "description", "legacy status"),), limit
            ),
        )

    observations = [item[1] for item in checks_by_producer.values()]
    observations.extend(statuses_by_context.values())
    observations.sort(
        key=lambda item: (
            0 if item.source == "check_run" else 1,
            item.context if item.source == "check_run" else item.context.casefold(),
            -1 if item.app_id is None else item.app_id,
        )
    )
    return tuple(observations)
