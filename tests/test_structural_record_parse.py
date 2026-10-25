"""Structural findings remain partitioned through review and resume."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ResultEvent
from daydream.backends.pi import PiBackend
from daydream.deep.detection import StackAssignment
from daydream.deep.review_steps import _per_stack_body, _step_per_stack_parse
from daydream.extensions import Registry, get_registry
from daydream.flows.engine import FlowContext
from daydream.hunk_index import write_hunk_index
from daydream.run_context import InteractionPolicy, RunContext
from tests.harness.review_result import records_artifact, review_coverage
from tests.harness.stub_backend import review_stage_result
from tests.harness.table_cases import isolated_case


@pytest.mark.parametrize("start_at", [None, "merge", "fix"])
@pytest.mark.parametrize("uids", [[], ["structure:1", "structure:3"]])
async def test_parse_preserves_structural_partition_on_resume(
    tmp_path: Path, make_config: Any, make_work: Any, start_at: str | None, uids: list[str],
) -> None:
    dd = tmp_path / ".daydream/deep"
    dd.mkdir(parents=True)
    coverage = review_coverage(files=("api.py",), phases=())
    primary = records_artifact(coverage, "python")
    (dd / "stack-python-records.json").write_text(json.dumps(primary))
    issues = [{"id": 1, "uid": uid, "file": "api.py", "line": 1, "description": "Boundary mismatch",
               "severity": "high", "confidence": "HIGH", "rationale": "Shared contract", "evidence": "api.py:1"}
              for uid in uids]
    structural = records_artifact(coverage, "structure", issues)
    path = dd / "stack-structure-records.json"
    path.write_text(json.dumps(structural))
    ctx = FlowContext(config=make_config(tmp_path, start_at=start_at), work=make_work(tmp_path), registry=Registry(),
        data={"dd": dd, "review_coverage": coverage, "stacks": [StackAssignment("python", ["api.py"]),
                                    StackAssignment("structure", ["api.py"])], "failed_stacks": {}},
    )
    assert await _step_per_stack_parse(ctx) is None
    assert json.loads(path.read_text()) == structural
    assert ctx.data["record_pool"].structural == issues
    assert ctx.data["record_pool"].language == []

@pytest.mark.parametrize("start_at", [None, "per-stack"])
async def test_per_stack_rerun_clears_stale_structural_outputs_before_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any, make_work: Any, start_at: str | None,
) -> None:
    dd = tmp_path / ".daydream/deep"
    dd.mkdir(parents=True)
    artifacts = [dd / name for name in ("stack-structure-records.json", "stack-structure-review.md",)]
    for path in artifacts:
        path.write_text("STALE")
    import hashlib

    from daydream.review_result import AnalyzedRevision, PlannedScope, ReviewCoverage
    from tests.harness.git_helpers import git, seed_feature_branch

    seed_feature_branch(tmp_path, base={"api.py": "value = 0\n"}, feature={"api.py": "value = 1\n"})
    diff = dd / "diff.patch"
    diff.write_text("diff --git a/api.py b/api.py\n--- a/api.py\n+++ b/api.py\n"
                    "@@ -1 +1 @@\n-value = 0\n+value = 1\n")
    write_hunk_index(dd, diff.read_text())
    intent = dd / "intent.md"
    intent.write_text("Preserve behavior")
    alternatives = dd / "alternatives.json"
    alternatives.write_text("[]")
    attempted: list[str] = []

    async def review(self: PiBackend, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        assert not artifacts[0].exists()
        assert "STALE" not in prompt
        attempted.append("structure" if "repository-wide interactions" in prompt else "primary")
        yield ResultEvent(structured_output=review_stage_result(prompt, []), continuation=None)

    monkeypatch.setattr(PiBackend, "execute", review)
    backend = PiBackend(model="test", reasoning_effort="high")
    ctx = FlowContext(
        config=make_config(tmp_path, start_at=start_at), work=make_work(tmp_path), registry=get_registry(),
        allow_standalone_artifacts=True, run_context=RunContext(InteractionPolicy(interactive=False)),
        _backend_factory=lambda *_: backend,
        data={"dd": dd, "review_coverage": ReviewCoverage("source-test", AnalyzedRevision(
                  git(tmp_path, "rev-parse", "HEAD"), git(tmp_path, "rev-parse", "main"),
                  hashlib.sha256(diff.read_bytes()).hexdigest()),
                  [PlannedScope(scope, scope, ("api.py",)) for scope in ("python", "structure")], ()) ,
              "diff_path": diff, "diff": diff.read_text(), "intent_path": intent,
              "alts_path": alternatives, "exploration_dir": None, "failed_stacks": {},
              "stacks": [StackAssignment("python", ["api.py"]), StackAssignment("structure", ["api.py"])]},
    )
    await _per_stack_body(ctx, include_alternatives=False)
    assert sorted(attempted) == ["primary", "structure"]
    assert json.loads(artifacts[0].read_text())["issues"] == []
    assert artifacts[1].read_text().startswith("# Review")
    assert ctx.data["failed_stacks"] == {}


def test_record_artifact_rejects_noncanonical_ordinals(
    subtests: pytest.Subtests,
) -> None:
    cases: list[str] = ["²", "١", "01", "0", "-1"]
    for ordinal in cases:
        with subtests.test(ordinal=ordinal):
            from daydream.phases.review import valid_record_artifact
            from tests.harness.review_result import records_artifact, review_coverage

            coverage = review_coverage()
            artifact = records_artifact(coverage, "python", [{"uid": f"python:{ordinal}"}])
            assert not valid_record_artifact(artifact, scope_id="python", analyzed_revision=coverage.revision.to_dict())


def test_report_collection_validates_optional_bound_sidecar(
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    """Unavailable/tampered rationale never erases valid source records or invents attribution."""
    table_root = tmp_path
    row_index = 0
    cases: list[str] = [
        "missing",
        "current",
        "unattributed",
        "counts",
        "notes",
        "secret",
        "origin",
        "scope",
        "revision",
        "run",
        "nested-checks",
        "nested-sidecar",
    ]
    for context_kind in cases:
        row_index += 1
        with subtests.test(context_kind=context_kind):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                from daydream.deep.report_context import collect_report_context
                from daydream.phases.review import valid_record_artifact

                coverage = review_coverage()
                artifact = records_artifact(coverage, "python")
                context: dict[str, Any] = {
                    "admitted_stages": 1,
                    "target_decisions": {"required": 1, "reviewed": 1, "not_reviewed": 0},
                    "notes": [
                        {
                            "text": "Source inspection found compatible callers.",
                            "files": ["api.py"],
                            "supporting_files": [],
                            "origin": {
                                "scope_id": "python",
                                "stage": "first_pass",
                                "sequence": 1,
                                "judgment_origin": "reviewer",
                                "analyzed_revision": coverage.revision.to_dict(),
                            },
                        }
                    ],
                    "notes_omitted": 0,
                }
                sidecar = {key: value for key, value in artifact.items() if key != "issues"}
                sidecar["review_context"] = context
                if context_kind == "unattributed":
                    context["notes"][0]["origin"] = None
                elif context_kind == "counts":
                    context["target_decisions"]["not_reviewed"] = 1
                elif context_kind == "notes":
                    context["notes"] *= 9
                elif context_kind == "secret":
                    context["notes"][0]["text"] = "ghp_1234567890"
                elif context_kind == "origin":
                    context["notes"][0]["origin"] = {"scope_id": "python"}
                elif context_kind == "scope":
                    sidecar["scope_id"] = "foreign"
                elif context_kind == "revision":
                    sidecar["analyzed_revision"] = {**coverage.revision.to_dict(), "head_sha": "foreign"}
                elif context_kind == "run":
                    sidecar["originating_run_id"] = "foreign"
                records_path = tmp_path / "stack-python-records.json"
                records_path.write_text(json.dumps(artifact))
                if context_kind == "nested-checks":
                    (tmp_path / "test-verdict.json").write_text("[" * 2000 + "]" * 2000)
                elif context_kind == "nested-sidecar":
                    context["notes"] = "nested_placeholder"
                    raw = json.dumps(sidecar).replace('"nested_placeholder"', "[" * 30000 + "0" + "]" * 30000)
                    assert len(raw.encode()) < 65536
                    (tmp_path / "stack-python-report-context.json").write_text(raw)
                elif context_kind != "missing":
                    (tmp_path / "stack-python-report-context.json").write_text(json.dumps(sidecar))
                collected = collect_report_context(tmp_path, coverage.to_dict())
                captured = collected["scopes"][0]
                if context_kind == "nested-checks":
                    assert collected["verification"] == {"status": "failed", "value": None}
                assert captured["originating_run_id"] == coverage.run_id
                assert captured["analyzed_revision"] == coverage.revision.to_dict()
                assert captured["review_context"] == (context if context_kind in {"current", "unattributed"} else None)
                assert json.loads(records_path.read_text()) == artifact
                assert valid_record_artifact(artifact, scope_id="python", analyzed_revision=coverage.revision.to_dict())
                assert not valid_record_artifact(
                    {**artifact, "review_context": context},
                    scope_id="python",
                    analyzed_revision=coverage.revision.to_dict(),
                )


def test_report_collection_admits_only_valid_author_context(
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    """Retained optional prose is shape-checked before it can enter rendering or a canonical handoff."""
    table_root = tmp_path
    row_index = 0
    cases: list[str] = [
        "current",
        "foreign-head",
        "unknown-head",
        "redacted",
        "null",
        "list",
        "missing-field",
        "extra-field",
        "source",
        "body-type",
        "body-length",
        "head-type",
        "head-length",
        "truncated-type",
        "nested",
        "file-size",
    ]
    for author_kind in cases:
        row_index += 1
        with subtests.test(author_kind=author_kind):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                from daydream.deep.report_context import collect_report_context
                from daydream.review_reporting import render_review_context

                coverage = review_coverage()
                record = records_artifact(coverage, "python")
                records_path = tmp_path / "stack-python-records.json"
                records_path.write_text(json.dumps(record))
                author: Any = {
                    "source": "pull_request_body",
                    "head_sha": coverage.revision.head_sha,
                    "body": "Retained author assertion",
                    "truncated": False,
                }
                if author_kind == "foreign-head":
                    author["head_sha"] = "f" * 40
                elif author_kind == "unknown-head":
                    author["head_sha"] = None
                elif author_kind == "redacted":
                    author["body"] = 'Retained author assertion {"token":"private-value"}'
                elif author_kind == "null":
                    author = None
                elif author_kind == "list":
                    author = []
                elif author_kind == "missing-field":
                    author.pop("truncated")
                elif author_kind == "extra-field":
                    author["invented"] = True
                elif author_kind == "source":
                    author["source"] = "host_execution"
                elif author_kind == "body-type":
                    author["body"] = []
                elif author_kind == "body-length":
                    author["body"] = "x" * 4097
                elif author_kind == "head-type":
                    author["head_sha"] = 7
                elif author_kind == "head-length":
                    author["head_sha"] = "h" * 513
                elif author_kind == "truncated-type":
                    author["truncated"] = "false"
                raw = (
                    "[" * 30000 + "0" + "]" * 30000
                    if author_kind == "nested"
                    else " " * 65536 + "null"
                    if author_kind == "file-size"
                    else json.dumps(author)
                )
                author_path = tmp_path / "review-author-context.json"
                author_path.write_text(raw)
                collected = collect_report_context(tmp_path, coverage.to_dict())
                retained = collected["author_reported"]
                valid = author_kind in {"current", "foreign-head", "unknown-head", "redacted"}
                if valid:
                    assert retained["source"] == "pull_request_body" and retained["head_sha"] == author["head_sha"]
                    assert retained["truncated"] is False and "Retained author assertion" in retained["body"]
                    if author_kind == "redacted":
                        assert "private-value" not in retained["body"] and retained["body"] != author["body"]
                    else:
                        assert retained == author
                    report = render_review_context(collected)
                    assert (
                        "f" * 40
                        if author_kind == "foreign-head"
                        else "unavailable"
                        if author_kind == "unknown-head"
                        else author["head_sha"]
                    ) in report
                else:
                    assert retained is None
                    assert "Author-reported check context unavailable." in render_review_context(collected)
                assert collected["scopes"][0]["originating_run_id"] == coverage.run_id
                assert collected["scopes"][0]["analyzed_revision"] == coverage.revision.to_dict()
                assert json.loads(records_path.read_text()) == record and author_path.read_text() == raw
                assert collected["verification"] == {"status": "unproduced", "value": None}


def test_report_collection_bounds_existing_artifacts_and_prioritizes_test_identity(
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    """Bound aggregate handoff size while explicitly accounting for omitted existing evidence."""
    table_root = tmp_path
    row_index = 0
    cases: list[str] = ["padding", "nested-arrays"]
    for shape in cases:
        row_index += 1
        with subtests.test(shape=shape):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
                from daydream.deep.report_context import collect_report_context
                from daydream.findings import MAX_ARTIFACT_BYTES, write_findings_artifact
                from daydream.review_reporting import render_review_context
                from tests.harness.review_result import findings_artifact

                coverage = review_coverage(scope_ids=[f"python-{index}" for index in range(32)])
                for scope in coverage.planned_scopes:
                    artifact = records_artifact(coverage, scope.scope_id)
                    (tmp_path / f"stack-{scope.scope_id}-records.json").write_text(json.dumps(artifact))
                    (tmp_path / f"stack-{scope.scope_id}-report-context.json").write_text(
                        json.dumps(
                            {
                                **{key: value for key, value in artifact.items() if key != "issues"},
                                "review_context": {
                                    "admitted_stages": 8,
                                    "target_decisions": {"required": 8, "reviewed": 8, "not_reviewed": 0},
                                    "notes": [{"text": "x" * 512, "files": [], "supporting_files": [], "origin": None}]
                                    * 8,
                                    "notes_omitted": 0,
                                },
                            }
                        )
                    )
                names = (
                    "test-verdict.json",
                    "remote-ci-verdict.json",
                    "fix-outcomes.json",
                    "fix-failures.json",
                    "fix-quality-gate.json",
                    "recommended-capture.json",
                )
                for name in names:
                    payload = (
                        {"existing_evidence": name, "padding": "x" * 60_000}
                        if shape == "padding"
                        else {"paths": [["x"]] * 10_000}
                    )
                    (tmp_path / name).write_text(json.dumps(payload, separators=(",", ":")))
                captured = collect_report_context(tmp_path, coverage.to_dict())
                assert 0 < len(captured["scopes"]) < 32
                assert captured["scopes_omitted"] == 32 - len(captured["scopes"])
                assert len(json.dumps(captured).encode()) < 400_000
                verification = captured["verification"]
                assert verification["status"] == "failed" and verification["omitted_artifacts"]
                assert set(verification["omitted_artifacts"]) | set(verification["value"]) >= set(names)
                if shape == "padding":
                    assert {"test-verdict.json", "remote-ci-verdict.json"} <= verification["value"].keys()
                else:
                    assert set(verification["omitted_artifacts"]) >= set(names)
                output = tmp_path / "findings.json"
                write_findings_artifact(
                    output, findings_artifact(review_context=captured, terminal_result=coverage.finalize("completed"))
                )
                assert output.stat().st_size < MAX_ARTIFACT_BYTES
                report = render_review_context(captured)
                assert "omitted due to the report evidence budget" in report
