"""Exercise the packaged native output extension with the installed Pi CLI."""
from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from daydream import committed_review
from daydream.agent import run_agent
from daydream.backends import (
    DiagnosticEvent,
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends._transport import CliTransport, teardown
from daydream.backends.pi import PiBackend, PiError
from daydream.committed_review import CommittedReviewRequest, run_committed_review
from daydream.committed_review_model import validate_committed_result
from daydream.host_tools import HostTool, HostToolBinding, HostToolFatal, HostToolRejected, NativeCapacityRequirement
from daydream.host_tools_server import PiHostToolServer
from daydream.review_budget import BudgetLedger
from daydream.review_source_refs import (
    canonicalize_source_selectors,
    part_source_mapping,
)
from daydream.run_config import RunConfig
from daydream.trajectory import DaydreamPhase
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.harbor_review import assignment_from_request, looked_up_targets
from tests.harness.native_pi import _NativeProvider, native_provider_server
from tests.harness.trajectory import make_recorder
from tests.test_committed_review import _adaptive_repo
from tests.test_deep_orchestrator import _profile_with_pipeline

# Declared loopback model limits must fit committed review's native-capacity preflight.
FIXTURE_CONTEXT_WINDOW, FIXTURE_MAX_TOKENS = 131072, 16384
HOSTED_TOOLS = {"read", "grep", "find", "ls", "review_source", "review_admit", "structured_output"}
_SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"verdict": {"type": "string"}, "findings": {"type": "array"}}, "required": ["verdict", "findings"]}


@pytest.fixture
def native_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_NativeProvider]:
    """Real installed Pi talks only to a counted, scripted loopback provider."""
    if shutil.which("pi") is None or shutil.which("node") is None:
        pytest.fail("The installed official Pi CLI and Node are required for native tool loading.")
    config = tmp_path / "isolated-host-pi"
    config.mkdir()
    (config / "settings.json").write_text(json.dumps({"retry": {"enabled": False, "maxRetries": 0}}))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(config))
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    for name in tuple(os.environ):
        if name.endswith(("_API_KEY", "_TOKEN", "_TOKEN_FILE")) or name.startswith(("AWS_", "AZURE_", "GOOGLE_")):
            monkeypatch.delenv(name, raising=False)
    fixture = _NativeProvider(config)

    with native_provider_server(fixture) as base_url:
        (config / "models.json").write_text(json.dumps({"providers": {"output-fixture": {
            "baseUrl": f"{base_url}/v1", "api": "openai-completions", "apiKey": "synthetic-loopback-only",
            "models": [{"id": "output-model", "reasoning": False, "input": ["text"],
                        "contextWindow": FIXTURE_CONTEXT_WINDOW, "maxTokens": FIXTURE_MAX_TOKENS}]}}}))
        monkeypatch.setenv("PI_PROVIDER", "output-fixture")
        yield fixture


@pytest.mark.parametrize("case", ["local", "hosted", "hosted-unsupported-node", "hosted-resources",
                                  "hosted-config-link", "hosted-model-fallback", "hosted-settings-encoding",
                                  "hosted-loader-override"])
async def test_committed_review_native_challenge_uses_fresh_captured_source_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, case: str,
) -> None:
    """Real Pi reviews frozen source with a fresh challenge and exact native selection.

    The hosted row retains operator policy across a late provider retry, which repeats the
    assignment within its original allocation, and excludes dirty working bytes; the local
    row proves stock reads/searches and supporting selectors share final working contents
    after original-checkout mutations. The hosted-* readiness rows reject versions,
    model/settings faults and executable resources before provider work.
    """
    local = case == "local"
    hosted = not local
    retried = case == "hosted"
    repo = tmp_path / "committed-source"
    original = "def ratio(total, count):\n    return total / count\n"
    captured = "def ratio(total, count):\n    return total / (count - 1)\n"
    helper = "VALUE = 'LOCAL_CAPTURE_HELPER'\n"
    untracked = "VALUE = 'LOCAL_CAPTURE_UNTRACKED'\n"
    supporting = {"supporting.py": helper} if local else {}
    context = "Review the committed ratio implementation carefully.\n"
    head = seed_feature_branch(repo, base={"ratio.py": original, "AGENTS.md": context, **supporting},
        feature={"ratio.py": original if local else captured, "AGENTS.md": context,
                 **({"supporting.py": "VALUE = 'HEAD_HELPER_MUST_REMAIN_UNREVIEWED'\n"} if local else {})})
    base = git(repo, "rev-parse", "main")
    working = captured if local else "def ratio(total, count):\n    return 'DIRTY_SOURCE_MUST_REMAIN_UNREVIEWED'\n"
    (repo / "ratio.py").write_text(working)
    if local:
        (repo / "supporting.py").write_text(helper)
        (repo / "untracked.py").write_text(untracked)
    else:
        (repo / "AGENTS.md").write_text("DIRTY_CONTEXT_MUST_REMAIN_UNREVIEWED\n")
    settings = native_provider.config / "settings.json"
    expected_model, expected_provider = "output-model", "output-fixture"
    settings.write_text(json.dumps({
        "defaultProvider": "output-fixture", "defaultModel": "output-model", "defaultThinkingLevel": "low",
        "defaultProjectTrust": "always", "modelThinkingLevels": {"output-fixture/output-model": "medium"},
        "retry": {"enabled": False, "maxRetries": 0},
    }))
    # The trusted project's thinking level wins locally; hosted rows overwrite it with ignored hostile settings.
    (repo / ".git/info/exclude").write_text(".pi/\n")
    project_settings = repo / ".pi/settings.json"
    project_settings.parent.mkdir()
    project_settings.write_text(json.dumps({"modelThinkingLevels": {"output-fixture/output-model": "low"}}))
    models_path = native_provider.config / "models.json"
    models = json.loads(models_path.read_text())
    models["providers"]["output-fixture"]["models"][0]["reasoning"] = True
    model = models["providers"]["output-fixture"]["models"][0]
    models_path.write_text(json.dumps(models))
    hosted_policy = None
    output_root = repo
    forbidden_marker = tmp_path / "forbidden-native-effect"
    if hosted:
        from daydream.hosted_review import HostedReviewPolicy
        from daydream.review_profile import ResolvedProfile, build_default_profile

        (repo / ".review-output.md").write_text("prior public report\n")
        (repo / ".daydream").mkdir()
        (repo / ".daydream/diff.patch").write_text("prior public artifact\n")
        (repo / ".daydream.toml").write_text('backend = "claude"\nmodel = "target-forbidden-model"\n'
            'review_profile = "hostile-missing-profile.toml"\n[phases.arbiter]\nbackend = "codex"\n'
            'model = "phase-forbidden-model"\n')
        # The model-fallback row requests a model the operator configuration does not serve.
        served = "deepseek/deepseek-v4.1-flash-1017"
        expected_provider = "nous"
        expected_model = "deepseek/deepseek-v4.1-flash" if case == "hosted-model-fallback" else served
        models_path.write_text(json.dumps({"providers": {expected_provider: {
            **models["providers"]["output-fixture"], "apiKey": "$NOUS_API_KEY", "authHeader": True,
            "models": [{**model, "id": served}],
        }}}))
        settings.write_text(json.dumps({"defaultProvider": "operator-unselected-provider",
            "defaultModel": "operator-unselected-model", "defaultProjectTrust": "always",
            "retry": {"enabled": False, "maxRetries": 0}}))
        (native_provider.config / "trust.json").write_text(json.dumps({
            str(native_provider.config.resolve()): True, str(repo.resolve()): True}))
        operator_environment = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "operator-home"),
                                "NOUS_API_KEY": "synthetic-hosted-credential", "DAYDREAM_PI_RETRY_ATTEMPTS": "1"}
        if retried:
            # The final discovery generation fails after all seven investigation starts.
            native_provider.errors[3] = "503 Service Unavailable"
            monkeypatch.setattr("daydream.agent_retry._sample_retry_delay", lambda cap: 0.0)
        elif case == "hosted-config-link":
            target_auth = repo / ".pi/auth.json"
            target_auth.write_text("{}")
            (native_provider.config / "auth.json").symlink_to(target_auth)
        elif case == "hosted-settings-encoding":
            settings.write_bytes(b"\xff")
        elif case == "hosted-loader-override":
            operator_environment["NODE_OPTIONS"] = f"--import={tmp_path / 'ambient-pi/preload.mjs'}"
        elif case == "hosted-unsupported-node":
            executable = shutil.which("node")
            assert executable is not None
            shim_dir = tmp_path / "unsupported-bin"
            shim_dir.mkdir()
            shim = shim_dir / "node"
            shim.write_text(f"#!{sys.executable}\nimport os, sys\n"
                "if sys.argv[1:] == ['--version']:\n print('v23.0.0')\n"
                f"else:\n os.execv({executable!r}, [{executable!r}, *sys.argv[1:]])\n")
            shim.chmod(0o700)
            operator_environment["PATH"] = f"{shim_dir}{os.pathsep}{operator_environment['PATH']}"
        output_root = tmp_path / "hosted-output"
        profile = build_default_profile()
        strategies = dict(profile.strategies)
        for key, marker in (("discovery.generic_fallback", "HOSTED_OPERATOR_DISCOVERY_POLICY"),
                            ("arbitration", "HOSTED_OPERATOR_CHALLENGE_POLICY")):
            strategies[key] = replace(strategies[key], content=marker)
        operator_profile = ResolvedProfile(profile=replace(
            profile, name="hosted-native-operator", strategies=strategies,
            pipeline=replace(profile.pipeline, reviewer_investigation_budget_s=60, review_wall_budget_s=170,
                             reviewer_tool_call_budget=30)),
            source_kind="operator")
        hosted_policy = HostedReviewPolicy(
            agent_dir=native_provider.config, environment=operator_environment,
            output_dir=output_root, private_base=tmp_path / "hosted-private",
            profile=operator_profile, trusted_prepared_source=True,
            provider=expected_provider, model=expected_model,
        )
        # Both ancestor/global instruction files and ambient resources are hostile.
        (tmp_path / "AGENTS.md").write_text("HOSTILE_ANCESTOR_POLICY\n")
        ambient = tmp_path / "ambient-pi"
        (ambient / "extensions").mkdir(parents=True)
        extension = ("import { writeFileSync } from 'node:fs';\n"
                     f"writeFileSync({json.dumps(str(forbidden_marker))}, 'resource executed');\n"
                     "export default function () {}\n")
        (ambient / "extensions/hostile.ts").write_text(extension)
        (ambient / "AGENTS.md").write_text("HOSTILE_GLOBAL_POLICY\n")
        (ambient / "trust.json").write_text(json.dumps({str(repo.resolve()): True}))
        (ambient / "settings.json").write_text(json.dumps({
            "defaultProjectTrust": "always", "defaultProvider": "ambient-forbidden-provider",
            "defaultModel": "ambient-forbidden-model", "packages": [str(ambient / "package")],
        }))
        package = ambient / "package"
        package.mkdir()
        (package / "package.json").write_text(json.dumps({"name": "hostile-local-package", "version": "1.0.0",
                                                         "pi": {"extensions": ["extension.ts"]}}))
        (package / "extension.ts").write_text(extension)
        if case == "hosted-resources":
            settings.write_text(json.dumps({"packages": [str(package)]}))
        preload = ambient / "preload.mjs"
        preload.write_text(extension.removesuffix("export default function () {}\n"))
        project_settings.write_text(json.dumps({"defaultProjectTrust": "always",
            "defaultProvider": "target-forbidden-provider", "defaultModel": "target-forbidden-model",
            "packages": [str(package)], "extensions": [str(package / "extension.ts")]}))
        monkeypatch.setenv("PI_CODING_AGENT_DIR", str(ambient))
        monkeypatch.setenv("PI_PROVIDER", "ambient-forbidden-provider")
        monkeypatch.setenv("PI_THINKING", "off")
        monkeypatch.setenv("NOUS_API_KEY", "ambient-forbidden-credential")
        monkeypatch.setenv("NODE_OPTIONS", f"--import={preload}")
    config_bytes = {path: path.read_bytes() for path in (settings, project_settings, models_path)}
    if local:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
        # Compare with the ordinary CLI, including its implicit default-provider authentication.
        control = await asyncio.to_thread(subprocess.run, [
            "pi", "--mode", "json", "--no-session", "--no-skills", "--no-extensions", "--no-tools",
            "--no-context-files", "-p", "Return the empty result."], cwd=repo, capture_output=True, timeout=30)
        assert control.returncode == 0, control.stderr.decode()
        assert len(native_provider.requests) == 1 and native_provider.requests[0]["model"] == expected_model
        assert native_provider.requests[0]["reasoning_effort"] == "low"
    native_provider.requests.clear()
    native_provider.authorizations.clear()
    assignments_seen: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}

    def choose(request: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        if hosted:
            systems = json.dumps([message for message in request["messages"]
                                  if message["role"] in {"system", "developer"}])
            assert context.strip() not in systems
            assert "HOSTILE_ANCESTOR_POLICY" not in systems and "HOSTILE_GLOBAL_POLICY" not in systems
            assert {tool["function"]["name"] for tool in request["tools"]} == HOSTED_TOOLS
        else:
            assert context.strip() in json.dumps(request["messages"])
        assert "DIRTY_CONTEXT_MUST_REMAIN_UNREVIEWED" not in json.dumps(request["messages"])
        assignment = assignment_from_request(request)
        kind = assignment["kind"]
        if hosted:
            assert assignment["profile"]["profile_name"] == operator_profile.name
            assert assignment["profile"]["profile_digest"] == operator_profile.digest
            assert assignment["profile"]["profile_source"] == "operator"
            assert assignment["profile"]["hosted_policy"]["provider"] == expected_provider
            assert assignment["profile"]["hosted_policy"]["model"] == expected_model
            assert assignment["profile"]["limits"]["effective_investigation_s"] == pytest.approx(50)
            marker = "HOSTED_OPERATOR_DISCOVERY_POLICY" if kind == "discovery" else "HOSTED_OPERATOR_CHALLENGE_POLICY"
            # Stable strategy policy survives compaction in system instructions, not the user turn.
            assert marker in json.dumps([m["content"] for m in request["messages"]
                                         if m["role"] in {"system", "developer"}])
            assert marker not in json.dumps([m["content"] for m in request["messages"] if m["role"] == "user"])
            assert assignment["profile"]["strategies"][kind] == marker
        revision = assignment["claims"][0]["revision"] if kind == "challenge" else 0
        call_kind = f"{kind}-{revision}" if hosted else kind
        tools = [message for message in request["messages"] if message["role"] == "tool"]
        # Targets and source handles come only from charged review_source lookups.
        lookups = [(f"{call_kind}-lookup-{index}", "review_source",
                    {"target_id": entry["target_id"], "path": None, "cursor": None})
                   for index, entry in enumerate(assignment["catalog"]["entries"])]
        if not tools:
            assignments_seen.append(assignment)
            histories[kind] = request["messages"]
            if hosted:
                assert not any(message["role"] in {"assistant", "tool"} for message in request["messages"])
                return [(f"{call_kind}-bash", "bash", {"command": f"touch {forbidden_marker}"}),
                        (f"{call_kind}-write", "write", {"path": str(forbidden_marker), "content": "forbidden"}),
                        (f"{call_kind}-edit", "edit", {"path": "ratio.py", "oldText": captured, "newText": "MUTATED"}),
                        (f"{call_kind}-ungranted", "publish_review", {"path": str(forbidden_marker)})]
            if kind == "discovery":
                for name in ("ratio.py", "supporting.py", "untracked.py", "late.py"):
                    (repo / name).write_text("VALUE = 'POST_CAPTURE_SOURCE_MUST_REMAIN_UNREVIEWED'\n")
            return [(f"{call_kind}-read", "read", {"path": "ratio.py"}), *lookups,
                    (f"{call_kind}-helper", "read", {"path": "supporting.py"}),
                    (f"{call_kind}-untracked", "read", {"path": "untracked.py"}),
                    (f"{call_kind}-grep", "grep", {"pattern": "LOCAL_CAPTURE", "path": "."}),
                    (f"{call_kind}-ls", "ls", {"path": "."})]
        if hosted and not any(tool["tool_call_id"] == f"{call_kind}-read" for tool in tools):
            assert len(tools) == 4
            assert all("not found" in tool["content"].lower() or "denied" in tool["content"].lower() for tool in tools)
            return [(f"{call_kind}-read", "read", {"path": "ratio.py"}), *lookups]
        read_result, = [tool for tool in tools if tool["tool_call_id"] == f"{call_kind}-read"]
        assert read_result["content"] == captured
        if local:
            observations = {tool["tool_call_id"]: tool["content"] for tool in tools}
            assert observations[f"{call_kind}-helper"] == helper
            assert observations[f"{call_kind}-untracked"] == untracked
            searched = observations[f"{call_kind}-grep"]
            assert "supporting.py:1:" in searched and helper.strip() in searched
            assert "untracked.py:1:" in searched and untracked.strip() in searched
            assert {entry for entry in observations[f"{call_kind}-ls"].splitlines() if entry.endswith(".py")} == {
                "ratio.py", "supporting.py", "untracked.py"}
            assert "POST_CAPTURE_SOURCE_MUST_REMAIN_UNREVIEWED" not in json.dumps(request["messages"])
        if hosted and not any(tool["tool_call_id"] == f"{call_kind}-forbidden-domain" for tool in tools):
            return [(f"{call_kind}-forbidden-domain", "structured_output", {"write_path": str(forbidden_marker)})]
        if hosted:
            rejection, = [tool for tool in tools if tool["tool_call_id"] == f"{call_kind}-forbidden-domain"]
            assert rejection["content"].startswith('Validation failed for tool "structured_output":')
        refs_by_target = {part["target_id"]: [{"source_id": next(
            option["source_id"] for option in part["source_options"] if option["side"] == "new"),
            "source_range": None, "basis": "supplied_diff"}] for part in looked_up_targets(request).values()}
        assert refs_by_target.keys() == {entry["target_id"] for entry in assignment["catalog"]["entries"]}
        ratio_part = next(part for part in looked_up_targets(request).values() if part["file"] == "ratio.py")
        refs = refs_by_target[ratio_part["target_id"]]
        if local:
            refs = [*refs, {"path": "supporting.py", "side": "new", "source_range": {"start": 1, "end": 1},
                           "basis": "source_inspection"}]
        if kind == "discovery":
            claim = {"uid": "", "revision": 0, "source_uids": [], "title": "Division can fail", "payload": {
                "claim": "The changed denominator divides by zero.", "trigger": "Call ratio(5, 1).",
                "consequence": "A normal input raises ZeroDivisionError.",
                "grounds": "Investigator is confident and considers its investigation complete.",
                "file": "ratio.py", "line": None, "evidence_refs": refs}}
            payload = {"targets": [{"target_id": target_id, "status": "reviewed", "reason": "Captured source checked.",
                                    "evidence_refs": target_refs} for target_id, target_refs in refs_by_target.items()],
                       "claims": [claim]}
        else:
            assert kind == "challenge" and len(assignment["claims"]) == 1
            claim = assignment["claims"][0]
            payload = {"verdicts": [{"uid": claim["uid"], "revision": claim["revision"], "verdict": "confirmed",
                "grounds": "At count=1 the captured denominator is zero.", "evidence_refs": refs}], "claims": []}
            if hosted and claim["revision"] == 1:
                payload["claims"] = [{key: claim[key] for key in ("uid", "revision", "source_uids", "title")} | {
                    "payload": {**claim["payload"], "grounds": "Captured denominator count-1 is zero at count=1.",
                                "evidence_refs": refs}}]
        return [(f"{call_kind}-submit", "structured_output", payload)]

    native_provider.choose = choose
    request = CommittedReviewRequest(
        repo, base, config=RunConfig(backend="codex" if hosted else "pi", output_mode="review", archive=False,
                                    run_eval=False, review_backend="claude" if hosted else None,
                                    review_model="request-phase-forbidden-model" if hosted else None,
                                    review_profile_path="hostile-missing-profile.toml" if hosted else None,
                                    model="request-forbidden-model" if hosted else None,
                                    review_profile=None if hosted else _profile_with_pipeline(
                                        reviewer_investigation_budget_s=90, reviewer_tool_call_budget=16)),
        hosted_policy=hosted_policy, local=local)
    if hosted and not retried:
        from daydream.hosted_review import UnsupportedHostedPolicy

        message = {
            "hosted-model-fallback": "Hosted Pi model selection",
            "hosted-unsupported-node": "node did not report v24[.]18[.]1",
            "hosted-loader-override": "Hosted Pi environment must not contain executable loader overrides",
        }.get(case, "Hosted|hosted")
        with pytest.raises(UnsupportedHostedPolicy, match=message):
            await run_committed_review(request)
        assert native_provider.requests == []
        assert not forbidden_marker.exists()
        assert (repo / ".review-output.md").read_text() == "prior public report\n"
        if case == "hosted-loader-override":
            assert not output_root.exists() and not (tmp_path / "hosted-private").exists()
        return
    result = await run_committed_review(request)
    assert result.complete and result.output_installed and result.capture_state == "complete"
    value = result.to_dict()
    validate_committed_result(value)
    assert value["analyzed_revision"]["head_sha"] == head
    assert value["analyzed_revision"]["merge_base_sha"] == base
    assert value["terminal_result"]["analysis_state"] == "complete"
    assert {part["file"] for part in value["assignments"]} == ({"ratio.py", "untracked.py"} if local else {"ratio.py"})
    expected_kinds = ["discovery", "challenge", *(["challenge"] if hosted else [])]
    if retried:
        # The late provider failure is retried in a fresh conversation with the identical assignment, and the
        # failed attempt's seven starts are refunded: the retry receives the original fifteen-call allocation.
        assert assignments_seen.pop(1) == assignments_seen[0]
        original, repeated = (json.dumps(native_provider.requests[index]["messages"]) for index in (0, 4))
        assert "tool-call allocation is 15 calls" in original and "tool-call allocation is 15 calls" in repeated
        assert ("Retry dispatch-time snapshot (supersedes the earlier invocation allocation): "
                "current invocation allocation: 15 tool calls.") in repeated
        # The refunded starts stay observed without overshoot; only the retry's eight are charged, so challenge
        # receives 30 - 8 and its re-dispatch only what the first challenge left.
        assert [(e["allocation"], e["observed_starts"], e["overshoot"]) for e in value["execution"]] == [
            (15, 15, 0), (22, 8, 0), (14, 8, 0)]
    assert [assignment["kind"] for assignment in assignments_seen] == expected_kinds
    assert len(native_provider.requests) == (4 if hosted else 2) * len(expected_kinds) + 4 * retried
    assert all(request["model"] == expected_model for request in native_provider.requests)
    if local:
        assert all(request["reasoning_effort"] == "low" for request in native_provider.requests)
    else:
        assert native_provider.authorizations == ["Bearer synthetic-hosted-credential"] * len(native_provider.requests)
        assert all({tool["function"]["name"] for tool in request["tools"]} == HOSTED_TOOLS
                   for request in native_provider.requests)
        assert not forbidden_marker.exists()
        assert value["settings"]["profile_name"] == operator_profile.name
        assert value["settings"]["profile_digest"] == operator_profile.digest
        assert value["settings"]["limits"]["effective_investigation_s"] == pytest.approx(50)
        assert value["settings"]["hosted_policy"]["provider"] == expected_provider
        assert value["settings"]["hosted_policy"]["model"] == expected_model
    assert not any(message["role"] in {"assistant", "tool"} for message in histories["challenge"])
    assert "discovery-read" not in json.dumps(histories["challenge"])
    assert assignments_seen[1]["captured_root"] == assignments_seen[0]["captured_root"]
    claim = value["claims"][0]
    assert claim["source_uids"] == [] and claim["revision"] == (2 if hosted else 1)
    assert claim["payload"]["line"] is None and claim["disposition"] == "confirmed"
    assert claim["challenge_evidence"][0]["revision"] == 1
    if hosted:
        assert [evidence["revision"] for evidence in claim["challenge_evidence"]] == [1, 2]
    assert len(value["findings"]) == 1
    if local:
        source_ref, = [ref for ref in claim["payload"]["evidence_refs"] if ref["path"] == "supporting.py"]
        assert source_ref["role"] == "supporting"
        assert source_ref["artifact_sha256"] == hashlib.sha256(helper.encode()).hexdigest()
    result_path = output_root / ("committed-review.json" if hosted else ".daydream/committed-review.json")
    assert json.loads(result_path.read_text()) == value
    report = (output_root / ("review-output.md" if hosted else ".review-output.md")).read_text()
    assert head in report and base in report and value["planned_work_id"] in report
    assert "Division can fail" in report and "Location: <code>ratio.py</code>." in report
    trajectory_path = (output_root / "trajectory.json" if hosted else
                       output_root / ".daydream/runs" / value["run_id"] / "trajectory.json")
    trajectory = json.loads(trajectory_path.read_text())
    calls = [call for step in trajectory["steps"] for call in step.get("tool_calls") or []]
    if local:
        assert [call["tool_call_id"] for call in calls] == [
            f"{kind}-{action}" for kind in expected_kinds
            for action in ("read", "lookup-0", "lookup-1", "helper", "untracked", "grep", "ls", "submit")]
    else:
        assert trajectory["extra"]["profile_name"] == operator_profile.name
        assert trajectory["extra"]["profile_digest"] == operator_profile.digest
        assert {"discovery-0-read", "challenge-1-read", "challenge-2-read"} <= {
            call["tool_call_id"] for call in calls}
        assert (repo / ".review-output.md").read_text() == "prior public report\n"
        assert (repo / ".daydream/diff.patch").read_text() == "prior public artifact\n"
        assert not (repo / ".daydream/committed-review.json").exists()
    invocations = trajectory["extra"]["subtrajectories"]
    expected_invocations = len(expected_kinds) + int(retried)
    assert len(invocations) == expected_invocations
    assert len({invocation["invocation_id"] for invocation in invocations}) == expected_invocations
    assert all(invocation["generation_lifecycle"]["drafts"] for invocation in invocations)
    generations = [generation for invocation in invocations
                   for generation in invocation["generation_lifecycle"]["drafts"]]
    assert all(generation["model_name"] == expected_model and generation["provider_name"] == expected_provider
               for generation in generations)
    assert all(execution["settled"] and execution["accepted"] and execution["cleanup_complete"]
               for execution in value["execution"])
    assert (repo / "ratio.py").read_text() == (
        "VALUE = 'POST_CAPTURE_SOURCE_MUST_REMAIN_UNREVIEWED'\n" if local else working)
    assert git(repo, "rev-parse", "HEAD") == head
    assert (repo / "AGENTS.md").read_text() == (context if local else "DIRTY_CONTEXT_MUST_REMAIN_UNREVIEWED\n")
    assert all(path.read_bytes() == original_bytes for path, original_bytes in config_bytes.items())


@pytest.mark.parametrize("fault", ["mixed-first", "mixed-last", "incomplete"])
async def test_native_terminal_batch_rejection_precedes_acceptance(
    tmp_path: Path, native_provider: _NativeProvider, fault: str,
) -> None:
    """Real provider batches cannot mutate review state before sole completed submission.

    Transcript replay can reject final output after mutation; only the installed
    extension proves the callback never sees the mixed or truncated generation.
    """
    (tmp_path / "source").write_text("Ordinary stock read\n")
    rejected = {"verdict": "REJECTED_TERMINAL_BATCH", "findings": []}
    accepted = {"verdict": "complete", "findings": []}
    terminal = ("rejected", "structured_output", rejected)
    read = ("read", "read", {"path": str((tmp_path / "source").resolve())})
    native_provider.turns = [[terminal] if fault == "incomplete" else (
        [read, terminal] if fault == "mixed-last" else [terminal, read]),
        [("accepted", "structured_output", accepted)]]
    if fault == "incomplete":
        native_provider.finish_reasons[0] = "length"
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(budget=BudgetLedger("native", deadline=time.monotonic() + 30, limit=5), accept=accept)
    result, _, reason = await run_agent(PiBackend(model="output-model"), tmp_path, "Submit the completed review.",
        phase=DaydreamPhase.REVIEW, output_schema=_SCHEMA, read_only=True, persist_session=False,
        host_tools=binding, progress_callback=lambda _: None)
    assert reason is None and result == accepted and installed == [accepted]
    assert len(native_provider.requests) == 2
    history = {message["tool_call_id"]: message["content"] for message in native_provider.requests[1]["messages"]
               if message["role"] == "tool"}
    assert "rejected" in history
    assert ("token limit" if fault == "incomplete" else "alone") in history["rejected"].lower()
    assert binding.accepted_result == accepted and binding.rejections >= 1


@pytest.mark.parametrize("lost_evidence", ["none", "fatal-details", "settlement"])
async def test_native_unavailable_host_input_stops_generation_before_retry_or_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, lost_evidence: str,
) -> None:
    """A real frozen-source failure ends /submit without correction or backend replay.

    Endpoint tests cannot prove that shipped extensions stop Pi's provider loop;
    admission exhaustion does not exercise /submit or authoritative classification
    when native error details or settlement disappear after the callback fails.
    """
    repo = tmp_path / "unavailable-source"
    head = seed_feature_branch(repo, base={"api.py": "VALUE = 0\n", "supporting.txt": "Frozen bytes\n"},
        feature={"api.py": "VALUE = 1\n", "supporting.txt": "Frozen bytes\n"})
    diff = git(repo, "diff", "main", "HEAD") + "\n"
    revision = {"head_sha": head, "merge_base_sha": git(repo, "rev-parse", "main")}
    assigned = part_source_mapping(diff, {"kind": "file", "file": "api.py"},
        artifact_sha256=hashlib.sha256(diff.encode()).hexdigest(), revision=revision,
        old_path="api.py", new_path="api.py")["source_refs"]
    selectors = [{"path": "supporting.txt", "side": "new", "source_range": {"start": 1, "end": 1},
                  "basis": "source_inspection"}]
    # Prove the submission and frozen reference are valid before the real Git
    # metadata disappears; selector errors cannot accidentally satisfy this test.
    assert canonicalize_source_selectors(selectors, assigned, revision=revision, source_root=repo,
                                        require_assigned=False)
    submitted = {"verdict": "complete", "findings": []}
    native_provider.turns = [[("failed-input", "structured_output", submitted)]]
    callback_calls = 0
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls == 1:
            (repo / ".git").rename(repo / ".git-unavailable")
        canonicalize_source_selectors(selectors, assigned, revision=revision, source_root=repo, require_assigned=False)
        return value
    binding = HostToolBinding(budget=BudgetLedger("native", deadline=time.monotonic() + 30, limit=4), accept=accept)
    native_rows: list[dict[str, Any]] = []
    lines = CliTransport.lines
    async def observe_native(transport: CliTransport, *args: Any, **kwargs: Any) -> Any:
        async for line in lines(transport, *args, **kwargs):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            native_rows.append(row)
            if lost_evidence == "settlement" and row.get("type") == "agent_settled":
                continue
            if lost_evidence == "fatal-details":
                if row.get("type") == "tool_execution_end" and row.get("toolCallId") == "failed-input":
                    row.get("result", {}).pop("details", None)
                elif row.get("type") == "message_end" and row.get("message", {}).get("role") == "toolResult":
                    row["message"].pop("details", None)
            yield json.dumps(row)
    monkeypatch.setattr(CliTransport, "lines", observe_native)
    backend = PiBackend(model="output-model")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = backend.retry_max_delay_s = 0
    recorder = make_recorder(tmp_path, backend_name="pi")
    async with recorder:
        with pytest.raises(PiError) as failure:
            await run_agent(backend, tmp_path, "Submit the frozen-source review.", phase=DaydreamPhase.REVIEW,
                host_tools=binding, output_schema=_SCHEMA, read_only=True, persist_session=False,
                progress_callback=lambda _: None)
    assert len(native_provider.requests) == 1, "fatal /submit must stop before any further provider request"
    assert callback_calls == 1 and binding.accepted_result is None
    assert failure.value.category == "HOST_TOOLS" and failure.value.retryable is False
    assert "supporting.txt" in str(failure.value) and "cannot" in str(failure.value).lower()
    assert not binding.authority_open and binding.cleanup_complete
    assert binding.native_settled is (lost_evidence != "settlement")
    errors = [row for row in native_rows if row.get("type") == "tool_execution_end"
              and row.get("toolCallId") == "failed-input" and row.get("isError") is True]
    assert len(errors) == 1 and "supporting.txt" in json.dumps(errors)
    trajectory = json.loads(recorder.path.read_text())
    diagnostics = [item for step in trajectory["steps"] for item in step.get("extra", {}).get("backend_diagnostics", [])
                   if item["code"] == "pi_host_tool_fatal"]
    assert len(diagnostics) == 1 and diagnostics[0]["metadata"]["code"] == "input_unavailable"
    assert diagnostics[0]["metadata"]["cause_type"] == "GitError"
    assert sum((step.get("metrics") or {}).get("prompt_tokens", 0) or 0 for step in trajectory["steps"]) == 10


@pytest.mark.parametrize("fault", ["missing-extension", "broken-extension", "unsupported-capability", "wrong-schema",
                                   "unreachable-endpoint", "unsupported-version"])
async def test_native_required_admission_readiness_fails_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, fault: str,
) -> None:
    """Native registration, active schema and endpoint handshake precede model work.

    A direct endpoint self-probe cannot catch Pi swallowing an extension load
    failure or a mismatched native definition and then dispatching a provider request.
    """
    root = tmp_path.resolve()
    binding = HostToolBinding(deadline=time.monotonic() + 30, accept=lambda value: value)
    reached: list[str] = []
    if fault in {"missing-extension", "broken-extension", "unsupported-capability"}:
        assets = root / "faulty-assets"
        assets.mkdir()
        packaged = files("daydream.backends")
        (assets / "pi_diagnostics.ts").write_text(packaged.joinpath("pi_diagnostics.ts").read_text())
        (assets / "pi_structured_output.ts").write_text(packaged.joinpath("pi_structured_output.ts").read_text())
        if fault == "broken-extension":
            (assets / "pi_host_tools.ts").write_text("this is an invalid TypeScript module {{")
        elif fault == "unsupported-capability":
            source = packaged.joinpath("pi_host_tools.ts").read_text()
            source = source.replace("export default function (pi: ExtensionAPI) {",
                "export default function (pi: ExtensionAPI) {\n  (pi as any).getAllTools = undefined;")
            (assets / "pi_host_tools.ts").write_text(source)
        monkeypatch.setattr("daydream.backends.pi.files", lambda _: assets)
    elif fault in {"wrong-schema", "unreachable-endpoint"}:
        packet = PiHostToolServer.packet
        def wrong_packet(server: PiHostToolServer) -> dict[str, Any]:
            reached.append(fault)
            value = packet(server)
            return {**value, **({"schema": {"type": "object", "properties": {"wrong": {"type": "string"}}}}
                if fault == "wrong-schema" else {"url": "http://127.0.0.1:1"})}
        monkeypatch.setattr(PiHostToolServer, "packet", wrong_packet)
    else:
        native_pi = shutil.which("pi")
        assert native_pi is not None
        shim = root / "bin"
        shim.mkdir()
        executable = shim / "pi"
        executable.write_text(f"#!{sys.executable}\nimport os, sys\n"
            "if sys.argv[1:] == ['--version']:\n print('1.0.4')\n"
            f"else:\n os.execv({native_pi!r}, [{native_pi!r}, *sys.argv[1:]])\n")
        executable.chmod(0o700)
        monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises((PiError, OSError)):
        async with asyncio.timeout(30):
            await run_agent(PiBackend(model="output-model"), root, "Require the bound terminal operation.",
                phase=DaydreamPhase.REVIEW, host_tools=binding, output_schema=_SCHEMA,
                persist_session=False, read_only=True, progress_callback=lambda _: None)
    if fault in {"wrong-schema", "unreachable-endpoint"}:
        assert reached
    assert native_provider.requests == [] and binding.calls == 0
    assert not binding.authority_open and not binding.native_settled and binding.cleanup_complete


@pytest.mark.parametrize("mode", ["tools-disabled", "finalization", "missing-output-schema", "unsupported-backend"])
async def test_required_terminal_admission_rejects_unsupported_dispatch_before_provider(
    tmp_path: Path, native_provider: _NativeProvider, mode: str,
) -> None:
    """A required acceptance operation cannot silently degrade into text-only work.

    Ordinary-mode tests have no domain callback whose omission would lose admission.
    """
    from tests.harness.backend import ScriptedBackend

    binding = HostToolBinding(deadline=time.monotonic() + 10, accept=lambda value: value)
    if mode == "finalization":
        with pytest.raises(PiError):
            _ = [event async for event in PiBackend(model="output-model").execute(
                tmp_path, "Serialize", finalization=True, host_tools=binding, output_schema=_SCHEMA)]
    else:
        backend = ScriptedBackend() if mode == "unsupported-backend" else PiBackend(model="output-model")
        options: dict[str, Any] = {"tools_disabled": True} if mode == "tools-disabled" else {}
        with pytest.raises((ValueError, NotImplementedError, PiError)):
            await run_agent(backend, tmp_path, "Require terminal acceptance.", phase=DaydreamPhase.REVIEW,
                host_tools=binding, output_schema=None if mode == "missing-output-schema" else _SCHEMA, **options)
        if isinstance(backend, ScriptedBackend):
            assert backend.calls == []
    assert native_provider.requests == [] and binding.calls == 0 and not binding.authority_open


async def test_native_reused_tool_identity_cannot_repeat_effect_or_hide_correlation_failure(
    tmp_path: Path, native_provider: _NativeProvider,
) -> None:
    """A repeated native call ID spends once and stops before a further generation.

    Unique-ID correction and idempotent host callback tests cannot detect a native
    adapter silently pairing two execution results with one original choice, or
    continuing after the host revokes authority for broken correlation.
    """
    source = tmp_path.resolve() / "source"
    source.write_text("first stock read only\n")
    repeated = ("same-native-id", "read", {"path": str(source)})
    accepted = {"verdict": "complete", "findings": []}
    native_provider.turns = [[repeated], [repeated], [("submit", "structured_output", accepted)]]
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 30, accept=accept)
    events = []
    with pytest.raises(PiError) as failure:
        async for event in PiBackend(model="output-model").execute(tmp_path, "Reject reused native identities.",
            output_schema=_SCHEMA, read_only=True, host_tools=binding, persist_session=False):
            events.append(event)
    assert len(native_provider.requests) == 2
    assert failure.value.category == "HOST_TOOLS" and failure.value.retryable is False
    assert "identity" in str(failure.value).lower()
    assert installed == [] and binding.accepted_result is None
    assert binding.calls == 1 and binding.cleanup_complete and binding.native_settled
    assert [event.id for event in events if isinstance(event, ToolStartEvent)] == ["same-native-id"]


_LOOKUP = {"type": "object", "additionalProperties": False, "properties": {"target_id": {"type": "string"}},
           "required": ["target_id"]}


@pytest.mark.parametrize("case", ["correctable", "oversized", "fatal", "capacity"])
async def test_native_host_lookup_is_registered_charged_and_served_with_exact_bytes(
    tmp_path: Path, native_provider: _NativeProvider, case: str,
) -> None:
    """Installed Pi exposes a host tool, serves its exact bytes once and refuses unfit models.

    Endpoint tests call /call directly; only the shipped extension proves Pi
    registers the declared schema, keeps frozen bytes out of JSON escaping,
    returns a correctable error to the model, stops after a fatal callback, and
    refuses native limits before the first provider request.
    """
    frozen = 'line "quoted"\n\ttabbed \\ bytes\n'
    served: list[Any] = []
    def lookup(arguments: dict[str, Any]) -> dict[str, Any]:
        served.append(arguments)
        if case == "correctable" and len(served) == 1:
            raise HostToolRejected("Unknown target_id; request the paged catalog first", code="unknown_target")
        if case == "fatal":
            raise HostToolFatal("Frozen review input is unavailable", code="input_unavailable")
        return {"target_id": arguments["target_id"], "content": frozen}
    accepted = {"verdict": "complete", "findings": []}
    # An oversized host-tool argument is a correctable payload fault, not an aborted invocation.
    first = {"target_id": "t" * 140_000} if case == "oversized" else {"target_id": "t1"}
    native_provider.turns = [[("lookup-1", "review_source", first)],
                             [("lookup-2", "review_source", {"target_id": "t1"})],
                             [("submit", "structured_output", accepted)]]
    binding = HostToolBinding(deadline=time.monotonic() + 30, accept=lambda value: value,
        host_tools={"review_source": HostTool(_LOOKUP, lookup)},
        # A largest tool result beyond the declared window cannot fit any invocation.
        native_requirement=NativeCapacityRequirement(400_000, 1_000) if case == "capacity" else None)
    events: list[Any] = []
    failure: PiError | None = None
    try:
        async for event in PiBackend(model="output-model").execute(tmp_path, "Look up the frozen target.",
            output_schema=_SCHEMA, read_only=True, host_tools=binding, tool_call_budget=6, persist_session=False):
            events.append(event)
    except PiError as exc:
        failure = exc
    request = next(event for event in events if isinstance(event, RequestEvent))
    assert isinstance(request.config, PiRequestConfig) and request.config.selected_tools_count == 6
    assert binding.prompt_bytes == len((request.system_prompt or "").encode()) + len(request.prompt.encode())
    if case == "capacity":
        assert failure is not None and failure.category == "HOST_TOOLS" and not failure.retryable
        assert "needs contextWindow >= " in str(failure)
        assert f"observed contextWindow {FIXTURE_CONTEXT_WINDOW}" in str(failure)
        assert any(isinstance(event, DiagnosticEvent) and event.code == "pi_host_tool_fatal"
                   and event.metadata["code"] == "native_capacity" for event in events)
        assert native_provider.requests == [] and served == [] and binding.calls == 0
        return
    capacity = next(event for event in events if isinstance(event, DiagnosticEvent)
                    and event.code == "pi_native_capacity")
    assert capacity.metadata["context_window"] == FIXTURE_CONTEXT_WINDOW
    assert capacity.metadata["max_tokens"] == FIXTURE_MAX_TOKENS
    assert capacity.metadata["reserve_tokens_source"] == "pi_1.4.0_default"
    tools = [tool["function"]["name"] for tool in native_provider.requests[0]["tools"]]
    assert sorted(tools) == ["find", "grep", "ls", "read", "review_source", "structured_output"]
    lookup_tool = next(tool["function"] for tool in native_provider.requests[0]["tools"]
                       if tool["function"]["name"] == "review_source")
    assert lookup_tool["parameters"] == _LOOKUP and "frozen captured review" in lookup_tool["description"]
    if case == "fatal":
        assert failure is not None and failure.category == "HOST_TOOLS" and not failure.retryable
        assert len(native_provider.requests) == 1 and binding.accepted_result is None
        assert not binding.authority_open and binding.cleanup_complete
        return
    assert failure is None and binding.accepted_result == accepted
    history = {message["tool_call_id"]: message["content"] for message in native_provider.requests[-1]["messages"]
               if message["role"] == "tool"}
    # The exact frozen bytes arrive unescaped, after the bounded metadata JSON block.
    assert frozen in history["lookup-2"] and '"capacity"' in history["lookup-2"]
    assert '"content"' not in history["lookup-2"]
    if case == "correctable":
        assert "Unknown target_id" in history["lookup-1"] and binding.rejections == 1
    if case == "oversized":
        assert "exceeds" in history["lookup-1"] and "split it into smaller batches" in history["lookup-1"]
        assert binding.fatal_error is None
    assert served == [{"target_id": "t1"}] * (2 if case == "correctable" else 1)
    assert binding.calls == 3


@pytest.mark.parametrize("cancel_at", ["active-read", "late-start"])
async def test_native_stock_read_cancellation_reaps_process_before_freeze_and_preserves_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, cancel_at: str,
) -> None:
    """Cancel real FIFO reads or late starts before freeze, preserving a cached-backend sibling."""
    root = tmp_path.resolve()
    blocked = root / "blocked-source"
    os.mkfifo(blocked)
    entered, release_spawn, authority_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    children: list[asyncio.subprocess.Process] = []
    fifo_writer: int | None = None
    spawn = asyncio.create_subprocess_exec
    async def delayed_spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        child = await spawn(*args, **kwargs)
        cancelled_invocation = any(isinstance(arg, str) and arg.startswith("@")
            and "cancelled-invocation" in Path(arg[1:]).read_text() for arg in args)
        if cancelled_invocation:
            children.append(child)
            entered.set()
            await release_spawn.wait()
        return child
    if cancel_at == "late-start":
        monkeypatch.setattr("daydream.backends._transport.asyncio.create_subprocess_exec", delayed_spawn)
    original = CliTransport.lines
    async def observe(transport: CliTransport, *args: Any, **kwargs: Any) -> Any:
        async for line in original(transport, *args, **kwargs):
            with suppress(json.JSONDecodeError):
                event = json.loads(line)
                if (cancel_at == "active-read" and event.get("type") == "tool_execution_start"
                        and event.get("toolCallId") == "blocked-read"):
                    children.extend(transport.processes)
                    entered.set()
            yield line
    monkeypatch.setattr(CliTransport, "lines", observe)
    survivor_waiting, release_survivor = threading.Event(), threading.Event()
    accepted = {"verdict": "complete", "findings": []}
    def choose(request: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        user = next(message for message in request["messages"] if message["role"] == "user")
        if "cancelled-invocation" in json.dumps(user):
            return [("blocked-read", "read", {"path": str(blocked)})]
        survivor_waiting.set()
        if not release_survivor.wait(10):
            raise RuntimeError("sibling provider release barrier was not reached")
        return [("sibling-submit", "structured_output", accepted)]
    native_provider.choose = choose
    backend = PiBackend(model="output-model")
    cancelled_results: list[Any] = []
    sibling_results: list[Any] = []
    def accept_cancelled(value: dict[str, Any]) -> dict[str, Any]:
        cancelled_results.append(value)
        return value
    def accept_sibling(value: dict[str, Any]) -> dict[str, Any]:
        sibling_results.append(value)
        return value
    cancelled_binding = HostToolBinding(deadline=time.monotonic() + 30, accept=accept_cancelled)
    close_attempt = cancelled_binding.close_attempt
    def observe_close() -> None:
        close_attempt()
        authority_closed.set()
    monkeypatch.setattr(cancelled_binding, "close_attempt", observe_close)
    sibling_binding = HostToolBinding(deadline=time.monotonic() + 30, accept=accept_sibling)
    cancelled_recorder = make_recorder(root, path=root / "cancelled.json", backend_name="pi")
    sibling_recorder = make_recorder(root, path=root / "sibling.json", backend_name="pi")
    async def invoke(binding: HostToolBinding, recorder: Any, prompt: str) -> Any:
        async with recorder:
            return await run_agent(backend, root, prompt, phase=DaydreamPhase.REVIEW, host_tools=binding,
                output_schema=_SCHEMA, read_only=True, persist_session=False, progress_callback=lambda _: None)
    cancelled = asyncio.create_task(invoke(cancelled_binding, cancelled_recorder, "cancelled-invocation"))
    sibling = asyncio.create_task(invoke(sibling_binding, sibling_recorder, "surviving-invocation"))
    try:
        await asyncio.wait_for(entered.wait(), 8)
        assert await asyncio.to_thread(survivor_waiting.wait, 5)
        # Keep the FIFO writer open without bytes so the real Node read cannot finish before cancellation.
        async with asyncio.timeout(5):
            while fifo_writer is None:
                try:
                    fifo_writer = os.open(blocked, os.O_WRONLY | os.O_NONBLOCK)
                except OSError as exc:
                    if exc.errno != errno.ENXIO:
                        raise
                    await asyncio.sleep(0.01)
        assert children[0].returncode is None
        cancelled.cancel()
        await asyncio.wait_for(authority_closed.wait(), 5)
        assert not cancelled_binding.authority_open
        release_spawn.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert len(children) == 1 and children[0].returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(children[0].pid, 0)
        frozen = cancelled_recorder.path.read_bytes()
        lifecycle = json.loads(frozen)["extra"]["subtrajectories"][0]["backend_lifecycle"]
        assert lifecycle["host_authority_closed"] and lifecycle["host_cleanup_complete"]
        assert not lifecycle["host_native_settled"] and cancelled_results == []
        assert not sibling.done() and sibling_binding.authority_open
        release_survivor.set()
        result, _, reason = await asyncio.wait_for(sibling, 8)
        assert reason is None and result == accepted and sibling_results == [accepted]
        assert cancelled_recorder.path.read_bytes() == frozen
        assert not cancelled_binding.authority_open and cancelled_binding.cleanup_complete
        assert sibling_binding.native_settled and sibling_binding.cleanup_complete
        assert cancelled_binding.calls == sibling_binding.calls == 1 and backend._transports == []
    finally:
        release_spawn.set()
        release_survivor.set()
        for task in (cancelled, sibling):
            if not task.done():
                task.cancel()
        await asyncio.gather(cancelled, sibling, return_exceptions=True)
        if fifo_writer is not None:
            os.close(fifo_writer)


@pytest.mark.parametrize("fault", ["disconnect", "uncorrelated-reply", "missing-callback"])
async def test_native_indeterminate_submission_stops_generation_and_never_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, fault: str,
) -> None:
    """Stop unconfirmed submissions, retaining accepted work and refusing replay.

    Missing callbacks prove fresh-conversation denial before acceptance; lost or uncorrelated
    receipts retain accepted work. The source-selector journey owns correction.
    """
    submitted = {"verdict": "submitted", "findings": []}
    canonical = {"verdict": "complete", "findings": []}
    native_provider.turns = [[("submit", "structured_output", submitted)]]
    installed: list[dict[str, Any]] = []
    callback_times: list[float] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        assert value == submitted
        callback_times.append(time.monotonic())
        binding.ensure_open()
        callback_times.append(time.monotonic())
        installed.append(canonical)
        return canonical
    binding = HostToolBinding(budget=BudgetLedger("native", deadline=time.monotonic() + 30, limit=4), accept=accept)
    dropped_submissions: list[bytes] = []
    if fault == "missing-callback":
        read_body = asyncio.StreamReader.readexactly
        async def lose_submit_body(reader: asyncio.StreamReader, size: int) -> bytes:
            body = await read_body(reader, size)
            if body and json.loads(body) == {"id": "submit", "arguments": submitted}:
                dropped_submissions.append(body)
                raise ConnectionResetError("Submission body read disconnected before callback dispatch")
            return body
        monkeypatch.setattr(asyncio.StreamReader, "readexactly", lose_submit_body)
    reply = PiHostToolServer._reply
    lost_replies: list[dict[str, Any]] = []
    async def lose_accepted_reply(server: PiHostToolServer, writer: asyncio.StreamWriter,
                                  status: int, value: dict[str, Any]) -> None:
        if value.get("accepted") is True:
            assert binding.accepted_result == canonical
            lost_replies.append(value)
            if fault == "disconnect":
                writer.close()
                await writer.wait_closed()
                return
            if fault == "uncorrelated-reply":
                value = {**value, "id": "foreign-submit"}
        await reply(server, writer, status, value)
    monkeypatch.setattr(PiHostToolServer, "_reply", lose_accepted_reply)
    native_rows: list[dict[str, Any]] = []
    lines = CliTransport.lines
    async def observe_native(transport: CliTransport, *args: Any, **kwargs: Any) -> Any:
        async for line in lines(transport, *args, **kwargs):
            with suppress(json.JSONDecodeError):
                native_rows.append(json.loads(line))
            yield line
    monkeypatch.setattr(CliTransport, "lines", observe_native)
    recorder = make_recorder(tmp_path, backend_name="pi")
    backend = PiBackend(model="output-model")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = backend.retry_max_delay_s = 0
    async with recorder:
        with pytest.raises(PiError) as failure:
            await run_agent(backend, tmp_path, "Submit the canonical review.", phase=DaydreamPhase.REVIEW,
                host_tools=binding, output_schema=_SCHEMA, read_only=True, persist_session=False,
                progress_callback=lambda _: None)
    if fault == "missing-callback":
        assert dropped_submissions and not installed and not callback_times and not lost_replies
        assert binding.accepted_result is None and not binding.possibly_accepted
    else:
        assert installed == [canonical] and binding.accepted_result == canonical and lost_replies
        assert len(callback_times) == 2 and callback_times[-1] < binding.deadline
    assert len(native_provider.requests) == 1, "indeterminate acceptance must stop further model generations"
    assert failure.value.category == "HOST_TOOLS" and failure.value.retryable is False
    assert binding.calls == 1 and binding.rejections == 0 and binding.cleanup_complete
    assert not binding.authority_open
    settlements = [row for row in native_rows if row.get("type") == "agent_settled"]
    assert len(settlements) == 1 and binding.native_settled
    errors = [row for row in native_rows if row.get("type") == "tool_execution_end"
              and row.get("toolCallId") == "submit" and row.get("isError") is True]
    assert errors and "indeterminate" in json.dumps(errors)
    trajectory = json.loads(recorder.path.read_text())
    lifecycle = trajectory["extra"]["subtrajectories"][0]["backend_lifecycle"]
    assert lifecycle["host_authority_closed"] and lifecycle["host_native_settled"]
    assert lifecycle["host_cleanup_complete"] and lifecycle["host_tool_call_count"] == 1
    calls = [call for step in trajectory["steps"] for call in step.get("tool_calls") or []]
    observations = {item["source_call_id"]: item["content"] for step in trajectory["steps"]
                    for item in (step.get("observation") or {}).get("results", [])}
    assert [call["tool_call_id"] for call in calls] == ["submit"] and "indeterminate" in observations["submit"]
    assert sum((step.get("metrics") or {}).get("prompt_tokens", 0) or 0 for step in trajectory["steps"]) == 10
    if fault == "missing-callback":
        with pytest.raises(PiError) as subsequent:
            await run_agent(backend, tmp_path, "Retry the same assignment in a new conversation.",
                phase=DaydreamPhase.REVIEW, host_tools=binding, output_schema=_SCHEMA, read_only=True,
                persist_session=False, progress_callback=lambda _: None)
        assert len(native_provider.requests) == 1 and not installed
        assert subsequent.value.category == "HOST_TOOLS" and not subsequent.value.retryable


async def test_native_pi_budget_warning_preserves_wire_roles_history_and_charged_calls(
    tmp_path: Path, native_provider: _NativeProvider,
) -> None:
    """When only submission capacity remains, the warning and recovery are system notes, never user turns.

    Actual wire requests catch custom notes becoming users or dropping genuine history, while failed
    and invalid batch calls stay charged across the threshold.
    """
    source = 'Captured source\n'
    (tmp_path / 'source.txt').write_text(source)
    prompt = 'Investigate the supplied source and submit the result.'
    assignment = {'objective': 'first_pass', 'target_ids': ['target-source'], 'candidate_ids': ['candidate-source']}
    policy = 'Stable original system policy: preserve this exact instruction.'
    instructions = policy + '\nDaydream final-submission assignment: ' + json.dumps(assignment)
    requests = native_provider.requests
    output = {'verdict': 'complete', 'findings': []}
    batches: list[list[tuple[str, str, dict[str, Any]]]] = [
        [('read-a', 'read', {'path': 'source.txt'}), ('read-missing', 'read', {'path': 'missing.txt'}),
         ('read-b', 'read', {'path': 'source.txt'})],
        [('failed-submit', 'structured_output', {'verdict': 'incomplete', 'findings': False})],
        [('read-c', 'read', {'path': 'source.txt'}), ('read-d', 'read', {'path': 'source.txt'})],
        [],  # A prose-only turn at the threshold receives recovery guidance.
        [('final-submit', 'structured_output', output)],
    ]

    def choose(request: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        index = len(requests) - 1
        native_provider.text = f'Original assistant reasoning summary {index}.'
        return batches[index]

    native_provider.choose = choose
    events = [event async for event in PiBackend(model='output-model').execute(
        tmp_path, prompt, output_schema=_SCHEMA, review_instructions=instructions,
        read_only=True, persist_session=False, tool_call_budget=7)]
    assert len(requests) == len(batches)
    recovery = 'Submit the final result now by calling structured_output alone. Assistant prose is not a submission.'
    warning_requests: list[int] = []
    for index, request in enumerate(requests):
        messages = request['messages']
        system = [message for message in messages if message['role'] in {'system', 'developer'}]
        assert system and messages[0] == system[0] and policy in json.dumps(system)
        users = [message for message in messages if message['role'] == 'user']
        assert len(users) == 1 and prompt in json.dumps(users[0])
        assert 'Final-submission budget warning' not in json.dumps(users) and recovery not in json.dumps(users)
        assistants = [message for message in messages if message['role'] == 'assistant']
        assert [message['content'] for message in assistants] == [
            f'Original assistant reasoning summary {earlier}.' for earlier in range(index)]
        warnings = [message for message in messages if 'Final-submission budget warning:' in json.dumps(message)]
        if warnings:
            warning_requests.append(index)
            assert len(warnings) == 1 and warnings[0]['role'] in {'system', 'developer'}
            assert 'Only submission capacity remains' in warnings[0]['content']
            assignment_text = warnings[0]['content'].split('Assigned objective and identities: ', 1)[1]
            assert json.JSONDecoder().raw_decode(assignment_text)[0] == assignment
        recovery_messages = [message for message in messages if recovery in json.dumps(message)]
        assert len(recovery_messages) == (1 if index == 4 else 0)
        assert all(message['role'] in {'system', 'developer'} for message in recovery_messages)
    assert warning_requests == [3]
    starts = [event.id for event in events if isinstance(event, ToolStartEvent)]
    assert starts == [call_id for batch in batches for call_id, _, _ in batch]
    failures = [event.id for event in events if isinstance(event, ToolResultEvent) and event.is_error]
    assert failures == ['read-missing', 'failed-submit']
    result = next(event for event in reversed(events) if isinstance(event, ResultEvent))
    assert result.structured_output == output


@pytest.mark.parametrize(("allowance", "malformation"), [(1, "schema"), (2, "argument-size")],
                         ids=["spent-by-native-schema-rejection", "argument-size-correction"])
async def test_native_schema_rejection_charges_host_before_corrected_acceptance(
    tmp_path: Path, native_provider: _NativeProvider, allowance: int, malformation: str,
) -> None:
    """A native schema failure spends host allowance before a corrected callback can mutate.

    Unbound warning tests and direct HTTP schema checks cannot catch charging
    after Pi validation, a misleading null-object error after native argument-size
    clipping, or a provider request after the last call fails. Direct execute also
    excludes run_agent's later observer.
    """
    accepted = {"verdict": "complete", "findings": []}
    invalid = ({"verdict": "incomplete", "findings": False} if malformation == "schema" else
               {"verdict": "x" * (128 * 1024), "findings": []})
    native_provider.turns = [[("invalid-schema", "structured_output", invalid)],
                             [("corrected", "structured_output", accepted)]]
    marker = tmp_path / "accepted-review"
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        marker.write_text(value["verdict"])
        installed.append(value)
        return value
    # The binding's ledger is the authority; execute's tool_call_budget only renders the preamble.
    binding = HostToolBinding(budget=BudgetLedger("native", deadline=time.monotonic() + 30, limit=allowance),
                              accept=accept)
    events = []
    failure: PiError | None = None
    try:
        async for event in PiBackend(model="output-model").execute(tmp_path, "Correct the native schema error.",
            output_schema=_SCHEMA, read_only=True, host_tools=binding, tool_call_budget=allowance,
            persist_session=False):
            events.append(event)
    except PiError as exc:
        failure = exc
    assert (failure is None) is (allowance == 2)
    if failure is not None:
        assert failure.category == "HOST_TOOLS" and failure.retryable is False
        assert "allocation exhausted" in str(failure).lower() and "cannot" in str(failure).lower()
    assert len(native_provider.requests) == allowance, "fatal admission must not dispatch another provider request"
    assert installed == ([accepted] if allowance == 2 else [])
    assert marker.exists() is (allowance == 2)
    if allowance == 2:
        assert marker.read_text() == "complete" and binding.accepted_result == accepted
        history = {message["tool_call_id"]: message["content"] for message in native_provider.requests[1]["messages"]
                   if message["role"] == "tool"}
        assert "argument object" in history["invalid-schema"] and "131072" in history["invalid-schema"]
        assert "reduce" in history["invalid-schema"].lower()
    else:
        assert binding.accepted_result is None
    assert binding.calls == allowance and binding.rejections == 1
    assert [event.id for event in events if isinstance(event, ToolStartEvent)] == (
        ["invalid-schema", "corrected"] if allowance == 2 else ["invalid-schema"])
    assert [event.id for event in events if isinstance(event, ToolResultEvent) and event.is_error] == (
        ["invalid-schema"])
    assert not binding.authority_open and binding.native_settled and binding.cleanup_complete


async def test_native_accepted_state_is_retained_when_native_teardown_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider,
) -> None:
    """A teardown failure after acceptance keeps the accepted result, never replays it, and reports cleanup open."""
    accepted = {"verdict": "complete", "findings": []}
    native_provider.turns = [[("accepted", "structured_output", accepted)]]
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 30, accept=accept)
    async def failing_teardown(transport: CliTransport, transports: list[CliTransport], **options: Any) -> None:
        await teardown(transport, transports, **options)
        if binding.accepted:
            raise RuntimeError("Native teardown failed after process cleanup")
    monkeypatch.setattr("daydream.backends.pi.teardown", failing_teardown)
    backend = PiBackend(model="output-model")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = backend.retry_max_delay_s = 0
    recorder = make_recorder(tmp_path, backend_name="pi")
    async with recorder:
        with pytest.raises(RuntimeError, match="Native teardown failed"):
            await run_agent(backend, tmp_path, "Submit the review.", phase=DaydreamPhase.REVIEW,
                host_tools=binding, output_schema=_SCHEMA, read_only=True, persist_session=False,
                progress_callback=lambda _: None)
    assert installed == [accepted] and binding.accepted_result == accepted
    assert len(native_provider.requests) == 1, "a possibly accepted mutation cannot be replayed"
    lifecycle = json.loads(recorder.path.read_text())["extra"]["subtrajectories"][0]["backend_lifecycle"]
    assert binding.native_settled and lifecycle["host_native_settled"]
    assert not binding.cleanup_complete and lifecycle["host_cleanup_complete"] is False


async def test_committed_navigation_survives_native_compaction_on_one_cumulative_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider,
) -> None:
    """Proof B: installed Pi compacts after 52 of 96 discovery starts; host authority is unchanged.

    Scripted journeys never reached native compaction, so dropped compaction events, a
    renewed allowance after summarization, navigation that lived only in summarized
    history, or unavailable selections erasing earlier admissions all went unnoticed.
    Investigation then continues toward exhaustion: a two-member batch after 95 of 96
    starts previously spent the protected submission start and killed the assignment;
    an unresolved cross-source concern is retained and re-challenged once, after another
    claim progressed, then stops rather than spending reserves on evidence-free repeats.
    The scripted model proves software behavior, not review quality.
    """
    repo = tmp_path / "source"
    files = {name: f"VALUE_{index} = {index}\n" for index, name in enumerate(("a.py", "b.py", "c.py", "d.py"))}
    head = seed_feature_branch(repo, base={name: "VALUE = 0\n" for name in files}, feature=files)
    base = git(repo, "rev-parse", "main")
    settings = native_provider.config / "settings.json"
    settings.write_text(json.dumps({"defaultProvider": "output-fixture", "defaultModel": "output-model",
                                    "defaultProjectTrust": "always", "retry": {"enabled": False, "maxRetries": 0},
                                    "compaction": {"enabled": True, "reserveTokens": 16384, "keepRecentTokens": 64}}))
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    bindings: list[HostToolBinding] = []
    real_open = committed_review.Review.open

    def open_work(self: Any, work_id: str, **kwargs: Any) -> Any:
        binding, schema = real_open(self, work_id, **kwargs)
        bindings.append(binding)
        return binding, schema

    monkeypatch.setattr(committed_review.Review, "open", open_work)
    state: dict[str, Any] = {"starts": 0, "remaining": [], "summaries": 0, "denied": 0}
    # The 52nd start's generation reports a context above contextWindow - reserveTokens.
    native_provider.usage[51] = {"prompt_tokens": 120000, "completion_tokens": 5, "total_tokens": 120005}

    def call(name: str, arguments: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        state["starts"] += 1
        return [(f"call-{state['starts']}", name, arguments)]

    def choose(request: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        text = json.dumps(request["messages"])
        if "context summarization assistant" in text:
            state["summaries"] += 1
            native_provider.text = "Summary: earlier lookups and one admitted batch."
            return []
        native_provider.text = None
        assignment = assignment_from_request(request)
        binding = bindings[-1]
        assert "Navigation:" in text and assignment["catalog"]["total_targets"] == 4
        if assignment["kind"] == "discovery":
            # One cumulative 96-start ledger: neither compaction nor summarization renews it.
            # Denied protected starts are observed, never charged against the allocation.
            assert binding.remaining_tool_calls == 96 - (state["starts"] - state["denied"])
        if assignment["kind"] == "challenge":
            if not any(m["role"] == "tool" for m in request["messages"]):
                return [("challenge-lookup", "review_source", {"target_id": state["a"], "path": None, "cursor": None})]
            # The cross-source concern stays unresolved: evidence-free repetition must stop, not spend reserves.
            return [("challenge-final", "structured_output", {"verdicts": [{
                "uid": claim["uid"], "revision": claim["revision"],
                **({"verdict": "confirmed", "grounds": "The reread frozen bytes confirm the changed value."}
                   if claim["title"] == "Changed value" else
                   {"verdict": "unresolved", "grounds": "No counterevidence settles the a.py/b.py interaction."}),
                "evidence_refs": state["refs"]["a.py"]} for claim in assignment["claims"]],
                "claims": []})]
        targets = {target["file"]: target for target in looked_up_targets(request).values()}
        starts = state["starts"]
        if state["summaries"]:
            # Before the next charged start the host ledger still reflects all 52 charged starts.
            state["remaining"].append(binding.remaining_tool_calls)
            if starts == 52:
                assert binding.remaining_tool_calls == 44
                assert state["a_bytes"] not in text  # Exact detail now lies outside retained context.
                return call("review_source", {"target_id": state["a"], "path": None, "cursor": None})
            if starts == 53:
                assert targets["a.py"]["target_id"] == state["a"]
                reread = next(m for m in request["messages"] if m["role"] == "tool"
                              and m["tool_call_id"] == "call-53")
                assert reread["content"].split("\n", 1)[1] == state["a_bytes"]
                return call("review_source", {"target_id": None, "path": "missing.py", "cursor": None})
            if starts == 54:
                unavailable = json.loads(next(m for m in request["messages"] if m.get("tool_call_id") == "call-54")
                                         ["content"].split("\n", 1)[0])
                assert unavailable["status"] == "unavailable" and unavailable["capacity"]["remaining_tool_calls"] == 42
                return call("review_admit", {"targets": [{"target_id": state["b"], "status": "reviewed", "reason": "",
                    "evidence_refs": [{"path": "missing.py", "side": "new", "source_range": {"start": 1, "end": 1},
                                       "basis": "source_inspection"}]}], "claims": []})
            if starts < 95:
                return call("read", {"path": "b.py"})
            if starts == 95:
                # A two-member investigation batch after 95 of 96 starts: both would consume the
                # protected structured_output start, so both are denied without closing authority.
                state["starts"], state["denied"] = 97, 2
                return [("call-96", "read", {"path": "c.py"}),
                        ("call-97", "review_source", {"target_id": state["c"], "path": None, "cursor": None})]
            assert starts == 97 and binding.authority_open and binding.denied_starts == 2
            denials = [m["content"] for m in request["messages"] if m.get("tool_call_id") in {"call-96", "call-97"}]
            assert len(denials) == 2 and all("reserved for structured_output" in denial for denial in denials)
            return call("structured_output", {"targets": [
                {"target_id": state[name[0]], "status": "reviewed", "reason": "", "evidence_refs": state["refs"][name]}
                for name in ("c.py", "d.py")], "claims": [{
                    "uid": "", "revision": 0, "source_uids": [], "title": "Cross-source value contract",
                    "payload": {"claim": "a.py and b.py disagree on the changed value contract.",
                                "trigger": "Import both modules.", "consequence": "Callers combine 0 and 1.",
                                "grounds": "Both frozen captured files change VALUE.", "file": "a.py", "line": None,
                                "evidence_refs": state["refs"]["a.py"] + state["refs"]["b.py"]}}]})
        if starts == 0:
            return call("review_source", {"target_id": None, "path": None, "cursor": None})
        if starts <= 4:
            return call("review_source", {"target_id": assignment["catalog"]["entries"][starts - 1]["target_id"],
                                          "path": None, "cursor": None})
        if starts == 5:
            state["refs"] = {name: [{"source_id": next(o["source_id"] for o in target["source_options"]
                                                       if o["side"] == "new"),
                                     "basis": "supplied_diff", "source_range": None}]
                          for name, target in targets.items()}
            for name in files:
                state[name[0]] = targets[name]["target_id"]
            state["a_bytes"] = next(m for m in request["messages"] if m.get("tool_call_id") == "call-2")[
                "content"].split("\n", 1)[1]
            return call("review_admit", {"targets": [{"target_id": state["a"], "status": "reviewed", "reason": "",
                "evidence_refs": state["refs"]["a.py"]}], "claims": [{"uid": "", "revision": 0, "source_uids": [],
                "title": "Changed value", "payload": {"claim": "VALUE_0 changes.", "trigger": "Import a.py.",
                    "consequence": "Callers see 0.", "grounds": "Frozen captured bytes.", "file": "a.py", "line": None,
                    "evidence_refs": state["refs"]["a.py"]}}]})
        return call("read", {"path": "a.py"})

    native_provider.choose = choose
    result = await run_committed_review(CommittedReviewRequest(
        repo, base, config=RunConfig(backend="pi", output_mode="review", archive=False, run_eval=False,
                                     review_profile=_profile_with_pipeline(reviewer_investigation_budget_s=240,
                                                                           reviewer_tool_call_budget=192))))
    value = result.to_dict()
    validate_committed_result(value)
    assert state["summaries"] == 1 and state["remaining"][:4] == [44, 43, 42, 41] and state["remaining"][-1] == 1
    discovery = bindings[0]
    # Every observed start is accounted once: 96 charged plus two denied members; none renewed.
    assert discovery.accepted
    assert (discovery.calls, discovery.denied_starts, discovery.remaining_tool_calls) == (98, 2, 0)
    local, *challenges = value["execution"]
    assert (local["allocation"], local["observed_starts"], local["overshoot"]) == (96, 98, 2)
    # Overshoot is reported, never charged against the protected challenge allocation.
    assert challenges[0]["allocation"] == 96 and [e["assignment"] for e in challenges] == ["challenge"] * 2
    spanning = next(c for c in value["claims"] if c["title"] == "Cross-source value contract")
    assert spanning["disposition"] == "unresolved" and len(spanning["challenge_evidence"]) == 2
    decisions = {d["target_id"]: d for scope in value["terminal_result"]["stack_outcomes"]
                 for d in scope["target_decisions"]}
    assert {t for t, d in decisions.items() if d["status"] == "reviewed"} == {state["a"], state["c"], state["d"]}
    assert decisions[state["b"]]["reason"].startswith("unsubmitted:")
    assert len(value["findings"]) == 1 and value["terminal_result"]["analysis_state"] == "incomplete"
    assert value["analyzed_revision"]["head_sha"] == head
    trajectory = json.loads((repo / ".daydream/runs" / value["run_id"] / "trajectory.json").read_text())
    compaction = [item for step in trajectory["steps"] for item in step.get("extra", {}).get("backend_diagnostics", [])
                  if item["code"] == "pi_compaction"]
    assert [item["metadata"]["phase"] for item in compaction] == ["start", "end"]
    assert compaction[1]["metadata"]["usage"]["input"] == 10 and compaction[1]["metadata"]["summary_bytes"] > 0
    assert "earlier lookups and one admitted batch" not in json.dumps(compaction)
    # Every provider request is billed once, including the summarization call that is not a generation.
    billed = [native_provider.usage.get(index, {"prompt_tokens": 10, "completion_tokens": 5})
              for index in range(len(native_provider.requests))]
    assert trajectory["final_metrics"]["total_prompt_tokens"] == sum(usage["prompt_tokens"] for usage in billed)
    assert trajectory["final_metrics"]["total_completion_tokens"] == sum(u["completion_tokens"] for u in billed)
    rejection = next(m for r in native_provider.requests for m in r["messages"] if m.get("tool_call_id") == "call-55")
    assert "no available regular blob" in rejection["content"]


async def test_adaptive_native_review_challenges_a_spanning_defect_in_a_fresh_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider,
) -> None:
    """Installed Pi runs concurrent local assignments, required cross-assignment and fresh challenge.

    Scripted program journeys substitute run_agent, so they cannot show that the host
    dispatches cross-assignment through real native sessions when every local
    assignment is empty, that each sibling's native starts are charged to its own
    share, or that challenge of the spanning claim carries none of its author's history.
    """
    repo, base = _adaptive_repo(tmp_path)
    (native_provider.config / "settings.json").write_text(json.dumps({
        "defaultProvider": "output-fixture", "defaultModel": "output-model", "defaultProjectTrust": "always",
        "retry": {"enabled": False, "maxRetries": 0}}))
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    seen: dict[str, list[dict[str, Any]]] = {}

    def choose(request: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        assignment = assignment_from_request(request)
        work = assignment["work"]
        seen.setdefault(work["work_id"], []).append(request)
        if work["kind"] == "discovery":
            # Every local assignment is empty: decided targets, no concerns.
            return [(f"{work['work_id']}-final", "structured_output", {"targets": [
                {"target_id": t, "status": "reviewed", "reason": "", "evidence_refs": []}
                for t in work["targets"]["ids"]], "claims": []})]
        if work["kind"] == "cross_assignment":
            targets = looked_up_targets(request)
            if not targets:
                # A parallel two-member lookup of original source across both assignments.
                return [(f"cross-{name}", "review_source", {"target_id": name, "path": None, "cursor": None})
                        for name in ("a.py", "f.py")]
            refs = [{"source_id": next(o["source_id"] for o in targets[name]["source_options"] if o["side"] == "new"),
                     "basis": "supplied_diff", "source_range": None} for name in ("a.py", "f.py")]
            return [("cross-final", "structured_output", {"targets": [], "claims": [{
                "uid": "", "revision": 0, "source_uids": [], "title": "Caller divides by zero",
                "payload": {"claim": "ratio(4, 1) now divides by zero.", "trigger": "Import f.py.",
                            "consequence": "ZeroDivisionError at import.", "grounds": "a.py uses count - 1.",
                            "file": "f.py", "line": None, "evidence_refs": refs}}]})]
        claim, = assignment["claims"]
        return [("challenge-final", "structured_output", {"verdicts": [{
            "uid": claim["uid"], "revision": claim["revision"], "verdict": "confirmed",
            "grounds": "No guard precedes the division in captured a.py.", "evidence_refs": []}], "claims": []})]

    native_provider.choose = choose
    result = await run_committed_review(CommittedReviewRequest(
        repo, base, assignment_policy="adaptive",
        config=RunConfig(backend="pi", output_mode="review", archive=False, run_eval=False,
                         review_profile=_profile_with_pipeline(reviewer_investigation_budget_s=240,
                                                               reviewer_tool_call_budget=40))))
    value = result.to_dict()
    validate_committed_result(value)
    assert result.complete
    entries = {e["work_id"]: e for e in value["execution"]}
    kinds = {r["work_id"]: r["kind"] for r in value["work"]}
    assert sorted(kinds.values()) == ["challenge", "cross_assignment", "discovery", "discovery"]
    cross = next(w for w, k in kinds.items() if k == "cross_assignment")
    challenge = next(w for w, k in kinds.items() if k == "challenge")
    # Native starts are charged per work item from its own stream: siblings 1 each, cross 2 lookups + submit.
    assert sorted((entries[w]["allocation"], entries[w]["observed_starts"])
                  for w, k in kinds.items() if k == "discovery") == [(7, 1), (8, 1)]
    assert (entries[cross]["allocation"], entries[cross]["observed_starts"]) == (18, 3)
    assert entries[challenge]["allocation"] == 40 - 5
    finding, = value["findings"]
    assert finding["work_id"] == cross and finding["challenge_evidence"][0]["work_id"] == challenge
    # Fresh challenge context: its first native request carries no cross-assignment history.
    first = seen[challenge][0]
    assert [m["role"] for m in first["messages"] if m["role"] not in {"system", "developer"}] == ["user"]
    assert "cross-final" not in json.dumps(seen[challenge]) and "cross-a.py" not in json.dumps(seen[challenge])
