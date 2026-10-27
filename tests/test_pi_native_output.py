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
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from daydream.agent import run_agent
from daydream.backends import (
    PiRequestConfig,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.backends._transport import CliTransport, teardown
from daydream.backends.pi import PiBackend, PiError
from daydream.committed_review import CommittedReviewRequest, run_committed_review
from daydream.committed_review_model import validate_committed_result
from daydream.host_tools import HostToolBinding
from daydream.host_tools_server import PiHostToolServer
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.review_source_refs import (
    SOURCE_REFS_SCHEMA,
    SOURCE_SELECTORS_SCHEMA,
    canonicalize_source_selectors,
    part_source_mapping,
    source_options,
)
from daydream.run_config import RunConfig
from daydream.trajectory import DaydreamPhase
from tests.harness.git_helpers import git, seed_feature_branch
from tests.harness.otlp import _loopback_http_server, _QuietHTTPHandler
from tests.harness.trajectory import make_recorder

_SCHEMA = {"type": "object", "additionalProperties": False,
    "properties": {"verdict": {"type": "string"}, "findings": {"type": "array"}}, "required": ["verdict", "findings"]}


@dataclass
class _NativeProvider:
    config: Path
    requests: list[dict[str, Any]] = field(default_factory=list)
    authorizations: list[str | None] = field(default_factory=list)
    turns: list[list[tuple[str, str, dict[str, Any]]]] = field(default_factory=list)
    choose: Callable[[dict[str, Any]], list[tuple[str, str, dict[str, Any]]]] | None = None
    text: str | None = None
    errors: dict[int, str] = field(default_factory=dict)
    finish_reasons: dict[int, str] = field(default_factory=dict)


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

    class Provider(_QuietHTTPHandler):
        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fixture.requests.append(request)
            fixture.authorizations.append(self.headers.get("Authorization"))
            index = len(fixture.requests) - 1
            if index in fixture.errors:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": fixture.errors[index]}}).encode())
                return
            batch = (fixture.choose(request) if fixture.choose is not None else
                     fixture.turns[index] if index < len(fixture.turns) else [])
            delta: dict[str, Any] = {"role": "assistant", "content": fixture.text or f"Native reasoning {index}."}
            if batch:
                delta["tool_calls"] = [{"index": i, "id": call_id, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)}}
                    for i, (call_id, name, arguments) in enumerate(batch)]
            finish_reason = fixture.finish_reasons.get(index, "tool_calls" if batch else "stop")
            chunks: list[dict[str, Any]] = [
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(("data: " + json.dumps({"id": f"host-output-{index}",
                    "object": "chat.completion.chunk", "created": 1, "model": request["model"], **chunk})
                    + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    with _loopback_http_server(Provider) as base_url:
        (config / "models.json").write_text(json.dumps({"providers": {"output-fixture": {
            "baseUrl": f"{base_url}/v1", "api": "openai-completions", "apiKey": "synthetic-loopback-only",
            "models": [{"id": "output-model", "reasoning": False, "input": ["text"],
                        "contextWindow": 32768, "maxTokens": 8192}]}}}))
        monkeypatch.setenv("PI_PROVIDER", "output-fixture")
        yield fixture


@pytest.mark.parametrize("case", ["defect", "empty", "false-positive", "fallback", "hosted",
                                  "hosted-unsupported-pi", "hosted-unsupported-node", "hosted-resources",
                                  "hosted-capability", "hosted-config-link", "hosted-challenge-capability",
                                  "hosted-model-fallback", "hosted-settings-encoding", "hosted-loader-override"])
async def test_committed_review_native_challenge_uses_fresh_captured_source_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, case: str,
) -> None:
    """Production composition needs fresh challenge, frozen reads and sealed output.

    Blank, absent and explicit selections are compared with Pi itself;
    direct adapter journeys cannot detect host selection or shared-history bugs.
    The hosted row additionally proves that native resource/context discovery and
    attempted effects cannot broaden the operator policy across fresh challenges.
    Unsupported versions must retain the command and expected version in their
    mismatch diagnostic; a generic readiness rejection would hide the cause.
    Suffix-only model registries and invalid UTF-8 settings exercise readiness
    faults that the exact-model, valid-JSON success path cannot detect.
    An explicit operator loader override must be rejected before its preload can
    execute; hostile ambient loader settings alone only prove environment isolation.
    """
    empty = case in {"empty", "fallback"}
    hosted = case.startswith("hosted")
    defect = case == "defect" or hosted
    repo = tmp_path / "committed-source"
    original = "def ratio(total, count):\n    return total / count\n"
    captured = ("def ratio(total, count):\n    return total / (count - 1)\n" if defect else
                "def ratio(total, count):\n    return 0 if count == 0 else total / count\n")
    context = "Review the committed ratio implementation carefully.\n"
    head = seed_feature_branch(repo, base={"ratio.py": original, "AGENTS.md": context},
                               feature={"ratio.py": captured, "AGENTS.md": context})
    base = git(repo, "rev-parse", "main")
    working = ("def ratio(total, count):\n    return 'DIRTY_SOURCE_MUST_REMAIN_UNREVIEWED'\n"
               if defect else captured)
    if defect:
        (repo / "ratio.py").write_text(working)
        (repo / "AGENTS.md").write_text("DIRTY_CONTEXT_MUST_REMAIN_UNREVIEWED\n")
    settings = native_provider.config / "settings.json"
    expected_model = "native-inherited-model" if case == "fallback" else "alternate-model" if empty else "output-model"
    expected_provider = "nous" if case == "fallback" else "output-fixture"
    settings.write_text(json.dumps({
        **({"defaultProvider": "nous" if case == "false-positive" else "output-fixture"}
           if case != "fallback" else {}),
        **({"defaultModel": "output-model"} if case != "fallback" else {}),
        "defaultThinkingLevel": "low",
        "defaultProjectTrust": "ask" if case == "defect" else "always",
        "modelThinkingLevels": {"output-fixture/output-model": "high" if case == "defect" else "medium",
                                "output-fixture/alternate-model": "high", "nous/native-inherited-model": "high"},
        **({"enabledModels": ["output-fixture/alternate-model", "output-fixture/output-model"]}
           if case == "empty" else {"enabledModels": ["output-fixture/alternate-model"]}
           if case == "false-positive" else {}),
        "retry": {"enabled": False, "maxRetries": 0},
    }))
    # Untrusted overrides are ignored; trusted blanks reset selection, while explicit thinking wins.
    (repo / ".git/info/exclude").write_text(".pi/\n")
    project_settings = repo / ".pi/settings.json"
    project_settings.parent.mkdir()
    project_settings.write_text(json.dumps({
        **({"defaultModel": "alternate-model"} if case == "defect" else
           {"defaultModel": ""} if case == "empty" else {}),
        "modelThinkingLevels": ({"unrelated-provider/model": "off"} if empty else
                                {"output-fixture/output-model": "low"}),
    }))
    models_path = native_provider.config / "models.json"
    models = json.loads(models_path.read_text())
    models["providers"]["output-fixture"]["models"][0]["reasoning"] = True
    model = models["providers"]["output-fixture"]["models"][0]
    models["providers"]["output-fixture"]["models"].append({**model, "id": "alternate-model"})
    if case == "fallback":
        models["providers"] = {"nous": {**models["providers"]["output-fixture"],
            "apiKey": "$NOUS_API_KEY", "authHeader": True, "models": [{**model, "id": expected_model}]}}
        monkeypatch.setenv("PI_API_KEY", "synthetic-implicit-fallback-alias")
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
            'review_profile = "hostile-missing-profile.toml"\n[phases.arbiter]\nbackend = "codex"\n')
        expected_provider, expected_model = "nous", "deepseek/deepseek-v4.1-flash"
        models_path.write_text(json.dumps({"providers": {"nous": {
            **models["providers"]["output-fixture"], "apiKey": "$NOUS_API_KEY", "authHeader": True,
            "models": [{**model,
                        "id": expected_model + "-1017" if case == "hosted-model-fallback" else expected_model}],
        }}}))
        settings.write_text('{"defaultProjectTrust": "always", "retry": {"enabled": false, "maxRetries": 0}}')
        (native_provider.config / "trust.json").write_text(json.dumps({
            str(native_provider.config.resolve()): True, str(repo.resolve()): True}))
        operator_environment = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "operator-home"),
                                "NOUS_API_KEY": "synthetic-hosted-credential",
                                "DAYDREAM_PI_RETRY_ATTEMPTS": "1", "DAYDREAM_PI_RETRY_BASE_DELAY_S": "0",
                                "DAYDREAM_PI_RETRY_MAX_DELAY_S": "0"}
        if case == "hosted":
            native_provider.errors[0] = "503 Service Unavailable"
        elif case in {"hosted-capability", "hosted-challenge-capability"}:
            assets = tmp_path / "unsupported-assets"
            assets.mkdir()
            packaged = files("daydream.backends")
            for name in ("pi_structured_output.ts", "pi_host_tools.ts"):
                source = packaged.joinpath(name).read_text()
                if name == "pi_host_tools.ts":
                    if case == "hosted-capability":
                        source = source.replace("export default function (pi: ExtensionAPI) {",
                            "export default function (pi: ExtensionAPI) {\n  (pi as any).getAllTools = undefined;")
                    else:
                        source = source.replace('packet = JSON.parse(readFileSync(fd).toString("utf8"));',
                            'packet = JSON.parse(readFileSync(fd).toString("utf8"));\n'
                            '      if ("verdicts" in packet.schema.properties) (pi as any).getAllTools = undefined;')
                (assets / name).write_text(source)
            monkeypatch.setattr("daydream.backends.pi.files", lambda _: assets)
        elif case == "hosted-config-link":
            target_auth = repo / ".pi/auth.json"
            target_auth.write_text("{}")
            (native_provider.config / "auth.json").symlink_to(target_auth)
        elif case == "hosted-settings-encoding":
            settings.write_bytes(b"\xff")
        elif case == "hosted-loader-override":
            operator_environment["NODE_OPTIONS"] = f"--import={tmp_path / 'ambient-pi/preload.mjs'}"
        if case.startswith("hosted-unsupported"):
            executable_name = "node" if case.endswith("node") else "pi"
            executable = shutil.which(executable_name)
            assert executable is not None
            shim_dir = tmp_path / "unsupported-bin"
            shim_dir.mkdir()
            shim = shim_dir / executable_name
            unsupported_version = "v23.0.0" if executable_name == "node" else "1.3.0"
            shim.write_text(f"#!{sys.executable}\nimport os, sys\n"
                f"if sys.argv[1:] == ['--version']:\n print({unsupported_version!r})\n"
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
            pipeline=replace(profile.pipeline, reviewer_investigation_budget_s=60, review_wall_budget_s=80)),
            source_kind="operator")
        hosted_policy = HostedReviewPolicy(
            agent_dir=native_provider.config, environment=operator_environment,
            output_dir=output_root, private_base=tmp_path / "hosted-private",
            profile=operator_profile,
            limits=ReviewLimits(90, 30, 30), trusted_prepared_source=True,
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
    if not hosted:
        monkeypatch.delenv("PI_PROVIDER", raising=False)
    # Compare with the ordinary CLI, including its implicit default-provider authentication.
    arguments = ["pi", "--mode", "json", "--no-session", "--no-skills", "--no-extensions", "--no-tools",
                 "--no-context-files"]
    if case == "false-positive":
        arguments.extend(["--model", "output-model:high"])
    control_env = os.environ.copy()
    if case == "fallback":
        control_env["NOUS_API_KEY"] = "synthetic-implicit-fallback-alias"
    if not hosted:
        control = await asyncio.to_thread(subprocess.run, [*arguments, "-p", "Return the empty result."],
                                          cwd=repo, env=control_env, capture_output=True, timeout=30)
        assert control.returncode == 0, control.stderr.decode()
        assert len(native_provider.requests) == 1 and native_provider.requests[0]["model"] == expected_model
    native_provider.requests.clear()
    native_provider.authorizations.clear()
    assignments_seen: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    counterevidence = "The captured count == 0 guard returns zero before division; the proposed trigger cannot fail."

    def choose(request: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        if hosted:
            systems = json.dumps([message for message in request["messages"]
                                  if message["role"] in {"system", "developer"}])
            assert context.strip() not in systems
            assert "HOSTILE_ANCESTOR_POLICY" not in systems and "HOSTILE_GLOBAL_POLICY" not in systems
            assert {tool["function"]["name"] for tool in request["tools"]} == {
                "read", "grep", "find", "ls", "structured_output"}
        else:
            assert context.strip() in json.dumps(request["messages"])
        assert "DIRTY_CONTEXT_MUST_REMAIN_UNREVIEWED" not in json.dumps(request["messages"])
        users = [message for message in request["messages"] if message["role"] == "user"]
        assert len(users) == 1
        content = users[0]["content"]
        text = content if isinstance(content, str) else "\n".join(block.get("text", "") for block in content)
        assignment = json.loads(text.split("\nCommitted review assignment:\n", 1)[1].split(
            "\nEnd committed review assignment.\n", 1)[0])
        kind = assignment["kind"]
        if hosted:
            assert assignment["profile"]["profile_name"] == operator_profile.name
            assert assignment["profile"]["profile_digest"] == operator_profile.digest
            assert assignment["profile"]["profile_source"] == "operator"
            assert assignment["profile"]["limits"]["effective_investigation_s"] == pytest.approx(50)
            marker = "HOSTED_OPERATOR_DISCOVERY_POLICY" if kind == "discovery" else "HOSTED_OPERATOR_CHALLENGE_POLICY"
            assert marker in text and assignment["profile"]["strategies"][kind] == marker
        revision = assignment["claims"][0]["revision"] if kind == "challenge" else 0
        call_kind = f"{kind}-{revision}" if hosted else kind
        tools = [message for message in request["messages"] if message["role"] == "tool"]
        if not tools:
            assignments_seen.append(assignment)
            histories[kind] = request["messages"]
            if hosted:
                assert not any(message["role"] in {"assistant", "tool"} for message in request["messages"])
                return [(f"{call_kind}-bash", "bash", {"command": f"touch {forbidden_marker}"}),
                        (f"{call_kind}-write", "write", {"path": str(forbidden_marker), "content": "forbidden"}),
                        (f"{call_kind}-edit", "edit", {"path": "ratio.py", "oldText": captured, "newText": "MUTATED"}),
                        (f"{call_kind}-ungranted", "publish_review", {"path": str(forbidden_marker)})]
            return [(f"{call_kind}-read", "read", {"path": "ratio.py"})]
        if hosted and not any(tool["tool_call_id"] == f"{call_kind}-read" for tool in tools):
            assert len(tools) == 4
            assert all("not found" in tool["content"].lower() or "denied" in tool["content"].lower() for tool in tools)
            return [(f"{call_kind}-read", "read", {"path": "ratio.py"})]
        read_result, = [tool for tool in tools if tool["tool_call_id"] == f"{call_kind}-read"]
        assert read_result["content"] == captured
        if hosted and not any(tool["tool_call_id"] == f"{call_kind}-forbidden-domain" for tool in tools):
            return [(f"{call_kind}-forbidden-domain", "structured_output", {"write_path": str(forbidden_marker)})]
        if hosted:
            rejection, = [tool for tool in tools if tool["tool_call_id"] == f"{call_kind}-forbidden-domain"]
            assert rejection["content"].startswith('Validation failed for tool "structured_output":')
        refs_by_target = {part["target_id"]: [{"source_id": next(
            option["source_id"] for option in part["source_options"] if option["side"] == "new"),
            "source_range": None, "basis": "supplied_diff"}] for part in assignment["assignments"]}
        refs = next(iter(refs_by_target.values()))
        if kind == "discovery":
            claim = {"uid": "", "revision": 0, "source_uids": [], "title": "Division can fail", "payload": {
                "claim": "The changed denominator divides by zero.",
                "trigger": "Call ratio(5, 1)." if defect else "Call ratio(5, 0).",
                "consequence": "A normal input raises ZeroDivisionError.",
                "grounds": "Investigator is confident and considers its investigation complete.",
                "file": "ratio.py", "line": None if defect else 2, "evidence_refs": refs}}
            payload = {"targets": [{"target_id": target_id, "status": "reviewed", "reason": "Captured source checked.",
                                    "evidence_refs": target_refs} for target_id, target_refs in refs_by_target.items()],
                       "claims": [] if empty else [claim]}
        else:
            assert kind == "challenge" and len(assignment["claims"]) == 1
            claim = assignment["claims"][0]
            payload = {"verdicts": [{"uid": claim["uid"], "revision": claim["revision"],
                "verdict": "confirmed" if defect else "rejected",
                "grounds": "At count=1 the captured denominator is zero." if defect else counterevidence,
                "evidence_refs": refs}], "claims": []}
            if hosted and claim["revision"] == 1:
                payload["claims"] = [{key: claim[key] for key in ("uid", "revision", "source_uids", "title")} | {
                    "payload": {**claim["payload"], "grounds": "Captured denominator count-1 is zero at count=1.",
                                "evidence_refs": refs}}]
        return [(f"{call_kind}-submit", "structured_output", payload)]

    native_provider.choose = choose
    request = CommittedReviewRequest(
        repo, base, config=RunConfig(backend="codex" if hosted else "pi", output_mode="review", archive=False,
                                    run_eval=False, review_backend="claude" if hosted else None,
                                    review_profile_path="hostile-missing-profile.toml" if hosted else None,
                                    model="output-model:high" if case == "false-positive" else None),
        limits=ReviewLimits(90, 12, 12), hosted_policy=hosted_policy)
    unsupported = {"hosted-resources", "hosted-capability", "hosted-config-link", "hosted-challenge-capability",
                   "hosted-model-fallback", "hosted-settings-encoding", "hosted-loader-override"}
    if case.startswith("hosted-unsupported") or case in unsupported:
        from daydream.hosted_review import UnsupportedHostedPolicy

        message = {
            "hosted-model-fallback": "Hosted Pi model selection",
            "hosted-unsupported-pi": "pi did not report 1[.]4[.]0",
            "hosted-unsupported-node": "node did not report v24[.]18[.]1",
            "hosted-loader-override": "Hosted Pi environment must not contain executable loader overrides",
        }.get(case, "Hosted|hosted")
        with pytest.raises(UnsupportedHostedPolicy, match=message):
            await run_committed_review(request)
        assert len(native_provider.requests) == (4 if case == "hosted-challenge-capability" else 0)
        assert not forbidden_marker.exists()
        assert (repo / ".review-output.md").read_text() == "prior public report\n"
        if case == "hosted-loader-override":
            assert not output_root.exists() and not (tmp_path / "hosted-private").exists()
        if case in {"hosted-capability", "hosted-challenge-capability"}:
            partial = json.loads((output_root / "committed-review.json").read_text())
            validate_committed_result(partial)
            assert partial["terminal_result"]["analysis_state"] == (
                "incomplete" if case == "hosted-challenge-capability" else "failed")
            assert any(execution["reason"] == "backend_failure" for execution in partial["execution"])
            if case == "hosted-challenge-capability":
                assert [assignment["kind"] for assignment in assignments_seen] == ["discovery"]
                assert partial["analyzed_revision"] == assignments_seen[0]["analyzed_revision"]
                assert partial["run_id"] == assignments_seen[0]["run_id"]
                assert partial["assignments"] == assignments_seen[0]["assignments"]
                report = (output_root / "review-output.md").read_text()
                identities = (partial["planned_work_id"], partial["run_id"], head, base)
                assert all(identity in report for identity in identities)
                claim, = partial["claims"]
                assert claim["revision"] == 1 and claim["disposition"] == "unresolved"
                assert claim["payload"]["file"] == "ratio.py" and claim["challenge_evidence"] == []
        return
    result = await run_committed_review(request)
    assert result.complete and result.output_installed and result.capture_state == "complete"
    value = result.to_dict()
    validate_committed_result(value)
    assert value["analyzed_revision"]["head_sha"] == head
    assert value["analyzed_revision"]["merge_base_sha"] == base
    assert value["terminal_result"]["analysis_state"] == "complete"
    assert {part["file"] for part in value["assignments"]} == {"ratio.py"}
    expected_kinds = ["discovery"] if empty else ["discovery", "challenge", *(["challenge"] if hosted else [])]
    assert [assignment["kind"] for assignment in assignments_seen] == expected_kinds
    assert len(native_provider.requests) == (4 if hosted else 2) * len(expected_kinds) + int(hosted)
    assert all(request["model"] == expected_model for request in native_provider.requests)
    if not hosted:
        assert all(request["reasoning_effort"] == "high" for request in native_provider.requests)
    else:
        assert native_provider.authorizations == ["Bearer synthetic-hosted-credential"] * len(native_provider.requests)
        assert all({tool["function"]["name"] for tool in request["tools"]} == {
            "read", "grep", "find", "ls", "structured_output"} for request in native_provider.requests)
        assert not forbidden_marker.exists()
        assert value["settings"]["profile_name"] == operator_profile.name
        assert value["settings"]["profile_digest"] == operator_profile.digest
        assert value["settings"]["limits"]["effective_investigation_s"] == pytest.approx(50)
    if case == "fallback":
        assert native_provider.authorizations == ["Bearer synthetic-implicit-fallback-alias"] * len(
            native_provider.requests)
    if not empty:
        assert not any(message["role"] in {"assistant", "tool"} for message in histories["challenge"])
        assert "discovery-read" not in json.dumps(histories["challenge"])
        assert assignments_seen[1]["captured_root"] == assignments_seen[0]["captured_root"]
        claim = value["claims"][0]
        assert claim["source_uids"] == [] and claim["revision"] == (2 if hosted else 1)
        assert claim["payload"]["line"] == (None if defect else 2)
        assert claim["disposition"] == ("confirmed" if defect else "rejected")
        assert claim["challenge_evidence"][0]["revision"] == 1
        if hosted:
            assert [evidence["revision"] for evidence in claim["challenge_evidence"]] == [1, 2]
        assert len(value["findings"]) == int(defect)
        if case == "false-positive":
            assert claim["challenge_evidence"][0]["grounds"] == counterevidence
    else:
        assert value["claims"] == value["findings"] == []
    result_path = output_root / ("committed-review.json" if hosted else ".daydream/committed-review.json")
    assert json.loads(result_path.read_text()) == value
    report = (output_root / ("review-output.md" if hosted else ".review-output.md")).read_text()
    assert head in report and base in report and value["planned_work_id"] in report
    assert ("Division can fail" in report) == (not empty)
    if not empty:
        line_suffix = "" if defect else ":<code>2</code>"
        assert f"Location: <code>ratio.py</code>{line_suffix}." in report
    if case == "false-positive":
        assert counterevidence in report and "Rejected concerns" in report
    trajectory_path = (output_root / "trajectory.json" if hosted else
                       output_root / ".daydream/runs" / value["run_id"] / "trajectory.json")
    trajectory = json.loads(trajectory_path.read_text())
    if hosted:
        assert trajectory["extra"]["profile_name"] == operator_profile.name
        assert trajectory["extra"]["profile_digest"] == operator_profile.digest
    calls = [call for step in trajectory["steps"] for call in step.get("tool_calls") or []]
    if not hosted:
        assert [call["tool_call_id"] for call in calls] == [
            f"{kind}-{action}" for kind in expected_kinds for action in ("read", "submit")]
    else:
        assert {"discovery-0-read", "challenge-1-read", "challenge-2-read"} <= {
            call["tool_call_id"] for call in calls}
        assert (repo / ".review-output.md").read_text() == "prior public report\n"
        assert (repo / ".daydream/diff.patch").read_text() == "prior public artifact\n"
        assert not (repo / ".daydream/committed-review.json").exists()
    invocations = trajectory["extra"]["subtrajectories"]
    expected_invocations = len(expected_kinds) + int(hosted)
    assert len(invocations) == expected_invocations
    assert len({invocation["invocation_id"] for invocation in invocations}) == expected_invocations
    assert all(invocation["generation_lifecycle"]["drafts"] for invocation in invocations)
    generations = [generation for invocation in invocations
                   for generation in invocation["generation_lifecycle"]["drafts"]]
    assert all(generation["model_name"] == expected_model and generation["provider_name"] == expected_provider
               for generation in generations)
    assert all(execution["settled"] and execution["accepted"] and execution["cleanup_complete"]
               for execution in value["execution"])
    assert (repo / "ratio.py").read_text() == working and git(repo, "rev-parse", "HEAD") == head
    assert (repo / "AGENTS.md").read_text() == (
        "DIRTY_CONTEXT_MUST_REMAIN_UNREVIEWED\n" if defect else context)
    assert all(path.read_bytes() == original_bytes for path, original_bytes in config_bytes.items())


async def test_native_stock_pagination_and_corrected_source_submission_are_admitted_once(
    tmp_path: Path, native_provider: _NativeProvider,
) -> None:
    """Real Pi corrects host-rejected selectors and terminates on canonical acceptance.

    Source-owner tests cannot catch lost native error correlation, replaced stock
    pagination, the obsolete eight-call ceiling, or an extra generation after acceptance.
    """
    repo = tmp_path / "source-repo"
    supporting = "".join(f"Page {index:02}: café 界 independently known bytes\n" for index in range(1, 22))
    sources = {"supporting.txt": supporting, "oversized.txt": "x" * (64 * 1024 + 1) + "\n"}
    head = seed_feature_branch(repo, base={"api.py": "VALUE = 0\n", **sources},
                               feature={"api.py": "VALUE = 1\n", **sources})
    base = git(repo, "rev-parse", "main")
    diff = git(repo, "diff", "--no-ext-diff", "main", "HEAD") + "\n"
    revision = {"merge_base_sha": base, "head_sha": head}
    assigned = part_source_mapping(diff, {"kind": "file", "file": "api.py"},
        artifact_sha256=hashlib.sha256(diff.encode()).hexdigest(), revision=revision,
        old_path="api.py", new_path="api.py")["source_refs"]
    offered = source_options(assigned)
    source = next(option for option in offered if option["side"] == "new")
    selectors = [{"source_id": source["source_id"], "source_range": None, "basis": "supplied_diff"},
                 {"path": "supporting.txt", "side": "new", "source_range": {"start": 2, "end": 3},
                  "basis": "source_inspection"}]
    canonical_refs = [{
        "artifact_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "part_sha256": hashlib.sha256(diff.encode()).hexdigest(), "revision": head,
        "path": "api.py", "side": "new", "source_range": {"start": 1, "end": 1}, "fragment": None,
        "basis": "supplied_diff", "role": "assigned",
    }, {
        "artifact_sha256": hashlib.sha256(supporting.encode()).hexdigest(), "part_sha256": None,
        "revision": head, "path": "supporting.txt", "side": "new", "source_range": {"start": 2, "end": 3},
        "fragment": None, "basis": "source_inspection", "role": "supporting",
    }]
    def schema(refs: dict[str, Any]) -> dict[str, Any]:
        return {"type": "object", "additionalProperties": False, "properties": {
            "evidence_refs": refs, "note": {"type": "string"}}, "required": ["evidence_refs", "note"]}
    submitted = {"evidence_refs": selectors, "note": "Accepted source judgment"}
    expected = {"evidence_refs": canonical_refs, "note": submitted["note"]}
    rejected = {"evidence_refs": [{"source_id": "source:" + "0" * 64, "source_range": None,
                                   "basis": "supplied_diff"}], "note": "REJECTED_SEMANTIC_NOTE"}
    invalid_range = {"evidence_refs": [{"source_id": source["source_id"],
        "source_range": {"start": 1, "end": 99}, "basis": "supplied_diff"}], "note": "Invalid range"}
    oversized = {"evidence_refs": [selectors[0], {"path": "oversized.txt", "side": "new",
        "source_range": {"start": 1, "end": 1}, "basis": "source_inspection"}], "note": "One line of a large blob"}
    installed: list[dict[str, Any]] = []
    def accept(value: Any) -> dict[str, Any]:
        canonical = {"evidence_refs": canonicalize_source_selectors(
            value["evidence_refs"], assigned, revision=revision, source_root=repo), "note": value["note"]}
        installed.append(canonical)
        return canonical
    path = str((repo / "supporting.txt").resolve())
    native_provider.turns = [[(f"page-{index}", "read", {"path": path, "offset": 2 * index + 1, "limit": 2})]
                             for index in range(10)]
    native_provider.turns += [[("rejected", "structured_output", rejected)],
                              [("invalid-range", "structured_output", invalid_range)],
                              [("oversized", "structured_output", oversized)],
                              [("accepted", "structured_output", submitted)]]
    count = 14
    budget = ReviewInvestigationBudget.from_limits(ReviewLimits(60, count, count))
    binding = HostToolBinding(deadline=time.monotonic() + 60, accept=accept)
    recorder = make_recorder(repo, agent_model_name="output-model", backend_name="pi")
    async with recorder:
        result, _, reason = await run_agent(PiBackend(model="output-model"), repo,
            "Inspect the supplied source pages and submit a source judgment.", phase=DaydreamPhase.REVIEW,
            read_only=True, persist_session=False, output_schema=schema(SOURCE_SELECTORS_SCHEMA),
            result_schema=schema(SOURCE_REFS_SCHEMA), investigation_budget=budget, tool_call_budget=count,
            host_tools=binding, wall_budget_s=60, progress_callback=lambda _: None)
    assert reason is None and result == expected
    assert installed == [expected] and binding.accepted and binding.accepted_result == expected
    assert len(native_provider.requests) == count, "acceptance must terminate without a further model generation"
    names = {tool["function"]["name"] for tool in native_provider.requests[0]["tools"]}
    assert {"read", "grep", "find", "ls", "structured_output"} <= names
    assert not {"bash", "edit", "write"} & names
    tool = next(tool for tool in native_provider.requests[0]["tools"]
                if tool["function"]["name"] == "structured_output")
    assert tool["function"]["parameters"] == schema(SOURCE_SELECTORS_SCHEMA)
    for index, request in enumerate(native_provider.requests):
        messages = request["messages"]
        assert len([message for message in messages if message["role"] == "user"]) == 1
        assistants = [message for message in messages if message["role"] == "assistant"]
        assert [message["content"] for message in assistants] == [f"Native reasoning {i}." for i in range(index)]
        previous = [call for batch in native_provider.turns[:index] for call in batch]
        history = {message["tool_call_id"]: message["content"] for message in messages if message["role"] == "tool"}
        assert set(history) == {call_id for call_id, _, _ in previous}
        for page in range(min(index, 10)):
            lines = supporting.splitlines(keepends=True)
            assert "".join(lines[2 * page:2 * page + 2]) in history[f"page-{page}"]
            assert lines[2 * page + 2].strip() not in history[f"page-{page}"]
            assert f"offset={2 * page + 3}" in history[f"page-{page}"]
        if index >= 11:
            assert "assigned inventory" in history["rejected"]
            assert "REJECTED_SEMANTIC_NOTE" in json.dumps(messages)
        if index >= 12:
            assert "source_range" in history["invalid-range"] and source["source_id"] in history["invalid-range"]
        if index >= 13:
            assert "oversized.txt" in history["oversized"] and "whole" in history["oversized"].lower()
            assert "65536" in history["oversized"] and "range" in history["oversized"].lower()
    trajectory = json.loads(recorder.path.read_text())
    calls = [call for step in trajectory["steps"] for call in step.get("tool_calls") or []]
    observations = {item["source_call_id"]: item["content"] for step in trajectory["steps"]
                    for item in (step.get("observation") or {}).get("results", [])}
    expected_ids = [f"page-{index}" for index in range(10)] + ["rejected", "invalid-range", "oversized", "accepted"]
    assert [call["tool_call_id"] for call in calls] == expected_ids
    assert "assigned inventory" in observations["rejected"]
    assert json.loads(observations["accepted"])["details"] == expected
    assert "REJECTED_SEMANTIC_NOTE" in json.dumps(trajectory), "rejection remains raw native evidence"
    assert all(step["source"] != "user" or "REJECTED_SEMANTIC_NOTE" not in json.dumps(step)
               for step in trajectory["steps"])
    assert binding.calls == budget.observed_tool_starts == count and budget.remaining_tool_calls == 0
    assert binding.rejections == 3 and not binding.authority_open
    assert binding.native_settled and binding.cleanup_complete
    lifecycle = trajectory["extra"]["subtrajectories"][0]["backend_lifecycle"]
    assert lifecycle["host_authority_closed"] and lifecycle["host_native_settled"]
    assert lifecycle["host_cleanup_complete"]
    assert lifecycle["host_tool_call_count"] == count and lifecycle["host_tool_rejection_count"] == 3
    assert sum((step.get("metrics") or {}).get("prompt_tokens", 0) or 0 for step in trajectory["steps"]) == 10 * count


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
    binding = HostToolBinding(deadline=time.monotonic() + 30, accept=accept)
    result, _, reason = await run_agent(PiBackend(model="output-model"), tmp_path, "Submit the completed review.",
        phase=DaydreamPhase.REVIEW, output_schema=_SCHEMA, read_only=True, persist_session=False,
        tool_call_budget=5, host_tools=binding, progress_callback=lambda _: None)
    assert reason is None and result == accepted and installed == [accepted]
    assert len(native_provider.requests) == 2
    history = {message["tool_call_id"]: message["content"] for message in native_provider.requests[1]["messages"]
               if message["role"] == "tool"}
    assert "rejected" in history
    assert ("token limit" if fault == "incomplete" else "alone") in history["rejected"].lower()
    assert binding.accepted_result == accepted and binding.rejections >= 1


@pytest.mark.parametrize("allowance", [1, 2], ids=["spent-by-native-schema-rejection", "correction-capacity"])
@pytest.mark.parametrize("malformation", ["schema", "argument-size"])
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
    binding = HostToolBinding(deadline=time.monotonic() + 30, accept=accept)
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
        assert "allowance" in str(failure).lower() and "cannot" in str(failure).lower()
    assert len(native_provider.requests) == allowance, "fatal admission must not dispatch another provider request"
    assert installed == ([accepted] if allowance == 2 else [])
    assert marker.exists() is (allowance == 2)
    if allowance == 2:
        assert marker.read_text() == "complete" and binding.accepted_result == accepted
        history = {message["tool_call_id"]: message["content"] for message in native_provider.requests[1]["messages"]
                   if message["role"] == "tool"}
        if malformation == "schema":
            assert history["invalid-schema"].startswith('Validation failed for tool "structured_output":')
        else:
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
    binding = HostToolBinding(deadline=time.monotonic() + 30, accept=accept)
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
                tool_call_budget=4, progress_callback=lambda _: None)
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


@pytest.mark.parametrize("fault", ["missing-extension", "broken-extension", "unsupported-capability",
                                  "wrong-schema", "unreachable-endpoint", "unsupported-version"])
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


@pytest.mark.parametrize("cancel_at", ["active-read", "late-start"])
async def test_native_stock_read_cancellation_reaps_process_before_freeze_and_preserves_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, cancel_at: str,
) -> None:
    """A blocked built-in read closes only its invocation before recorded evidence freezes.

    Fake subprocess cancellation misses Node filesystem work, a real child whose
    spawn returns after cancellation, and sibling authority on one cached backend.
    """
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
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
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
    close_authority = cancelled_binding.close_authority
    def observe_close() -> None:
        close_authority()
        authority_closed.set()
    monkeypatch.setattr(cancelled_binding, "close_authority", observe_close)
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
        # A nonblocking writer can open only after the real Node reader opens
        # this FIFO. Keep it open without bytes so the child cannot finish its
        # read naturally while the spawn-return barrier is still pending.
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


@pytest.mark.parametrize("fault", ["agent_settled", "tool_execution_end", "teardown-error", "teardown-timeout"])
async def test_native_accepted_state_is_retained_when_terminal_settlement_or_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, fault: str,
) -> None:
    """Native failure cannot undo acceptance, replay its mutation, or overstate cleanup.

    Domain stage tests cannot establish native terminal evidence or charged usage.
    Endpoint cleanup alone also misses a separately failing native process teardown.
    """
    accepted = {"verdict": "complete", "findings": []}
    native_provider.turns = [[("accepted", "structured_output", accepted)]]
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 30, accept=accept)
    servers: list[PiHostToolServer] = []
    start_server = PiHostToolServer.start
    async def observe_server(server: PiHostToolServer) -> None:
        servers.append(server)
        await start_server(server)
    monkeypatch.setattr(PiHostToolServer, "start", observe_server)
    if fault.startswith("teardown-"):
        async def failing_teardown(transport: CliTransport, transports: list[CliTransport],
                                   **options: Any) -> None:
            await teardown(transport, transports, **options)
            if binding.accepted:
                if fault == "teardown-error":
                    raise RuntimeError("Native teardown failed after process cleanup")
                async with asyncio.timeout(0):
                    await asyncio.Event().wait()
        monkeypatch.setattr("daydream.backends.pi.teardown", failing_teardown)
    else:
        original = CliTransport.lines
        async def lose_event(transport: CliTransport, *args: Any, **kwargs: Any) -> Any:
            async for line in original(transport, *args, **kwargs):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    yield line
                    continue
                if event.get("type") != fault:
                    yield line
        monkeypatch.setattr(CliTransport, "lines", lose_event)
    backend = PiBackend(model="output-model")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = backend.retry_max_delay_s = 0
    recorder = make_recorder(tmp_path, backend_name="pi")
    expected_error = (RuntimeError if fault == "teardown-error" else
                      TimeoutError if fault == "teardown-timeout" else PiError)
    async with recorder:
        with pytest.raises(expected_error):
            await run_agent(backend, tmp_path, "Submit the review.", phase=DaydreamPhase.REVIEW,
                host_tools=binding, output_schema=_SCHEMA, read_only=True, persist_session=False,
                progress_callback=lambda _: None)
    assert installed == [accepted] and binding.accepted and binding.accepted_result == accepted
    assert len(native_provider.requests) == 1, "a possibly accepted mutation cannot be replayed"
    assert servers and all(server.closed for server in servers), "endpoint closure is an independent cleanup outcome"
    trajectory = json.loads(recorder.path.read_text())
    lifecycle = trajectory["extra"]["subtrajectories"][0]["backend_lifecycle"]
    assert lifecycle["host_authority_closed"] and not binding.authority_open
    if fault.startswith("teardown-"):
        assert binding.native_settled and lifecycle["host_native_settled"]
        assert not binding.cleanup_complete and lifecycle["host_cleanup_complete"] is False
    else:
        assert binding.cleanup_complete and lifecycle["host_cleanup_complete"]
        # Missing execution completion fails at the following toolResult, before
        # the adapter can consume the later agent_settled event.
        assert not binding.native_settled and lifecycle["host_native_settled"] is False
    assert sum((step.get("metrics") or {}).get("prompt_tokens", 0) or 0 for step in trajectory["steps"]) == 10


@pytest.mark.parametrize("fault", ["timeout", "disconnect", "uncorrelated-reply"])
async def test_native_indeterminate_submission_stops_generation_and_preserves_host_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_provider: _NativeProvider, fault: str,
) -> None:
    """Losing the acceptance reply ends native work without undoing an accepted canonical result.

    A native error assertion alone misses an unwanted next generation. Successful
    semantic correction is separately covered by the real source-selector journey.
    """
    submitted = {"verdict": "submitted", "findings": []}
    canonical = {"verdict": "complete", "findings": []}
    native_provider.turns = [[("submit", "structured_output", submitted)]]
    installed: list[dict[str, Any]] = []
    callback_times: list[float] = []
    timeout_s = 1.0
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        assert value == submitted
        callback_times.append(time.monotonic())
        if fault == "timeout":
            # Synchronous frozen-source work can outlast Node's independent
            # HTTP timer while the original domain deadline remains open.
            time.sleep(timeout_s + 0.25)
        binding.ensure_open()
        callback_times.append(time.monotonic())
        installed.append(canonical)
        return canonical
    binding = HostToolBinding(deadline=time.monotonic() + 30, accept=accept)
    packet = PiHostToolServer.packet
    def timed_packet(server: PiHostToolServer) -> dict[str, Any]:
        value = packet(server)
        return {**value, "request_timeout_ms": int(timeout_s * 1000)}
    monkeypatch.setattr(PiHostToolServer, "packet", timed_packet)
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
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            native_rows.append(row)
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
                tool_call_budget=4, progress_callback=lambda _: None)
    assert installed == [canonical] and binding.accepted_result == canonical and lost_replies
    assert len(callback_times) == 2 and callback_times[-1] < binding.deadline
    if fault == "timeout":
        assert callback_times[-1] - callback_times[0] > timeout_s
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


async def test_installed_pi_validation_optout_preserves_text(
    tmp_path: Path, native_provider: _NativeProvider,
) -> None:
    native_provider.text = json.dumps({"verdict": "complete", "findings": []})
    events = [event async for event in PiBackend(model="output-model").execute(
        tmp_path, "Return the requested result.", output_schema=_SCHEMA, read_only=True, persist_session=False,
        validate_structured_output=False, tool_call_budget=8)]
    request = next(event for event in events if isinstance(event, RequestEvent))
    assert isinstance(request.config, PiRequestConfig) and request.config.no_extensions is False
    assert request.output_schema == _SCHEMA
    assert not any(isinstance(event, ToolStartEvent) for event in events)
    assert next(event for event in reversed(events) if isinstance(event, ResultEvent)).structured_output is None
    assert [event.text for event in events if isinstance(event, TextEvent)] == [native_provider.text]


@pytest.mark.parametrize('allowance,submit', [(8, True), (7, True), (6, True), (2, True), (1, True), (None, True),
                                            (7, False), (6, False)],
                         ids=['ample', 'submission-only', 'batch-exhausted', 'batch-overdrawn',
                              'initial-only', 'unbounded', 'bounded-recovery', 'exhausted-no-recovery'])
async def test_native_pi_budget_warning_preserves_wire_roles_history_and_charged_calls(
    tmp_path: Path, native_provider: _NativeProvider, allowance: int | None, submit: bool,
) -> None:
    """Actual wire requests catch custom notes becoming users or dropping genuine history.

    The immediate-submission journey misses threshold transitions, failed/invalid batch
    charges and missing-submission recovery at an unchanged remaining allowance.
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
        *([[]] if allowance in (8, 7, None) else []),  # Repeat threshold through bounded recovery.
        [('final-submit', 'structured_output', output)] if submit else [],
    ]

    def choose(request: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
        index = len(requests) - 1
        native_provider.text = f'Original assistant reasoning summary {index}.'
        return batches[index]

    native_provider.choose = choose
    events = [event async for event in PiBackend(model='output-model').execute(
        tmp_path, prompt, output_schema=_SCHEMA, review_instructions=instructions,
        read_only=True, persist_session=False, tool_call_budget=allowance)]

    assert len(requests) == len(batches)
    request_event = next(event for event in events if isinstance(event, RequestEvent))
    assert isinstance(request_event.config, PiRequestConfig) and request_event.config.no_extensions is True
    assert request_event.output_schema == _SCHEMA
    completions: dict[str, str] = {}
    for event in events:
        if isinstance(event, ToolResultEvent):
            # The host retains native details as JSON; OpenAI's tool content carries text only.
            completions[event.id] = (''.join(block['text'] for block in json.loads(event.output)['content'])
                                     if event.output.startswith('{"content":') else event.output)
    warning_requests: list[int] = []
    recovery = 'Submit the final result now by calling structured_output alone. Assistant prose is not a submission.'
    warning_instruction = ('Final-submission budget warning: Only submission capacity remains. '
                           'Submit now using structured_output alone; report unfinished assigned work '
                           'with truthful incomplete decisions. Assigned objective and identities: '
                           + json.dumps(assignment, separators=(',', ':')))
    original_system: dict[str, Any] | None = None
    for index, request in enumerate(requests):
        messages = request['messages']
        system = [message for message in messages if message['role'] in {'system', 'developer'}]
        assert system and messages[0] == system[0]
        assert policy in json.dumps(system)
        assert 'Attempt dispatch-time snapshot:' in json.dumps(system)
        preserved_system = dict(system[0])
        preserved_system['content'] = (
            system[0]['content'].replace(warning_instruction, '').replace(recovery, '').strip())
        if original_system is None:
            original_system = preserved_system
        assert preserved_system == original_system
        users = [message for message in messages if message['role'] == 'user']
        assert len(users) == 1 and prompt in json.dumps(users[0])
        assert 'Live native Pi attempt budget:' not in json.dumps(messages)
        assert 'daydream_native_tool_budget' not in json.dumps(messages)
        assert request['tools'] == requests[0]['tools']
        assert next(tool for tool in request['tools'] if tool['function']['name'] == 'structured_output')[
            'function']['parameters'] == _SCHEMA
        assistants = [message for message in messages if message['role'] == 'assistant']
        assert len(assistants) == index
        for earlier, assistant in enumerate(assistants):
            assert assistant['content'] == f'Original assistant reasoning summary {earlier}.'
            assert [(call['id'], call['function']['name'], json.loads(call['function']['arguments']))
                    for call in assistant.get('tool_calls', [])] == batches[earlier]
        history = {message['tool_call_id']: message['content'] for message in messages if message['role'] == 'tool'}
        previous = [call for batch in batches[:index] for call in batch]
        assert set(history) == {call_id for call_id, _, _ in previous}
        for call_id, name, arguments in previous:
            assert history[call_id] == completions[call_id]
            if name == 'read' and arguments['path'] == 'source.txt':
                assert history[call_id] == source
        warnings = [message for message in messages if 'Final-submission budget warning:' in json.dumps(message)]
        if warnings:
            warning_requests.append(index)
            assert len(warnings) == 1 and warnings[0]['role'] in {'system', 'developer'}
            warning = json.dumps(warnings[0])
            assert 'Only submission capacity remains' in warning and 'truthful incomplete decisions' in warning
            assignment_text = warnings[0]['content'].split('Assigned objective and identities: ', 1)[1]
            assert json.JSONDecoder().raw_decode(assignment_text)[0] == assignment
        recovery_messages = [message for message in messages if recovery in json.dumps(message)]
        assert len(recovery_messages) == (1 if index == 4 else 0)
        assert all(message['role'] in {'system', 'developer'} for message in recovery_messages)
    assert warning_requests == ([3] if allowance == 7 else [0] if allowance == 1 else [])
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    assert [event.id for event in starts] == [call_id for batch in batches for call_id, _, _ in batch]
    failures = [event.id for event in events if isinstance(event, ToolResultEvent) and event.is_error]
    assert failures == ['read-missing', 'failed-submit']
    assert completions['read-missing'] == f"ENOENT: no such file or directory, access '{tmp_path / 'missing.txt'}'"
    assert completions['failed-submit'].startswith('Validation failed for tool "structured_output":\n')
    assert completions['failed-submit'].endswith(
        'Received arguments:\n{\n  "verdict": "incomplete",\n  "findings": false\n}')
    result = next(event for event in reversed(events) if isinstance(event, ResultEvent))
    assert result.structured_output == (output if submit else None)


@pytest.mark.parametrize("host_bound", [False, True], ids=["serializer", "host-admission"])
async def test_native_pi_retry_refreshes_attempt_count_without_renewing_host_allowance(
    tmp_path: Path, native_provider: _NativeProvider, host_bound: bool,
) -> None:
    """A real provider failure restarts warnings without refunding previously charged native calls."""
    (tmp_path / "source.txt").write_text("Captured source\n")
    requests = native_provider.requests
    output = {"verdict": "complete", "findings": []}
    native_provider.errors[1] = "503 Service Unavailable"
    native_provider.turns = [
        [("attempt-1-a", "read", {"path": "source.txt"}), ("attempt-1-b", "read", {"path": "missing.txt"})],
        [], [("attempt-2-submit", "structured_output", output)],
    ]
    backend = PiBackend(model="output-model")
    backend.retry_attempts = 1
    backend.retry_base_delay_s = backend.retry_max_delay_s = 0
    budget = ReviewInvestigationBudget.from_limits(ReviewLimits(60, 3, 3))
    installed: list[Any] = []
    def accept(value: dict[str, Any]) -> dict[str, Any]:
        installed.append(value)
        return value
    binding = HostToolBinding(deadline=time.monotonic() + 60, accept=accept) if host_bound else None
    deadline = binding.deadline if binding is not None else None
    result, _, reason = await run_agent(
        backend, tmp_path, "Investigate the supplied source and submit the result.",
        phase=DaydreamPhase.DEEP, output_schema=_SCHEMA, read_only=True, persist_session=False,
        investigation_budget=budget, tool_call_budget=3, host_tools=binding, progress_callback=lambda _: None)

    if binding is not None:
        assert installed == [output] and binding.accepted_result == output and binding.calls == 3
        assert deadline is not None and binding.deadline <= deadline and binding.cleanup_complete
    assert reason is None and result == output
    assert budget.observed_tool_starts == 3 and budget.remaining_tool_calls == 0
    assert len(requests) == 3
    for index, request in enumerate(requests):
        notes = [message for message in request['messages']
                 if 'Final-submission budget warning:' in json.dumps(message)]
        assert len(notes) == (0 if index == 0 else 1)
        assert all(message['role'] in {'system', 'developer'} for message in notes)
        assert len([message for message in request['messages'] if message['role'] == 'user']) == 1
        assert 'Live native Pi attempt budget:' not in json.dumps(request['messages'])
    assert 'tool-call allocation is 3 calls' in json.dumps(requests[0]['messages'])
    assert 'tool-call allocation is 1 calls' in json.dumps(requests[2]['messages'])
    assert 'attempt-1-a' not in json.dumps(requests[2]['messages'])
