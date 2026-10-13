"""Matched live deep-review trials; no posting, fixing, or cache reuse.

First create each fixture with --prepare --language python|go --variant
clean|local|cross. Then review that immutable repository with --fixture,
--source, --output (a new directory), and --mode off|auto|cap8.
Repeat paired trials with identical provider/model/effort and fixture HEAD.
Findings require independent adjudication; keyword counts are not quality scores.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _helper_sources(language: str) -> tuple[dict[str, str], dict[str, str]]:
    """Behavior-preserving API additions with separately introduced defects.

    Local: empty-batch mean raises or returns NaN. Cross: a caller passes milliseconds
    to the seconds-based expiry helper, making expired tokens valid again.
    Neither defect is described in repository instructions or review prompts.
    """
    if language == "python":
        before = {
            "stats.py": "def mean(values):\n    if not values:\n        return 0.0\n"
            "    return sum(values) / len(values)\n",
            "expiry.py": "def is_expired(expires_at_s, now_s):\n    return now_s >= expires_at_s\n",
            "tokens.py": "from expiry import is_expired\n\ndef valid_token(expires_at_ms, now_ms):\n"
            "    return not is_expired(expires_at_ms / 1000, now_ms / 1000)\n",
            "formatting.py": "def format_count(count):\n    return str(count)\n",
            "limits.py": "def clamp(value, maximum):\n    return max(0, min(value, maximum))\n",
            "paging.py": "def page(values, offset, limit):\n    return values[offset:offset + limit]\n",
            "sorting.py": "def sorted_ids(values):\n    return sorted(values)\n",
            "test_legacy.py": "from stats import mean\n\ndef test_old_mean():\n    assert mean([2, 4]) == 3\n",
        }
        after = {path: content + "\n# Public helpers remain stable for existing callers.\n"
                 for path, content in before.items() if path != "test_legacy.py"}
        after["stats.py"] += "\ndef total(values):\n    return sum(values)\n"
        after["expiry.py"] += "\ndef remaining_seconds(expires_at_s, now_s):\n    return max(0, expires_at_s - now_s)\n"
        after["tokens.py"] += "\ndef normalized_token(value):\n    return value.strip()\n"
        after["formatting.py"] += "\ndef format_label(value):\n    return value.strip()\n"
        after["limits.py"] += "\ndef within_limit(value, maximum):\n    return 0 <= value <= maximum\n"
        after["paging.py"] += "\ndef first_page(values, limit):\n    return page(values, 0, limit)\n"
        after["sorting.py"] += "\ndef unique_ids(values):\n    return sorted(set(values))\n"
        after["test_helpers.py"] = (
            "from stats import mean, total\nfrom tokens import valid_token\n\n"
            "def test_nonempty_statistics():\n    assert mean([2, 4]) == 3\n    assert total([2, 4]) == 6\n\n"
            "def test_future_token():\n    assert valid_token(5000, 1000)\n"
        )
    else:
        before = {
            "stats.go": "package helpers\n\nfunc Mean(values []float64) float64 {\n"
            "\tif len(values) == 0 { return 0 }\n\tvar sum float64\n\tfor _, value := range values { sum += value }\n"
            "\treturn sum / float64(len(values))\n}\n",
            "expiry.go": "package helpers\n\nfunc IsExpired(expiresAtS, nowS int64) bool {\n"
            "\treturn nowS >= expiresAtS\n}\n",
            "tokens.go": "package helpers\n\nfunc ValidToken(expiresAtMS, nowMS int64) bool {\n"
            "\treturn !IsExpired(expiresAtMS / 1000, nowMS / 1000)\n}\n",
            "formatting.go": "package helpers\n\nimport \"strconv\"\n\n"
            "func FormatCount(count int) string { return strconv.Itoa(count) }\n",
            "limits.go": "package helpers\n\nfunc Clamp(value, maximum int) int {\n"
            "\tif value < 0 { return 0 }; if value > maximum { return maximum }; return value\n}\n",
            "paging.go": "package helpers\n\nfunc Page(values []int, offset, limit int) []int {\n"
            "\tif offset >= len(values) { return []int{} }; end := offset + limit\n"
            "\tif end > len(values) { end = len(values) }; return values[offset:end]\n}\n",
            "sorting.go": "package helpers\n\nimport \"sort\"\n\nfunc SortedIDs(values []int) []int {\n"
            "\tresult := append([]int{}, values...); sort.Ints(result); return result\n}\n",
            "legacy_test.go": "package helpers\n\nimport \"testing\"\n\nfunc TestOldMean(t *testing.T) {\n"
            "\tif Mean([]float64{2, 4}) != 3 { t.Fatal(\"mean\") }\n}\n",
            "go.mod": "module example.com/helpers\n\ngo 1.23\n",
        }
        after = {path: content + ("\n// Public helpers remain stable for existing callers.\n"
                                 if path.endswith(".go") else "")
                 for path, content in before.items() if path != "legacy_test.go"}
        after["stats.go"] += "\nfunc Total(values []float64) float64 {\n\tvar sum float64\n"
        after["stats.go"] += "\tfor _, value := range values { sum += value }; return sum\n}\n"
        after["expiry.go"] += "\nfunc RemainingSeconds(expiresAtS, nowS int64) int64 {\n"
        after["expiry.go"] += "\tif nowS >= expiresAtS { return 0 }; return expiresAtS - nowS\n}\n"
        after["tokens.go"] += "\nfunc TokenPresent(value string) bool { return value != \"\" }\n"
        after["formatting.go"] += "\nfunc FormatTotal(counts []int) string {\n\ttotal := 0\n"
        after["formatting.go"] += "\tfor _, count := range counts { total += count }; return FormatCount(total)\n}\n"
        after["limits.go"] += "\nfunc WithinLimit(value, maximum int) bool { return value >= 0 && value <= maximum }\n"
        after["paging.go"] += "\nfunc FirstPage(values []int, limit int) []int { return Page(values, 0, limit) }\n"
        after["sorting.go"] += "\nfunc Smallest(values []int) (int, bool) {\n"
        after["sorting.go"] += "\tif len(values) == 0 { return 0, false }; return SortedIDs(values)[0], true\n}\n"
        after["helpers_test.go"] = (
            "package helpers\n\nimport \"testing\"\n\nfunc TestHelpers(t *testing.T) {\n"
            "\tif Mean([]float64{2, 4}) != 3 || Total([]float64{2, 4}) != 6 { t.Fatal(\"stats\") }\n"
            "\tif !ValidToken(5000, 1000) { t.Fatal(\"future token\") }\n}\n"
        )
    before["README.md"] = (
        "# Shared helpers\n\nThe mean of an empty batch is zero. Token expiry is supplied in milliseconds; "
        "expiry primitives accept Unix seconds. A token is invalid at or after its expiry. "
        "Paging callers supply nonnegative offsets and limits. Sorting never mutates caller input.\n"
    )
    after["README.md"] = before["README.md"] + "\nAdd convenience helpers while preserving existing behavior.\n"
    return before, after


def fixture_files(language: str, variant: str) -> tuple[dict[str, str], dict[str, str]]:
    before, after = _helper_sources(language)
    before["README.md"] += "\nLimit helpers receive nonnegative maximum bounds.\n"
    if language == "python":
        before["test_legacy.py"] = """from stats import mean
from tokens import valid_token
from expiry import is_expired
from formatting import format_count
from limits import clamp
from paging import page
from sorting import sorted_ids

def test_original_statistics():
    assert mean([]) == 0.0
    assert mean([2, 4]) == 3.0
    assert mean([-2, 2]) == 0.0

def test_original_expiry():
    assert not is_expired(5, 4)
    assert is_expired(5, 5)
    assert is_expired(5, 6)
    assert valid_token(5000, 1000)
    assert not valid_token(5000, 5000)
    assert not valid_token(5000, 6000)
    assert valid_token(1500, 1000)
    assert valid_token(1500, 1499)
    assert not valid_token(1500, 1500)
    assert not valid_token(1500, 1501)

def test_original_helpers():
    assert format_count(0) == "0"
    assert format_count(12) == "12"
    assert clamp(-1, 10) == 0
    assert clamp(5, 10) == 5
    assert clamp(11, 10) == 10
    assert clamp(0, 0) == 0
    assert page([1, 2, 3], 1, 2) == [2, 3]
    assert page([1, 2, 3], 4, 2) == []
    assert page([1, 2, 3], 0, 0) == []
    assert page([], 0, 2) == []
    assert page([1, 2, 3], 1, 1 << 100) == [2, 3]
    values = [3, 1, 2]
    assert sorted_ids(values) == [1, 2, 3]
    assert values == [3, 1, 2]
    assert sorted_ids([]) == []
"""
        after["test_helpers.py"] = before["test_legacy.py"] + """
from stats import total
from tokens import normalized_token
from expiry import remaining_seconds
from formatting import format_label
from limits import within_limit
from paging import first_page
from sorting import unique_ids

def test_added_statistics_and_expiry():
    assert total([]) == 0
    assert total([2, 4]) == 6
    assert total([-2, 2]) == 0
    assert remaining_seconds(5, 1) == 4
    assert remaining_seconds(5, 5) == 0
    assert remaining_seconds(5, 6) == 0
    assert remaining_seconds(1.5, 1.0) == 0.5

def test_added_strings_and_bounds():
    assert normalized_token("  token\\n") == "token"
    assert normalized_token("") == ""
    assert format_label("  label\\n") == "label"
    assert format_label(" ") == ""
    assert within_limit(0, 10)
    assert within_limit(10, 10)
    assert not within_limit(-1, 10)
    assert not within_limit(11, 10)
    assert within_limit(0, 0)

def test_added_sequences():
    assert first_page([1, 2, 3], 2) == [1, 2]
    assert first_page([1], 3) == [1]
    assert first_page([], 2) == []
    assert first_page([1], 0) == []
    assert first_page([1, 2, 3], 1 << 100) == [1, 2, 3]
    values = [3, 1, 3, 2]
    assert unique_ids(values) == [1, 2, 3]
    assert values == [3, 1, 3, 2]
    assert unique_ids([]) == []
"""
        if variant == "local":
            after["stats.py"] = after["stats.py"].replace("    if not values:\n        return 0.0\n", "")
        elif variant == "cross":
            after["tokens.py"] = after["tokens.py"].replace("expires_at_ms / 1000, now_ms / 1000",
                                                                         "expires_at_ms, now_ms / 1000")
        api_description = (
            "Convenience helpers include total for numeric batches, remaining_seconds for nonnegative time-to-expiry, "
            "normalized_token and format_label for trimming strings, within_limit for inclusive bounds, first_page "
            "for leading slices, and unique_ids for sorted duplicate-free IDs. "
            "Existing helper contracts remain unchanged."
        )
    else:
        for files in [before, after]:
            files["paging.go"] = files["paging.go"].replace(
                "\tif offset >= len(values) { return []int{} }; end := offset + limit\n"
                "\tif end > len(values) { end = len(values) }; return values[offset:end]\n",
                "\tif offset >= len(values) { return []int{} }\n"
                "\tremaining := len(values) - offset\n"
                "\tif limit > remaining { limit = remaining }\n"
                "\treturn values[offset:offset + limit]\n")
            files["expiry.go"] = files["expiry.go"].replace(
                "func IsExpired(expiresAtS, nowS int64) bool", "func IsExpired(expiresAtS, nowS float64) bool")
            files["tokens.go"] = files["tokens.go"].replace(
                "IsExpired(expiresAtMS / 1000, nowMS / 1000)",
                "IsExpired(float64(expiresAtMS) / 1000, float64(nowMS) / 1000)")
        # Share the ordinary total operation instead of repeating its loop.
        after["stats.go"] = after["stats.go"].replace(
            "\tvar sum float64\n\tfor _, value := range values { sum += value }\n"
            "\treturn sum / float64(len(values))", "\treturn Total(values) / float64(len(values))")
        before["legacy_test.go"] = (
            'package helpers\n'
            '\n'
            'import (\n'
            '    "reflect"\n'
            '    "testing"\n'
            ')\n'
            '\n'
            'func TestOriginalStatistics(t *testing.T) {\n'
            '    for _, values := range [][]float64{nil, {}, {2, 4}, {-2, 2}} {\n'
            '        want := 0.0\n'
            '        if len(values) == 2 && values[0] == 2 { want = 3 }\n'
            '        if got := Mean(values); got != want { t.Fatalf("Mean(%v)=%v want%v"'
            ', values, got, want) }\n'
            '    }\n'
            '}\n'
            'func TestOriginalExpiry(t *testing.T) {\n'
            '    for _, tc := range []struct{ expiry, now int64; valid bool }{\n'
            '        {5000,1000,true}, {5000,5000,false}, {5000,6000,false},\n'
            '        {1500,1000,true}, {1500,1499,true}, {1500,1500,false}, {1500,1501,f'
            'alse},\n'
            '    } {\n'
            '        if got := ValidToken(tc.expiry,tc.now); got != tc.valid { t.Fatalf('
            '"ValidToken(%d,%d)=%v want%v",tc.expiry,tc.now,got,tc.valid) }\n'
            '    }\n'
            '    if IsExpired(5,4) || !IsExpired(5,5) || !IsExpired(5,6) { t.Fatal("expi'
            'ry boundary") }\n'
            '}\n'
            'func TestOriginalHelpers(t *testing.T) {\n'
            '    if FormatCount(0)!="0" || FormatCount(12)!="12" { t.Fatal("count format'
            'ting") }\n'
            '    if Clamp(-1,10)!=0 || Clamp(5,10)!=5 || Clamp(11,10)!=10 || Clamp(0,0)!'
            '=0 { t.Fatal("clamp bounds") }\n'
            '    if !reflect.DeepEqual(Page([]int{1,2,3},1,2),[]int{2,3}) || len(Page([]'
            'int{1,2,3},4,2))!=0 || len(Page([]int{1},0,0))!=0 || len(Page(nil,0,2))!=0 '
            '{ t.Fatal("page edges") }\n'
            '    maxInt := int(^uint(0) >> 1)\n'
            '    if !reflect.DeepEqual(Page([]int{1,2,3},1,maxInt),[]int{2,3}) { t.Fatal'
            '("page limit overflow") }\n'
            '    values:=[]int{3,1,2}; got:=SortedIDs(values)\n'
            '    if !reflect.DeepEqual(got,[]int{1,2,3}) || !reflect.DeepEqual(values,[]'
            'int{3,1,2}) || len(SortedIDs(nil))!=0 { t.Fatal("sorted IDs or input mutati'
            'on") }\n'
            '}\n'
        )
        after["helpers_test.go"] = before["legacy_test.go"] + (
            '\n'
            'func TestAddedStatisticsAndExpiry(t *testing.T) {\n'
            '    if Total(nil)!=0 || Total([]float64{2,4})!=6 || Total([]float64{-2,2})!'
            '=0 { t.Fatal("totals") }\n'
            '    if RemainingSeconds(5,1)!=4 || RemainingSeconds(5,5)!=0 || RemainingSec'
            'onds(5,6)!=0 { t.Fatal("remaining seconds") }\n'
            '}\n'
            'func TestAddedFormattingAndBounds(t *testing.T) {\n'
            '    if !TokenPresent("token") || TokenPresent("") { t.Fatal("token presence'
            '") }\n'
            '    if FormatTotal(nil)!="0" || FormatTotal([]int{2,4})!="6" || FormatTotal'
            '([]int{-2,2})!="0" { t.Fatal("formatted totals") }\n'
            '    if !WithinLimit(0,10) || !WithinLimit(10,10) || WithinLimit(-1,10) || W'
            'ithinLimit(11,10) || !WithinLimit(0,0) { t.Fatal("inclusive bounds") }\n'
            '}\n'
            'func TestAddedSequences(t *testing.T) {\n'
            '    if !reflect.DeepEqual(FirstPage([]int{1,2,3},2),[]int{1,2}) || !reflect'
            '.DeepEqual(FirstPage([]int{1},3),[]int{1}) || len(FirstPage(nil,2))!=0 || l'
            'en(FirstPage([]int{1},0))!=0 { t.Fatal("first page") }\n'
            '    maxInt := int(^uint(0) >> 1)\n'
            '    if !reflect.DeepEqual(FirstPage([]int{1,2,3},maxInt),[]int{1,2,3}) { t.'
            'Fatal("first page huge limit") }\n'
            '    values:=[]int{3,1,2};got,ok:=Smallest(values)\n'
            '    if got!=1 || !ok || !reflect.DeepEqual(values,[]int{3,1,2}) { t.Fatal("'
            'smallest or input mutation") }\n'
            '    if _,ok:=Smallest(nil);ok { t.Fatal("empty smallest") }\n'
            '}\n'
        )
        if variant == "local":
            after["stats.go"] = after["stats.go"].replace("\tif len(values) == 0 { return 0 }\n", "")
        elif variant == "cross":
            after["tokens.go"] = after["tokens.go"].replace(
                "float64(expiresAtMS) / 1000, float64(nowMS) / 1000",
                "float64(expiresAtMS), float64(nowMS) / 1000")
        api_description = (
            "Convenience helpers include Total for numeric batches, RemainingSeconds for nonnegative time-to-expiry, "
            "TokenPresent for nonempty token strings, FormatTotal for count batches, WithinLimit for inclusive bounds, "
            "FirstPage for leading slices, and Smallest for nonmutating minimum lookup. "
            "Existing helper contracts remain unchanged."
        )
    after["README.md"] = before["README.md"] + "\n" + api_description + "\n"
    return before, after


def prepare(repo: Path, language: str, variant: str) -> None:
    repo.mkdir(parents=True, exist_ok=False)
    git(repo, "init", "-q", "-b", "main")
    before, after = fixture_files(language, variant)
    for path, text in before.items():
        (repo / path).write_text(text)
    git(repo, "add", *sorted(before))
    git(repo, "commit", "-S", "-qm", "test: establish shared helper contracts")
    git(repo, "checkout", "-qb", "feature/convenience-helpers")
    for path in before.keys() - after.keys():
        git(repo, "rm", path)
    for path, text in after.items():
        (repo / path).write_text(text)
    git(repo, "add", *sorted(after))
    git(repo, "commit", "-S", "-qm", "feat: add shared convenience helpers")


async def measure(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    from daydream import runner
    from daydream.artifacts.models import PrivateRootLocations
    from daydream.backends import Backend, effective_fanout_concurrency
    from daydream.review_budget import review_scale_for_diff
    from daydream.review_profile import build_default_profile
    from daydream.run_config import RunConfig

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = output / "repo"
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(args.fixture.resolve()), str(repo)], check=True)
    fixture_head = git(repo, "rev-parse", "HEAD")
    # A fixture may already be checked out on main. Keep its HEAD and branch
    # untouched, and give patch capture and the runner the same private base ref.
    base_ref = "refs/daydream/measurement-base"
    git(repo, "update-ref", base_ref, args.base_commit or "refs/remotes/origin/main")
    # Unlike scalar Git answers, a patch must retain its final newline for identity binding.
    patch = subprocess.check_output(["git", "-C", str(repo), "diff", f"{base_ref}...HEAD"]).decode()
    fixture_base = git(repo, "merge-base", base_ref, fixture_head)
    source_digest = hashlib.sha256()
    for path in sorted((source / "daydream").rglob("*.py")):
        source_digest.update(str(path.relative_to(source)).encode())
        source_digest.update(path.read_bytes())
    config = RunConfig(
        target=str(repo), base=base_ref, backend="pi", model=args.model, reasoning_effort=args.effort,
        latency_profile="balanced", output_mode="review", diagram="off", non_interactive=True,
        review_cache_enabled=False, archive=False, run_eval=False, dataset_capture_disabled=True,
        trajectory_path=output / "trajectory.json", cleanup=False,
        deep_shard_enabled=False if args.mode == "off" else None,
        deep_shard_fanout_cap=8 if args.mode == "cap8" else None,
    )
    os.environ["PI_PROVIDER"] = args.provider
    constructed_backends: list[dict[str, Any]] = []
    create_backend: Callable[..., Backend] = getattr(runner, "create_backend")

    def measured_backend(*backend_args: Any, **backend_kwargs: Any) -> Backend:
        backend = create_backend(*backend_args, **backend_kwargs)
        constructed_backends.append({
            "model": backend.model,
            "reasoning_effort": getattr(backend, "reasoning_effort", None),
            "fanout_hint": getattr(backend, "fanout_concurrency", None),
            "review_capacity": effective_fanout_concurrency(10, backend),
        })
        return backend

    setattr(runner, "create_backend", measured_backend)
    started = time.monotonic()
    failure = None
    exit_code = None
    try:
        exit_code = await runner.run(
            config, private_roots=PrivateRootLocations(output / "private", output / "workspaces"),
        )
    except Exception as exc:
        failure = type(exc).__name__
    finally:
        setattr(runner, "create_backend", create_backend)
    elapsed = time.monotonic() - started
    deep = repo / ".daydream" / "deep"
    coverage_path = deep / "review-coverage.json"
    coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else None
    patch_digest = hashlib.sha256(patch.encode()).hexdigest()
    assert git(repo, "rev-parse", "HEAD") == fixture_head
    assert not git(repo, "diff", "HEAD", "--name-only"), "review changed tracked fixture source"
    if coverage is not None:
        assert coverage["analyzed_revision"]["head_sha"] == fixture_head
        assert coverage["analyzed_revision"]["merge_base_sha"] == fixture_base
        assert coverage["analyzed_revision"]["diff_key"] == patch_digest
    trajectory_path = output / "trajectory.json"
    trajectory = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else {}
    trajectories = [trajectory]
    trajectories.extend(json.loads(path.read_text()) for path in (repo / ".daydream" / "runs").rglob("*.json")
                        if "trajectories" in path.parts)
    steps = [step for item in trajectories for step in item.get("steps", [])]
    scope_count = len(coverage["planned_scopes"]) if coverage else 0
    scale = review_scale_for_diff(patch)
    report: dict[str, Any] = {
        "mode": args.mode, "fixture_head": fixture_head, "fixture_base": fixture_base,
        "diff_sha256": patch_digest, "diff_bytes": len(patch.encode()), "diff_lines": patch.count("\n"),
        "diff_scale": scale,
        "source_head": git(source, "rev-parse", "HEAD"), "source_sha256": source_digest.hexdigest(),
        "model": args.model, "provider": args.provider, "effort": args.effort, "latency_profile": "balanced",
        "constructed_backends": constructed_backends,
        "profile_digest": build_default_profile().digest, "cache": False, "diagram": "off", "elapsed_s": elapsed,
        "exit_code": exit_code, "failure_type": failure, "coverage": coverage,
        "scope_count": scope_count,
        "allowances": {"per_scope_investigation_s": 480 * scale, "per_scope_tool_starts": 48 * scale,
                       "aggregate_scope_investigation_s": scope_count * 480 * scale,
                       "aggregate_scope_tool_starts": scope_count * 48 * scale,
                       "shared_pipeline_s": 2700 * scale},
        "final_metrics": trajectory.get("final_metrics"),
        "model_requests": sum(step.get("llm_call_count", 0) for step in steps),
        "tool_starts": sum(len(step.get("tool_calls", [])) for step in steps),
        # Invocation exceptions and host stops are different from failed tool calls.
        # Coverage classifies terminal scope failures; retries may have earlier exceptions.
        "invocation_errors": [{"phase": step.get("extra", {}).get("daydream_phase"),
                               "error_subtype": step.get("extra", {}).get("error_subtype")}
                              for step in steps if step.get("extra", {}).get("error")],
        "host_stops": [step["extra"]["stop_reason"] for step in steps
                       if step.get("extra", {}).get("stop_reason")],
        "scope_failures": [{"scope_id": scope["scope_id"], "status": scope["status"],
                            "reason_codes": scope["reason_codes"]}
                           for scope in coverage["stack_outcomes"] if scope["status"] != "complete"]
                          if coverage else None,
        "unsuccessful_phases": [{"phase": event["phase"], "status": event["status"],
                                 "reason_code": event.get("reason_code")}
                                for item in trajectories for event in item.get("extra", {}).get("phase_events", [])
                                if event.get("event") == "phase_end" and event["status"] != "succeeded"],
        "quality_adjudication": "pending independent source/evidence review",
    }
    items = deep / "merged-items.json"
    report["items"] = json.loads(items.read_text()) if items.exists() else None
    (output / "measurement.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("mode", "elapsed_s", "exit_code", "failure_type", "scope_count")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--language", choices=("python", "go"))
    parser.add_argument("--variant", choices=("clean", "local", "cross"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--base-commit", help="Exact frozen merge base; otherwise use the fixture's origin/main")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("off", "auto", "cap8"), default="auto")
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--effort", default="medium")
    args = parser.parse_args()
    if args.prepare:
        if not args.language or not args.variant:
            parser.error("--prepare requires --language and --variant")
        prepare(args.prepare.resolve(), args.language, args.variant)
    elif args.source and args.fixture and args.output:
        asyncio.run(measure(args))
    else:
        parser.error("a trial requires --source, --fixture, and --output")


if __name__ == "__main__":
    main()
