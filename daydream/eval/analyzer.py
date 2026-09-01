"""Quantitative analysis of ATIF v1.7 trajectory data from daydream runs.

Parses trajectory JSON files and deep review artifacts, computes metrics
across cost, tool usage, file coverage, finding quality, grounding, and
training signal dimensions. Output is a JSON-serializable report for archive
inspection and downstream training analysis.

Usage::

    from daydream.eval.analyzer import analyze_session
    report = analyze_session(Path("/path/to/.daydream"))
"""

import json
import math
import re
import shlex
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from daydream._tree_sitter_safety import TreeSitterBadVersionError, assert_tree_sitter_safe
from daydream.generated_files import is_generated_file
from daydream.timeutil import parse_iso_timestamp

# Trajectory loading

def _latest_main_trajectory(daydream_dir: Path) -> Path | None:
    """Return the most recent main trajectory by mtime, or None.

    New layout: ``runs/<session_id>/trajectory.json``.
    """
    candidates = list(daydream_dir.glob("runs/*/trajectory.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_dir_trajectory_paths(run_dir: Path) -> list[Path]:
    """``trajectory.json`` plus sorted ``trajectories/*.json`` directly under *run_dir*."""
    paths: list[Path] = []
    main_path = run_dir / "trajectory.json"
    if main_path.is_file():
        paths.append(main_path)
    siblings_dir = run_dir / "trajectories"
    if siblings_dir.is_dir():
        paths.extend(f for f in sorted(siblings_dir.glob("*.json")) if f.is_file())
    return paths


def collect_trajectory_paths(run_dir: Path) -> list[Path]:
    """Collect trajectory files for *run_dir*, main trajectory first.

    Looks for ``trajectory.json`` plus ``trajectories/*.json`` directly under
    *run_dir*; when neither exists, falls back to the most recently modified
    ``runs/<session_id>/`` run beneath it.
    """
    paths = _run_dir_trajectory_paths(run_dir)
    if not paths:
        latest = _latest_main_trajectory(run_dir)
        if latest:
            paths = _run_dir_trajectory_paths(latest.parent)
    return paths


def load_trajectories(daydream_dir: Path, session_id: str | None = None) -> dict[str, Any]:
    """Load trajectory files for a single session from a .daydream directory.

    New layout: ``runs/<session_id>/trajectory.json`` plus
    ``runs/<session_id>/trajectories/<descriptor>.json``.

    Args:
        daydream_dir: Path to the ``.daydream`` directory.
        session_id: Optional session ID (or prefix) to filter to. When None,
            the most recent main trajectory is used.

    Returns:
        Dict with ``main`` (root trajectory or None) and ``forked`` (list of
        subagent trajectories belonging to the same session).
    """
    main = None
    forked: list[dict[str, Any]] = []
    runs_dir = daydream_dir / "runs"

    # --- Resolve the run directory ------------------------------------------
    run_dir: Path | None = None
    if session_id:
        # Exact match first, then prefix match on run directory names
        exact = runs_dir / session_id
        if exact.is_dir():
            run_dir = exact
        elif runs_dir.is_dir():
            matches = sorted(
                d for d in runs_dir.iterdir()
                if d.is_dir() and d.name.startswith(session_id)
            )
            if len(matches) == 1:
                run_dir = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"Session prefix '{session_id}' matches multiple runs")
    else:
        latest = _latest_main_trajectory(daydream_dir)
        if latest:
            # latest is runs/<session_id>/trajectory.json — parent is the run dir
            run_dir = latest.parent

    # --- Resolve the main and forked trajectories ---------------------------
    if run_dir:
        for path in _run_dir_trajectory_paths(run_dir):
            data = json.loads(path.read_text())
            data["_source_file"] = path.name
            if path.name == "trajectory.json":
                main = data
            else:
                forked.append(data)

    return {"main": main, "forked": forked}


def _agent_label(filename: str) -> str:
    """Human-readable label from trajectory filename.

    New layout::

        ``trajectory.json`` → ``main``
        ``deep-python.json`` → ``deep-python``

    Legacy layout::

        ``trajectory-20260429-121816-5f0088a9.json`` → ``main``
        ``10073f9b.deep-python.json`` → ``deep-python``
    """
    if filename.startswith("trajectory"):
        return "main"
    parts = filename.rsplit(".", 2)
    if len(parts) >= 3:
        return parts[1]
    return filename.replace(".json", "")


def _extract_tool_calls(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        for tc in step.get("tool_calls") or []:
            calls.append({
                "step_id": step["step_id"],
                "function_name": tc["function_name"],
                "arguments": tc.get("arguments", {}),
                "phase": (step.get("extra") or {}).get("daydream_phase", "unknown"),
            })
    return calls


def _files_from_diff(diff_path: Path) -> list[str]:
    if not diff_path.exists():
        return []
    text = diff_path.read_text()
    files: set[str] = set()
    for m in re.finditer(r"^diff --git a/.+? b/(.+?)$", text, re.MULTILINE):
        files.add(m.group(1))
    return sorted(files)


_READ_VERBS = ("sed", "nl", "cat", "rg", "grep", "head", "tail", "awk", "wc")
_IMPORT_ONLY_ALTERNATIVE_RE = re.compile(r"^\^?(?:from\b|import\b)")


def _is_import_only_pattern(pattern: str) -> bool:
    """Whether a ``Grep`` pattern references only module imports, no content.

    Returns ``False`` for an empty/absent pattern (an import-only rule must
    never gate a pathless Grep call). Otherwise ``True`` iff every ``|``-
    separated alternative, stripped, matches the anchor-or-bare ``from`` /
    ``import`` predicate — so ``^from|^import`` and ``from |import `` qualify,
    while any alternative naming content (a ``class ``/``def `` body or a
    symbol) makes it ``False``.
    """
    if not pattern:
        return False
    return all(
        _IMPORT_ONLY_ALTERNATIVE_RE.match(alt.strip())
        for alt in pattern.split("|")
    )


_SED_RANGE_RE = re.compile(r"^\d+(?:,\d*)?\$?p$")
_REDIRECT_RE = re.compile(r"^(\d*)([<>]+|&>)(.*)$")
_SEGMENT_SEPARATORS = frozenset(("&&", ";", "&"))
_RG_LONG_VALUE_OPTS = frozenset(
    {
        "context",
        "context-before",
        "context-after",
        "glob",
        "ignore-file",
        "engine",
        "regexp",
        "file",
        "type",
        "type-not",
        "type-add",
        "max-columns",
        "max-count",
        "max-filesize",
        "threads",
        "pre",
        "pre-glob",
        "replace",
        "sort",
        "sortr",
    }
)
_RG_SHORT_VALUE_OPTS = frozenset("ABCefgMmtTr")
_GREP_LONG_VALUE_OPTS = frozenset(
    {
        "context",
        "before-context",
        "after-context",
        "max-count",
        "regexp",
        "file",
        "include",
        "exclude",
    }
)
_GREP_SHORT_VALUE_OPTS = frozenset("efABC")


def _tokenize_command(command: str) -> list[str]:
    """Split a shell command into tokens, honoring quotes and separators.

    ``shlex`` preserves quoted operands as single tokens (``'my file.py'``
    stays intact) while punctuation mode keeps ``&&``/``;``/``&`` as distinct
    separator tokens even without surrounding whitespace.

    Malformed tool-call data (unbalanced quotes) must never crash the
    eval/archive pipeline; on ``ValueError`` we fall back to a whitespace
    split so recoverable path operands still surface, except recoverable
    operands adjacent to a separator, which the whitespace split glues to
    the separator and are lost (issue #327).
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return command.split()


def _option_info(
    tok: str,
    long_value_opts: frozenset[str],
    short_value_opts: frozenset[str],
) -> tuple[int, bool]:
    """How many tokens a ``rg``/``grep``-family option occupies, and whether it supplies the pattern.

    ``-C 3`` → (2, False); ``--glob=*.py`` → (1, False) (value attached);
    ``-n`` → (1, False); ``-e PAT``/``--regexp=PAT`` → (…, True) because an
    explicit pattern leaves the next positional operand as a path, not a
    pattern. Combined short flags (``-ni``) skip only their own token; a
    value-taking short flag with an attached value (``-C3``) also consumes one.

    The value sets are the verb's own — ``_RG_{LONG,SHORT}_VALUE_OPTS`` for
    ``rg`` and ``_GREP_{LONG,SHORT}_VALUE_OPTS`` for ``grep`` — so each verb
    consumes exactly the options that take a value in its own table. Which
    options supply the search pattern is shared across both: ``--regexp`` and
    ``--file`` long, ``-e`` and ``-f`` short.
    """
    if tok.startswith("--"):
        if "=" in tok:
            name = tok[2:].split("=", 1)[0]
            return 1, name in ("regexp", "file")
        name = tok[2:]
        return (2 if name in long_value_opts else 1), name in ("regexp", "file")
    body = tok[1:]
    if body and body[0] in short_value_opts:
        return (1 if len(body) > 1 else 2), body[0] in ("e", "f")
    return 1, False


def _rg_option_info(tok: str) -> tuple[int, bool]:
    """How many tokens an ``rg`` option occupies, and whether it supplies the pattern."""
    return _option_info(tok, _RG_LONG_VALUE_OPTS, _RG_SHORT_VALUE_OPTS)


def _grep_option_info(tok: str) -> tuple[int, bool]:
    """How many tokens a ``grep`` option occupies, and whether it supplies the pattern.

    Uses grep's own value sets (``--context``/``--before-context``/``--max-count``/…
    long; ``-e``/``-f``/``-A``/``-B``/``-C`` short) so only grep's value-taking
    options consume a following token.
    """
    return _option_info(tok, _GREP_LONG_VALUE_OPTS, _GREP_SHORT_VALUE_OPTS)


def _read_paths_for_segment(verb: str, operands: list[str]) -> set[str]:
    """File-path operands of one command segment for a given read verb.

    Redirection operators are consumed together with their targets (separated
    ``> target`` or attached ``2>/dev/null``) so a redirect target is never
    recorded as a read. Sed address ranges and flags are filtered, ``rg``
    skips option values plus the search pattern, and the inspection verbs
    behave likewise: ``grep`` skips option values and its first positional
    (the pattern) and credits no operand when that pattern is import-only
    (issue #739), ``awk`` skips options and its first positional (the
    program), ``head``/``tail`` skip options and their ``-n``/``-c`` values,
    and ``wc`` skips options. ``cat`` operands pass through verbatim.
    """
    paths: set[str] = set()
    i = 0
    n = len(operands)
    seen_pattern = False
    seen_program = False
    after_ddash = False
    while i < n:
        tok = operands[i]
        m = _REDIRECT_RE.match(tok)
        if m:
            i += 1 if m.group(3) else 2
            continue
        # ``rg`` and ``grep`` share the same shape: ``--`` flips to literal
        # operand parsing, a leading ``-`` (unless ``-`` itself) is an option
        # whose consumed tokens + pattern-supplying status come from the verb's
        # own option table, and the first remaining positional is the pattern.
        if verb in ("rg", "grep"):
            option_info = _rg_option_info if verb == "rg" else _grep_option_info
            if tok == "--":
                after_ddash = True
                i += 1
                continue
            if not after_ddash and tok.startswith("-") and tok != "-":
                skip, supplies_pattern = option_info(tok)
                i += skip
                if supplies_pattern:
                    seen_pattern = True
                continue
            if not seen_pattern:
                seen_pattern = True
                if verb == "grep" and _is_import_only_pattern(tok):
                    # A grep whose pattern is only an import anchor is a
                    # module-import scan, not a content read (issue #739):
                    # crediting its operands as read paths would mis-credit
                    # AC2/AC3 import coverage, exactly like the Grep-tool
                    # branch's carve-out on the same shared seam.
                    return paths
                i += 1
                continue
        elif verb in ("head", "tail"):
            if tok.startswith("-"):
                body = tok[1:]
                if body and body[0] in ("n", "c"):
                    i += 1 if len(body) > 1 else 2
                else:
                    i += 1
                continue
        elif verb in ("awk", "wc"):
            if tok.startswith("-"):
                i += 1
                continue
            if verb == "awk" and not seen_program:
                seen_program = True
                i += 1
                continue
        elif verb in ("sed", "nl", "cat"):
            if tok.startswith("-"):
                i += 1
                continue
            if verb == "sed" and _SED_RANGE_RE.fullmatch(tok):
                i += 1
                continue
        paths.add(tok)
        i += 1
    return paths


def _paths_from_command(command: str) -> set[str]:
    """Extract file-path operands from a codex ``shell`` / pi ``bash`` command.

    Reviewers read files through these verbs: ``sed -n '1,240p'``, ``nl -ba``,
    ``cat``, ``rg``, ``grep``, ``head``, ``tail``, ``awk``, and ``wc``.
    Commands may chain segments with ``&&``/``;`` and redirect with
    ``2>/dev/null``. Tokenization is shell-aware: quoted paths survive,
    redirection targets are consumed with their operator, and option values
    plus the ``rg``/``grep`` search pattern and the ``awk`` program are
    filtered. Extraction is deliberately permissive — ``_path_matches``
    matches by ``endswith``, so a stray operand simply never matches a diff
    file — but options, redirect targets, sed address ranges, and the
    ``rg``/``grep`` search patterns are filtered.
    """
    paths: set[str] = set()
    tokens = _tokenize_command(command)
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok in _SEGMENT_SEPARATORS:
            i += 1
            continue
        m = _REDIRECT_RE.match(tok)
        if m:
            i += 1 if m.group(3) else 2
            continue
        verb = tok.split("/")[-1]
        if verb not in _READ_VERBS:
            i += 1
            continue
        i += 1
        operands: list[str] = []
        while i < n and tokens[i] not in _SEGMENT_SEPARATORS:
            operands.append(tokens[i])
            i += 1
        paths.update(_read_paths_for_segment(verb, operands))
    return paths


def _read_paths_for_call(tc: dict[str, Any]) -> list[str]:
    """All file paths a single tool call reads, across backends.

    ``function_name`` is case-folded so ``Bash`` (Claude) / ``bash`` (pi) /
    ``shell`` (codex) all route to the shell-command parser, and Claude's
    ``Read`` / pi's ``read`` collapse into one branch that accepts either the
    ``file_path`` or ``path`` argument key.

    - claude: ``Read`` → ``arguments.file_path``; ``Grep`` → ``arguments.path``
      (credited only when the pattern is not import-only)
    - pi:     lowercase ``read`` → ``arguments.path``
    - codex/pi: ``shell``/``bash`` → paths embedded in ``arguments.command``
    """
    fn = tc["function_name"].lower()
    args = tc["arguments"]
    if fn == "read":
        p = args.get("file_path") or args.get("path", "")
        return [p] if p else []
    if fn == "grep":
        p = args.get("path", "")
        if not p:
            return []
        pattern = args.get("pattern", "")
        if pattern and _is_import_only_pattern(pattern):
            return []
        return [p]
    if fn in ("shell", "bash"):
        return sorted(_paths_from_command(args.get("command", "")))
    return []


def _files_read(tool_calls: list[dict[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for tc in tool_calls:
        paths.update(_read_paths_for_call(tc))
    return paths


def _path_matches(absolute: str, relative: str) -> bool:
    """Check if an absolute tool-call path corresponds to a relative diff path."""
    return absolute.endswith(relative) or absolute.endswith("/" + relative)


# Analysis functions

def _all_trajectories(trajectories: dict[str, Any]) -> list[dict[str, Any]]:
    all_trajs: list[dict[str, Any]] = []
    if trajectories["main"]:
        all_trajs.append(trajectories["main"])
    all_trajs.extend(trajectories["forked"])
    return all_trajs


def analyze_costs(trajectories: dict[str, Any]) -> dict[str, Any]:
    """Cost and token breakdown across all agents.

    Run totals come from the root trajectory's ``final_metrics`` alone: the
    recorder folds each fork's totals into the parent at fork close, so the
    root is already whole-run truth and re-summing the sibling files would
    double-count. ``by_agent`` rows keep fork-level detail by listing each
    fork separately, with cost and token rows excluding their direct children.
    Step counts remain document-local.

    ``cached_tokens`` is a subset of ``prompt_tokens``, so
    ``total_input_tokens`` is the prompt-token total without adding the cache
    hit subset again.
    """
    agents: list[dict[str, Any]] = []
    forked = trajectories.get("forked") or []
    forks_by_source = {fork["_source_file"]: fork for fork in forked}
    for traj in _all_trajectories(trajectories):
        label = _agent_label(traj["_source_file"])
        fm = traj.get("final_metrics") or {}
        child_sources = {
            Path(ref).name
            for subtrajectory in (traj.get("extra") or {}).get("subtrajectories") or []
            if isinstance(subtrajectory, dict)
            for ref in [subtrajectory.get("sibling_trajectory_ref")]
            if isinstance(ref, str)
        }
        direct_forks = [forks_by_source[source] for source in child_sources if source in forks_by_source]
        fold_cost = sum((f.get("final_metrics") or {}).get("total_cost_usd") or 0.0 for f in direct_forks)
        fold_prompt = sum((f.get("final_metrics") or {}).get("total_prompt_tokens") or 0 for f in direct_forks)
        fold_completion = sum((f.get("final_metrics") or {}).get("total_completion_tokens") or 0 for f in direct_forks)
        fold_cached = sum((f.get("final_metrics") or {}).get("total_cached_tokens") or 0 for f in direct_forks)
        agents.append({
            "agent": label,
            "cost_usd": (fm.get("total_cost_usd") or 0.0) - fold_cost,
            "prompt_tokens": (fm.get("total_prompt_tokens") or 0) - fold_prompt,
            "completion_tokens": (fm.get("total_completion_tokens") or 0) - fold_completion,
            "cached_tokens": (fm.get("total_cached_tokens") or 0) - fold_cached,
            "steps": fm.get("total_steps") or len(traj.get("steps", [])),
            "model": traj.get("agent", {}).get("model_name", "unknown"),
        })

    # Marked roots have fork-inclusive final_metrics. Older trajectories lack
    # the marker and retain their legacy per-file aggregation.
    root = trajectories.get("main")
    root_metrics_include_forks = (
        ((root or {}).get("final_metrics") or {}).get("extra") or {}
    ).get("daydream_metric_scope") == "whole_run_including_forks"
    if root and root_metrics_include_forks:
        root_fm = root.get("final_metrics") or {}
        total_cost = root_fm.get("total_cost_usd") or 0.0
        total_prompt = root_fm.get("total_prompt_tokens") or 0
        total_completion = root_fm.get("total_completion_tokens") or 0
        total_cached = root_fm.get("total_cached_tokens") or 0
    else:
        total_cost = sum(a["cost_usd"] for a in agents)
        total_prompt = sum(a["prompt_tokens"] for a in agents)
        total_completion = sum(a["completion_tokens"] for a in agents)
        total_cached = sum(a["cached_tokens"] for a in agents)

    total_input = total_prompt
    cache_hit_rate = total_cached / total_input if total_input > 0 else 0.0

    return {
        "total_cost_usd": total_cost,
        "total_input_tokens": total_input,
        "total_prompt_tokens_raw": total_prompt,
        "total_completion_tokens": total_completion,
        "total_cached_tokens": total_cached,
        "cache_hit_rate": round(cache_hit_rate, 4),
        "by_agent": sorted(agents, key=lambda a: a["cost_usd"], reverse=True),
    }


def analyze_tools(trajectories: dict[str, Any]) -> dict[str, Any]:
    """Tool call counts, per-agent breakdown, and redundancy detection."""
    total_counts: Counter[str] = Counter()
    by_agent: dict[str, dict[str, Any]] = {}
    redundant_reads: list[dict[str, Any]] = []

    for traj in _all_trajectories(trajectories):
        label = _agent_label(traj["_source_file"])
        calls = _extract_tool_calls(traj)
        counts = Counter(tc["function_name"] for tc in calls)
        by_agent[label] = dict(counts)
        total_counts.update(counts)

        read_paths = [
            tc["arguments"].get("file_path", "")
            for tc in calls
            if tc["function_name"] == "Read"
        ]
        for path, count in Counter(read_paths).items():
            if count > 1 and path:
                redundant_reads.append({"agent": label, "file": path, "read_count": count})

    total = sum(total_counts.values())
    write_count = total_counts.get("Write", 0)

    return {
        "total_calls": total,
        "by_type": dict(total_counts.most_common()),
        "by_agent": by_agent,
        "write_ratio": round(write_count / total, 4) if total > 0 else 0,
        "redundant_reads": redundant_reads,
    }


def analyze_coverage(trajectories: dict[str, Any], daydream_dir: Path) -> dict[str, Any]:
    """File review coverage: files in diff vs files read by review agents."""
    diff_files = _files_from_diff(daydream_dir / "diff.patch")

    review_reads: set[str] = set()
    for traj in trajectories["forked"]:
        label = _agent_label(traj["_source_file"])
        if label.startswith("deep-"):
            review_reads.update(_files_read(_extract_tool_calls(traj)))

    if trajectories["main"]:
        for tc in _extract_tool_calls(trajectories["main"]):
            if tc["phase"] in ("deep", "alternatives"):
                review_reads.update(_read_paths_for_call(tc))

    covered = {df for df in diff_files if any(_path_matches(r, df) for r in review_reads)}
    uncovered = sorted(set(diff_files) - covered)

    return {
        "files_in_diff": len(diff_files),
        "files_read_by_reviewers": len(covered),
        "coverage_ratio": round(len(covered) / len(diff_files), 4) if diff_files else 1.0,
        "uncovered_files": uncovered,
    }


def _records_issues(records: Any) -> list[Any] | None:
    """Normalize a loaded per-stack records file to its bare issues list.

    Issue #742: fresh-run per-stack records files carry the dict shape
    ``{"issues": [...], "verdicts": [...]}``; legacy and primed fixtures use
    the bare list. Returns the issues list when ``records`` is a bare list or
    a dict whose ``issues`` is a list; ``None`` when it is neither (callers
    keep their own fail-open / warn-and-continue handling for non-list
    shapes). This is the canonical records-shape normalization shared by
    every per-stack records reader (coverage sweep, merge resume, analyzer,
    phases, and the test harness); a future shape change lands here only.
    """
    if isinstance(records, dict):
        issues = records.get("issues")
        return issues if isinstance(issues, list) else None
    return records if isinstance(records, list) else None


def _records_issues_or_empty(records: Any) -> list[Any]:
    """Normalize a loaded per-stack records file to a bare issues list.

    Collapses the ``None`` -> ``[]`` fallback idiom that every per-stack
    records reader previously re-implemented verbatim (orchestrator merge
    resume, analyzer findings, and the test harness). A non-list load yields
    the same degenerate value as the callers' explicit fallback
    (``[]`` for a dict, otherwise the raw load), preserving prior
    warn-and-continue semantics exactly.
    """
    issues = _records_issues(records)
    if issues is None:
        return [] if isinstance(records, dict) else records
    return issues


_EMPTY_PER_LENS = {"wonder": 0, "per-stack": 0, "uncovered": 0, "structure": 0}


def _bucketed_lens_counts(deep_dir: Path) -> dict[str, int]:
    """Attribute findings across the wonder / per-stack / uncovered / structure lenses.

    ``wonder`` reads the bare-list ``alternatives.json`` (the canonical wonder
    artifact); the existing ``stack-*-records.json`` glob buckets into
    per-stack / uncovered / structure by stack name. These lens counts are raw,
    pre-merge attribution -- they are *not* derived from the shipped
    ``merged-items.json`` set, so they need not sum to, or relate to, the
    shipped ``total`` reported by ``_shipped_counts``. The distinction is
    deliberate and documented: reconciling a lens to the shipped set would hide
    how many items survived merge/dedup, so report readers should treat the
    lens as raw attribution rather than a partition of the shipped set.
    """
    per_lens = dict(_EMPTY_PER_LENS)
    alts_path = deep_dir / "alternatives.json"
    if alts_path.exists():
        try:
            alternatives = json.loads(alts_path.read_text())
        except json.JSONDecodeError:
            # A present-but-malformed alternatives.json must not take down
            # analyze_findings / analyze_session; leave wonder attribution at 0.
            alternatives = None
        if isinstance(alternatives, list):
            per_lens["wonder"] = len(alternatives)
    for f in sorted(deep_dir.glob("stack-*-records.json")):
        stack_name = f.stem.replace("stack-", "").replace("-records", "")
        records = _records_issues_or_empty(json.loads(f.read_text()))
        if stack_name == "uncovered":
            per_lens["uncovered"] += len(records)
        elif stack_name == "structure":
            per_lens["structure"] += len(records)
        else:
            per_lens["per-stack"] += len(records)
    return per_lens


def _shipped_counts(
    deep_dir: Path, all_findings: list[dict[str, Any]], merged_review: dict[str, Any]
) -> tuple[int, dict[str, int]]:
    """Select the shipped review set and report (total, by_confidence) together.

    ``merged-items.json`` is authoritative when present (present-but-empty means
    "shipped nothing"); every item it carries is shipped, including wonder-lens
    findings (the renderer emits them, so they count -- issue #741). When that
    file is absent the count falls back to the ``merged_finding_count`` regex on
    ``review-output.md``; when neither exists it falls back to the pre-merge
    per-stack total so archived runs never regress to zero. ``by_confidence``
    is in every branch derived from the artifact that supplies ``total``.

    A present-but-corrupt ``merged-items.json`` is a data-integrity error that
    propagates rather than being silently hidden behind the fallback: a
    *syntax*-invalid file surfaces ``JSONDecodeError`` from ``json.loads``, and a
    well-formed file with the wrong shape (top-level non-object, or an ``items``
    field that is not a list) raises ``ValueError`` from the explicit shape
    check. Both are the same documented error class -- a bogus shipped set is
    never silently counted.
    """
    merged_items_file = deep_dir / "merged-items.json"
    if merged_items_file.exists():
        merged = json.loads(merged_items_file.read_text())
        # Shape-check before use so a well-formed-but-wrong-shape file (top-level
        # non-object, or ``items`` not a list) is treated as the documented
        # data-integrity error instead of silently yielding a bogus count. The
        # writer always emits the ``items`` key, so a missing ``items`` is a
        # wrong shape too -- only ``items": []`` is "shipped nothing".
        if not isinstance(merged, dict) or not isinstance(merged.get("items"), list):
            raise ValueError(
                "merged-items.json must be an object whose ``items`` is a list; "
                f"got top-level type {type(merged).__name__}"
            )
        merged_items = merged["items"]
        # Every merged item is shipped: the renderer now emits wonder-lens
        # findings too (issue #741), so the shipped set equals the posted
        # review. A present-but-empty list means "shipped nothing".
        by_confidence = dict(
            Counter(i.get("confidence", "UNKNOWN") for i in merged_items)
        )
        return len(merged_items), by_confidence
    if merged_review.get("merged_finding_count"):
        # The review is the shipped rendering but carries no confidence values,
        # so no confidence attribution is possible without another source. Return
        # an empty rather than cross-mix the pre-merge per-stack confidences.
        return merged_review["merged_finding_count"], {}
    return len(all_findings), dict(
        Counter(f.get("confidence", "UNKNOWN") for f in all_findings)
    )


def analyze_findings(daydream_dir: Path) -> dict[str, Any]:
    """Parse per-stack records, dedup stats, and merged review.

    ``total``/``by_confidence`` report the shipped review set: ``merged-items.json``
    is authoritative when present (it is the canonical set the posted review is
    rendered from). When that file is absent, the count falls back to the
    ``merged_finding_count`` regex on ``review-output.md``, then to the pre-merge
    per-stack total so archived runs never regress to zero. ``per_lens`` attributes
    findings across the wonder (``alternatives.json``), per-stack, uncovered, and
    structure lenses.
    """
    deep_dir = daydream_dir / "deep"
    if not deep_dir.is_dir():
        return {
            "total": 0,
            "by_confidence": {},
            "findings": [],
            "stacks": [],
            "dedup": {},
            "merged_review": {},
            "per_lens": dict(_EMPTY_PER_LENS),
        }

    all_findings: list[dict[str, Any]] = []
    stacks: list[dict[str, Any]] = []

    per_lens = _bucketed_lens_counts(deep_dir)

    for f in sorted(deep_dir.glob("stack-*-records.json")):
        stack_name = f.stem.replace("stack-", "").replace("-records", "")
        records = _records_issues_or_empty(json.loads(f.read_text()))
        stacks.append({"name": stack_name, "finding_count": len(records)})
        for r in records:
            r["_stack"] = stack_name
            all_findings.append(r)

    dedup_stats: dict[str, Any] = {}
    dedup_path = deep_dir / "dedup-candidates.json"
    if dedup_path.exists():
        dedup = json.loads(dedup_path.read_text())
        pairs = dedup.get("record_alt_pairs", [])
        dupes = dedup.get("record_duplicate_pairs", [])
        avg_sim = sum(p.get("similarity", 0) for p in pairs) / len(pairs) if pairs else 0
        dedup_stats = {
            "record_alt_overlaps": len(pairs),
            "record_duplicates": len(dupes),
            "avg_overlap_similarity": round(avg_sim, 4),
        }

    merged_review: dict[str, Any] = {}
    review_path = deep_dir / "review-output.md"
    if review_path.exists():
        text = review_path.read_text()
        merged_review["merged_finding_count"] = len(
            re.findall(r"^\d+\.\s+\[", text, re.MULTILINE)
        )

    total, by_confidence = _shipped_counts(deep_dir, all_findings, merged_review)

    return {
        "total": total,
        "by_confidence": by_confidence,
        "findings": all_findings,
        "stacks": stacks,
        "dedup": dedup_stats,
        "merged_review": merged_review,
        "per_lens": per_lens,
    }


def analyze_grounding(trajectories: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Tier 1 grounding: verify cited files were actually read by the agent.

    The denominator is the pre-merge per-stack finding list held in
    ``findings_data["findings"]`` -- those are the records tagged with
    ``_stack`` to match against a deep-stack reader -- not the shipped ``total``
    from ``merged-items.json``. Shipped wonder/structural items are absent from
    this set because they have no per-stack reader stream to ground to, so they
    neither inflate the denominator nor get grounding credit here.

    ``grounding_rate`` is ``None`` over an empty finding set: the ratio is
    undefined, not perfect. It feeds the RL/SFT reward as a credit axis
    (``daydream/training/reward.py``), where scoring absence of evidence as 1.0
    made "report nothing" the cheapest path to a maximal composite.
    """
    agent_reads: dict[str, set[str]] = {}
    for traj in trajectories["forked"]:
        label = _agent_label(traj["_source_file"])
        agent_reads[label] = _files_read(_extract_tool_calls(traj))

    grounded: list[dict[str, Any]] = []
    ungrounded: list[dict[str, Any]] = []

    for finding in findings:
        stack = finding.get("_stack", "")
        reads = agent_reads.get(f"deep-{stack}", set())

        cited_file = finding.get("file", "")
        rationale = finding.get("rationale", "")

        file_was_read = any(_path_matches(r, cited_file) for r in reads)

        rationale_refs = re.findall(r"[\w/.:-]+\.(?:md|json|py|ts|tsx|js|txt|yaml|yml|in|toml|cfg)", rationale)
        unread_refs = [
            ref for ref in rationale_refs
            if not any(_path_matches(r, ref) for r in reads)
        ]

        entry = {
            "id": finding.get("id"),
            "stack": stack,
            "file": cited_file,
            "confidence": finding.get("confidence", "UNKNOWN"),
            "file_was_read": file_was_read,
            "unread_rationale_refs": unread_refs,
            "grounded": file_was_read and len(unread_refs) == 0,
        }
        (grounded if entry["grounded"] else ungrounded).append(entry)

    total = len(findings)
    return {
        "total_findings": total,
        "grounded_count": len(grounded),
        "ungrounded_count": len(ungrounded),
        "grounding_rate": round(len(grounded) / total, 4) if total > 0 else None,
        "grounded": grounded,
        "ungrounded": ungrounded,
    }


def analyze_exploration_utilization(trajectories: dict[str, Any]) -> dict[str, Any]:
    """Check whether review agents read the exploration-path artifacts.

    Any tool call contributing a read path beneath an ``exploration/`` directory
    (e.g. the deterministic ``affected_files.md`` index, ``summary.md``, or
    ``conventions.md``) counts as exploration utilization, whichever backend
    performed the read — ``Read``/``Grep`` tool calls and ``shell``/``bash``
    commands alike route through ``_read_paths_for_call``.
    """
    results: list[dict[str, Any]] = []

    for traj in trajectories["forked"]:
        label = _agent_label(traj["_source_file"])
        if not label.startswith("deep-"):
            continue

        calls = _extract_tool_calls(traj)
        exploration_refs: list[str] = []
        total_reads = 0

        for tc in calls:
            read_paths = _read_paths_for_call(tc)
            if not read_paths:
                continue
            total_reads += 1
            for path in read_paths:
                if "/exploration/" in path:
                    exploration_refs.append(path)

        results.append({
            "agent": label,
            "total_reads": total_reads,
            "exploration_reads": len(exploration_refs),
            "utilized": len(exploration_refs) > 0,
        })

    utilized = sum(1 for r in results if r["utilized"])
    total_reviewers = len(results)

    return {
        "reviewers_utilizing_exploration": utilized,
        "total_reviewers": total_reviewers,
        "utilization_rate": round(utilized / total_reviewers, 4) if total_reviewers > 0 else 0,
        "by_agent": results,
    }


def analyze_timing(trajectories: dict[str, Any]) -> dict[str, Any]:
    """Wall-clock timing from step timestamps."""
    all_timestamps: list[datetime] = []
    agent_timings: list[dict[str, Any]] = []

    for traj in _all_trajectories(trajectories):
        label = _agent_label(traj["_source_file"])
        ts_list = [
            parse_iso_timestamp(s["timestamp"])
            for s in traj.get("steps", [])
            if s.get("timestamp")
        ]
        if len(ts_list) >= 2:
            duration = (ts_list[-1] - ts_list[0]).total_seconds()
            agent_timings.append({"agent": label, "duration_seconds": round(duration, 1)})
        all_timestamps.extend(ts_list)

    total_duration = 0.0
    if len(all_timestamps) >= 2:
        total_duration = (max(all_timestamps) - min(all_timestamps)).total_seconds()

    return {
        "total_wall_clock_seconds": round(total_duration, 1),
        "by_agent": sorted(agent_timings, key=lambda a: a["duration_seconds"], reverse=True),
    }


def analyze_training_signals(
    trajectories: dict[str, Any],
    findings: list[dict[str, Any]],
    grounding: dict[str, Any],
) -> dict[str, Any]:
    """Assess trajectory quality for ML training purposes."""
    signals: list[dict[str, Any]] = []

    for traj in trajectories["forked"]:
        label = _agent_label(traj["_source_file"])
        steps = traj.get("steps", [])

        has_reasoning = any(s.get("reasoning_content") for s in steps)
        total_tool_calls = sum(len(s.get("tool_calls") or []) for s in steps)

        # Reasoning token fraction (approximation from char length)
        reasoning_chars = sum(len(s.get("reasoning_content") or "") for s in steps)
        message_chars = sum(
            len(s["message"]) for s in steps
            if s.get("source") == "agent" and isinstance(s.get("message"), str)
        )
        total_output_chars = reasoning_chars + message_chars
        reasoning_fraction = (
            round(reasoning_chars / total_output_chars, 4)
            if total_output_chars > 0
            else 0
        )

        noise_flags: list[str] = []
        for s in steps:
            obs = s.get("observation")
            if obs:
                for r in obs.get("results", []):
                    content = r.get("content", "")
                    if isinstance(content, str) and content.strip() == "":
                        noise_flags.append("empty_tool_result")
                        break

        # Extract stack name from agent label (e.g. "deep-python" → "python")
        agent_stack = label.removeprefix("deep-").removeprefix("explore-")
        agent_ungrounded = [
            g for g in grounding.get("ungrounded", [])
            if g.get("stack", "") == agent_stack
        ]
        if agent_ungrounded:
            noise_flags.append(f"ungrounded_findings:{len(agent_ungrounded)}")

        signals.append({
            "trajectory": label,
            "source_file": traj["_source_file"],
            "steps": len(steps),
            "has_reasoning": has_reasoning,
            "tool_calls": total_tool_calls,
            "reasoning_fraction": reasoning_fraction,
            "noise_flags": noise_flags,
            "training_quality": "clean" if not noise_flags else "review",
        })

    clean = sum(1 for s in signals if s["training_quality"] == "clean")

    return {
        "total_trajectories": len(signals),
        "clean_for_training": clean,
        "needs_review": len(signals) - clean,
        "trajectories": signals,
    }


# Quality metrics (issue #316) ----------------------------------------------

_QUALITY_EXCLUDED_DIRS = frozenset(
    {
        ".git", ".daydream", "node_modules", ".venv", "venv", "__pycache__", ".worktrees",
        "dist", "build", "vendor", "third_party", "migrations",
        # ``atif`` is daydream/atif, explicitly vendored from Harbor
        # (see daydream/atif/NOTICE) — out of metric scope (Finding #5).
        "atif",
    }
)

# Node types that add cyclomatic complexity (tree-sitter-python). ``else_clause``
# is intentionally absent: an else adds no new path.
_CC_DECISION_TYPES = frozenset(
    {
        "if_statement",
        "elif_clause",
        "for_statement",
        "while_statement",
        "except_clause",
        "with_statement",
        "assert_statement",
        "conditional_expression",
        "boolean_operator",
        "case_clause",
        "for_in_clause",
    }
)

_COMPREHENSION_TYPES = frozenset(
    {
        "list_comprehension",
        "set_comprehension",
        "dictionary_comprehension",
        "generator_expression",
    }
)

_MIN_CLONE_BLOCK = 3
_MAX_CLONE_BLOCK = 20

_QUALITY_CALIBRATION = {
    "human_verbosity": 0.19,
    "human_erosion": 0.34,
    "paper": "arXiv:2603.24755",
}


@lru_cache(maxsize=1)
def _quality_python_parser() -> Any | None:
    """Cached tree-sitter Python parser.

    Lazy factory mirroring ``daydream/tree_sitter_index.py``; returns ``None``
    when tree-sitter-python is unavailable so quality analysis degrades to an
    empty shape instead of crashing ``analyze_session``.  Consults the shared
    version guard first: a known-bad installed tree-sitter (issue #1087) raises
    :class:`daydream._tree_sitter_safety.TreeSitterBadVersionError` out of the
    factory instead of constructing a parser that would corrupt memory — the
    orchestrator's fail-open wrapper then reports the quality gate explicitly
    unavailable instead of silently skipping it.
    """
    try:
        assert_tree_sitter_safe()
        import tree_sitter_python
        from tree_sitter import Language, Parser

        return Parser(Language(tree_sitter_python.language()))
    except TreeSitterBadVersionError:
        raise
    except Exception:
        return None


def _iter_tree(node: Any) -> Iterator[Any]:
    """Yield *node* and all descendants in a deterministic depth-first order."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _iter_function(func: Any) -> Iterator[Any]:
    """Yield *func* and its descendants, skipping nested function definitions.

    Nested ``def`` nodes are yielded but their subtrees are not descended into,
    because each nested def is measured as its own function.
    """
    stack = [func]
    while stack:
        node = stack.pop()
        yield node
        if node is not func and node.type == "function_definition":
            continue
        stack.extend(reversed(node.children))


def _is_comprehension_filter(node: Any) -> bool:
    """An ``if_clause`` attached to a comprehension, not a match-case guard."""
    parent = node.parent
    if parent is None:
        return False
    if parent.type in _COMPREHENSION_TYPES:
        return True
    grandparent = parent.parent
    return grandparent is not None and grandparent.type in _COMPREHENSION_TYPES


def _is_wildcard_case(node: Any) -> bool:
    """A ``case _:`` clause (no guard) — matches any value, adds no path."""
    if node.type != "case_clause":
        return False
    if node.child_by_field_name("guard") is not None:
        return False
    pattern = next((c for c in node.children if c.type == "case_pattern"), None)
    return pattern is not None and pattern.text.decode().strip() == "_"


def _count_decision_nodes(node: Any) -> int:
    """Cyclomatic decision count of a subtree, skipping nested functions."""
    total = 1 if node.type in _CC_DECISION_TYPES else 0
    if node.type == "case_clause" and _is_wildcard_case(node):
        total -= 1
    elif node.type == "if_clause" and _is_comprehension_filter(node):
        total += 1
    for child in node.children:
        if child.type == "function_definition":
            continue
        total += _count_decision_nodes(child)
    return total


def _scoped_python_files(
    workspace: Path, need_lines: bool = False
) -> list[tuple[Path, str, list[str]]]:
    """All ``*.py`` files under *workspace* minus excluded dirs, as ``(path, rel, lines)``.

    Excluded directories are matched on exact path components so a nested
    ``node_modules_extra/`` or a sibling ``.worktrees`` checkout never shadows
    real source. Vendored subtrees with non-obvious basenames (``atif``, see
    ``daydream/atif/NOTICE``) are excluded by name too. Generated files are
    also out of scope: path-based rules
    (``*_generated.py``, ``*.pb.py``, ``migrations/*.py``, …) and the
    generated-file header marker are both applied via ``is_generated_file``,
    so vendor/third_party trees and generated artifacts never reach the
    metric denominators (Finding #8).

    The third element is the file's raw decoded lines, decoded from the same
    ``content`` bytes already read for ``is_generated_file`` with the same
    ``utf-8`` + ``errors="replace"`` decode ``_parse_python_file`` uses, so
    candidate-scoped analysis (issue #457) can index a non-candidate peer's
    text for cross-file clone attribution without parsing it. It is only
    populated when *need_lines* is set (candidate-scoped analysis); the default
    whole-workspace path never consumes it, so the redundant decode+splitlines
    is skipped there and no dead third element is materialized.
    """
    files: list[tuple[Path, str, list[str]]] = []
    for path in workspace.rglob("*.py"):
        try:
            rel = path.relative_to(workspace)
        except ValueError:
            continue
        if any(part in _QUALITY_EXCLUDED_DIRS for part in rel.parts):
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if is_generated_file(str(rel), content):
            continue
        if need_lines:
            raw_lines = content.decode("utf-8", errors="replace").splitlines()
        else:
            raw_lines = []
        files.append((path, str(rel), raw_lines))
    return sorted(files, key=lambda item: item[1])


def _parse_python_file(path: Path) -> tuple[Any, list[str]] | None:
    """Parse *path* once; return ``(root, lines)`` or ``None`` on any failure.

    A file that cannot be parsed is counted in ``scoped_files`` by the caller
    but excluded from the aggregates — quality analysis never crashes.
    Tree-sitter returns a PARTIAL tree with ERROR/missing nodes for malformed
    Python, so a root whose tree carries any error is treated as a parse
    failure instead of aggregating garbage (Finding #9).
    """
    parser = _quality_python_parser()
    if parser is None:
        return None
    try:
        source = path.read_bytes()
    except OSError:
        return None
    try:
        tree = parser.parse(source)
    except Exception:
        return None
    if tree is None:
        return None
    root = tree.root_node
    if root.has_error:
        return None
    lines = source.decode("utf-8", errors="replace").splitlines()
    return root, lines


def _file_quality_from_tree(
    root: Any,
    lines: list[str],
    cross_file_flagged: set[int] | None = None,
) -> dict[str, Any]:
    """Erosion + verbosity metrics from an already-parsed file.

    *cross_file_flagged* carries line rows (0-based) that duplicate a block also
    present in another scoped file; they are OR'd into the verbosity set before
    the ratio is computed (Finding #6).

    Per-file ratios are ``None`` when their denominator is zero: ``erosion``
    with no functions, ``verbosity`` with no non-blank lines. The numeric mass
    and line counts are still returned so the caller can pool them for the
    workspace aggregate (Finding #2).
    """
    # Erosion: pooled cyclomatic mass of functions with CC > 10.
    functions = [node for node in _iter_tree(root) if node.type == "function_definition"]
    metrics: list[tuple[int, int, float]] = []  # (cc, sloc, mass)
    for func in functions:
        body = func.child_by_field_name("body")
        cc = 1 + _count_decision_nodes(body) if body is not None else 1
        sloc = func.end_point.row - func.start_point.row + 1
        if sloc < 1:
            continue
        metrics.append((cc, sloc, cc * math.sqrt(sloc)))

    total_mass = sum(mass for _, _, mass in metrics)
    high_mass = sum(mass for cc, _, mass in metrics if cc > 10)
    erosion = high_mass / total_mass if total_mass > 0 else None

    # Verbosity: deterministic rule subset + clone detection over lines.
    sloc_file = sum(1 for line in lines if line.strip())
    flagged = _verbosity_flagged_lines(root)
    flagged |= _clone_flagged_lines(lines)
    if cross_file_flagged:
        flagged |= cross_file_flagged
    flagged &= {i for i, line in enumerate(lines) if line.strip()}
    verbosity = len(flagged) / sloc_file if sloc_file > 0 else None

    return {
        "entry": {
            "erosion": round(erosion, 4) if erosion is not None else None,
            "verbosity": round(verbosity, 4) if verbosity is not None else None,
            "sloc": sloc_file,
            "functions": len(metrics),
            "high_cc_functions": sum(1 for cc, _, _ in metrics if cc > 10),
        },
        "mass": total_mass,
        "high_mass": high_mass,
        "flagged": len(flagged),
        "loc": sloc_file,
    }


def _verbosity_flagged_lines(root: Any) -> set[int]:
    """Line rows (0-based) flagged by the deterministic taxonomy subset."""
    flagged: set[int] = set()
    flagged |= _identity_comprehension_lines(root)
    flagged |= _empty_list_guard_lines(root)
    flagged |= _single_use_variable_lines(root)
    flagged |= _trivial_wrapper_lines(root)
    flagged |= _nested_ladder_lines(root)
    return flagged


def _comprehension_body(node: Any) -> Any | None:
    """The output expression of a comprehension (skipping delimiters).

    A generator expression wraps its output expression in ``(``/``)``, so
    parens are skipped alongside the list/set/dict brackets.
    """
    for child in node.children:
        if child.type not in ("[", "]", "{", "}", "(", ")"):
            return child
    return None


def _identity_comprehension_lines(root: Any) -> set[int]:
    """``[x for x in items]``-style comprehensions whose output is the loop var.

    Only a single unfiltered generator whose body is exactly the generator
    target qualifies; a filter or extra generator means the comprehension
    does real work and is never flagged.
    """
    flagged: set[int] = set()
    for node in _iter_tree(root):
        if node.type not in _COMPREHENSION_TYPES:
            continue
        for_clauses = [child for child in node.children if child.type == "for_in_clause"]
        if len(for_clauses) != 1:
            continue
        target = for_clauses[0].child_by_field_name("left")
        if target is None or target.type != "identifier":
            continue
        has_filter = any(
            child.type == "if_clause"
            or (
                child.type == "for_in_clause"
                and any(grandchild.type == "if_clause" for grandchild in child.children)
            )
            for child in node.children
        )
        if has_filter:
            continue
        body = _comprehension_body(node)
        if body is not None and body.text == target.text:
            flagged.update(range(node.start_point.row, node.end_point.row + 1))
    return flagged


def _empty_guard_variable(if_node: Any) -> str | None:
    """The variable tested by ``len(x) == 0`` or ``not x``, else ``None``."""
    cond = if_node.child_by_field_name("condition")
    if cond is None:
        return None
    if cond.type == "not_operator":
        arg = cond.child_by_field_name("argument")
        if arg is not None and arg.type == "identifier":
            return str(arg.text.decode())
        return None
    if cond.type == "comparison_operator":
        call = None
        zero = None
        for child in cond.children:
            if child.type == "call":
                call = child
            elif child.type == "integer":
                zero = child
        if call is None or zero is None or zero.text.decode().strip() != "0":
            return None
        fn = call.child_by_field_name("function")
        args = call.child_by_field_name("arguments")
        if fn is None or fn.type != "identifier" or fn.text.decode() != "len" or args is None:
            return None
        arg_ids = [child for child in args.children if child.type == "identifier"]
        if len(arg_ids) != 1:
            return None
        return str(arg_ids[0].text.decode())
    return None


def _len_call_argument(node: Any) -> str | None:
    """The single argument name of a ``len(x)`` call, else ``None``."""
    if node is None or node.type != "call":
        return None
    fn = node.child_by_field_name("function")
    args = node.child_by_field_name("arguments")
    if fn is None or fn.type != "identifier" or fn.text.decode() != "len" or args is None:
        return None
    arg_ids = [child for child in args.children if child.type == "identifier"]
    if len(arg_ids) == 1:
        return str(arg_ids[0].text.decode())
    return None


def _empty_guard_collection(condition: Any) -> str | None:
    """The collection a ``while`` condition proves nonempty, else ``None``.

    Only exact shapes prove it: the bare collection (``while items:``), a
    bare ``len`` call (``while len(items):``), or a positive length
    comparison (``while len(items) > 0:``). A predicate that merely receives
    the collection (``while should_continue(items):``) does not, so the guard
    may be required.
    """
    if condition is None:
        return None
    if condition.type == "identifier":
        return str(condition.text.decode())
    name = _len_call_argument(condition)
    if name is not None:
        return name
    if condition.type == "comparison_operator":
        operands = list(condition.children)
        if len(operands) == 3 and operands[1].type == ">":
            left, _, right = operands
            if right.type == "integer" and right.text.decode().strip() == "0":
                return _len_call_argument(left)
    return None


def _statement_mutates(node: Any, name: str) -> bool:
    """May *node* mutate the collection *name*?

    A method call on the collection (``items.pop()``, ``items.clear()``,
    ``items.remove()``, …) or a reassignment of *name* invalidates the
    loop-header's nonemptiness proof, so a later empty guard is meaningful.
    """
    for sub in _iter_tree(node):
        if sub.type in ("assignment", "augmented_assignment"):
            target = sub.child_by_field_name("left")
            if target is not None and target.type == "identifier" and target.text.decode() == name:
                return True
        if sub.type == "call":
            fn = sub.child_by_field_name("function")
            if fn is not None and fn.type == "attribute":
                obj = fn.child_by_field_name("object")
                if obj is not None and obj.type == "identifier" and obj.text.decode() == name:
                    return True
    return False


def _empty_list_guard_lines(root: Any) -> set[int]:
    """``if len(x) == 0`` / ``if not x`` guard directly inside a loop over ``x``.

    The guard is redundant only at a program point where the loop header's
    nonemptiness proof still holds. A statement between the header and the
    guard that may mutate the collection (``items.pop()``, ``items.clear()``,
    reassignment) invalidates that proof, so the guard is a necessary
    post-mutation termination check and is not flagged (Finding #3).
    """
    flagged: set[int] = set()
    for node in _iter_tree(root):
        if node.type not in ("for_statement", "while_statement"):
            continue
        body = node.child_by_field_name("body")
        if body is None:
            continue
        if node.type == "for_statement":
            iterable = node.child_by_field_name("right")
            if iterable is None or iterable.type != "identifier":
                continue
            guarded = iterable.text.decode()
        else:
            guarded = _empty_guard_collection(node.child_by_field_name("condition"))
            if guarded is None:
                continue
        mutated = False
        for child in body.children:
            if child.type == "if_statement":
                guard_var = _empty_guard_variable(child)
                if guard_var == guarded and not mutated:
                    flagged.update(range(child.start_point.row, child.end_point.row + 1))
            if _statement_mutates(child, guarded):
                mutated = True
    return flagged


def _count_later_references(func: Any, name: str, after_row: int) -> int:
    """Identifier occurrences of *name* after *after_row* inside *func*."""
    return sum(
        1
        for node in _iter_function(func)
        if node.type == "identifier"
        and node.text.decode() == name
        and node.start_point.row > after_row
    )


def _single_use_variable_lines(root: Any) -> set[int]:
    """Assignments to a plain identifier referenced exactly once later in the fn.

    Tuple unpacking, attribute, and subscript targets are excluded; names
    starting with ``_`` are ignored.
    """
    flagged: set[int] = set()
    for func in _iter_tree(root):
        if func.type != "function_definition":
            continue
        for node in _iter_function(func):
            if node.type != "assignment":
                continue
            target = node.child_by_field_name("left")
            if target is None or target.type != "identifier":
                continue
            name = target.text.decode()
            if name.startswith("_"):
                continue
            if _count_later_references(func, name, node.start_point.row) == 1:
                flagged.add(node.start_point.row)
    return flagged


def _return_value(return_node: Any) -> Any | None:
    """The expression a ``return`` yields (the ``return`` keyword is a token)."""
    for child in return_node.children:
        if child.type != "return":
            return child
    return None


def _param_definitions(params: Any) -> list[tuple[str, bool]] | None:
    """Ordered ``(name, has_default)`` per parameter, or ``None`` if complex.

    ``None`` covers positional-only ``/``, keyword-only ``*``, ``*args``, and
    ``**kwargs``: a wrapper cannot forward those as plain positional names.
    """
    definitions: list[tuple[str, bool]] = []
    for child in params.children:
        if child.type in (",", "(", ")"):
            continue
        if child.type in (
            "/",
            "*",
            "**",
            "positional_separator",
            "keyword_separator",
            "list_splat",
            "dictionary_splat",
        ):
            return None
        if child.type == "identifier":
            definitions.append((child.text.decode(), False))
        elif child.type == "typed_parameter":
            name = next((c for c in child.children if c.type == "identifier"), None)
            if name is None:
                return None
            definitions.append((name.text.decode(), False))
        elif child.type in ("default_parameter", "typed_default_parameter"):
            name = child.child_by_field_name("name")
            if name is None or name.type != "identifier":
                return None
            definitions.append((name.text.decode(), True))
        else:
            return None
    return definitions


def _forwarded_argument_names(args: Any) -> list[str] | None:
    """Positional argument names of a call, or ``None`` if not plain identifiers.

    A literal, keyword, ``*args``/``**kwargs``, or any other non-identifier
    argument makes the call non-trivial (``None``).
    """
    names: list[str] = []
    for child in args.children:
        if child.type in (",", "(", ")"):
            continue
        if child.type == "identifier":
            names.append(child.text.decode())
        else:
            return None
    return names


def _is_docstring_statement(node: Any) -> bool:
    """An ``expression_statement`` whose whole content is a string literal."""
    if node.type != "expression_statement":
        return False
    contents = [child for child in node.children if child.type != ";"]
    return len(contents) == 1 and contents[0].type == "string"


def _trivial_wrapper(func: Any) -> set[int] | None:
    """Flag a function whose whole body is ``return other(same args)``.

    A leading function-docstring is not an executable statement, so a
    pass-through wrapper with a docstring is still a wrapper (Finding #4).
    """
    body = func.child_by_field_name("body")
    if body is None:
        return None
    statements = list(body.children)
    if statements and _is_docstring_statement(statements[0]):
        statements = statements[1:]
    if len(statements) != 1 or statements[0].type != "return_statement":
        return None
    value = _return_value(statements[0])
    if value is None or value.type != "call":
        return None
    fn_name = value.child_by_field_name("function")
    args = value.child_by_field_name("arguments")
    params = func.child_by_field_name("parameters")
    if fn_name is None or fn_name.type != "identifier" or args is None or params is None:
        return None
    param_defs = _param_definitions(params)
    arg_names = _forwarded_argument_names(args)
    if param_defs is None or arg_names is None:
        return None
    if len(param_defs) != len(arg_names):
        return None
    if any(has_default for _, has_default in param_defs):
        return None
    if [name for name, _ in param_defs] != arg_names:
        return None
    return set(range(func.start_point.row, func.end_point.row + 1))


def _trivial_wrapper_lines(root: Any) -> set[int]:
    flagged: set[int] = set()
    for func in _iter_tree(root):
        if func.type != "function_definition":
            continue
        lines = _trivial_wrapper(func)
        if lines is not None:
            flagged.update(lines)
    return flagged


def _direct_nested_ifs(if_node: Any) -> list[Any]:
    """Direct ``if`` children of *if_node*'s consequence/alternative blocks."""
    nested: list[Any] = []
    consequence = if_node.child_by_field_name("consequence")
    if consequence is not None:
        nested.extend(child for child in consequence.children if child.type == "if_statement")
    alternative = if_node.child_by_field_name("alternative")
    if alternative is not None and alternative.type in ("elif_clause", "else_clause"):
        block = alternative.child_by_field_name("consequence") or alternative.child_by_field_name("body")
        if block is not None:
            nested.extend(child for child in block.children if child.type == "if_statement")
    return nested


def _nested_ladder_lines(root: Any) -> set[int]:
    """Innermost ``if`` of any ≥3-deep directly-nested if ladder."""
    flagged: set[int] = set()

    def visit(if_node: Any, depth: int) -> None:
        nested = _direct_nested_ifs(if_node)
        if nested:
            for child in nested:
                visit(child, depth + 1)
        elif depth >= 3:
            flagged.update(range(if_node.start_point.row, if_node.end_point.row + 1))

    for node in _iter_tree(root):
        if node.type == "if_statement":
            visit(node, 1)
    return flagged


def _clone_flagged_lines(lines: list[str]) -> set[int]:
    """Line rows in ≥2 occurrences of an identical contiguous block (3..20 lines).

    Lines are normalized by stripping; blocks containing blank lines are
    skipped so whitespace runs are never counted as clones.
    """
    stripped = [line.strip() for line in lines]
    n = len(stripped)
    flagged: set[int] = set()
    for length in range(_MIN_CLONE_BLOCK, _MAX_CLONE_BLOCK + 1):
        if length > n:
            break
        by_first: dict[str, list[int]] = {}
        for i in range(n - length + 1):
            if any(not stripped[j] for j in range(i, i + length)):
                continue
            by_first.setdefault(stripped[i], []).append(i)
        for starts in by_first.values():
            if len(starts) < 2:
                continue
            by_block: dict[tuple[str, ...], list[int]] = {}
            for i in starts:
                by_block.setdefault(tuple(stripped[i : i + length]), []).append(i)
            for occurrences in by_block.values():
                if len(occurrences) < 2:
                    continue
                for i in occurrences:
                    flagged.update(range(i, i + length))
    return flagged


def _cross_file_clone_flagged_lines(
    file_lines: list[tuple[Path, list[str]]],
    target_paths: set[Path] | None = None,
) -> dict[Path, set[int]]:
    """Line rows whose block also appears verbatim in another scoped file.

    Same rule as ``_clone_flagged_lines`` (exact stripped contiguous block of
    3..20 lines; blank-containing blocks skipped) but the match set spans
    files: a block present in ≥2 distinct files flags every occurrence in
    every file that holds it. Within-file duplicates stay
    ``_clone_flagged_lines``'s job and are still counted per file, so the two
    passes compose — a block duplicated twice inside ``a.py`` and once in
    ``b.py`` is flagged in both, once per occurrence.

    *target_paths* (issue #457) restricts which indexed files can be flagged:
    when provided, a block flags a path ONLY if that path is a target, while
    the ≥2-distinct-files gate still counts every indexed file (targets and
    peers) as a distinct source; blocks present in no target flag nothing. The
    returned dict still keys every indexed path (empty sets for non-flagged).
    ``None`` keeps the flag-every-file behavior.
    """
    stripped_lines = {path: [line.strip() for line in lines] for path, lines in file_lines}
    flagged: dict[Path, set[int]] = {path: set() for path in stripped_lines}
    target_set = target_paths if target_paths is not None else set(stripped_lines)

    for length in range(_MIN_CLONE_BLOCK, _MAX_CLONE_BLOCK + 1):
        by_first: dict[str, list[tuple[Path, int]]] = {}
        for path, stripped in stripped_lines.items():
            n = len(stripped)
            if length > n:
                continue
            for i in range(n - length + 1):
                if any(not stripped[j] for j in range(i, i + length)):
                    continue
                by_first.setdefault(stripped[i], []).append((path, i))
        for starts in by_first.values():
            if len({path for path, _ in starts}) < 2:
                continue
            by_block: dict[tuple[str, ...], list[tuple[Path, int]]] = {}
            for path, i in starts:
                by_block.setdefault(tuple(stripped_lines[path][i : i + length]), []).append((path, i))
            for occurrences in by_block.values():
                if len({path for path, _ in occurrences}) < 2:
                    continue
                for path, i in occurrences:
                    if path not in target_set:
                        continue
                    flagged[path].update(range(i, i + length))
    return flagged


def _aggregate_per_file(
    candidates: list[tuple[Path, str, list[str]]],
    parsed: dict[Path, Any],
    parsed_lines: dict[Path, list[str]],
    cross_file_flagged: dict[Path, set[int]],
) -> tuple[dict[str, dict[str, Any]], float, float, int, int]:
    """Terminal per-file aggregation over the parsed candidates.

    Walks the candidate list once, folding each successfully parsed file's
    quality entry and mass/flagged/loc totals; a candidate whose parse failed
    stays in ``scoped_files`` but contributes nothing here (Finding #1).
    Returns ``(per_file, total_mass, high_mass, total_flagged, total_loc)``.
    """
    per_file: dict[str, dict[str, Any]] = {}
    total_mass = 0.0
    high_mass = 0.0
    total_flagged = 0
    total_loc = 0
    for path, rel, _raw in candidates:
        root = parsed.get(path)
        if root is None:
            continue
        quality = _file_quality_from_tree(
            root,
            parsed_lines[path],
            cross_file_flagged=cross_file_flagged.get(path),
        )
        per_file[rel] = quality["entry"]
        total_mass += quality["mass"]
        high_mass += quality["high_mass"]
        total_flagged += quality["flagged"]
        total_loc += quality["loc"]
    return per_file, total_mass, high_mass, total_flagged, total_loc


def analyze_quality(
    daydream_dir: str | Path, candidate_paths: set[str] | None = None
) -> dict[str, Any]:
    """Structural erosion and verbosity of the post-fix workspace (issue #316).

    Computes SlopCodeBench-style metrics (arXiv:2603.24755) over every scoped
    ``*.py`` file in ``daydream_dir.parent`` — the live reviewed tree at eval
    time, after daydream's fix phase ran. Pure and deterministic: no backend,
    no network, no randomness. A file whose tree-sitter parse fails is counted
    in ``scoped_files`` but excluded from the aggregates and from the
    cross-file clone index, so malformed source never shifts valid files'
    verbosity (Finding #1).

    *candidate_paths* (issue #457) opt-in scopes the parse, per-file metrics,
    totals, and scoped-file count to the exact workspace-relative ``*.py``
    paths listed (a candidate that fails the eligibility rules is absent from
    ``_scoped_python_files`` and so is simply not a candidate). ``None`` (the
    default) preserves the whole-workspace behavior above. An explicitly empty
    set returns a zero-count empty report without enumerating the workspace.

    A candidate is still checked for cross-file clone duplication against every
    scoped peer, so a candidate's verbosity verdict still reflects clones in any
    peer file — candidate-scoped reporting does not weaken verdict meaning. A
    peer that fails to parse is excluded from the clone index (Finding #1 holds
    in candidate mode too), so malformed source never flags a valid candidate.

    Returns:
        ``{erosion, verbosity, per_file, calibration, scoped_files}``.
        ``erosion``/``verbosity`` are ``None`` only when no functions / no
        lines are in scope; per-file entries use ``None`` for a ratio whose
        denominator is zero (no functions / no non-blank lines). ``per_file``
        keys are workspace-relative paths.
    """
    daydream_dir = Path(daydream_dir)
    workspace = daydream_dir.parent

    # An explicitly empty candidate set is a zero-count empty report: never
    # walk (or parse) the workspace (issue #457).
    if candidate_paths is not None and not candidate_paths:
        return {
            "erosion": None,
            "verbosity": None,
            "per_file": {},
            "calibration": dict(_QUALITY_CALIBRATION),
            "scoped_files": 0,
        }

    scoped_files = _scoped_python_files(workspace, need_lines=candidate_paths is not None)
    if candidate_paths is None:
        candidates = scoped_files
        candidate_path_set: set[Path] | None = None
    else:
        # Exact workspace-relative match: a candidate that fails eligibility is
        # absent from ``scoped_files`` and so is not a candidate (issue #457).
        candidates = [t for t in scoped_files if t[1] in candidate_paths]
        candidate_path_set = {path for path, _rel, _raw in candidates}
    scoped = len(candidates)

    # Parse and validate each candidate ONCE. Only successfully parsed files
    # feed the per-file aggregates; a malformed file stays in ``scoped`` but is
    # omitted from the aggregates (Finding #1). In whole-workspace mode every
    # candidate is a scoped file (byte-identical to today). In candidate mode
    # non-candidate peers join the clone index via their raw decoded ``lines``
    # (3rd tuple element) without being parsed, so a candidate's cross-file
    # clone attribution still sees every peer (issue #457). The parse results
    # are reused below, so no file is ever parsed twice.
    parsed: dict[Path, Any] = {}
    parsed_lines: dict[Path, list[str]] = {}
    file_lines: list[tuple[Path, list[str]]] = []
    for path, _rel, raw_lines in scoped_files:
        if candidate_path_set is not None and path not in candidate_path_set:
            # Non-candidate peer: index it only if it parses, so a malformed
            # peer never enters the clone index and cannot flag a valid
            # candidate (Finding #1 holds in candidate mode too). The parsed
            # lines are byte-identical to the raw lines for a valid file, so
            # indexing is unchanged for well-formed peers.
            peer = _parse_python_file(path)
            if peer is None:
                continue
            file_lines.append((path, peer[1]))
            continue
        result = _parse_python_file(path)
        if result is None:
            continue
        root, lines = result
        parsed[path] = root
        parsed_lines[path] = lines
        file_lines.append((path, lines))

    # Cross-file clone pass over the index (parsed candidates + raw peers):
    # find blocks duplicated verbatim across >=2 files, then feed each target's
    # cross-file line rows into its per-file verbosity below (Finding #6). In
    # candidate mode only parsed candidate paths are flagged (``target_paths``),
    # so peers stay unmeasured while still contributing clone sources, and an
    # unparseable candidate stays out of both the index and the flag targets.
    # Within-file duplicates are still handled per file inside
    # _file_quality_from_tree.
    cross_file_flagged = _cross_file_clone_flagged_lines(
        file_lines,
        target_paths=set(parsed) if candidate_paths is not None else None,
    )

    per_file, total_mass, high_mass, total_flagged, total_loc = _aggregate_per_file(
        candidates, parsed, parsed_lines, cross_file_flagged
    )

    return {
        "erosion": round(high_mass / total_mass, 4) if total_mass > 0 else None,
        "verbosity": round(total_flagged / total_loc, 4) if total_loc > 0 else None,
        "per_file": per_file,
        "calibration": dict(_QUALITY_CALIBRATION),
        "scoped_files": scoped,
    }


# Top-level entry point

def analyze_session(daydream_dir: str | Path, session_id: str | None = None) -> dict[str, Any]:
    """Run full quantitative analysis on a .daydream directory.

    Args:
        daydream_dir: Path to the ``.daydream`` directory from a completed run.
        session_id: Optional session ID (or prefix) to analyze. Defaults to the
            most recent session.
    """
    daydream_dir = Path(daydream_dir)
    trajectories = load_trajectories(daydream_dir, session_id=session_id)

    if not trajectories["main"] and not trajectories["forked"]:
        return {"error": f"No trajectory files found in {daydream_dir}"}

    main = trajectories["main"] or trajectories["forked"][0]
    session_id = main.get("session_id", "unknown")
    agent_info = main.get("agent", {})

    # Extract PR metadata from trajectory extra (set by TrajectoryRecorder)
    traj_extra = main.get("extra") or {}
    pr_number = traj_extra.get("pr_number")
    pr_repo = traj_extra.get("pr_repo")

    costs = analyze_costs(trajectories)
    tools = analyze_tools(trajectories)
    findings_data = analyze_findings(daydream_dir)
    coverage = analyze_coverage(trajectories, daydream_dir)
    grounding = analyze_grounding(trajectories, findings_data["findings"])
    exploration = analyze_exploration_utilization(trajectories)
    timing = analyze_timing(trajectories)
    training = analyze_training_signals(
        trajectories, findings_data["findings"], grounding,
    )
    quality = analyze_quality(daydream_dir)

    finding_count = findings_data["total"]
    cost_per_finding = (
        round(costs["total_cost_usd"] / finding_count, 4)
        if finding_count > 0
        else None
    )

    result: dict[str, Any] = {
        "session_id": session_id,
        "agent": agent_info,
        "daydream_dir": str(daydream_dir),
        "trajectory_count": len(_all_trajectories(trajectories)),
        "cost": costs,
        "timing": timing,
        "tools": tools,
        "coverage": coverage,
        "findings": {
            "total": finding_count,
            "by_confidence": findings_data["by_confidence"],
            "stacks": findings_data["stacks"],
            "dedup": findings_data["dedup"],
            "merged_review": findings_data.get("merged_review", {}),
            "per_lens": findings_data.get("per_lens", {}),
        },
        "grounding": grounding,
        "exploration_utilization": exploration,
        "training_signals": training,
        "quality": quality,
        "derived": {
            "cost_per_finding_usd": cost_per_finding,
        },
    }

    if pr_number is not None or pr_repo is not None:
        result["pr"] = {"pr_number": pr_number, "pr_repo": pr_repo}

    return result
