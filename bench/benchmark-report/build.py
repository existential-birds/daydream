#!/usr/bin/env python3
"""Benchmark report generator: daydream vs the SaaS field on the code-review-benchmark.

Reads the benchmark corpus (read-only) ACROSS ALL JUDGE MODELS, recomputes every
SaaS tool on the SAME PR subset daydream covered (per judge), synthesizes daydream
cost from measured tokens, and emits a self-contained offline ``index.html``.

The judge model is NOT fixed: the generator DISCOVERS every
``<results>/<judge>/evaluations.json`` and normalizes the vendor prefix so the
harness dir (``claude-opus-4-5-20251101``) and the published SaaS-field dir
(``anthropic_claude-opus-4-5-20251101``) collapse to one judge. Re-run it against
any future corpus with a different judge, daydream label, or backend price card.

Usage:
    python3 build.py <RESULTS_ROOT> [--daydream-tool daydream-owl-alpha]
        [--exclude-tool daydream-glm] [--price-model glm-5.2]
        [--trajectories <dir>] [--pr-labels <file>] [--dashboard <file>]
        [--speed-analysis <file>] [--out <report_dir>]

Every figure is traceable to a source file; numbers that cannot be cited are dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ── Price cards (per 1M tokens). Mirrors bench/review-bot-compare/compare.py PRICING.
# Counters are DISJOINT: prompt = fresh input, cached = cache reads, completion = output.
PRICE_CARDS: dict[str, dict[str, float]] = {
    "glm-5.2": {"input": 1.40, "cached": 0.26, "output": 4.40},
}

# Judge-error guard from daydream/benchmark/score.py: above this ratio of failed
# comparisons the tp/fp/fn collapse to noise, not a real zero.
JUDGE_ERROR_RATIO_THRESHOLD = 0.5

# Vendor prefixes the published SaaS field carries; the harness writes daydream
# with no prefix. Stripping these collapses the two dirs onto one judge id.
_VENDOR_PREFIXES = ("anthropic_", "openai_", "google_", "xai_", "zai_")

# A judge needs at least this many distinct SaaS tools to anchor a comparison panel.
_MIN_SAAS_TOOLS_FOR_PANEL = 5


def normalize_judge(dir_name: str) -> str:
    """Collapse a results dir name to a canonical judge id by stripping the vendor prefix."""
    for p in _VENDOR_PREFIXES:
        if dir_name.startswith(p):
            return dir_name[len(p):]
    return dir_name


def judge_display(canon: str) -> str:
    """Human label for a canonical judge id."""
    c = canon.replace("claude-", "").replace("-20251101", "").replace("-20250929", "")
    pretty = {"opus-4-5": "Opus 4.5", "sonnet-4-5": "Sonnet 4.5", "gpt-5.2": "GPT-5.2", "gpt-4o-mini": "GPT-4o-mini"}
    for k, v in pretty.items():
        c = c.replace(k, v)
    return c


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def f1(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r)


def discover_judges(results_root: Path, exclude_tool: str) -> dict[str, dict[str, Any]]:
    """Discover every ``<judge>/evaluations.json`` and merge by normalized judge id.

    Returns canonical-judge-id -> {dirs, evals (PR -> tool -> leaf, exclude_tool dropped)}.
    Leaves from multiple physical dirs sharing a judge id are unioned per PR.
    """
    judges: dict[str, dict[str, Any]] = {}
    for d in sorted(results_root.iterdir()):
        ev_file = d / "evaluations.json"
        if not ev_file.is_file():
            continue
        canon = normalize_judge(d.name)
        raw = json.loads(ev_file.read_text())
        bucket = judges.setdefault(canon, {"dirs": [], "evals": {}})
        bucket["dirs"].append(str(ev_file))
        for pr_url, tools in raw.items():
            merged = bucket["evals"].setdefault(pr_url, {})
            for tool, leaf in tools.items():
                if tool == exclude_tool:
                    continue
                merged[tool] = leaf  # later dir wins on collision (none observed)
    return judges


def _leaf_present(leaf: dict[str, Any]) -> bool:
    return leaf is not None and not leaf.get("skipped", False)


def aggregate_tool(evals: dict[str, dict], tool: str, subset: set[str]) -> dict[str, Any] | None:
    """Micro-aggregate one tool over ``subset`` PRs. Mirrors score.parse_daydream_scores.

    Returns None when the tool has no present leaf anywhere in the subset.
    """
    tp = fp = fn = errors = comparisons = 0
    n = 0
    sample_fp: list[str] = []
    for pr in subset:
        leaf = evals.get(pr, {}).get(tool)
        if not _leaf_present(leaf):
            continue
        n += 1
        tp += int(leaf.get("tp", 0))
        fp += int(leaf.get("fp", 0))
        fn += int(leaf.get("fn", 0))
        errors += int(leaf.get("errors_count", 0))
        comparisons += int(leaf.get("total_candidates", 0)) * int(leaf.get("total_golden", 0))
        for c in (leaf.get("false_positives") or [])[:2]:
            txt = c.get("candidate") if isinstance(c, dict) else str(c)
            if txt and len(sample_fp) < 6:
                sample_fp.append(txt)
    if n == 0:
        return None
    invalid = bool(comparisons) and (errors / comparisons) >= JUDGE_ERROR_RATIO_THRESHOLD
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return {
        "tool": tool, "n_prs": n, "tp": tp, "fp": fp, "fn": fn,
        "errors": errors, "comparisons": comparisons, "invalid": invalid,
        "precision": p, "recall": r, "f1": f1(p, r),
        "fp_per_tp": _safe_div(fp, tp) if tp else (float("inf") if fp else 0.0),
        "sample_fp": sample_fp,
    }


def rank_of(rows: list[dict], target_tool: str, key: str) -> tuple[int, int]:
    """1-based rank of target_tool among valid rows by ``key`` (higher is better), and field size."""
    valid = [r for r in rows if not r["invalid"]]
    ordered = sorted(valid, key=lambda r: r[key], reverse=True)
    for i, r in enumerate(ordered, 1):
        if r["tool"] == target_tool:
            return i, len(ordered)
    return 0, len(ordered)


def load_trajectories(traj_dir: Path, price: dict[str, float], price_model: str) -> dict[str, dict]:
    """Map PR-url -> {tokens, synth cost, wall seconds, steps} from ATIF trajectories.

    Join key: (repo-last-segment, pr-number) parsed from the filename. Cost is
    SYNTHESIZED from measured tokens (the backend records $0 for GLM via z.ai).
    """
    out: dict[str, dict] = {}
    if not traj_dir.is_dir():
        return out
    for f in sorted(traj_dir.glob("*.json")):
        if f.name.endswith(".partial"):
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        stem = f.stem  # e.g. "cal.com-10600"
        repo_short, _, num = stem.rpartition("-")
        if not num.isdigit():
            continue
        fm = d.get("final_metrics", {}) or {}
        prompt = int(fm.get("total_prompt_tokens", 0))
        completion = int(fm.get("total_completion_tokens", 0))
        cached = int(fm.get("total_cached_tokens", 0))
        cost = prompt / 1e6 * price["input"] + cached / 1e6 * price["cached"] + completion / 1e6 * price["output"]
        events = (d.get("extra", {}) or {}).get("phase_events", []) or []
        stamps = [e.get("timestamp") for e in events if e.get("timestamp")]
        wall = None
        if len(stamps) >= 2:
            def _p(s: str) -> datetime:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            wall = (_p(max(stamps)) - _p(min(stamps))).total_seconds()
        out[f"{repo_short}/{num}"] = {
            "repo_short": repo_short, "pr_number": num,
            "pr_repo": (d.get("extra", {}) or {}).get("pr_repo", ""),
            "prompt_tokens": prompt, "completion_tokens": completion, "cached_tokens": cached,
            "cost_usd": cost, "wall_seconds": wall, "steps": int(fm.get("total_steps", 0)),
            "price_model": price_model,
        }
    return out


def traj_key_for_pr(pr_url: str) -> str:
    """(repo-last-segment, number) join key for a PR url, matching trajectory filenames."""
    # https://github.com/calcom/cal.com/pull/10600 -> "cal.com/10600"
    parts = pr_url.rstrip("/").split("/")
    num = parts[-1]
    repo_last = parts[-3] if len(parts) >= 3 else ""
    return f"{repo_last}/{num}"


def slice_daydream(evals: dict, subset: set[str], labels: dict, dd_tool: str, dim_path: tuple[str, str]) -> list[dict]:
    """Micro P/R/F1 for daydream grouped by one PR-label dimension."""
    section, field = dim_path
    groups: dict[str, dict[str, int]] = {}
    for pr in subset:
        leaf = evals.get(pr, {}).get(dd_tool)
        if not _leaf_present(leaf):
            continue
        lab = labels.get(pr, {}).get(section, {}).get(field)
        if lab is None:
            continue
        lab = str(lab)
        g = groups.setdefault(lab, {"tp": 0, "fp": 0, "fn": 0, "n": 0})
        g["tp"] += int(leaf.get("tp", 0))
        g["fp"] += int(leaf.get("fp", 0))
        g["fn"] += int(leaf.get("fn", 0))
        g["n"] += 1
    out = []
    for lab, g in groups.items():
        p = _safe_div(g["tp"], g["tp"] + g["fp"])
        r = _safe_div(g["tp"], g["tp"] + g["fn"])
        out.append({"label": lab, "n_prs": g["n"], "tp": g["tp"], "fp": g["fp"], "fn": g["fn"],
                    "precision": p, "recall": r, "f1": f1(p, r)})
    return sorted(out, key=lambda x: x["n_prs"], reverse=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    results_root = Path(args.results_root).resolve()
    dd_tool = args.daydream_tool
    price_model = args.price_model
    price = PRICE_CARDS.get(price_model)
    if price is None:
        raise SystemExit(f"unknown --price-model {price_model!r}; known: {', '.join(PRICE_CARDS)}")

    dashboard = json.loads(Path(args.dashboard).read_text()) if Path(args.dashboard).is_file() else {}
    display_names = dashboard.get("tool_display_names", {})
    tool_colors = dashboard.get("tool_colors", {})
    labels = json.loads(Path(args.pr_labels).read_text()) if Path(args.pr_labels).is_file() else {}

    judges_raw = discover_judges(results_root, args.exclude_tool)

    # The daydream-scored PR set is the fixed cross-judge anchor (daydream only ran on these).
    dd_subset: set[str] = set()
    dd_source_dir = ""
    for canon, b in judges_raw.items():
        prs = {pr for pr, t in b["evals"].items() if _leaf_present(t.get(dd_tool))}
        if len(prs) > len(dd_subset):
            dd_subset, dd_source_dir = prs, canon
    if not dd_subset:
        raise SystemExit(f"no judge has a present leaf for --daydream-tool {dd_tool!r}")

    trajectories = load_trajectories(Path(args.trajectories), price, price_model)
    speed = {}
    if args.speed_analysis and Path(args.speed_analysis).is_file():
        speed = json.loads(Path(args.speed_analysis).read_text())

    judges_out = []
    skipped_judges = []
    for canon in sorted(judges_raw):
        b = judges_raw[canon]
        evals = b["evals"]
        all_tools: set[str] = set()
        for t in evals.values():
            all_tools.update(t.keys())
        saas_tools = sorted(t for t in all_tools if t != dd_tool)
        if len(saas_tools) < _MIN_SAAS_TOOLS_FOR_PANEL:
            # Verify daydream-subset overlap so future re-judges still anchor cleanly.
            overlap = len(dd_subset & set(evals.keys()))
            skipped_judges.append({"id": canon, "dirs": b["dirs"], "saas_tools": len(saas_tools),
                                   "reason": "no SaaS field (superseded or partial run)",
                                   "daydream_overlap": overlap})
            continue

        overlap = dd_subset & set(evals.keys())
        has_dd = bool({pr for pr in dd_subset if _leaf_present(evals.get(pr, {}).get(dd_tool))})

        field = []
        for tool in saas_tools:
            agg = aggregate_tool(evals, tool, dd_subset)
            if agg is None:
                continue
            agg["display"] = display_names.get(tool, tool)
            agg["color"] = tool_colors.get(tool, "#5B7C99")
            field.append(agg)

        dd_agg = aggregate_tool(evals, dd_tool, dd_subset) if has_dd else None
        ranks = {}
        if dd_agg:
            dd_agg["display"] = "daydream (owl-alpha)"
            dd_agg["color"] = "#FD8017"
            combined = field + [dd_agg]
            ranks = {
                "precision": rank_of(combined, dd_tool, "precision"),
                "recall": rank_of(combined, dd_tool, "recall"),
                "f1": rank_of(combined, dd_tool, "f1"),
            }

        judges_out.append({
            "id": canon,
            "display": judge_display(canon),
            "dirs": b["dirs"],
            "subset_pr_count": len(overlap),
            "has_daydream": has_dd,
            "daydream": dd_agg,
            "ranks": ranks,
            "field": sorted(field, key=lambda r: r["f1"], reverse=True),
        })

    # ── daydream economy (judge-independent: it ran once under owl-alpha) ──
    anchor_judge = next((j for j in judges_out if j["has_daydream"]), None)
    anchor_evals = judges_raw[dd_source_dir]["evals"]
    per_pr_rows = []
    tot_prompt = tot_completion = tot_cached = 0
    tot_cost = 0.0
    walls = []
    for pr in sorted(dd_subset):
        leaf = anchor_evals.get(pr, {}).get(dd_tool, {})
        tj = trajectories.get(traj_key_for_pr(pr), {})
        lab = labels.get(pr, {})
        tp, fp, fn = int(leaf.get("tp", 0)), int(leaf.get("fp", 0)), int(leaf.get("fn", 0))
        cost = tj.get("cost_usd")
        if tj:
            tot_prompt += tj.get("prompt_tokens", 0)
            tot_completion += tj.get("completion_tokens", 0)
            tot_cached += tj.get("cached_tokens", 0)
            tot_cost += cost or 0.0
            if tj.get("wall_seconds") is not None:
                walls.append(tj["wall_seconds"])
        per_pr_rows.append({
            "pr_url": pr,
            "repo": "/".join(pr.split("/")[3:5]),
            "pr_number": pr.rstrip("/").split("/")[-1],
            "language": lab.get("derived", {}).get("language"),
            "domain": lab.get("llm_pr_labels", {}).get("domain"),
            "risk": lab.get("llm_pr_labels", {}).get("risk_level"),
            "change_type": lab.get("llm_pr_labels", {}).get("change_type"),
            "complexity": lab.get("llm_pr_labels", {}).get("code_complexity"),
            "golden": int(leaf.get("total_golden", 0)),
            "candidates": int(leaf.get("total_candidates", 0)),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": _safe_div(tp, tp + fp), "recall": _safe_div(tp, tp + fn),
            "cost_usd": cost,
            "wall_seconds": tj.get("wall_seconds"),
            "prompt_tokens": tj.get("prompt_tokens"),
            "completion_tokens": tj.get("completion_tokens"),
            "cached_tokens": tj.get("cached_tokens"),
            "steps": tj.get("steps"),
        })

    n_with_traj = sum(1 for r in per_pr_rows if r["cost_usd"] is not None)
    economy = {
        "price_model": price_model, "price_card": price,
        "n_prs": len(dd_subset), "n_with_trajectory": n_with_traj,
        "total_prompt_tokens": tot_prompt, "total_completion_tokens": tot_completion,
        "total_cached_tokens": tot_cached, "total_cost_usd": tot_cost,
        "cost_per_pr": _safe_div(tot_cost, n_with_traj) if n_with_traj else 0.0,
        "median_wall_seconds": (sorted(walls)[len(walls) // 2] if walls else None),
        "mean_wall_seconds": (sum(walls) / len(walls) if walls else None),
        "n_with_wall": len(walls),
    }

    # ── label slices for daydream (under the anchor judge) ──
    slice_dims = [
        ("Language", ("derived", "language")),
        ("Domain", ("llm_pr_labels", "domain")),
        ("Risk", ("llm_pr_labels", "risk_level")),
        ("Change type", ("llm_pr_labels", "change_type")),
        ("Complexity", ("llm_pr_labels", "code_complexity")),
    ]
    slices = []
    if anchor_judge:
        for title, dim in slice_dims:
            rows = slice_daydream(anchor_evals, dd_subset, labels, dd_tool, dim)
            if rows:
                slices.append({"title": title, "rows": rows})

    # ── SaaS latency (only if a speed_analysis.json was supplied; never fabricated) ──
    latency_field = []
    for tool, data in (speed or {}).items():
        st = data.get("stats")
        if st:
            latency_field.append({"tool": tool, "display": display_names.get(tool, tool),
                                  "color": tool_colors.get(tool, "#5B7C99"),
                                  "median_seconds": st["median_seconds"], "n": st["count"]})
    latency_field.sort(key=lambda r: r["median_seconds"])

    return {
        "meta": {
            "generated_from": str(results_root),
            "daydream_tool": dd_tool,
            "excluded_tool": args.exclude_tool,
            "anchor_judge": dd_source_dir,
            "subset_pr_count": len(dd_subset),
            "subset_prs": sorted(dd_subset),
            "judge_error_ratio_threshold": JUDGE_ERROR_RATIO_THRESHOLD,
            "metric_def": ("micro precision=ΣTP/(ΣTP+ΣFP), recall=ΣTP/(ΣTP+ΣFN), "
                           "F1=2PR/(P+R) — daydream/benchmark/score.py"),
            "price_source": "bench/review-bot-compare/compare.py PRICING",
            "speed_analysis_present": bool(speed),
            "total_daydream_attempts": 26,
        },
        "judges": judges_out,
        "skipped_judges": skipped_judges,
        "economy": economy,
        "per_pr": per_pr_rows,
        "slices": slices,
        "latency_field": latency_field,
    }


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_root", help="Path to the benchmark results/ dir (contains <judge>/evaluations.json)")
    ap.add_argument("--daydream-tool", default="daydream-owl-alpha")
    ap.add_argument("--exclude-tool", default="daydream-glm", help="Stale daydream label to drop")
    ap.add_argument("--price-model", default="glm-5.2", help=f"Price card; known: {', '.join(PRICE_CARDS)}")
    ap.add_argument("--trajectories", default="", help="Path to .daydream-bench/trajectories")
    ap.add_argument("--pr-labels", default="", help="Path to results/pr_labels.json")
    ap.add_argument("--dashboard", default="", help="Path to analysis/benchmark_dashboard.json")
    ap.add_argument("--speed-analysis", default="", help="Optional path to a speed_analysis.json (SaaS latency)")
    ap.add_argument("--out-root", default=str(here / "runs"),
                    help="Parent dir for per-run report folders (default: bench/benchmark-report/runs)")
    ap.add_argument("--run-id", default="",
                    help="Report folder name under --out-root (default: UTC timestamp + corpus fingerprint)")
    ap.add_argument("--out", default="",
                    help="Explicit full output dir; overrides --out-root/--run-id (writes exactly here)")
    args = ap.parse_args()

    # Sensible sibling-corpus defaults so a bare run just works.
    root = Path(args.results_root)
    if not args.trajectories:
        args.trajectories = str(root.parent / ".daydream-bench" / "trajectories")
    if not args.pr_labels:
        args.pr_labels = str(root / "pr_labels.json")
    if not args.dashboard:
        args.dashboard = str(root.parent / "analysis" / "benchmark_dashboard.json")

    data = build(args)

    # ── Per-run output folder — never overwrite a prior report. ──
    # The benchmark corpus is read-only, so reports are archived under daydream,
    # one self-contained folder per generation (mirrors osprey's <RUN>/report/).
    # run-id = UTC timestamp + a corpus fingerprint (judge dirs + scored PR set),
    # so repeated generations are distinct yet a re-run of the *same* corpus at the
    # same second lands on the same addressable folder.
    generated_at = datetime.now(UTC)
    fp_dirs = "|".join(sorted(d for j in data["judges"] for d in j["dirs"]))
    fp_src = fp_dirs + "||" + "|".join(data["meta"]["subset_prs"])
    fingerprint = hashlib.sha256(fp_src.encode()).hexdigest()[:8]
    run_id = args.run_id or f"{generated_at:%Y-%m-%d__%H-%M-%S}__{data['meta']['daydream_tool']}__{fingerprint}"
    data["meta"]["generated_at"] = generated_at.isoformat()
    data["meta"]["run_id"] = run_id
    data["meta"]["corpus_fingerprint"] = fingerprint

    out_dir = Path(args.out) if args.out else Path(args.out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.json").write_text(json.dumps(data, indent=2))

    template = (here / "template.html").read_text()
    html = template.replace("__DATA__", json.dumps(data))
    (out_dir / "index.html").write_text(html)
    # Each report is self-contained (offline file://): copy the vendored htmx beside it.
    htmx_src = here / "htmx.min.js"
    if htmx_src.is_file():
        shutil.copyfile(htmx_src, out_dir / "htmx.min.js")
    # A convenience 'latest' pointer to the freshest report (best-effort; never fatal).
    if not args.out:
        latest = Path(args.out_root) / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(out_dir.resolve(), target_is_directory=True)
        except OSError:
            pass

    # ── Stdout: cited headline aggregates per judge ──
    m = data["meta"]
    print(f"\n■ Benchmark report built → {out_dir/'index.html'}")
    print(f"  run-id: {run_id}  (generated {m['generated_at']})")
    print(f"  corpus: {m['generated_from']}  daydream tool: {m['daydream_tool']}  "
          f"(excluded stale: {m['excluded_tool']})")
    print(f"  metric: {m['metric_def']}")
    print(f"  anchor judge (daydream scored): {m['anchor_judge']}  subset: "
          f"{m['subset_pr_count']} PRs of {m['total_daydream_attempts']} attempted\n")
    for j in data["judges"]:
        print(f"── JUDGE {j['display']}  [{j['id']}]")
        for src in j["dirs"]:
            print(f"     source: {src}")
        if j["has_daydream"] and j["daydream"]:
            d = j["daydream"]
            rk = j["ranks"]
            print(f"     daydream  tp={d['tp']} fp={d['fp']} fn={d['fn']}  "
                  f"P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f}  ({d['n_prs']} PRs)")
            print(f"        rank among field (valid):  P #{rk['precision'][0]}/{rk['precision'][1]}  "
                  f"R #{rk['recall'][0]}/{rk['recall'][1]}  F1 #{rk['f1'][0]}/{rk['f1'][1]}")
        else:
            print("     daydream: NOT YET SCORED under this judge (placeholder; re-judge to fill in)")
        top = j["field"][0] if j["field"] else None
        if top:
            print(f"     field: {len(j['field'])} SaaS tools on same {j['subset_pr_count']}-PR subset; "
                  f"best-F1 = {top['display']} (F1={top['f1']:.3f})")
        print()
    if data["skipped_judges"]:
        print("── discovered but skipped (no SaaS field):")
        for s in data["skipped_judges"]:
            print(f"     {s['id']}  ({s['saas_tools']} saas tools, "
                  f"daydream_overlap={s['daydream_overlap']})  — {s['reason']}")
    e = data["economy"]
    print(f"\n── daydream economy (computed from measured tokens @ {e['price_model']} price card "
          f"{e['price_card']['input']}/{e['price_card']['cached']}/{e['price_card']['output']} per 1M):")
    print(f"     total ${e['total_cost_usd']:.4f} over {e['n_with_trajectory']} PRs  "
          f"(${e['cost_per_pr']:.4f}/PR);  tokens in={e['total_prompt_tokens']:,} "
          f"cached={e['total_cached_tokens']:,} out={e['total_completion_tokens']:,}")
    if e["median_wall_seconds"] is not None:
        print(f"     review compute time: median {e['median_wall_seconds']:.0f}s, "
              f"mean {e['mean_wall_seconds']:.0f}s ({e['n_with_wall']} PRs)")
    speed_msg = "loaded" if m["speed_analysis_present"] else "NOT in this snapshot (live timeline crawl not run)"
    print(f"     SaaS latency (speed_analysis.json): {speed_msg}")
    print("\n✓ regenerate success")


if __name__ == "__main__":
    main()
