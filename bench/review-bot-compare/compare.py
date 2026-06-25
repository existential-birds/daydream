#!/usr/bin/env python3
"""Compare a review bot vs daydream on the same PR snapshots.

Aligns the bot's inline comments against daydream's findings per PR using a
deterministic matcher (same file + line proximity + body token overlap), then
scores:
  - issues raised (counts, per-severity)
  - overlap / bot-only / daydream-only
  - "acted upon" precision: fraction of each side's findings that line up with
    a RESOLVED bot thread (the closest available ground-truth signal)
  - cost (daydream measured $ + tokens; bot cost is not observable — see notes)
  - latency (daydream wall-clock)

Deterministic matching is the pilot baseline; phrasing/line drift means it is a
LOWER BOUND on true semantic overlap. Layer an LLM judge on top later for the
final numbers.

Output: <in>/<owner__repo>/report.md and comparison.csv

Usage:
  python compare.py --repo owner/repo --bot-name coderabbit --in ./out
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from common import read_json, repo_slug, write_json

LINE_DELTA = 12  # lines apart still considered the same locus
JACCARD_MIN = 0.10  # min body token overlap to confirm a same-locus match

# Per-1M-token prices (USD). Used to synthesize a $ cost when the backend
# reports none (pi/GLM via z.ai subscription reports $0 but real tokens).
PRICING = {
    # GLM-5.2 on z.ai: input $1.40 / cached-input $0.26 / output $4.40 per 1M.
    "glm-5.2": {"input": 1.40, "cached": 0.26, "output": 4.40},
}


def synth_cost(prompt: int, completion: int, cached: int, model: str) -> float | None:
    """Synthesize $ cost from token counts using the model's price card.

    These counters are DISJOINT, matching provider usage semantics (and how the
    pi backend maps them): ``prompt`` = fresh input (usage.input, billed at the
    full input rate), ``cached`` = cache reads (usage.cacheRead, billed at the
    cached-input rate), ``completion`` = output. Do not subtract cached from
    prompt — for GLM, cache reads dwarf fresh input (e.g. 980k vs 107k).
    """
    price = PRICING.get(model)
    if not price:
        return None
    return (
        (prompt or 0) / 1e6 * price["input"]
        + (cached or 0) / 1e6 * price["cached"]
        + (completion or 0) / 1e6 * price["output"]
    )

_WORD = re.compile(r"[a-z0-9_]+")
_STOP = {
    "the", "a", "an", "is", "are", "to", "of", "in", "on", "and", "or", "this",
    "that", "it", "be", "should", "could", "will", "with", "for", "you", "if",
    "as", "at", "by", "not", "can", "may", "from", "use", "using", "code",
}


def tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def norm_path(p: str | None) -> str:
    return (p or "").lstrip("./").strip()


def bot_findings(record: dict) -> list[dict]:
    """One finding per inline comment, tagged with thread-resolution status."""
    # Index resolved status by (path, line) from review threads.
    resolved_loci = {
        (norm_path(t.get("path")), t.get("line"))
        for t in record.get("threads", [])
        if t.get("is_resolved")
    }
    out = []
    for c in record.get("comments", []):
        if c.get("in_reply_to_id"):
            continue  # skip follow-up replies; keep the originating comment
        line = c.get("line") or c.get("original_line") or c.get("start_line")
        path = norm_path(c.get("path"))
        out.append({
            "path": path,
            "line": line,
            "text": c.get("body") or "",
            "resolved": (path, line) in resolved_loci or (path, c.get("line")) in resolved_loci,
        })
    return out


def dd_findings(art: dict) -> list[dict]:
    out = []
    for f in art.get("findings", []):
        out.append({
            "path": norm_path(f.get("path")),
            "line": f.get("line"),
            "text": f"{f.get('title') or ''}\n{f.get('body') or ''}",
            "severity": f.get("severity"),
        })
    return out


def findings_digest(bot: list[dict], dd: list[dict]) -> str:
    """Stable hash of the exact (path, line, text) inputs a judge run scored.

    judge.py records this; compare.py recomputes it from the current findings and
    rejects a judge artifact whose digest no longer matches (e.g. replay was rerun
    with a different backend), so stale semantic matches can't skew the report.
    """
    payload = json.dumps(
        {
            "bot": [[f["path"], f["line"], f["text"]] for f in bot],
            "dd": [[f["path"], f["line"], f["text"]] for f in dd],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def match(bot: list[dict], dd: list[dict]) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matches: list of (bot_idx, dd_idx, score)."""
    cands = []
    for i, b in enumerate(bot):
        bt = tokens(b["text"])
        for j, d in enumerate(dd):
            if b["path"] != d["path"]:
                continue
            if b["line"] and d["line"] and abs(b["line"] - d["line"]) > LINE_DELTA:
                continue
            score = jaccard(bt, tokens(d["text"]))
            # Same file + close line is itself weak evidence; require some overlap
            # unless lines coincide almost exactly.
            close = b["line"] and d["line"] and abs(b["line"] - d["line"]) <= 3
            if score >= JACCARD_MIN or close:
                cands.append((score + (0.2 if close else 0.0), i, j))
    cands.sort(reverse=True)
    used_b, used_d, pairs = set(), set(), []
    for score, i, j in cands:
        if i in used_b or j in used_d:
            continue
        used_b.add(i)
        used_d.add(j)
        pairs.append((i, j, round(score, 3)))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--bot-name", default="bot", help="display name, e.g. coderabbit")
    ap.add_argument("--in", dest="in_root", default="./out", type=Path)
    ap.add_argument("--price-model", default="glm-5.2",
                    help=f"price card for synthetic cost when backend reports $0 "
                         f"(known: {', '.join(PRICING)})")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="min confidence for an LLM-judge match to count (default 0.5)")
    args = ap.parse_args()

    base = args.in_root / repo_slug(args.repo)
    replay = read_json(base / "replay" / "summary.json")["results"]

    rows = []
    measured_cost = False  # any PR where the backend itself reported a real $
    synth_cost_used = False  # any PR where we synthesized $ from tokens
    incomplete_threads = []  # PRs whose GraphQL thread fetch was truncated

    for res in replay:
        n = res["pr_number"]
        record = read_json(base / f"pr-{n}.json")
        if record.get("threads_complete") is False:
            incomplete_threads.append(n)
        bf = bot_findings(record)
        ok = res.get("status") == "ok"
        # A run is USABLE for comparison if it produced a valid findings
        # artifact, even when it exited nonzero afterward (e.g. a late-phase
        # usage-limit error). "complete" = clean exit; "partial" = errored but
        # findings emitted; "failed" = no findings at all.
        fp = base / "findings" / f"pr-{n}.json"
        art = read_json(fp) if fp.exists() else None
        df = dd_findings(art) if art else []
        usable = ok or len(df) > 0
        state = "complete" if ok else ("partial" if df else "failed")

        pairs = match(bf, df) if usable else []
        overlap = len(pairs)
        bot_resolved = sum(1 for b in bf if b["resolved"])
        overlap_resolved = sum(1 for i, _, _ in pairs if bf[i]["resolved"])

        # Prefer the LLM judge's semantic matches when a judge artifact exists —
        # but only if it was scored against THESE findings. A digest mismatch
        # means replay was rerun since the judge ran, so the matches are stale;
        # reject them and fall back to the deterministic lower bound.
        overlap_source = "deterministic"
        jp = base / "judge" / f"pr-{n}.json"
        if usable and jp.exists():
            jrec = read_json(jp)
            fresh = jrec.get("findings_digest") == findings_digest(bf, df)
            if not jrec.get("error") and fresh:
                jm = [m for m in jrec.get("matches", [])
                      if float(m.get("confidence", 0)) >= args.min_conf]
                overlap = len(jm)
                overlap_resolved = sum(1 for m in jm if m.get("a_resolved"))
                overlap_source = "llm-judge"

        # File-level overlap: a no-LLM signal that both tools targeted the same
        # file, even when line/wording drift defeats finding-level matching.
        bot_files = {b["path"] for b in bf if b["path"]}
        dd_files = {d["path"] for d in df if d["path"]}
        common_files = bot_files & dd_files

        ptok = res.get("prompt_tokens") or 0
        ctok = res.get("completion_tokens") or 0
        cached = res.get("cached_tokens") or 0

        # Prefer the backend's own $ if it reported a real (non-zero) cost;
        # otherwise synthesize from tokens via the price card. Unknown if a
        # partial run never wrote token metrics.
        backend_cost = res.get("cost_usd")
        if backend_cost:
            cost = backend_cost
            measured_cost = True
            cost_kind = "measured"
        elif ptok or ctok or cached:
            cost = synth_cost(ptok, ctok, cached, args.price_model)
            if cost is not None:
                synth_cost_used = True
            cost_kind = "synth"
        else:
            cost = None
            cost_kind = "unknown"

        rows.append({
            "pr": n,
            "title": record["title"],
            "state": state,
            "usable": usable,
            "bot_findings": len(bf),
            "dd_findings": len(df) if usable else None,
            "overlap": overlap if usable else None,
            "bot_only": (len(bf) - overlap) if usable else None,
            "dd_only": (len(df) - overlap) if usable else None,
            "bot_resolved": bot_resolved,
            "dd_on_resolved": overlap_resolved if usable else None,
            "overlap_source": overlap_source if usable else None,
            "bot_files": len(bot_files),
            "dd_files": len(dd_files) if usable else None,
            "common_files": len(common_files) if usable else None,
            "cost_usd": round(cost, 4) if cost is not None else None,
            "cost_kind": cost_kind,
            "prompt_tok": ptok,
            "completion_tok": ctok,
            "cached_tok": cached,
            "wall_s": res.get("wall_seconds"),
            "status": res.get("status"),
        })

    # Comparison metrics aggregate over USABLE runs (complete + partial). A
    # crash with no findings is excluded — it is not "daydream found nothing".
    done = [r for r in rows if r["usable"]]
    failed = [r for r in rows if not r["usable"]]
    tot = {
        "bot": sum(r["bot_findings"] for r in done),
        "dd": sum(r["dd_findings"] for r in done),
        "overlap": sum(r["overlap"] for r in done),
        "bot_only": sum(r["bot_only"] for r in done),
        "dd_only": sum(r["dd_only"] for r in done),
        "bot_resolved": sum(r["bot_resolved"] for r in done),
        "overlap_resolved": sum(r["dd_on_resolved"] for r in done),
        "common_files": sum(r["common_files"] for r in done),
        "bot_files": sum(r["bot_files"] for r in done),
        "dd_files": sum(r["dd_files"] for r in done),
        "cost": sum(r["cost_usd"] or 0 for r in done),
        "ptok": sum(r["prompt_tok"] for r in done),
        "ctok": sum(r["completion_tok"] for r in done),
        "cached": sum(r["cached_tok"] for r in done),
        "secs": sum(r["wall_s"] or 0 for r in done),
        "wasted_cost": sum(r["cost_usd"] or 0 for r in failed),
    }

    # CSV
    csv_path = base / "comparison.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["pr"])
        w.writeheader()
        w.writerows(rows)

    def cell(v: object) -> str:
        return "—" if v is None else str(v)

    # Markdown report
    bn = args.bot_name
    nd = len(done)
    n_complete = sum(1 for r in rows if r["state"] == "complete")
    n_partial = sum(1 for r in rows if r["state"] == "partial")
    glyph = {"complete": "✓", "partial": "◐", "failed": "✗"}
    md = [f"# {bn} vs daydream — {args.repo}\n"]
    md.append(f"PRs: **{len(rows)}** total — daydream **{n_complete}** complete, "
              f"**{n_partial}** partial (findings emitted then errored), "
              f"**{len(failed)}** failed (no findings).\n")
    md.append("## Per-PR\n")
    md.append(f"| PR | dd | {bn} | daydream | overlap | file-overlap | "
              f"{bn} resolved | cost $ | wall s |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        dd_disp = cell(r["dd_findings"]) if r["usable"] else f"✗ {r['status']}"
        files = (f"{r['common_files']}/{r['dd_files']}∩{r['bot_files']}"
                 if r["usable"] else "—")
        md.append(
            f"| #{r['pr']} | {glyph[r['state']]} | {r['bot_findings']} | {dd_disp} "
            f"| {cell(r['overlap'])} | {files} "
            f"| {r['bot_resolved']} | {cell(r['cost_usd'])} | {cell(r['wall_s'])} |"
        )
    md.append("\n_overlap: same underlying issue — per the LLM judge where a judge "
              "artifact exists, else deterministic file+line+wording (a lower bound). "
              "file-overlap: common/daydream∩bot files both flagged._")

    if failed:
        md.append(f"\n**Failed runs ({len(failed)}, no findings — excluded):** "
                  + ", ".join(f"#{r['pr']} ({r['status']})" for r in failed) + "\n")
    if n_partial:
        md.append(f"**Partial runs ({n_partial}):** findings were emitted before a "
                  f"late error (e.g. usage limit); included in counts, but their "
                  f"cost/tokens may be missing and the review may be pre-merge.\n")
    if incomplete_threads:
        md.append(f"**Unreliable resolved-thread counts ({len(incomplete_threads)}):** "
                  + ", ".join(f"#{n}" for n in incomplete_threads)
                  + " — the GraphQL thread fetch was truncated at harvest, so the "
                  "acted-upon proxy understates these PRs. Re-harvest to refresh.\n")

    md.append(f"\n## Totals (over {nd} usable daydream runs: "
              f"{n_complete} complete + {n_partial} partial)\n")
    if nd:
        md.append(f"- **{bn} findings:** {tot['bot']}")
        md.append(f"- **daydream findings:** {tot['dd']}")
        sources = {r["overlap_source"] for r in done if r["overlap_source"]}
        osrc = ("LLM-judge" if sources == {"llm-judge"}
                else "deterministic (lower bound — run judge.py for semantic overlap)"
                if sources == {"deterministic"} else "mixed judge/deterministic")
        md.append(f"- **Issue-level overlap [{osrc}]:** {tot['overlap']}")
        md.append(f"- **File-level overlap:** {tot['common_files']} files flagged by both "
                  f"({tot['bot_files']} {bn} files, {tot['dd_files']} daydream files)")
        md.append(f"- **{bn}-only:** {tot['bot_only']}  |  **daydream-only:** {tot['dd_only']}")
        if tot["bot"]:
            md.append(f"- **Overlap as % of {bn}:** {100*tot['overlap']/tot['bot']:.0f}%")
        if tot["dd"]:
            md.append(f"- **Overlap as % of daydream:** {100*tot['overlap']/tot['dd']:.0f}%")
        md.append(f"- **{bn} resolved threads (acted-upon proxy):** {tot['bot_resolved']}"
                  f" / {tot['bot']}")
        md.append(f"- **daydream findings that hit a resolved {bn} thread:** "
                  f"{tot['overlap_resolved']}")
    else:
        md.append("_No completed daydream runs — re-run on a more stable backend._")

    md.append("\n### Cost & throughput (daydream, completed runs)\n")
    if measured_cost and not synth_cost_used:
        cost_label = "measured by backend"
    elif synth_cost_used and not measured_cost:
        cost_label = f"synthesized from tokens @ {args.price_model} price card"
    else:
        cost_label = "mixed (measured where reported, else synthesized from tokens)"
    npr = nd or 1
    md.append(f"- **Total cost:** ${tot['cost']:.4f} ({cost_label})")
    md.append(f"- **Avg $/PR:** ${tot['cost']/npr:.4f}")
    md.append(f"- **Tokens:** {tot['ptok']:,} fresh input / {tot['cached']:,} cached-read "
              f"/ {tot['ctok']:,} output")
    if synth_cost_used:
        pc = PRICING.get(args.price_model, {})
        md.append(f"- **Price card ({args.price_model}):** "
                  f"${pc.get('input', '?')}/1M in, ${pc.get('cached', '?')}/1M cached-in, "
                  f"${pc.get('output', '?')}/1M out (disjoint counters)")
    md.append(f"- **Wall-clock:** {tot['secs']:.0f}s total"
              f" ({tot['secs']/npr:.0f}s/PR avg)")
    if tot["wasted_cost"]:
        md.append(f"- **Spend wasted on {len(failed)} failed runs:** ${tot['wasted_cost']:.4f}")
    md.append(f"\n> {bn} per-review LLM cost is **not observable** (SaaS). Compare "
              f"daydream's $/PR above against {bn}'s amortized list price "
              f"(plan ÷ PRs reviewed).\n")
    md.append("> Overlap is a deterministic **lower bound** (file+line+token "
              "match). Add an LLM judge for semantic alignment.\n")

    report_path = base / "report.md"
    report_path.write_text("\n".join(md))
    write_json(base / "comparison.json", {"repo": args.repo, "totals": tot, "rows": rows})

    print("\n".join(md))
    print(f"\nWrote {report_path}\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
