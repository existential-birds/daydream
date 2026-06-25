#!/usr/bin/env python3
"""LLM judge: semantically match a bot's findings against daydream's findings.

The deterministic matcher in compare.py only catches findings that share a file,
a nearby line, AND overlapping wording — a strict LOWER BOUND. Two tools almost
always phrase the same issue differently and anchor it to different lines, so
that matcher reports ~0 overlap even when the tools agree. This judge asks an
LLM to decide, per PR, which findings describe the SAME underlying issue.

Per PR it makes ONE LLM call (a light text task — no tools, no repo), so it is
far cheaper/more reliable than a full review and works on subscription backends
that rate-limit deep mode. Output: <in>/<owner__repo>/judge/pr-<N>.json, which
compare.py picks up automatically to compute semantic overlap.

Usage:
  python judge.py --repo owner/repo --in ./out \
      --backend pi --provider zai --model glm-5.2 [--pr 1293] [--min-conf 0.5]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from common import read_json, repo_slug, run, write_json
from compare import bot_findings, dd_findings, findings_digest

MAX_TEXT = 600  # per-finding char cap to bound prompt size

_TRANSIENT = ("429", "overloaded", "rate limit", "try again later", "temporarily")

PROMPT_HEADER = """You are comparing two automated code-review tools that reviewed the SAME pull \
request. Tool A is the existing review bot; Tool B is "daydream".

Match findings that describe the SAME underlying issue — same file and same root \
problem — even when the wording differs or the line numbers differ. Do NOT match \
findings that merely touch the same file but raise different problems.

Return ONLY a JSON object of this exact shape, no prose:
{"matches": [{"a": <A index>, "b": <B index>, "confidence": <0.0-1.0>, "reason": "<short>"}]}
Include a pair only when you are reasonably confident it is the same issue. \
Each A index and each B index may appear at most once.
"""


def _fmt(findings: list[dict], kind: str) -> str:
    lines = []
    for i, f in enumerate(findings):
        text = (f.get("text") or "").strip().replace("\n", " ")
        if len(text) > MAX_TEXT:
            text = text[:MAX_TEXT] + "…"
        loc = f"{f.get('path')}:{f.get('line')}"
        lines.append(f"[{kind}{i}] {loc} — {text}")
    return "\n".join(lines) if lines else f"(no {kind} findings)"


def _build_cmd(backend: str, provider: str, model: str, prompt: str) -> list[str]:
    if backend == "pi":
        return ["pi", "-p", "--no-tools", "--provider", provider, "--model", model, prompt]
    if backend == "codex":
        return ["codex", "exec", prompt]
    raise ValueError(f"unsupported judge backend: {backend}")


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of model output (tolerates code fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the largest brace-balanced span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def call_llm(backend: str, provider: str, model: str, prompt: str,
             retries: int, backoff: int) -> str:
    last = ""
    for attempt in range(1, retries + 2):
        proc = run(_build_cmd(backend, provider, model, prompt), check=False, timeout=300)
        out = (proc.stdout or "").strip()
        if proc.returncode == 0 and out:
            return out
        last = out or (proc.stderr or "")
        if attempt <= retries and any(s in last.lower() for s in _TRANSIENT):
            wait = backoff * (2 ** (attempt - 1))
            print(f"    transient judge error; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        break
    raise RuntimeError(f"judge LLM call failed: {last[:200]}")


def judge_pr(record: dict, art: dict, backend: str, provider: str, model: str,
             retries: int, backoff: int) -> dict:
    a = bot_findings(record)
    b = dd_findings(art)
    n = record["pr_number"]
    # Bind this judgement to the exact findings it scored + the snapshot it ran
    # against, so compare.py can reject it if replay is later rerun with different
    # findings (a stale judge artifact would otherwise silently skew overlap).
    provenance = {
        "review_commit_id": record.get("review_commit_id"),
        "findings_digest": findings_digest(a, b),
    }
    if not a or not b:
        return {"pr_number": n, "a_count": len(a), "b_count": len(b), "matches": [],
                "note": "one side empty", **provenance}

    prompt = (
        f"{PROMPT_HEADER}\n\n=== Tool A findings ({len(a)}) ===\n{_fmt(a, 'A')}\n\n"
        f"=== Tool B (daydream) findings ({len(b)}) ===\n{_fmt(b, 'B')}\n"
    )
    raw = call_llm(backend, provider, model, prompt, retries, backoff)
    parsed = _extract_json(raw)
    if not parsed or "matches" not in parsed:
        return {"pr_number": n, "a_count": len(a), "b_count": len(b), "matches": [],
                "error": "unparseable judge output", "raw": raw[:500], **provenance}

    def _confidence(match: dict) -> float:
        try:
            return float(match.get("confidence", 1.0))
        except (TypeError, ValueError, AttributeError):
            return -1.0

    # Validate indices and dedupe so each side is used at most once. Process by
    # descending confidence so a conflicting index keeps the stronger pair rather
    # than whichever the model happened to emit first.
    seen_a, seen_b, clean = set(), set(), []
    for m in sorted(parsed["matches"], key=_confidence, reverse=True):
        try:
            ai, bi = int(m["a"]), int(m["b"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= ai < len(a) and 0 <= bi < len(b)):
            continue
        if ai in seen_a or bi in seen_b:
            continue
        seen_a.add(ai)
        seen_b.add(bi)
        clean.append({
            "a": ai, "b": bi,
            "confidence": float(m.get("confidence", 1.0)),
            "reason": str(m.get("reason", ""))[:200],
            "a_resolved": bool(a[ai].get("resolved")),
            "a_path": a[ai].get("path"),
            "b_path": b[bi].get("path"),
        })
    return {"pr_number": n, "backend": backend, "model": model,
            "a_count": len(a), "b_count": len(b), "matches": clean, **provenance}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--in", dest="in_root", default="./out", type=Path)
    ap.add_argument("--backend", default="pi", choices=["pi", "codex"])
    ap.add_argument("--provider", default="zai", help="pi provider (default: zai)")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--pr", type=int, action="append", help="judge only these PRs")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=int, default=20)
    args = ap.parse_args()

    base = args.in_root / repo_slug(args.repo)
    findings_dir = base / "findings"
    # Judge every PR that has BOTH a harvest record and a daydream findings file.
    targets = []
    for fp in sorted(findings_dir.glob("pr-*.json")):
        n = int(fp.stem.split("-")[1])
        if args.pr and n not in set(args.pr):
            continue
        rec_path = base / f"pr-{n}.json"
        if rec_path.exists():
            targets.append((n, rec_path, fp))

    print(f"Judging {len(targets)} PRs with {args.backend}/{args.model}", file=sys.stderr)
    for n, rec_path, fp in targets:
        try:
            art = read_json(fp)
            if not art.get("findings"):
                print(f"  PR #{n}: no daydream findings; skip", file=sys.stderr)
                continue
            res = judge_pr(read_json(rec_path), art, args.backend, args.provider,
                           args.model, args.retries, args.backoff)
        except (RuntimeError, ValueError, OSError) as e:
            res = {"pr_number": n, "error": str(e), "matches": []}
        write_json(base / "judge" / f"pr-{n}.json", res)
        print(f"  PR #{n}: {len(res.get('matches', []))} semantic matches "
              f"(A={res.get('a_count')}, B={res.get('b_count')})"
              + (f" [{res['error']}]" if res.get("error") else ""), file=sys.stderr)

    print(f"\nJudge complete -> {base / 'judge'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
