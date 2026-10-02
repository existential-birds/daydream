"""Read-only recommendation and retained-patch verification prompts.

Schemas travel through the backend output contract rather than being repeated in
prompt text. The host persists each verdict and interprets missing entries.
"""

from pathlib import Path
from typing import Any

from daydream.phases.inputs import _render_bash_allowlist
from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION


def _read_only_contract(*, depth: bool = False) -> str:
    """Share the non-mutating tool contract; optionally bound reads to the retained patch."""
    parts = [
        "Read-only contract (MANDATORY):\n"
        "  - Allowed tools: Read, Grep, Glob, Bash.\n"
        f"  - Bash is restricted to non-mutating commands only: {_render_bash_allowlist()}.\n"
        "  - Do NOT write, edit, or move files. Do NOT run `git commit`, "
        "`git add`, `git checkout`, `git reset`, `git stash`, or any other "
        "state-changing command."
    ]
    if depth:
        parts.append(
            "  - Depth: inspect the retained patch's files, but do not roam the whole "
            "tree; prefer Grep/Glob to narrow."
        )
    return "\n".join(parts)

def build_verification_prompt(
    *,
    strategy: str,
    items: list[dict[str, Any]],
    cwd: Path,
    output_path: Path,
) -> str:
    """Check recommendations against interfaces, sibling implementations, and evidence.

    The caller excludes structural findings. Verdicts are advisory and keyed by
    canonical issue ID; an empty issue list produces no verdicts. output_path is
    accepted for extension compatibility, but the host writes the verdict file.
    """
    from daydream.deep.render import render_report

    parts: list[str] = []
    parts.append(
        strategy + "\n\n"
        f"{CWD_GROUNDING_INSTRUCTION.format(cwd=cwd)}\n"
        "The numbered findings to verify (each `issue_id` in your output MUST "
        "match the leading number `N.` of the finding it verifies):\n\n"
        + render_report(items)
        + "\nDo NOT re-run any reviews."
    )
    parts.append(_read_only_contract())
    parts.append(
        "Turn budget: cap your investigation at 25 turns total. Prefer Grep/Glob "
        "to narrow the search before opening files with Read."
    )
    parts.append(
        "Gate-0 anti-confabulation (MANDATORY — applies before any verdict):\n"
        "  Before issuing ANY verdict (consistent/contradicts/uncertain), you MUST "
        "echo the exact artifact you are judging, quoted from a source read in THIS "
        "turn:\n"
        "    - The file:line plus the cited code, read freshly now (not recalled "
        "from earlier in the session).\n"
        "  The artifact is the only source of truth. A verdict issued without a "
        "same-turn echo of its target is INVALID — emit the echo first, or do not "
        "emit the verdict."
    )
    parts.append(
        "For EACH numbered issue in the merged report, perform these five steps:\n\n"
        "  1. Locate the `impl` / interface / protocol declaration the changed "
        "code participates in. If absent, set `verdict=consistent` only if no "
        "sibling implementations exist.\n"
        "  2. Locate every sibling implementation using the Grep tool "
        "(e.g. search for `impl <Trait> for` or `class X(<Iface>)`).\n"
        "  3. Locate the trait/interface doc-comment that specifies the behavior "
        "being changed.\n"
        "  4. Compare the recommendation against those. Verdicts:\n"
        "     - `consistent` -- recommendation aligns with the trait doc and at "
        "least one sibling. Cite one line of evidence.\n"
        "     - `contradicts` -- recommendation would make this impl diverge "
        "from the trait doc OR from a sibling that the trait doc agrees with. "
        "Cite the conflicting line.\n"
        "     - `uncertain` -- cannot decide from the codebase. List the "
        "assumption that would need to hold.\n"
        "  5. Additionally: list any *transitive properties* the recommendation "
        "asserts about functions it does not modify (`unverified_assumptions`). "
        'Example: "assumes `osprey_home()` always returns an absolute path."'
    )
    parts.append(
        "Empty-input rule: if the merged report contains no numbered issues "
        "under `## Issues` or `## Cross-Stack Issues`, emit an empty `verdicts` "
        "array. This is NOT an error."
    )
    parts.append(
        "Every verdict entry MUST include all four required fields, even when "
        "`unverified_assumptions` is an empty array."
    )
    return "\n\n".join(parts)

def build_fix_verify_prompt(
    *,
    items: list[dict[str, Any]],
    changed_hunks: str,
    cwd: Path,
    round_number: int = 1,
) -> str:
    """Audit the complete retained patch against every canonical finding.

    Return one resolved, unresolved, wrong_target, or regressed verdict per item;
    wrong_target and regressed must identify the affected path. Empty hunks mean
    the retained tree matches its stable base.
    """
    from daydream.deep.render import render_report

    hunks_block = changed_hunks if changed_hunks.strip() else "(no hunks provided)"
    parts: list[str] = []
    parts.append(
        "You are the post-fix fix-verifier agent (the `fix-verify` step). The "
        "fix cycle is stabilizing this "
        "worktree; your job is to audit the complete retained result and return "
        "EXACTLY one verdict per finding below. This is a READ-ONLY pass: you "
        "inspect the diff and the code, you do not edit anything.\n\n"
        f"Verification pass {round_number}.\n\n"
        f"{CWD_GROUNDING_INSTRUCTION.format(cwd=cwd)}\n"
        "Audit this complete current retained patch:\n"
        "\n"
        "<changed-hunks>\n"
        f"{hunks_block}\n"
        "</changed-hunks>\n\n"
        "Audit all canonical findings (each `issue_id` in your "
        "output MUST match the leading number `N.` of the finding it "
        "verifies):\n\n"
        + render_report(items)
        + "\nDo NOT re-run reviews; do NOT re-dispatch anything."
    )
    parts.append(
        "Verdict semantics (MANDATORY):\n"
        "  - `resolved` -- the named defect is gone from the retained result.\n"
        "  - `unresolved` -- the defect is still present, wholly or partly.\n"
        "  - `wrong_target` -- the defect lives in a file the finding did NOT "
        "name; emit `path` with the corrected repo-relative file.\n"
        "  - `regressed` -- the retained result contains a NEW instance of the named "
        "defect; emit `path` with the file where it now appears.\n"
        "  - `wrong_target` and `regressed` verdicts MUST carry `path`; the "
        "other two never carry it.\n"
        "  - Emit one verdict entry for EVERY numbered finding. A finding you "
        "omit is treated as `unresolved`; there is no skip verdict."
    )
    parts.append(_read_only_contract(depth=True))
    return "\n\n".join(parts)
