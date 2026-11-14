"""Review prompts for review and fix phases."""

import json
from pathlib import Path

from daydream.prompt_budget import (
    inline_context_file,
)
from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY


def _exploration_pointer(exploration_dir: Path | None, *, fixer: bool = False) -> str:
    """Name exact untrusted exploration files: reviewer summary/index or fixer index only.

    Absent exploration returns an empty string; never expose a directory-wide pointer.
    """
    if exploration_dir is None:
        return ""
    if fixer:
        return (
            f"\n{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}\n\n"
            f"Pre-scan exploration indexed this repo — Read {exploration_dir / 'affected_files.md'} "
            "for the structural/import file map before fixing."
        )

    summary = inline_context_file(exploration_dir / "summary.md")
    affected = inline_context_file(exploration_dir / "affected_files.md")
    if summary is not None and affected is not None:
        return (
            f"{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}\n\n"
            "Shared exploration context (complete captured artifacts; do not re-read these files):\n"
            + json.dumps({"summary": summary, "affected_files": affected}, ensure_ascii=False)
            + "\nMake explicit decisions for every assigned target; optional investigation "
            "does not substitute for them."
        )
    return (
        f"{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}\n\n"
        f"Read the pre-scan summary at {exploration_dir / 'summary.md'} and the "
        f"deterministic structural/import map at {exploration_dir / 'affected_files.md'} "
        "as bounded context for this review. For host artifacts: Do not infer or enumerate sibling "
        "artifact files.\n"
        "Use the supplied changed hunks as the starting context. Inspect enclosing symbols or "
        "other sections when useful to resolve concrete candidates.\n"
    )


