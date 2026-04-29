"""CLI entry point: ``python -m daydream.eval <.daydream dir> [--session ID] [--pr owner/repo#number]``."""

import json
import sys
from pathlib import Path

from daydream.eval.analyzer import analyze_session


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage: python -m daydream.eval <.daydream dir> [--session ID] [--pr owner/repo#number]",
            file=sys.stderr,
        )
        sys.exit(1)

    daydream_dir = Path(args[0])
    if not daydream_dir.is_dir():
        print(f"Error: {daydream_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Optional --session filter
    session_id = None
    for i, arg in enumerate(args):
        if arg == "--session" and i + 1 < len(args):
            session_id = args[i + 1]

    result = analyze_session(daydream_dir, session_id=session_id)

    # Optional PR feedback
    pr_spec = None
    for i, arg in enumerate(args):
        if arg == "--pr" and i + 1 < len(args):
            pr_spec = args[i + 1]

    # Auto-detect PR from trajectory metadata when --pr is not provided
    if pr_spec is None:
        pr_meta = result.get("pr") or {}
        auto_repo = pr_meta.get("pr_repo")
        auto_number = pr_meta.get("pr_number")
        if auto_repo and auto_number:
            pr_spec = f"{auto_repo}#{auto_number}"
            print(
                f"Auto-detected PR: {auto_repo}#{auto_number} from trajectory metadata",
                file=sys.stderr,
            )

    if pr_spec:
        from daydream.eval.pr_feedback import fetch_pr_feedback

        # Parse owner/repo#number
        if "#" in pr_spec:
            repo, num = pr_spec.rsplit("#", 1)
            pr_number = int(num)
        else:
            print(f"Error: PR spec must be owner/repo#number, got {pr_spec}", file=sys.stderr)
            sys.exit(1)

        findings = []
        deep_dir = daydream_dir / "deep"
        if deep_dir.is_dir():
            import json as _json

            for f in sorted(deep_dir.glob("stack-*-records.json")):
                findings.extend(_json.loads(f.read_text()))

        result["pr_feedback"] = fetch_pr_feedback(repo, pr_number, findings)

    json.dump(result, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
