"""Host-owned comment markers shared by posting and reconciliation."""
import re

import daydream

DAYDREAM_REPO_URL = "https://github.com/existential-birds/daydream"


DAYDREAM_FOOTER = (
    f"<sub>🧙 Posted by [daydream v{daydream.__version__}]({DAYDREAM_REPO_URL})</sub>"
)


# Hidden HTML-comment marker embedded in posted comment bodies so later runs
# can recognise their own findings (cross-run dedup). Invisible in rendered
# markdown, present in the raw body fetched via the API.
FINDING_MARKER_RE = re.compile(r"<!-- daydream-finding: ([0-9a-f]{64}) -->")


def finding_marker(fingerprint: str) -> str:
    """Render the hidden finding marker comment for a fingerprint."""
    return f"<!-- daydream-finding: {fingerprint} -->"


def parse_finding_markers(text: str) -> list[str]:
    """Return all finding fingerprints embedded in ``text``, in order."""
    return FINDING_MARKER_RE.findall(text)


# Hidden marker for a standalone grounded-diagram comment (issue #1113). One
# per rendered kind, so a later diagram-only run of the SAME kind can find and
# minimize its own prior comment without touching the other kind's.
DIAGRAM_MARKER_RE = re.compile(r"<!-- daydream-diagram: ([a-z]+) ([0-9a-f]{7,40}) -->")


def diagram_marker(kind: str, head_sha: str) -> str:
    """Render the hidden diagram marker comment for one kind at one head."""
    return f"<!-- daydream-diagram: {kind} {head_sha} -->"


def parse_diagram_markers(text: str) -> list[tuple[str, str]]:
    """Return all ``(kind, head_sha)`` diagram markers in ``text``, in order."""
    return [(kind, sha) for kind, sha in DIAGRAM_MARKER_RE.findall(text)]
