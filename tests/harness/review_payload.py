"""Test-side builder for the rendered review dict the payload assertions inspect.

Production builds a :class:`ReviewPayload` from an already-authorized
``ReviewEvent`` and posts it; the authorization decision itself lives in
``post_review_findings`` (``_is_clean_review``). These tests pin rendering and
the approval gate, so they reach both halves through the same two owners that
production uses.
"""

from __future__ import annotations

from typing import Any

from daydream.pr_review import (
    ClassifiedIssues,
    PRInfo,
    ReviewEvent,
    ReviewRenderers,
    _is_clean_review,
    _review_payload_dict,
    build_payload_for_event,
)


def payload_for(
    pr: PRInfo,
    classified: ClassifiedIssues,
    *,
    renderers: ReviewRenderers,
    run_info: str = "",
    approve_on_clean: bool = False,
    diagram_blocks: str | None = None,
) -> dict[str, Any]:
    """Return the dict ``transport.post_review`` would receive for this review."""
    event = (
        ReviewEvent.APPROVE
        if _is_clean_review(classified, approve_on_clean)
        else ReviewEvent.COMMENT
    )
    return _review_payload_dict(
        build_payload_for_event(
            pr,
            classified,
            event=event,
            run_info=run_info,
            renderers=renderers,
            diagram_blocks=diagram_blocks,
        )
    )
