"""The author-intent precedence rule.

This module holds one shared definition of the PR-description "authoritative"
precedence rule. Both consumers reference it by import rather than duplicating
the text inline:

- ``daydream.phases.build_intent_prompt`` composes it into the intent-phase
  prompt after the ``"The author supplied the following pull-request description"``
  opener.
- ``daydream.deep.prompts._context_pointers`` injects it into the four
  finding-producing review prompts (per-stack, structural, generic-fallback,
  arbiter) when a fresh PR body was ingested.

The rule is worded to be context-neutral: it reads correctly whether it
follows the "author supplied" opener in the intent prompt or a pointer to
``intent_path`` in the downstream prompts.

Introduced to fix GitHub issue #279.
"""

from __future__ import annotations

AUTHORITATIVE_INTENT_RULE = (
    "Treat this author-stated intent as AUTHORITATIVE: where the description "
    "and the intent you would infer from the diff conflict, the description "
    "outranks the diff. Crucially, when the description says something is "
    "deliberate but the diff appears to contradict it — a near-1.0 ratio that "
    "looks inert, a guard that looks like a no-op, a pass-through that looks "
    "unfinished — that is a deliberate design decision to preserve, NOT a "
    "defect to surface or 'complete'."
)

__all__ = ["AUTHORITATIVE_INTENT_RULE"]
