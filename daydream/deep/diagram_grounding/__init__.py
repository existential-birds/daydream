"""Ground diagram proposals against repository evidence before rendering.

Each grounder checks, prunes, caps, then evaluates omission floors. Share one
RepoSymbols index across kinds and repair turns; render only when omit_reasons
is empty. spec_final contains schema keys only for posting to revalidate.
"""

from daydream.deep.diagram_grounding.evidence import RepoSymbols as RepoSymbols
from daydream.deep.diagram_grounding.flowchart import ground_flowchart as ground_flowchart
from daydream.deep.diagram_grounding.models import (
    OMIT_REASONS as OMIT_REASONS,
    REASON_CODES as REASON_CODES,
    ElementCheck as ElementCheck,
    GroundingReport as GroundingReport,
)
from daydream.deep.diagram_grounding.sequence import ground_sequence as ground_sequence
