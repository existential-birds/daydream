"""Pure review scoring core for the Daydream Harbor verifier.

Stdlib-only (dataclasses, hashlib, json, re, math) so this module can be
copied byte-for-byte into a judge-free Harbor verifier image. No daydream
source, no pydantic, no third-party imports.
"""

from __future__ import annotations

import re

MAX_ARTIFACT_BYTES = 1_048_576
MAX_CANDIDATE_FINDINGS = 100
MAX_GOLD_FINDINGS = 50
CONFIDENCE_THRESHOLD = 0.7

_SEP = "\x1f"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class VerifierError(Exception):
    """Raised on any invalid verifier input (mirrors a pydantic error)."""
