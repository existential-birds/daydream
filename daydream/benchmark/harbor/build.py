"""Deterministic, leak-resistant content compiler for private PR benchmarks (issue #778).

Consumes a curated private-PR benchmark workspace and compiles each compilable
case into an opaque-keyed ``harbor/`` task tree: opaque task keys, a bounded
delimited PR context block, provenance-free hidden gold + Oracle candidate
artifact, byte-identical verifier/solution template assets, and an exact
inventory + private ``benchmark.lock.json``. Stdlib-only and deterministic
(no timestamps); no CLI here -- issue 9 owns packaging and the ``build-harbor``
command surface.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

TEMPLATE_VERSION = "1"


class CompileError(Exception):
    """Raised on any compile/leakage/validation rejection."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def derive_task_key(case_id: str) -> str:
    """Return the opaque ``case-<sha256(case_id)[:12]>`` task directory key."""
    return "case-" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:12]