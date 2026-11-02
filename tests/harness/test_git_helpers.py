"""Contract tests for the shared real-git subprocess helpers.

The seed family (``SEED_ENV`` + ``write_and_stage`` + ``seeded_commit``) is what makes
every benchmark suite's fixtures byte-reproducible, so the SHAs it produces are
pinned here: this module is the regression gate for the seed identity and the
commit step. Values were measured on this repository with these helpers.
"""

from __future__ import annotations

_SHA_BASE2 = "d35f2cbffc81b6292f67cf891ac1c4256fe948a4"

