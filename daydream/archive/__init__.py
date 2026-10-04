"""Publish frozen run artifacts, a manifest, and a cross-project SQLite index.

``finalize_archive_run`` assembles a complete attested bundle transactionally
under ``get_archive_dir()/runs/{session_id}``; failure raises
``ArchiveFinalizationError`` and removes this attempt's archive outputs.
"""

from daydream.archive.errors import (
    ArchiveFinalizationError as ArchiveFinalizationError,
    ArchiveIntegrityError as ArchiveIntegrityError,
    ArchivePublicationError as ArchivePublicationError,
)
from daydream.archive.finalize import (
    finalize_archive_run as finalize_archive_run,
    get_archive_dir as get_archive_dir,
)
