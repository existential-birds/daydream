"""Distinguish optional persistence failures from protected publication failures."""


class ArchiveFinalizationError(RuntimeError):
    """Optional run-data persistence did not complete successfully."""


class ArchiveIntegrityError(ArchiveFinalizationError):
    """Protected runtime evidence failed validation; publication must stop."""


class ArchivePublicationError(ArchiveFinalizationError):
    """An explicitly requested diagnostic output could not be published."""
