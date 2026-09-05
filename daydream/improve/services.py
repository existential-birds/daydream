"""Re-export shim for the shared service-root discovery module.

Service discovery moved to :mod:`daydream.services` (issue #1113) so the
grounded-diagram flow and the improve flow resolve monorepo service roots from
one implementation. This shim keeps the historical
``daydream.improve.services`` import path working; ``__all__`` is explicit
because both ruff (F401) and mypy's ``no_implicit_reexport`` require a
re-export to be declared.
"""

from __future__ import annotations

from daydream.services import Service, enumerate_services, filter_scope

__all__ = ["Service", "enumerate_services", "filter_scope"]
