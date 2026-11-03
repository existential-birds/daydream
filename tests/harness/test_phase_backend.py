"""Real-path tests for the shared ``PhaseDispatchBackend``.

Drives the production shallow mode through ``runner.run`` with the shared
phase-dispatch fake injected at the ``daydream.runner.create_backend`` seam.
Asserts the observable outcome and that the removed parse phase is not invoked.
"""


