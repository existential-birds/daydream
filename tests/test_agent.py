"""Tests for daydream.agent module-level state accessors."""

from daydream.agent import (
    get_non_interactive,
    is_environmental_failure,
    reset_state,
    set_non_interactive,
)


def test_set_and_get_non_interactive() -> None:
    try:
        set_non_interactive(True)
        assert get_non_interactive() is True
    finally:
        reset_state()


def test_reset_state_clears_non_interactive() -> None:
    set_non_interactive(True)
    reset_state()
    assert get_non_interactive() is False


def test_is_environmental_failure_both_directions() -> None:
    environmental = [
        "The dev Postgres container is not running",
        "could not connect to server: Connection refused",
        "localhost:5432",
        "ECONNREFUSED",
    ]
    for output in environmental:
        assert is_environmental_failure(output) is True, output

    ordinary = [
        "AssertionError: assert 1 == 2",
        "1 failed, 3 passed",
        "ValueError: bad input",
    ]
    for output in ordinary:
        assert is_environmental_failure(output) is False, output


def test_scrubbed_supervisor_error_scrubs_all_str_surfaces() -> None:
    """_scrubbed_supervisor_error must never re-surface a redactable value.

    Regression for issue #702 round 2: the args-scrub must hold for
    OSError-family types (whose str() is built from errno/strerror, not args)
    and for types overriding __str__/__repr__, and must preserve the
    retryable discriminator on the reconstruction path too.
    """
    from daydream.agent import (
        _RedactedSupervisorError,
        _scrubbed_supervisor_error,
    )

    credential = "ZAI_API_KEY=credential-shaped-supervisor-value"

    # OSError-family: real (sub)type preserved, str() scrubbed
    err = OSError(2, f"failed auth with {credential}")
    rebuilt = _scrubbed_supervisor_error(err)
    assert type(rebuilt) is type(err)
    assert isinstance(rebuilt, OSError)
    assert credential not in str(rebuilt)
    assert "[REDACTED_ENV_VAR]" in str(rebuilt)

    # custom __str__ override: fail closed to the already-redacted stand-in
    class CustomLeak(Exception):
        def __str__(self) -> str:
            return f"boom {self.args}"

    custom = CustomLeak((credential,))
    stand_in = _scrubbed_supervisor_error(custom)
    assert type(stand_in) is _RedactedSupervisorError
    assert credential not in str(stand_in)
    assert stand_in.original_type_name == "CustomLeak"

    # reconstruction path must preserve retryable even when it is an
    # instance attribute set from a non-args kwarg (e.g. BackendError).
    class RetryableBackendError(RuntimeError):
        def __init__(self, message: str, *, retryable: bool = False) -> None:
            super().__init__(message)
            self.retryable = retryable

    backend = RetryableBackendError(f"boom {credential}", retryable=True)
    rebuilt_retryable = _scrubbed_supervisor_error(backend)
    assert type(rebuilt_retryable) is RetryableBackendError
    assert credential not in str(rebuilt_retryable)
    assert getattr(rebuilt_retryable, "retryable", False) is True
