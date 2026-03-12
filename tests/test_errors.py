"""Tests for madakit._errors.

Covers the full error hierarchy: AgentError base, ProviderError with
optional status_code, MiddlewareError, CircuitOpenError, and
RetryExhaustedError with last_error chaining. Tests isinstance hierarchy,
message propagation, and attribute access.
"""

from __future__ import annotations

import pytest

from madakit._errors import (
    AgentError,
    CircuitOpenError,
    MiddlewareError,
    ProviderError,
    RetryExhaustedError,
)


class TestAgentError:
    """Tests for the AgentError base exception."""

    def test_is_exception(self) -> None:
        """AgentError inherits from the built-in Exception."""
        assert issubclass(AgentError, Exception)

    def test_can_be_raised(self) -> None:
        """AgentError can be raised and caught."""
        with pytest.raises(AgentError):
            raise AgentError("something went wrong")

    def test_message_preserved(self) -> None:
        """The error message is accessible via str()."""
        err = AgentError("boom")
        assert str(err) == "boom"

    def test_caught_as_exception(self) -> None:
        """AgentError is caught by a bare except Exception handler."""
        with pytest.raises(Exception):
            raise AgentError("caught as Exception")

    def test_empty_message(self) -> None:
        """AgentError can be raised with no message."""
        err = AgentError()
        assert str(err) == ""


class TestProviderError:
    """Tests for the ProviderError exception."""

    def test_is_agent_error(self) -> None:
        """ProviderError is a subclass of AgentError."""
        assert issubclass(ProviderError, AgentError)

    def test_is_exception(self) -> None:
        """ProviderError is a subclass of Exception."""
        assert issubclass(ProviderError, Exception)

    def test_message_preserved(self) -> None:
        """The error message is accessible via str()."""
        err = ProviderError("rate limited")
        assert str(err) == "rate limited"

    def test_status_code_default_none(self) -> None:
        """status_code defaults to None when not provided."""
        err = ProviderError("network error")
        assert err.status_code is None

    def test_status_code_stored(self) -> None:
        """status_code is stored when explicitly provided."""
        err = ProviderError("not found", status_code=404)
        assert err.status_code == 404

    def test_status_code_500(self) -> None:
        """status_code stores server-error codes correctly."""
        err = ProviderError("internal server error", status_code=500)
        assert err.status_code == 500

    def test_status_code_429(self) -> None:
        """status_code stores rate-limit code correctly."""
        err = ProviderError("too many requests", status_code=429)
        assert err.status_code == 429

    def test_caught_as_agent_error(self) -> None:
        """ProviderError is caught by an AgentError handler."""
        with pytest.raises(AgentError):
            raise ProviderError("upstream failed", status_code=503)

    def test_caught_as_exception(self) -> None:
        """ProviderError is caught by a bare Exception handler."""
        with pytest.raises(Exception):
            raise ProviderError("upstream failed")


class TestMiddlewareError:
    """Tests for the MiddlewareError base exception."""

    def test_is_agent_error(self) -> None:
        """MiddlewareError is a subclass of AgentError."""
        assert issubclass(MiddlewareError, AgentError)

    def test_is_exception(self) -> None:
        """MiddlewareError is a subclass of Exception."""
        assert issubclass(MiddlewareError, Exception)

    def test_message_preserved(self) -> None:
        """The error message is accessible via str()."""
        err = MiddlewareError("middleware failed")
        assert str(err) == "middleware failed"

    def test_can_be_raised(self) -> None:
        """MiddlewareError can be raised and caught directly."""
        with pytest.raises(MiddlewareError):
            raise MiddlewareError("oops")

    def test_caught_as_agent_error(self) -> None:
        """MiddlewareError is caught by an AgentError handler."""
        with pytest.raises(AgentError):
            raise MiddlewareError("caught upstream")

    def test_not_a_provider_error(self) -> None:
        """MiddlewareError is not a subclass of ProviderError."""
        assert not issubclass(MiddlewareError, ProviderError)


class TestCircuitOpenError:
    """Tests for the CircuitOpenError exception."""

    def test_is_middleware_error(self) -> None:
        """CircuitOpenError is a subclass of MiddlewareError."""
        assert issubclass(CircuitOpenError, MiddlewareError)

    def test_is_agent_error(self) -> None:
        """CircuitOpenError is a subclass of AgentError."""
        assert issubclass(CircuitOpenError, AgentError)

    def test_is_exception(self) -> None:
        """CircuitOpenError is a subclass of Exception."""
        assert issubclass(CircuitOpenError, Exception)

    def test_message_preserved(self) -> None:
        """The error message is accessible via str()."""
        err = CircuitOpenError("circuit is open")
        assert str(err) == "circuit is open"

    def test_can_be_raised(self) -> None:
        """CircuitOpenError can be raised and caught directly."""
        with pytest.raises(CircuitOpenError):
            raise CircuitOpenError("open")

    def test_caught_as_middleware_error(self) -> None:
        """CircuitOpenError is caught by a MiddlewareError handler."""
        with pytest.raises(MiddlewareError):
            raise CircuitOpenError("open")

    def test_caught_as_agent_error(self) -> None:
        """CircuitOpenError is caught by an AgentError handler."""
        with pytest.raises(AgentError):
            raise CircuitOpenError("open")

    def test_not_a_provider_error(self) -> None:
        """CircuitOpenError is not a subclass of ProviderError."""
        assert not issubclass(CircuitOpenError, ProviderError)


class TestRetryExhaustedError:
    """Tests for the RetryExhaustedError exception."""

    def test_is_middleware_error(self) -> None:
        """RetryExhaustedError is a subclass of MiddlewareError."""
        assert issubclass(RetryExhaustedError, MiddlewareError)

    def test_is_agent_error(self) -> None:
        """RetryExhaustedError is a subclass of AgentError."""
        assert issubclass(RetryExhaustedError, AgentError)

    def test_is_exception(self) -> None:
        """RetryExhaustedError is a subclass of Exception."""
        assert issubclass(RetryExhaustedError, Exception)

    def test_message_preserved(self) -> None:
        """The error message is accessible via str()."""
        err = RetryExhaustedError("all retries failed", last_error=ValueError("x"))
        assert str(err) == "all retries failed"

    def test_last_error_stored(self) -> None:
        """last_error holds the exception from the final attempt."""
        cause = ProviderError("timeout", status_code=503)
        err = RetryExhaustedError("retries exhausted", last_error=cause)
        assert err.last_error is cause

    def test_last_error_type_preserved(self) -> None:
        """last_error retains its original type."""
        cause = ProviderError("bad gateway", status_code=502)
        err = RetryExhaustedError("done", last_error=cause)
        assert isinstance(err.last_error, ProviderError)
        assert err.last_error.status_code == 502

    def test_caught_as_middleware_error(self) -> None:
        """RetryExhaustedError is caught by a MiddlewareError handler."""
        with pytest.raises(MiddlewareError):
            raise RetryExhaustedError("exhausted", last_error=OSError("conn"))

    def test_caught_as_agent_error(self) -> None:
        """RetryExhaustedError is caught by an AgentError handler."""
        with pytest.raises(AgentError):
            raise RetryExhaustedError("exhausted", last_error=OSError("conn"))

    def test_not_a_circuit_open_error(self) -> None:
        """RetryExhaustedError is not a subclass of CircuitOpenError."""
        assert not issubclass(RetryExhaustedError, CircuitOpenError)

    def test_not_a_provider_error(self) -> None:
        """RetryExhaustedError is not a subclass of ProviderError."""
        assert not issubclass(RetryExhaustedError, ProviderError)


class TestHierarchy:
    """Consolidated isinstance checks across the full error tree."""

    def test_provider_error_isinstance_chain(self) -> None:
        """A ProviderError instance satisfies the full upward chain."""
        err = ProviderError("fail", status_code=500)
        assert isinstance(err, ProviderError)
        assert isinstance(err, AgentError)
        assert isinstance(err, Exception)

    def test_circuit_open_error_isinstance_chain(self) -> None:
        """A CircuitOpenError instance satisfies the full upward chain."""
        err = CircuitOpenError("open")
        assert isinstance(err, CircuitOpenError)
        assert isinstance(err, MiddlewareError)
        assert isinstance(err, AgentError)
        assert isinstance(err, Exception)

    def test_retry_exhausted_error_isinstance_chain(self) -> None:
        """A RetryExhaustedError instance satisfies the full upward chain."""
        err = RetryExhaustedError("done", last_error=OSError("x"))
        assert isinstance(err, RetryExhaustedError)
        assert isinstance(err, MiddlewareError)
        assert isinstance(err, AgentError)
        assert isinstance(err, Exception)

    def test_provider_and_middleware_are_siblings(self) -> None:
        """ProviderError and MiddlewareError are independent branches under AgentError."""
        assert not issubclass(ProviderError, MiddlewareError)
        assert not issubclass(MiddlewareError, ProviderError)

    def test_all_errors_are_agent_errors(self) -> None:
        """Every concrete error type is an AgentError."""
        for cls in (ProviderError, MiddlewareError, CircuitOpenError, RetryExhaustedError):
            assert issubclass(cls, AgentError), f"{cls.__name__} must be an AgentError"

    def test_middleware_subtypes_are_not_provider_errors(self) -> None:
        """No middleware subtype bleeds into the ProviderError branch."""
        for cls in (MiddlewareError, CircuitOpenError, RetryExhaustedError):
            assert not issubclass(cls, ProviderError), f"{cls.__name__} must not be a ProviderError"


class TestIntegration:
    """Realistic error-handling patterns across the hierarchy."""

    def test_retry_wraps_provider_error(self) -> None:
        """RetryExhaustedError correctly wraps a ProviderError as last_error."""
        cause = ProviderError("upstream timeout", status_code=503)
        err = RetryExhaustedError("3 attempts failed", last_error=cause)
        assert isinstance(err.last_error, ProviderError)
        assert err.last_error.status_code == 503
        assert str(err) == "3 attempts failed"
        assert str(err.last_error) == "upstream timeout"

    def test_circuit_open_and_retry_both_caught_as_middleware(self) -> None:
        """Both MiddlewareError subtypes are caught by a single MiddlewareError handler."""
        errors: list[Exception] = [
            CircuitOpenError("open"),
            RetryExhaustedError("exhausted", last_error=OSError("net")),
        ]
        for exc in errors:
            with pytest.raises(MiddlewareError):
                raise exc

    def test_generic_agent_error_handler_catches_all(self) -> None:
        """A single AgentError handler catches every concrete error type."""
        errors: list[Exception] = [
            AgentError("base"),
            ProviderError("provider"),
            MiddlewareError("middleware"),
            CircuitOpenError("circuit"),
            RetryExhaustedError("retry", last_error=ValueError("v")),
        ]
        for exc in errors:
            with pytest.raises(AgentError):
                raise exc

    def test_status_code_accessible_after_catch(self) -> None:
        """status_code is readable on a caught ProviderError."""
        try:
            raise ProviderError("rate limited", status_code=429)
        except ProviderError as exc:
            assert exc.status_code == 429

    def test_last_error_accessible_after_catch(self) -> None:
        """last_error is readable on a caught RetryExhaustedError."""
        cause = ProviderError("bad gateway", status_code=502)
        try:
            raise RetryExhaustedError("gave up", last_error=cause)
        except RetryExhaustedError as exc:
            assert exc.last_error is cause
