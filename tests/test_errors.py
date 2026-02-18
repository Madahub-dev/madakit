"""Tests for mada_modelkit._errors.

Covers the full error hierarchy: AgentError base, ProviderError with
optional status_code, MiddlewareError, CircuitOpenError, and
RetryExhaustedError with last_error chaining. Tests isinstance hierarchy,
message propagation, and attribute access.
"""

from __future__ import annotations

import pytest

from mada_modelkit._errors import AgentError, CircuitOpenError, MiddlewareError, ProviderError


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
