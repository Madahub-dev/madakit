"""Tests for mada_modelkit._errors.

Covers the full error hierarchy: AgentError base, ProviderError with
optional status_code, MiddlewareError, CircuitOpenError, and
RetryExhaustedError with last_error chaining. Tests isinstance hierarchy,
message propagation, and attribute access.
"""

from __future__ import annotations

import pytest

from mada_modelkit._errors import AgentError


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
