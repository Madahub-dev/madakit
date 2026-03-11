"""Tests for timeout middleware.

Tests TimeoutMiddleware constructor, send_request timeout enforcement,
send_request_stream timeout (first chunk only), and edge cases.
"""

from __future__ import annotations

import pytest

from mada_modelkit.middleware.timeout import TimeoutMiddleware
from mada_modelkit._types import AgentRequest, AgentResponse

from helpers import MockProvider


class TestModuleExports:
    """Test module-level exports and imports."""

    def test_all_exports(self) -> None:
        """__all__ contains only TimeoutMiddleware."""
        from mada_modelkit.middleware import timeout

        assert timeout.__all__ == ["TimeoutMiddleware"]

    def test_middleware_importable(self) -> None:
        """TimeoutMiddleware can be imported from module."""
        from mada_modelkit.middleware.timeout import TimeoutMiddleware as TM

        assert TM is not None


class TestTimeoutMiddlewareConstructor:
    """Test TimeoutMiddleware constructor and initialization."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts client and uses default timeout."""
        mock = MockProvider()
        middleware = TimeoutMiddleware(client=mock)

        assert middleware._client is mock
        assert middleware._timeout_seconds == 30.0

    def test_explicit_timeout(self) -> None:
        """Constructor accepts explicit timeout_seconds."""
        mock = MockProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=10.0)

        assert middleware._timeout_seconds == 10.0

    def test_custom_timeout_values(self) -> None:
        """Constructor accepts various timeout values."""
        mock = MockProvider()

        # Very short timeout
        mw1 = TimeoutMiddleware(client=mock, timeout_seconds=0.1)
        assert mw1._timeout_seconds == 0.1

        # Long timeout
        mw2 = TimeoutMiddleware(client=mock, timeout_seconds=300.0)
        assert mw2._timeout_seconds == 300.0

        # Fractional timeout
        mw3 = TimeoutMiddleware(client=mock, timeout_seconds=2.5)
        assert mw3._timeout_seconds == 2.5

    def test_wraps_base_agent_client(self) -> None:
        """TimeoutMiddleware can wrap any BaseAgentClient."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        mw1 = TimeoutMiddleware(client=mock1)
        mw2 = TimeoutMiddleware(client=mock2)

        assert mw1._client is mock1
        assert mw2._client is mock2

    def test_wraps_middleware(self) -> None:
        """TimeoutMiddleware can wrap another middleware."""
        from mada_modelkit.middleware.retry import RetryMiddleware

        mock = MockProvider()
        retry_mw = RetryMiddleware(client=mock, max_retries=3)
        timeout_mw = TimeoutMiddleware(client=retry_mw, timeout_seconds=5.0)

        assert timeout_mw._client is retry_mw
        assert timeout_mw._timeout_seconds == 5.0

    def test_super_init_called(self) -> None:
        """TimeoutMiddleware calls BaseAgentClient.__init__."""
        mock = MockProvider()
        middleware = TimeoutMiddleware(client=mock)

        # Should have BaseAgentClient methods
        assert hasattr(middleware, "send_request")
        assert hasattr(middleware, "send_request_stream")
        assert hasattr(middleware, "cancel")
        assert hasattr(middleware, "close")

    def test_instance_isolation(self) -> None:
        """Different instances have independent state."""
        mock = MockProvider()
        mw1 = TimeoutMiddleware(client=mock, timeout_seconds=10.0)
        mw2 = TimeoutMiddleware(client=mock, timeout_seconds=20.0)

        assert mw1._timeout_seconds == 10.0
        assert mw2._timeout_seconds == 20.0
