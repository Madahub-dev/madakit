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


class TestSendRequestWithTimeout:
    """Test send_request timeout enforcement."""

    @pytest.mark.asyncio
    async def test_fast_request_succeeds(self) -> None:
        """Fast requests complete successfully within timeout."""
        mock = MockProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert response.content == "mock"
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_slow_request_raises_timeout(self) -> None:
        """Slow requests raise asyncio.TimeoutError."""
        import asyncio

        # Mock with 0.5s latency
        mock = MockProvider(latency=0.5)
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=0.1)

        request = AgentRequest(prompt="test")

        with pytest.raises(asyncio.TimeoutError):
            await middleware.send_request(request)

    @pytest.mark.asyncio
    async def test_request_at_timeout_boundary(self) -> None:
        """Request that completes exactly at timeout boundary."""
        import asyncio

        # Request takes ~0.05s, timeout is 0.1s
        mock = MockProvider(latency=0.05)
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=0.1)

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert response.content == "mock"

    @pytest.mark.asyncio
    async def test_timeout_value_respected(self) -> None:
        """Different timeout values are respected."""
        import asyncio
        import time

        # Request takes 0.2s
        mock = MockProvider(latency=0.2)

        # Short timeout fails
        mw_short = TimeoutMiddleware(client=mock, timeout_seconds=0.1)
        request = AgentRequest(prompt="test")

        with pytest.raises(asyncio.TimeoutError):
            await mw_short.send_request(request)

        # Longer timeout succeeds
        mock2 = MockProvider(latency=0.2)
        mw_long = TimeoutMiddleware(client=mock2, timeout_seconds=0.5)

        response = await mw_long.send_request(request)
        assert response.content == "mock"

    @pytest.mark.asyncio
    async def test_response_returned_correctly(self) -> None:
        """Response from wrapped client is returned unchanged."""
        custom_response = AgentResponse(
            content="custom", model="test-model", input_tokens=100, output_tokens=200
        )
        mock = MockProvider(responses=[custom_response])
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert response is custom_response
        assert response.content == "custom"
        assert response.model == "test-model"
        assert response.input_tokens == 100

    @pytest.mark.asyncio
    async def test_exception_propagation_before_timeout(self) -> None:
        """Exceptions from wrapped client propagate before timeout."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error", status_code=500)

        mock = FailingProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError, match="API error"):
            await middleware.send_request(request)

    @pytest.mark.asyncio
    async def test_very_short_timeout(self) -> None:
        """Very short timeouts (milliseconds) work correctly."""
        import asyncio

        mock = MockProvider(latency=0.1)
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=0.01)

        request = AgentRequest(prompt="test")

        with pytest.raises(asyncio.TimeoutError):
            await middleware.send_request(request)

    @pytest.mark.asyncio
    async def test_multiple_requests_independent_timeouts(self) -> None:
        """Each request has independent timeout enforcement."""
        import asyncio

        mock = MockProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=0.1)

        request = AgentRequest(prompt="test")

        # First request succeeds
        await middleware.send_request(request)

        # Second request with slow mock times out
        mock_slow = MockProvider(latency=0.2)
        mw_slow = TimeoutMiddleware(client=mock_slow, timeout_seconds=0.1)

        with pytest.raises(asyncio.TimeoutError):
            await mw_slow.send_request(request)

        # Third request succeeds again
        await middleware.send_request(request)

        assert mock.call_count == 2
