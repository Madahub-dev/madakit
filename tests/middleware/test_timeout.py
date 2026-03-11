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


class TestSendRequestStreamWithTimeout:
    """Test send_request_stream timeout enforcement (first chunk only)."""

    @pytest.mark.asyncio
    async def test_fast_stream_succeeds(self) -> None:
        """Fast streams complete successfully within timeout."""

        class MultiChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(delta="chunk2", is_final=False)
                yield StreamChunk(delta="chunk3", is_final=True)

        mock = MultiChunkProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta == "chunk1"
        assert chunks[2].delta == "chunk3"
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_slow_first_chunk_raises_timeout(self) -> None:
        """Stream with slow first chunk raises asyncio.TimeoutError."""
        import asyncio

        class SlowFirstChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                await asyncio.sleep(0.3)  # Delay before first chunk
                yield StreamChunk(delta="chunk1", is_final=True)

        mock = SlowFirstChunkProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=0.1)

        request = AgentRequest(prompt="test")

        with pytest.raises(asyncio.TimeoutError):
            async for _ in middleware.send_request_stream(request):
                pass

    @pytest.mark.asyncio
    async def test_subsequent_chunks_not_subject_to_timeout(self) -> None:
        """After first chunk, subsequent chunks can be slow without timeout."""
        import asyncio

        class SlowSubsequentChunksProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                # First chunk arrives quickly
                yield StreamChunk(delta="chunk1", is_final=False)
                # Subsequent chunks are slow
                await asyncio.sleep(0.3)
                yield StreamChunk(delta="chunk2", is_final=False)
                await asyncio.sleep(0.3)
                yield StreamChunk(delta="chunk3", is_final=True)

        mock = SlowSubsequentChunksProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=0.1)

        request = AgentRequest(prompt="test")
        chunks = []

        # Should succeed despite slow subsequent chunks
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta == "chunk1"
        assert chunks[1].delta == "chunk2"
        assert chunks[2].delta == "chunk3"

    @pytest.mark.asyncio
    async def test_single_chunk_stream(self) -> None:
        """Stream with single chunk works correctly."""

        class SingleChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="only chunk", is_final=True)

        mock = SingleChunkProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].delta == "only chunk"
        assert chunks[0].is_final is True

    @pytest.mark.asyncio
    async def test_stream_exception_propagation(self) -> None:
        """Exceptions from wrapped client stream propagate correctly."""
        from mada_modelkit._errors import ProviderError

        class FailingStreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                raise ProviderError("Stream error", status_code=500)
                yield  # Make it a generator

        mock = FailingStreamProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")

        with pytest.raises(ProviderError, match="Stream error"):
            async for _ in middleware.send_request_stream(request):
                pass

    @pytest.mark.asyncio
    async def test_timeout_value_respected_for_first_chunk(self) -> None:
        """Different timeout values are respected for first chunk."""
        import asyncio

        class DelayedFirstChunkProvider(MockProvider):
            def __init__(self, delay):
                super().__init__()
                self.delay = delay

            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                await asyncio.sleep(self.delay)
                yield StreamChunk(delta="chunk", is_final=True)

        request = AgentRequest(prompt="test")

        # 0.2s delay with 0.1s timeout fails
        mock1 = DelayedFirstChunkProvider(delay=0.2)
        mw1 = TimeoutMiddleware(client=mock1, timeout_seconds=0.1)

        with pytest.raises(asyncio.TimeoutError):
            async for _ in mw1.send_request_stream(request):
                pass

        # 0.2s delay with 0.5s timeout succeeds
        mock2 = DelayedFirstChunkProvider(delay=0.2)
        mw2 = TimeoutMiddleware(client=mock2, timeout_seconds=0.5)

        chunks = []
        async for chunk in mw2.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 1

    @pytest.mark.asyncio
    async def test_empty_stream_times_out(self) -> None:
        """Stream that yields no chunks times out waiting for first chunk."""
        import asyncio

        class EmptyStreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                await asyncio.sleep(0.5)
                return
                yield  # Make it a generator

        mock = EmptyStreamProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=0.1)

        request = AgentRequest(prompt="test")

        with pytest.raises(asyncio.TimeoutError):
            async for _ in middleware.send_request_stream(request):
                pass


class TestTimeoutComprehensive:
    """Comprehensive integration tests for TimeoutMiddleware."""

    @pytest.mark.asyncio
    async def test_mixed_request_types_both_enforced(self) -> None:
        """Both send_request and send_request_stream respect timeout."""
        import asyncio

        class SlowProvider(MockProvider):
            async def send_request(self, request):
                await asyncio.sleep(0.3)
                return await super().send_request(request)

            async def send_request_stream(self, request):
                await asyncio.sleep(0.3)
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="test", is_final=True)

        mock = SlowProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=0.1)

        request = AgentRequest(prompt="test")

        # send_request times out
        with pytest.raises(asyncio.TimeoutError):
            await middleware.send_request(request)

        # send_request_stream times out
        with pytest.raises(asyncio.TimeoutError):
            async for _ in middleware.send_request_stream(request):
                pass

    @pytest.mark.asyncio
    async def test_middleware_composition_with_retry(self) -> None:
        """TimeoutMiddleware stacks with RetryMiddleware."""
        import asyncio
        from mada_modelkit.middleware.retry import RetryMiddleware
        from mada_modelkit._errors import ProviderError

        class FlakeyProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.attempt = 0

            async def send_request(self, request):
                self.attempt += 1
                if self.attempt == 1:
                    raise ProviderError("Transient error", status_code=500)
                return await super().send_request(request)

        mock = FlakeyProvider()
        retry_mw = RetryMiddleware(client=mock, max_retries=2, backoff_base=0.01)
        timeout_mw = TimeoutMiddleware(client=retry_mw, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")
        response = await timeout_mw.send_request(request)

        assert response.content == "mock"
        assert mock.attempt == 2  # Retried once

    @pytest.mark.asyncio
    async def test_middleware_composition_with_cost_control(self) -> None:
        """TimeoutMiddleware stacks with CostControlMiddleware."""
        import asyncio
        from mada_modelkit.middleware.cost_control import CostControlMiddleware

        mock = MockProvider()
        cost_fn = lambda resp: 1.5
        cost_mw = CostControlMiddleware(client=mock, cost_fn=cost_fn)
        timeout_mw = TimeoutMiddleware(client=cost_mw, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")
        response = await timeout_mw.send_request(request)

        assert response.content == "mock"
        assert cost_mw.total_spend == 1.5

    @pytest.mark.asyncio
    async def test_timeout_wraps_cost_control_slow_request(self) -> None:
        """Timeout prevents slow request from incurring cost."""
        import asyncio
        from mada_modelkit.middleware.cost_control import CostControlMiddleware

        mock = MockProvider(latency=0.5)
        cost_fn = lambda resp: 10.0
        cost_mw = CostControlMiddleware(client=mock, cost_fn=cost_fn)
        timeout_mw = TimeoutMiddleware(client=cost_mw, timeout_seconds=0.1)

        request = AgentRequest(prompt="test")

        with pytest.raises(asyncio.TimeoutError):
            await timeout_mw.send_request(request)

        # No cost incurred because request timed out before completion
        assert cost_mw.total_spend == 0.0

    @pytest.mark.asyncio
    async def test_context_manager_support(self) -> None:
        """TimeoutMiddleware works as async context manager."""
        mock = MockProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")

        async with middleware:
            response = await middleware.send_request(request)

        assert response.content == "mock"

    @pytest.mark.asyncio
    async def test_concurrent_requests_independent_timeouts(self) -> None:
        """Concurrent requests have independent timeout enforcement."""
        import asyncio

        class VariableDelayProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.delays = [0.0, 0.3, 0.0]  # Second request is slow
                self.index = 0

            async def send_request(self, request):
                delay = self.delays[self.index]
                self.index += 1
                await asyncio.sleep(delay)
                return await super().send_request(request)

        mock = VariableDelayProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=0.1)

        request = AgentRequest(prompt="test")

        # Run 3 concurrent requests
        results = await asyncio.gather(
            middleware.send_request(request),
            middleware.send_request(request),
            middleware.send_request(request),
            return_exceptions=True,
        )

        # First and third succeed, second times out
        assert isinstance(results[0], AgentResponse)
        assert isinstance(results[1], asyncio.TimeoutError)
        assert isinstance(results[2], AgentResponse)

    @pytest.mark.asyncio
    async def test_timeout_doesnt_swallow_other_exceptions(self) -> None:
        """Timeout middleware doesn't hide non-timeout exceptions."""
        from mada_modelkit._errors import ProviderError

        class FailingProvider(MockProvider):
            async def send_request(self, request):
                raise ProviderError("API error", status_code=500)

        mock = FailingProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")

        # ProviderError propagates, not TimeoutError
        with pytest.raises(ProviderError, match="API error"):
            await middleware.send_request(request)

    @pytest.mark.asyncio
    async def test_long_timeout_for_expensive_operations(self) -> None:
        """Long timeouts work for expensive operations."""
        import asyncio

        # Simulate expensive operation (0.5s)
        mock = MockProvider(latency=0.5)
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=10.0)

        request = AgentRequest(prompt="test")

        # Should succeed with long timeout
        response = await middleware.send_request(request)
        assert response.content == "mock"

    @pytest.mark.asyncio
    async def test_zero_timeout_immediately_fails(self) -> None:
        """Zero or very small timeout causes immediate failure."""
        import asyncio

        mock = MockProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=0.0)

        request = AgentRequest(prompt="test")

        # Zero timeout should fail immediately
        with pytest.raises(asyncio.TimeoutError):
            await middleware.send_request(request)

    @pytest.mark.asyncio
    async def test_stream_timeout_with_metadata_in_final_chunk(self) -> None:
        """Stream timeout works when final chunk has metadata."""

        class MetadataStreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="Hello", is_final=False)
                yield StreamChunk(delta=" world", is_final=False)
                yield StreamChunk(
                    delta="!",
                    is_final=True,
                    metadata={"model": "test-model", "input_tokens": 10, "output_tokens": 20},
                )

        mock = MetadataStreamProvider()
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[2].is_final is True
        assert chunks[2].metadata["model"] == "test-model"
        assert chunks[2].metadata["input_tokens"] == 10

    @pytest.mark.asyncio
    async def test_timeout_preserves_response_metadata(self) -> None:
        """Timeout middleware preserves response metadata."""
        custom_response = AgentResponse(
            content="test",
            model="test-model",
            input_tokens=50,
            output_tokens=100,
            metadata={"custom_key": "custom_value"},
        )
        mock = MockProvider(responses=[custom_response])
        middleware = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert response.metadata["custom_key"] == "custom_value"
        assert response.model == "test-model"
        assert response.input_tokens == 50

    @pytest.mark.asyncio
    async def test_multiple_middleware_layers_with_timeouts(self) -> None:
        """Multiple TimeoutMiddleware layers work correctly."""
        import asyncio

        mock = MockProvider(latency=0.2)

        # Inner timeout is longer (1.0s)
        inner_timeout = TimeoutMiddleware(client=mock, timeout_seconds=1.0)

        # Outer timeout is shorter (0.1s)
        outer_timeout = TimeoutMiddleware(client=inner_timeout, timeout_seconds=0.1)

        request = AgentRequest(prompt="test")

        # Outer timeout triggers first
        with pytest.raises(asyncio.TimeoutError):
            await outer_timeout.send_request(request)

    @pytest.mark.asyncio
    async def test_timeout_with_streaming_and_composition(self) -> None:
        """Timeout works correctly with streaming in middleware stack."""
        import asyncio
        from mada_modelkit.middleware.cost_control import CostControlMiddleware

        class TokenCountingStreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk

                yield StreamChunk(delta="test", is_final=False)
                yield StreamChunk(delta="", is_final=True, metadata={"input_tokens": 10, "output_tokens": 20})

        mock = TokenCountingStreamProvider()
        cost_fn = lambda resp: resp.input_tokens * 0.01 + resp.output_tokens * 0.02
        cost_mw = CostControlMiddleware(client=mock, cost_fn=cost_fn)
        timeout_mw = TimeoutMiddleware(client=cost_mw, timeout_seconds=1.0)

        request = AgentRequest(prompt="test")
        chunks = []
        async for chunk in timeout_mw.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 2
        # Cost should be tracked: 10 * 0.01 + 20 * 0.02 = 0.5
        assert cost_mw.total_spend == 0.5
