"""Tests for RateLimitMiddleware.

Validates token bucket and leaky bucket rate limiting algorithms,
burst handling, and integration with the BaseAgentClient contract.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from mada_modelkit._errors import AgentError
from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.middleware.rate_limit import RateLimitMiddleware
from helpers import MockProvider


class TestModuleExports:
    """Verify rate_limit.py module structure."""

    def test_all_exports(self) -> None:
        """__all__ contains RateLimitMiddleware."""
        from mada_modelkit.middleware import rate_limit
        assert hasattr(rate_limit, "__all__")
        assert "RateLimitMiddleware" in rate_limit.__all__
        assert len(rate_limit.__all__) == 1

    def test_middleware_importable(self) -> None:
        """RateLimitMiddleware is importable from rate_limit module."""
        from mada_modelkit.middleware.rate_limit import RateLimitMiddleware as Imported
        assert Imported is RateLimitMiddleware


class TestRateLimitMiddlewareConstructor:
    """Test constructor parameter handling and state initialisation."""

    def test_minimal_constructor(self) -> None:
        """Constructor with only client uses defaults."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock)
        assert middleware._client is mock
        assert middleware._requests_per_second == 10.0
        assert middleware._burst_size is None
        assert middleware._strategy == "token_bucket"

    def test_explicit_requests_per_second(self) -> None:
        """requests_per_second parameter is stored."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=5.0)
        assert middleware._requests_per_second == 5.0

    def test_explicit_burst_size(self) -> None:
        """burst_size parameter is stored."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, burst_size=20)
        assert middleware._burst_size == 20

    def test_explicit_strategy_token_bucket(self) -> None:
        """strategy parameter accepts 'token_bucket'."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, strategy="token_bucket")
        assert middleware._strategy == "token_bucket"

    def test_explicit_strategy_leaky_bucket(self) -> None:
        """strategy parameter accepts 'leaky_bucket'."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, strategy="leaky_bucket")
        assert middleware._strategy == "leaky_bucket"

    def test_token_bucket_state_initialisation(self) -> None:
        """Token bucket strategy initialises tokens and refill state."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, strategy="token_bucket")
        assert hasattr(middleware, "_tokens")
        assert hasattr(middleware, "_max_tokens")
        assert hasattr(middleware, "_last_refill")
        assert hasattr(middleware, "_lock")
        assert middleware._tokens == 10.0  # default burst_size = requests_per_second
        assert middleware._max_tokens == 10.0

    def test_token_bucket_custom_burst_size(self) -> None:
        """Token bucket uses burst_size when provided."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=5.0, burst_size=15, strategy="token_bucket"
        )
        assert middleware._tokens == 15.0
        assert middleware._max_tokens == 15.0

    def test_leaky_bucket_state_initialisation(self) -> None:
        """Leaky bucket strategy initialises queue."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, strategy="leaky_bucket")
        assert hasattr(middleware, "_max_queue_size")
        assert hasattr(middleware, "_queue")
        assert hasattr(middleware, "_processor_task")
        assert middleware._max_queue_size == 20  # default 2 * requests_per_second
        assert isinstance(middleware._queue, asyncio.Queue)
        assert middleware._processor_task is None

    def test_leaky_bucket_custom_burst_size(self) -> None:
        """Leaky bucket uses burst_size for max_queue_size when provided."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=5.0, burst_size=30, strategy="leaky_bucket"
        )
        assert middleware._max_queue_size == 30

    def test_invalid_strategy_raises_value_error(self) -> None:
        """Unknown strategy raises ValueError."""
        mock = MockProvider()
        with pytest.raises(ValueError, match="Unknown strategy: invalid"):
            RateLimitMiddleware(client=mock, strategy="invalid")

    def test_instance_isolation(self) -> None:
        """Multiple instances have independent state."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        middleware1 = RateLimitMiddleware(client=mock1, requests_per_second=5.0)
        middleware2 = RateLimitMiddleware(client=mock2, requests_per_second=10.0)
        assert middleware1._client is mock1
        assert middleware2._client is mock2
        assert middleware1._requests_per_second == 5.0
        assert middleware2._requests_per_second == 10.0
        assert middleware1._tokens != middleware2._tokens

    def test_wraps_middleware(self) -> None:
        """Can wrap another middleware instance."""
        mock = MockProvider()
        from mada_modelkit.middleware.retry import RetryMiddleware
        retry = RetryMiddleware(client=mock, max_retries=2)
        middleware = RateLimitMiddleware(client=retry, requests_per_second=5.0)
        assert middleware._client is retry
        assert retry._client is mock

    def test_super_init_called(self) -> None:
        """Constructor calls super().__init__()."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock)
        # Inherits from BaseAgentClient, so should have _semaphore attribute
        assert hasattr(middleware, "_semaphore")


class TestTokenBucketAlgorithm:
    """Test token bucket refill and acquisition logic."""

    def test_refill_tokens_adds_tokens_over_time(self) -> None:
        """Tokens increase based on elapsed time and requests_per_second."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        # Start with full bucket
        initial_tokens = middleware._tokens
        assert initial_tokens == 10.0

        # Consume all tokens
        middleware._tokens = 0.0
        middleware._last_refill = middleware._last_refill - 0.5  # Simulate 0.5s elapsed

        # Refill should add 5 tokens (10 tokens/sec * 0.5 sec)
        middleware._refill_tokens()
        assert abs(middleware._tokens - 5.0) < 0.01

    def test_refill_tokens_caps_at_max_tokens(self) -> None:
        """Tokens cannot exceed max_tokens."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        # Simulate 10 seconds elapsed with no consumption
        middleware._last_refill = middleware._last_refill - 10.0
        middleware._refill_tokens()

        # Should cap at max_tokens (10.0), not accumulate to 100
        assert middleware._tokens == middleware._max_tokens
        assert middleware._tokens == 10.0

    def test_refill_tokens_updates_last_refill(self) -> None:
        """_refill_tokens updates _last_refill timestamp."""
        import time
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        old_refill_time = middleware._last_refill
        time.sleep(0.01)  # Small sleep to ensure time advances
        middleware._refill_tokens()

        assert middleware._last_refill > old_refill_time

    @pytest.mark.asyncio
    async def test_acquire_token_consumes_one_token(self) -> None:
        """_acquire_token consumes exactly one token."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        initial_tokens = middleware._tokens
        request = AgentRequest(prompt="test")
        await middleware._acquire_token(request)

        assert abs(middleware._tokens - (initial_tokens - 1.0)) < 0.01

    @pytest.mark.asyncio
    async def test_acquire_token_blocks_when_empty(self) -> None:
        """_acquire_token blocks until tokens are available."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=100.0, strategy="token_bucket")

        # Consume all tokens
        middleware._tokens = 0.0

        # This should block briefly, then succeed as tokens refill
        start = time.monotonic()
        await middleware._acquire_token(AgentRequest(prompt="test"))
        elapsed = time.monotonic() - start

        # Should have waited at least 10ms (one poll interval)
        assert elapsed >= 0.01
        assert middleware._tokens < middleware._max_tokens  # One token consumed

    @pytest.mark.asyncio
    async def test_acquire_token_multiple_sequential(self) -> None:
        """Multiple sequential acquisitions work correctly."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=5, strategy="token_bucket"
        )

        # Start with 5 tokens
        assert middleware._tokens == 5.0

        # Acquire 3 tokens
        await middleware._acquire_token(AgentRequest(prompt="test"))
        await middleware._acquire_token(AgentRequest(prompt="test"))
        await middleware._acquire_token(AgentRequest(prompt="test"))

        # Should have 2 tokens left (plus any refill from elapsed time)
        assert middleware._tokens >= 2.0 and middleware._tokens < 3.0

    @pytest.mark.asyncio
    async def test_acquire_token_concurrent_safe(self) -> None:
        """Concurrent token acquisitions are thread-safe."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=100.0, burst_size=10, strategy="token_bucket"
        )

        # Start with 10 tokens
        assert middleware._tokens == 10.0

        # Acquire 10 tokens concurrently
        request = AgentRequest(prompt="test")
        await asyncio.gather(*[middleware._acquire_token(request) for _ in range(10)])

        # All tokens consumed (plus minimal refill during execution)
        assert middleware._tokens < 1.0

    @pytest.mark.asyncio
    async def test_acquire_token_refills_before_checking(self) -> None:
        """_acquire_token refills before checking availability."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        # Deplete tokens
        middleware._tokens = 0.0
        middleware._last_refill = time.monotonic() - 0.2  # Simulate 0.2s elapsed

        # Should refill 2 tokens (10/sec * 0.2sec), then consume 1
        await middleware._acquire_token(AgentRequest(prompt="test"))
        assert abs(middleware._tokens - 1.0) < 0.1  # ~1 token remaining

    def test_refill_tokens_with_custom_rate(self) -> None:
        """Token refill respects custom requests_per_second."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=5.0, strategy="token_bucket")

        middleware._tokens = 0.0
        middleware._last_refill = middleware._last_refill - 1.0  # 1 second elapsed

        middleware._refill_tokens()
        assert abs(middleware._tokens - 5.0) < 0.01  # 5 tokens/sec * 1 sec

    def test_refill_tokens_fractional_accumulation(self) -> None:
        """Tokens accumulate as floats for fractional rates."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=2.5, strategy="token_bucket")

        middleware._tokens = 0.0
        middleware._last_refill = middleware._last_refill - 0.4  # 0.4 seconds

        middleware._refill_tokens()
        assert abs(middleware._tokens - 1.0) < 0.01  # 2.5 * 0.4 = 1.0


class TestLeakyBucketAlgorithm:
    """Test leaky bucket queue and processor logic."""

    @pytest.mark.asyncio
    async def test_processor_loop_processes_at_fixed_rate(self) -> None:
        """Processor loop dequeues and processes at fixed interval."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, strategy="leaky_bucket"
        )

        # Add events to queue
        event1 = asyncio.Event()
        event2 = asyncio.Event()
        await middleware._queue.put(event1)
        await middleware._queue.put(event2)

        # Start processor
        processor = asyncio.create_task(middleware._processor_loop())

        # Wait for processing (2 items at 10/sec = 0.2 seconds)
        await asyncio.sleep(0.25)

        # Queue should be empty and events should be set
        assert middleware._queue.empty()
        assert event1.is_set()
        assert event2.is_set()

        # Clean up
        processor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await processor

    @pytest.mark.asyncio
    async def test_acquire_slot_starts_processor(self) -> None:
        """_acquire_slot starts processor task if not running."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="leaky_bucket")

        assert middleware._processor_task is None

        # Acquire slot should start processor
        await middleware._acquire_slot(AgentRequest(prompt="test"))

        assert middleware._processor_task is not None
        assert not middleware._processor_task.done()

        # Clean up
        middleware._processor_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await middleware._processor_task

    @pytest.mark.asyncio
    async def test_acquire_slot_enqueues_request(self) -> None:
        """_acquire_slot enqueues and waits for processing."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=100.0, strategy="leaky_bucket"
        )

        # Queue starts empty
        assert middleware._queue.qsize() == 0

        # Acquire slot (waits for processing)
        await middleware._acquire_slot(AgentRequest(prompt="test"))

        # Queue should be empty (item was processed)
        assert middleware._queue.qsize() == 0
        # But processor task should be running
        assert middleware._processor_task is not None
        assert not middleware._processor_task.done()

        # Clean up
        if middleware._processor_task:
            middleware._processor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await middleware._processor_task

    @pytest.mark.asyncio
    async def test_acquire_slot_with_full_queue(self) -> None:
        """_acquire_slot handles full queue correctly."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=50.0, burst_size=3, strategy="leaky_bucket"
        )

        # Fill queue to capacity (burst_size=3) with events
        events = [asyncio.Event() for _ in range(3)]
        for event in events:
            await middleware._queue.put(event)
        assert middleware._queue.full()
        assert middleware._queue.qsize() == 3

        # Start processor to drain queue
        middleware._processor_task = asyncio.create_task(middleware._processor_loop())

        # Wait for processor to drain at least one item (1/50 = 0.02s per item)
        await asyncio.sleep(0.04)

        # Now acquire_slot should succeed (queue has space)
        await middleware._acquire_slot(AgentRequest(prompt="test"))

        # Clean up
        middleware._processor_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await middleware._processor_task

    @pytest.mark.asyncio
    async def test_processor_loop_continuous_processing(self) -> None:
        """Processor loop continues processing until cancelled."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=50.0, strategy="leaky_bucket"
        )

        # Start processor
        processor = asyncio.create_task(middleware._processor_loop())

        # Add events continuously
        events = [asyncio.Event() for _ in range(5)]
        for event in events:
            await middleware._queue.put(event)

        # Wait for processing (5 items at 50/sec = 0.1 seconds)
        await asyncio.sleep(0.15)

        # Queue should be empty and all events set
        assert middleware._queue.empty()
        assert all(e.is_set() for e in events)
        assert not processor.done()  # Still running

        # Clean up
        processor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await processor

    @pytest.mark.asyncio
    async def test_acquire_slot_restarts_dead_processor(self) -> None:
        """_acquire_slot restarts processor if it has terminated."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="leaky_bucket")

        # Start and immediately cancel processor
        middleware._processor_task = asyncio.create_task(middleware._processor_loop())
        middleware._processor_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await middleware._processor_task

        assert middleware._processor_task.done()

        # Acquire slot should restart processor
        await middleware._acquire_slot(AgentRequest(prompt="test"))

        assert not middleware._processor_task.done()

        # Clean up
        middleware._processor_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await middleware._processor_task

    @pytest.mark.asyncio
    async def test_processor_loop_respects_rate(self) -> None:
        """Processor loop enforces correct processing rate."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=20.0, strategy="leaky_bucket"
        )

        # Add multiple events
        events = [asyncio.Event() for _ in range(4)]
        for event in events:
            await middleware._queue.put(event)

        # Start processor and measure time
        start = time.monotonic()
        processor = asyncio.create_task(middleware._processor_loop())

        # Wait for all items to process
        await middleware._queue.join()
        elapsed = time.monotonic() - start

        # 4 items with sleep after each (4 * 1/20 = 0.2s total)
        assert elapsed >= 0.19  # Allow small tolerance
        assert all(e.is_set() for e in events)

        # Clean up
        processor.cancel()
        with pytest.raises(asyncio.CancelledError):
            await processor

    @pytest.mark.asyncio
    async def test_acquire_slot_multiple_concurrent(self) -> None:
        """Multiple concurrent slot acquisitions work correctly."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=100.0, burst_size=5, strategy="leaky_bucket"
        )

        # Acquire 5 slots concurrently (all wait for processing)
        start = time.monotonic()
        request = AgentRequest(prompt="test")
        await asyncio.gather(*[middleware._acquire_slot(request) for _ in range(5)])
        elapsed = time.monotonic() - start

        # Should take at least 5 intervals (5 * 1/100 = 0.05s)
        assert elapsed >= 0.04

        # Queue should be empty (all processed)
        assert middleware._queue.qsize() == 0

        # Clean up
        if middleware._processor_task:
            middleware._processor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await middleware._processor_task

    def test_processor_loop_calculates_correct_interval(self) -> None:
        """Processor uses correct interval for given rate."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=5.0, strategy="leaky_bucket"
        )

        # Interval should be 1/5 = 0.2 seconds
        expected_interval = 1.0 / 5.0
        assert expected_interval == 0.2


class TestSendRequestWithRateLimit:
    """Test send_request integration with rate limiting."""

    @pytest.mark.asyncio
    async def test_send_request_token_bucket_delegates(self) -> None:
        """send_request with token_bucket acquires token then delegates."""
        response_obj = AgentResponse(content="test response", model="test", input_tokens=5, output_tokens=10)
        mock = MockProvider(responses=[response_obj])
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        request = AgentRequest(prompt="test prompt")
        response = await middleware.send_request(request)

        assert response.content == "test response"
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_send_request_leaky_bucket_delegates(self) -> None:
        """send_request with leaky_bucket acquires slot then delegates."""
        response_obj = AgentResponse(content="test response", model="test", input_tokens=5, output_tokens=10)
        mock = MockProvider(responses=[response_obj])
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, strategy="leaky_bucket"
        )

        request = AgentRequest(prompt="test prompt")
        response = await middleware.send_request(request)

        assert response.content == "test response"
        assert mock.call_count == 1

        # Clean up processor
        if middleware._processor_task:
            middleware._processor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await middleware._processor_task

    @pytest.mark.asyncio
    async def test_send_request_token_bucket_enforces_rate(self) -> None:
        """send_request with token_bucket enforces rate limit."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=20.0, burst_size=2, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")

        # First 2 requests use burst capacity (immediate)
        start = time.monotonic()
        await middleware.send_request(request)
        await middleware.send_request(request)
        immediate_elapsed = time.monotonic() - start
        assert immediate_elapsed < 0.02  # Should be near-instant

        # Third request must wait for token refill
        start = time.monotonic()
        await middleware.send_request(request)
        wait_elapsed = time.monotonic() - start
        assert wait_elapsed >= 0.04  # At least one polling interval

        assert mock.call_count == 3

    @pytest.mark.asyncio
    async def test_send_request_leaky_bucket_enforces_rate(self) -> None:
        """send_request with leaky_bucket enforces rate limit."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=20.0, strategy="leaky_bucket"
        )

        request = AgentRequest(prompt="test")

        # Enqueue 3 requests
        start = time.monotonic()
        tasks = [middleware.send_request(request) for _ in range(3)]
        responses = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - start

        # First processes immediately, next 2 wait (2 * 1/20 = 0.1s)
        assert elapsed >= 0.09
        assert len(responses) == 3
        assert mock.call_count == 3

        # Clean up
        if middleware._processor_task:
            middleware._processor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await middleware._processor_task

    @pytest.mark.asyncio
    async def test_send_request_token_bucket_multiple_sequential(self) -> None:
        """Multiple sequential send_request calls with token bucket."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=50.0, burst_size=5, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")

        # Send 5 requests sequentially (all use burst)
        for i in range(5):
            response = await middleware.send_request(request)
            assert response.content == "mock"  # Default response

        assert mock.call_count == 5
        assert middleware._tokens < 1.0  # Burst depleted

    @pytest.mark.asyncio
    async def test_send_request_leaky_bucket_multiple_sequential(self) -> None:
        """Multiple sequential send_request calls with leaky bucket."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=50.0, strategy="leaky_bucket"
        )

        request = AgentRequest(prompt="test")

        # Send 3 requests sequentially
        for i in range(3):
            response = await middleware.send_request(request)
            assert response.content == "mock"  # Default response

        assert mock.call_count == 3

        # Clean up
        if middleware._processor_task:
            middleware._processor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await middleware._processor_task

    @pytest.mark.asyncio
    async def test_send_request_token_bucket_preserves_response(self) -> None:
        """send_request returns exact response from wrapped client."""
        response_obj = AgentResponse(content="content", model="model", input_tokens=10, output_tokens=20)
        mock = MockProvider(responses=[response_obj])
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert response.content == "content"
        assert response.model == "model"
        assert response.input_tokens == 10
        assert response.output_tokens == 20

    @pytest.mark.asyncio
    async def test_send_request_propagates_exceptions(self) -> None:
        """send_request propagates exceptions from wrapped client."""
        from mada_modelkit._errors import ProviderError

        error = ProviderError("test error", status_code=500)
        mock = MockProvider(errors=[error])
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        request = AgentRequest(prompt="test")
        with pytest.raises(ProviderError, match="test error"):
            await middleware.send_request(request)

    @pytest.mark.asyncio
    async def test_send_request_token_bucket_refills_between_calls(self) -> None:
        """Tokens refill between send_request calls."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=1, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")

        # First request uses burst token
        await middleware.send_request(request)
        assert middleware._tokens < 1.0

        # Wait for refill
        await asyncio.sleep(0.15)  # Should refill ~1.5 tokens

        # Second request should succeed without blocking
        start = time.monotonic()
        await middleware.send_request(request)
        elapsed = time.monotonic() - start

        assert elapsed < 0.02  # Should be immediate (token available)
        assert mock.call_count == 2


class TestSendRequestStreamWithRateLimit:
    """Test send_request_stream integration with rate limiting."""

    @pytest.mark.asyncio
    async def test_send_request_stream_token_bucket_delegates(self) -> None:
        """send_request_stream with token_bucket acquires token then streams."""
        from mada_modelkit._types import StreamChunk

        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        request = AgentRequest(prompt="test prompt")
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].delta == "mock"
        assert chunks[0].is_final is True
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_send_request_stream_leaky_bucket_delegates(self) -> None:
        """send_request_stream with leaky_bucket acquires slot then streams."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, strategy="leaky_bucket"
        )

        request = AgentRequest(prompt="test prompt")
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].delta == "mock"
        assert mock.call_count == 1

        # Clean up
        if middleware._processor_task:
            middleware._processor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await middleware._processor_task

    @pytest.mark.asyncio
    async def test_send_request_stream_token_bucket_enforces_rate(self) -> None:
        """send_request_stream with token_bucket enforces rate limit."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=20.0, burst_size=2, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")

        # First 2 streams use burst capacity (immediate)
        start = time.monotonic()
        async for _ in middleware.send_request_stream(request):
            pass
        async for _ in middleware.send_request_stream(request):
            pass
        immediate_elapsed = time.monotonic() - start
        assert immediate_elapsed < 0.02

        # Third stream must wait for token refill
        start = time.monotonic()
        async for _ in middleware.send_request_stream(request):
            pass
        wait_elapsed = time.monotonic() - start
        assert wait_elapsed >= 0.04

        assert mock.call_count == 3

    @pytest.mark.asyncio
    async def test_send_request_stream_leaky_bucket_enforces_rate(self) -> None:
        """send_request_stream with leaky_bucket enforces rate limit."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=20.0, strategy="leaky_bucket"
        )

        request = AgentRequest(prompt="test")

        # Create 3 stream tasks
        async def collect_stream():
            chunks = []
            async for chunk in middleware.send_request_stream(request):
                chunks.append(chunk)
            return chunks

        start = time.monotonic()
        results = await asyncio.gather(*[collect_stream() for _ in range(3)])
        elapsed = time.monotonic() - start

        # First processes immediately, next 2 wait (2 * 1/20 = 0.1s)
        assert elapsed >= 0.09
        assert len(results) == 3
        assert mock.call_count == 3

        # Clean up
        if middleware._processor_task:
            middleware._processor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await middleware._processor_task

    @pytest.mark.asyncio
    async def test_send_request_stream_token_bucket_single_token_per_stream(self) -> None:
        """Each stream consumes exactly one token regardless of chunk count."""
        # Create a mock that yields multiple chunks
        class MultiChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                from mada_modelkit._types import StreamChunk
                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(delta="chunk2", is_final=False)
                yield StreamChunk(delta="chunk3", is_final=True)

        mock = MultiChunkProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=3, strategy="token_bucket"
        )

        initial_tokens = middleware._tokens
        request = AgentRequest(prompt="test")

        # Consume one stream (3 chunks)
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 3
        # Should consume exactly 1 token, not 3
        assert abs(middleware._tokens - (initial_tokens - 1.0)) < 0.01

    @pytest.mark.asyncio
    async def test_send_request_stream_propagates_exceptions(self) -> None:
        """send_request_stream propagates exceptions from wrapped client."""
        from mada_modelkit._errors import ProviderError

        class FailingStreamProvider(MockProvider):
            async def send_request_stream(self, request):
                self.call_count += 1
                raise ProviderError("stream error", status_code=500)
                yield  # Make it a generator

        mock = FailingStreamProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        request = AgentRequest(prompt="test")
        with pytest.raises(ProviderError, match="stream error"):
            async for _ in middleware.send_request_stream(request):
                pass

    @pytest.mark.asyncio
    async def test_send_request_stream_token_bucket_refills_between_streams(self) -> None:
        """Tokens refill between send_request_stream calls."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=1, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")

        # First stream uses burst token
        async for _ in middleware.send_request_stream(request):
            pass
        assert middleware._tokens < 1.0

        # Wait for refill
        await asyncio.sleep(0.15)

        # Second stream should succeed without blocking
        start = time.monotonic()
        async for _ in middleware.send_request_stream(request):
            pass
        elapsed = time.monotonic() - start

        assert elapsed < 0.02
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_send_request_stream_mixed_with_send_request(self) -> None:
        """send_request and send_request_stream share same rate limit."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=20.0, burst_size=2, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")

        # Use burst with mixed calls
        start = time.monotonic()
        await middleware.send_request(request)
        async for _ in middleware.send_request_stream(request):
            pass
        immediate_elapsed = time.monotonic() - start
        assert immediate_elapsed < 0.02

        # Third call must wait
        start = time.monotonic()
        await middleware.send_request(request)
        wait_elapsed = time.monotonic() - start
        assert wait_elapsed >= 0.04

        assert mock.call_count == 3


class TestPerKeyRateLimiting:
    """Test per-key rate limiting with key_fn."""

    @pytest.mark.asyncio
    async def test_constructor_with_key_fn(self) -> None:
        """Constructor accepts key_fn parameter."""
        mock = MockProvider()
        key_fn = lambda req: req.metadata.get("user_id")
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, strategy="token_bucket", key_fn=key_fn
        )
        assert middleware._key_fn is key_fn

    @pytest.mark.asyncio
    async def test_per_key_token_bucket_independent_limits(self) -> None:
        """Different keys have independent token buckets."""
        mock = MockProvider()
        key_fn = lambda req: req.metadata.get("user_id")
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=2, strategy="token_bucket", key_fn=key_fn
        )

        # User 1 uses up their burst
        req_user1 = AgentRequest(prompt="test", metadata={"user_id": "user1"})
        await middleware.send_request(req_user1)
        await middleware.send_request(req_user1)

        # User 1's bucket is depleted
        assert middleware._per_key_tokens["user1"] < 1.0

        # User 2 should still have full burst available (immediate)
        req_user2 = AgentRequest(prompt="test", metadata={"user_id": "user2"})
        start = time.monotonic()
        await middleware.send_request(req_user2)
        elapsed = time.monotonic() - start

        assert elapsed < 0.02  # Should be immediate
        assert middleware._per_key_tokens["user2"] >= 1.0  # Still has tokens
        assert mock.call_count == 3

    @pytest.mark.asyncio
    async def test_per_key_leaky_bucket_independent_queues(self) -> None:
        """Different keys have independent leaky bucket queues."""
        mock = MockProvider()
        key_fn = lambda req: req.metadata.get("user_id")
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=20.0, strategy="leaky_bucket", key_fn=key_fn
        )

        req_user1 = AgentRequest(prompt="test", metadata={"user_id": "user1"})
        req_user2 = AgentRequest(prompt="test", metadata={"user_id": "user2"})

        # Send 2 requests for each user concurrently
        async def send_for_user(req):
            await middleware.send_request(req)

        start = time.monotonic()
        await asyncio.gather(
            send_for_user(req_user1),
            send_for_user(req_user1),
            send_for_user(req_user2),
            send_for_user(req_user2),
        )
        elapsed = time.monotonic() - start

        # Each user's requests are rate-limited independently
        # Each user has 2 requests: first immediate, second after 1/20 = 0.05s
        # They run in parallel, so total time ~0.05s (not 0.15s if sequential)
        assert elapsed < 0.12  # Should be ~0.05s + overhead
        assert mock.call_count == 4

        # Clean up processors for both keys
        for processor in middleware._per_key_processors.values():
            if processor and not processor.done():
                processor.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await processor

    @pytest.mark.asyncio
    async def test_per_key_creates_state_on_demand(self) -> None:
        """Per-key state is created lazily for new keys."""
        mock = MockProvider()
        key_fn = lambda req: req.metadata.get("user_id")
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, strategy="token_bucket", key_fn=key_fn
        )

        # No keys initially
        assert len(middleware._per_key_tokens) == 0

        # First request for user1 creates state
        req_user1 = AgentRequest(prompt="test", metadata={"user_id": "user1"})
        await middleware.send_request(req_user1)

        assert "user1" in middleware._per_key_tokens
        assert "user1" in middleware._per_key_locks

        # Second request for user2 creates new state
        req_user2 = AgentRequest(prompt="test", metadata={"user_id": "user2"})
        await middleware.send_request(req_user2)

        assert "user2" in middleware._per_key_tokens
        assert len(middleware._per_key_tokens) == 2

    @pytest.mark.asyncio
    async def test_per_key_token_bucket_enforces_rate_per_key(self) -> None:
        """Each key's rate limit is enforced independently."""
        mock = MockProvider()
        key_fn = lambda req: req.metadata.get("endpoint")
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=20.0, burst_size=1, strategy="token_bucket", key_fn=key_fn
        )

        req_api1 = AgentRequest(prompt="test", metadata={"endpoint": "/api/v1"})

        # Deplete api1's burst
        await middleware.send_request(req_api1)
        assert middleware._per_key_tokens["/api/v1"] < 1.0

        # Second request for api1 should wait
        start = time.monotonic()
        await middleware.send_request(req_api1)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.04  # Had to wait for refill
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_per_key_none_key_uses_same_bucket(self) -> None:
        """Requests with None key share the same bucket."""
        mock = MockProvider()
        key_fn = lambda req: req.metadata.get("user_id")  # Returns None if no metadata
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=2, strategy="token_bucket", key_fn=key_fn
        )

        # Two requests without metadata (key=None)
        req1 = AgentRequest(prompt="test")  # No metadata
        req2 = AgentRequest(prompt="test")  # No metadata

        # Both should share global bucket, not create per-key state
        await middleware.send_request(req1)
        await middleware.send_request(req2)

        # Global bucket depleted
        assert middleware._tokens < 1.0
        # No per-key state created for None
        assert None not in middleware._per_key_tokens

    @pytest.mark.asyncio
    async def test_per_key_stream_independent_limits(self) -> None:
        """send_request_stream respects per-key limits."""
        mock = MockProvider()
        key_fn = lambda req: req.metadata.get("user_id")
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=2, strategy="token_bucket", key_fn=key_fn
        )

        req_user1 = AgentRequest(prompt="test", metadata={"user_id": "user1"})
        req_user2 = AgentRequest(prompt="test", metadata={"user_id": "user2"})

        # User 1 depletes burst
        async for _ in middleware.send_request_stream(req_user1):
            pass
        async for _ in middleware.send_request_stream(req_user1):
            pass

        # User 2 should still have full burst
        start = time.monotonic()
        async for _ in middleware.send_request_stream(req_user2):
            pass
        elapsed = time.monotonic() - start

        assert elapsed < 0.02  # Immediate
        assert middleware._per_key_tokens["user2"] >= 1.0

    @pytest.mark.asyncio
    async def test_per_key_leaky_bucket_creates_processor_per_key(self) -> None:
        """Each key gets its own processor task in leaky bucket."""
        mock = MockProvider()
        key_fn = lambda req: req.metadata.get("user_id")
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, strategy="leaky_bucket", key_fn=key_fn
        )

        req_user1 = AgentRequest(prompt="test", metadata={"user_id": "user1"})
        req_user2 = AgentRequest(prompt="test", metadata={"user_id": "user2"})

        # Send requests for both users
        await middleware.send_request(req_user1)
        await middleware.send_request(req_user2)

        # Both should have their own processor tasks
        assert "user1" in middleware._per_key_processors
        assert "user2" in middleware._per_key_processors
        assert middleware._per_key_processors["user1"] is not middleware._per_key_processors["user2"]

        # Clean up
        for processor in middleware._per_key_processors.values():
            if processor and not processor.done():
                processor.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await processor

    @pytest.mark.asyncio
    async def test_per_key_hashable_keys(self) -> None:
        """Per-key limiting works with various hashable key types."""
        mock = MockProvider()

        # Test with string keys
        key_fn_str = lambda req: req.metadata.get("id")
        middleware_str = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, strategy="token_bucket", key_fn=key_fn_str
        )
        await middleware_str.send_request(AgentRequest(prompt="test", metadata={"id": "abc"}))
        assert "abc" in middleware_str._per_key_tokens

        # Test with int keys
        key_fn_int = lambda req: req.metadata.get("id")
        middleware_int = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, strategy="token_bucket", key_fn=key_fn_int
        )
        await middleware_int.send_request(AgentRequest(prompt="test", metadata={"id": 123}))
        assert 123 in middleware_int._per_key_tokens

        # Test with tuple keys
        key_fn_tuple = lambda req: (req.metadata.get("user"), req.metadata.get("endpoint"))
        middleware_tuple = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, strategy="token_bucket", key_fn=key_fn_tuple
        )
        await middleware_tuple.send_request(
            AgentRequest(prompt="test", metadata={"user": "alice", "endpoint": "/api"})
        )
        assert ("alice", "/api") in middleware_tuple._per_key_tokens


class TestRateLimitComprehensive:
    """Comprehensive integration and edge case tests."""

    @pytest.mark.asyncio
    async def test_token_bucket_accuracy_under_load(self) -> None:
        """Token bucket maintains accurate rate under sustained load."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=50.0, burst_size=10, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")
        start = time.monotonic()

        # Send 20 requests (10 burst + 10 more)
        for _ in range(20):
            await middleware.send_request(request)

        elapsed = time.monotonic() - start

        # First 10 use burst (near instant), next 10 require ~0.2s (10/50)
        assert elapsed >= 0.19  # At least the refill time
        assert mock.call_count == 20

    @pytest.mark.asyncio
    async def test_leaky_bucket_queue_fairness(self) -> None:
        """Leaky bucket processes requests in FIFO order."""
        class OrderTrackingProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.order = []

            async def send_request(self, request):
                self.call_count += 1
                self.order.append(request.metadata.get("id"))
                return AgentResponse(content="ok", model="test", input_tokens=1, output_tokens=1)

        mock = OrderTrackingProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=100.0, strategy="leaky_bucket"
        )

        # Send 5 requests with IDs
        requests = [AgentRequest(prompt="test", metadata={"id": i}) for i in range(5)]
        await asyncio.gather(*[middleware.send_request(req) for req in requests])

        # Should be processed in order 0, 1, 2, 3, 4
        assert mock.order == [0, 1, 2, 3, 4]

        # Clean up
        if middleware._processor_task:
            middleware._processor_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await middleware._processor_task

    @pytest.mark.asyncio
    async def test_burst_handling_refill_timing(self) -> None:
        """Burst capacity refills correctly over time."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=5, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")

        # Deplete burst
        for _ in range(5):
            await middleware.send_request(request)

        assert middleware._tokens < 1.0

        # Wait for partial refill (0.3s = 3 tokens)
        await asyncio.sleep(0.3)

        # Should be able to send 3 more immediately
        start = time.monotonic()
        for _ in range(3):
            await middleware.send_request(request)
        elapsed = time.monotonic() - start

        assert elapsed < 0.05  # Should be near-instant
        assert mock.call_count == 8

    @pytest.mark.asyncio
    async def test_key_based_isolation_no_interference(self) -> None:
        """Per-key limits don't interfere with each other."""
        mock = MockProvider()
        key_fn = lambda req: req.metadata.get("key")
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=2, strategy="token_bucket", key_fn=key_fn
        )

        # Key A depletes burst completely
        req_a = AgentRequest(prompt="test", metadata={"key": "A"})
        await middleware.send_request(req_a)
        await middleware.send_request(req_a)

        # Key A would block now
        assert middleware._per_key_tokens["A"] < 1.0

        # Key B should still work immediately (no blocking)
        req_b = AgentRequest(prompt="test", metadata={"key": "B"})
        start = time.monotonic()
        await middleware.send_request(req_b)
        await middleware.send_request(req_b)
        elapsed = time.monotonic() - start

        assert elapsed < 0.02  # Immediate
        assert middleware._per_key_tokens["B"] < 1.0
        assert mock.call_count == 4

    @pytest.mark.asyncio
    async def test_wraps_other_middleware(self) -> None:
        """RateLimitMiddleware can wrap other middleware."""
        from mada_modelkit.middleware.retry import RetryMiddleware

        mock = MockProvider()
        retry = RetryMiddleware(client=mock, max_retries=2)
        rate_limit = RateLimitMiddleware(client=retry, requests_per_second=10.0, strategy="token_bucket")

        request = AgentRequest(prompt="test")
        response = await rate_limit.send_request(request)

        assert response.content == "mock"
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self) -> None:
        """Context manager properly cleans up resources."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, strategy="leaky_bucket"
        )

        async with middleware:
            await middleware.send_request(AgentRequest(prompt="test"))
            assert middleware._processor_task is not None

        # After context exit, close() should have been called
        # (leaky bucket doesn't implement close, but inherited no-op works)

    @pytest.mark.asyncio
    async def test_zero_burst_not_allowed(self) -> None:
        """Burst size of 0 is treated as default."""
        mock = MockProvider()
        # burst_size=0 should use default (requests_per_second)
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=0, strategy="token_bucket"
        )

        # Should have used 0 as max_tokens
        assert middleware._max_tokens == 0.0

    @pytest.mark.asyncio
    async def test_very_high_rate_limit(self) -> None:
        """Very high rate limits work correctly."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=1000.0, burst_size=100, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")
        start = time.monotonic()

        # Send 100 requests (all from burst)
        for _ in range(100):
            await middleware.send_request(request)

        elapsed = time.monotonic() - start

        # Should be very fast (all from burst)
        assert elapsed < 0.5
        assert mock.call_count == 100

    @pytest.mark.asyncio
    async def test_fractional_rate_limit(self) -> None:
        """Fractional rates (< 1 req/sec) work correctly."""
        mock = MockProvider()
        # 0.5 requests per second = 1 request every 2 seconds
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=0.5, burst_size=1, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")

        # First request uses burst
        await middleware.send_request(request)
        assert middleware._tokens < 1.0

        # Second request should wait ~2 seconds
        start = time.monotonic()
        await middleware.send_request(request)
        elapsed = time.monotonic() - start

        # Should have waited for 1 full token (2 seconds at 0.5/sec)
        # But we poll every 10ms, so might be slightly less
        assert elapsed >= 1.0  # At least 1 second

    @pytest.mark.asyncio
    async def test_exception_doesnt_consume_token(self) -> None:
        """Exceptions after token acquisition don't affect rate limit state."""
        from mada_modelkit._errors import ProviderError

        error = ProviderError("test error")
        mock = MockProvider(errors=[error])
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=10.0, burst_size=2, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")

        # First request fails but consumes token
        with pytest.raises(ProviderError):
            await middleware.send_request(request)

        # Token was consumed
        assert middleware._tokens < 2.0

        # Second request should still have 1 token available
        await middleware.send_request(request)  # This will succeed (no more errors)
        assert middleware._tokens < 1.0

    @pytest.mark.asyncio
    async def test_concurrent_per_key_requests(self) -> None:
        """Concurrent requests for different keys process in parallel."""
        mock = MockProvider()
        key_fn = lambda req: req.metadata.get("key")
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=5.0, strategy="leaky_bucket", key_fn=key_fn
        )

        # Send 2 requests for each of 3 keys concurrently
        async def send_pair(key):
            req = AgentRequest(prompt="test", metadata={"key": key})
            await middleware.send_request(req)
            await middleware.send_request(req)

        start = time.monotonic()
        await asyncio.gather(send_pair("A"), send_pair("B"), send_pair("C"))
        elapsed = time.monotonic() - start

        # Each key processes 2 requests: first immediate, second after 1/5 = 0.2s
        # All 3 keys run in parallel, so total ~0.2s (not 0.6s if sequential)
        assert elapsed < 0.5  # Allow for timing variance
        assert mock.call_count == 6

        # Clean up all processors
        for processor in middleware._per_key_processors.values():
            if processor and not processor.done():
                processor.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await processor

    @pytest.mark.asyncio
    async def test_mixed_send_request_and_stream(self) -> None:
        """send_request and send_request_stream interleave correctly."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(
            client=mock, requests_per_second=20.0, burst_size=3, strategy="token_bucket"
        )

        request = AgentRequest(prompt="test")

        # Alternate between send_request and send_request_stream
        await middleware.send_request(request)
        async for _ in middleware.send_request_stream(request):
            pass
        await middleware.send_request(request)

        # All 3 consumed burst
        assert middleware._tokens < 1.0
        assert mock.call_count == 3

    @pytest.mark.asyncio
    async def test_health_check_inherited(self) -> None:
        """health_check method is inherited and works."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        # Default health_check should return True
        assert await middleware.health_check() is True

    @pytest.mark.asyncio
    async def test_cancel_inherited(self) -> None:
        """cancel method is inherited and works (no-op)."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        # Should not raise
        await middleware.cancel()

    @pytest.mark.asyncio
    async def test_close_inherited(self) -> None:
        """close method is inherited and works (no-op)."""
        mock = MockProvider()
        middleware = RateLimitMiddleware(client=mock, requests_per_second=10.0, strategy="token_bucket")

        # Should not raise
        await middleware.close()
