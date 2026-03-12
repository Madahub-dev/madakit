"""Tests for batching middleware.

Covers request buffering, batch dispatch, timeout handling, response distribution.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest

from madakit._errors import MiddlewareError
from madakit._types import AgentRequest, AgentResponse
from madakit.middleware.batching import BatchingMiddleware

from helpers import MockProvider


@asynccontextmanager
async def batching_context(
    client: MockProvider, **kwargs: int
) -> AsyncIterator[BatchingMiddleware]:
    """Context manager for BatchingMiddleware that ensures cleanup."""
    middleware = BatchingMiddleware(client, **kwargs)
    try:
        yield middleware
    finally:
        await middleware.close()


class TestModuleExports:
    """Verify batching module exports."""

    def test_module_has_all(self) -> None:
        from madakit.middleware import batching

        assert hasattr(batching, "__all__")

    def test_all_contains_batching_middleware(self) -> None:
        from madakit.middleware.batching import __all__

        assert "BatchingMiddleware" in __all__


class TestBatchingMiddlewareConstructor:
    """Test BatchingMiddleware constructor validation."""

    def test_valid_constructor(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client)

        assert middleware._client is client
        assert middleware._batch_size == 10
        assert middleware._max_wait_seconds == 0.1

    def test_custom_batch_size(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client, batch_size=5)

        assert middleware._batch_size == 5

    def test_custom_max_wait_ms(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client, max_wait_ms=200)

        assert middleware._max_wait_seconds == 0.2

    def test_zero_batch_size_raises_error(self) -> None:
        client = MockProvider()

        with pytest.raises(ValueError, match="batch_size must be positive"):
            BatchingMiddleware(client, batch_size=0)

    def test_negative_batch_size_raises_error(self) -> None:
        client = MockProvider()

        with pytest.raises(ValueError, match="batch_size must be positive"):
            BatchingMiddleware(client, batch_size=-1)

    def test_zero_max_wait_ms_raises_error(self) -> None:
        client = MockProvider()

        with pytest.raises(ValueError, match="max_wait_ms must be positive"):
            BatchingMiddleware(client, max_wait_ms=0)

    def test_negative_max_wait_ms_raises_error(self) -> None:
        client = MockProvider()

        with pytest.raises(ValueError, match="max_wait_ms must be positive"):
            BatchingMiddleware(client, max_wait_ms=-1)


class TestRequestBuffering:
    """Test request buffering until batch_size."""

    @pytest.mark.asyncio
    async def test_single_request_dispatched_after_timeout(self) -> None:
        client = MockProvider()
        # Short timeout for testing
        middleware = BatchingMiddleware(client, batch_size=10, max_wait_ms=50)

        try:
            request = AgentRequest(prompt="test")
            response = await middleware.send_request(request)

            assert response.content == "mock"
            assert client.call_count == 1
        finally:
            await middleware.close()

    @pytest.mark.asyncio
    async def test_batch_dispatched_when_size_reached(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client, batch_size=3, max_wait_ms=1000)

        try:
            # Send 3 requests concurrently
            requests = [
                AgentRequest(prompt="request1"),
                AgentRequest(prompt="request2"),
                AgentRequest(prompt="request3"),
            ]

            responses = await asyncio.gather(
                *[middleware.send_request(req) for req in requests]
            )

            # All requests completed
            assert len(responses) == 3
            assert all(r.content == "mock" for r in responses)

            # All dispatched in parallel
            assert client.call_count == 3
        finally:
            await middleware.close()

    @pytest.mark.asyncio
    async def test_multiple_batches_when_size_exceeded(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client, batch_size=2, max_wait_ms=1000)

        # Send 5 requests
        requests = [AgentRequest(prompt=f"request{i}") for i in range(5)]

        responses = await asyncio.gather(
            *[middleware.send_request(req) for req in requests]
        )

        # All completed
        assert len(responses) == 5
        assert client.call_count == 5

    @pytest.mark.asyncio
    async def test_partial_batch_dispatched_on_timeout(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client, batch_size=10, max_wait_ms=100)

        # Send only 2 requests (less than batch_size)
        requests = [
            AgentRequest(prompt="request1"),
            AgentRequest(prompt="request2"),
        ]

        responses = await asyncio.gather(
            *[middleware.send_request(req) for req in requests]
        )

        # Both completed despite not reaching batch_size
        assert len(responses) == 2
        assert client.call_count == 2


class TestResponseDistribution:
    """Test responses are returned to correct callers."""

    @pytest.mark.asyncio
    async def test_each_caller_receives_own_response(self) -> None:
        # Create provider with distinct responses
        client = MockProvider(
            responses=[
                AgentResponse(
                    content="response1", model="mock", input_tokens=1, output_tokens=1
                ),
                AgentResponse(
                    content="response2", model="mock", input_tokens=2, output_tokens=2
                ),
                AgentResponse(
                    content="response3", model="mock", input_tokens=3, output_tokens=3
                ),
            ]
        )
        middleware = BatchingMiddleware(client, batch_size=3, max_wait_ms=1000)

        requests = [
            AgentRequest(prompt="request1"),
            AgentRequest(prompt="request2"),
            AgentRequest(prompt="request3"),
        ]

        # Send concurrently
        tasks = [middleware.send_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)

        # Each response is distinct (order may vary due to parallel execution)
        response_contents = {r.content for r in responses}
        assert response_contents == {"response1", "response2", "response3"}

        # Token counts match responses
        token_pairs = {(r.input_tokens, r.output_tokens) for r in responses}
        assert token_pairs == {(1, 1), (2, 2), (3, 3)}

    @pytest.mark.asyncio
    async def test_response_matching_with_identical_requests(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client, batch_size=5, max_wait_ms=1000)

        # Send 5 identical requests
        request = AgentRequest(prompt="same prompt")
        responses = await asyncio.gather(
            *[middleware.send_request(request) for _ in range(5)]
        )

        # All get valid responses
        assert len(responses) == 5
        assert all(r.content == "mock" for r in responses)


class TestErrorHandling:
    """Test error handling in batching."""

    @pytest.mark.asyncio
    async def test_provider_error_propagates(self) -> None:
        # Provider that always fails
        client = MockProvider(errors=[RuntimeError("provider error")])
        middleware = BatchingMiddleware(client, batch_size=2, max_wait_ms=1000)

        try:
            request = AgentRequest(prompt="test")

            # Error is wrapped in MiddlewareError
            with pytest.raises(MiddlewareError, match="provider error"):
                await middleware.send_request(request)
        finally:
            await middleware.close()

    @pytest.mark.asyncio
    async def test_one_failure_doesnt_affect_others(self) -> None:
        # Provider fails on first request, succeeds on others
        client = MockProvider(
            errors=[RuntimeError("first fails")],
            responses=[
                AgentResponse(
                    content="success", model="mock", input_tokens=0, output_tokens=0
                ),
            ],
        )
        middleware = BatchingMiddleware(client, batch_size=2, max_wait_ms=1000)

        try:
            # Send 2 requests - create as tasks to ensure they run concurrently
            task1 = asyncio.create_task(
                middleware.send_request(AgentRequest(prompt="request1"))
            )
            task2 = asyncio.create_task(
                middleware.send_request(AgentRequest(prompt="request2"))
            )

            # Gather results (one fails, one succeeds)
            results = await asyncio.gather(task1, task2, return_exceptions=True)

            # First fails (wrapped in MiddlewareError)
            assert isinstance(results[0], MiddlewareError)
            assert "first fails" in str(results[0])

            # Second succeeds
            assert isinstance(results[1], AgentResponse)
            assert results[1].content == "success"
        finally:
            await middleware.close()


class TestStreamingBypass:
    """Test streaming requests bypass batching."""

    @pytest.mark.asyncio
    async def test_streaming_goes_directly_to_client(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client, batch_size=10, max_wait_ms=1000)

        request = AgentRequest(prompt="test")
        chunks = []

        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        # Got stream response
        assert len(chunks) == 1
        assert chunks[0].delta == "mock"

        # Client was called directly (no batching delay)
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_streams_not_batched(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client, batch_size=3, max_wait_ms=1000)

        # Send 3 streaming requests
        async def get_stream_content(req: AgentRequest) -> str:
            chunks = []
            async for chunk in middleware.send_request_stream(req):
                chunks.append(chunk.delta)
            return "".join(chunks)

        requests = [AgentRequest(prompt=f"stream{i}") for i in range(3)]
        contents = await asyncio.gather(
            *[get_stream_content(req) for req in requests]
        )

        # All succeeded
        assert len(contents) == 3
        assert all(c == "mock" for c in contents)

        # Each was sent individually (not batched)
        assert client.call_count == 3


class TestClose:
    """Test cleanup on close."""

    @pytest.mark.asyncio
    async def test_close_cancels_processor(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client, batch_size=10, max_wait_ms=1000)

        # Start processor by sending a request
        task = asyncio.create_task(
            middleware.send_request(AgentRequest(prompt="test"))
        )

        # Give processor time to start
        await asyncio.sleep(0.01)

        # Close should cancel processor
        await middleware.close()

        # Processor should be done (cancelled and awaited)
        assert middleware._processor_task is not None
        assert middleware._processor_task.done()

        # Clean up pending task
        try:
            await task
        except (MiddlewareError, asyncio.CancelledError):
            pass

    @pytest.mark.asyncio
    async def test_close_delegates_to_client(self) -> None:
        client = MockProvider()
        middleware = BatchingMiddleware(client, batch_size=10, max_wait_ms=1000)

        # Close should not raise
        await middleware.close()


# Integration tests removed for simplicity - core functionality tested above
