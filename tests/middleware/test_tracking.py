"""Tests for TrackingMiddleware constructor (task 2.4.1).

Covers: client storage, default cost_fn (None), custom cost_fn storage,
_stats initialised as a fresh TrackingStats instance (all counters at zero,
total_cost_usd=None), BaseAgentClient inheritance, semaphore absence,
per-instance stats isolation, stub delegation to the wrapped client, and
middleware composition.
"""

from __future__ import annotations

import pytest

from helpers import MockProvider
from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk, TrackingStats
from mada_modelkit.middleware.tracking import TrackingMiddleware


class TestTrackingMiddlewareConstructor:
    """TrackingMiddleware.__init__ — attribute storage and initial stats state."""

    def test_stores_client(self) -> None:
        """Asserts that the wrapped client is stored as _client."""
        provider = MockProvider()
        middleware = TrackingMiddleware(client=provider)
        assert middleware._client is provider

    def test_default_cost_fn_is_none(self) -> None:
        """Asserts that cost_fn defaults to None."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert middleware._cost_fn is None

    def test_custom_cost_fn_is_stored(self) -> None:
        """Asserts that a custom cost_fn callable is stored correctly."""
        fn = lambda resp: 0.01  # noqa: E731
        middleware = TrackingMiddleware(client=MockProvider(), cost_fn=fn)
        assert middleware._cost_fn is fn

    def test_stats_is_tracking_stats_instance(self) -> None:
        """Asserts that _stats is a TrackingStats instance."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert isinstance(middleware._stats, TrackingStats)

    def test_stats_initial_total_requests_is_zero(self) -> None:
        """Asserts that _stats.total_requests starts at 0."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert middleware._stats.total_requests == 0

    def test_stats_initial_input_tokens_is_zero(self) -> None:
        """Asserts that _stats.total_input_tokens starts at 0."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert middleware._stats.total_input_tokens == 0

    def test_stats_initial_output_tokens_is_zero(self) -> None:
        """Asserts that _stats.total_output_tokens starts at 0."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert middleware._stats.total_output_tokens == 0

    def test_stats_initial_inference_ms_is_zero(self) -> None:
        """Asserts that _stats.total_inference_ms starts at 0.0."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert middleware._stats.total_inference_ms == 0.0

    def test_stats_initial_ttft_ms_is_zero(self) -> None:
        """Asserts that _stats.total_ttft_ms starts at 0.0."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert middleware._stats.total_ttft_ms == 0.0

    def test_stats_initial_cost_usd_is_none(self) -> None:
        """Asserts that _stats.total_cost_usd starts at None."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert middleware._stats.total_cost_usd is None

    def test_is_base_agent_client(self) -> None:
        """Asserts that TrackingMiddleware is a BaseAgentClient subclass."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert isinstance(middleware, BaseAgentClient)

    def test_no_semaphore_by_default(self) -> None:
        """Asserts that _semaphore is None when max_concurrent is not passed."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert middleware._semaphore is None

    def test_each_instance_has_independent_stats(self) -> None:
        """Asserts that two instances do not share the same TrackingStats object."""
        m1 = TrackingMiddleware(client=MockProvider())
        m2 = TrackingMiddleware(client=MockProvider())
        assert m1._stats is not m2._stats

    def test_mutating_one_stats_does_not_affect_another(self) -> None:
        """Asserts that incrementing one instance's stats leaves the other unchanged."""
        m1 = TrackingMiddleware(client=MockProvider())
        m2 = TrackingMiddleware(client=MockProvider())
        m1._stats.total_requests = 5
        assert m2._stats.total_requests == 0

    def test_client_can_be_another_middleware(self) -> None:
        """Asserts that a TrackingMiddleware can wrap another middleware (composition)."""
        from mada_modelkit.middleware.cache import CachingMiddleware

        inner = CachingMiddleware(client=MockProvider())
        outer = TrackingMiddleware(client=inner)
        assert outer._client is inner


class TestStubDelegation:
    """TrackingMiddleware stub — send_request and send_request_stream delegate to client."""

    @pytest.mark.asyncio
    async def test_send_request_delegates_to_client(self) -> None:
        """Asserts that send_request returns the wrapped client's response."""
        expected = AgentResponse(content="tracked", model="m", input_tokens=1, output_tokens=1)
        middleware = TrackingMiddleware(client=MockProvider(responses=[expected]))
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected

    @pytest.mark.asyncio
    async def test_send_request_stream_delegates_to_client(self) -> None:
        """Asserts that send_request_stream yields chunks from the wrapped client."""
        middleware = TrackingMiddleware(client=MockProvider())
        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="hi")):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].delta == "mock"

    @pytest.mark.asyncio
    async def test_send_request_increments_client_call_count(self) -> None:
        """Asserts that send_request invokes the wrapped client exactly once."""
        provider = MockProvider()
        middleware = TrackingMiddleware(client=provider)
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert provider.call_count == 1
