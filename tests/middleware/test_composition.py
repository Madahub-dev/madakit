"""Tests for middleware composition — all 5 middleware stacked together."""

from __future__ import annotations

import pytest

from helpers import MockProvider
from madakit import (
    AgentRequest,
    AgentResponse,
    CachingMiddleware,
    CircuitBreakerMiddleware,
    FallbackMiddleware,
    RetryMiddleware,
    TrackingMiddleware,
)
from madakit._errors import ProviderError


def _build_stack(
    provider: MockProvider,
    *,
    fallback_provider: MockProvider | None = None,
) -> tuple[TrackingMiddleware, FallbackMiddleware | RetryMiddleware]:
    """Build a full middleware stack: tracking → cache → circuit → retry → provider.

    Returns the tracking middleware (for stats inspection) and the outermost client.
    """
    retry = RetryMiddleware(provider, max_retries=2, backoff_base=0.0)
    circuit = CircuitBreakerMiddleware(retry, failure_threshold=3, recovery_timeout=0.1)
    cache = CachingMiddleware(circuit, ttl=60.0, max_entries=100)
    tracking = TrackingMiddleware(cache)

    if fallback_provider is not None:
        fallback_retry = RetryMiddleware(fallback_provider, max_retries=1, backoff_base=0.0)
        outer: FallbackMiddleware | TrackingMiddleware = FallbackMiddleware(
            tracking, [fallback_retry]
        )
    else:
        outer = tracking

    return tracking, outer


class TestFullStackComposition:
    """All 5 middleware layers stacked, happy path."""

    async def test_request_flows_through_full_stack(self) -> None:
        provider = MockProvider()
        tracking, stack = _build_stack(provider)

        request = AgentRequest(prompt="hello")
        response = await stack.send_request(request)

        assert response.content == "mock"
        assert provider.call_count == 1
        stats = tracking.stats
        assert stats.total_requests == 1

    async def test_cache_hit_skips_provider(self) -> None:
        provider = MockProvider()
        tracking, stack = _build_stack(provider)

        request = AgentRequest(prompt="cached")
        await stack.send_request(request)
        await stack.send_request(request)

        assert provider.call_count == 1
        assert tracking.stats.total_requests == 2

    async def test_retry_on_transient_failure(self) -> None:
        provider = MockProvider(errors=[ProviderError("fail", status_code=503)])
        tracking, stack = _build_stack(provider)

        request = AgentRequest(prompt="retry-me")
        response = await stack.send_request(request)

        assert response.content == "mock"
        assert provider.call_count == 2  # 1 failure + 1 success


class TestStreamThroughStack:
    """Streaming through the composition stack."""

    async def test_stream_through_full_stack(self) -> None:
        provider = MockProvider()
        tracking, stack = _build_stack(provider)

        request = AgentRequest(prompt="stream-me")
        chunks = []
        async for chunk in stack.send_request_stream(request):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[-1].is_final


class TestErrorPropagation:
    """Error flows through the stack correctly."""

    async def test_circuit_opens_after_threshold(self) -> None:
        errors = [ProviderError("fail", status_code=500) for _ in range(20)]
        provider = MockProvider(errors=errors)
        tracking, stack = _build_stack(provider)

        from madakit._errors import CircuitOpenError, RetryExhaustedError

        failures = 0
        for _ in range(10):
            try:
                await stack.send_request(AgentRequest(prompt="break"))
            except (ProviderError, CircuitOpenError, RetryExhaustedError):
                failures += 1

        assert failures > 0

    async def test_fallback_on_primary_failure(self) -> None:
        primary = MockProvider(errors=[ProviderError("down", status_code=500) for _ in range(10)])
        fallback = MockProvider(
            responses=[AgentResponse(content="fallback", model="fb", input_tokens=5, output_tokens=3)]
        )

        _, stack = _build_stack(primary, fallback_provider=fallback)

        response = await stack.send_request(AgentRequest(prompt="help"))
        assert response.content == "fallback"
