"""End-to-end integration tests for mada-modelkit."""

from __future__ import annotations

import pytest

from helpers import MockProvider
from mada_modelkit import (
    AgentRequest,
    AgentResponse,
    CachingMiddleware,
    CircuitBreakerMiddleware,
    RetryMiddleware,
    TrackingMiddleware,
)
from mada_modelkit._errors import ProviderError


class TestCloudE2EHappyPath:
    """Simulates a cloud provider flow with tracking and caching."""

    async def test_request_response_cycle(self) -> None:
        provider = MockProvider(
            responses=[
                AgentResponse(content="Hello!", model="gpt-4", input_tokens=15, output_tokens=8)
            ]
        )
        tracking = TrackingMiddleware(provider)

        response = await tracking.send_request(
            AgentRequest(prompt="Say hello", max_tokens=100)
        )

        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert tracking.stats.total_requests == 1
        assert tracking.stats.total_input_tokens == 15
        assert tracking.stats.total_output_tokens == 8

    async def test_streaming_with_tracking(self) -> None:
        provider = MockProvider()
        tracking = TrackingMiddleware(provider)

        chunks = []
        async for chunk in tracking.send_request_stream(AgentRequest(prompt="stream")):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[-1].is_final
        # streaming tracks timing but does not increment total_requests
        assert tracking.stats.total_inference_ms > 0


class TestFallbackWithCircuitBreaker:
    """Circuit breaker triggers fallback recovery."""

    async def test_circuit_breaker_recovery(self) -> None:
        from mada_modelkit._errors import RetryExhaustedError

        provider = MockProvider(
            errors=[ProviderError("timeout", status_code=503)],
        )
        circuit = CircuitBreakerMiddleware(
            RetryMiddleware(provider, max_retries=0, backoff_base=0.0),
            failure_threshold=3,
            recovery_timeout=0.01,
        )

        with pytest.raises(RetryExhaustedError):
            await circuit.send_request(AgentRequest(prompt="fail"))

        response = await circuit.send_request(AgentRequest(prompt="recover"))
        assert response.content == "mock"


class TestCacheTrackingInteraction:
    """Cache and tracking work correctly together."""

    async def test_cache_hit_still_tracked(self) -> None:
        provider = MockProvider(
            responses=[
                AgentResponse(content="cached", model="m", input_tokens=10, output_tokens=5)
            ]
        )
        cache = CachingMiddleware(provider, ttl=60.0)
        tracking = TrackingMiddleware(cache)

        req = AgentRequest(prompt="same")
        r1 = await tracking.send_request(req)
        r2 = await tracking.send_request(req)

        assert r1.content == "cached"
        assert r2.content == "cached"
        assert provider.call_count == 1
        assert tracking.stats.total_requests == 2

    async def test_cache_miss_hits_provider(self) -> None:
        provider = MockProvider()
        cache = CachingMiddleware(provider, ttl=60.0)
        tracking = TrackingMiddleware(cache)

        await tracking.send_request(AgentRequest(prompt="a"))
        await tracking.send_request(AgentRequest(prompt="b"))

        assert provider.call_count == 2
        assert tracking.stats.total_requests == 2
