"""Tests for load balancing middleware.

Tests LoadBalancingMiddleware constructor, weighted round-robin, health-based
routing, latency-based selection, and error handling.
"""

from __future__ import annotations

import asyncio

import pytest

from mada_modelkit._errors import MiddlewareError
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk
from mada_modelkit.middleware.load_balancing import LoadBalancingMiddleware

from helpers import MockProvider


class TestModuleExports:
    """Test module-level exports and imports."""

    def test_all_exports(self) -> None:
        """__all__ contains only LoadBalancingMiddleware."""
        from mada_modelkit.middleware import load_balancing

        assert load_balancing.__all__ == ["LoadBalancingMiddleware"]

    def test_middleware_importable(self) -> None:
        """LoadBalancingMiddleware can be imported from module."""
        from mada_modelkit.middleware.load_balancing import (
            LoadBalancingMiddleware as LBM,
        )

        assert LBM is not None


class TestLoadBalancingMiddlewareConstructor:
    """Test LoadBalancingMiddleware constructor and initialization."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts providers list with default strategy."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        providers = [(mock1, 1.0), (mock2, 1.0)]

        middleware = LoadBalancingMiddleware(providers=providers)

        assert middleware._providers == providers
        assert middleware._strategy == "weighted"

    def test_with_weighted_strategy(self) -> None:
        """Constructor accepts weighted strategy."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="weighted",
        )

        assert middleware._strategy == "weighted"

    def test_with_health_strategy(self) -> None:
        """Constructor accepts health strategy."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="health",
        )

        assert middleware._strategy == "health"

    def test_with_latency_strategy(self) -> None:
        """Constructor accepts latency strategy."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="latency",
        )

        assert middleware._strategy == "latency"

    def test_empty_providers_raises(self) -> None:
        """Empty providers list raises ValueError."""
        with pytest.raises(ValueError, match="At least one provider is required"):
            LoadBalancingMiddleware(providers=[])

    def test_zero_weight_raises(self) -> None:
        """Zero weight raises ValueError."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        with pytest.raises(ValueError, match="All provider weights must be positive"):
            LoadBalancingMiddleware(providers=[(mock1, 1.0), (mock2, 0.0)])

    def test_negative_weight_raises(self) -> None:
        """Negative weight raises ValueError."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        with pytest.raises(ValueError, match="All provider weights must be positive"):
            LoadBalancingMiddleware(providers=[(mock1, 1.0), (mock2, -0.5)])

    def test_unknown_strategy_raises(self) -> None:
        """Unknown strategy raises ValueError."""
        mock = MockProvider()

        with pytest.raises(ValueError, match="Unknown strategy"):
            LoadBalancingMiddleware(
                providers=[(mock, 1.0)],
                strategy="unknown",
            )

    def test_super_init_called(self) -> None:
        """LoadBalancingMiddleware calls BaseAgentClient.__init__."""
        mock = MockProvider()
        middleware = LoadBalancingMiddleware(providers=[(mock, 1.0)])

        assert hasattr(middleware, "send_request")
        assert hasattr(middleware, "send_request_stream")


class TestWeightedRoundRobin:
    """Test weighted round-robin selection."""

    @pytest.mark.asyncio
    async def test_equal_weights_round_robin(self) -> None:
        """Equal weights distribute requests evenly."""
        call_counts = [0, 0]

        class CountingProvider(MockProvider):
            def __init__(self, index):
                super().__init__()
                self.index = index

            async def send_request(self, request):
                call_counts[self.index] += 1
                return await super().send_request(request)

        mock1 = CountingProvider(0)
        mock2 = CountingProvider(1)

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="weighted",
        )

        # Send multiple requests
        for _ in range(10):
            await middleware.send_request(AgentRequest(prompt="test"))

        # Should distribute roughly evenly
        assert 3 <= call_counts[0] <= 7
        assert 3 <= call_counts[1] <= 7

    @pytest.mark.asyncio
    async def test_weighted_distribution(self) -> None:
        """Unequal weights distribute proportionally."""
        call_counts = [0, 0]

        class CountingProvider(MockProvider):
            def __init__(self, index):
                super().__init__()
                self.index = index

            async def send_request(self, request):
                call_counts[self.index] += 1
                return await super().send_request(request)

        mock1 = CountingProvider(0)
        mock2 = CountingProvider(1)

        # 3:1 ratio
        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 3.0), (mock2, 1.0)],
            strategy="weighted",
        )

        # Send many requests
        for _ in range(100):
            await middleware.send_request(AgentRequest(prompt="test"))

        # Should approximate 3:1 ratio (75% vs 25%)
        assert 60 <= call_counts[0] <= 90  # ~75%
        assert 10 <= call_counts[1] <= 40  # ~25%

    @pytest.mark.asyncio
    async def test_single_provider_always_selected(self) -> None:
        """Single provider receives all requests."""
        call_count = 0

        class CountingProvider(MockProvider):
            async def send_request(self, request):
                nonlocal call_count
                call_count += 1
                return await super().send_request(request)

        mock = CountingProvider()
        middleware = LoadBalancingMiddleware(
            providers=[(mock, 1.0)],
            strategy="weighted",
        )

        for _ in range(5):
            await middleware.send_request(AgentRequest(prompt="test"))

        assert call_count == 5


class TestHealthBasedRouting:
    """Test health-based provider selection."""

    @pytest.mark.asyncio
    async def test_routes_to_healthy_provider(self) -> None:
        """Routes to healthy provider when one is unhealthy."""
        call_counts = [0, 0]

        class HealthAwareProvider(MockProvider):
            def __init__(self, index, is_healthy):
                super().__init__()
                self.index = index
                self.is_healthy = is_healthy

            async def health_check(self):
                return self.is_healthy

            async def send_request(self, request):
                call_counts[self.index] += 1
                return await super().send_request(request)

        mock1 = HealthAwareProvider(0, False)  # Unhealthy
        mock2 = HealthAwareProvider(1, True)   # Healthy

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="health",
        )

        for _ in range(5):
            await middleware.send_request(AgentRequest(prompt="test"))

        # All requests should go to healthy provider
        assert call_counts[0] == 0
        assert call_counts[1] == 5

    @pytest.mark.asyncio
    async def test_all_unhealthy_raises(self) -> None:
        """Raises MiddlewareError when all providers unhealthy."""

        class UnhealthyProvider(MockProvider):
            async def health_check(self):
                return False

        mock1 = UnhealthyProvider()
        mock2 = UnhealthyProvider()

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="health",
        )

        with pytest.raises(MiddlewareError, match="No healthy providers available"):
            await middleware.send_request(AgentRequest(prompt="test"))

    @pytest.mark.asyncio
    async def test_health_check_exception_treated_unhealthy(self) -> None:
        """Provider with failing health_check treated as unhealthy."""
        call_counts = [0, 0]

        class FailingHealthProvider(MockProvider):
            def __init__(self, index, fail_health):
                super().__init__()
                self.index = index
                self.fail_health = fail_health

            async def health_check(self):
                if self.fail_health:
                    raise Exception("Health check failed")
                return True

            async def send_request(self, request):
                call_counts[self.index] += 1
                return await super().send_request(request)

        mock1 = FailingHealthProvider(0, True)   # Health check throws
        mock2 = FailingHealthProvider(1, False)  # Healthy

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="health",
        )

        await middleware.send_request(AgentRequest(prompt="test"))

        # Should route to provider 2
        assert call_counts[0] == 0
        assert call_counts[1] == 1

    @pytest.mark.asyncio
    async def test_all_healthy_selects_first(self) -> None:
        """Selects first healthy provider when all are healthy."""
        call_counts = [0, 0]

        class HealthyProvider(MockProvider):
            def __init__(self, index):
                super().__init__()
                self.index = index

            async def health_check(self):
                return True

            async def send_request(self, request):
                call_counts[self.index] += 1
                return await super().send_request(request)

        mock1 = HealthyProvider(0)
        mock2 = HealthyProvider(1)

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="health",
        )

        for _ in range(5):
            await middleware.send_request(AgentRequest(prompt="test"))

        # All should go to first provider
        assert call_counts[0] == 5
        assert call_counts[1] == 0


class TestLatencyBasedRouting:
    """Test latency-based provider selection."""

    @pytest.mark.asyncio
    async def test_routes_to_fastest_provider(self) -> None:
        """Routes to provider with lowest latency."""
        call_counts = [0, 0]

        class LatencyProvider(MockProvider):
            def __init__(self, index, latency):
                super().__init__()
                self.index = index
                self.latency = latency

            async def send_request(self, request):
                call_counts[self.index] += 1
                await asyncio.sleep(self.latency)
                return await super().send_request(request)

        mock1 = LatencyProvider(0, 0.1)   # Slow
        mock2 = LatencyProvider(1, 0.01)  # Fast

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="latency",
        )

        # First request goes to provider 0 (no latency data yet)
        await middleware.send_request(AgentRequest(prompt="test"))

        # Second request should go to same provider (still exploring)
        await middleware.send_request(AgentRequest(prompt="test"))

        # After gathering latency data, should prefer faster provider
        # Let's send a few more to build up data
        for _ in range(5):
            await middleware.send_request(AgentRequest(prompt="test"))

        # Provider 2 should have more requests (it's faster)
        # This is approximate since first requests distribute evenly
        assert call_counts[1] > call_counts[0]

    @pytest.mark.asyncio
    async def test_latency_window_limits_history(self) -> None:
        """Latency window limits stored history."""
        mock = MockProvider()
        middleware = LoadBalancingMiddleware(
            providers=[(mock, 1.0)],
            strategy="latency",
        )

        # Record many latencies
        for i in range(20):
            middleware._record_latency(0, float(i))

        # Should only keep last 10
        assert len(middleware._latencies[0]) == 10
        assert middleware._latencies[0][-1] == 19.0

    @pytest.mark.asyncio
    async def test_no_latency_data_uses_first_provider(self) -> None:
        """With no latency data, uses first provider."""
        call_counts = [0, 0]

        class CountingProvider(MockProvider):
            def __init__(self, index):
                super().__init__()
                self.index = index

            async def send_request(self, request):
                call_counts[self.index] += 1
                return await super().send_request(request)

        mock1 = CountingProvider(0)
        mock2 = CountingProvider(1)

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="latency",
        )

        # First request with no latency data
        await middleware.send_request(AgentRequest(prompt="test"))

        # Should go to first provider (both have inf latency)
        assert call_counts[0] == 1
        assert call_counts[1] == 0


class TestRequestRouting:
    """Test request routing through middleware."""

    @pytest.mark.asyncio
    async def test_send_request_returns_response(self) -> None:
        """send_request returns response from selected provider."""
        mock = MockProvider()
        middleware = LoadBalancingMiddleware(
            providers=[(mock, 1.0)],
            strategy="weighted",
        )

        response = await middleware.send_request(AgentRequest(prompt="test"))

        assert response.content == "mock"
        assert response.model == "mock"

    @pytest.mark.asyncio
    async def test_send_request_preserves_response_data(self) -> None:
        """send_request preserves all response data."""
        mock = MockProvider(
            responses=[
                AgentResponse(
                    content="custom",
                    model="custom-model",
                    input_tokens=100,
                    output_tokens=50,
                    metadata={"key": "value"},
                )
            ]
        )
        middleware = LoadBalancingMiddleware(
            providers=[(mock, 1.0)],
            strategy="weighted",
        )

        response = await middleware.send_request(AgentRequest(prompt="test"))

        assert response.content == "custom"
        assert response.model == "custom-model"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.metadata == {"key": "value"}

    @pytest.mark.asyncio
    async def test_send_request_records_latency(self) -> None:
        """send_request records latency for provider."""
        mock = MockProvider()
        middleware = LoadBalancingMiddleware(
            providers=[(mock, 1.0)],
            strategy="latency",
        )

        await middleware.send_request(AgentRequest(prompt="test"))

        # Should have recorded latency
        assert len(middleware._latencies[0]) == 1
        assert middleware._latencies[0][0] > 0


class TestStreamRouting:
    """Test streaming request routing."""

    @pytest.mark.asyncio
    async def test_send_request_stream_yields_chunks(self) -> None:
        """send_request_stream yields chunks from provider."""

        class StreamProvider(MockProvider):
            async def send_request_stream(self, request):
                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(delta="chunk2", is_final=True)

        mock = StreamProvider()
        middleware = LoadBalancingMiddleware(
            providers=[(mock, 1.0)],
            strategy="weighted",
        )

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].delta == "chunk1"
        assert chunks[1].delta == "chunk2"

    @pytest.mark.asyncio
    async def test_send_request_stream_records_latency(self) -> None:
        """send_request_stream records latency to first chunk."""

        class StreamProvider(MockProvider):
            async def send_request_stream(self, request):
                yield StreamChunk(delta="test", is_final=True)

        mock = StreamProvider()
        middleware = LoadBalancingMiddleware(
            providers=[(mock, 1.0)],
            strategy="latency",
        )

        async for _ in middleware.send_request_stream(AgentRequest(prompt="test")):
            pass

        # Should have recorded latency
        assert len(middleware._latencies[0]) == 1

    @pytest.mark.asyncio
    async def test_send_request_stream_with_health_strategy(self) -> None:
        """send_request_stream works with health strategy."""
        call_counts = [0, 0]

        class HealthStreamProvider(MockProvider):
            def __init__(self, index, is_healthy):
                super().__init__()
                self.index = index
                self.is_healthy = is_healthy

            async def health_check(self):
                return self.is_healthy

            async def send_request_stream(self, request):
                call_counts[self.index] += 1
                yield StreamChunk(delta="test", is_final=True)

        mock1 = HealthStreamProvider(0, False)
        mock2 = HealthStreamProvider(1, True)

        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 1.0), (mock2, 1.0)],
            strategy="health",
        )

        async for _ in middleware.send_request_stream(AgentRequest(prompt="test")):
            pass

        # Should route to healthy provider
        assert call_counts[0] == 0
        assert call_counts[1] == 1


class TestIntegration:
    """Test full integration scenarios."""

    @pytest.mark.asyncio
    async def test_weighted_distribution_over_many_requests(self) -> None:
        """Weighted strategy distributes correctly over many requests."""
        call_counts = [0, 0, 0]

        class CountingProvider(MockProvider):
            def __init__(self, index):
                super().__init__()
                self.index = index

            async def send_request(self, request):
                call_counts[self.index] += 1
                return await super().send_request(request)

        mock1 = CountingProvider(0)
        mock2 = CountingProvider(1)
        mock3 = CountingProvider(2)

        # 5:3:2 ratio
        middleware = LoadBalancingMiddleware(
            providers=[(mock1, 5.0), (mock2, 3.0), (mock3, 2.0)],
            strategy="weighted",
        )

        for _ in range(100):
            await middleware.send_request(AgentRequest(prompt="test"))

        # Check approximate distribution (50%, 30%, 20%)
        assert 35 <= call_counts[0] <= 65  # ~50%
        assert 15 <= call_counts[1] <= 45  # ~30%
        assert 5 <= call_counts[2] <= 35   # ~20%

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """LoadBalancingMiddleware works as context manager."""
        mock = MockProvider()
        middleware = LoadBalancingMiddleware(providers=[(mock, 1.0)])

        async with middleware:
            response = await middleware.send_request(AgentRequest(prompt="test"))
            assert response.content == "mock"
