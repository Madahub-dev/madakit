"""Tests for A/B testing middleware.

Tests ABTestMiddleware constructor, traffic splitting, variant selection,
deterministic routing, and metadata tagging.
"""

from __future__ import annotations

import pytest

from madakit._types import AgentRequest, AgentResponse
from madakit.middleware.ab_test import ABTestMiddleware

from helpers import MockProvider


class TestModuleExports:
    """Test module-level exports and imports."""

    def test_all_exports(self) -> None:
        """__all__ contains only ABTestMiddleware."""
        from madakit.middleware import ab_test

        assert ab_test.__all__ == ["ABTestMiddleware"]

    def test_middleware_importable(self) -> None:
        """ABTestMiddleware can be imported from module."""
        from madakit.middleware.ab_test import ABTestMiddleware as ABM

        assert ABM is not None


class TestABTestMiddlewareConstructor:
    """Test ABTestMiddleware constructor and initialization."""

    def test_minimal_constructor(self) -> None:
        """Constructor accepts variants list."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        variants = [(mock1, 1.0), (mock2, 1.0)]

        middleware = ABTestMiddleware(variants=variants)

        assert middleware._variants == variants
        assert middleware._key_fn is None

    def test_with_custom_key_fn(self) -> None:
        """Constructor accepts custom key function."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        def custom_key(request: AgentRequest) -> str:
            return request.metadata.get("user_id", "default")

        middleware = ABTestMiddleware(
            variants=[(mock1, 1.0), (mock2, 1.0)],
            key_fn=custom_key,
        )

        assert middleware._key_fn is custom_key

    def test_weights_normalized(self) -> None:
        """Variant weights are normalized to sum to 1.0."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        mock3 = MockProvider()

        middleware = ABTestMiddleware(
            variants=[(mock1, 2.0), (mock2, 3.0), (mock3, 5.0)]
        )

        # Total weight is 10.0, so normalized: [0.2, 0.3, 0.5]
        assert abs(middleware._normalized_weights[0] - 0.2) < 0.001
        assert abs(middleware._normalized_weights[1] - 0.3) < 0.001
        assert abs(middleware._normalized_weights[2] - 0.5) < 0.001

    def test_cumulative_weights_computed(self) -> None:
        """Cumulative weights computed for distribution."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        middleware = ABTestMiddleware(
            variants=[(mock1, 0.3), (mock2, 0.7)]
        )

        # Cumulative: [0.3, 1.0]
        assert abs(middleware._cumulative_weights[0] - 0.3) < 0.001
        assert abs(middleware._cumulative_weights[1] - 1.0) < 0.001

    def test_empty_variants_raises(self) -> None:
        """Empty variants list raises ValueError."""
        with pytest.raises(ValueError, match="At least one variant is required"):
            ABTestMiddleware(variants=[])

    def test_zero_weight_raises(self) -> None:
        """Zero weight raises ValueError."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        with pytest.raises(ValueError, match="All variant weights must be positive"):
            ABTestMiddleware(variants=[(mock1, 1.0), (mock2, 0.0)])

    def test_negative_weight_raises(self) -> None:
        """Negative weight raises ValueError."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        with pytest.raises(ValueError, match="All variant weights must be positive"):
            ABTestMiddleware(variants=[(mock1, 1.0), (mock2, -0.5)])

    def test_super_init_called(self) -> None:
        """ABTestMiddleware calls BaseAgentClient.__init__."""
        mock = MockProvider()
        middleware = ABTestMiddleware(variants=[(mock, 1.0)])

        assert hasattr(middleware, "send_request")
        assert hasattr(middleware, "send_request_stream")


class TestTrafficSplitting:
    """Test traffic splitting and variant selection."""

    def test_deterministic_routing(self) -> None:
        """Same key always routes to same variant."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        middleware = ABTestMiddleware(variants=[(mock1, 1.0), (mock2, 1.0)])

        request = AgentRequest(prompt="test prompt")

        # Get variant for same request multiple times
        variants = []
        for _ in range(10):
            _, variant_index = middleware._select_variant(request)
            variants.append(variant_index)

        # All should be the same
        assert len(set(variants)) == 1

    def test_different_keys_different_variants(self) -> None:
        """Different keys can route to different variants."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        middleware = ABTestMiddleware(variants=[(mock1, 1.0), (mock2, 1.0)])

        # Try many different prompts until we see both variants
        variants_seen = set()
        for i in range(100):
            request = AgentRequest(prompt=f"test prompt {i}")
            _, variant_index = middleware._select_variant(request)
            variants_seen.add(variant_index)

            if len(variants_seen) == 2:
                break

        # Should see both variants with 50/50 split
        assert len(variants_seen) == 2

    def test_weight_distribution_approximate(self) -> None:
        """Variant selection approximates weight distribution."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        # 80/20 split
        middleware = ABTestMiddleware(variants=[(mock1, 0.8), (mock2, 0.2)])

        # Count selections over many requests
        counts = [0, 0]
        for i in range(1000):
            request = AgentRequest(prompt=f"request {i}")
            _, variant_index = middleware._select_variant(request)
            counts[variant_index] += 1

        # Check approximate distribution (allow ±10% tolerance)
        variant1_ratio = counts[0] / 1000
        variant2_ratio = counts[1] / 1000

        assert 0.7 < variant1_ratio < 0.9  # ~80%
        assert 0.1 < variant2_ratio < 0.3  # ~20%

    def test_custom_key_fn_used(self) -> None:
        """Custom key function is used for routing."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        def key_fn(request: AgentRequest) -> str:
            return request.metadata.get("user_id", "default")

        middleware = ABTestMiddleware(
            variants=[(mock1, 1.0), (mock2, 1.0)],
            key_fn=key_fn,
        )

        # Same user_id should route to same variant
        request1 = AgentRequest(prompt="prompt1", metadata={"user_id": "user123"})
        request2 = AgentRequest(prompt="prompt2", metadata={"user_id": "user123"})

        _, variant1 = middleware._select_variant(request1)
        _, variant2 = middleware._select_variant(request2)

        assert variant1 == variant2

    def test_single_variant_always_selected(self) -> None:
        """Single variant is always selected."""
        mock = MockProvider()
        middleware = ABTestMiddleware(variants=[(mock, 1.0)])

        for i in range(10):
            request = AgentRequest(prompt=f"test {i}")
            _, variant_index = middleware._select_variant(request)
            assert variant_index == 0


class TestMetadataTagging:
    """Test variant metadata tagging in responses."""

    @pytest.mark.asyncio
    async def test_send_request_adds_variant_metadata(self) -> None:
        """send_request adds variant index to response metadata."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        middleware = ABTestMiddleware(variants=[(mock1, 1.0), (mock2, 1.0)])

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert "variant" in response.metadata
        assert response.metadata["variant"] in [0, 1]

    @pytest.mark.asyncio
    async def test_send_request_preserves_response_data(self) -> None:
        """send_request preserves response content and tokens."""
        mock = MockProvider()
        middleware = ABTestMiddleware(variants=[(mock, 1.0)])

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert response.content == "mock"
        assert response.model == "mock"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    @pytest.mark.asyncio
    async def test_send_request_preserves_existing_metadata(self) -> None:
        """send_request preserves existing response metadata."""
        class MetadataProvider(MockProvider):
            async def send_request(self, request):
                response = await super().send_request(request)
                return AgentResponse(
                    content=response.content,
                    model=response.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    metadata={"custom": "value"},
                )

        mock = MetadataProvider()
        middleware = ABTestMiddleware(variants=[(mock, 1.0)])

        request = AgentRequest(prompt="test")
        response = await middleware.send_request(request)

        assert response.metadata["custom"] == "value"
        assert response.metadata["variant"] == 0

    @pytest.mark.asyncio
    async def test_send_request_stream_adds_variant_to_final_chunk(self) -> None:
        """send_request_stream adds variant to final chunk metadata."""
        class StreamProvider(MockProvider):
            async def send_request_stream(self, request):
                from madakit._types import StreamChunk

                yield StreamChunk(delta="chunk1", is_final=False)
                yield StreamChunk(delta="chunk2", is_final=True, metadata={"model": "test"})

        mock = StreamProvider()
        middleware = ABTestMiddleware(variants=[(mock, 1.0)])

        request = AgentRequest(prompt="test")
        chunks = []
        async for chunk in middleware.send_request_stream(request):
            chunks.append(chunk)

        # First chunk unchanged
        assert chunks[0].is_final is False
        assert "variant" not in chunks[0].metadata

        # Final chunk has variant
        assert chunks[1].is_final is True
        assert chunks[1].metadata["variant"] == 0
        assert chunks[1].metadata["model"] == "test"

    @pytest.mark.asyncio
    async def test_send_request_stream_preserves_chunks(self) -> None:
        """send_request_stream preserves all chunks."""
        class MultiChunkProvider(MockProvider):
            async def send_request_stream(self, request):
                from madakit._types import StreamChunk

                yield StreamChunk(delta="a", is_final=False)
                yield StreamChunk(delta="b", is_final=False)
                yield StreamChunk(delta="c", is_final=True)

        mock = MultiChunkProvider()
        middleware = ABTestMiddleware(variants=[(mock, 1.0)])

        request = AgentRequest(prompt="test")
        deltas = []
        async for chunk in middleware.send_request_stream(request):
            deltas.append(chunk.delta)

        assert deltas == ["a", "b", "c"]


class TestVariantDistribution:
    """Test variant distribution accuracy."""

    @pytest.mark.asyncio
    async def test_equal_weights_approximate_equal_distribution(self) -> None:
        """Equal weights produce approximately equal distribution."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        middleware = ABTestMiddleware(variants=[(mock1, 1.0), (mock2, 1.0)])

        variant_counts = [0, 0]
        for i in range(200):
            request = AgentRequest(prompt=f"request {i}")
            response = await middleware.send_request(request)
            variant_counts[response.metadata["variant"]] += 1

        # Allow ±20% tolerance for 50/50 split
        assert 80 < variant_counts[0] < 120
        assert 80 < variant_counts[1] < 120

    @pytest.mark.asyncio
    async def test_unequal_weights_approximate_distribution(self) -> None:
        """Unequal weights produce approximately weighted distribution."""
        mock1 = MockProvider()
        mock2 = MockProvider()
        mock3 = MockProvider()

        # 50%, 30%, 20% split
        middleware = ABTestMiddleware(
            variants=[(mock1, 0.5), (mock2, 0.3), (mock3, 0.2)]
        )

        variant_counts = [0, 0, 0]
        for i in range(500):
            request = AgentRequest(prompt=f"request {i}")
            response = await middleware.send_request(request)
            variant_counts[response.metadata["variant"]] += 1

        # Check approximate distribution (±15% tolerance)
        assert 175 < variant_counts[0] < 325  # ~50%
        assert 75 < variant_counts[1] < 225   # ~30%
        assert 25 < variant_counts[2] < 175   # ~20%

    @pytest.mark.asyncio
    async def test_consistent_routing_per_key(self) -> None:
        """Same key consistently routes to same variant."""
        mock1 = MockProvider()
        mock2 = MockProvider()

        def key_fn(request: AgentRequest) -> str:
            return request.metadata.get("session_id", "")

        middleware = ABTestMiddleware(
            variants=[(mock1, 1.0), (mock2, 1.0)],
            key_fn=key_fn,
        )

        # Same session_id should always get same variant
        session_id = "session_abc"
        variants = []
        for i in range(10):
            request = AgentRequest(
                prompt=f"different prompt {i}",
                metadata={"session_id": session_id},
            )
            response = await middleware.send_request(request)
            variants.append(response.metadata["variant"])

        # All should be the same variant
        assert len(set(variants)) == 1
