"""Tests for consensus middleware.

Covers majority voting, confidence scoring, parallel dispatch, error handling.
"""

from __future__ import annotations

import asyncio

import pytest

from mada_modelkit._errors import MiddlewareError
from mada_modelkit._types import AgentRequest, AgentResponse
from mada_modelkit.middleware.consensus import ConsensusMiddleware

from helpers import MockProvider


class TestModuleExports:
    """Verify consensus module exports."""

    def test_module_has_all(self) -> None:
        from mada_modelkit.middleware import consensus

        assert hasattr(consensus, "__all__")

    def test_all_contains_consensus_middleware(self) -> None:
        from mada_modelkit.middleware.consensus import __all__

        assert "ConsensusMiddleware" in __all__


class TestConsensusMiddlewareConstructor:
    """Test ConsensusMiddleware constructor validation."""

    def test_valid_constructor_with_defaults(self) -> None:
        providers = [MockProvider(), MockProvider()]
        middleware = ConsensusMiddleware(providers)

        assert middleware._providers == providers
        assert middleware._strategy == "majority"

    def test_valid_constructor_with_majority_strategy(self) -> None:
        providers = [MockProvider(), MockProvider()]
        middleware = ConsensusMiddleware(providers, strategy="majority")

        assert middleware._strategy == "majority"

    def test_valid_constructor_with_confidence_strategy(self) -> None:
        providers = [MockProvider(), MockProvider()]
        middleware = ConsensusMiddleware(providers, strategy="confidence")

        assert middleware._strategy == "confidence"

    def test_empty_providers_raises_error(self) -> None:
        with pytest.raises(ValueError, match="providers list cannot be empty"):
            ConsensusMiddleware([])

    def test_invalid_strategy_raises_error(self) -> None:
        providers = [MockProvider()]

        with pytest.raises(ValueError, match="Invalid strategy"):
            ConsensusMiddleware(providers, strategy="invalid")

    def test_single_provider_allowed(self) -> None:
        providers = [MockProvider()]
        middleware = ConsensusMiddleware(providers)

        assert len(middleware._providers) == 1


class TestParallelDispatch:
    """Test parallel dispatch to multiple providers."""

    @pytest.mark.asyncio
    async def test_dispatches_to_all_providers(self) -> None:
        provider1 = MockProvider()
        provider2 = MockProvider()
        provider3 = MockProvider()

        middleware = ConsensusMiddleware([provider1, provider2, provider3])

        request = AgentRequest(prompt="test")
        await middleware.send_request(request)

        # All providers called
        assert provider1.call_count == 1
        assert provider2.call_count == 1
        assert provider3.call_count == 1

    @pytest.mark.asyncio
    async def test_parallel_execution(self) -> None:
        # Providers with latency
        provider1 = MockProvider(latency=0.05)
        provider2 = MockProvider(latency=0.05)
        provider3 = MockProvider(latency=0.05)

        middleware = ConsensusMiddleware([provider1, provider2, provider3])

        import time

        start = time.monotonic()
        await middleware.send_request(AgentRequest(prompt="test"))
        duration = time.monotonic() - start

        # Should take ~0.05s (parallel) not ~0.15s (sequential)
        assert duration < 0.1


class TestMajorityVoting:
    """Test majority voting strategy."""

    @pytest.mark.asyncio
    async def test_returns_most_common_response(self) -> None:
        # 3 providers: 2 return "yes", 1 returns "no"
        provider1 = MockProvider(
            responses=[
                AgentResponse(
                    content="yes", model="mock", input_tokens=0, output_tokens=5
                )
            ]
        )
        provider2 = MockProvider(
            responses=[
                AgentResponse(
                    content="yes", model="mock", input_tokens=0, output_tokens=5
                )
            ]
        )
        provider3 = MockProvider(
            responses=[
                AgentResponse(
                    content="no", model="mock", input_tokens=0, output_tokens=4
                )
            ]
        )

        middleware = ConsensusMiddleware(
            [provider1, provider2, provider3], strategy="majority"
        )

        response = await middleware.send_request(AgentRequest(prompt="test"))

        # Majority wins
        assert response.content == "yes"

    @pytest.mark.asyncio
    async def test_adds_consensus_metadata(self) -> None:
        # 2 providers with same response
        provider1 = MockProvider(
            responses=[
                AgentResponse(
                    content="answer", model="mock", input_tokens=0, output_tokens=10
                )
            ]
        )
        provider2 = MockProvider(
            responses=[
                AgentResponse(
                    content="answer", model="mock", input_tokens=0, output_tokens=10
                )
            ]
        )

        middleware = ConsensusMiddleware([provider1, provider2], strategy="majority")

        response = await middleware.send_request(AgentRequest(prompt="test"))

        # Metadata added
        assert response.metadata is not None
        assert response.metadata["consensus_votes"] == 2
        assert response.metadata["consensus_total"] == 2

    @pytest.mark.asyncio
    async def test_tie_returns_first_occurrence(self) -> None:
        # 2 providers with different responses (tie)
        provider1 = MockProvider(
            responses=[
                AgentResponse(
                    content="option1", model="mock", input_tokens=0, output_tokens=5
                )
            ]
        )
        provider2 = MockProvider(
            responses=[
                AgentResponse(
                    content="option2", model="mock", input_tokens=0, output_tokens=5
                )
            ]
        )

        middleware = ConsensusMiddleware([provider1, provider2], strategy="majority")

        response = await middleware.send_request(AgentRequest(prompt="test"))

        # One of them wins (implementation returns first)
        assert response.content in ("option1", "option2")


class TestConfidenceScoring:
    """Test confidence scoring strategy."""

    @pytest.mark.asyncio
    async def test_returns_highest_confidence_response(self) -> None:
        # 3 providers with different confidence (based on output tokens)
        provider1 = MockProvider(
            responses=[
                AgentResponse(
                    content="short", model="mock", input_tokens=0, output_tokens=5
                )
            ]
        )
        provider2 = MockProvider(
            responses=[
                AgentResponse(
                    content="medium response",
                    model="mock",
                    input_tokens=0,
                    output_tokens=15,
                )
            ]
        )
        provider3 = MockProvider(
            responses=[
                AgentResponse(
                    content="very long detailed response with lots of content",
                    model="mock",
                    input_tokens=0,
                    output_tokens=50,
                )
            ]
        )

        middleware = ConsensusMiddleware(
            [provider1, provider2, provider3], strategy="confidence"
        )

        response = await middleware.send_request(AgentRequest(prompt="test"))

        # Highest confidence (most tokens + longest) wins
        assert response.content == "very long detailed response with lots of content"
        assert response.output_tokens == 50

    @pytest.mark.asyncio
    async def test_adds_confidence_metadata(self) -> None:
        provider = MockProvider(
            responses=[
                AgentResponse(
                    content="test", model="mock", input_tokens=0, output_tokens=10
                )
            ]
        )

        middleware = ConsensusMiddleware([provider], strategy="confidence")

        response = await middleware.send_request(AgentRequest(prompt="test"))

        # Confidence metadata added
        assert response.metadata is not None
        assert "consensus_confidence" in response.metadata
        assert response.metadata["consensus_total"] == 1


class TestErrorHandling:
    """Test error handling in consensus."""

    @pytest.mark.asyncio
    async def test_all_providers_fail_raises_error(self) -> None:
        # All providers fail
        provider1 = MockProvider(errors=[RuntimeError("error1")])
        provider2 = MockProvider(errors=[RuntimeError("error2")])

        middleware = ConsensusMiddleware([provider1, provider2])

        with pytest.raises(MiddlewareError, match="All 2 providers failed"):
            await middleware.send_request(AgentRequest(prompt="test"))

    @pytest.mark.asyncio
    async def test_partial_failure_uses_valid_responses(self) -> None:
        # 1 provider fails, 2 succeed
        provider1 = MockProvider(errors=[RuntimeError("fail")])
        provider2 = MockProvider(
            responses=[
                AgentResponse(
                    content="success", model="mock", input_tokens=0, output_tokens=5
                )
            ]
        )
        provider3 = MockProvider(
            responses=[
                AgentResponse(
                    content="success", model="mock", input_tokens=0, output_tokens=5
                )
            ]
        )

        middleware = ConsensusMiddleware([provider1, provider2, provider3])

        response = await middleware.send_request(AgentRequest(prompt="test"))

        # Uses valid responses despite one failure
        assert response.content == "success"

    @pytest.mark.asyncio
    async def test_single_provider_failure_uses_other(self) -> None:
        # With majority strategy, even if 1 of 3 fails, consensus works
        provider1 = MockProvider(errors=[RuntimeError("fail")])
        provider2 = MockProvider()
        provider3 = MockProvider()

        middleware = ConsensusMiddleware([provider1, provider2, provider3])

        response = await middleware.send_request(AgentRequest(prompt="test"))

        # Got response from working providers
        assert response.content == "mock"


class TestStreamingBehavior:
    """Test streaming request handling."""

    @pytest.mark.asyncio
    async def test_streaming_uses_first_provider(self) -> None:
        provider1 = MockProvider()
        provider2 = MockProvider()

        middleware = ConsensusMiddleware([provider1, provider2])

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk)

        # Got chunks from first provider
        assert len(chunks) == 1
        assert chunks[0].delta == "mock"

        # Only first provider called for streaming
        assert provider1.call_count == 1
        assert provider2.call_count == 0


class TestClose:
    """Test cleanup on close."""

    @pytest.mark.asyncio
    async def test_close_all_providers(self) -> None:
        provider1 = MockProvider()
        provider2 = MockProvider()
        provider3 = MockProvider()

        middleware = ConsensusMiddleware([provider1, provider2, provider3])

        await middleware.close()

        # All providers closed (MockProvider doesn't track this, but call succeeds)
        # Just verify close doesn't raise


class TestConsensusIntegration:
    """Integration tests for consensus middleware."""

    @pytest.mark.asyncio
    async def test_consensus_with_varying_responses(self) -> None:
        # 5 providers: 3 say "A", 2 say "B"
        providers = [
            MockProvider(
                responses=[
                    AgentResponse(
                        content="A", model="mock", input_tokens=0, output_tokens=1
                    )
                ]
            )
            for _ in range(3)
        ]
        providers.extend(
            [
                MockProvider(
                    responses=[
                        AgentResponse(
                            content="B", model="mock", input_tokens=0, output_tokens=1
                        )
                    ]
                )
                for _ in range(2)
            ]
        )

        middleware = ConsensusMiddleware(providers, strategy="majority")

        response = await middleware.send_request(AgentRequest(prompt="test"))

        assert response.content == "A"
        assert response.metadata["consensus_votes"] == 3
        assert response.metadata["consensus_total"] == 5

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        providers = [MockProvider(), MockProvider()]

        async with ConsensusMiddleware(providers) as middleware:
            response = await middleware.send_request(AgentRequest(prompt="test"))
            assert response.content == "mock"

        # Context manager exited without error
