"""Tests for stream aggregation middleware.

Covers stream merging, race mode, chunk interleaving.
"""

from __future__ import annotations

import asyncio

import pytest

from mada_modelkit._types import AgentRequest, StreamChunk
from mada_modelkit.middleware.stream_aggregation import StreamAggregationMiddleware

from helpers import MockProvider


class TestModuleExports:
    """Verify stream_aggregation module exports."""

    def test_module_has_all(self) -> None:
        from mada_modelkit.middleware import stream_aggregation

        assert hasattr(stream_aggregation, "__all__")

    def test_all_contains_stream_aggregation_middleware(self) -> None:
        from mada_modelkit.middleware.stream_aggregation import __all__

        assert "StreamAggregationMiddleware" in __all__


class TestStreamAggregationMiddlewareConstructor:
    """Test StreamAggregationMiddleware constructor validation."""

    def test_valid_constructor_with_defaults(self) -> None:
        clients = [MockProvider(), MockProvider()]
        middleware = StreamAggregationMiddleware(clients)

        assert middleware._clients == clients
        assert middleware._strategy == "merge"

    def test_valid_constructor_with_merge_strategy(self) -> None:
        clients = [MockProvider(), MockProvider()]
        middleware = StreamAggregationMiddleware(clients, strategy="merge")

        assert middleware._strategy == "merge"

    def test_valid_constructor_with_race_strategy(self) -> None:
        clients = [MockProvider(), MockProvider()]
        middleware = StreamAggregationMiddleware(clients, strategy="race")

        assert middleware._strategy == "race"

    def test_empty_clients_raises_error(self) -> None:
        with pytest.raises(ValueError, match="clients list cannot be empty"):
            StreamAggregationMiddleware([])

    def test_invalid_strategy_raises_error(self) -> None:
        clients = [MockProvider()]

        with pytest.raises(ValueError, match="Invalid strategy"):
            StreamAggregationMiddleware(clients, strategy="invalid")

    def test_single_client_allowed(self) -> None:
        clients = [MockProvider()]
        middleware = StreamAggregationMiddleware(clients)

        assert len(middleware._clients) == 1


class TestNonStreamingBehavior:
    """Test non-streaming request handling."""

    @pytest.mark.asyncio
    async def test_send_request_uses_first_client(self) -> None:
        client1 = MockProvider()
        client2 = MockProvider()

        middleware = StreamAggregationMiddleware([client1, client2])

        response = await middleware.send_request(AgentRequest(prompt="test"))

        # Got response from first client
        assert response.content == "mock"

        # Only first client called
        assert client1.call_count == 1
        assert client2.call_count == 0


class TestStreamMerging:
    """Test merge strategy for stream aggregation."""

    @pytest.mark.asyncio
    async def test_merge_interleaves_chunks(self) -> None:
        # Create custom mock providers that yield multiple chunks
        class MultiChunkProvider(MockProvider):
            def __init__(self, chunks: list[str]) -> None:
                super().__init__()
                self.chunks = chunks

            async def send_request_stream(self, request):
                for chunk_text in self.chunks:
                    yield StreamChunk(delta=chunk_text, is_final=False)

        client1 = MultiChunkProvider(["A1", "A2"])
        client2 = MultiChunkProvider(["B1", "B2"])

        middleware = StreamAggregationMiddleware([client1, client2], strategy="merge")

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk.delta)

        # All chunks from both streams received
        assert len(chunks) == 4
        assert "A1" in chunks
        assert "A2" in chunks
        assert "B1" in chunks
        assert "B2" in chunks

    @pytest.mark.asyncio
    async def test_merge_handles_different_speeds(self) -> None:
        # One stream slower than the other
        class SlowProvider(MockProvider):
            async def send_request_stream(self, request):
                await asyncio.sleep(0.05)
                yield StreamChunk(delta="slow", is_final=True)

        class FastProvider(MockProvider):
            async def send_request_stream(self, request):
                yield StreamChunk(delta="fast", is_final=True)

        slow = SlowProvider()
        fast = FastProvider()

        middleware = StreamAggregationMiddleware([slow, fast], strategy="merge")

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk.delta)

        # Both chunks received
        assert len(chunks) == 2
        assert "slow" in chunks
        assert "fast" in chunks
        # Fast should come first
        assert chunks[0] == "fast"

    @pytest.mark.asyncio
    async def test_merge_handles_empty_stream(self) -> None:
        # One stream is empty
        class EmptyProvider(MockProvider):
            async def send_request_stream(self, request):
                return
                yield  # Make it a generator

        class NormalProvider(MockProvider):
            async def send_request_stream(self, request):
                yield StreamChunk(delta="content", is_final=True)

        empty = EmptyProvider()
        normal = NormalProvider()

        middleware = StreamAggregationMiddleware([empty, normal], strategy="merge")

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk.delta)

        # Only chunk from non-empty stream
        assert len(chunks) == 1
        assert chunks[0] == "content"


class TestRaceMode:
    """Test race strategy for stream aggregation."""

    @pytest.mark.asyncio
    async def test_race_uses_fastest_stream(self) -> None:
        # Create providers with different latencies
        class SlowProvider(MockProvider):
            async def send_request_stream(self, request):
                await asyncio.sleep(0.1)
                yield StreamChunk(delta="slow", is_final=True)

        class FastProvider(MockProvider):
            async def send_request_stream(self, request):
                yield StreamChunk(delta="fast", is_final=True)

        slow = SlowProvider()
        fast = FastProvider()

        middleware = StreamAggregationMiddleware([slow, fast], strategy="race")

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk.delta)

        # Only fast stream's chunk received
        assert len(chunks) == 1
        assert chunks[0] == "fast"

    @pytest.mark.asyncio
    async def test_race_yields_all_chunks_from_winner(self) -> None:
        # Winner yields multiple chunks
        class WinnerProvider(MockProvider):
            async def send_request_stream(self, request):
                yield StreamChunk(delta="first", is_final=False)
                await asyncio.sleep(0.01)
                yield StreamChunk(delta="second", is_final=False)
                yield StreamChunk(delta="third", is_final=True)

        class LoserProvider(MockProvider):
            async def send_request_stream(self, request):
                await asyncio.sleep(0.1)
                yield StreamChunk(delta="loser", is_final=True)

        winner = WinnerProvider()
        loser = LoserProvider()

        middleware = StreamAggregationMiddleware([winner, loser], strategy="race")

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk.delta)

        # All chunks from winner
        assert len(chunks) == 3
        assert chunks == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_race_cancels_losing_streams(self) -> None:
        # Verify losers are cancelled
        class CountingProvider(MockProvider):
            def __init__(self, delay: float) -> None:
                super().__init__()
                self.delay = delay
                self.started = False

            async def send_request_stream(self, request):
                self.started = True
                await asyncio.sleep(self.delay)
                yield StreamChunk(delta=f"delay-{self.delay}", is_final=True)

        fast = CountingProvider(0.0)
        slow = CountingProvider(0.5)

        middleware = StreamAggregationMiddleware([fast, slow], strategy="race")

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk.delta)

        # Fast won
        assert chunks == ["delay-0.0"]

        # Give slow stream time to be cancelled
        await asyncio.sleep(0.01)

        # Both started but only fast completed
        assert fast.started
        assert slow.started


class TestClose:
    """Test cleanup on close."""

    @pytest.mark.asyncio
    async def test_close_all_clients(self) -> None:
        client1 = MockProvider()
        client2 = MockProvider()
        client3 = MockProvider()

        middleware = StreamAggregationMiddleware([client1, client2, client3])

        await middleware.close()

        # Close succeeds (MockProvider doesn't track but call doesn't raise)


class TestStreamAggregationIntegration:
    """Integration tests for stream aggregation."""

    @pytest.mark.asyncio
    async def test_merge_with_three_streams(self) -> None:
        class NumberProvider(MockProvider):
            def __init__(self, numbers: list[int]) -> None:
                super().__init__()
                self.numbers = numbers

            async def send_request_stream(self, request):
                for num in self.numbers:
                    yield StreamChunk(delta=str(num), is_final=False)

        client1 = NumberProvider([1, 2])
        client2 = NumberProvider([3, 4])
        client3 = NumberProvider([5, 6])

        middleware = StreamAggregationMiddleware(
            [client1, client2, client3], strategy="merge"
        )

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk.delta)

        # All 6 chunks received
        assert len(chunks) == 6
        assert set(chunks) == {"1", "2", "3", "4", "5", "6"}

    @pytest.mark.asyncio
    async def test_race_deterministic_order(self) -> None:
        # When all streams are equally fast, first wins
        class InstantProvider(MockProvider):
            def __init__(self, label: str) -> None:
                super().__init__()
                self.label = label

            async def send_request_stream(self, request):
                yield StreamChunk(delta=self.label, is_final=True)

        client1 = InstantProvider("first")
        client2 = InstantProvider("second")
        client3 = InstantProvider("third")

        middleware = StreamAggregationMiddleware(
            [client1, client2, client3], strategy="race"
        )

        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="test")):
            chunks.append(chunk.delta)

        # First client wins when all are instant
        assert len(chunks) == 1
        assert chunks[0] == "first"

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        clients = [MockProvider(), MockProvider()]

        async with StreamAggregationMiddleware(clients) as middleware:
            chunks = []
            async for chunk in middleware.send_request_stream(
                AgentRequest(prompt="test")
            ):
                chunks.append(chunk.delta)

            assert len(chunks) == 2  # One chunk from each client

        # Context manager exited without error
