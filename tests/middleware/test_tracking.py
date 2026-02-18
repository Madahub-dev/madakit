"""Tests for TrackingMiddleware (tasks 2.4.1–2.4.5).

Covers: constructor attribute storage, initial stats state, BaseAgentClient
inheritance, semaphore absence, per-instance stats isolation, middleware
composition, send_request timing and token accumulation, cost_fn invocation
and accumulation, send_request_stream TTFT measurement and inference timing,
token accumulation from final-chunk metadata, exception safety, stats
property identity, stats.reset() interaction, module exports, virtual method
defaults, and end-to-end integration scenarios.
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import patch

import pytest

from helpers import MockProvider
from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk, TrackingStats
from mada_modelkit.middleware.tracking import TrackingMiddleware


class _MultiChunkProvider(BaseAgentClient):
    """Provider that yields a pre-specified list of StreamChunks in order."""

    def __init__(self, chunks: list[StreamChunk]) -> None:
        """Initialise with the list of chunks to yield."""
        super().__init__()
        self._chunks = chunks

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Return a placeholder response (stream tests do not use this path)."""
        return AgentResponse(content="", model="m", input_tokens=0, output_tokens=0)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:  # type: ignore[override]
        """Yield all pre-loaded chunks in order."""
        for chunk in self._chunks:
            yield chunk


class TestStatsProperty:
    """TrackingMiddleware.stats — returns the live TrackingStats instance."""

    def test_returns_tracking_stats_instance(self) -> None:
        """Asserts that stats returns a TrackingStats object."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert isinstance(middleware.stats, TrackingStats)

    def test_returns_same_object_as_internal_stats(self) -> None:
        """Asserts that stats is the same object as _stats (no copy)."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert middleware.stats is middleware._stats

    def test_reflects_mutations_made_via_internal_stats(self) -> None:
        """Asserts that changes to _stats are visible through stats."""
        middleware = TrackingMiddleware(client=MockProvider())
        middleware._stats.total_requests = 7
        assert middleware.stats.total_requests == 7

    @pytest.mark.asyncio
    async def test_reflects_live_state_after_send_request(self) -> None:
        """Asserts that stats reflects updated counts after send_request completes."""
        middleware = TrackingMiddleware(client=MockProvider())
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert middleware.stats.total_requests == 1

    def test_two_instances_have_independent_stats_properties(self) -> None:
        """Asserts that stats on two instances are not the same object."""
        m1 = TrackingMiddleware(client=MockProvider())
        m2 = TrackingMiddleware(client=MockProvider())
        assert m1.stats is not m2.stats


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


class TestSendRequest:
    """TrackingMiddleware.send_request — timing, token accumulation, and cost tracking."""

    @pytest.mark.asyncio
    async def test_increments_total_requests_by_one(self) -> None:
        """Asserts that total_requests is 1 after a single send_request call."""
        middleware = TrackingMiddleware(client=MockProvider())
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert middleware._stats.total_requests == 1

    @pytest.mark.asyncio
    async def test_accumulates_total_requests_across_calls(self) -> None:
        """Asserts that total_requests accumulates correctly over multiple calls."""
        middleware = TrackingMiddleware(client=MockProvider())
        await middleware.send_request(AgentRequest(prompt="a"))
        await middleware.send_request(AgentRequest(prompt="b"))
        assert middleware._stats.total_requests == 2

    @pytest.mark.asyncio
    async def test_accumulates_input_tokens(self) -> None:
        """Asserts that total_input_tokens sums input_tokens from all responses."""
        resp1 = AgentResponse(content="r1", model="m", input_tokens=3, output_tokens=1)
        resp2 = AgentResponse(content="r2", model="m", input_tokens=7, output_tokens=1)
        middleware = TrackingMiddleware(client=MockProvider(responses=[resp1, resp2]))
        await middleware.send_request(AgentRequest(prompt="a"))
        await middleware.send_request(AgentRequest(prompt="b"))
        assert middleware._stats.total_input_tokens == 10

    @pytest.mark.asyncio
    async def test_accumulates_output_tokens(self) -> None:
        """Asserts that total_output_tokens sums output_tokens from all responses."""
        resp1 = AgentResponse(content="r1", model="m", input_tokens=1, output_tokens=4)
        resp2 = AgentResponse(content="r2", model="m", input_tokens=1, output_tokens=6)
        middleware = TrackingMiddleware(client=MockProvider(responses=[resp1, resp2]))
        await middleware.send_request(AgentRequest(prompt="a"))
        await middleware.send_request(AgentRequest(prompt="b"))
        assert middleware._stats.total_output_tokens == 10

    @pytest.mark.asyncio
    async def test_inference_ms_is_positive_after_call(self) -> None:
        """Asserts that total_inference_ms is greater than zero after one call."""
        middleware = TrackingMiddleware(client=MockProvider())
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert middleware._stats.total_inference_ms > 0.0

    @pytest.mark.asyncio
    async def test_inference_ms_uses_perf_counter(self) -> None:
        """Asserts that inference_ms is derived from time.perf_counter readings."""
        side_effects = [1.0, 1.5]  # start=1.0, end=1.5 → 500 ms
        middleware = TrackingMiddleware(client=MockProvider())
        with patch("mada_modelkit.middleware.tracking.time.perf_counter", side_effect=side_effects):
            await middleware.send_request(AgentRequest(prompt="hi"))
        assert middleware._stats.total_inference_ms == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_inference_ms_accumulates_across_calls(self) -> None:
        """Asserts that total_inference_ms sums elapsed time across multiple calls."""
        side_effects = [0.0, 0.1, 0.0, 0.2]  # 100 ms + 200 ms = 300 ms
        middleware = TrackingMiddleware(client=MockProvider())
        with patch("mada_modelkit.middleware.tracking.time.perf_counter", side_effect=side_effects):
            await middleware.send_request(AgentRequest(prompt="a"))
            await middleware.send_request(AgentRequest(prompt="b"))
        assert middleware._stats.total_inference_ms == pytest.approx(300.0)

    @pytest.mark.asyncio
    async def test_no_cost_fn_leaves_cost_usd_none(self) -> None:
        """Asserts that total_cost_usd remains None when cost_fn is not provided."""
        middleware = TrackingMiddleware(client=MockProvider())
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert middleware._stats.total_cost_usd is None

    @pytest.mark.asyncio
    async def test_cost_fn_called_with_response(self) -> None:
        """Asserts that cost_fn receives the AgentResponse returned by the client."""
        received: list[AgentResponse] = []
        resp = AgentResponse(content="x", model="m", input_tokens=1, output_tokens=1)
        middleware = TrackingMiddleware(
            client=MockProvider(responses=[resp]),
            cost_fn=lambda r: received.append(r) or 0.0,  # type: ignore[return-value]
        )
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert len(received) == 1
        assert received[0] is resp

    @pytest.mark.asyncio
    async def test_cost_fn_result_stored_in_total_cost_usd(self) -> None:
        """Asserts that the cost_fn return value is stored in total_cost_usd."""
        middleware = TrackingMiddleware(client=MockProvider(), cost_fn=lambda r: 0.05)
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert middleware._stats.total_cost_usd == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_cost_fn_accumulates_across_calls(self) -> None:
        """Asserts that total_cost_usd sums cost_fn results over multiple calls."""
        middleware = TrackingMiddleware(client=MockProvider(), cost_fn=lambda r: 0.03)
        await middleware.send_request(AgentRequest(prompt="a"))
        await middleware.send_request(AgentRequest(prompt="b"))
        assert middleware._stats.total_cost_usd == pytest.approx(0.06)

    @pytest.mark.asyncio
    async def test_exception_does_not_update_stats(self) -> None:
        """Asserts that stats remain unchanged when the wrapped client raises."""
        middleware = TrackingMiddleware(client=MockProvider(errors=[RuntimeError("boom")]))
        with pytest.raises(RuntimeError):
            await middleware.send_request(AgentRequest(prompt="hi"))
        assert middleware._stats.total_requests == 0
        assert middleware._stats.total_input_tokens == 0
        assert middleware._stats.total_output_tokens == 0
        assert middleware._stats.total_inference_ms == 0.0
        assert middleware._stats.total_cost_usd is None

    @pytest.mark.asyncio
    async def test_returns_response_from_client(self) -> None:
        """Asserts that send_request returns exactly the response from the wrapped client."""
        expected = AgentResponse(content="z", model="m", input_tokens=2, output_tokens=3)
        middleware = TrackingMiddleware(client=MockProvider(responses=[expected]))
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected


class TestSendRequestStream:
    """TrackingMiddleware.send_request_stream — TTFT, inference timing, token tracking."""

    @pytest.mark.asyncio
    async def test_ttft_ms_set_in_first_chunk_metadata(self) -> None:
        """Asserts that the first yielded chunk has ttft_ms in its metadata dict."""
        middleware = TrackingMiddleware(client=MockProvider())
        chunks = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert "ttft_ms" in chunks[0].metadata

    @pytest.mark.asyncio
    async def test_ttft_ms_is_non_negative(self) -> None:
        """Asserts that metadata["ttft_ms"] on the first chunk is >= 0.0."""
        middleware = TrackingMiddleware(client=MockProvider())
        chunks = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks[0].metadata["ttft_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_ttft_ms_uses_perf_counter(self) -> None:
        """Asserts that ttft_ms equals the difference between start and first-chunk perf_counter."""
        # Single-chunk stream: perf_counter called as start, ttft, elapsed (3 calls).
        side_effects = [0.0, 0.3, 0.5]
        middleware = TrackingMiddleware(client=MockProvider())
        with patch("mada_modelkit.middleware.tracking.time.perf_counter", side_effect=side_effects):
            chunks = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks[0].metadata["ttft_ms"] == pytest.approx(300.0)

    @pytest.mark.asyncio
    async def test_total_ttft_ms_accumulates_across_calls(self) -> None:
        """Asserts that total_ttft_ms sums TTFT values across multiple stream calls."""
        # Each single-chunk stream uses 3 perf_counter calls: start, ttft, elapsed.
        side_effects = [0.0, 0.1, 0.2,   # call 1: ttft = 100 ms
                        0.0, 0.3, 0.6]   # call 2: ttft = 300 ms
        middleware = TrackingMiddleware(client=MockProvider())
        with patch("mada_modelkit.middleware.tracking.time.perf_counter", side_effect=side_effects):
            async for _ in middleware.send_request_stream(AgentRequest(prompt="a")):
                pass
            async for _ in middleware.send_request_stream(AgentRequest(prompt="b")):
                pass
        assert middleware._stats.total_ttft_ms == pytest.approx(400.0)

    @pytest.mark.asyncio
    async def test_inference_ms_accumulated_on_final_chunk(self) -> None:
        """Asserts that total_inference_ms is positive after the stream completes."""
        middleware = TrackingMiddleware(client=MockProvider())
        async for _ in middleware.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert middleware._stats.total_inference_ms > 0.0

    @pytest.mark.asyncio
    async def test_inference_ms_uses_perf_counter(self) -> None:
        """Asserts that inference_ms equals elapsed time from start to the final chunk."""
        side_effects = [0.0, 0.1, 0.5]  # start=0, ttft at 0.1, elapsed at 0.5 → 500 ms
        middleware = TrackingMiddleware(client=MockProvider())
        with patch("mada_modelkit.middleware.tracking.time.perf_counter", side_effect=side_effects):
            async for _ in middleware.send_request_stream(AgentRequest(prompt="hi")):
                pass
        assert middleware._stats.total_inference_ms == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_non_first_chunk_has_no_ttft_in_metadata(self) -> None:
        """Asserts that subsequent chunks do not have ttft_ms injected into metadata."""
        chunks_in = [
            StreamChunk(delta="a"),
            StreamChunk(delta="b", is_final=True),
        ]
        middleware = TrackingMiddleware(client=_MultiChunkProvider(chunks_in))
        chunks_out = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert "ttft_ms" not in chunks_out[1].metadata

    @pytest.mark.asyncio
    async def test_single_chunk_stream_has_ttft_when_is_final(self) -> None:
        """Asserts that a single-chunk stream with is_final=True still gets ttft_ms."""
        middleware = TrackingMiddleware(client=MockProvider())
        chunks = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks[0].is_final is True
        assert "ttft_ms" in chunks[0].metadata

    @pytest.mark.asyncio
    async def test_input_tokens_accumulated_from_final_chunk_metadata(self) -> None:
        """Asserts that input_tokens from the final chunk's metadata is accumulated."""
        chunks_in = [StreamChunk(delta="x", is_final=True, metadata={"input_tokens": 5})]
        middleware = TrackingMiddleware(client=_MultiChunkProvider(chunks_in))
        async for _ in middleware.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert middleware._stats.total_input_tokens == 5

    @pytest.mark.asyncio
    async def test_output_tokens_accumulated_from_final_chunk_metadata(self) -> None:
        """Asserts that output_tokens from the final chunk's metadata is accumulated."""
        chunks_in = [StreamChunk(delta="x", is_final=True, metadata={"output_tokens": 3})]
        middleware = TrackingMiddleware(client=_MultiChunkProvider(chunks_in))
        async for _ in middleware.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert middleware._stats.total_output_tokens == 3

    @pytest.mark.asyncio
    async def test_tokens_default_to_zero_when_absent_from_metadata(self) -> None:
        """Asserts that token stats remain zero if the final chunk has no token metadata."""
        middleware = TrackingMiddleware(client=MockProvider())
        async for _ in middleware.send_request_stream(AgentRequest(prompt="hi")):
            pass
        assert middleware._stats.total_input_tokens == 0
        assert middleware._stats.total_output_tokens == 0

    @pytest.mark.asyncio
    async def test_exception_before_first_chunk_does_not_update_stats(self) -> None:
        """Asserts that stats remain unchanged when the client raises before yielding."""
        middleware = TrackingMiddleware(client=MockProvider(errors=[RuntimeError("fail")]))
        with pytest.raises(RuntimeError):
            async for _ in middleware.send_request_stream(AgentRequest(prompt="hi")):
                pass
        assert middleware._stats.total_ttft_ms == 0.0
        assert middleware._stats.total_inference_ms == 0.0
        assert middleware._stats.total_input_tokens == 0
        assert middleware._stats.total_output_tokens == 0

    @pytest.mark.asyncio
    async def test_original_chunk_delta_preserved(self) -> None:
        """Asserts that the delta value is preserved when ttft_ms metadata is injected."""
        chunks_in = [StreamChunk(delta="hello", is_final=True)]
        middleware = TrackingMiddleware(client=_MultiChunkProvider(chunks_in))
        chunks_out = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks_out[0].delta == "hello"

    @pytest.mark.asyncio
    async def test_original_chunk_is_final_preserved(self) -> None:
        """Asserts that is_final is preserved when ttft_ms metadata is injected."""
        chunks_in = [StreamChunk(delta="x", is_final=True)]
        middleware = TrackingMiddleware(client=_MultiChunkProvider(chunks_in))
        chunks_out = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks_out[0].is_final is True


class TestModuleExports:
    """Module-level exports — __all__ and public name availability."""

    def test_all_is_defined(self) -> None:
        """Asserts that __all__ is defined in the tracking module."""
        import mada_modelkit.middleware.tracking as mod
        assert hasattr(mod, "__all__")

    def test_tracking_middleware_in_all(self) -> None:
        """Asserts that 'TrackingMiddleware' is listed in __all__."""
        import mada_modelkit.middleware.tracking as mod
        assert "TrackingMiddleware" in mod.__all__

    def test_tracking_middleware_importable(self) -> None:
        """Asserts that TrackingMiddleware can be imported from the tracking module."""
        from mada_modelkit.middleware.tracking import TrackingMiddleware as TM
        assert TM is TrackingMiddleware


class TestVirtualMethodDefaults:
    """TrackingMiddleware inherited virtual methods — health_check, cancel, close."""

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self) -> None:
        """Asserts that health_check returns True (inherited default)."""
        middleware = TrackingMiddleware(client=MockProvider())
        assert await middleware.health_check() is True

    @pytest.mark.asyncio
    async def test_cancel_is_no_op(self) -> None:
        """Asserts that cancel() completes without raising."""
        middleware = TrackingMiddleware(client=MockProvider())
        await middleware.cancel()  # must not raise

    @pytest.mark.asyncio
    async def test_close_is_no_op(self) -> None:
        """Asserts that close() completes without raising."""
        middleware = TrackingMiddleware(client=MockProvider())
        await middleware.close()  # must not raise


class TestIntegration:
    """TrackingMiddleware end-to-end — mixed calls, reset, stacking, and context manager."""

    @pytest.mark.asyncio
    async def test_mixed_send_request_and_stream_accumulate_all_stats(self) -> None:
        """Asserts that send_request and send_request_stream both contribute to stats."""
        resp = AgentResponse(content="r", model="m", input_tokens=3, output_tokens=2)
        stream_chunks = [StreamChunk(delta="s", is_final=True, metadata={"input_tokens": 1, "output_tokens": 4})]
        provider_seq = MockProvider(responses=[resp])
        middleware = TrackingMiddleware(client=provider_seq)
        await middleware.send_request(AgentRequest(prompt="a"))
        # override client for stream call
        middleware._client = _MultiChunkProvider(stream_chunks)
        async for _ in middleware.send_request_stream(AgentRequest(prompt="b")):
            pass
        assert middleware.stats.total_requests == 1       # only send_request increments
        assert middleware.stats.total_input_tokens == 4   # 3 + 1
        assert middleware.stats.total_output_tokens == 6  # 2 + 4
        assert middleware.stats.total_inference_ms > 0.0
        assert middleware.stats.total_ttft_ms > 0.0

    @pytest.mark.asyncio
    async def test_stats_reset_returns_snapshot_with_accumulated_values(self) -> None:
        """Asserts that reset() returns a TrackingStats with the pre-reset counters."""
        resp = AgentResponse(content="x", model="m", input_tokens=5, output_tokens=3)
        middleware = TrackingMiddleware(
            client=MockProvider(responses=[resp]),
            cost_fn=lambda r: 0.02,
        )
        await middleware.send_request(AgentRequest(prompt="hi"))
        snapshot = middleware.stats.reset()
        assert snapshot.total_requests == 1
        assert snapshot.total_input_tokens == 5
        assert snapshot.total_output_tokens == 3
        assert snapshot.total_cost_usd == pytest.approx(0.02)

    @pytest.mark.asyncio
    async def test_stats_reset_zeroes_live_counters(self) -> None:
        """Asserts that after reset(), stats shows zeroed counters."""
        middleware = TrackingMiddleware(client=MockProvider(), cost_fn=lambda r: 0.01)
        await middleware.send_request(AgentRequest(prompt="hi"))
        middleware.stats.reset()
        assert middleware.stats.total_requests == 0
        assert middleware.stats.total_input_tokens == 0
        assert middleware.stats.total_output_tokens == 0
        assert middleware.stats.total_inference_ms == 0.0
        assert middleware.stats.total_ttft_ms == 0.0
        assert middleware.stats.total_cost_usd is None

    @pytest.mark.asyncio
    async def test_accumulation_continues_from_zero_after_reset(self) -> None:
        """Asserts that new requests after reset() accumulate from zero."""
        middleware = TrackingMiddleware(client=MockProvider())
        await middleware.send_request(AgentRequest(prompt="first"))
        middleware.stats.reset()
        await middleware.send_request(AgentRequest(prompt="second"))
        assert middleware.stats.total_requests == 1

    @pytest.mark.asyncio
    async def test_stacked_with_caching_middleware(self) -> None:
        """Asserts that TrackingMiddleware correctly wraps CachingMiddleware."""
        from mada_modelkit.middleware.cache import CachingMiddleware

        inner = CachingMiddleware(client=MockProvider())
        tracking = TrackingMiddleware(client=inner)
        await tracking.send_request(AgentRequest(prompt="cached"))
        await tracking.send_request(AgentRequest(prompt="cached"))
        # Both calls pass through TrackingMiddleware; second is served from cache
        assert tracking.stats.total_requests == 2

    @pytest.mark.asyncio
    async def test_context_manager_usage(self) -> None:
        """Asserts that TrackingMiddleware works as an async context manager."""
        async with TrackingMiddleware(client=MockProvider()) as middleware:
            await middleware.send_request(AgentRequest(prompt="hi"))
            assert middleware.stats.total_requests == 1
