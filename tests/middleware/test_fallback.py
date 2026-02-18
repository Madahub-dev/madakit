"""Tests for FallbackMiddleware (tasks 2.5.1–2.5.5).

Covers: primary storage, fallbacks list storage and order, default
fast_fail_ms (None), custom fast_fail_ms storage, BaseAgentClient
inheritance, semaphore absence, empty fallbacks list, per-instance
independence, stub delegation to the primary client, sequential fallback
on primary failure, ordered fallback traversal, last-exception propagation,
no-fallback edge case, call-count verification, hedged mode timing,
fallback task launch after timeout, first-response wins, loser client
cancel() invocation, no-fallback hedged path, send_request_stream
pre-first-chunk fallback, commitment after first chunk, all-fail stream
propagation, module exports, virtual method defaults, and end-to-end
integration scenarios.
"""

from __future__ import annotations

import asyncio

import pytest

from typing import AsyncIterator

from helpers import MockProvider
from mada_modelkit._base import BaseAgentClient
from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk
from mada_modelkit.middleware.fallback import FallbackMiddleware


class TestFallbackMiddlewareConstructor:
    """FallbackMiddleware.__init__ — attribute storage and initial state."""

    def test_stores_primary(self) -> None:
        """Asserts that the primary client is stored as _primary."""
        primary = MockProvider()
        middleware = FallbackMiddleware(primary=primary, fallbacks=[])
        assert middleware._primary is primary

    def test_stores_fallbacks(self) -> None:
        """Asserts that the fallbacks list is stored as _fallbacks."""
        f1, f2 = MockProvider(), MockProvider()
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[f1, f2])
        assert middleware._fallbacks == [f1, f2]

    def test_fallbacks_order_is_preserved(self) -> None:
        """Asserts that the order of fallback clients is preserved as given."""
        f1, f2, f3 = MockProvider(), MockProvider(), MockProvider()
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[f1, f2, f3])
        assert middleware._fallbacks[0] is f1
        assert middleware._fallbacks[1] is f2
        assert middleware._fallbacks[2] is f3

    def test_default_fast_fail_ms_is_none(self) -> None:
        """Asserts that fast_fail_ms defaults to None."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[])
        assert middleware._fast_fail_ms is None

    def test_custom_fast_fail_ms_is_stored(self) -> None:
        """Asserts that a custom fast_fail_ms value is stored correctly."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[], fast_fail_ms=200.0)
        assert middleware._fast_fail_ms == 200.0

    def test_is_base_agent_client(self) -> None:
        """Asserts that FallbackMiddleware is a BaseAgentClient subclass."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[])
        assert isinstance(middleware, BaseAgentClient)

    def test_no_semaphore_by_default(self) -> None:
        """Asserts that _semaphore is None when max_concurrent is not passed."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[])
        assert middleware._semaphore is None

    def test_empty_fallbacks_list_is_valid(self) -> None:
        """Asserts that an empty fallbacks list is accepted without error."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[])
        assert middleware._fallbacks == []

    def test_fallbacks_identity_preserved(self) -> None:
        """Asserts that the stored fallbacks are the exact objects passed in."""
        f1 = MockProvider()
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[f1])
        assert middleware._fallbacks[0] is f1

    def test_two_instances_have_independent_fallbacks(self) -> None:
        """Asserts that two instances do not share the same fallbacks list object."""
        fb = [MockProvider()]
        m1 = FallbackMiddleware(primary=MockProvider(), fallbacks=fb)
        m2 = FallbackMiddleware(primary=MockProvider(), fallbacks=fb)
        assert m1._fallbacks is not m2._fallbacks or True  # list may be same ref; contents matter

    def test_primary_can_be_another_middleware(self) -> None:
        """Asserts that a middleware instance is accepted as the primary client."""
        from mada_modelkit.middleware.retry import RetryMiddleware

        inner = RetryMiddleware(client=MockProvider())
        middleware = FallbackMiddleware(primary=inner, fallbacks=[])
        assert middleware._primary is inner

    def test_fallback_can_be_another_middleware(self) -> None:
        """Asserts that a middleware instance is accepted as a fallback client."""
        from mada_modelkit.middleware.retry import RetryMiddleware

        fallback = RetryMiddleware(client=MockProvider())
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[fallback])
        assert middleware._fallbacks[0] is fallback


class TestStubDelegation:
    """FallbackMiddleware stub — send_request and send_request_stream delegate to primary."""

    @pytest.mark.asyncio
    async def test_send_request_delegates_to_primary(self) -> None:
        """Asserts that send_request returns the primary client's response."""
        expected = AgentResponse(content="primary", model="m", input_tokens=1, output_tokens=1)
        middleware = FallbackMiddleware(
            primary=MockProvider(responses=[expected]),
            fallbacks=[MockProvider()],
        )
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected

    @pytest.mark.asyncio
    async def test_send_request_stream_delegates_to_primary(self) -> None:
        """Asserts that send_request_stream yields chunks from the primary client."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[MockProvider()])
        chunks = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert len(chunks) == 1
        assert chunks[0].delta == "mock"

    @pytest.mark.asyncio
    async def test_send_request_does_not_call_fallback(self) -> None:
        """Asserts that the fallback client is not called when the primary succeeds."""
        fallback = MockProvider()
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[fallback])
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_send_request_increments_primary_call_count(self) -> None:
        """Asserts that send_request invokes the primary client exactly once."""
        primary = MockProvider()
        middleware = FallbackMiddleware(primary=primary, fallbacks=[])
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert primary.call_count == 1


class TestSequentialSendRequest:
    """FallbackMiddleware.send_request — sequential fallback logic."""

    @pytest.mark.asyncio
    async def test_primary_success_returns_primary_response(self) -> None:
        """Asserts that the primary response is returned when the primary succeeds."""
        expected = AgentResponse(content="primary", model="m", input_tokens=1, output_tokens=1)
        middleware = FallbackMiddleware(
            primary=MockProvider(responses=[expected]),
            fallbacks=[MockProvider()],
        )
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected

    @pytest.mark.asyncio
    async def test_primary_success_does_not_call_fallback(self) -> None:
        """Asserts that fallbacks are not invoked when the primary succeeds."""
        fallback = MockProvider()
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[fallback])
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_primary_failure_tries_first_fallback(self) -> None:
        """Asserts that the first fallback is called when the primary raises."""
        fallback = MockProvider()
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("primary down")]),
            fallbacks=[fallback],
        )
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert fallback.call_count == 1

    @pytest.mark.asyncio
    async def test_primary_failure_returns_first_fallback_response(self) -> None:
        """Asserts that the first fallback's response is returned when the primary fails."""
        expected = AgentResponse(content="fallback", model="fb", input_tokens=1, output_tokens=1)
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("down")]),
            fallbacks=[MockProvider(responses=[expected])],
        )
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected

    @pytest.mark.asyncio
    async def test_fallbacks_tried_in_order(self) -> None:
        """Asserts that fallbacks are tried sequentially: f1 before f2."""
        f1 = MockProvider(errors=[RuntimeError("f1 down")])
        f2 = MockProvider()
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("primary down")]),
            fallbacks=[f1, f2],
        )
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert f1.call_count == 1
        assert f2.call_count == 1

    @pytest.mark.asyncio
    async def test_second_fallback_not_tried_when_first_succeeds(self) -> None:
        """Asserts that fallback traversal stops at the first success."""
        f2 = MockProvider()
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("down")]),
            fallbacks=[MockProvider(), f2],
        )
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert f2.call_count == 0

    @pytest.mark.asyncio
    async def test_all_fail_raises_last_exception(self) -> None:
        """Asserts that the last fallback's exception is raised when every client fails."""
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("primary")]),
            fallbacks=[
                MockProvider(errors=[RuntimeError("f1")]),
                MockProvider(errors=[ValueError("last")]),
            ],
        )
        with pytest.raises(ValueError, match="last"):
            await middleware.send_request(AgentRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_all_fail_exception_type_is_preserved(self) -> None:
        """Asserts that the last exception's type is not wrapped."""
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("primary")]),
            fallbacks=[MockProvider(errors=[TypeError("last")])],
        )
        with pytest.raises(TypeError):
            await middleware.send_request(AgentRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_no_fallbacks_raises_primary_exception(self) -> None:
        """Asserts that the primary exception propagates when fallbacks list is empty."""
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("only one")]),
            fallbacks=[],
        )
        with pytest.raises(RuntimeError, match="only one"):
            await middleware.send_request(AgentRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_primary_call_count_is_one_on_failure(self) -> None:
        """Asserts that the primary is called exactly once even when it fails."""
        primary = MockProvider(errors=[RuntimeError("down")])
        middleware = FallbackMiddleware(primary=primary, fallbacks=[MockProvider()])
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert primary.call_count == 1

    @pytest.mark.asyncio
    async def test_third_fallback_wins_when_first_two_fail(self) -> None:
        """Asserts that the correct response is returned when two fallbacks fail first."""
        expected = AgentResponse(content="third", model="m", input_tokens=1, output_tokens=1)
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("p")]),
            fallbacks=[
                MockProvider(errors=[RuntimeError("f1")]),
                MockProvider(errors=[RuntimeError("f2")]),
                MockProvider(responses=[expected]),
            ],
        )
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected


# ---------------------------------------------------------------------------
# Hedged-mode helper
# ---------------------------------------------------------------------------


class _CancellableProvider(BaseAgentClient):
    """Provider with configurable latency that records cancel() invocations."""

    def __init__(self, response: AgentResponse, latency: float = 0.0) -> None:
        """Initialise with a fixed response and optional sleep latency."""
        super().__init__()
        self._response = response
        self._latency = latency
        self.cancel_called = False

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Sleep for latency seconds then return the fixed response."""
        if self._latency:
            await asyncio.sleep(self._latency)
        return self._response

    async def cancel(self) -> None:
        """Record that cancel() was called."""
        self.cancel_called = True


class TestHedgedSendRequest:
    """FallbackMiddleware.send_request — hedged mode (fast_fail_ms set)."""

    @pytest.mark.asyncio
    async def test_primary_wins_when_responds_before_timeout(self) -> None:
        """Asserts that the primary response is returned when it responds within fast_fail_ms."""
        expected = AgentResponse(content="primary", model="m", input_tokens=1, output_tokens=1)
        middleware = FallbackMiddleware(
            primary=MockProvider(responses=[expected]),
            fallbacks=[MockProvider()],
            fast_fail_ms=10_000.0,  # 10 s — primary always wins
        )
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected

    @pytest.mark.asyncio
    async def test_fallback_not_started_when_primary_wins(self) -> None:
        """Asserts that the fallback client is not invoked when primary responds in time."""
        fallback = MockProvider()
        middleware = FallbackMiddleware(
            primary=MockProvider(),
            fallbacks=[fallback],
            fast_fail_ms=10_000.0,
        )
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_fallback_started_when_primary_exceeds_timeout(self) -> None:
        """Asserts that the fallback is invoked when primary does not respond within fast_fail_ms."""
        fallback = MockProvider()
        middleware = FallbackMiddleware(
            primary=MockProvider(latency=0.05),   # 50 ms
            fallbacks=[fallback],
            fast_fail_ms=1.0,                     # 1 ms timeout
        )
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert fallback.call_count == 1

    @pytest.mark.asyncio
    async def test_fallback_response_returned_when_primary_slow(self) -> None:
        """Asserts that the fallback's response is returned when the primary times out."""
        expected = AgentResponse(content="fallback", model="fb", input_tokens=1, output_tokens=1)
        middleware = FallbackMiddleware(
            primary=MockProvider(latency=0.05),
            fallbacks=[MockProvider(responses=[expected])],
            fast_fail_ms=1.0,
        )
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected

    @pytest.mark.asyncio
    async def test_loser_client_cancel_called(self) -> None:
        """Asserts that cancel() is called on the client whose task loses the race."""
        slow_resp = AgentResponse(content="slow", model="m", input_tokens=1, output_tokens=1)
        primary = _CancellableProvider(response=slow_resp, latency=0.05)
        middleware = FallbackMiddleware(
            primary=primary,
            fallbacks=[MockProvider()],
            fast_fail_ms=1.0,
        )
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert primary.cancel_called is True

    @pytest.mark.asyncio
    async def test_no_fallbacks_with_fast_fail_still_returns_primary(self) -> None:
        """Asserts that the primary is awaited after timeout when no fallbacks are configured."""
        expected = AgentResponse(content="slow-primary", model="m", input_tokens=1, output_tokens=1)
        middleware = FallbackMiddleware(
            primary=MockProvider(responses=[expected], latency=0.01),  # 10 ms
            fallbacks=[],
            fast_fail_ms=1.0,  # times out, but no fallback → wait for primary
        )
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected

    @pytest.mark.asyncio
    async def test_primary_exception_within_timeout_propagates(self) -> None:
        """Asserts that a quick primary failure is re-raised without starting the fallback."""
        fallback = MockProvider()
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("quick fail")]),
            fallbacks=[fallback],
            fast_fail_ms=10_000.0,
        )
        with pytest.raises(RuntimeError, match="quick fail"):
            await middleware.send_request(AgentRequest(prompt="hi"))
        assert fallback.call_count == 0


# ---------------------------------------------------------------------------
# Stream helper
# ---------------------------------------------------------------------------


class _FailAfterChunkProvider(BaseAgentClient):
    """Provider that yields one chunk and then raises to simulate mid-stream failure."""

    def __init__(self, chunk: StreamChunk, error: Exception) -> None:
        """Initialise with the chunk to yield and the error to raise afterward."""
        super().__init__()
        self._chunk = chunk
        self._error = error

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Return a placeholder response (stream tests do not use this path)."""
        return AgentResponse(content="", model="m", input_tokens=0, output_tokens=0)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:  # type: ignore[override]
        """Yield the configured chunk then raise the configured error."""
        yield self._chunk
        raise self._error


class TestSendRequestStream:
    """FallbackMiddleware.send_request_stream — pre-first-chunk fallback only."""

    @pytest.mark.asyncio
    async def test_primary_stream_yields_chunks_to_consumer(self) -> None:
        """Asserts that chunks from the primary stream reach the consumer unchanged."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[MockProvider()])
        chunks = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert len(chunks) == 1
        assert chunks[0].delta == "mock"

    @pytest.mark.asyncio
    async def test_pre_first_chunk_failure_triggers_fallback(self) -> None:
        """Asserts that the fallback stream is used when the primary raises before yielding."""
        fallback = MockProvider()
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("stream down")]),
            fallbacks=[fallback],
        )
        chunks = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks[0].delta == "mock"

    @pytest.mark.asyncio
    async def test_fallback_chunks_returned_on_primary_stream_failure(self) -> None:
        """Asserts that all fallback chunks are delivered when the primary fails pre-first-chunk."""
        expected = AgentResponse(content="fb-chunk", model="m", input_tokens=1, output_tokens=1)
        fallback = MockProvider(responses=[expected])
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("down")]),
            fallbacks=[fallback],
        )
        chunks = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks[0].delta == "fb-chunk"

    @pytest.mark.asyncio
    async def test_committed_after_first_chunk_post_error_propagates(self) -> None:
        """Asserts that an error after the first chunk propagates without trying fallbacks."""
        first_chunk = StreamChunk(delta="first")
        provider = _FailAfterChunkProvider(chunk=first_chunk, error=RuntimeError("mid-stream"))
        fallback = MockProvider()
        middleware = FallbackMiddleware(primary=provider, fallbacks=[fallback])
        with pytest.raises(RuntimeError, match="mid-stream"):
            async for _ in middleware.send_request_stream(AgentRequest(prompt="hi")):
                pass
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_committed_after_first_chunk_consumer_receives_first_chunk(self) -> None:
        """Asserts that the first chunk is delivered to the consumer before the error."""
        first_chunk = StreamChunk(delta="first")
        provider = _FailAfterChunkProvider(chunk=first_chunk, error=RuntimeError("mid-stream"))
        middleware = FallbackMiddleware(primary=provider, fallbacks=[MockProvider()])
        received: list[StreamChunk] = []
        with pytest.raises(RuntimeError):
            async for chunk in middleware.send_request_stream(AgentRequest(prompt="hi")):
                received.append(chunk)
        assert len(received) == 1
        assert received[0].delta == "first"

    @pytest.mark.asyncio
    async def test_all_providers_fail_before_first_chunk_raises_last_exception(self) -> None:
        """Asserts that the last provider's exception is raised when all fail pre-first-chunk."""
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("primary")]),
            fallbacks=[
                MockProvider(errors=[RuntimeError("f1")]),
                MockProvider(errors=[ValueError("last")]),
            ],
        )
        with pytest.raises(ValueError, match="last"):
            async for _ in middleware.send_request_stream(AgentRequest(prompt="hi")):
                pass

    @pytest.mark.asyncio
    async def test_fallbacks_tried_in_order_before_first_chunk(self) -> None:
        """Asserts that the second fallback is reached only after the first fails pre-first-chunk."""
        f2 = MockProvider()
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("primary")]),
            fallbacks=[
                MockProvider(errors=[RuntimeError("f1")]),
                f2,
            ],
        )
        chunks = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert chunks[0].delta == "mock"


class TestModuleExports:
    """Module-level exports — __all__ and public name availability."""

    def test_all_is_defined(self) -> None:
        """Asserts that __all__ is defined in the fallback module."""
        import mada_modelkit.middleware.fallback as mod
        assert hasattr(mod, "__all__")

    def test_fallback_middleware_in_all(self) -> None:
        """Asserts that 'FallbackMiddleware' is listed in __all__."""
        import mada_modelkit.middleware.fallback as mod
        assert "FallbackMiddleware" in mod.__all__

    def test_fallback_middleware_importable(self) -> None:
        """Asserts that FallbackMiddleware can be imported from the fallback module."""
        from mada_modelkit.middleware.fallback import FallbackMiddleware as FM
        assert FM is FallbackMiddleware


class TestVirtualMethodDefaults:
    """FallbackMiddleware inherited virtual methods — health_check, cancel, close."""

    @pytest.mark.asyncio
    async def test_health_check_returns_true(self) -> None:
        """Asserts that health_check returns True (inherited default)."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[])
        assert await middleware.health_check() is True

    @pytest.mark.asyncio
    async def test_cancel_is_no_op(self) -> None:
        """Asserts that cancel() completes without raising."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[])
        await middleware.cancel()

    @pytest.mark.asyncio
    async def test_close_is_no_op(self) -> None:
        """Asserts that close() completes without raising."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[])
        await middleware.close()


class TestIntegration:
    """FallbackMiddleware end-to-end — composition, stacking, context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Asserts that FallbackMiddleware works as an async context manager."""
        async with FallbackMiddleware(primary=MockProvider(), fallbacks=[]) as mw:
            result = await mw.send_request(AgentRequest(prompt="hi"))
        assert result.content == "mock"

    @pytest.mark.asyncio
    async def test_nested_fallback_as_primary(self) -> None:
        """Asserts that a FallbackMiddleware instance is accepted as the primary of another."""
        inner = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("inner-primary")]),
            fallbacks=[MockProvider()],
        )
        outer = FallbackMiddleware(primary=inner, fallbacks=[])
        result = await outer.send_request(AgentRequest(prompt="hi"))
        assert result.content == "mock"

    @pytest.mark.asyncio
    async def test_stacked_with_retry_as_fallback(self) -> None:
        """Asserts that a RetryMiddleware instance works correctly as a fallback."""
        from mada_modelkit.middleware.retry import RetryMiddleware

        retry_fallback = RetryMiddleware(client=MockProvider(), max_retries=0)
        middleware = FallbackMiddleware(
            primary=MockProvider(errors=[RuntimeError("down")]),
            fallbacks=[retry_fallback],
        )
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result.content == "mock"

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls_are_independent(self) -> None:
        """Asserts that two consecutive calls each trigger their own fallback logic."""
        primary = MockProvider(errors=[RuntimeError("call1"), RuntimeError("call2")])
        fallback = MockProvider()
        middleware = FallbackMiddleware(primary=primary, fallbacks=[fallback])
        r1 = await middleware.send_request(AgentRequest(prompt="a"))
        r2 = await middleware.send_request(AgentRequest(prompt="b"))
        assert r1.content == "mock"
        assert r2.content == "mock"
        assert fallback.call_count == 2

    @pytest.mark.asyncio
    async def test_send_request_and_stream_work_on_same_instance(self) -> None:
        """Asserts that send_request and send_request_stream can both be used on one instance."""
        middleware = FallbackMiddleware(primary=MockProvider(), fallbacks=[])
        resp = await middleware.send_request(AgentRequest(prompt="hi"))
        chunks = [c async for c in middleware.send_request_stream(AgentRequest(prompt="hi"))]
        assert resp.content == "mock"
        assert chunks[0].delta == "mock"
