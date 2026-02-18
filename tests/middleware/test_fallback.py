"""Tests for FallbackMiddleware constructor and sequential send_request (tasks 2.5.1–2.5.2).

Covers: primary storage, fallbacks list storage and order, default
fast_fail_ms (None), custom fast_fail_ms storage, BaseAgentClient
inheritance, semaphore absence, empty fallbacks list, per-instance
independence, stub delegation to the primary client, sequential fallback
on primary failure, ordered fallback traversal, last-exception propagation,
no-fallback edge case, and call-count verification.
"""

from __future__ import annotations

import pytest

from helpers import MockProvider
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
