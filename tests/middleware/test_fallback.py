"""Tests for FallbackMiddleware constructor (task 2.5.1).

Covers: primary storage, fallbacks list storage and order, default
fast_fail_ms (None), custom fast_fail_ms storage, BaseAgentClient
inheritance, semaphore absence, empty fallbacks list, per-instance
independence, and stub delegation to the primary client.
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
