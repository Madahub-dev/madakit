"""Tests for CachingMiddleware constructor (task 2.3.1).

Covers: client storage, default parameter values (ttl, max_entries, key_fn),
custom parameter storage, initial state of _cache and _in_flight (both empty
dicts), BaseAgentClient inheritance, semaphore absence, per-instance isolation,
stub delegation to the wrapped client, and middleware composition.
"""

from __future__ import annotations

import asyncio

import pytest

from helpers import MockProvider
from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk
from mada_modelkit.middleware.cache import CachingMiddleware


class TestCachingMiddlewareConstructor:
    """CachingMiddleware.__init__ — attribute storage and initial state."""

    def test_stores_client(self) -> None:
        """Asserts that the wrapped client is stored as _client."""
        provider = MockProvider()
        middleware = CachingMiddleware(client=provider)
        assert middleware._client is provider

    def test_default_ttl(self) -> None:
        """Asserts that ttl defaults to 3600.0."""
        middleware = CachingMiddleware(client=MockProvider())
        assert middleware._ttl == 3600.0

    def test_custom_ttl(self) -> None:
        """Asserts that a custom ttl is stored correctly."""
        middleware = CachingMiddleware(client=MockProvider(), ttl=60.0)
        assert middleware._ttl == 60.0

    def test_default_max_entries(self) -> None:
        """Asserts that max_entries defaults to 1000."""
        middleware = CachingMiddleware(client=MockProvider())
        assert middleware._max_entries == 1000

    def test_custom_max_entries(self) -> None:
        """Asserts that a custom max_entries is stored correctly."""
        middleware = CachingMiddleware(client=MockProvider(), max_entries=50)
        assert middleware._max_entries == 50

    def test_default_key_fn_is_none(self) -> None:
        """Asserts that key_fn defaults to None."""
        middleware = CachingMiddleware(client=MockProvider())
        assert middleware._key_fn is None

    def test_custom_key_fn_is_stored(self) -> None:
        """Asserts that a custom key_fn callable is stored correctly."""
        fn = lambda req: req.prompt  # noqa: E731
        middleware = CachingMiddleware(client=MockProvider(), key_fn=fn)
        assert middleware._key_fn is fn

    def test_cache_initialises_as_empty_dict(self) -> None:
        """Asserts that _cache starts as an empty dict."""
        middleware = CachingMiddleware(client=MockProvider())
        assert isinstance(middleware._cache, dict)
        assert len(middleware._cache) == 0

    def test_in_flight_initialises_as_empty_dict(self) -> None:
        """Asserts that _in_flight starts as an empty dict."""
        middleware = CachingMiddleware(client=MockProvider())
        assert isinstance(middleware._in_flight, dict)
        assert len(middleware._in_flight) == 0

    def test_is_base_agent_client(self) -> None:
        """Asserts that CachingMiddleware is a BaseAgentClient subclass."""
        middleware = CachingMiddleware(client=MockProvider())
        assert isinstance(middleware, BaseAgentClient)

    def test_no_semaphore_by_default(self) -> None:
        """Asserts that _semaphore is None when max_concurrent is not passed."""
        middleware = CachingMiddleware(client=MockProvider())
        assert middleware._semaphore is None

    def test_each_instance_has_independent_cache(self) -> None:
        """Asserts that two instances do not share the same _cache dict."""
        m1 = CachingMiddleware(client=MockProvider())
        m2 = CachingMiddleware(client=MockProvider())
        assert m1._cache is not m2._cache

    def test_each_instance_has_independent_in_flight(self) -> None:
        """Asserts that two instances do not share the same _in_flight dict."""
        m1 = CachingMiddleware(client=MockProvider())
        m2 = CachingMiddleware(client=MockProvider())
        assert m1._in_flight is not m2._in_flight

    def test_mutating_one_cache_does_not_affect_another(self) -> None:
        """Asserts that writing to one instance's _cache leaves the other untouched."""
        m1 = CachingMiddleware(client=MockProvider())
        m2 = CachingMiddleware(client=MockProvider())
        m1._cache["key"] = (AgentResponse(content="x", model="m", input_tokens=1, output_tokens=1), 0.0)
        assert "key" not in m2._cache

    def test_client_can_be_another_middleware(self) -> None:
        """Asserts that a CachingMiddleware can wrap another middleware (composition)."""
        from mada_modelkit.middleware.circuit_breaker import CircuitBreakerMiddleware

        inner = CircuitBreakerMiddleware(client=MockProvider())
        outer = CachingMiddleware(client=inner)
        assert outer._client is inner


class TestStubDelegation:
    """CachingMiddleware stub — send_request and send_request_stream delegate to client."""

    @pytest.mark.asyncio
    async def test_send_request_delegates_to_client(self) -> None:
        """Asserts that send_request returns the wrapped client's response."""
        expected = AgentResponse(content="cached", model="m", input_tokens=1, output_tokens=1)
        middleware = CachingMiddleware(client=MockProvider(responses=[expected]))
        result = await middleware.send_request(AgentRequest(prompt="hi"))
        assert result is expected

    @pytest.mark.asyncio
    async def test_send_request_stream_delegates_to_client(self) -> None:
        """Asserts that send_request_stream yields chunks from the wrapped client."""
        middleware = CachingMiddleware(client=MockProvider())
        chunks = []
        async for chunk in middleware.send_request_stream(AgentRequest(prompt="hi")):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].delta == "mock"

    @pytest.mark.asyncio
    async def test_send_request_increments_client_call_count(self) -> None:
        """Asserts that send_request invokes the wrapped client exactly once."""
        provider = MockProvider()
        middleware = CachingMiddleware(client=provider)
        await middleware.send_request(AgentRequest(prompt="hi"))
        assert provider.call_count == 1
