"""Tests for CachingMiddleware constructor and default cache key function (tasks 2.3.1–2.3.2).

Covers: client storage, default parameter values (ttl, max_entries, key_fn),
custom parameter storage, initial state of _cache and _in_flight (both empty
dicts), BaseAgentClient inheritance, semaphore absence, per-instance isolation,
stub delegation to the wrapped client, middleware composition, and
_default_key_fn (determinism, field sensitivity, SHA-256 hex output, stop list
conversion, callable as a static method).
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


class TestDefaultKeyFn:
    """CachingMiddleware._default_key_fn — stable hash of cacheable request fields."""

    def test_same_request_produces_same_key(self) -> None:
        """Asserts that calling _default_key_fn twice with an identical request returns the same key."""
        req = AgentRequest(prompt="hello")
        assert CachingMiddleware._default_key_fn(req) == CachingMiddleware._default_key_fn(req)

    def test_different_prompts_produce_different_keys(self) -> None:
        """Asserts that requests differing only in prompt produce distinct keys."""
        k1 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hello"))
        k2 = CachingMiddleware._default_key_fn(AgentRequest(prompt="world"))
        assert k1 != k2

    def test_different_system_prompt_produces_different_keys(self) -> None:
        """Asserts that requests differing only in system_prompt produce distinct keys."""
        k1 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", system_prompt=None))
        k2 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", system_prompt="sys"))
        assert k1 != k2

    def test_different_max_tokens_produces_different_keys(self) -> None:
        """Asserts that requests differing only in max_tokens produce distinct keys."""
        k1 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", max_tokens=100))
        k2 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", max_tokens=200))
        assert k1 != k2

    def test_different_temperature_produces_different_keys(self) -> None:
        """Asserts that requests differing only in temperature produce distinct keys."""
        k1 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", temperature=0.0))
        k2 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", temperature=1.0))
        assert k1 != k2

    def test_different_stop_produces_different_keys(self) -> None:
        """Asserts that requests differing only in stop sequences produce distinct keys."""
        k1 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", stop=["END"]))
        k2 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", stop=["STOP"]))
        assert k1 != k2

    def test_stop_none_vs_empty_list_produce_different_keys(self) -> None:
        """Asserts that stop=None and stop=[] are treated as distinct cache keys."""
        k1 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", stop=None))
        k2 = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", stop=[]))
        assert k1 != k2

    def test_stop_list_does_not_raise(self) -> None:
        """Asserts that a stop list is accepted without raising a TypeError."""
        key = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi", stop=["a", "b"]))
        assert isinstance(key, str)

    def test_returns_nonempty_string(self) -> None:
        """Asserts that the returned cache key is a non-empty string."""
        key = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi"))
        assert isinstance(key, str)
        assert len(key) > 0

    def test_returns_64_char_hex_string(self) -> None:
        """Asserts that the key is a 64-character lowercase hex string (SHA-256 digest)."""
        key = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi"))
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_callable_as_static_method(self) -> None:
        """Asserts that _default_key_fn can be called on the class without an instance."""
        key = CachingMiddleware._default_key_fn(AgentRequest(prompt="static"))
        assert isinstance(key, str)

    def test_metadata_and_attachments_do_not_affect_key(self) -> None:
        """Asserts that metadata and attachments are excluded from the cache key."""
        req1 = AgentRequest(prompt="hi", metadata={"x": 1})
        req2 = AgentRequest(prompt="hi", metadata={"y": 2})
        assert CachingMiddleware._default_key_fn(req1) == CachingMiddleware._default_key_fn(req2)
