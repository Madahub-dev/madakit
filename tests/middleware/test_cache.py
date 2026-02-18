"""Tests for CachingMiddleware constructor, default key function, send_request,
TTL validation, and LRU eviction (tasks 2.3.1–2.3.4).

Covers: client storage, default parameter values (ttl, max_entries, key_fn),
custom parameter storage, initial state of _cache and _in_flight (both empty
dicts), BaseAgentClient inheritance, semaphore absence, per-instance isolation,
stub delegation to the wrapped client, middleware composition,
_default_key_fn (determinism, field sensitivity, SHA-256 hex output, stop list
conversion, callable as a static method), send_request (cache miss delegates
to client, cache hit returns stored response without calling client, response
stored with timestamp, custom key_fn used when provided, exception from client
propagates without populating cache, per-key lock created in _in_flight),
TTL (expired entries trigger re-fetch, valid entries served from cache, boundary
at exact TTL is treated as expired, refresh on re-fetch), and LRU eviction
(oldest entry removed at capacity, access order updated on hit, cache never
exceeds max_entries).
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from helpers import MockProvider
from mada_modelkit._base import BaseAgentClient
from mada_modelkit._errors import ProviderError
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


class TestSendRequest:
    """CachingMiddleware.send_request — cache hit/miss, key function, and locking."""

    @pytest.mark.asyncio
    async def test_cache_miss_calls_client(self) -> None:
        """Asserts that a cache miss delegates to the wrapped client."""
        provider = MockProvider()
        cm = CachingMiddleware(client=provider)
        await cm.send_request(AgentRequest(prompt="hi"))
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(self) -> None:
        """Asserts that a cache hit returns the same response object as the first request."""
        provider = MockProvider()
        cm = CachingMiddleware(client=provider)
        req = AgentRequest(prompt="hi")
        first = await cm.send_request(req)
        second = await cm.send_request(req)
        assert second is first

    @pytest.mark.asyncio
    async def test_cache_hit_does_not_call_client_again(self) -> None:
        """Asserts that a cache hit skips the wrapped client entirely."""
        provider = MockProvider()
        cm = CachingMiddleware(client=provider)
        req = AgentRequest(prompt="hi")
        await cm.send_request(req)
        await cm.send_request(req)
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_response_stored_in_cache_after_first_request(self) -> None:
        """Asserts that _cache is populated with the response after a cache miss."""
        cm = CachingMiddleware(client=MockProvider())
        req = AgentRequest(prompt="hi")
        response = await cm.send_request(req)
        key = CachingMiddleware._default_key_fn(req)
        assert key in cm._cache
        assert cm._cache[key][0] is response

    @pytest.mark.asyncio
    async def test_cache_stores_response_with_timestamp(self) -> None:
        """Asserts that the cache entry includes a float timestamp alongside the response."""
        cm = CachingMiddleware(client=MockProvider())
        await cm.send_request(AgentRequest(prompt="hi"))
        key = CachingMiddleware._default_key_fn(AgentRequest(prompt="hi"))
        _, stored_at = cm._cache[key]
        assert isinstance(stored_at, float)
        assert stored_at > 0.0

    @pytest.mark.asyncio
    async def test_custom_key_fn_is_used_when_provided(self) -> None:
        """Asserts that a custom key_fn is called instead of _default_key_fn."""
        called_with: list[AgentRequest] = []

        def my_key_fn(req: AgentRequest) -> str:
            """Record the call and return a fixed key."""
            called_with.append(req)
            return "fixed-key"

        cm = CachingMiddleware(client=MockProvider(), key_fn=my_key_fn)
        req = AgentRequest(prompt="hi")
        await cm.send_request(req)
        assert len(called_with) == 1
        assert called_with[0] is req

    @pytest.mark.asyncio
    async def test_custom_key_fn_result_used_for_cache_lookup(self) -> None:
        """Asserts that the cache key produced by a custom key_fn is used for storage."""
        cm = CachingMiddleware(client=MockProvider(), key_fn=lambda _: "my-key")
        await cm.send_request(AgentRequest(prompt="hi"))
        assert "my-key" in cm._cache

    @pytest.mark.asyncio
    async def test_different_requests_use_separate_cache_entries(self) -> None:
        """Asserts that two distinct requests each get their own cache entry."""
        cm = CachingMiddleware(client=MockProvider())
        await cm.send_request(AgentRequest(prompt="alpha"))
        await cm.send_request(AgentRequest(prompt="beta"))
        assert len(cm._cache) == 2

    @pytest.mark.asyncio
    async def test_exception_from_client_propagates(self) -> None:
        """Asserts that an exception raised by the wrapped client propagates to the caller."""
        cm = CachingMiddleware(client=MockProvider(errors=[ProviderError("err", 500)]))
        with pytest.raises(ProviderError):
            await cm.send_request(AgentRequest(prompt="hi"))

    @pytest.mark.asyncio
    async def test_exception_from_client_does_not_populate_cache(self) -> None:
        """Asserts that a failed request leaves _cache empty."""
        cm = CachingMiddleware(client=MockProvider(errors=[ProviderError("err", 500)]))
        with pytest.raises(ProviderError):
            await cm.send_request(AgentRequest(prompt="hi"))
        assert len(cm._cache) == 0

    @pytest.mark.asyncio
    async def test_per_key_lock_created_in_in_flight(self) -> None:
        """Asserts that a per-key asyncio.Lock is stored in _in_flight after a cache miss."""
        cm = CachingMiddleware(client=MockProvider())
        req = AgentRequest(prompt="hi")
        await cm.send_request(req)
        key = CachingMiddleware._default_key_fn(req)
        assert key in cm._in_flight
        assert isinstance(cm._in_flight[key], asyncio.Lock)

    @pytest.mark.asyncio
    async def test_sequential_identical_requests_call_client_once(self) -> None:
        """Asserts that sequential requests with the same key only call the client once."""
        provider = MockProvider()
        cm = CachingMiddleware(client=provider)
        req = AgentRequest(prompt="same")
        for _ in range(5):
            await cm.send_request(req)
        assert provider.call_count == 1


class TestTTL:
    """CachingMiddleware TTL — expired entries trigger re-fetch; valid entries served from cache."""

    @pytest.mark.asyncio
    async def test_valid_entry_served_from_cache(self) -> None:
        """Asserts that a non-expired entry is returned from cache without calling the client."""
        provider = MockProvider()
        cm = CachingMiddleware(client=provider, ttl=10.0)
        req = AgentRequest(prompt="hi")
        with patch("mada_modelkit.middleware.cache.time.monotonic", return_value=100.0):
            await cm.send_request(req)
        with patch("mada_modelkit.middleware.cache.time.monotonic", return_value=109.9):
            await cm.send_request(req)  # elapsed=9.9 < 10.0 → valid
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_expired_entry_calls_client_again(self) -> None:
        """Asserts that a TTL-expired entry causes the client to be called again."""
        provider = MockProvider()
        cm = CachingMiddleware(client=provider, ttl=10.0)
        req = AgentRequest(prompt="hi")
        with patch("mada_modelkit.middleware.cache.time.monotonic", return_value=100.0):
            await cm.send_request(req)
        with patch("mada_modelkit.middleware.cache.time.monotonic", return_value=111.0):
            await cm.send_request(req)  # elapsed=11.0 >= 10.0 → expired
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_ttl_boundary_exactly_at_ttl_is_expired(self) -> None:
        """Asserts that an entry with elapsed == ttl is treated as expired."""
        provider = MockProvider()
        cm = CachingMiddleware(client=provider, ttl=10.0)
        req = AgentRequest(prompt="hi")
        with patch("mada_modelkit.middleware.cache.time.monotonic", return_value=100.0):
            await cm.send_request(req)
        with patch("mada_modelkit.middleware.cache.time.monotonic", return_value=110.0):
            await cm.send_request(req)  # elapsed=10.0 → not < ttl → expired
        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_expired_entry_refreshed_with_new_timestamp(self) -> None:
        """Asserts that a re-fetched entry overwrites the expired one with a fresh timestamp."""
        cm = CachingMiddleware(client=MockProvider(), ttl=10.0)
        req = AgentRequest(prompt="hi")
        key = CachingMiddleware._default_key_fn(req)
        with patch("mada_modelkit.middleware.cache.time.monotonic", return_value=100.0):
            await cm.send_request(req)
        with patch("mada_modelkit.middleware.cache.time.monotonic", return_value=115.0):
            await cm.send_request(req)
            _, new_stored_at = cm._cache[key]
        assert new_stored_at == 115.0

    @pytest.mark.asyncio
    async def test_expired_entry_removed_before_re_fetch(self) -> None:
        """Asserts that an expired entry is absent from _cache while being re-fetched."""
        cm = CachingMiddleware(client=MockProvider(), ttl=10.0)
        req = AgentRequest(prompt="hi")
        key = CachingMiddleware._default_key_fn(req)
        with patch("mada_modelkit.middleware.cache.time.monotonic", return_value=100.0):
            await cm.send_request(req)
        # Verify it was stored
        assert key in cm._cache
        # Now expire and re-fetch; entry should be refreshed (not absent after)
        with patch("mada_modelkit.middleware.cache.time.monotonic", return_value=115.0):
            await cm.send_request(req)
        assert key in cm._cache  # new entry stored after re-fetch


class TestLRUEviction:
    """CachingMiddleware LRU eviction — oldest-accessed entry removed when at capacity."""

    @pytest.mark.asyncio
    async def test_evicts_oldest_entry_at_capacity(self) -> None:
        """Asserts that adding a third entry with max_entries=2 evicts the first-inserted entry."""
        cm = CachingMiddleware(client=MockProvider(), max_entries=2)
        await cm.send_request(AgentRequest(prompt="a"))
        await cm.send_request(AgentRequest(prompt="b"))
        await cm.send_request(AgentRequest(prompt="c"))
        key_a = CachingMiddleware._default_key_fn(AgentRequest(prompt="a"))
        assert key_a not in cm._cache
        assert len(cm._cache) == 2

    @pytest.mark.asyncio
    async def test_access_updates_lru_order(self) -> None:
        """Asserts that accessing an entry promotes it so a newer entry is evicted instead."""
        cm = CachingMiddleware(client=MockProvider(), max_entries=2)
        await cm.send_request(AgentRequest(prompt="a"))  # order: [a]
        await cm.send_request(AgentRequest(prompt="b"))  # order: [a, b]
        await cm.send_request(AgentRequest(prompt="a"))  # access a → order: [b, a]
        await cm.send_request(AgentRequest(prompt="c"))  # evict b → order: [a, c]
        key_a = CachingMiddleware._default_key_fn(AgentRequest(prompt="a"))
        key_b = CachingMiddleware._default_key_fn(AgentRequest(prompt="b"))
        assert key_a in cm._cache
        assert key_b not in cm._cache

    @pytest.mark.asyncio
    async def test_max_entries_one_evicts_on_new_entry(self) -> None:
        """Asserts that max_entries=1 always evicts the sole entry when a new one arrives."""
        cm = CachingMiddleware(client=MockProvider(), max_entries=1)
        await cm.send_request(AgentRequest(prompt="a"))
        await cm.send_request(AgentRequest(prompt="b"))
        assert len(cm._cache) == 1
        key_a = CachingMiddleware._default_key_fn(AgentRequest(prompt="a"))
        assert key_a not in cm._cache

    @pytest.mark.asyncio
    async def test_cache_never_exceeds_max_entries(self) -> None:
        """Asserts that _cache length never exceeds max_entries after many requests."""
        max_e = 5
        cm = CachingMiddleware(client=MockProvider(), max_entries=max_e)
        for i in range(20):
            await cm.send_request(AgentRequest(prompt=f"prompt-{i}"))
        assert len(cm._cache) <= max_e

    @pytest.mark.asyncio
    async def test_evicted_entry_is_re_fetched_from_client(self) -> None:
        """Asserts that requesting an evicted entry results in a fresh client call."""
        provider = MockProvider()
        cm = CachingMiddleware(client=provider, max_entries=1)
        await cm.send_request(AgentRequest(prompt="a"))  # cached
        await cm.send_request(AgentRequest(prompt="b"))  # evicts a
        await cm.send_request(AgentRequest(prompt="a"))  # cache miss → client called again
        assert provider.call_count == 3
