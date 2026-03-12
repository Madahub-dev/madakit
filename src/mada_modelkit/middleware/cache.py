"""Caching middleware for mada-modelkit.

Wraps any BaseAgentClient and caches responses with configurable TTL and LRU
eviction. Supports request coalescing (singleflight) for concurrent identical
requests. Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["CachingMiddleware"]


class CachingMiddleware(BaseAgentClient):
    """Middleware that caches responses with TTL, LRU eviction, and request coalescing."""

    def __init__(
        self,
        client: BaseAgentClient,
        ttl: float = 3600.0,
        max_entries: int = 1000,
        key_fn: Callable[[AgentRequest], str] | None = None,
    ) -> None:
        """Initialise with a wrapped client and cache configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            ttl: Time-to-live in seconds for cached responses. Defaults to 3600.0.
            max_entries: Maximum number of cache entries before LRU eviction kicks in.
                Defaults to 1000.
            key_fn: Optional callable that produces a string cache key from an
                AgentRequest. Defaults to _default_key_fn when None.
        """
        super().__init__()
        self._client = client
        self._ttl = ttl
        self._max_entries = max_entries
        self._key_fn = key_fn
        self._cache: OrderedDict[str, tuple[AgentResponse, float]] = OrderedDict()
        self._in_flight: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _default_key_fn(request: AgentRequest) -> str:
        """Return a stable hex-digest cache key derived from the request's cacheable fields.

        Hashes the tuple (prompt, system_prompt, max_tokens, temperature, stop) using
        SHA-256. The stop list is converted to a tuple before hashing to ensure a
        consistent string representation.
        """
        stop = tuple(request.stop) if request.stop is not None else None
        key_data = str((
            request.prompt,
            request.system_prompt,
            request.max_tokens,
            request.temperature,
            stop,
        ))
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request with caching, TTL validation, and LRU eviction.

        Fast path: if the key is in _cache and elapsed < ttl, move the entry to the
        end of the LRU order and return immediately. Expired entries are removed and
        execution falls through to the slow path.

        Slow path: acquire the per-key asyncio.Lock from _in_flight (creating it if
        absent), re-check the cache with TTL after the lock is acquired, call the
        wrapped client, evict the least-recently-used entry if _cache is at capacity,
        store (response, timestamp), and return. Exceptions propagate without
        populating the cache.
        """
        key_fn = self._key_fn if self._key_fn is not None else self._default_key_fn
        key = key_fn(request)

        # Fast path: cache hit with TTL check
        if key in self._cache:
            response, stored_at = self._cache[key]
            if time.monotonic() - stored_at < self._ttl:
                self._cache.move_to_end(key)  # mark as most-recently-used
                return response
            del self._cache[key]  # TTL expired; fall through to slow path

        # Slow path: per-key lock + double-checked locking
        if key not in self._in_flight:
            self._in_flight[key] = asyncio.Lock()
        async with self._in_flight[key]:
            # Re-check with TTL after acquiring lock
            if key in self._cache:
                response, stored_at = self._cache[key]
                if time.monotonic() - stored_at < self._ttl:
                    self._cache.move_to_end(key)
                    return response
                del self._cache[key]  # TTL expired
            response = await self._client.send_request(request)
            # LRU eviction: remove oldest-accessed entry if at capacity
            if len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
            self._cache[key] = (response, time.monotonic())
            return response

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks with caching and TTL validation.

        Fast path: if the key is in _cache and the entry has not expired, yield the
        cached content as a single StreamChunk(delta=content, is_final=True) without
        calling the wrapped client.

        Slow path: stream chunks through to the consumer immediately while accumulating
        delta values in a buffer. When the chunk with is_final=True arrives the buffer
        is joined, LRU eviction is applied if _cache is at capacity, and the response
        is written to _cache with the current timestamp. Any exception discards the
        buffer and re-raises without populating the cache. Streams that end without an
        is_final=True chunk are never cached.
        """
        key_fn = self._key_fn if self._key_fn is not None else self._default_key_fn
        key = key_fn(request)

        # Fast path: cache hit with TTL check
        if key in self._cache:
            response, stored_at = self._cache[key]
            if time.monotonic() - stored_at < self._ttl:
                self._cache.move_to_end(key)
                yield StreamChunk(delta=response.content, is_final=True)
                return
            del self._cache[key]  # TTL expired; fall through to slow path

        # Slow path: stream-through with deferred cache write on is_final
        buffer: list[str] = []
        try:
            async for chunk in self._client.send_request_stream(request):
                buffer.append(chunk.delta)
                yield chunk
                if chunk.is_final:
                    content = "".join(buffer)
                    if len(self._cache) >= self._max_entries:
                        self._cache.popitem(last=False)
                    self._cache[key] = (
                        AgentResponse(
                            content=content, model="", input_tokens=0, output_tokens=0
                        ),
                        time.monotonic(),
                    )
        except Exception:
            raise  # buffer discarded; do not cache
