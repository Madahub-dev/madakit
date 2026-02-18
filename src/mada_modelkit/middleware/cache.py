"""Caching middleware for mada-modelkit.

Wraps any BaseAgentClient and caches responses with configurable TTL and LRU
eviction. Supports request coalescing (singleflight) for concurrent identical
requests. Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
import hashlib
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
        self._cache: dict[str, tuple[AgentResponse, float]] = {}
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
        """Delegate to wrapped client (stub; full cache logic added in task 2.3.3)."""
        return await self._client.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Delegate to wrapped client (stub; full stream-through logic added in task 2.3.6)."""
        async for chunk in self._client.send_request_stream(request):
            yield chunk
