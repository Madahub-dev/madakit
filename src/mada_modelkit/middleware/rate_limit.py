"""Rate limiting middleware for mada-modelkit.

Implements token bucket and leaky bucket algorithms for request rate limiting.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any, AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["RateLimitMiddleware"]


class RateLimitMiddleware(BaseAgentClient):  # pylint: disable=too-many-instance-attributes
    """Middleware that enforces rate limiting using token bucket or leaky bucket algorithms."""

    def __init__(
        self,
        client: BaseAgentClient,
        requests_per_second: float = 10.0,
        burst_size: int | None = None,
        strategy: str = "token_bucket",
        key_fn: Callable[[AgentRequest], Any] | None = None,
    ) -> None:
        """Initialise with a wrapped client and rate limiting configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            requests_per_second: Maximum sustained request rate per second.
            burst_size: Maximum burst capacity. Defaults to requests_per_second for
                token_bucket, and 2 * requests_per_second for leaky_bucket.
            strategy: Rate limiting algorithm ("token_bucket" or "leaky_bucket").
            key_fn: Optional function to extract a key from requests for per-key
                rate limiting. If None, uses global rate limit.
        """
        super().__init__()
        self._client = client
        self._requests_per_second = requests_per_second
        self._burst_size = burst_size
        self._strategy = strategy
        self._key_fn = key_fn

        # Computed constants (needed for both global and per-key)
        if strategy == "token_bucket":
            self._max_tokens = float(burst_size if burst_size is not None else requests_per_second)
        elif strategy == "leaky_bucket":
            self._max_queue_size = int(burst_size if burst_size is not None else 2 * requests_per_second)

        # Initialize state based on strategy
        if strategy == "token_bucket":
            # Global state (for non-per-key or None keys when per-key is enabled)
            self._tokens = self._max_tokens
            self._last_refill = time.monotonic()
            self._lock = asyncio.Lock()
            # Per-key state storage (when key_fn is provided)
            if key_fn is not None:
                self._per_key_tokens: dict[Any, float] = {}
                self._per_key_last_refill: dict[Any, float] = {}
                self._per_key_locks: dict[Any, asyncio.Lock] = {}
        elif strategy == "leaky_bucket":
            # Global state (for non-per-key or None keys when per-key is enabled)
            self._queue: asyncio.Queue[asyncio.Event] = asyncio.Queue(maxsize=self._max_queue_size)
            self._processor_task: asyncio.Task[None] | None = None
            # Per-key state storage (when key_fn is provided)
            if key_fn is not None:
                self._per_key_queues: dict[Any, asyncio.Queue[asyncio.Event]] = {}
                self._per_key_processors: dict[Any, asyncio.Task[None] | None] = {}

        if strategy not in ("token_bucket", "leaky_bucket"):
            raise ValueError(f"Unknown strategy: {strategy}")

    def _get_or_create_token_bucket(self, key: Any) -> tuple[asyncio.Lock, float, float]:
        """Get or create token bucket state for a key (token bucket + per-key only).

        Returns:
            Tuple of (lock, current_tokens, last_refill_time).
        """
        if key not in self._per_key_locks:
            self._per_key_locks[key] = asyncio.Lock()
            self._per_key_tokens[key] = self._max_tokens
            self._per_key_last_refill[key] = time.monotonic()
        return (
            self._per_key_locks[key],
            self._per_key_tokens[key],
            self._per_key_last_refill[key],
        )

    def _get_or_create_leaky_bucket(self, key: Any) -> tuple[asyncio.Queue[asyncio.Event], asyncio.Task[None] | None]:
        """Get or create leaky bucket state for a key (leaky bucket + per-key only).

        Returns:
            Tuple of (queue, processor_task).
        """
        if key not in self._per_key_queues:
            self._per_key_queues[key] = asyncio.Queue(maxsize=self._max_queue_size)
            self._per_key_processors[key] = None  # Will be created when needed
        return (self._per_key_queues[key], self._per_key_processors.get(key))

    def _refill_tokens(self, key: Any | None = None) -> None:
        """Refill tokens based on elapsed time since last refill (token bucket only).

        Called within lock context. Updates tokens up to max_tokens based on
        requests_per_second rate and elapsed time.

        Args:
            key: If provided and key_fn is set, refills for that key's bucket.
                Otherwise refills global bucket.
        """
        now = time.monotonic()
        if key is not None and self._key_fn is not None:
            # Per-key refill
            elapsed = now - self._per_key_last_refill[key]
            self._per_key_tokens[key] = min(
                self._max_tokens,
                self._per_key_tokens[key] + elapsed * self._requests_per_second
            )
            self._per_key_last_refill[key] = now
        else:
            # Global refill
            elapsed = now - self._last_refill
            self._tokens = min(self._max_tokens, self._tokens + elapsed * self._requests_per_second)
            self._last_refill = now

    async def _acquire_token(self, request: AgentRequest) -> None:
        """Acquire a token from the bucket, blocking until one is available (token bucket only).

        Refills tokens, then waits until at least one token is available.
        Consumes exactly one token before returning.

        Args:
            request: The request to potentially extract a key from (if key_fn is set).
        """
        # Determine key
        key = self._key_fn(request) if self._key_fn is not None else None

        if key is not None:
            # Per-key token bucket
            lock, _, _ = self._get_or_create_token_bucket(key)
            while True:
                async with lock:
                    self._refill_tokens(key)
                    if self._per_key_tokens[key] >= 1.0:
                        self._per_key_tokens[key] -= 1.0
                        return
                await asyncio.sleep(0.01)
        else:
            # Global token bucket
            while True:
                async with self._lock:
                    self._refill_tokens()
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                await asyncio.sleep(0.01)

    async def _processor_loop(self, key: Any | None = None) -> None:
        """Process queued requests at fixed rate (leaky bucket only).

        Continuously dequeues events from queue, sleeps for interval,
        then signals the event. Runs until cancelled.

        Args:
            key: If provided and key_fn is set, processes that key's queue.
                Otherwise processes global queue.
        """
        interval = 1.0 / self._requests_per_second
        queue = self._per_key_queues[key] if key is not None and self._key_fn is not None else self._queue
        while True:
            event = await queue.get()
            await asyncio.sleep(interval)
            event.set()
            queue.task_done()

    async def _acquire_slot(self, request: AgentRequest) -> None:
        """Acquire a slot in the leaky bucket queue (leaky bucket only).

        Ensures the processor task is running, enqueues an event, and waits
        for the processor to signal the event. This enforces the rate limit.

        Args:
            request: The request to potentially extract a key from (if key_fn is set).
        """
        # Determine key
        key = self._key_fn(request) if self._key_fn is not None else None

        if key is not None:
            # Per-key leaky bucket
            queue, processor = self._get_or_create_leaky_bucket(key)
            if processor is None or processor.done():
                self._per_key_processors[key] = asyncio.create_task(self._processor_loop(key))
            event = asyncio.Event()
            await queue.put(event)
            await event.wait()
        else:
            # Global leaky bucket
            if self._processor_task is None or self._processor_task.done():
                self._processor_task = asyncio.create_task(self._processor_loop())
            event = asyncio.Event()
            await self._queue.put(event)
            await event.wait()

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request after acquiring rate limit token/slot.

        For token_bucket: waits until a token is available, then delegates.
        For leaky_bucket: queues request and waits for processor slot, then delegates.
        If key_fn is set, uses per-key rate limiting.
        """
        if self._strategy == "token_bucket":
            await self._acquire_token(request)
        elif self._strategy == "leaky_bucket":
            await self._acquire_slot(request)

        return await self._client.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks after acquiring rate limit token/slot.

        Rate limiting applies before the first chunk, same as send_request.
        Consumes exactly one token/slot regardless of stream length.
        If key_fn is set, uses per-key rate limiting.
        """
        if self._strategy == "token_bucket":
            await self._acquire_token(request)
        elif self._strategy == "leaky_bucket":
            await self._acquire_slot(request)

        async for chunk in self._client.send_request_stream(request):
            yield chunk
