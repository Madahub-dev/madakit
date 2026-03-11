"""Rate limiting middleware for mada-modelkit.

Implements token bucket and leaky bucket algorithms for request rate limiting.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

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
    ) -> None:
        """Initialise with a wrapped client and rate limiting configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            requests_per_second: Maximum sustained request rate per second.
            burst_size: Maximum burst capacity. Defaults to requests_per_second for
                token_bucket, and 2 * requests_per_second for leaky_bucket.
            strategy: Rate limiting algorithm ("token_bucket" or "leaky_bucket").
        """
        super().__init__()
        self._client = client
        self._requests_per_second = requests_per_second
        self._burst_size = burst_size
        self._strategy = strategy

        # Token bucket state
        if strategy == "token_bucket":
            self._tokens = float(burst_size if burst_size is not None else requests_per_second)
            self._max_tokens = float(burst_size if burst_size is not None else requests_per_second)
            self._last_refill = time.monotonic()
            self._lock = asyncio.Lock()
        # Leaky bucket state
        elif strategy == "leaky_bucket":
            default_queue_size = 2 * requests_per_second
            self._max_queue_size = int(burst_size if burst_size is not None else default_queue_size)
            self._queue: asyncio.Queue[asyncio.Event] = asyncio.Queue(maxsize=self._max_queue_size)
            self._processor_task: asyncio.Task[None] | None = None
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time since last refill (token bucket only).

        Called within _lock context. Updates _tokens up to _max_tokens based on
        requests_per_second rate and elapsed time.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._requests_per_second)
        self._last_refill = now

    async def _acquire_token(self) -> None:
        """Acquire a token from the bucket, blocking until one is available (token bucket only).

        Refills tokens, then waits (with exponential backoff polling) until at least
        one token is available. Consumes exactly one token before returning.
        """
        while True:
            async with self._lock:
                self._refill_tokens()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            # No token available: sleep and retry
            await asyncio.sleep(0.01)  # 10ms poll interval

    async def _processor_loop(self) -> None:
        """Process queued requests at fixed rate (leaky bucket only).

        Continuously dequeues events from _queue, sleeps for interval,
        then signals the event. Runs until cancelled.
        """
        interval = 1.0 / self._requests_per_second
        while True:
            event = await self._queue.get()
            await asyncio.sleep(interval)
            event.set()
            self._queue.task_done()

    async def _acquire_slot(self) -> None:
        """Acquire a slot in the leaky bucket queue (leaky bucket only).

        Ensures the processor task is running, enqueues an event, and waits
        for the processor to signal the event. This enforces the rate limit.
        """
        # Start processor if not running
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._processor_loop())

        # Create event for this request
        event = asyncio.Event()

        # Enqueue event (blocks if queue full)
        await self._queue.put(event)

        # Wait for processor to signal our turn
        await event.wait()

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request after acquiring rate limit token/slot.

        For token_bucket: waits until a token is available, then delegates.
        For leaky_bucket: queues request and waits for processor slot, then delegates.
        """
        if self._strategy == "token_bucket":
            await self._acquire_token()
        elif self._strategy == "leaky_bucket":
            await self._acquire_slot()

        return await self._client.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks after acquiring rate limit token/slot.

        Rate limiting applies before the first chunk, same as send_request.
        """
        # Stub: will implement in task 8.1.5
        async for chunk in self._client.send_request_stream(request):
            yield chunk
