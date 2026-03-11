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
            self._queue: asyncio.Queue[None] = asyncio.Queue(maxsize=self._max_queue_size)
            self._processor_task: asyncio.Task[None] | None = None
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request after acquiring rate limit token/slot.

        For token_bucket: waits until a token is available.
        For leaky_bucket: queues request and waits for processor.
        """
        # Stub: will implement in task 8.1.4
        return await self._client.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks after acquiring rate limit token/slot.

        Rate limiting applies before the first chunk, same as send_request.
        """
        # Stub: will implement in task 8.1.5
        async for chunk in self._client.send_request_stream(request):
            yield chunk
