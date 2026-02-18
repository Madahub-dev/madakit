"""Circuit breaker middleware for mada-modelkit.

Wraps any BaseAgentClient and opens a circuit after a configurable failure
threshold, preventing cascading failures. Supports closed, open, and half-open
states with timeout-based recovery. Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["CircuitBreakerMiddleware"]


class CircuitBreakerMiddleware(BaseAgentClient):
    """Middleware that opens a circuit after repeated failures to prevent cascading errors."""

    def __init__(
        self,
        client: BaseAgentClient,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        """Initialise with a wrapped client and circuit breaker configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            failure_threshold: Number of consecutive failures before the circuit opens.
            recovery_timeout: Seconds to wait in the open state before transitioning
                to half-open and probing the backend.
        """
        super().__init__()
        self._client = client
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failure_count: int = 0
        self._state: str = "closed"
        self._last_failure_time: float | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Delegate to the wrapped client (circuit logic added in task 2.2.3)."""
        return await self._client.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Delegate streaming to the wrapped client (circuit logic added in task 2.2.4)."""
        async for chunk in self._client.send_request_stream(request):
            yield chunk
