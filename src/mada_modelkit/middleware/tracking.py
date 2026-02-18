"""Tracking middleware for mada-modelkit.

Wraps any BaseAgentClient and records aggregate statistics: request count,
token usage, wall-clock inference time, time-to-first-token, and optional
cost estimation. Zero external dependencies — stdlib only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk, TrackingStats

__all__ = ["TrackingMiddleware"]


class TrackingMiddleware(BaseAgentClient):
    """Middleware that records per-request timing, token usage, and optional cost."""

    def __init__(
        self,
        client: BaseAgentClient,
        cost_fn: Callable[[AgentResponse], float] | None = None,
    ) -> None:
        """Initialise with a wrapped client and an optional cost function.

        Args:
            client: The underlying BaseAgentClient to wrap.
            cost_fn: Optional callable that accepts an AgentResponse and returns
                the estimated cost in USD. When None, total_cost_usd remains None.
        """
        super().__init__()
        self._client = client
        self._cost_fn = cost_fn
        self._stats: TrackingStats = TrackingStats()

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Delegate to wrapped client (stub; full tracking logic added in task 2.4.2)."""
        return await self._client.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Delegate to wrapped client (stub; full TTFT tracking added in task 2.4.3)."""
        async for chunk in self._client.send_request_stream(request):
            yield chunk
