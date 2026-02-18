"""Fallback middleware for mada-modelkit.

Wraps a primary BaseAgentClient and one or more fallback clients. Supports
sequential fallback (try each client in order on failure) and hedged requests
(race primary against fallbacks after a fast-fail timeout). Zero external
dependencies — stdlib only.
"""

from __future__ import annotations

from typing import AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["FallbackMiddleware"]


class FallbackMiddleware(BaseAgentClient):
    """Middleware that routes requests to fallback clients when the primary fails."""

    def __init__(
        self,
        primary: BaseAgentClient,
        fallbacks: list[BaseAgentClient],
        fast_fail_ms: float | None = None,
    ) -> None:
        """Initialise with a primary client, ordered fallbacks, and optional hedge timeout.

        Args:
            primary: The first client to attempt for every request.
            fallbacks: Ordered sequence of clients to try after the primary fails.
                When ``fast_fail_ms`` is set, the first fallback is also used as the
                hedge target if the primary does not respond in time.
            fast_fail_ms: Optional timeout in milliseconds. When provided and the
                primary has not returned a response within this window, the first
                fallback is started in parallel and the first result wins.
        """
        super().__init__()
        self._primary = primary
        self._fallbacks = fallbacks
        self._fast_fail_ms = fast_fail_ms

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Delegate to primary (stub; sequential and hedged logic added in tasks 2.5.2–2.5.3)."""
        return await self._primary.send_request(request)

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Delegate to primary (stub; pre-first-chunk fallback added in task 2.5.4)."""
        async for chunk in self._primary.send_request_stream(request):
            yield chunk
