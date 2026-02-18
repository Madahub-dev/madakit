"""Fallback middleware for mada-modelkit.

Wraps a primary BaseAgentClient and one or more fallback clients. Supports
sequential fallback (try each client in order on failure) and hedged requests
(race primary against fallbacks after a fast-fail timeout). Zero external
dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
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
        """Try primary then each fallback; hedges when fast_fail_ms is set.

        Sequential mode (fast_fail_ms is None): iterates ``[primary] + fallbacks``
        in order, returning the first successful response. Re-raises the last
        exception if every client fails.

        Hedged mode (fast_fail_ms is not None): starts the primary as an asyncio
        task and awaits it for up to ``fast_fail_ms`` milliseconds. If it responds
        in time, that result is returned immediately. If it times out and a fallback
        exists, the first fallback is started in parallel; whichever task completes
        first wins and the loser is cancelled via ``cancel()``. If no fallback
        exists, waits for the primary to finish.
        """
        if self._fast_fail_ms is not None:
            return await self._hedged_send_request(request)
        last_exc: Exception | None = None
        for client in [self._primary, *self._fallbacks]:
            try:
                return await client.send_request(request)
            except Exception as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def _hedged_send_request(self, request: AgentRequest) -> AgentResponse:
        """Race primary against first fallback after fast_fail_ms timeout.

        Starts the primary as an asyncio task and waits up to ``fast_fail_ms``
        milliseconds. If the primary responds (successfully or with an exception)
        within the window, that result is returned immediately. If it times out,
        the first fallback client is launched in parallel via
        ``asyncio.create_task``; whichever task completes first wins, and the
        losing task is cancelled (task cancellation + ``client.cancel()`` call)
        before the winner's result is returned. When no fallbacks are configured,
        the primary is simply awaited without a deadline.
        """
        assert self._fast_fail_ms is not None
        timeout_s = self._fast_fail_ms / 1000.0

        primary_task = asyncio.create_task(self._primary.send_request(request))

        # Wait for the primary within the timeout window.
        done, _ = await asyncio.wait({primary_task}, timeout=timeout_s)
        if done:
            # Primary responded (or failed) before the timeout; use its result.
            return primary_task.result()

        # Primary timed out. If no fallback, simply wait for it to finish.
        if not self._fallbacks:
            return await primary_task

        # Start the first fallback and race both tasks.
        fallback_client = self._fallbacks[0]
        fallback_task = asyncio.create_task(fallback_client.send_request(request))

        done, pending = await asyncio.wait(
            {primary_task, fallback_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        winner_task = done.pop()

        # Cancel the loser: signal the asyncio task and notify the client.
        for loser_task in pending:
            loser_task.cancel()
            loser_client = self._primary if loser_task is primary_task else fallback_client
            await loser_client.cancel()
            try:
                await loser_task
            except (asyncio.CancelledError, Exception):
                pass

        return winner_task.result()

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Delegate to primary (stub; pre-first-chunk fallback added in task 2.5.4)."""
        async for chunk in self._primary.send_request_stream(request):
            yield chunk
