"""Timeout middleware for mada-modelkit.

Request-level timeout enforcement independent of HTTP client timeouts.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["TimeoutMiddleware"]


class TimeoutMiddleware(BaseAgentClient):
    """Middleware that enforces timeouts on requests and streams."""

    def __init__(self, client: BaseAgentClient, timeout_seconds: float = 30.0) -> None:
        """Initialise with a wrapped client and timeout configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            timeout_seconds: Maximum time in seconds to wait for a response.
                For streams, applies only to first chunk arrival.
        """
        super().__init__()
        self._client = client
        self._timeout_seconds = timeout_seconds

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request with timeout.

        Delegates to wrapped client with asyncio.wait_for timeout enforcement.
        Raises asyncio.TimeoutError if the request exceeds timeout_seconds.

        Args:
            request: The request to send.

        Returns:
            The response from the wrapped client.

        Raises:
            asyncio.TimeoutError: If the request exceeds the timeout.
        """
        return await asyncio.wait_for(
            self._client.send_request(request), timeout=self._timeout_seconds
        )

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks with timeout on first chunk.

        Timeout applies only to the arrival of the first chunk.
        Once streaming begins, subsequent chunks are not subject to timeout.

        Args:
            request: The request to send.

        Yields:
            Stream chunks from the wrapped client.

        Raises:
            asyncio.TimeoutError: If the first chunk does not arrive within timeout_seconds.
        """
        stream = self._client.send_request_stream(request)
        iterator = stream.__aiter__()

        # Apply timeout to first chunk only
        first_chunk = await asyncio.wait_for(
            iterator.__anext__(), timeout=self._timeout_seconds
        )
        yield first_chunk

        # Stream remaining chunks without timeout
        async for chunk in iterator:
            yield chunk
