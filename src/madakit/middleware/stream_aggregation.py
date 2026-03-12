"""Stream aggregation middleware for mada-modelkit.

Combine multiple streams via merging or racing strategies.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from madakit._base import BaseAgentClient
from madakit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["StreamAggregationMiddleware"]


class StreamAggregationMiddleware(BaseAgentClient):
    """Middleware for aggregating multiple streams.

    Supports merge (interleave chunks) and race (first wins) strategies.
    """

    def __init__(
        self,
        clients: list[BaseAgentClient],
        strategy: str = "merge",
    ) -> None:
        """Initialize stream aggregation middleware.

        Args:
            clients: List of clients to aggregate streams from.
            strategy: Aggregation strategy ("merge" or "race").

        Raises:
            ValueError: If clients list is empty or strategy is invalid.
        """
        super().__init__()

        if not clients:
            raise ValueError("clients list cannot be empty")

        if strategy not in ("merge", "race"):
            raise ValueError(f"Invalid strategy: {strategy}")

        self._clients = clients
        self._strategy = strategy

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Send request to first client only (non-streaming).

        Args:
            request: The request to send.

        Returns:
            Response from the first client.
        """
        # Non-streaming uses first client only
        return await self._clients[0].send_request(request)

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming request with aggregation.

        Args:
            request: The request to send.

        Yields:
            Aggregated stream chunks based on strategy.
        """
        if self._strategy == "merge":
            async for chunk in self._merge_streams(request):
                yield chunk
        else:  # race
            async for chunk in self._race_streams(request):
                yield chunk

    async def _merge_streams(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Merge multiple streams by interleaving chunks.

        Args:
            request: The request to send to all clients.

        Yields:
            Chunks from all streams, interleaved as they arrive.
        """
        # Create async iterators for all clients
        iterators = [
            client.send_request_stream(request).__aiter__()
            for client in self._clients
        ]

        # Track which iterators are still active
        active = set(range(len(iterators)))
        pending_tasks: dict[asyncio.Task[StreamChunk], int] = {}

        # Create initial tasks (one per iterator)
        for i in active:
            task = asyncio.create_task(iterators[i].__anext__())
            pending_tasks[task] = i

        # Process chunks as they arrive
        while pending_tasks:
            # Wait for next chunk from any stream
            done, pending = await asyncio.wait(
                pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                stream_idx = pending_tasks.pop(task)

                try:
                    chunk = await task
                    yield chunk

                    # Create next task for this iterator
                    next_task = asyncio.create_task(iterators[stream_idx].__anext__())
                    pending_tasks[next_task] = stream_idx

                except StopAsyncIteration:
                    # This stream is exhausted
                    active.discard(stream_idx)

        # All streams exhausted

    async def _race_streams(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Race multiple streams, using first to produce output.

        Args:
            request: The request to send to all clients.

        Yields:
            Chunks from the winning stream only.
        """
        # Create async iterators for all clients
        iterators = [
            client.send_request_stream(request).__aiter__()
            for client in self._clients
        ]

        # Create tasks for first chunk from each stream
        tasks = [
            asyncio.create_task(iterator.__anext__())
            for iterator in iterators
        ]

        # Wait for first chunk from any stream
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel all pending tasks (losing streams)
        for task in pending:
            task.cancel()

        # Get the winning task
        winner_task = list(done)[0]
        winner_idx = tasks.index(winner_task)

        # Yield first chunk from winner
        try:
            first_chunk = await winner_task
            yield first_chunk
        except StopAsyncIteration:
            # Winner stream was empty, shouldn't happen but handle gracefully
            return

        # Continue yielding from winning stream only
        winner_iterator = iterators[winner_idx]
        try:
            async for chunk in winner_iterator:
                yield chunk
        except StopAsyncIteration:
            pass

    async def close(self) -> None:
        """Close all client streams."""
        # Close all clients in parallel
        tasks = [client.close() for client in self._clients]
        await asyncio.gather(*tasks, return_exceptions=True)
