"""Batching middleware for mada-modelkit.

Collects multiple requests and dispatches them as a batch for efficiency.
Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from madakit._base import BaseAgentClient
from madakit._errors import MiddlewareError
from madakit._types import AgentRequest, AgentResponse, StreamChunk

__all__ = ["BatchingMiddleware"]


class BatchingMiddleware(BaseAgentClient):
    """Middleware for batching multiple requests together.

    Collects requests until batch_size is reached or max_wait_ms timeout,
    then dispatches them as a batch to the underlying provider.
    """

    def __init__(
        self,
        client: BaseAgentClient,
        batch_size: int = 10,
        max_wait_ms: int = 100,
    ) -> None:
        """Initialize batching middleware.

        Args:
            client: Wrapped client to delegate batched requests to.
            batch_size: Maximum number of requests per batch (default 10).
            max_wait_ms: Maximum wait time in milliseconds before dispatching
                partial batch (default 100).

        Raises:
            ValueError: If batch_size or max_wait_ms is not positive.
        """
        super().__init__()
        self._client = client

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_wait_ms <= 0:
            raise ValueError("max_wait_ms must be positive")

        self._batch_size = batch_size
        self._max_wait_seconds = max_wait_ms / 1000.0

        # Queue for pending requests
        self._queue: asyncio.Queue[tuple[AgentRequest, asyncio.Future[AgentResponse]]] = asyncio.Queue()

        # Background task for batch processing
        self._processor_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._shutdown = False

    async def _start_processor_if_needed(self) -> None:
        """Start the batch processor task if not already running."""
        async with self._lock:
            if self._processor_task is None or self._processor_task.done():
                self._processor_task = asyncio.create_task(self._process_batches())

    async def _process_batches(self) -> None:
        """Background task that processes batches of requests."""
        try:
            while not self._shutdown:
                batch: list[tuple[AgentRequest, asyncio.Future[AgentResponse]]] = []
                deadline = time.monotonic() + self._max_wait_seconds

                try:
                    # Collect first request (blocking)
                    try:
                        first_item = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=self._max_wait_seconds,
                        )
                        batch.append(first_item)
                    except asyncio.TimeoutError:
                        # No requests, check shutdown and continue waiting
                        if self._shutdown:
                            break
                        continue

                    # Collect additional requests until batch_size or timeout
                    while len(batch) < self._batch_size:
                        remaining_time = deadline - time.monotonic()
                        if remaining_time <= 0:
                            # Deadline reached, dispatch what we have
                            break

                        try:
                            # Use max to ensure non-negative timeout
                            timeout = max(0.001, remaining_time)
                            item = await asyncio.wait_for(
                                self._queue.get(),
                                timeout=timeout,
                            )
                            batch.append(item)
                        except asyncio.TimeoutError:
                            # Timeout reached, dispatch partial batch
                            break

                    # Dispatch batch
                    if batch:
                        await self._dispatch_batch(batch)

                except Exception as e:
                    # On error, fail all requests in the batch
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(
                                MiddlewareError(f"Batch processing failed: {e}")
                            )
        except asyncio.CancelledError:
            # Task was cancelled, exit gracefully
            pass

    async def _dispatch_batch(
        self,
        batch: list[tuple[AgentRequest, asyncio.Future[AgentResponse]]],
    ) -> None:
        """Dispatch a batch of requests and distribute responses.

        Args:
            batch: List of (request, future) tuples to process.
        """
        # Execute all requests in parallel
        tasks_with_futures = []
        for request, future in batch:
            if not future.done():
                try:
                    task = asyncio.create_task(self._client.send_request(request))
                    tasks_with_futures.append((task, future))
                except Exception as e:
                    # Task creation failed, set exception immediately
                    future.set_exception(e)

        # Gather all tasks (with return_exceptions to continue on errors)
        if tasks_with_futures:
            tasks = [t for t, _ in tasks_with_futures]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Set futures with results or exceptions
            for (_, future), result in zip(tasks_with_futures, results):
                if not future.done():
                    if isinstance(result, Exception):
                        future.set_exception(result)
                    else:
                        future.set_result(result)

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Send request through batching layer.

        Args:
            request: The request to send.

        Returns:
            Response from the provider.

        Raises:
            MiddlewareError: If batching or request fails.
        """
        # Start processor if needed
        await self._start_processor_if_needed()

        # Create future for this request
        future: asyncio.Future[AgentResponse] = asyncio.Future()

        # Add to queue
        await self._queue.put((request, future))

        # Wait for response
        try:
            return await future
        except Exception as e:
            # Wrap any exceptions in MiddlewareError
            if isinstance(e, MiddlewareError):
                raise
            raise MiddlewareError(f"Batched request failed: {e}") from e

    async def send_request_stream(
        self, request: AgentRequest
    ) -> AsyncIterator[StreamChunk]:
        """Send streaming request.

        Note: Streaming is not batched. Requests are sent individually.

        Args:
            request: The request to send.

        Yields:
            Stream chunks from the provider.
        """
        # Streaming bypasses batching and goes directly to client
        async for chunk in self._client.send_request_stream(request):
            yield chunk

    async def close(self) -> None:
        """Close the middleware and cancel background tasks.

        Waits for pending requests to complete before closing.
        """
        # Signal shutdown
        self._shutdown = True

        # Cancel processor task
        if self._processor_task is not None and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        # Close wrapped client
        await self._client.close()

    def __del__(self) -> None:
        """Cancel processor task on garbage collection."""
        if hasattr(self, "_processor_task"):
            if self._processor_task is not None and not self._processor_task.done():
                try:
                    self._processor_task.cancel()
                except RuntimeError:
                    # Event loop is closed, can't cancel
                    pass
