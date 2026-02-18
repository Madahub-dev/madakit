"""Tracking middleware for mada-modelkit.

Wraps any BaseAgentClient and records aggregate statistics: request count,
token usage, wall-clock inference time, time-to-first-token, and optional
cost estimation. Zero external dependencies — stdlib only.
"""

from __future__ import annotations

import time
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
        """Execute request and record wall-clock time, token usage, and optional cost.

        Measures elapsed time via time.perf_counter, increments total_requests,
        accumulates input/output tokens and inference_ms. If cost_fn is set, calls it
        with the response and accumulates the result into total_cost_usd. Exceptions
        from the wrapped client propagate without updating stats.
        """
        start = time.perf_counter()
        response = await self._client.send_request(request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._stats.total_requests += 1
        self._stats.total_input_tokens += response.input_tokens
        self._stats.total_output_tokens += response.output_tokens
        self._stats.total_inference_ms += elapsed_ms
        if self._cost_fn is not None:
            cost = self._cost_fn(response)
            if self._stats.total_cost_usd is None:
                self._stats.total_cost_usd = 0.0
            self._stats.total_cost_usd += cost
        return response

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream chunks with TTFT measurement, per-stream timing, and token tracking.

        Times the interval from stream start to the first yielded chunk (TTFT),
        sets ``metadata["ttft_ms"]`` on that chunk, and accumulates
        ``total_ttft_ms``. On the chunk where ``is_final=True``, accumulates
        ``total_inference_ms`` (measured from stream start) and reads
        ``input_tokens``/``output_tokens`` from the chunk's metadata (defaulting
        to 0 if absent). Exceptions raised before the first chunk propagate
        without updating any stats.
        """
        start = time.perf_counter()
        first_chunk = True
        async for chunk in self._client.send_request_stream(request):
            if first_chunk:
                ttft_ms = (time.perf_counter() - start) * 1000.0
                chunk = StreamChunk(
                    delta=chunk.delta,
                    is_final=chunk.is_final,
                    metadata={**chunk.metadata, "ttft_ms": ttft_ms},
                )
                self._stats.total_ttft_ms += ttft_ms
                first_chunk = False
            if chunk.is_final:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._stats.total_inference_ms += elapsed_ms
                self._stats.total_input_tokens += chunk.metadata.get("input_tokens", 0)
                self._stats.total_output_tokens += chunk.metadata.get("output_tokens", 0)
            yield chunk
