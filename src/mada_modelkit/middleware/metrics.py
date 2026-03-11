"""Metrics middleware for mada-modelkit.

Prometheus-compatible metrics export for request/response monitoring.
Requires prometheus_client (optional dependency).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, AsyncIterator

from mada_modelkit._base import BaseAgentClient
from mada_modelkit._types import AgentRequest, AgentResponse, StreamChunk

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

__all__ = ["MetricsMiddleware"]


class MetricsMiddleware(BaseAgentClient):
    """Middleware that exports Prometheus-compatible metrics."""

    def __init__(
        self,
        client: BaseAgentClient,
        registry: CollectorRegistry | None = None,
        prefix: str = "madakit",
    ) -> None:
        """Initialise with a wrapped client and metrics configuration.

        Args:
            client: The underlying BaseAgentClient to wrap.
            registry: Optional Prometheus registry. If None, uses default registry.
            prefix: Metric name prefix (default "madakit").

        Raises:
            ImportError: If prometheus_client is not installed.
        """
        super().__init__()
        self._client = client
        self._prefix = prefix

        # Deferred import of prometheus_client
        try:
            from prometheus_client import CollectorRegistry, Counter

            self._prometheus_available = True
        except ImportError as exc:
            raise ImportError(
                "prometheus_client is required for MetricsMiddleware. "
                "Install with: pip install prometheus-client"
            ) from exc

        # Use provided registry or default
        if registry is None:
            from prometheus_client import REGISTRY

            self._registry = REGISTRY
        else:
            self._registry = registry

        # Initialize counter metrics
        self._init_counters()

        # Initialize histogram metrics
        self._init_histograms()

        # Initialize gauge metrics
        self._init_gauges()

    def _init_counters(self) -> None:
        """Initialize Prometheus counter metrics."""
        from prometheus_client import Counter

        # Total requests counter
        self._requests_total = Counter(
            name=f"{self._prefix}_requests_total",
            documentation="Total number of requests",
            registry=self._registry,
        )

        # Errors by type counter
        self._errors_total = Counter(
            name=f"{self._prefix}_errors_total",
            documentation="Total number of errors by type",
            labelnames=["error_type"],
            registry=self._registry,
        )

    def _init_histograms(self) -> None:
        """Initialize Prometheus histogram metrics."""
        from prometheus_client import Histogram

        # Request duration histogram
        self._request_duration_seconds = Histogram(
            name=f"{self._prefix}_request_duration_seconds",
            documentation="Request latency distribution in seconds",
            registry=self._registry,
        )

        # Input tokens histogram
        self._input_tokens = Histogram(
            name=f"{self._prefix}_input_tokens",
            documentation="Input token count distribution",
            registry=self._registry,
        )

        # Output tokens histogram
        self._output_tokens = Histogram(
            name=f"{self._prefix}_output_tokens",
            documentation="Output token count distribution",
            registry=self._registry,
        )

        # Time to first token histogram (for streaming)
        self._ttft_seconds = Histogram(
            name=f"{self._prefix}_ttft_seconds",
            documentation="Time to first token in seconds",
            registry=self._registry,
        )

    def _init_gauges(self) -> None:
        """Initialize Prometheus gauge metrics."""
        from prometheus_client import Gauge

        # Active requests gauge
        self._active_requests = Gauge(
            name=f"{self._prefix}_active_requests",
            documentation="Current number of in-flight requests",
            registry=self._registry,
        )

    async def send_request(self, request: AgentRequest) -> AgentResponse:
        """Execute request with metrics collection.

        Records request count, latency, and token usage.
        """
        # Increment total requests counter
        self._requests_total.inc()

        # Track active request
        self._active_requests.inc()

        start_time = time.perf_counter()
        try:
            response = await self._client.send_request(request)

            # Record duration
            duration_seconds = time.perf_counter() - start_time
            self._request_duration_seconds.observe(duration_seconds)

            # Record token counts
            if response.input_tokens is not None:
                self._input_tokens.observe(response.input_tokens)
            if response.output_tokens is not None:
                self._output_tokens.observe(response.output_tokens)

            return response
        except Exception as exc:
            # Increment error counter by type
            error_type = type(exc).__name__
            self._errors_total.labels(error_type=error_type).inc()
            raise
        finally:
            # Decrement active request count
            self._active_requests.dec()

    async def send_request_stream(self, request: AgentRequest) -> AsyncIterator[StreamChunk]:
        """Stream response chunks with metrics collection.

        Records request count, latency, TTFT, and token usage.
        """
        # Increment total requests counter
        self._requests_total.inc()

        # Track active request
        self._active_requests.inc()

        start_time = time.perf_counter()
        first_chunk_received = False
        final_chunk = None

        try:
            async for chunk in self._client.send_request_stream(request):
                # Record TTFT on first chunk
                if not first_chunk_received:
                    ttft_seconds = time.perf_counter() - start_time
                    self._ttft_seconds.observe(ttft_seconds)
                    first_chunk_received = True

                # Track final chunk for metadata
                if chunk.is_final:
                    final_chunk = chunk

                yield chunk

            # Record total duration and token counts from final chunk
            duration_seconds = time.perf_counter() - start_time
            self._request_duration_seconds.observe(duration_seconds)

            if final_chunk is not None:
                input_tokens = final_chunk.metadata.get("input_tokens")
                output_tokens = final_chunk.metadata.get("output_tokens")

                if input_tokens is not None:
                    self._input_tokens.observe(input_tokens)
                if output_tokens is not None:
                    self._output_tokens.observe(output_tokens)

        except Exception as exc:
            # Increment error counter by type
            error_type = type(exc).__name__
            self._errors_total.labels(error_type=error_type).inc()
            raise
        finally:
            # Decrement active request count
            self._active_requests.dec()
